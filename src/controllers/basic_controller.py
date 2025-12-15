from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args

        #计算输入的维度
        input_shape = self._get_input_shape(scheme)
        # agent网络创建
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        # action选择器创建
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.save_probs = getattr(self.args, 'save_probs', False)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        #输入可选动作、和agent对当前所有动作的Q值，让动作选择器根据不同的策略输出确定动作
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if test_mode:
            self.agent.eval()
        #输入到网络中
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e5

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        obs_t = batch["obs"][:, t] # (bs, n_agents, obs_shape)

        # --- (KeyAgent) ---
        if getattr(self.args, "use_critical_agent_obs", False):
            try:
                # 检查 "时间门控" 标志 (t >= 50k)
                # (bs, 1) -> (bs, 1, 1)
                is_active_flag = batch["critical_id_active"][:, t].float().view(bs, 1, 1)

                # 1. 获取 critical_id
                crit_id_scalar = batch["critical_id"][:, t] # (bs, 1)
                
                # 2. 收集关键观测 (key_obs_t)
                # (bs, 1, 1) -> (bs, 1, obs_shape)
                crit_id_gather = crit_id_scalar.reshape(-1, 1, 1).expand(-1, -1, self.args.obs_shape)
                # (bs, 1, obs_shape)
                key_obs_t = th.gather(obs_t, dim=1, index=crit_id_gather.long()) 
                # (bs, n_agents, obs_shape)
                key_obs_t_broadcast = key_obs_t.repeat(1, self.n_agents, 1)
                
                # 3. 创建关键 ID 的 one-hot (key_id_onehot)
                # (bs, 1) -> (bs, n_agents)
                key_id_onehot = th.zeros(bs, self.n_agents, device=batch.device)
                key_id_onehot.scatter_(dim=1, index=crit_id_scalar.long(), value=1)
                # (bs, n_agents) -> (bs, n_agents, n_agents)
                key_id_onehot_broadcast = key_id_onehot.unsqueeze(1).repeat(1, self.n_agents, 1)
                
                # 4. (关键) 应用时间门控
                # 在 t <= vae_activation 时, is_active_flag = 0.0, 辅助信息被清零
                # 在 t >= vae_activation 时, is_active_flag = 1.0, 辅助信息被保留
                key_obs_t_gated = key_obs_t_broadcast * is_active_flag
                key_id_onehot_gated = key_id_onehot_broadcast * is_active_flag

                # 5. 拼接
                # [自己的观测, 关键智能体的观测, 关键智能体的ID]
                aug_obs_t = th.cat([obs_t, key_obs_t_gated, key_id_onehot_gated], dim=-1)
                
                inputs.append(aug_obs_t)
                
            except Exception as e:
                print(f"Error in _build_inputs augmentation: {e}")
                inputs.append(obs_t) # 出错时回退
        else:
            inputs.append(obs_t)  # 正常
        # --- (更新结束) ---


        if self.args.obs_last_action:
            if t == 0:
                # 初始时刻没有上一个动作，用0代替
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            # 创建智能体ID的one-hot编码 [1, 智能体数, 智能体数]
            # 并扩展到整个批次 [批次大小, 智能体数, 智能体数]
            # agent_ids = th.eye(self.n_agents, device=batch.device) #创建一个单位矩阵生成每个agent的one-hot编码 (n_agents*n_agents)
            # tensor([
            #     [1., 0., 0.],  # 智能体 0 的 one-hot 编码
            #     [0., 1., 0.],  # 智能体 1 的 one-hot 编码
            #     [0., 0., 1.]   # 智能体 2 的 one-hot 编码
            # ], device='cuda:0')

            # agent_ids = agent_ids.unsqueeze(0).expand(bs, -1, -1) 在第 0 维度增加一个大小为 1 的维度，
            # 扩展到整个批次 为批次维度预留位置 (batch_size*n_agents*n_agents)
            # inputs.append(agent_ids)
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        
        # 合并所有输入，并将其展平为 [batch_size, n_agents, -1]
        # 此时 inputs = [obs_tensor, last_action_tensor, agent_id_tensor]
        # [x.reshape(bs, self.n_agents, -1) for x in inputs] 其中 -1 代表自动计算该维度的大小，并保证元素总数不变，（因为不同任务下智能体数量不同）
        # th.cat(..., dim=-1) 沿最后一个维度（特征维度）拼接所有张量。[bs, n_agents, obs_dim + action_dim + n_agents]
        # 如果 inputs 包含：

        # 观测值:       [2, 3, 4]
        # 上一动作:     [2, 3, 2]
        # 智能体ID：    [2, 3, 3]
        # 拼接结果：    [2, 3, 4 + 2 + 3] = [2, 3, 9]

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    # 计算输入的维度：观测obs的维度+上一步动作的维度+智能体数量
    # 加入智能体数量的原因：在共享的智能体网络输入中，最后加上agent 的id
    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            #为什么不是+1？因为使用的one-hot编码 ：[0,0,...,1,0,...,0]
            input_shape += self.n_agents

        if getattr(self.args, "use_critical_agent_obs", False):
            # 将关键智能体的观测维度也加进去
            # scheme["obs"]["vshape"] == self.args.obs_shape
            input_shape += scheme["obs"]["vshape"]
            # 2. 加上关键智能体的 ID 维度 (one-hot)
            input_shape += self.n_agents

        return input_shape
