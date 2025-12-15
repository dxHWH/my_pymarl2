import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionQMixer(nn.Module):
    def __init__(self, args):
        super(AttentionQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.obs_dim = int(np.prod(args.obs_shape))
        self.n_actions = args.n_actions

        self.embed_dim = args.mixing_embed_dim
        self.attention_heads = getattr(args, 'attention_heads', 4)
        self.abs = getattr(self.args, 'abs', True)

        # 用于处理每个智能体的状态信息
        self.agent_state_encoder = nn.Linear(self.obs_dim, self.embed_dim)
        
        # 用于处理动作信息
        self.action_encoder = nn.Linear(self.n_actions, self.embed_dim)
        
        # 用于处理Q值信息
        self.q_encoder = nn.Linear(1, self.embed_dim)
        
        # 多头注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.attention_heads,
            batch_first=True
        )
        
        # 用于将注意力输出映射到QMIX的输入
        self.attention_to_qmix = nn.Linear(self.embed_dim, 1)
        
        # 原始的QMIX网络结构
        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, actions=None, observations=None, all_qvals=None):
        """
        Args:
            agent_qs: [batch_size, seq_len, n_agents] - 每个智能体选择的动作的Q值
            states: [batch_size, seq_len, state_dim] - 全局状态
            actions: [batch_size, seq_len, n_agents] - 每个智能体选择的动作 (可选)
            observations: [batch_size, seq_len, n_agents, obs_dim] - 每个智能体的观察 (可选)
            all_qvals: [batch_size, seq_len, n_agents, n_actions] - 每个智能体对所有动作的Q值 (可选)
        """
        bs, seq_len = agent_qs.size(0), agent_qs.size(1)
        
        # 如果没有提供actions和observations，则使用原始的QMIX
        if actions is None or observations is None:
            return self._original_qmix_forward(agent_qs, states)
        
        # 处理状态信息
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, self.n_agents)
        actions = actions.reshape(-1, self.n_agents)
        observations = observations.reshape(-1, self.n_agents, self.obs_dim)
        
        # 为每个智能体创建增强的特征表示
        enhanced_qs = []
        
        for i in range(self.n_agents):
            # 获取第i个智能体的信息
            agent_obs = observations[:, i, :]  # [batch*seq, obs_dim]
            agent_action = actions[:, i]  # [batch*seq]
            agent_q = agent_qs[:, i]  # [batch*seq]
            
            # 编码观察信息
            obs_encoded = self.agent_state_encoder(agent_obs)  # [batch*seq, embed_dim]
            
            # 编码动作信息 (one-hot)
            action_onehot = th.zeros(agent_action.size(0), self.n_actions, device=agent_action.device)
            action_onehot.scatter_(1, agent_action.unsqueeze(1), 1)
            action_encoded = self.action_encoder(action_onehot)  # [batch*seq, embed_dim]
            
            # 编码Q值信息 - 使用选中动作的Q值
            q_encoded = self.q_encoder(agent_q.unsqueeze(1))  # [batch*seq, embed_dim]
            
            # 如果有完整的Q值信息，可以进一步利用
            if all_qvals is not None:
                all_qvals_flat = all_qvals.reshape(-1, self.n_agents, self.n_actions)
                agent_all_qvals = all_qvals_flat[:, i, :]  # [batch*seq, n_actions]
                # 可以添加对所有Q值的统计信息，如最大值、平均值等
                max_q = th.max(agent_all_qvals, dim=1, keepdim=True)[0]  # [batch*seq, 1]
                mean_q = th.mean(agent_all_qvals, dim=1, keepdim=True)  # [batch*seq, 1]
                q_stats = th.cat([max_q, mean_q], dim=1)  # [batch*seq, 2]
                q_stats_encoded = self.q_encoder(q_stats)  # [batch*seq, embed_dim]
                q_encoded = q_encoded + q_stats_encoded  # 融合选中动作Q值和统计信息
            
            # 拼接所有信息
            agent_features = obs_encoded + action_encoded + q_encoded  # [batch*seq, embed_dim]
            enhanced_qs.append(agent_features)
        
        # 堆叠所有智能体的特征 [batch*seq, n_agents, embed_dim]
        enhanced_qs = th.stack(enhanced_qs, dim=1)
        
        # 应用多头注意力机制
        # 将每个智能体的特征作为query, key, value
        attended_features, attention_weights = self.attention(
            enhanced_qs, enhanced_qs, enhanced_qs
        )
        
        # 将注意力输出映射回Q值维度
        attended_qs = self.attention_to_qmix(attended_features)  # [batch*seq, n_agents, 1]
        attended_qs = attended_qs.squeeze(-1)  # [batch*seq, n_agents]
        
        # 使用增强后的Q值进行QMIX混合
        attended_qs = attended_qs.reshape(bs, seq_len, self.n_agents)
        return self._original_qmix_forward(attended_qs, states.reshape(bs, seq_len, -1))

    def _original_qmix_forward(self, agent_qs, states):
        """原始的QMIX前向传播逻辑"""
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
        
        # use sinh
        if self.args.use_sinh:
            sinh_agent_qs = th.sinh(agent_qs).detach()
            agent_qs = agent_qs + (sinh_agent_qs - agent_qs)

        # First layer
        w1 = self.hyper_w_1(states).abs() if self.abs else self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        
        # Second layer
        w_final = self.hyper_w_final(states).abs() if self.abs else self.hyper_w_final(states)
        w_final = w_final.view(-1, self.embed_dim, 1)
        v = self.V(states).view(-1, 1, 1)
        y = th.bmm(hidden, w_final) + v
        q_tot = y.view(bs, -1, 1)
        
        return q_tot

    def k(self, states):
        """用于向外分析智能体贡献"""
        bs = states.size(0)
        w1 = th.abs(self.hyper_w_1(states))
        w_final = th.abs(self.hyper_w_final(states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        w_final = w_final.view(-1, self.embed_dim, 1)
        k = th.bmm(w1, w_final).view(bs, -1, self.n_agents)
        k = k / th.sum(k, dim=2, keepdim=True)
        return k
    
    def b(self, states):
        """分析环境基础值"""
        bs = states.size(0)
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        b1 = self.hyper_b_1(states)
        b1 = b1.view(-1, 1, self.embed_dim)
        v = self.V(states).view(-1, 1, 1)
        b = th.bmm(b1, w_final) + v
        return b
