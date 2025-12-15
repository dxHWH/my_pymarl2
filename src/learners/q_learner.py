import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from utils.rl_utils import build_td_lambda_targets
from envs.matrix_game import print_matrix_status
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np

class QLearner:
    def __init__(self, mac, scheme, logger, args): 
        self.args = args
        self.mac = mac
        self.logger = logger
        #将controller的参数加入的参数列表中
        self.params = list(mac.parameters())

        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda  else 'cpu')
        #根据传入的参数信息来选择mixer
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            #将mixer的参数加入的参数列表中；
            #注意，如果要在Qlearner的的基础上diy新的神经网络部分，需要将新的神经网络加入到当前learner的参数列表中
            self.params += list(self.mixer.parameters())
            #深度拷贝出目标网络，用于计算loss
            self.target_mixer = copy.deepcopy(self.mixer)

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params,  lr=args.lr)
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        
        self.train_t = 0

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        #这里的batch: batch_size(n条轨迹)*max_seq_length（轨迹长度）的数据
        # 分解轨迹信息
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            #对每一个时间步，将batch中的信息给到mac中，得到一个batch下，对于所有episode在t时间步下，每个智能体对当前动作的Qi
            agent_outs = self.mac.forward(batch, t=t)# batch_size n_agents n_actions
            mac_out.append(agent_outs)#加入到列表中
        #将列表处理为向量，这里的mac_out代表一个batch下每个轨迹上从0-t时刻所有agent的动作的Q值向量
        # (batch_size, max_seq_length, n_agents, n_actions)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time  # batch_size, max_seq_length, n_agents , n_actions

        # Pick the Q-Values for the actions taken by each agent
        # 第0步-倒数第二步的动作的Q值
        # 在dim=3 动作维度上收集数据，用Q值替换轨迹中agent确定选择出的动作
        # (batch_size, max_seq_length - 1, n_agents, 1)，这里的1代表当前轨迹中agent的采取动作的Q值，已经无意义
        # .squeeze(3)​​: 移除长度为1的动作维度，得到 (batch_size, max_seq_length - 1, n_agents)
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals_back = chosen_action_qvals
        
        #目标网络同理
        # Calculate the Q-Values necessary for the target
        target_mac_out = [] 
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        # 
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        # mask：将无法选的动作的Q估值设置为负无穷
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        # double_q：是否使用double q，即target的更新值是选用Q网络的最大动作还是目标网络的最大动作
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        # 更新公式，时序差分目标：  当前步奖励+折扣系数*（是否到最后一步的更新标志）*（下一步的最大Q值）
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        # 时序差分误差：  当前步的Q估值与更新目标的差值
        # 这里的chosen_action_qvals_back代表当前步的Q估值，targets.detach()代表更新目标,因为targets不随着Qgen的更新而更新，所以detach()从pytorch计算图中剥离
        td_error = (chosen_action_qvals - targets.detach())

        #拓展掩码维度
        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        # 将无用数据mask掉，（之前为了对齐维度到max_seq_length产生了大量无用数据）
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        # 损失，求出各个轨迹的损失求和，除以mask的元素个数，得到平均损失，避免梯度过大不稳定
        loss = 0.5 * (masked_td_error ** 2).sum() / mask.sum()
        
        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        #训练时间超过目标网络更新设定，更新目标网络
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
        
        # 输出log
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
            
            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                print_matrix_status(batch, self.mixer, mac_out)
    
    # 目标网络更新实现，通过load
    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
