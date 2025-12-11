import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dmaq_general import DMAQer as DMAQ_QattenMixer
from modules.mixers.cfmix import DMAQ_QattenMixer as CF_Mixer
import torch as th
import numpy as np
from torch.optim import RMSprop, Adam
from utils.th_utils import get_params_size
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from envs.matrix_game import print_matrix_status

# 计算张量的归一化熵，用于衡量分布的不确定性
def entropy(x, dim=-1):
    max_entropy = np.log(x.shape[dim])
    x = (x+1e-8) / th.sum(x+1e-8, dim, keepdim=True)
    return (-th.log(x)*x).sum(dim) / max_entropy


class DMAQ_qattenLearner:
    def __init__(self, mac, scheme, logger, args):
        # 初始化学习器对象，设置参数、控制器、日志记录器等
        self.args = args
        self.mac = mac
        self.logger = logger

        # 获取控制器的参数
        self.params = list(mac.parameters())

        # 初始化目标网络更新的计数器
        self.last_target_update_episode = 0

        # 初始化混合器（Mixer）
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == 'dmaq_qatten':
                self.mixer = DMAQ_QattenMixer(args)
            elif args.mixer == "cfmix":
                self.mixer = CF_Mixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # 初始化优化器（Adam 或 RMSprop）
        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params,  lr=args.lr)
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # 深拷贝目标网络（包括控制器和混合器）
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1

        # 设置动作数量
        self.n_actions = self.args.n_actions

    def sub_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, mac, mixer, optimiser, params,
                  show_demo=False, save_data=None):
        # 执行一次训练迭代
        # 从 batch 中提取奖励、动作、终止标志等数据
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]

        # 计算 Q 值（mac_out 和 q_self_out）
        mac_out = []
        q_self_out = []
        onehot_out = []
        eps = 1e-8
        eye = th.eye(self.args.n_agents).reshape(-1).to(self.args.device)
        eye = th.cat([eye] * self.args.att_heads, dim=0)
        self.mac.init_hidden(batch.batch_size)
        att_out = []
        for t in range(batch.max_seq_length):
            agent_outs = mac.forward(batch, t=t)
            q_self = mac.q_self_out
            att = self.mac.agent.att.dot
            att = att.view(batch.batch_size,-1)
            onehot_out.append(F.kl_div((att+eps).log(), eye, reduction='none').mean(dim=-1))
            mac_out.append(agent_outs)
            q_self_out.append(q_self)
        mac_out = th.stack(mac_out, dim=1)  # 按时间维度拼接
        q_self_out = th.stack(q_self_out, dim=1)
        onehot_out = th.stack(onehot_out, dim=1)

        # 选择动作对应的 Q 值
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)
        chosen_q_self = th.gather(q_self_out[:, :-1], dim=3, index=actions).squeeze(3)
        
        x_mac_out = mac_out.clone().detach()
        x_q_self_out = q_self_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        x_q_self_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)
        max_q_self, max_q_self_index = x_q_self_out[:, :-1].max(dim=3)
        M1 = (max_action_qvals - max_q_self).squeeze(-1)
        M2 = (chosen_action_qvals - chosen_q_self)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        if show_demo:
            q_i_data = chosen_action_qvals.detach().cpu().numpy()
            q_data = (max_action_qvals - chosen_action_qvals).detach().cpu().numpy()

        # 计算目标 Q 值
        target_mac_out = []
        target_q_self_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_q_self_out_i =  self.target_mac.q_self_out
            target_mac_out.append(target_agent_outs)
            target_q_self_out.append(target_q_self_out_i)

        # 忽略第一个时间步的 Q 值
        target_mac_out = th.stack(target_mac_out[1:], dim=1)
        target_q_self_out = th.stack(target_q_self_out[1:], dim=1)

        # 屏蔽不可用动作
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        target_q_self_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # 获取最大化当前 Q 值的动作（用于双重 Q 学习）
            mac_out_detach = mac_out.clone().detach()
            q_self_out_detach = q_self_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            q_self_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]

            target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_chosen_q_self = th.gather(target_q_self_out, 3, cur_max_actions).squeeze(3)
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_max_q_self = target_q_self_out.max(dim=3)[0]
            target_next_actions = cur_max_actions.detach()

            cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,)).cuda()
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
        else:
            # 计算目标 Q 值
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            # 忽略第一个时间步的 Q 值
            target_mac_out = th.stack(target_mac_out[1:], dim=1)
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if mixer is not None:
            if self.args.mixer == "dmaq_qatten":

                ans_chosen, q_attend_regs, head_entropies = \
                    mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv, _, _ = mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                      max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv
            elif self.args.mixer == "cfmix":
                if self.args.max_action_qvals==True:
                    ans_chosen, q_attend_regs, head_entropies= mixer(max_action_qvals, batch["state"][:, :-1], is_v=True)
                else:
                    ans_chosen, q_attend_regs, head_entropies= mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                
                ans_adv ,_,_= mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                max_q_i=max_action_qvals, q_self_out=chosen_q_self,max_q_self=max_q_self , is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv
            else:
                ans_chosen = mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv = mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv

            if self.args.double_q:
                if self.args.mixer == "dmaq_qatten":
                    target_chosen, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                         actions=cur_max_actions_onehot,
                                                         max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
                elif self.args.mixer == "cfmix":
                    if self.args.max_action_qvals==True:
                        target_chosen, _, _= self.target_mixer(target_max_qvals, batch["state"][:, :-1], is_v=True)
                    else:
                        target_chosen, _, _= self.target_mixer(target_chosen_qvals, batch["state"][:, :-1], is_v=True)
                
                    target_adv ,_,_= self.target_mixer(target_chosen_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                    max_q_i=target_max_qvals, q_self_out=target_chosen_q_self,max_q_self=target_max_q_self , is_v=False)
                    target_max_qvals = target_chosen + target_adv
                else:
                    target_chosen = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                   actions=cur_max_actions_onehot,
                                                   max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], is_v=True)

        # 计算 1 步 Q 学习目标
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        if show_demo:
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()
            tot_target = targets.detach().cpu().numpy()
            print('action_pair_%d_%d' % (save_data[0], save_data[1]), np.squeeze(q_data[:, 0]),
                  np.squeeze(q_i_data[:, 0]), np.squeeze(tot_q_data[:, 0]), np.squeeze(tot_target[:, 0]))
            self.logger.log_stat('action_pair_%d_%d' % (save_data[0], save_data[1]),
                                 np.squeeze(tot_q_data[:, 0]), t_env)
            return

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        if self.args.mixer == "dmaq_qatten" or self.args.mixer == "cfmix":
            loss = (masked_td_error ** 2).sum() / mask.sum() + q_attend_regs
        else:
            loss = (masked_td_error ** 2).sum() / mask.sum()

        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        # Optimise
        if self.args.learner == "qplex_learner" or self.args.learner =="cf_learner" and t_env > self.args.breakpoint:
            onehot_out = onehot_out[:, :-1].unsqueeze(-1) * mask
            loss = loss + self.args.alpha * onehot_out.sum() / mask.sum()
            
        mixer.update_lambda(M1, M2, x_mac_out)
        optimiser.zero_grad()
        loss.backward(retain_graph=True)
        grad_norm = th.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip)
        optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("train/loss", loss.item(), t_env)
            self.logger.log_stat("train/hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("train/grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("train/td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("train/q_taken_mean", (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("train/target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

            if self.args.env == "one_step_matrix_game":
                print_matrix_status(batch, self.mixer, mac_out)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        # 训练函数，执行一次子训练并根据需要更新目标网络
        self.sub_train(batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,
                       show_demo=show_demo, save_data=save_data)
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
    def sub_evaluate(self, batch: EpisodeBatch, t_env: int, episode_num: int, mac, mixer, optimiser, params,
                  show_demo=False, save_data=None):
        # 执行一次评估迭代
        # 从 batch 中提取奖励、动作、终止标志等数据
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]

        # 计算估计的 Q 值
        mac_out = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # 按时间维度拼接

        # 选择动作对应的 Q 值
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        # 计算目标 Q 值
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # 忽略第一个时间步的 Q 值
        target_mac_out = th.stack(target_mac_out[1:], dim=1)

        # 屏蔽不可用动作
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # 获取最大化当前 Q 值的动作（用于双重 Q 学习）
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_next_actions = cur_max_actions.detach()

            cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,)).cuda()
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
        else:
            # 计算目标 Q 值
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            # 忽略第一个时间步的 Q 值
            target_mac_out = th.stack(target_mac_out[1:], dim=1)
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        for i in range(chosen_action_qvals.shape[-2]):
            # print("i=",i," agent_qvals= ",chosen_action_qvals[0][i])
            draw.value.append(chosen_action_qvals[0][i])
            draw.step.append(i)
        if mixer is not None:
            if self.args.mixer == "dmaq_qatten":
                ans_chosen, q_attend_regs, head_entropies = \
                    mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv, _, _ = mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                      max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv
                for t in range(batch.max_seq_length - 1):
                    td_error1 = (chosen_action_qvals[0][t][0]) 
                    loss1 = td_error1
                    # Optimise
                    self.optimiser.zero_grad()
                    loss1.backward(retain_graph=True)
                    k = copy.deepcopy(global_Grad.x.grad[0,t])
                    k = k.view(global_Grad.x.grad.shape[-1])
                    draw.value_grad.append(k)
                En = th.stack(draw.value_grad)
                grad_entropy = entropy(En).unsqueeze(-1) * mask
                grad_entropy = grad_entropy.sum(dim=-1)  # / mask.sum()
                fl = 0
                for i in range(batch.max_seq_length - 1):
                    if(mask[0][i][0]==0.0):
                        fl =i
                        break
                grad_entropy = grad_entropy[0][:fl]

                print("grad_entropy:", grad_entropy.mean())

                draw.main(self.args)
                return
            else:
                ans_chosen = mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv = mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv

            if self.args.double_q:
                if self.args.mixer == "dmaq_qatten":
                    target_chosen, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                         actions=cur_max_actions_onehot,
                                                         max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
                else:
                    target_chosen = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                   actions=cur_max_actions_onehot,
                                                   max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], is_v=True)

    def evaluate(self,batch: EpisodeBatch, t_env = None, episode_num = None, show_demo=False, save_data=None):
        # 评估函数，执行一次子评估
        self.sub_evaluate(batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,
                       show_demo=show_demo, save_data=save_data)

    def output_grad(self,mac_out,batch):
        # 计算并返回梯度信息，用于分析和可视化
        actions = batch["actions"][:, :-1]
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]
        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        qvals = chosen_action_qvals
        ans_chosen, q_attend_regs, head_entropies = \
            self.mixer(qvals, batch["state"][:, :-1], is_v=True)
        ans_adv, _, _ = self.mixer(qvals,  batch["state"][:, :-1], actions=actions_onehot,
                              max_q_i=max_action_qvals, is_v=False)
        chosen_action_qvals = ans_chosen + ans_adv
        # for classifier
        grad = th.autograd.grad(chosen_action_qvals.sum(), qvals, create_graph=True)[0]
        return grad

    def _update_targets(self):
        # 更新目标网络的参数
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        # 将模型参数转移到 GPU
        #self.classifier.cuda()
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        # 保存模型参数
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        # 加载模型参数
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path),
                                                      map_location=lambda storage, loc: storage))
