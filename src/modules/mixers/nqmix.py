import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NQMixer(nn.Module):
    def __init__(self, args):
        super(NQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim
        self.abs = getattr(self.args, 'abs', True)

        self.power_num = getattr(args, "power_num", 16)
        # 生成矩阵
        self.hyper_list = nn.ModuleList()

        if getattr(args, "hypernet_layers", 1) == 1:
            for i in range(self.power_num):
                hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
                hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
                hyper_unit = nn.ModuleList([
                    hyper_w_1,
                    hyper_w_final
                ])
                self.hyper_list.append(hyper_unit)
        elif getattr(args, "hypernet_layers", 1) == 2:
            for i in range(self.power_num):
                hypernet_embed = self.args.hypernet_embed
                hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
                hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hypernet_embed, self.embed_dim))
                hyper_unit = nn.ModuleList([
                    hyper_w_1,
                    hyper_w_final
                ])
                self.hyper_list.append(hyper_unit)
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.embed_dim, 1))
        
        # weights_last_layer
        self.weigthts_last_layer = nn.Linear(self.state_dim, self.power_num)
        

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)

        # 创建一个y的列表
        y_list = []
        # 计算每个y
        for i in range(self.power_num):
            hyper_w_1, hyper_w_final = self.hyper_list[i]

            
            # First layer
            w1 = hyper_w_1(states).abs() if self.abs else hyper_w_1(states)
            b1 = self.hyper_b_1(states)
            w1 = w1.view(-1, self.n_agents, self.embed_dim)
            b1 = b1.view(-1, 1, self.embed_dim)
            hidden = F.elu(th.bmm(agent_qs, w1) + b1)

            # Second layer
            w_final = hyper_w_final(states).abs() if self.abs else hyper_w_final(states)
            w_final = w_final.view(-1, self.embed_dim, 1)
            # State-dependent bias
            v = self.V(states).view(-1, 1, 1)
            # Compute final output
            y = th.bmm(hidden, w_final) + v #(batch, 1, 1)
            # Reshape and return
            y = y.pow(2*i+1)
            y_list.append(y)
        y = th.cat(y_list, dim=-1) # (batch*max_seq_len,1, power_num)
        # 计算权重
        weights = self.weigthts_last_layer(states).abs() if self.abs else self.weigthts_last_layer(states) #(batch*max_seq_len, power_num)
        weights = weights.view(-1, self.power_num, 1) # (batch*max_seq_len, power_num, 1)
        # 计算加权和
        y = th.bmm(y, weights)
        # 计算最终的输出
        q_tot = y.view(bs, -1, 1)
        # 计算最终的输出
        return q_tot
            
    def k(self, states):
        bs = states.size(0)
        w1 = th.abs(self.hyper_w_1(states))
        w_final = th.abs(self.hyper_w_final(states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        w_final = w_final.view(-1, self.embed_dim, 1)
        k = th.bmm(w1,w_final).view(bs, -1, self.n_agents)
        k = k / th.sum(k, dim=2, keepdim=True)
        return k

    def b(self, states):
        bs = states.size(0)
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        b1 = self.hyper_b_1(states)
        b1 = b1.view(-1, 1, self.embed_dim)
        v = self.V(states).view(-1, 1, 1)
        b = th.bmm(b1, w_final) + v
        return b
