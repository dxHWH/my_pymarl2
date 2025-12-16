import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 1. 定义多头图注意力层 (Multi-Head GAT) ---
# 对应论文公式 (8), (9), (10)
# 这里的图是全连接的 (Fully Connected)，所以我们不需要邻接矩阵，直接做 Attention
class MultiHeadGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads):
        super(MultiHeadGAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        # W 矩阵: 将输入的 hidden_state 映射到 GAT 的特征空间
        # 输出维度: n_heads * hidden_dim
        self.W = nn.Linear(input_dim, n_heads * hidden_dim, bias=False)
        
        # Attention 向量 a: 用于计算节点间的注意力权重
        # 输入是拼接的两个节点特征 [Wh_i || Wh_j]，所以是 2 * hidden_dim
        self.att_a = nn.Parameter(th.Tensor(1, n_heads, 2 * hidden_dim)) # 创建一个矩阵并且将其变成可学习的
        nn.init.xavier_uniform_(self.att_a.data, gain=1.414) # 初始化
        
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h):
        # h shape: (batch_size, n_agents, input_dim)
        bs, n_agents, _ = h.size()
        
        # 1. 线性变换: h -> Wh
        # shape: (bs, n_agents, n_heads * hidden_dim)
        h_prime = self.W(h) 
        # 重塑为多头: (bs, n_agents, n_heads, hidden_dim)
        h_prime = h_prime.view(bs, n_agents, self.n_heads, self.hidden_dim)
        # 转置以便后续广播: (bs, n_heads, n_agents, hidden_dim)，变换维度
        h_prime = h_prime.permute(0, 2, 1, 3)
        
        # 2. 构建注意力机制 (Fully Connected Graph)
        # 我们需要计算所有 i 和 j 的组合。
        # 扩展维度进行广播:
        # h_i: (bs, heads, n_agents, 1, hidden_dim)
        # h_j: (bs, heads, 1, n_agents, hidden_dim)
        h_i = h_prime.unsqueeze(3)
        h_j = h_prime.unsqueeze(2)
        
        # 拼接: [Wh_i || Wh_j] -> (bs, heads, n_agents, n_agents, 2*hidden_dim)
        # repeat 使得维度匹配
        h_cat = th.cat([h_i.repeat(1, 1, 1, n_agents, 1), 
                        h_j.repeat(1, 1, n_agents, 1, 1)], dim=-1)
        
        # 计算 e_ij (公式 10)
        # (bs, heads, n_agents, n_agents, 2*hid) * (1, heads, 1, 1, 2*hid) -> sum -> scalar
        # attention score: (bs, heads, n_agents, n_agents)
        e = (h_cat * self.att_a.unsqueeze(2).unsqueeze(3)).sum(dim=-1)
        e = self.leaky_relu(e)
        
        # 计算 alpha_ij (公式 9)
        attention = F.softmax(e, dim=-1) # 对 j 维度做 softmax
        
        # 3. 聚合信息 (公式 8)
        # 这里实现了加权求和的操作，其中attention就相当于权重，h_prime就是两个智能体之间的特征信息
        # alpha: (bs, heads, n_agents, n_agents)
        # h_prime (h_j): (bs, heads, n_agents, hidden_dim)
        # matmul: (bs, heads, n_agents, n_agents) @ (bs, heads, n_agents, hidden_dim)
        #      -> (bs, heads, n_agents, hidden_dim)
        h_new = th.matmul(attention, h_prime)
        
        # 应用激活函数 sigma (通常是 ELU 或 ReLU)
        h_new = F.elu(h_new)
        
        return h_new # 输出 G^d (batch_size, n_heads, n_agents, hidden_dim)


# --- 2. 定义 DVD Mixer ---
class DVDMixer(nn.Module):
    def __init__(self, args):
        super(DVDMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim
        self.abs = getattr(self.args, 'abs', True) # QMIX 的单调性约束
        
        # DVD 特有参数
        # 这里的 hidden_state_dim 需要和你使用的 RNN (GRU/LSTM) 的输出维度一致
        self.rnn_hidden_dim = args.rnn_hidden_dim 
        self.n_heads = getattr(args, 'dvd_heads', 4) # 采样次数 D, 论文推荐 4 或 8
        self.gat_dim = getattr(args, 'gat_embed_dim', 32) # GAT 内部特征维度

        # --- 组件 1: 轨迹图生成器 (GAT) ---
        self.gat = MultiHeadGAT(self.rnn_hidden_dim, self.gat_dim, self.n_heads)

        # --- 组件 2: 状态超网络 (用于生成 W1) ---
        # 论文公式 (11): K = |f_s(s) * G|
        # 我们需要生成一个矩阵来处理 GAT 的输出，最终得到 (n_agents, embed_dim) 的权重
        # f_s(s) 输出维度: n_heads * embed_dim * gat_dim
        self.hyper_w_1_state = nn.Linear(self.state_dim, self.n_heads * self.embed_dim * self.gat_dim)
        
        # W1 的偏置 (和 QMIX 一样，仅依赖 State)
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # --- 组件 3: 第二层混合 (W_final) ---
        # 对于第二层，通常保持 QMIX 的原样，或者也用 DVD。
        # 为了稳定性，通常只将 DVD 应用于第一层 (Credit Assignment 主要是第一层的作用)
        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        else:
            # 2层超网络
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.ReLU(inplace=True),
                                               nn.Linear(hypernet_embed, self.embed_dim))
            
        # V(s) 状态价值偏置
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, hidden_states):
        """
        Args:
            agent_qs: (batch, T, n_agents)
            states: (batch, T, state_dim)
            hidden_states: (batch, T, n_agents, rnn_hidden_dim) [新增输入!]
        """
        bs = agent_qs.size(0) # batch_size
        
        # 数据展平，处理时间维度的 Batch
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
        # hidden_states: (bs*T, n_agents, rnn_dim)
        hidden_states = hidden_states.reshape(-1, self.n_agents, self.rnn_hidden_dim)

        # -----------------------------------------------------------
        # Step 1: 通过 GAT 构建轨迹图并采样 (论文公式 8-10)
        # -----------------------------------------------------------
        # graphs_out: (bs*T, n_heads, n_agents, gat_dim)
        # 这就是论文中的 G^d
        graphs_out = self.gat(hidden_states)

        # -----------------------------------------------------------
        # Step 2: 计算第一层权重 W1 (论文公式 11 & 12)
        # -----------------------------------------------------------
        # 2.1 生成状态表示 f_s(s)
        # w1_state: (bs*T, n_heads * embed_dim * gat_dim)
        w1_state = self.hyper_w_1_state(states)
        # 重塑: (bs*T, n_heads, embed_dim, gat_dim)
        w1_state = w1_state.view(-1, self.n_heads, self.embed_dim, self.gat_dim)
        
        # 2.2 结合 State 和 Graph 计算 K^d (Credits)
        # 这里的运算对应公式 (11): f_s(s) * G^d
        # 我们使用矩阵乘法来实现: (embed, gat) @ (gat, agents) -> (embed, agents)
        
        # 调整 G 的维度: (bs*T, n_heads, gat_dim, n_agents)
        graphs_T = graphs_out.permute(0, 1, 3, 2)
        
        # MatMul: 
        # (bs*T, heads, embed, gat) @ (bs*T, heads, gat, agents) 
        # -> (bs*T, heads, embed, agents)
        w1_heads = th.matmul(w1_state, graphs_T)
        
        # 2.3 绝对值约束 (Monotonicity)
        if self.abs:
            w1_heads = th.abs(w1_heads)
            
        # 2.4 后门调整 (平均化) - 公式 (12)
        # 对 n_heads 维度求平均
        # (bs*T, heads, embed, agents) -> (bs*T, embed, agents)
        w1 = w1_heads.mean(dim=1)
        
        # 转置以匹配 QMIX 的乘法顺序: (bs*T, n_agents, embed_dim)
        w1 = w1.permute(0, 2, 1)

        # -----------------------------------------------------------
        # Step 3: 标准的 QMIX 计算流程
        # -----------------------------------------------------------
        # 生成 b1
        b1 = self.hyper_b_1(states).view(-1, 1, self.embed_dim)
        
        # 第1层混合: hidden = ELU( Q * W1 + b1 )
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        
        # 生成 W_final
        w_final = self.hyper_w_final(states)
        if self.abs:
            w_final = th.abs(w_final)
        w_final = w_final.view(-1, self.embed_dim, 1)
        
        # 生成 V(s)
        v = self.V(states).view(-1, 1, 1)
        
        # 第2层混合: Q_tot = hidden * W_final + V
        y = th.bmm(hidden, w_final) + v
        
        # 重塑回 (batch, T, 1)
        q_tot = y.view(bs, -1, 1)
        
        return q_tot


# import torch as th
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np


# class MultiHeadGAT(nn.Module):
#     """
#     多头图注意力层 (Multi-Head Graph Attention Layer)
#     对应论文公式 (8), (9), (10) [cite: 188-196]
#     作用: 构建轨迹图 (Trajectory Graph) 并提取智能体间的关系特征
#     """

# # --- 1. 定义多头图注意力层 (Multi-Head GAT) ---
# # 对应论文公式 (8), (9), (10)
# # 这里的图是全连接的 (Fully Connected)，所以我们不需要邻接矩阵，直接做 Attention
# class MultiHeadGAT(nn.Module):

#     def __init__(self, input_dim, hidden_dim, n_heads):
#         super(MultiHeadGAT, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.n_heads = n_heads

#         # W 矩阵: 线性变换 h -> h' [cite: 189]
#         self.W = nn.Linear(input_dim, n_heads * hidden_dim, bias=False)
        
#         # Attention 向量 a: 用于计算 e_ij [cite: 193]
#         # 输入是拼接的 [Wh_i || Wh_j], 维度为 2 * hidden_dim
#         self.att_a = nn.Parameter(th.Tensor(1, n_heads, 2 * hidden_dim))
#         nn.init.xavier_uniform_(self.att_a.data, gain=1.414)

        
#         self.leaky_relu = nn.LeakyReLU(0.2)

#     def forward(self, h):

#         # h: (batch_size, n_agents, input_dim)
#         bs, n_agents, _ = h.size()
        
#         # 1. 线性变换与多头重塑
#         h_prime = self.W(h) # (bs, n_agents, n_heads * hidden_dim)
#         h_prime = h_prime.view(bs, n_agents, self.n_heads, self.hidden_dim)
#         h_prime = h_prime.permute(0, 2, 1, 3) # (bs, heads, n_agents, hidden_dim)
        
#         # 2. 构建全连接图的注意力机制 [cite: 183]
#         # 准备广播以计算所有 i, j 对
#         h_i = h_prime.unsqueeze(3) # (bs, heads, n_agents, 1, hidden)
#         h_j = h_prime.unsqueeze(2) # (bs, heads, 1, n_agents, hidden)
        
#         # 拼接特征
#         h_cat = th.cat([h_i.repeat(1, 1, 1, n_agents, 1), 
#                         h_j.repeat(1, 1, n_agents, 1, 1)], dim=-1)
        
#         # 计算注意力系数 e_ij (公式 10)
#         e = (h_cat * self.att_a.unsqueeze(2).unsqueeze(3)).sum(dim=-1)
#         e = self.leaky_relu(e)
        
#         # 计算归一化权重 alpha_ij (公式 9)
#         attention = F.softmax(e, dim=-1) 
        
#         # 3. 聚合信息生成新特征 (公式 8)
#         # (bs, heads, n_agents, n_agents) @ (bs, heads, n_agents, hidden)
#         h_new = th.matmul(attention, h_prime)
#         h_new = F.elu(h_new) # 论文公式 (8) 使用 sigma 激活函数
        
#         return h_new # 输出 G^d, shape: (bs, heads, n_agents, hidden_dim)


# class DVDMixer(nn.Module):
#     """
#     Deconfounded Value Decomposition (DVD) Mixer
#     基于 QMIX 架构，但在 Credit Assignment (Layer 2) 引入因果推断的轨迹图
#     """

#     def __init__(self, args):
#         super(DVDMixer, self).__init__()
#         self.args = args
#         self.n_agents = args.n_agents
#         self.state_dim = int(np.prod(args.state_shape))
#         self.embed_dim = args.mixing_embed_dim

#         self.abs = getattr(self.args, 'abs', True) # 单调性约束 [cite: 200]
        
#         # DVD 参数
#         self.rnn_hidden_dim = args.rnn_hidden_dim 
#         self.n_heads = getattr(args, 'dvd_heads', 4) # 采样次数 D [cite: 186]
#         self.gat_dim = getattr(args, 'gat_embed_dim', 32)

#         # --- 1. 轨迹图生成器 (G, Trajectory Graph) ---
#         # 对应图 1(c) 中的 G 生成过程
#         self.gat = MultiHeadGAT(self.rnn_hidden_dim, self.gat_dim, self.n_heads)

#         # --- 2. 第一层混合网络 (QMIX Standard) ---
#         # 论文提到 DVD 也可以应用于 QMIX[cite: 59], 
#         # 这里的 Factorize 过程 (Q_local -> Q_inter) 通常保持标准 QMIX 方式
#         # 生成 W1: (State) -> (n_agents * embed_dim)
#         self.hyper_w_1 = nn.Linear(self.state_dim, self.n_agents * self.embed_dim)
#         # 生成 b1: (State) -> (embed_dim)
#         self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

#         # --- 3. 第二层混合网络 (DVD Credit Assignment) ---
#         # 对应论文中的 "Credit Assignment" 步骤 (公式 11-13)
#         # 这里的 K (即 W_final) 需要由 State 和 Graph 共同决定 [cite: 168]
#         # 公式 (11): K^d = |f_s(s) * G^d|
        
#         # 我们将 G^d 展平: (n_agents * gat_dim)
#         # f_s(s) 需要输出一个矩阵，能把 G^d 映射为 (embed_dim, 1)
#         # 所以 f_s(s) 输出维度: n_heads * embed_dim * (n_agents * gat_dim)
#         self.hyper_w_final_dvd = nn.Linear(self.state_dim, 
#                                            self.n_heads * self.embed_dim * (self.n_agents * self.gat_dim))

#         # V(s): 全局状态价值偏置 (公式 14 中的 Bias 部分)

#         self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
#                                nn.ReLU(inplace=True),
#                                nn.Linear(self.embed_dim, 1))

#     def forward(self, agent_qs, states, hidden_states):
#         """
#         Args:
#             agent_qs: (batch, T, n_agents)
#             states: (batch, T, state_dim)

#             hidden_states: (batch, T, n_agents, rnn_hidden_dim) - 用于构建图
#         """
#         bs = agent_qs.size(0)
#         states = states.reshape(-1, self.state_dim)
#         agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
#         hidden_states = hidden_states.reshape(-1, self.n_agents, self.rnn_hidden_dim)

#         # -----------------------------------------------------------
#         # Step 1: 标准 QMIX 第一层 (Mixing)
#         # Q_tot = W2 * (W1 * Q + b1) + V
#         # 这里计算 Q_inter = W1 * Q + b1
#         # -----------------------------------------------------------
        
#         # 生成 W1: (bs*T, n_agents, embed_dim)
#         w1 = th.abs(self.hyper_w_1(states)) if self.abs else self.hyper_w_1(states)
#         w1 = w1.view(-1, self.n_agents, self.embed_dim)
        
#         # 生成 b1: (bs*T, 1, embed_dim)
#         b1 = self.hyper_b_1(states).view(-1, 1, self.embed_dim)
        
#         # 计算中间层 hidden: (bs*T, 1, embed_dim)
#         # 公式 (17): Q_inter = W * Q_local [cite: 587]
#         hidden = F.elu(th.bmm(agent_qs, w1) + b1)

#         # -----------------------------------------------------------
#         # Step 2: DVD Credit Assignment (计算 K / W_final)
#         # -----------------------------------------------------------
        
#         # 2.1 生成轨迹图 G^d (公式 8-10)
#         # graphs_out: (bs*T, heads, n_agents, gat_dim)
#         graphs_out = self.gat(hidden_states)
        
#         # 展平 Graph 以便进行矩阵运算: (bs*T, heads, n_agents * gat_dim, 1)
#         # 这里我们将图特征视为一个整体向量
#         graphs_flat = graphs_out.reshape(-1, self.n_heads, self.n_agents * self.gat_dim, 1)

#         # 2.2 生成状态表示 f_s(s) (公式 11 中的 f_s)
#         # weights_dvd: (bs*T, heads, embed_dim, n_agents * gat_dim)
#         weights_dvd = self.hyper_w_final_dvd(states)
#         weights_dvd = weights_dvd.view(-1, self.n_heads, self.embed_dim, self.n_agents * self.gat_dim)

#         # 2.3 计算 Credits K^d (公式 11)
#         # K^d = |f_s(s) * G^d|
#         # Matmul: (heads, embed, agent*gat) @ (heads, agent*gat, 1) -> (heads, embed, 1)
#         k_d = th.matmul(weights_dvd, graphs_flat)

#         # 2.4 后门调整/平均化 (公式 12)
#         # K = Mean(K^d)
#         # w_final: (bs*T, embed_dim, 1)
#         w_final = k_d.mean(dim=1) 
        
#         # 绝对值约束 (Monotonicity) [cite: 200]
#         if self.abs:
#             w_final = th.abs(w_final)

#         # -----------------------------------------------------------
#         # Step 3: 最终聚合
#         # -----------------------------------------------------------
        
#         # 生成全局 V(s)
#         v = self.V(states).view(-1, 1, 1)
        
#         # 计算 Q_tot (公式 13)
#         # Q_tot = Sum(K * Q_inter) + V
#         # bmm: (bs*T, 1, embed) @ (bs*T, embed, 1) -> (bs*T, 1, 1)

#         y = th.bmm(hidden, w_final) + v
        
#         # 重塑回 (batch, T, 1)
#         q_tot = y.view(bs, -1, 1)
        
#         return q_tot
