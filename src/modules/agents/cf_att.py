import math
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from modules.layer.self_atten import SelfAttention
from torch.nn.parameter import Parameter

# 定义 Kaiming 均匀初始化函数
# 用于初始化权重和偏置
# 参数 mode 决定 fan 的计算方式，gain 是缩放因子
def kaiming_uniform_(tensor_w, tensor_b, mode='fan_in', gain=12 ** (-0.5)):
    fan = nn.init._calculate_correct_fan(tensor_w.data, mode)  # 计算 fan 值
    std = gain / math.sqrt(fan)  # 计算标准差
    bound_w = math.sqrt(3.0) * std  # 权重的均匀分布范围
    bound_b = 1 / math.sqrt(fan)  # 偏置的均匀分布范围
    with th.no_grad():
        tensor_w.data.uniform_(-bound_w, bound_w)  # 初始化权重
        if tensor_b is not None:
            tensor_b.data.uniform_(-bound_b, bound_b)  # 初始化偏置

# 定义一个合并多头注意力输出的模块
class Merger(nn.Module):
    def __init__(self, head, fea_dim):
        super(Merger, self).__init__()
        self.head = head  # 注意力头的数量
        if head > 1:
            self.weight = Parameter(th.Tensor(1, head, fea_dim).fill_(1.))  # 初始化权重参数
            self.softmax = nn.Softmax(dim=1)  # 定义 softmax 函数

    def forward(self, x):
        """
        :param x: 输入张量，形状为 [bs, n_head, fea_dim]
        :return: 输出张量，形状为 [bs, fea_dim]
        """
        if self.head > 1:
            # 如果有多个头，使用 softmax 计算权重并加权求和
            return th.sum(self.softmax(self.weight) * x, dim=1, keepdim=False)
        else:
            # 如果只有一个头，直接返回输入
            return th.squeeze(x, dim=1)

# 定义基于 RNN 和注意力机制的Q值网络
class ATTRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(ATTRNNAgent, self).__init__()
        self.args = args  # 保存参数
        self.input_shape = input_shape  # 输入的形状
        self.q_self = None  # 初始化自注意力 Q 值
        self.n_agents = args.n_agents  # 智能体数量
        self.n_heads = args.hpn_head_num  # 注意力头的数量
        self.rnn_hidden_dim = args.rnn_hidden_dim  # RNN 隐藏层维度
        self.use_q_v = False  # 是否使用 Q 值

        # 定义网络层
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)  # 输入层
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)  # GRU 单元
        
        #定义一个超网络模块 （HPN相关方法）
        self.hyper_ally = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.hpn_hyper_dim),  # 超网络的第一层
            nn.ReLU(inplace=True),  # 激活函数
            nn.Linear(args.hpn_hyper_dim, args.rnn_hidden_dim * args.rnn_hidden_dim * self.n_heads)  # 超网络的第二层
        )
        self.unify_input_heads = Merger(self.n_heads, self.rnn_hidden_dim)  # 合并多头注意力输出

        #注意力模块
        self.att = SelfAttention(input_shape, args.att_heads, args.att_embed_dim) # obsshape + last_a + agent_id==> attention embed size

        self.selfpadding = "Zero"  # 自填充策略

        # 定义全连接层
        self.fc2 = nn.Linear(args.att_heads *  args.att_embed_dim, args.rnn_hidden_dim)
        self.fc_inter = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim * 2, args.rnn_hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.fc_last = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.rnn_hidden_dim, args.n_actions)
        )

    # 重置超网络的参数
    # HPN相关
    def _reset_hypernet_parameters(self, init_type='kaiming'):
        gain = 2 ** (-0.5)  # 缩放因子
        for m in self.hyper_enemy.modules():  # 遍历敌方超网络的所有模块
            if isinstance(m, nn.Linear):  # 如果是线性层
                if init_type == "kaiming":
                    kaiming_uniform_(m.weight, m.bias, gain=gain)  # 使用 Kaiming 初始化
                else:
                    nn.init.xavier_normal_(m.weight.data)  # 使用 Xavier 初始化
                    m.bias.data.fill_(0.)  # 初始化偏置为 0
        for m in self.hyper_ally.modules():  # 遍历友方超网络的所有模块
            if isinstance(m, nn.Linear):
                if init_type == "kaiming":
                    kaiming_uniform_(m.weight, m.bias, gain=gain)
                else:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.)

    # 初始化隐藏状态
    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()  # 返回零张量

    # 前向传播
    def forward(self, inputs, hidden_state):
        device = self.fc1.weight.device  # 获取设备信息
        inputs = inputs.to(device)  # 将输入迁移到设备
        hidden_state = hidden_state.to(device)  # 将隐藏状态迁移到设备

        # RNN 部分
        e = inputs.shape[-1]  # 获取输入的最后一维  # agent_id + obs + last_a
        inputs = inputs.reshape(-1, self.args.n_agents, e)  # 重塑输入张量 #   bs * n * e   
        b, a, e = inputs.size()  # 获取批量大小、智能体数量和特征维度 # 
        x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)  # FC1 [b, a, e]->[b*a,e]->[b*a, h]
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)  # 重塑隐藏状态
        h = self.rnn(x, h_in).view(b, a, -1)  # 通过 GRU 计算隐藏状态 # b , a , h_out

        # 注意力部分
        att = self.att(inputs.view(b, a, -1))    # 计算注意力 # bs ,n , e 
        att = F.relu(self.fc2(att), inplace=True).view(b, a, -1)  # 全连接层处理注意力输出 # b*a*att_out

        # Q 值计算
        q = th.cat((h, att), dim=-1)  # 拼接隐藏状态和注意力输出  # b * a * (h_out + att_out)
        if self.selfpadding == "Zero":
            allay_mask = th.zeros_like(att)  # 零填充
        else:
            allay_mask = th.randn_like(att)  # 随机填充
        q_self = th.cat((h, allay_mask), dim=-1)  # 拼接隐藏状态和填充

        # 全连接层处理 Q 值
        inter = self.fc_inter(q)  # 中间层 
        q = self.fc_last(inter)  # 最后一层 b * a  

        with th.no_grad():
            q_self = self.fc_last(self.fc_inter(q_self))  # 计算自注意力 Q 值

        self.q_self = q_self.view(b, a, -1)  # 保存自注意力 Q 值
        return q.view(b, a, -1), inter.view(b, a, -1), h.view(b, a, -1)  # 返回 Q 值、中间层输出和隐藏状态