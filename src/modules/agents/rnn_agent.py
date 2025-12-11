import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        # VAE
        # 如果使用了关键智能体，输入维度翻倍
        if getattr(args, "use_critical_agent_obs", False):
            self.logger.info("Using Critical Agent Obs augmentation. Input shape doubled.")
            input_shape = input_shape * 2

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state=None):
        #通过view()改变张量形状（shape），但不会改变张量实际存储的数据（即不复制数据，仅改变“视图”）
        b, a, e = inputs.size() # bs * n * (agent_id + obs + last_a)
        
        x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)#[b, a, e]->[b*a,e]->[b*a, h]
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, hidden_state)
        q = self.fc2(h)
        return q.view(b, a, -1), h.view(b, a, -1)#[b*a,h]->[b,a,h]