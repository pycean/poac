import torch.nn as nn
import torch.nn.functional as F
import torch as th

class RNNAgent(nn.Module):
    def __init__(self, task2input_shape_info, task2n_agents,args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.n_agents = task2n_agents
        self.task_repre_dim = args.task_repre_dim
        input_shape = task2input_shape_info["transbq1"]["input_shape"]
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim + self.task_repre_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state,task_repre,task):
        # bs = inputs.shape[0] // self.n_agents[task]
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        h_new = th.cat([h, task_repre], dim=-1)
        q = self.fc2(h_new)
        return q, h

