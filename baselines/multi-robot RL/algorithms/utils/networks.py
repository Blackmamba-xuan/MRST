import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

device = torch.device("cuda:0")
class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=False, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = torch.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out

class RnnNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):

        super(RnnNetwork, self).__init__()
        self.hidden_dim=hidden_dim
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.GRUCell(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X, hidden_state):
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h_in=hidden_state.reshape(-1,self.hidden_dim)
        h_out = self.fc2(h1, h_in)
        h2 = self.nonlin(h_out)
        out = self.out_fn(self.fc3(h2))
        return out, h_out

class QMixNet(nn.Module):
    def __init__(self, nagents, state_space, hyper_hidden_dim, qmix_hidden_dim):
        super(QMixNet, self).__init__()
        self.nagents=nagents
        self.qmix_hidden_dim=qmix_hidden_dim
        self.hyper_w1 = nn.Sequential(nn.Linear(state_space, hyper_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hyper_hidden_dim, self.nagents * qmix_hidden_dim)).to(device)
        self.hyper_b1 = nn.Linear(state_space, qmix_hidden_dim).to(device)

        self.hyper_w2 = nn.Sequential(nn.Linear(state_space, hyper_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hyper_hidden_dim, qmix_hidden_dim)).to(device)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_space, qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(qmix_hidden_dim, 1)).to(device)
    def forward(self,q_values, states):
        q_values = q_values.view(-1, 1, self.nagents)
        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = w1.view(-1, self.nagents, self.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.qmix_hidden_dim)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = w2.view(-1, self.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)  # (1920, 1， 1)

        q_total = torch.bmm(hidden, w2) + b2
        return q_total.view(-1, 1)