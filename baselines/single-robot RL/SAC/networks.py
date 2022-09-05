import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=128, fc2_dims=128,
            name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.critic_conv1 = nn.Conv2d(input_dims[0], 16, kernel_size=5, stride=2)
        self.critic_bn1 = nn.BatchNorm2d(16)
        self.critic_conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.critic_bn2 = nn.BatchNorm2d(32)
        self.critic_conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.critic_bn3 = nn.BatchNorm2d(32)
        critic_convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(input_dims[1])))
        critic_convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(input_dims[2])))
        critic_linear_input_size = critic_convw * critic_convh * 32

        self.fc1 = nn.Linear(critic_linear_input_size+2+n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def conv2d_size_out(self,size, kernel_size=5, stride=2):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, state,speeds, action):
        x = F.relu(self.critic_bn1(self.critic_conv1(state)))
        x = F.relu(self.critic_bn2(self.critic_conv2(x)))
        x = F.relu(self.critic_bn3(self.critic_conv3(x)))
        linear_in = T.cat((x.view(x.size(0), -1), speeds), dim=1)
        linear_in = T.cat((linear_in,action),dim=1)
        # linear_in = x.view(x.size(0), -1)

        action_value = self.fc1(linear_in)
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self, index):
        save_name = os.path.join(self.checkpoint_dir, self.name + '_'+ str(index) + '_sac')
        T.save(self.state_dict(), save_name)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=128, fc2_dims=128,
            name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.critic_conv1 = nn.Conv2d(input_dims[0], 16, kernel_size=5, stride=2)
        #self.critic_bn1 = nn.BatchNorm2d(16)
        self.critic_conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        #self.critic_bn2 = nn.BatchNorm2d(32)
        self.critic_conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        #self.critic_bn3 = nn.BatchNorm2d(32)
        critic_convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(input_dims[1])))
        critic_convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(input_dims[2])))
        critic_linear_input_size = critic_convw * critic_convh * 32

        self.fc1 = nn.Linear(critic_linear_input_size+2, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def conv2d_size_out(self,size, kernel_size=5, stride=2):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, state, speeds):
        # x = F.relu(self.critic_bn1(self.critic_conv1(state)))
        # x = F.relu(self.critic_bn2(self.critic_conv2(x)))
        # x = F.relu(self.critic_bn3(self.critic_conv3(x)))
        x = F.relu(self.critic_conv1(state))
        x = F.relu(self.critic_conv2(x))
        x = F.relu(self.critic_conv3(x))
        linear_in = T.cat((x.view(x.size(0), -1), speeds), dim=1)
        # linear_in = x.view(x.size(0), -1)

        state_value = self.fc1(linear_in)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self, index):
        save_name = os.path.join(self.checkpoint_dir, self.name+'_'+str(index) +'_sac')
        T.save(self.state_dict(), save_name)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=128,
            fc2_dims=128, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.actor_conv1 = nn.Conv2d(input_dims[0], 16, kernel_size=5, stride=2)
        self.actor_bn1 = nn.BatchNorm2d(16)
        self.actor_conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.actor_bn2 = nn.BatchNorm2d(32)
        self.actor_conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.actor_bn3 = nn.BatchNorm2d(32)
        actor_convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(input_dims[1])))
        actor_convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(input_dims[2])))
        actor_linear_input_size = actor_convw * actor_convh * 32

        self.fc1 = nn.Linear(actor_linear_input_size+2, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def conv2d_size_out(self,size, kernel_size=5, stride=2):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, state, speeds):
        x = F.relu(self.actor_bn1(self.actor_conv1(state)))
        x = F.relu(self.actor_bn2(self.actor_conv2(x)))
        x = F.relu(self.actor_bn3(self.actor_conv3(x)))
        linear_in = T.cat((x.view(x.size(0), -1), speeds), dim=1)

        prob = self.fc1(linear_in)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state,speeds, reparameterize=True):
        mu, sigma = self.forward(state,speeds)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self, index):
        save_name=os.path.join(self.checkpoint_dir, self.name+'_'+str(index) +'_sac')
        T.save(self.state_dict(), save_name)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))