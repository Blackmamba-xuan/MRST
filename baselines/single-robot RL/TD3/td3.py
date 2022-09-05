import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple, deque
from itertools import count
import random
import math
from torch.autograd import Variable
from misc import hard_update, gumbel_softmax, onehot_from_logits, average_gradients, soft_update
from torch.optim import Adam
from noise import OUNoise
from torch import Tensor
import numpy as np

device = torch.device("cpu")
MSELoss = torch.nn.MSELoss()

class MLPNetwork(nn.Module):

    def __init__(self, h, w,n_actions,attention_dim=2):
        super(MLPNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.n_actions=n_actions

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, 32)
        self.out = nn.Linear(32+attention_dim, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, speed):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))
        linear_in = torch.cat((x.view(x.size(0), -1), speed), dim=1)
        return self.out(linear_in)

class TD3Agent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, h, w, n_actions, hidden_dim=64,
                 lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = MLPNetwork(h, w, n_actions).to(device)
        self.critic_1 = MLPNetwork(h, w, 1, attention_dim=11).to(device)
        self.critic_2 = MLPNetwork(h, w, 1, attention_dim=11).to(device)
        self.target_policy = MLPNetwork(h, w, n_actions).to(device)
        self.target_critic_1 = MLPNetwork(h, w, 1, attention_dim=11).to(device)
        self.target_critic_2 = MLPNetwork(h, w, 1, attention_dim=11).to(device)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic_1, self.critic_1)
        hard_update(self.target_critic_2, self.critic_2)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer_1 = Adam(self.critic_1.parameters(), lr=lr)
        self.critic_optimizer_2 = Adam(self.critic_2.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(n_actions)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

        self.gamma = 0.95
        self.niter=0
        self.tau = 0.01
        self.pol_dev="cpu"
        self.critic_dev="cpu"
        self.trgt_pol_dev="cpu"
        self.trgt_critic_dev="cpu"

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def choose_action(self, state,speed, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(state,speed)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic_1': self.critic_1.state_dict(),
                'critic_2': self.critic_2.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic_1': self.target_critic_1.state_dict(),
                'target_critic_2': self.target_critic_2.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer_1': self.critic_optimizer_1.state_dict(),
                'critic_optimizer_2': self.critic_optimizer_2.state_dict(),
                }

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic_1.load_state_dict(params['critic_1'])
        self.critic_2.load_state_dict(params['critic_2'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic_1.load_state_dict(params['target_critic_1'])
        self.target_critic_2.load_state_dict(params['target_critic_2'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer_1.load_state_dict(params['critic_optimizer_1'])
        self.critic_optimizer_2.load_state_dict(params['critic_optimizer_2'])

    def update(self, sample, agent_i,alg_type='MADDPG',parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs,speeds, acs, rews, next_obs,next_speeds, dones = sample
        for speed in speeds[0]:
            print('speed : ', speed)
        self.critic_optimizer_1.zero_grad()
        self.critic_optimizer_2.zero_grad()
        if self.discrete_action:
            trgt_acs=onehot_from_logits(self.target_policy(next_obs[agent_i], next_speeds[agent_i]))
        else:
            trgt_acs=self.target_policy(next_obs[agent_i], next_speeds[agent_i])
        trgt_acs = trgt_acs + torch.clamp(torch.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        trgt_vf_in = torch.cat((next_speeds[agent_i], trgt_acs), dim=1)
        #tmp=curr_agent.target_critic(trgt_vf_in)
        q1=self.target_critic_1(next_obs[agent_i],trgt_vf_in)
        q2=self.target_critic_2(next_obs[agent_i],trgt_vf_in)
        critic_value=torch.min(q1,q2)
        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        critic_value *(1 - dones[agent_i].view(-1, 1)))
        vf_in = torch.cat((speeds[agent_i], acs[agent_i]), dim=1)
        actual_value_1 = self.critic_1(obs[agent_i],vf_in)
        actual_value_2 = self.critic_2(obs[agent_i],vf_in)
        vf_loss_1 = MSELoss(actual_value_1, target_value.detach())
        vf_loss_2 = MSELoss(actual_value_2, target_value.detach())
        vf_loss=vf_loss_1+vf_loss_2
        vf_loss.backward()
        if parallel:
            average_gradients(self.critic_1)
            average_gradients(self.critic_2)
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 0.5)
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.step()

        self.policy_optimizer.zero_grad()

        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = self.policy(obs[agent_i],speeds[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = self.policy(obs[agent_i],speeds[agent_i])
            curr_pol_vf_in = curr_pol_out

        vf_in = torch.cat((speeds[agent_i], curr_pol_vf_in),
                            dim=1)
        pol_loss = -self.critic_1(obs[agent_i],vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()
        if parallel:
            average_gradients(self.policy)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic_1, self.critic_1, self.tau)
        soft_update(self.target_critic_2, self.critic_2, self.tau)
        soft_update(self.target_policy, self.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        self.policy.train()
        self.critic_1.train()
        self.critic_2.train()
        self.target_policy.train()
        self.target_critic_1.train()
        self.target_critic_2.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            self.policy = fn(self.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic_1 = fn(self.critic_1)
            self.critic_2 = fn(self.critic_2)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            self.target_policy = fn(self.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic_1 = fn(self.target_critic_1)
            self.target_critic_2 = fn(self.target_critic_2)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        #print('enter prep_rollouts')
        self.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            self.policy = fn(self.policy)
            self.pol_dev = device