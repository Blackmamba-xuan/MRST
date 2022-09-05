import torch
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from SMART.baselines.utils.misc import hard_update, gumbel_softmax, onehot_from_logits
import numpy as np
from SMART.baselines.utils.pmNets import AttentionMLPNetwork, MLPNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise
from .policies import DiscretePolicy

class PMAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol,attention_dim, num_in_pm, hidden_dim=16,
                 lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = AttentionMLPNetwork(num_in_pol, num_out_pol,attention_dim,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False,
                                 discrete_action=discrete_action)
        self.target_policy = AttentionMLPNetwork(num_in_pol, num_out_pol,attention_dim,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False,
                                        discrete_action=discrete_action)
        self.pmNet_1 = MLPNetwork(num_in_pm, num_out_pol,norm_in=False,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.pmNet_2 = MLPNetwork(num_in_pm, num_out_pol,norm_in=False,
                                  hidden_dim=hidden_dim,
                                  constrain_out=False)

        hard_update(self.target_policy, self.policy)
        self.num_out_pol=num_out_pol
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.pmNet_optimizer1 = Adam(self.pmNet_1.parameters(), lr=lr)
        self.pmNet_optimizer2 = Adam(self.pmNet_2.parameters(), lr=lr)
        self.EPSILON = 1.0
        self.EPS_MIN = 0.01
        self.EPS_DEC = 0.996

    def step(self, obs_x1,obs_x2, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        rand = np.random.random()
        if rand> self.EPSILON:
            action_prob1=self.pmNet_1.forward(obs_x1)
            action_prob1_soft=torch.softmax(action_prob1,dim=1)
            action_prob2=self.pmNet_2.forward(obs_x1)
            action_prob2_soft=torch.softmax(action_prob2,dim=1)
            partn_ac_prob=torch.cat((action_prob1_soft,action_prob2_soft),dim=1)
            q_values = self.policy.forward(obs_x1,torch.cat((obs_x2,partn_ac_prob),dim=1))
            action_index = q_values.max(1)[1].item()
        else:
            action_index = np.random.choice(self.num_out_pol)
        action = np.array([0, 0, 0, 0])
        action[action_index] = 1
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'pmNet_1': self.pmNet_1.state_dict(),
                'pmNet_2': self.pmNet_2.state_dict()
                }

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.pmNet_1.load_state_dict(params['pmNet_1'])
        self.pmNet_2.load_state_dict(params['pmNet_2'])

class PMAttentionAgent(object):
    """
    General class for Attention agents (policy, target policy)
    """
    def __init__(self, num_in_pol, num_out_pol, hidden_dim=64,
                 lr=0.01, onehot_dim=0):

        self.policy = DiscretePolicy(num_in_pol, num_out_pol,
                                     hidden_dim=hidden_dim,
                                     onehot_dim=onehot_dim)
        self.target_policy = DiscretePolicy(num_in_pol,
                                            num_out_pol,
                                            hidden_dim=hidden_dim,
                                            onehot_dim=onehot_dim)
        self.pmNet_1 = MLPNetwork(num_in_pol, num_out_pol, norm_in=False,
                                  hidden_dim=hidden_dim,
                                  constrain_out=False)
        self.pmNet_2 = MLPNetwork(num_in_pol, num_out_pol, norm_in=False,
                                  hidden_dim=hidden_dim,
                                  constrain_out=False)

        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

    def step(self, obs, explore=False):

        return self.policy(obs, sample=explore)

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'pmNet_1': self.pmNet_1.state_dict(),
                'pmNet_2': self.pmNet_2.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.pmNet_1.load_state_dict(params['pmNet_1'])
        self.pmNet_2.load_state_dict(params['pmNet_2'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])