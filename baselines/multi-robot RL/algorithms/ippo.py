import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from SMART.baselines.utils.networks import MLPNetwork
from SMART.baselines.utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from SMART.baselines.utils.ppo import PPOAgent
import numpy as np
import copy

MSELoss = torch.nn.MSELoss()

class IPPO(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=32,
                 discrete_action=True):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = 6
        self.alg_types = alg_types
        self.agents = [PPOAgent(**params)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.niter = 0
        self.pol_dev='cpu'
        self.TARGET_UPDATE = 10
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics


    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        # return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
        #                                                          observations)]
        actions,action_one_hots, probs, vals=[],[],[],[]
        for a, obs in zip(self.agents,observations):
            action, prob, val=a.choose_action(obs)
            action_one_hot = np.array([0, 0, 0, 0,0])
            action_one_hot[action] = 1
            actions.append(action)
            action_one_hots.append(action_one_hot)
            probs.append(prob)
            vals.append(val)
        return actions,action_one_hots,probs,vals

    def push(self,observations,actions,probs,vals,rewards,dones):
        #
        #print(rewards)
        for a,observation,action,prob,val,reward,done in zip(self.agents,observations,actions,probs,vals,rewards,dones):
            a.remember(observation,action,prob,val,reward,done)

    def clear(self):
        for a in self.agents:
            a.memory.clear_memory()


    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.actor.train()
            a.critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.actor = fn(a.actor)
                a.critic = fn(a.critic)
            self.pol_dev = device

    def prep_rollouts(self, device='gpu'):
        for a in self.agents:
            a.actor.eval()
            a.critic.eval()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        for a in self.agents:
            a.actor = fn(a.actor)
            a.critic=fn(a.critic)

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                    'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, batch_size=5, n_epochs=12, agent_num=6, agent_alg="PPO", num_in_pol=40, num_out_pol=5,
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=32):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []

        for i in range(agent_num):
            agent_init_params.append({'input_dims': num_in_pol,
                                      'n_actions': num_out_pol,
                                      'batch_size':batch_size,
                                      'n_epochs':n_epochs
                                      })
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': agent_alg,
                     'agent_init_params': agent_init_params,
                    }
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename, map_location="cuda:0")
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance