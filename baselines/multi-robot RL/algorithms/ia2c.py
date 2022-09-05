import random

import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from SMART.baselines.utils.networks import MLPNetwork
from SMART.baselines.utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from SMART.baselines.utils.agents import A2CAgent

MSELoss = torch.nn.MSELoss()

class IA2C(object):
    def __init__(self, agent_init_params, alg_types,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False):

        self.alg_types = alg_types
        self.agents = [A2CAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim,
                                 **params)
                       for params in agent_init_params]
        self.nagents = len(self.agents)
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def step(self,index,obs,explore=False):

        return self.agents[index].step(obs,explore)

    def stepAll(self,observations, explore=False):
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                                 observations)]
    def stepNoiseAll(self,observations, explore=False):
        if random.random()<0.1:
            return [a.step(obs+torch.randn_like(obs)*0.1, explore=explore) for a, obs in zip(self.agents,
                                                                   observations)]
        else:
            return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                                   observations)]

    def update(self, sample, agent_i,alg_type='IA2C',parallel=False, logger=None):
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]
        curr_agent.critic_optimizer.zero_grad()
        curr_agent.policy_optimizer.zero_grad()

        trgt_vf_in = torch.cat((*next_obs,),
                                   dim=1)
        #tmp=curr_agent.target_critic(trgt_vf_in)
        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1)))

        vf_in = torch.cat((*obs,), dim=1)
        actual_value = curr_agent.critic(vf_in)
        delta = actual_value-target_value.detach()
        value_loss=MSELoss(actual_value,target_value.detach())
        curr_ac, log_pi, reg = curr_agent.policy(obs[agent_i],return_log_pi=True, regularize=True)

        pol_loss = (-log_pi*delta).mean()
        pol_loss += 1e-3 * reg[0]
        # value_loss.backward(retain_graph=True)
        # pol_loss.backward()
        (value_loss+pol_loss).backward()

        if parallel:
            average_gradients(curr_agent.critic)
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()
        curr_agent.policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': value_loss,
                                'log_pi':log_pi.mean(),
                                'pol_loss': pol_loss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='gpu'):
        #print('enter prep_rollouts')
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, agent_num=12, agent_alg="IA2C",num_in_pol=40,num_out_pol=5,discrete_action=True,
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=32):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []

        for i in range(agent_num):
            num_in_critic = num_in_pol * agent_num
            agent_init_params.append({'num_in_pol': num_in_pol,
                                        'num_out_pol': num_out_pol,
                                        'num_in_critic': num_in_critic})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': agent_alg,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance