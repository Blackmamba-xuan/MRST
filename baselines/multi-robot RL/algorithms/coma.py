import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from SMART.baselines.utils.networks import MLPNetwork
from SMART.baselines.utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax, hard_update
from SMART.baselines.utils.agents import ComaAgent
from torch.optim import Adam
import copy

MSELoss = torch.nn.MSELoss()

class COMA(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params,alg_types,num_in_critic,num_out_critic,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False, batch_size=1024):
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
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agents = [ComaAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim,
                                 **params)
                       for params in agent_init_params]
        self.critic = MLPNetwork(num_in_critic, num_out_critic,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.target_critic = MLPNetwork(num_in_critic, num_out_critic,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
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
        self.batch_size=batch_size

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    # step all agents
    def stepAll(self,observations,last_actions, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, ac, explore=explore) for a, obs, ac in zip(self.agents,
                                                                       observations, last_actions)]

    def update(self, sample, agent_i,alg_type='multi-agent',parallel=False, logger=None):
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
        obs, last_acs, acs, rews, next_obs, dones = sample
        curr_agent=self.agents[agent_i]
        agent_encode = torch.zeros([1, self.nagents]).cuda()
        agent_encode[0][agent_i] = 1
        agent_encode = agent_encode.repeat(self.batch_size, 1)

        self.critic_optimizer.zero_grad()
        oth_agents = [i for i in range(self.nagents) if (i != agent_i)]  # index of other agents

        if self.discrete_action:  # one-hot encode action
            other_trgt_acs = [
                onehot_from_logits(self.agents[i].policy(torch.cat((next_obs[i],acs[i]),dim=1)))for i in oth_agents]
        else:
            other_trgt_acs = [self.agents[i].policy(torch.cat((next_obs[i],acs[i]),dim=1))
                              for i in oth_agents]
        # the input of critic is: other_agents'actions, state, curr_agent_id, curr_agent's obs
        trgt_s_in = torch.cat((*other_trgt_acs, *next_obs), dim=1)
        trgt_vf_in = torch.cat((trgt_s_in, next_obs[agent_i]), dim=1)
        trgt_vf_in = torch.cat((trgt_vf_in, acs[agent_i]), dim=1)
        trgt_vf_in = torch.cat((trgt_vf_in, agent_encode), dim=1)
        # tmp=curr_agent.target_critic(trgt_vf_in)

        target_value=self.target_critic(trgt_vf_in).detach()
        ac_index=acs[agent_i].max(dim=1)[1].unsqueeze(1)
        target_value=target_value.gather(1, ac_index)

        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        target_value *
                        (1 - dones[agent_i].view(-1, 1)))
        # construct curr critic input
        other_acs = copy.deepcopy(acs)
        other_acs.pop(agent_i)
        s_in = torch.cat((*other_acs, *obs), dim=1)
        vf_in = torch.cat((s_in, obs[agent_i]), dim=1)
        vf_in = torch.cat((vf_in, last_acs[agent_i]), dim=1)
        vf_in = torch.cat((vf_in, agent_encode), dim=1)
        actual_value = self.critic(vf_in)
        actual_value = actual_value.gather(1,ac_index)
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        # if parallel:
        #     average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        curr_agent.policy_optimizer.zero_grad()

        # update critic
        if self.discrete_action:
            curr_pol_out = curr_agent.policy(torch.cat((obs[agent_i],last_acs[agent_i]),dim=1))
            probs = F.softmax(curr_pol_out, dim=1)
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(torch.cat((obs[agent_i],last_acs[agent_i]),dim=1))
            probs = 1
            curr_pol_vf_in = curr_pol_out
        all_pol_other_acs = [onehot_from_logits(self.agents[i].policy(torch.cat((obs[i],last_acs[i]),dim=1)))  for i in oth_agents]
        s_in = torch.cat((*all_pol_other_acs, *obs), dim=1)
        vf_in = torch.cat((s_in, obs[agent_i]), dim=1)
        vf_in = torch.cat((vf_in, last_acs[agent_i]), dim=1)
        vf_in = torch.cat((vf_in,agent_encode),dim=1)
        # calculate advantage
        q_values = self.critic(vf_in)
        baseline = (q_values * probs).sum(dim=1).unsqueeze(1).detach()
        ac_index = torch.max(curr_pol_vf_in, dim=1)[1].unsqueeze(1)
        q_taken = q_values.gather(1, ac_index)
        advantage = (q_taken - baseline).detach()
        log_probs = F.log_softmax(curr_pol_out,dim=1)
        log_probs = log_probs.gather(1, ac_index)
        pol_loss = -(log_probs * advantage).mean()
        # pol_loss = -advantage.mean()
        # pol_loss += (curr_pol_out ** 2).mean() * 1e-3
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()
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
        #soft_update(self.agent.target_policy, self.agent.policy, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic=fn(self.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='gpu'):
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
                     'critic':self.critic.state_dict(),
                     'target_critic':self.target_critic.state_dict(),
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    # 1834=362*4+362+3*5+5+4
    @classmethod
    def init_from_env(cls, agent_num=4, agent_alg="COMA", num_in_pol=367,num_out_pol=5,num_in_critic=1834,
                      discrete_action=True,adversary_alg="COMA", gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [agent_alg for i in range(agent_num)]
        for i in range(agent_num):
            num_in_pol=num_in_pol
            num_out_pol=num_out_pol
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'num_in_critic': num_in_critic,
                     'num_out_critic': num_out_pol,
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
        instance.critic=save_dict['critic']
        instance.target_critic=save_dict['target_critic']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance