import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam,RMSprop

from SMART.baselines.utils.networks import MLPNetwork,QMixNet
from SMART.baselines.utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from SMART.baselines.utils.agents import DQNAgent
import numpy as np

device = torch.device("cuda:0")
MSELoss = torch.nn.MSELoss()

class QMIX(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types,state_space,qmix_hidden_dim,hyper_hidden_dim,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=32,
                 discrete_action=True):

        self.nagents = 3
        self.alg_types = alg_types
        self.qmix_hidden_dim=qmix_hidden_dim
        self.agents = [DQNAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim,
                                 **params)
                       for params in agent_init_params]
        self.qmix_net=QMixNet(nagents=self.nagents,state_space=state_space,qmix_hidden_dim=qmix_hidden_dim,hyper_hidden_dim=hyper_hidden_dim)
        self.qmix_net_optimizer=RMSprop(self.qmix_net.parameters(), lr=lr)

        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.rw_scale=0.1
        self.niter = 0
        self.pol_dev='cpu'
        self.TARGET_UPDATE = 10
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

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
        actionList=[]
        for a, obs in zip(self.agents,observations):
            action=a.step(obs)
            actionList.append(action)
        return actionList

    def stepEval(self, observations):
        actionList = []
        for a, obs in zip(self.agents, observations):
            action = a.evalStep(obs)
            actionList.append(action)
        return actionList

    def update(self, sample, parallel=False, logger=None):
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
        obs, acs, rews, next_obs, dones = sample
        states=torch.cat((*obs,),dim=1)
        next_states=torch.cat((*next_obs,),dim=1)
        actual_q_list=[]
        next_q_list=[]
        self.qmix_net_optimizer.zero_grad()
        for agent_i in range(self.nagents):
            curr_agent = self.agents[agent_i]
            curr_agent.policy_optimizer.zero_grad()
            curr_obs=obs[agent_i]
            curr_acs=acs[agent_i]
            curr_rews=rews[agent_i]
            curr_next_obs=next_obs[agent_i]

            #vf_in=torch.cat((*obs,),dim=1)
            curr_acs_index=curr_acs.max(1)[1].view(-1,1)
            #print(curr_acs_index)
            curr_actual_values=curr_agent.policy(curr_obs).gather(1,curr_acs_index)
            curr_next_q_values=curr_agent.target_policy(curr_next_obs).max(1)[0].unsqueeze(1).detach()
            actual_q_list.append(curr_actual_values)
            next_q_list.append(curr_next_q_values)
        q_values=torch.cat((*actual_q_list,),dim=1)
        next_q_values=torch.cat((*next_q_list,),dim=1)
        q_total=self.qmix_net.forward(q_values,states)
        next_q_total=self.qmix_net.forward(next_q_values,next_states).detach()

        target_values=rews[0].view(-1, 1)*self.rw_scale+self.gamma*(1 - dones[0].view(-1, 1))*next_q_total
        loss = MSELoss(q_total, target_values.detach())
        loss.backward()
        for param in self.qmix_net.parameters():
            param.grad.data.clamp_(-0.5, 0.5)
        self.qmix_net_optimizer.step()
        for agent_i in range(self.nagents):
            curr_agent=self.agents[agent_i]
            for param in curr_agent.policy.parameters():
                param.grad.data.clamp_(-0.5, 0.5)
            curr_agent.policy_optimizer.step()
            niter=self.niter+1
            curr_agent.EPSILON = curr_agent.EPSILON * curr_agent.EPS_DEC if curr_agent.EPSILON > \
                                                        curr_agent.EPS_MIN else curr_agent.EPS_MIN
            if logger is not None:
                logger.add_scalars('agent%i/losses' % agent_i,
                                   {'loss': loss},
                                   niter)
        self.niter+=1

    def update_all_targets(self):
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)

    def prep_training(self, device='gpu'):
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
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device

    def prep_rollouts(self, device='gpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        for a in self.agents:
            a.policy = fn(a.policy)
            a.target_policy=fn(a.target_policy)

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'qmix_net':self.qmix_net,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, agent_num=3, agent_alg="QMIX", num_in_pol=363, num_out_pol=4, discrete_action=True,
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=32, qmix_hidden_dim=32,hyper_hidden_dim=64):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        state_space=num_in_pol*agent_num

        for i in range(agent_num):
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': agent_alg,
                     'state_space':state_space,
                     'qmix_hidden_dim':qmix_hidden_dim,
                     'hyper_hidden_dim':hyper_hidden_dim,
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
        instance.qmix_net = save_dict['qmix_net']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
            optimizer_to(a.policy_optimizer,device)
        return instance

def optimizer_to(optim, device):
    for param in optim.state.values():
         # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)