import gym
import torch
import torch.nn as nn
import torch.optim as opt
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple
from itertools import count
import matplotlib
import os

matplotlib.use('agg')
import matplotlib.pyplot as plt
from os.path import join as joindir
from os import makedirs as mkdir
import pandas as pd
import numpy as np
import argparse
import datetime
import math
from env_laneChange import Env
import time
from tensorboardX import SummaryWriter

Transition = namedtuple('Transition', ('state', 'speed', 'value', 'action', 'logproba', 'mask', 'next_state', 'next_speed' ,'reward'))
EPS = 1e-10
RESULT_DIR = joindir('../result', '.'.join(__file__.split('.')[:-1]))
mkdir(RESULT_DIR, exist_ok=True)


class args(object):
    env_name = 'Auto-driving'
    seed = 1234
    num_episode = 2000
    batch_size = 256
    max_step_per_round = 5
    gamma = 0.995
    lamda = 0.97
    log_num_episode = 1
    num_epoch = 6
    minibatch_size = 64
    clip = 0.2
    loss_coeff_value = 0.5
    loss_coeff_entropy = 0.01
    lr = 3e-4
    num_parallel_run = 1
    # tricks
    schedule_adam = 'linear'
    schedule_clip = 'linear'
    layer_norm = True
    state_norm = True
    advantage_norm = True
    lossvalue_norm = True


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


class ActorCritic(nn.Module):
    def __init__(self, input_dims, num_outputs, layer_norm=True):
        super(ActorCritic, self).__init__()
        #Actor Network
        self.actor_conv1 = nn.Conv2d(input_dims[0], 16, kernel_size=5, stride=2)
        self.actor_bn1 = nn.BatchNorm2d(16)
        self.actor_conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.actor_bn2 = nn.BatchNorm2d(32)
        self.actor_conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.actor_bn3 = nn.BatchNorm2d(32)
        actor_convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(input_dims[1])))
        actor_convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(input_dims[2])))
        actor_linear_input_size=actor_convw*actor_convh*32
        self.actor_fc = nn.Linear(actor_linear_input_size+2, num_outputs) # 2 is the dimension of speed
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_outputs))
        #critic Network
        self.critic_conv1 = nn.Conv2d(input_dims[0], 16, kernel_size=5, stride=2)
        self.critic_bn1 = nn.BatchNorm2d(16)
        self.critic_conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.critic_bn2 = nn.BatchNorm2d(32)
        self.critic_conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.critic_bn3= nn.BatchNorm2d(32)
        critic_convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(input_dims[1])))
        critic_convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(input_dims[2])))
        critic_linear_input_size = critic_convw * critic_convh * 32
        self.critic_fc = nn.Linear(critic_linear_input_size+2, 1) # 2 is the dimension of speed
        # self.actor_fc1 = nn.Linear(num_inputs, 64)
        # self.actor_fc2 = nn.Linear(64, 64)
        # self.actor_fc3 = nn.Linear(64, num_outputs)
        # self.actor_logstd = nn.Parameter(torch.zeros(1, num_outputs))
        #
        # self.critic_fc1 = nn.Linear(num_inputs, 64)
        # self.critic_fc2 = nn.Linear(64, 64)
        # self.critic_fc3 = nn.Linear(64, 1)

        # if layer_norm:
        #     self.layer_norm(self.actor_fc1, std=1.0)
        #     self.layer_norm(self.actor_fc2, std=1.0)
        #     self.layer_norm(self.actor_fc3, std=0.01)
        #
        #     self.layer_norm(self.critic_fc1, std=1.0)
        #     self.layer_norm(self.critic_fc2, std=1.0)
        #     self.layer_norm(self.critic_fc3, std=1.0)

    def conv2d_size_out(self,size, kernel_size=5, stride=2):
        return (size - (kernel_size - 1) - 1) // stride + 1

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, states, speeds):
        """
        run policy network (actor) as well as value network (critic)
        :param states: a Tensor2 represents states
        :return: 3 Tensor2
        """
        action_mean, action_logstd = self._forward_actor(states, speeds)
        critic_value = self._forward_critic(states, speeds)
        return action_mean, action_logstd, critic_value

    def _forward_actor(self, states, speeds):
        x = F.relu(self.actor_bn1(self.actor_conv1(states)))
        x = F.relu(self.actor_bn2(self.actor_conv2(x)))
        x = F.relu(self.actor_bn3(self.actor_conv3(x)))
        # x = torch.tanh(self.actor_fc1(states))
        # x = torch.tanh(self.actor_fc2(x))
        linear_in=torch.cat((x.view(x.size(0), -1),speeds), dim=1)
        action_mean = self.actor_fc(linear_in)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def _forward_critic(self, states, speeds):
        x = F.relu(self.critic_bn1(self.critic_conv1(states)))
        x = F.relu(self.critic_bn2(self.critic_conv2(x)))
        x = F.relu(self.critic_bn3(self.critic_conv3(x)))
        # x = torch.tanh(self.critic_fc1(states))
        # x = torch.tanh(self.critic_fc2(x))
        linear_in = torch.cat((x.view(x.size(0), -1), speeds), dim=1)
        critic_value = self.critic_fc(linear_in)
        return critic_value

    def select_action(self, action_mean, action_logstd, return_logproba=True):
        """
        given mean and std, sample an action from normal(mean, std)
        also returns probability of the given chosen
        """
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        if return_logproba:
            logproba = self._normal_logproba(action, action_mean, action_logstd, action_std)
        return action, logproba

    @staticmethod
    def _normal_logproba(x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)

        std_sq = std.pow(2)
        logproba = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
        return logproba.sum(1)

    def get_logproba(self, states, speeds, actions):
        """
        return probability of chosen the given actions under corresponding states of current network
        :param states: Tensor
        :param actions: Tensor
        """
        action_mean, action_logstd = self._forward_actor(states, speeds)
        logproba = self._normal_logproba(actions, action_mean, action_logstd)
        return logproba


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


def ppo(args):
    log_dir = 'checkpoints_PPO_11_lanechange_6_minSpeed/'
    run_dir = log_dir + 'logs/'
    logger = SummaryWriter(str(log_dir))
    learn_step=0
    #os.makedirs(run_dir)

    env = Env()
    input_dim=(4,120, 200)
    num_actions=2

    #env.seed(args.seed)
    torch.manual_seed(args.seed)

    network = ActorCritic(input_dim, num_actions, layer_norm=args.layer_norm)
    optimizer = opt.Adam(network.parameters(), lr=args.lr)

    #running_state = ZFilter((num_inputs,), clip=5.0)

    # record average 1-round cumulative reward in every episode
    reward_record = []
    global_steps = 0
    global_episode = 0

    lr_now = args.lr
    clip_now = args.clip

    save_step=2

    for i_episode in range(args.num_episode):
        print('i_episode: ', i_episode)
        # step1: perform current policy to collect trajectories
        # this is an on-policy method!
        memory = Memory()
        num_steps = 0
        reward_list = []
        len_list = []
        reward_sum=0
        while num_steps < args.batch_size:
            global_episode+=1
            state,speed = env.reset()
            # if args.state_norm:
            #     state = running_state(state)
            reward_sum = 0
            for t in range(args.max_step_per_round):
                action_mean, action_logstd, value = network(Tensor(state).unsqueeze(0), Tensor(speed).unsqueeze(0))
                action, logproba = network.select_action(action_mean, action_logstd)
                action = action.data.numpy()[0]
                logproba = logproba.data.numpy()[0]
                next_state, next_speed, reward, done,successFlag = env.step(action)
                print('reward: ', reward)
                reward_sum += reward
                mask = 0 if done else 1
                memory.push(state, speed, value, action, logproba, mask, next_state, next_speed, reward)
                if done:
                    break

                state = next_state
                speed = next_speed
            env.turn_back()
            num_steps += (t + 1)
            global_steps += (t + 1)
            reward_list.append(reward_sum)
            len_list.append(t + 1)
            logger.add_scalar('episode_rewards', reward_sum, global_episode)
            logger.add_scalar('successFlag', successFlag, global_episode)
            logger.add_scalar('mean_episode_rewards', np.mean(reward_list), global_episode)
        reward_record.append({
            'episode': i_episode,
            'steps': global_steps,
            'meanepreward': np.mean(reward_list),
            'meaneplen': np.mean(len_list)})

        batch = memory.sample()
        batch_size = len(memory)

        # step2: extract variables from trajectories
        rewards = Tensor(batch.reward)
        values = Tensor(batch.value)
        masks = Tensor(batch.mask)
        actions = Tensor(batch.action)
        states = Tensor(batch.state)
        speeds = Tensor(batch.speed)
        oldlogproba = Tensor(batch.logproba)

        returns = Tensor(batch_size)
        deltas = Tensor(batch_size)
        advantages = Tensor(batch_size)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(batch_size)):
            returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values[i]
            # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
            advantages[i] = deltas[i] + args.gamma * args.lamda * prev_advantage * masks[i]

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]
        if args.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

        for i_epoch in range(int(args.num_epoch * batch_size / args.minibatch_size)):
            # sample from current batch
            minibatch_ind = np.random.choice(batch_size, args.minibatch_size, replace=False)
            minibatch_states = states[minibatch_ind]
            minibatch_speeds = speeds[minibatch_ind]
            minibatch_actions = actions[minibatch_ind]
            minibatch_oldlogproba = oldlogproba[minibatch_ind]
            minibatch_newlogproba = network.get_logproba(minibatch_states, minibatch_speeds, minibatch_actions)
            minibatch_advantages = advantages[minibatch_ind]
            minibatch_returns = returns[minibatch_ind]
            minibatch_newvalues = network._forward_critic(minibatch_states, minibatch_speeds).flatten()

            ratio = torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
            surr1 = ratio * minibatch_advantages
            surr2 = ratio.clamp(1 - clip_now, 1 + clip_now) * minibatch_advantages
            loss_surr = - torch.mean(torch.min(surr1, surr2))

            # not sure the value loss should be clipped as well
            # clip example: https://github.com/Jiankai-Sun/Proximal-Policy-Optimization-in-Pytorch/blob/master/ppo.py
            # however, it does not make sense to clip score-like value by a dimensionless clipping parameter
            # moreover, original paper does not mention clipped value
            if args.lossvalue_norm:
                minibatch_return_6std = 6 * minibatch_returns.std()
                loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2)) / minibatch_return_6std
            else:
                loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))

            loss_entropy = torch.mean(torch.exp(minibatch_newlogproba) * minibatch_newlogproba)

            total_loss = loss_surr + args.loss_coeff_value * loss_value + args.loss_coeff_entropy * loss_entropy
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            print('enter learn process')
            learn_step+=1
            logger.add_scalar('value_loss', loss_value, learn_step)
            logger.add_scalar('total_loss', total_loss, learn_step)

        if args.schedule_clip == 'linear':
            ep_ratio = 1 - (i_episode / args.num_episode)
            clip_now = args.clip * ep_ratio

        if args.schedule_adam == 'linear':
            ep_ratio = 1 - (i_episode / args.num_episode)
            lr_now = args.lr * ep_ratio
            # set learning rate
            # ref: https://stackoverflow.com/questions/48324152/
            for g in optimizer.param_groups:
                g['lr'] = lr_now

        if i_episode % save_step ==0:
            print('enter save model')
            torch.save(network,run_dir+str(i_episode)+'model.pkl')
        if i_episode % args.log_num_episode == 0:
            print('Finished episode: {} Reward: {:.4f} total_loss = {:.4f} = {:.4f} + {} * {:.4f} + {} * {:.4f}' \
                  .format(i_episode, reward_record[-1]['meanepreward'], total_loss.data, loss_surr.data,
                          args.loss_coeff_value,
                          loss_value.data, args.loss_coeff_entropy, loss_entropy.data))
            print('-----------------')


    return reward_record


def main(args):
    record_dfs = []
    for i in range(args.num_parallel_run):
        args.seed += 1
        reward_record = pd.DataFrame(ppo(args))
        reward_record['#parallel_run'] = i
        record_dfs.append(reward_record)
    record_dfs = pd.concat(record_dfs, axis=0)
    record_dfs.to_csv(joindir(RESULT_DIR, 'ppo-record-{}.csv'.format(args.env_name)))


if __name__ == '__main__':
    args.env_name = 'AutoDriving'
    main(args)