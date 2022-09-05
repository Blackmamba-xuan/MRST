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
from sac_torch import Agent
from env_laneChange import Env

if __name__ == '__main__':
    log_dir = 'checkpoints_SAC_11_12_lanechange/'
    run_dir = log_dir + 'logs/'
    logger = SummaryWriter(str(log_dir))
    env=Env()
    input_dim = (4, 120, 200)
    action_dim = 2
    agent = Agent(input_dims=input_dim, env=env,
                  n_actions=action_dim, batch_size=256, log_dir=run_dir)
    n_episodes=50000
    episode_length=5
    steps_per_update=30
    save_interval=1000
    global_step=0
    reward_list=[]

    for ep_i in range(n_episodes):
        print('episode: ', ep_i)
        state, speed = env.reset()
        reward_sum=0
        successFlag=False

        for ep_t in range(episode_length):
            global_step+=1
            action = agent.choose_action(state,speed)
            next_state, next_speed ,reward, done,successFlag = env.step(action)
            agent.remember(state,speed, action, reward, next_state,next_speed, done)
            reward_sum+=reward
            state=next_state
            speed=next_speed
            if global_step % steps_per_update == 0:
                print('enter learn step')
                agent.learn()
            if done:
                break
        env.turn_back()
        reward_list.append(reward_sum)
        logger.add_scalar('episode_rewards', reward_sum, ep_i)
        logger.add_scalar('successFlag', successFlag, ep_i)
        logger.add_scalar('mean_episode_rewards', np.mean(reward_list), ep_i)
        if ep_i % save_interval ==0:
            agent.save_models(ep_i)




