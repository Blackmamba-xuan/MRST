import time

from env3 import *
from dqn import Agent
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from tensorboardX import SummaryWriter
import os
from torch.autograd import Variable

# if gpu is to be used
device = torch.device("cpu")
# device = torch.device("cpu")

if __name__ == '__main__':
    log_dir = 'checkpoints_DQN_8_9_minSpeed_0.04_env300_env650/'
    run_dir = log_dir + 'logs/'
    logger = SummaryWriter(str(log_dir))
    learn_step = 0
    os.makedirs(run_dir,exist_ok=True)

    env=Env()
    n_actions=9
    agent=Agent(120,200,n_actions)

    num_episodes = 30000
    epi_length = 30
    for i_episode in range(num_episodes):
        print("Episodes %i-%i of %i" % (i_episode + 1,
                                        i_episode + 2,
                                        num_episodes))
        state,speed=env.reset()
        reward_sum = 0
        reward_list = []
        state=Variable(Tensor(state).unsqueeze(0).to(device),requires_grad=False)
        speed=Variable(Tensor(speed).unsqueeze(0).to(device),requires_grad=False)
        for et_i in range(epi_length):
            action=agent.select_action(state, speed)
            next_state, next_speed, reward, done = env.step(action)
            state.to("cpu")
            speed.to("cpu")
            reward_sum += reward
            reward = torch.tensor([reward], device="cpu").float()
            if done:
                next_state = None
                next_speed = None
            else:
                next_state = Variable(Tensor(next_state).unsqueeze(0),requires_grad=False)
                next_speed = Variable(Tensor(next_speed).unsqueeze(0),requires_grad=False)
            # Store the transition in memory
            agent.memory.push(state, speed, action, next_state, next_speed,reward)
            if done:
                break
            else:
                # Move to the next state
                state = next_state.to(device)
                speed = next_speed.to(device)
                agent.learn()
                time.sleep(0.2)
        print('episode sum reward: ', reward_sum)
        if len(reward_list)<1000:
            reward_list.append(reward_sum)
            mean_reward=np.mean(reward_list)
        else:
            reward_list.pop(0)
            reward_list.append(reward_sum)
            mean_reward=np.mean(reward_list)
        if i_episode % agent.TARGET_UPDATE == 0:
            print('enter update target')
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        logger.add_scalar('episode_rewards', reward_sum, i_episode)
        logger.add_scalar('mean_episode_rewards', mean_reward, i_episode)
        print(len(agent.memory))
