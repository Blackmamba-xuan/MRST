import time

from env2 import *
from td3 import TD3Agent
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from tensorboardX import SummaryWriter
import os
from torch.autograd import Variable
from buffer import TD3ReplayBuffer
import numpy as np

device = torch.device("cpu")

n_exploration_eps=3000
final_noise_scale=0.0
init_noise_scale=0.3
buffer_length=int(5000)
batch_size=516
steps_per_update=100
USE_CUDA=False
if __name__ == '__main__':
    log_dir = 'checkpoints_TD3_7_8_minSpeed_0.04_env300_env650/'
    run_dir = log_dir + 'logs/'
    logger = SummaryWriter(str(log_dir))
    learn_step = 0
    os.makedirs(run_dir,exist_ok=True)

    env=Env()
    n_actions=9
    agent=TD3Agent(60,100,n_actions)
    replay_buffer = TD3ReplayBuffer(buffer_length, 1,
                                 [[4,60,100]] * 1,
                                 [9] * 1)

    num_episodes = 30000
    epi_length = 30
    for ep_i in range(num_episodes):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 2,
                                        num_episodes))
        state,speed=env.reset()
        agent.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, n_exploration_eps - ep_i) / n_exploration_eps
        agent.scale_noise(
            final_noise_scale + (init_noise_scale - final_noise_scale) * explr_pct_remaining)
        agent.reset_noise()
        reward_sum = 0
        reward_list = []
        t=0
        for et_i in range(epi_length):
            torch_state=Variable(Tensor(state).unsqueeze(0).to(device),requires_grad=False)
            torch_speed=Variable(Tensor(speed).unsqueeze(0).to(device),requires_grad=False)
            action=agent.choose_action(torch_state,torch_speed)
            _, agent_action=torch.max(action,dim=1)
            next_state, next_speed, reward, done = env.step(agent_action)
            reward_sum += reward
            reward=np.array([[reward]])
            if done:
                dones=np.full((1, 1), 1)
            else:
                dones=np.full((1, 1), 0)
            replay_buffer.push(np.array([[state]]),np.array([speed]),action.cpu().numpy(),reward,np.array([[next_state]]),np.array([next_speed]),dones)
            if done:
                break
            else:
                # Move to the next state
                state=next_state
                speed=next_speed
                t+=1
                if (len(replay_buffer) >= batch_size and
                        (t % steps_per_update) < 1):
                    if USE_CUDA:
                        agent.prep_training(device='cpu')
                    else:
                        agent.prep_training(device='cpu')
                    sample = replay_buffer.sample(batch_size,
                                                        to_gpu=USE_CUDA)
                    agent.update(sample, 0, logger=logger)
                    agent.update_all_targets()
                    agent.prep_rollouts(device='cpu')
            time.sleep(0.2)
        ep_rews = replay_buffer.get_average_rewards(
            epi_length)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
        print('episode sum reward: ', reward_sum)
        if len(reward_list) < 1000:
            reward_list.append(reward_sum)
            mean_reward = np.mean(reward_list)
        else:
            reward_list.pop(0)
            reward_list.append(reward_sum)
            mean_reward = np.mean(reward_list)
        logger.add_scalar('episode_rewards', reward_sum, ep_i)
        logger.add_scalar('mean_episode_rewards', mean_reward, ep_i)
