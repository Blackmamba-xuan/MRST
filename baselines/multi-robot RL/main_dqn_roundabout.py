from SMART.baselines.maddpg import MADDPG

import rospy
import argparse
import torch
import time
import os
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from SMART.baselines.dqn import DQN
from Env import Env
from SMART.baselines.utils.buffer import ReplayBuffer
import json


USE_CUDA = torch.cuda.is_available()


def run(config):
    log_dir = 'checkpoints_DQN_roundabout_6_6/'
    run_dir = log_dir + 'logs/'
    logger = SummaryWriter(str(log_dir))
    env_info = {}
    env_info['episode_length'] = config.episode_length
    env_info['batch_size'] = config.batch_size
    env_json = json.dumps(env_info)
    with open(log_dir + 'setting.json', 'w') as json_file:
        json.dump(env_json, json_file)

    env = Env(scenario='roundabout')
    dqn = DQN.init_from_env(agent_alg=config.agent_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    nagents=12
    replay_buffer = ReplayBuffer(config.buffer_length, dqn.nagents,
                                 [40]*nagents,
                                 [5]*nagents)
    # with open(run_dir + 'incrementaldqnBuffer8851.pkl', 'rb') as file:
    #     replay_buffer=pickle.load(file)
    #     print('len replay buffer: ', len(replay_buffer))
    t = 0
    for ep_i in range(0, config.n_episodes):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 2,
                                        config.n_episodes))
        obs = env.reset()
        dqn.prep_rollouts(device='gpu')


        collision_flag = 0
        speeds_list = [[] for i in range(nagents)]
        for et_i in range(config.episode_length):
            print(et_i)
            collision_flag = collision_flag + 1
            # env.render(close=False)
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])).cuda(),
                                  requires_grad=False)
                         for i in range(dqn.nagents)]
            # get actions as torch Variables
            agent_actions = dqn.step(torch_obs, explore=True)
            actions = [agent_actions for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, info = env.step(actions,isTeamReward=True)
            speeds=info['speeds']
            #print(obs, agent_actions, rewards, next_obs, dones)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                    (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    dqn.prep_training(device='gpu')
                else:
                    dqn.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(dqn.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        dqn.update(sample, a_i, logger=logger)
                    dqn.update_all_targets()
                dqn.prep_rollouts(device='gpu')
            print(np.max(dones))
            for i, speed in enumerate(speeds):
                speeds_list[i].append(speed)
                print(speed)
            if np.max(dones)==1:
                break
            #time.sleep(0.7)

        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        print('ep rews: ', ep_rews)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
            logger.add_scalar('agent%i/mean_speed' % a_i, np.mean(speeds_list[a_i]), ep_i)
        logger.add_scalar('turn_success', env.scenario.turn_success_flag/nagents, ep_i)
        if collision_flag == config.episode_length:
            logger.add_scalar('collision', 0, ep_i)
        else:
            logger.add_scalar('collision',1, ep_i)
        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir + 'incremental', exist_ok=True)
            dqn.save(run_dir + 'incremental' + ('model_ep%i.pt' % (ep_i + 1)))
            dqn.save(run_dir + 'model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", help="Name of environment", default="intersection_two_90")
    parser.add_argument("--model_name",
                        help="Name of directory to store " +
                             "model/training contents", default="DQN")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e5), type=int)
    parser.add_argument("--n_episodes", default=30000, type=int)
    parser.add_argument("--episode_length", default=24, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=256, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=2000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=300, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action", default=True, type=bool)

    config = parser.parse_args()
    run(config)