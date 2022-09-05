#! /usr/bin/env python
import rospy
import argparse
import torch
import time
import os
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import json

from SMART.baselines.ippo import MAPPO
from Env import Env


USE_CUDA = torch.cuda.is_available()

def run(config):
    log_dir = 'checkpoints_IPPO_roundabout_6_8/'
    run_dir = log_dir + 'logs/'
    logger = SummaryWriter(str(log_dir))
    env_info = {}
    env_info['episode_length'] = config.episode_length
    env_info['batch_size'] = config.batch_size
    env_json = json.dumps(env_info)
    with open(log_dir + 'setting.json', 'w') as json_file:
        json.dump(env_json, json_file)

    env = Env(scenario='roundabout')
    mappo = MAPPO.init_from_env(agent_alg=config.agent_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    # update_rate
    N=24
    nagents=12
    learning_iter=0
    n_steps=0
    for ep_i in range(0, config.n_episodes):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 2,
                                        config.n_episodes))
        obs = env.reset()
        mappo.prep_rollouts(device='gpu')

        collision_flag=0
        score_hist=[[] for i in range(nagents)]
        speeds_list = [[] for i in range(nagents)]

        for et_i in range(config.episode_length):
            print(et_i)
            n_steps+=1
            collision_flag=collision_flag+1

            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])).cuda(),
                                  requires_grad=False)
                         for i in range(mappo.nagents)]

            actions, action_one_hots, probs, vals = mappo.step(torch_obs, explore=True)

            action_one_hots = [[ac for ac in action_one_hots] for i in range(config.n_rollout_threads)]
            #print(action_one_hots)
            next_obs, rewards, dones, info = env.step(action_one_hots,isTeamReward=True)
            print(rewards)
            speeds = info['speeds']
            # covert torch tensor to numpy
            obs_list=[torch_ob.cpu().squeeze(0).numpy().tolist() for torch_ob in torch_obs]
            rewards=rewards[0].tolist()
            dones=dones[0].tolist()
            mappo.push(obs_list,actions,probs,vals,rewards,dones)
            for agent_i in range(nagents):
                score_hist[agent_i].append(rewards[agent_i])
            obs = next_obs
            # if (et_i+1)%N == 0 or dones[0]==1:
            if n_steps % N ==0:
                if USE_CUDA:
                    mappo.prep_training(device='gpu')
                else:
                    mappo.prep_training(device='cpu')
                for agent_i, agent in enumerate(mappo.agents):
                    agent.learn(agent_i, learning_iter,logger)
                learning_iter+=1
                mappo.prep_rollouts(device='gpu')
            print(np.max(dones))
            for i, speed in enumerate(speeds):
                speeds_list[i].append(speed)
                print(speed)
            if np.max(dones)==1:
                break
        #mappo.clear()
        # ep_rews = replay_buffer.get_average_rewards(
        #     config.episode_length * config.n_rollout_threads)
        for index,scores in enumerate(score_hist):
            logger.add_scalar('agent%i/mean_episode_rewards' % index, np.mean(scores), ep_i)
            logger.add_scalar('agent%i/mean_speed' % index, np.mean(speeds_list[index]), ep_i)
        logger.add_scalar('turn_success', env.scenario.turn_success_flag / nagents, ep_i)
        if collision_flag == config.episode_length:
            logger.add_scalar('collision', 0, ep_i)
        else:
            logger.add_scalar('collision', 1, ep_i)
        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir + 'incremental', exist_ok=True)
            mappo.save(run_dir + 'incremental' + ('model_ep%i.pt' % (ep_i + 1)))
            mappo.save(run_dir + 'model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", help="Name of environment", default="Autodriving")
    parser.add_argument("--model_name",
                        help="Name of directory to store " +
                             "model/training contents", default="DQN")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=12, type=int)
    parser.add_argument("--buffer_length", default=int(1e5), type=int)
    parser.add_argument("--n_episodes", default=30000, type=int)
    parser.add_argument("--episode_length", default=24, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=3000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=200, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="PPO", type=str,
                        choices=['PPO', 'PPO'])
    parser.add_argument("--adversary_alg",
                        default="PPO", type=str,
                        choices=['PPO', 'PPO'])
    parser.add_argument("--discrete_action", default=True, type=bool)

    config = parser.parse_args()
    run(config)