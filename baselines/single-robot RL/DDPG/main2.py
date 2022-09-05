import gym
import numpy as np
from DeepDeterministic.utils import plotLearning
from DeepDeterministic.DDPG import *

if __name__=='__main__':
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=1e-4, beta=1e-3, input_dims=[8], tau=0.001, env=env,
                  batch_size=64, layer1_size=32, layer2_size=32, n_actions=2)

    # agent.load_models()
    np.random.seed(0)
    score_history = []
    for i in range(500):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            # 这里是神经网络模型做出动作的决策
            act = agent.choose_action(obs)
            #print(act)
            # 这个接口是把动作通过step接口给到环境，然后让环境去执行，值得注意的是，这里的返回值有执行后的new_state，还有done的标志服
            # 这个done表示游戏是不是结束，碰到东西，或者赢了
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
            # env.render()
        score_history.append(score)

        # if i % 25 == 0:
        #     agent.save_models()

        print('episode ', i, 'score %.2f' % score,
              'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

    filename = 'LunarLander-400-300-main2.png'
    plotLearning(score_history, filename, window=100)