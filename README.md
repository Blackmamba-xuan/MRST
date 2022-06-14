# MRST - A Scalable Training School for Multi-robot Reinforcement Learning
Multi-robot Reinforcement Learning Scalable Training School (MRST) is a training and evaluation platform for reinforcement learning reasearch.

Check out the background and project goals in our paper "From Simulation to Reality: A Scalable Emulation
Platform for Multi-robot Reinforcement Learning".

![](https://github.com/Blackmamba-xuan/MRST/blob/main/screenshoot/scenarios.png)

# Installation

Our project require the installation of the following third-parties before getting started:

- Ubuntu 18
- python 3.7+
- torch 1.8+
- [Ros Knetic or Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu)
- [Turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3)

## Getting Started

Create a workspace, clone the repo using git command:

```shell
git clone https://github.com/Blackmamba-xuan/MRST.git
```
Move the folder mrst_simulation to your catkin workspace and build for it

```shell
cd ~/catkin_ws/src/
catkin_make
```
## Multi-robot Reinforcement Learning Baselines

We implement several high impact reinforcement learning algorithms and extend them to multi-robot areas.

- [x] Indepent Q-learning
- [x] MADDPG
- [x] COMA
- [x] MAAC
- [x] MASAC
- [x] IPPO
- [x] MAPPO
- [x] VDN
- [x] QMIX 
- [x] Hierarchical RL
- [x] Others

## Running the Simulation

Use the roslaunch command to run the simulator:

```shell
# launch a crosslane environment
roslaunch mrst_simulation turtlebot3_autorace_crossLane.launch

# launch a roundabout environment
roslaunch mrst_simulation turtlebot3_autorace_roundabout.launch

```
## Simple Code

```shell
from Env import Env
   def main():
      env=Env(scenario="roundabout")
      n_episodes = 100
      n_agents=12
      episode_length=15
      for e in range(n_episodes):
         env.reset()
         for et_i in range(episode_length):
               print(et_i)
               actions=[[1] for i in range(n_agents)]
               next_obs, rewards, dones, speeds = env.step(actions, isTeamReward=True)
   if __name__ == "__main__":
      main()

```

## Documentation

For more detalis, please check our [documentaion](https://mrst.readthedocs.io/en/latest/).

### Simple Example


