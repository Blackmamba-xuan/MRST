.. MRST documentation master file, created by
   sphinx-quickstart on Tue Jun  7 02:59:05 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MRST's documentation!
================================

.. figure:: Media/scenarios.png
   :alt: scenarios
   :align: center

Multi-robot Reinforcement Learning Scalable Training School (MRST) is a training and evaluation platform for reinforcement learning reasearch.

Check out the paper "From Multi-agent to Multi-robot: Scalable Training Platform for Multi-robot Reinforcement Learning" for background on some of the project goals.

Simple Example
~~~~~~~~~~~~~~~

1. Launch the simulation environment
   
.. highlight:: sh
::

   roslaunch mrst_simulation turtlebot3_autorace_roundabout.launch
   roslaunch mrst_simulation turtlebot3_autorace_control_roundabout.launch

2. A simple code example for training
   
.. highlight:: sh
::

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

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   Getting start with MRST/index
   MRST Scenarios/index
   How to customize the robot/index
   Support for ROS Developer/index
   MRRL Research/index



