import math
import random

import rospy
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, UInt8, Bool
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from math import radians
import numpy as np
import copy
import time
import os, sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

class Scenario(object):
    def __init__(self,ScenarioID=0, require_social_agnet=False):
        print('creating intersection_two')
        rospy.init_node('gazebo_env_intersection_two')
        rate = rospy.Rate(1)
        self.nagents = 6
        names=['vehicle1','vehicle4','vehicle7','vehicle10','vehicle11','vehicle14']
        initPos = [[-4.130228, -0.068332], [-1.727619, -1.571675], [-2.218669, 2.050204], [2.432010,-1.295719],[4.063855,4.063855],[1.724398,2.408157]]
        self.eneities = [Entity(names[i], initPos[i]) for i in range(self.nagents)]
        self.lastPos = initPos
        self.rw_scale = 1
        self.turn_success_flag=0

    def reset(self):
        self.turn_success_flag = 0
        reset_simulation = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_simulation()
        obs = [self.eneities[i].getObs() for i in range(self.nagents)]
        self.lastPos = []
        for entity in self.eneities:
            self.lastPos.append(entity.getPos())
            entity.reset()
        time.sleep(1)
        return np.array([obs])

    def step(self, actions,isTeamReward=False):
        actions = actions[0]
        actions = [np.argmax(action) for action in actions]
        print('step actions: ', actions)
        for entity, action in zip(self.eneities, actions):
            entity.step(action)
        time.sleep(0.5)
        obs = []
        rewards = []
        speeds = []
        pos = []
        done_flag = 0
        # calculate reward
        if isTeamReward:
            team_reward=0
            for i, entity in zip(range(self.nagents), self.eneities):
                ob = entity.getObs()
                scan_msg = entity.scan_data
                # print(scan_data)
                front_data = [scan_msg[i] for i in range(-2, 2)]
                front_dist = np.min(front_data)
                # print(entity.name, 'front_dist: ', front_dist)
                lastP = self.lastPos[i]
                cur_pos = entity.pos
                pos.append(cur_pos)
                if front_dist < 0.2:
                    team_reward = -100
                    done_flag = 1
                    print('===================detect collision===================')
                    print('===================detect collision===================')
                    print('robot name: ', entity.name)
                else:
                    team_reward+= self.rw_scale * math.sqrt((lastP[0] - cur_pos[0]) ** 2 + (lastP[1] - cur_pos[1]) ** 2)
                    self.lastPos[i] = cur_pos
                    if entity.isFinishTurn:
                        team_reward+=25
                        entity.isFinishTurn=False
                        self.turn_success_flag+=1
                obs.append(ob)
                speeds.append(entity.speed_x)
            rewards=np.array([team_reward]*self.nagents)
        else:
            for i, entity in zip(range(self.nagents), self.eneities):
                ob = entity.getObs()
                scan_msg = entity.scan_data
                # print(scan_data)
                front_data = [scan_msg[i] for i in range(-2, 2)]
                front_dist = np.min(front_data)
                print(entity.name, 'front_dist: ', front_dist)
                lastP = self.lastPos[i]
                cur_pos = entity.pos
                pos.append(cur_pos)
                if front_dist < 0.2:
                    print('===================detect collision===================')
                    print('===================detect collision===================')
                    reward = -100
                    done_flag = 1
                else:
                    reward = self.rw_scale * math.sqrt((lastP[0] - cur_pos[0]) ** 2 + (lastP[1] - cur_pos[1]) ** 2)
                    if entity.isFinishTurn:
                        reward+=25
                        entity.isFinishTurn=False
                        self.turn_success_flag+=1
                    self.lastPos[i] = cur_pos
                rewards.append(reward)
                obs.append(ob)
                speeds.append(entity.speed_x)
        dones = np.full((1, self.nagents), done_flag)
        info = {
            'pos': np.array(pos),
            'speeds': speeds
        }

        # step the social agent
        # if self.require_social_agnet:
        #     for social_agent in self.social_agents:
        #         social_agent.step()
        return np.array([obs]), np.array([rewards]), dones,info

class Entity(object):

    def __init__(self,name,pos):
        self.counter = 1
        self.sub_scan = rospy.Subscriber(name + '/scan', LaserScan, self.scanCallback, queue_size=1)
        self.sub_odom = rospy.Subscriber(name + '/odom', Odometry, self.getOdometry, queue_size=1)
        self.sub_speed = rospy.Subscriber(name + '/cmd_vel', Twist, self.speedCallBack, queue_size=1)
        self.sub_start_turn = rospy.Subscriber(name + '/control/start_turn', Bool, self.startTrunCallBack,queue_size=1)
        self.sub_finish_turn = rospy.Subscriber(name + '/control/finish_turn', Bool, self.finishTrunCallBack, queue_size=1)
        self.pub_reSet = rospy.Publisher(name + '/reset_env', Bool, queue_size=1)
        self.pub_lane_behavior = rospy.Publisher(name + '/lane_behavior', UInt8, queue_size=1)
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.scan_data = [3.5]*36
        self.name=name
        self.speed_x=0.1
        self.pos=pos
        self.isFinishTurn=False
        self.isStartTurn=0


    def step(self,action):
        behavior_msg = UInt8()
        behavior_msg.data = np.uint8(action)
        self.pub_lane_behavior.publish(behavior_msg)

    def reset(self):
        self.isFinishTurn = False
        self.isTurnFlag = 0
        self.pub_reSet.publish(Bool(data=True))

    def getObs(self):
        obs=copy.deepcopy(self.scan_data)
        obs.append(self.isStartTurn)
        obs.append(self.speed_x)
        obs = np.append(obs, self.pos[0])
        obs = np.append(obs, self.pos[1])
        return np.array(obs)

    def scanCallback(self,data):
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1
        #print('enter scanCallback')
        scan = data
        scan_range = []
        # print('scan_data_lenth: ',len(scan.ranges))
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif scan.ranges[i]==float('inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(3.5)
            else:
                scan_range.append(scan.ranges[i])
        scan_range90 = [np.min(scan_range[i * 10:(i + 1) * 10]) for i in range(36)]
        self.scan_data = scan_range90

    def speedCallBack(self, msg):
        self.speed_x = msg.linear.x

    def getOdometry(self, odom):
        self.pos = [odom.pose.pose.position.x,odom.pose.pose.position.y]

    def finishTrunCallBack(self,msg):
        print('enter finish turn')
        self.isFinishTurn=True
    def startTrunCallBack(self,msg):
        print('enter start turn')
        self.isStartTurn=1

    def getPos(self):
        return self.pos


