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
        print('creating crosslane22')
        rospy.init_node('gazebo_env_crossLane')
        rate = rospy.Rate(1)
        self.nagents = 4
        initPos = [[2.108462, 0.112866], [-0.103539, 1.587786], [-1.785433, -0.116726], [0.107544, -0.116726]]
        if ScenarioID>0:
            self.eneities = [Entity('/robot' + str(i +9), initPos[i]) for i in range(self.nagents)]
        else:
            self.eneities = [Entity('/robot' + str(i + 5), initPos[i]) for i in range(self.nagents)]
        self.lastPos = []
        self.rw_scale = 5
        self.turn_success_flag=0

        # whether include social agent or not
        self.require_social_agnet=require_social_agnet
        if self.require_social_agnet:
            self.social_agents=[Entity('/agent' + str(i), initPos[i]) for i in range(2)]

    def reset(self):
        print('enter reset')
        self.turn_success_flag = 0
        reset_simulation = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_simulation()
        time.sleep(0.2)
        obs = [self.eneities[i].getObs() for i in range(self.nagents)]
        self.lastPos = []
        for entity in self.eneities:
            self.lastPos.append(entity.getPos())
            entity.reset()
        time.sleep(0.5)
        return np.array([obs])

    def step(self, actions,isTeamReward=False):
        actions = actions[0]
        actions = [np.argmax(action) for action in actions]
        for entity, action in zip(self.eneities, actions):
            entity.step(action)
        time.sleep(0.6)
        obs = []
        rewards = []
        done_flag = 0
        # calculate reward
        if isTeamReward:
            team_reward=0
            for i, entity in zip(range(self.nagents), self.eneities):
                ob = entity.getObs()
                scan_msg = entity.scan_data
                # print(scan_data)
                front_data = [scan_msg[i] for i in range(-30, 30)]
                front_dist = np.min(front_data)
                # print(entity.name, 'front_dist: ', front_dist)
                if front_dist < 0.2:
                    team_reward = -20
                    done_flag = 1
                else:
                    lastP = self.lastPos[i]
                    cur_pos = entity.pos
                    team_reward+= self.rw_scale * math.sqrt((lastP[0] - cur_pos[0]) ** 2 + (lastP[1] - cur_pos[1]) ** 2)
                    self.lastPos[i] = cur_pos
                    if entity.isFinishTurn:
                        team_reward+=100
                        entity.isFinishTurn=False
                        self.turn_success_flag+=1
                obs.append(ob)
            rewards=np.array([team_reward]*self.nagents)
        else:
            for i, entity in zip(range(self.nagents), self.eneities):
                ob = entity.getObs()
                scan_msg = entity.scan_data
                # print(scan_data)
                front_data = [scan_msg[i] for i in range(-30, 30)]
                front_dist = np.min(front_data)
                # print(entity.name, 'front_dist: ', front_dist)
                if front_dist < 0.2:
                    reward = -20
                    done_flag = 1
                else:
                    lastP = self.lastPos[i]
                    cur_pos = entity.pos
                    reward = self.rw_scale * math.sqrt((lastP[0] - cur_pos[0]) ** 2 + (lastP[1] - cur_pos[1]) ** 2)
                    if entity.isFinishTurn:
                        reward+=100
                        entity.isFinishTurn=False
                        self.turn_success_flag+=1
                    self.lastPos[i] = cur_pos
                rewards.append(reward)
                obs.append(ob)
        dones = np.full((1, self.nagents), done_flag)

        # step the social agent
        if self.require_social_agnet:
            for social_agent in self.social_agents:
                social_agent.step()
        info={}
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
        self.scan_data = [3.5]*360
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
        #print('')
        obs.append(self.isStartTurn)
        obs.append(self.speed_x)
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
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])
        self.scan_data=scan_range

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

class SocialAgent(object):
    def __init__(self,name,max_speed=0.2, max_acc_speed=0.04,decre_speed=0.02):
        self.name=name
        self.max_speed=max_speed
        self.a=max_acc_speed
        self.b=0.02
        self.safe_dist=0.2
        self.react_time=1
        self.speed_x=0.16

        # ROS Topic
        self.sub_scan = rospy.Subscriber(name + '/scan', LaserScan, self.scanCallback, queue_size=1)
        self.sub_odom = rospy.Subscriber(name + '/odom', Odometry, self.getOdometry, queue_size=1)
        self.sub_speed = rospy.Subscriber(name + '/cmd_vel', Twist, self.speedCallBack, queue_size=1)
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=1)

    def step(self, min_dist, fron_speed):
        print('enter social agent step')
        delta_v=self.speed_x-fron_speed
        delta_s=self.safe_dist+max(self.speed_x*self.react_time+(self.speed_x*delta_v/2*math.sqrt(self.a*self.b)),0)
        delta_sp=random.random()
        delta_s+=delta_sp
        acc_speed=self.a*(1-math.pow(self.speed_x/self.max_speed,4)-math.pow(delta_s,min_dist))
        self.speed_x=self.speed_x+acc_speed
        print('acc_speed: ', acc_speed, ' new speed: ', self.speed_x)
        twist=Twist()
        twist.linear.x=self.speed_x
        self.pub_cmd_vel.publish(twist)


    def scanCallback(self,data):
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1
        print('enter scanCallback')
        scan = data
        scan_range = []
        # print('scan_data_lenth: ',len(scan.ranges))
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])
        self.scan_data=scan_range

    def speedCallBack(self, msg):
        self.speed_x = msg.linear.x

    def getOdometry(self, odom):
        self.pos = [odom.pose.pose.position.x,odom.pose.pose.position.y]