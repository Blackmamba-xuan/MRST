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
        print('creating roundabout')
        rospy.init_node('gazebo_env_crossLane')
        rate = rospy.Rate(1)
        self.nagents = 12
        self.social_num=2
        names = ['vehicle1', 'vehicle2', 'vehicle3', 'vehicle5', 'vehicle6', 'vehicle8',
                 'vehicle9', 'vehicle11', 'vehicle12','vehicle13', 'vehicle15', 'vehicle16']
        initPos = [[-1.346932, 0.399113], [-3.043667, -0.407096], [-2.660860, -0.157350],
                   [0.394815, -3.139112],[-0.989011, 1.326105], [1.099457, -1.287574],
                   [3.337716,0.375457],[1.220001,-0.759366],[1.386080,0.924039],
                   [-0.127716,3.474771],[0.793265,1.183017],[-0.989011,1.326105]
                   ]
        self.eneities = [Entity(names[i], initPos[i]) for i in range(self.nagents)]
        self.lastPos = []
        self.rw_scale = 1
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
        self.lastPos = []
        for entity in self.eneities:
            entity.reset()
        time.sleep(0.3)

        self.socialAgentID = np.random.choice(self.nagents,self.social_num,replace=False).tolist()

        for entity in self.eneities:
            self.lastPos.append(entity.getPos())
        obs = [self.eneities[i].getObs() for i in range(self.nagents)]
        return np.array([obs])

    def step(self, actions,isTeamReward=False):
        actions = actions[0]
        actions = [np.argmax(action) for action in actions]
        print('step actions: ', actions)
        for i, action in enumerate(actions):
            if i == self.socialAgentID:
                print('step social agent: ', i)
                # calculate the mini distance
                social_agent = self.eneities[i]
                agent_dists = [math.sqrt((self.eneities[j].pos[0] - social_agent.pos[0]) ** 2 +
                                         (self.eneities[j].pos[1] - social_agent.pos[1]) ** 2) for j in
                               range(self.nagents) if j != i]
                min_dist = min(agent_dists)
                print('min_dist: ', min_dist)
                social_agent.stepIDM(min_dist, agent_dists.index(min_dist))
            else:
                entity = self.eneities[i]
                entity.step(action)
        time.sleep(0.6)
        obs = []
        rewards = []
        speeds = []
        pos=[]
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
                lastP = self.lastPos[i]
                cur_pos = entity.pos
                pos.append(cur_pos)
                if front_dist < 0.2:
                    team_reward = -100
                    done_flag = 1
                else:
                    team_reward+= self.rw_scale * math.sqrt((lastP[0] - cur_pos[0]) ** 2 + (lastP[1] - cur_pos[1]) ** 2)
                    self.lastPos[i] = cur_pos
                    if entity.isFinishTurn:
                        team_reward+=25
                        entity.isFinishTurn=False
                        self.turn_success_flag+=1
                speeds.append(entity.speed_x)
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
                lastP = self.lastPos[i]
                cur_pos = entity.pos
                pos.append(cur_pos)
                if front_dist < 0.2:
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
                speeds.append(entity.speed_x)
                obs.append(ob)
        dones = np.full((1, self.nagents), done_flag)
        info={
            'pos':np.array(pos),
            'speeds':speeds
        }
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
        #self.pub_cmd_vel = rospy.Publisher(name + 'cmd_vel', Twist, queue_size=1)
        self.scan_data = [3.5]*36
        self.name=name
        self.speed_x=0.1
        self.pos=pos
        self.isFinishTurn=False
        self.isStartTurn=0

        # parameters of IDM model
        self.max_speed = 0.2
        self.a = 0.02
        self.b = 0.02
        self.safe_dist = 0.2
        self.react_time = 1
        self.speed_x = 0.16


    def step(self,action):
        behavior_msg = UInt8()
        behavior_msg.data = np.uint8(action)
        self.pub_lane_behavior.publish(behavior_msg)

    def stepIDM(self,min_dist, fron_speed):
        print()
        delta_v = self.speed_x - fron_speed
        delta_s = self.safe_dist + max(
            self.speed_x * self.react_time + (self.speed_x * delta_v / 2 * math.sqrt(self.a * self.b)), 0)
        delta_sp = random.random()
        delta_s += delta_sp
        acc_speed = self.a * (1 - math.pow(self.speed_x / self.max_speed, 4) - math.pow(delta_s, min_dist))
        self.speed_x = self.speed_x + acc_speed

    def reset(self):
        self.isFinishTurn = False
        self.isTurnFlag = 0
        self.pub_reSet.publish(Bool(data=True))
        self.pub_reSet.publish(Bool(data=True))
        self.pub_reSet.publish(Bool(data=True))

    def getObs(self):
        obs=copy.deepcopy(self.scan_data)
        #print('')
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
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])
        scan_range36 = [np.min(scan_range[i * 10:(i + 1) * 10]) for i in range(36)]
        self.scan_data = scan_range36

    def speedCallBack(self, msg):
        #print('enter speed callback: ', msg.linear.x)
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