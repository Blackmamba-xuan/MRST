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
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import os, sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

class Scenario(object):
    def __init__(self,ScenarioID=0, require_social_agnet=False):
        print('creating crosslane22')
        rospy.init_node('gazebo_env_crossLane')
        rate = rospy.Rate(1)
        self.nagents = 3
        robot_pos1 = [[-0.574424, 0.633950]]
        robot_pos2 = [[-0.805519, 0.284848], [-0.796213, 0.177235]]
        robot_pos3 = [[-0.559980, 0.018121], [-0.553760, -0.099075], [-0.557075, -0.248758]]
        robot_pos4 = [[-0.769542, -0.682726], [-2.244528, -0.949694]]
        self.initPos = [robot_pos1, robot_pos2, robot_pos3, robot_pos4]
        lane_flag = [0, 1, 0]
        if ScenarioID>0:
            self.eneities = [Entity('/robot' + str(i +9), self.initPos[i],lane_flag[i]) for i in range(self.nagents)]
            self.names=['robot10','robot11','robot12']
        else:
            self.eneities = [Entity('/robot' + str(i + 5), self.initPos[i],lane_flag[i]) for i in range(self.nagents)]
            self.names = ['robot6', 'robot7', 'robot8']
        self.lastPos = []
        self.rw_scale = 5
        self.turn_success_flag=0

        # whether include social agent or not
        self.require_social_agnet=require_social_agnet
        # if self.require_social_agnet:
        #     self.social_agents=[Entity('/agent' + str(i+8), self.initPos[i]) for i in range(1)]

    def reset(self):
        print('enter reset')
        self.turn_success_flag=0
        reset_simulation = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_simulation()
        idx1 = np.random.randint(2)
        robot6_initPos = self.initPos[1][idx1]
        print(idx1)
        idx2 = np.random.randint(3)
        robot7_initPos = self.initPos[2][idx2]
        print(idx2)
        idx3 = np.random.randint(2)
        robot8_initPos = self.initPos[3][idx3]
        print('enter set model service')
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            state_msg = ModelState()
            state_msg.model_name = self.names[0]
            state_msg.pose.position.x = robot6_initPos[0]
            state_msg.pose.position.y = robot6_initPos[1]
            state_msg.pose.position.z = -0.001003
            state_msg.pose.orientation.x = 0.002640941576348912
            state_msg.pose.orientation.y = 0.002830898435420755
            state_msg.pose.orientation.z = -0.6842180448216592
            state_msg.pose.orientation.w = 0.7292672202848998
            resp = set_state(state_msg)

            state_msg2 = ModelState()
            state_msg2.model_name = self.names[1]
            state_msg2.pose.position.x = robot7_initPos[0]
            state_msg2.pose.position.y = robot7_initPos[1]
            state_msg2.pose.position.z = -0.001003
            state_msg2.pose.orientation.x = 0.00264318220817986
            state_msg2.pose.orientation.y = 0.002815092351699225
            state_msg2.pose.orientation.z = -0.6858120802965438
            state_msg2.pose.orientation.w = 0.7277684242684571
            resp2 = set_state(state_msg2)

            state_msg3 = ModelState()
            state_msg3.model_name = self.names[2]
            state_msg3.pose.position.x = robot8_initPos[0]
            state_msg3.pose.position.y = robot8_initPos[1]
            state_msg3.pose.position.z = -0.001003
            state_msg3.pose.orientation.x = 0.00264318220817986
            state_msg3.pose.orientation.y = 0.002815092351699225
            state_msg3.pose.orientation.z = -0.6858120802965438
            state_msg3.pose.orientation.w = 0.7277684242684571
            resp3 = set_state(state_msg3)

        except:
            print("set model state Service call failed: %s")
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
        speeds = []
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
                speeds.append(entity.speed_x)
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
                speeds.append(entity.speed_x)
        dones = np.full((1, self.nagents), done_flag)
        # step the social agent
        # if self.require_social_agnet:
        #     for social_agent in self.social_agents:
        #         social_agent.step()
        info={}
        return np.array([obs]), np.array([rewards]), dones,speeds

class Entity(object):

    def __init__(self,name,pos,lane_flag):
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
        self.lidar_frames = [[3.5] * 360 for i in range(1)]
        self.init_lane_flag = lane_flag
        self.lane_flag = self.init_lane_flag
        self.isFinishTurn=False
        self.isStartTurn=0


    def step(self,action):
        behavior_msg = UInt8()
        behavior_msg.data = np.uint8(action)
        self.pub_lane_behavior.publish(behavior_msg)

    def reset(self):
        self.isFinishTurn = False
        self.isTurnFlag = 0
        self.lane_flag=self.init_lane_flag
        self.pub_reSet.publish(Bool(data=True))

    def getObs(self):
        obs=copy.deepcopy(self.scan_data)
        #print('')
        obs = np.array(obs).reshape(-1)
        obs = np.append(obs, self.speed_x)
        obs = np.append(obs,self.isStartTurn)
        obs = np.append(obs,self.lane_flag)
        obs = np.append(obs,self.pos[0])
        obs = np.append(obs,self.pos[1])
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
            elif scan.ranges[i] == float('inf'):
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