import rospy
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, UInt8, Bool
from sensor_msgs.msg import LaserScan, CompressedImage
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from math import radians
import numpy as np
import copy
import time
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from skimage.transform import rotate, AffineTransform, warp
from PIL import Image, ImageEnhance
import random
import math

INPUT_DIM=(120, 200, 1)
transform_type_dict = dict(
    brightness=ImageEnhance.Brightness, contrast=ImageEnhance.Contrast,
    sharpness=ImageEnhance.Sharpness, color=ImageEnhance.Color
)

class ColorJitter(object):
    def __init__(self, transform_dict):
        self.transforms = [(transform_type_dict[k], transform_dict[k]) for k in transform_dict]

    def __call__(self, img):
        out = img
        rand_num = np.random.uniform(0, 1, len(self.transforms))
        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (rand_num[i] * 2.0 - 1.0) + 1  # r in [1-alpha, 1+alpha)
            out = transformer(out).enhance(r)
        return out
class Env(object):
    def __init__(self):
        rospy.init_node('gazebo_env_node')
        self.pub_cmd_vel = rospy.Publisher('/robot10/cmd_vel', Twist, queue_size=1)
        self.sub_odom = rospy.Subscriber('/robot10/odom', Odometry, self.getOdometry)
        self.sub_scan = rospy.Subscriber('/robot10/scan', LaserScan, self.scanCallback, queue_size=1)
        self.sub_img = rospy.Subscriber('/robot10/camera/image_projected/compressed', CompressedImage, self.imgCallback, queue_size=1)
        self.pub_construct_lane = rospy.Publisher('/robot10/detect/construct_lane', Bool, queue_size=1)
        self.init_speed=[0.1,0.2]
        self.init_pos_list=[[-0.805519,0.284848],[-0.796213,0.177235],[-0.785670,-0.067999],[-0.788997,0.049891],[-0.783987,-0.189060],[-0.769798,-0.328783]]
        self.last_pos=[-0.805519,0.284848]
        self.robot8_pos=[-0.769861,-0.682819]
        self.target_speed=[0.1,0]
        self.speed_record=[]
        self.cv_image=np.zeros([1,120,200])
        self.counter=1
        self.rw_scale=1
        self.im_frames=[]
        self.stop_counter=0
        self.scan_data = [3.5] * 360

    def reset(self):
        print('enter reset')
        self.target_speed = [0.1, 0]
        self.speed_record = []
        index = np.random.randint(len(self.init_pos_list))
        #print('index : ', index)
        self.init_pos=self.init_pos_list[index]
        self.last_pos = self.init_pos
        self.pos=self.init_pos
        self.stop_counter = 0
        reset_simulation = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_simulation()

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            state_msg = ModelState()
            state_msg.model_name = 'robot10'
            state_msg.pose.position.x = self.init_pos[0]
            state_msg.pose.position.y = self.init_pos[1]
            state_msg.pose.position.z = -0.001003
            state_msg.pose.orientation.x = 0.002640941576348912
            state_msg.pose.orientation.y = 0.002830898435420755
            state_msg.pose.orientation.z = -0.6842180448216592
            state_msg.pose.orientation.w = 0.7292672202848998
            resp = set_state(state_msg)

        except:
            print("set model state Service call failed: %s")
        self.pub_construct_lane.publish(Bool(data=False))
        time.sleep(0.1)
        self.pub_construct_lane.publish(Bool(data=False))
        time.sleep(0.1)
        twist = Twist()
        twist.linear.x = self.target_speed[0]
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = self.target_speed[1]
        print('linear_speed: ', self.target_speed[0], ' angular_speed: ', self.target_speed[1])
        self.pub_cmd_vel.publish(twist)
        obs=self.get_obs()
        return obs, self.target_speed

    def turn_back(self):
        self.speed_record.reverse()
        for speed in self.speed_record:
            twist = Twist()
            twist.linear.x = speed[0]
            twist.linear.y = 0
            twist.linear.z = 0
            twist.angular.x = 0
            twist.angular.y = 0
            twist.angular.z = -speed[1]
            print('linear_speed: ', self.target_speed[0], ' angular_speed: ', self.target_speed[1])
            self.pub_cmd_vel.publish(twist)
            time.sleep(0.6)

    def cal_Dist(self,x0,y0,x1,y1,x2,y2):
        k=(y1-y2)/(x1-x2)
        return abs(k*(x0-x1)-(y0-y1))/math.sqrt(1+k**2)

    def get_obs(self):
        if len(self.im_frames)!=4:
            obs=np.zeros([4,120,200])
        else:
            obs=np.vstack(self.im_frames)
        return obs

    def step(self, action):
        print('the action is: ', action)
        self.target_speed[0]=np.clip(action[0], 0.1, 0.2)
        self.target_speed[1]=np.clip(action[1], 0.12, 0.25)
        twist=Twist()
        twist.linear.x=self.target_speed[0]
        twist.linear.y =0
        twist.linear.z=0
        twist.angular.x=0
        twist.angular.y=0
        twist.angular.z=self.target_speed[1]
        #print('linear_speed: ', self.target_speed[0], ' angular_speed: ', self.target_speed[1])
        self.pub_cmd_vel.publish(twist)
        time.sleep(0.6)
        curr_pos=self.pos
        dist=self.cal_Dist(curr_pos[0],curr_pos[1],self.init_pos[0],self.init_pos[1],self.robot8_pos[0],self.robot8_pos[1])
        #print('dist: ', dist)
        front_data = [self.scan_data[i] for i in range(-40, 40)]
        front_dist = np.min(front_data)
        #print('front_dist: ', front_dist)
        # dist=0l.1292236
        successFlag=0
        if dist>0.13 or front_dist<0.3:
            done=True
            reward=-20
        elif 0.1<dist<0.13:
            done=True
            successFlag=1
            reward=20
        else:
            done=False
            reward=self.rw_scale*dist
        next_obs=self.get_obs()
        self.speed_record.append([twist.linear.x,twist.angular.z])
        return next_obs,self.target_speed, reward,done,successFlag

    def imgCallback(self, image_msg):
        if self.counter % 2 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1
        np_arr = np.fromstring(image_msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        im = cv2.resize(cv_image, (200, 120), interpolation=cv2.INTER_AREA)
        _transform_dict = {'brightness': 0.5, 'contrast': 0.3, 'sharpness': 0.8386, 'color': 0.1592}
        _color_jitter = ColorJitter(_transform_dict)
        im = Image.fromarray(im)
        im = _color_jitter(im)
        # 做完color jitter之后，再将Image对象转回numpy array
        im = np.array(im)  # 转换完之后，这里的img是unit8类型的
        # 2. gaussian filter
        im = cv2.GaussianBlur(im, (5, 5), 0)
        # 3. color to gray
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = im.reshape(INPUT_DIM)
        # transform the image x
        if random.random() > 0.9:
            angle = random.randint(-15, 15)
            transform = AffineTransform(translation=(angle, 0))  # (-200,0) are x and y coordinate
            im = warp(im, transform, mode="constant") * 255.
        im = im.reshape(1,120,200)
        #cv2.imwrite('test.png', self.cv_image)
        if len(self.im_frames)!=4:
            self.im_frames.append(im)
        else:
            self.im_frames.pop(0)
            self.im_frames.append(im)

    def getOdometry(self, odom):
        self.pos = [odom.pose.pose.position.x,odom.pose.pose.position.y]
        #print('pos callback: ',self.pos)

    def scanCallback(self,data):
        #print('enter scan call back')
        if self.counter % 2 != 0:
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
