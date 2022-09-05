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
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
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
        self.sub_lane = rospy.Subscriber('/robot5/detect/lane', Float64, self.cbFollowLane, queue_size=1)
        #self.sub_speed = rospy.Subscriber('/robot5/cmd_vel', Twist, self.speedCallBack, queue_size=1)
        self.pub_cmd_vel = rospy.Publisher('/robot5/cmd_vel', Twist, queue_size=1)
        self.sub_odom = rospy.Subscriber('/robot5/odom', Odometry, self.getOdometry)
        self.sub_scan = rospy.Subscriber('/robot5/camera/image_projected/compressed', CompressedImage, self.imgCallback, queue_size=1)
        self.pub_construct_lane = rospy.Publisher('/robot5/detect/construct_lane', Bool, queue_size=1)
        self.init_speed=[0.06,0.0]
        self.init_pos=[[-0.7774,-0.1255,-0.0010,0.0026,0.0027,-0.6905,0.7232],[-0.6073,-1.0843,-0.0010,0.0018,0.0033,-0.4817,0.8762],
                       [0.6954,-0.4487,-0.0010,-0.0033,0.0019,0.8678,0.4967],[0.5950,0.2485,-0.0010,-0.0022,0.0031,0.5694,0.8220]]
        self.last_pos=[-0.574910,0.627879]
        self.min_x = 0.04
        self.max_x = 0.16
        self.min_z = -0.2
        self.max_z = 0.2
        self.init_speed=[0.06,0.0]
        self.cur_center=500
        self.cv_image=np.zeros([1,120,200])
        self.counter=1
        self.rw_scale=20
        self.im_frames=[]
        self.stop_counter=0
        self.speed_x=0.06
        self.speed_z=0.0

    def reset(self):
        print('enter reset')
        self.init_speed = [0.06, 0.0]
        self.cur_center = 500
        index = np.random.randint(len(self.init_pos))
        print('index : ', index)
        init_pos=self.init_pos[index]
        self.last_pos = init_pos
        self.stop_counter = 0
        reset_simulation = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_simulation()

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            state_msg = ModelState()
            state_msg.model_name = 'robot5'
            state_msg.pose.position.x = init_pos[0]
            state_msg.pose.position.y = init_pos[1]
            state_msg.pose.position.z = init_pos[2]
            state_msg.pose.orientation.x = init_pos[3]
            state_msg.pose.orientation.y = init_pos[4]
            state_msg.pose.orientation.z = init_pos[5]
            state_msg.pose.orientation.w = init_pos[6]
            resp = set_state(state_msg)

        except:
            print("set model state Service call failed: %s")
        self.pub_construct_lane.publish(Bool(data=False))
        time.sleep(0.1)
        self.pub_construct_lane.publish(Bool(data=False))
        time.sleep(0.5)
        twist = Twist()
        twist.linear.x = self.init_speed[0]
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = self.init_speed[1]
        print('linear_speed: ', self.init_speed[0], ' angular_speed: ', self.init_speed[1])
        self.pub_cmd_vel.publish(twist)
        self.speed_x=0.06
        self.speed_z=0
        obs=self.get_obs()
        return obs, self.init_speed

    def get_obs(self):
        if len(self.im_frames)!=4:
            obs=np.zeros([4,120,200])
        else:
            obs=np.vstack(self.im_frames)
        return obs

    def step(self, action):
        print("action: ", action)
        if action == 0:
            self.speed_x = self.speed_x
            self.speed_z = self.speed_z
        elif action == 1:
            self.speed_x = self.speed_x - 0.02
        elif action == 2:
            self.speed_x = self.speed_x + 0.02
        elif action == 3:
            self.speed_z = self.speed_z - 0.02
        elif action == 4:
            self.speed_z = self.speed_z + 0.02
        elif action == 5:
            self.speed_x = self.speed_x - 0.02
            self.speed_z = self.speed_z - 0.02
        elif action == 6:
            self.speed_x = self.speed_x - 0.02
            self.speed_z = self.speed_z + 0.02
        elif action == 7:
            self.speed_x = self.speed_x + 0.02
            self.speed_z = self.speed_z - 0.02
        elif action == 8:
            self.speed_x = self.speed_x + 0.02
            self.speed_z = self.speed_z + 0.02
        self.speed_x = np.clip(self.speed_x, self.min_x, self.max_x)
        self.speed_z = np.clip(self.speed_z, self.min_z, self.max_z)
        twist = Twist()
        twist.linear.x = self.speed_x
        twist.angular.z = self.speed_z
        self.pub_cmd_vel.publish(twist)
        print('linear_speed: ', twist.linear.x, ' angular_speed: ', twist.angular.z)
        self.pub_cmd_vel.publish(twist)
        time.sleep(0.1)
        print('current center: ', self.cur_center)
        if 300<self.cur_center<650:
            done = False
            if self.speed_x==0:
                print('enter linear_speed==0')
                reward=0
                if self.stop_counter==5:
                    done=True
                else:
                    self.stop_counter+=1
            else:
                self.stop_counter=0
                offset_rw=-0.1*(1-abs(self.cur_center-500)/200)
                travel_rw = self.rw_scale * math.sqrt(
                    (self.last_pos[0] - self.pos[0]) ** 2 + (self.last_pos[1] - self.pos[1]) ** 2)
                print('offset_rw: ', offset_rw, 'travel_rw: ', travel_rw)
                reward=offset_rw+travel_rw
                self.last_pos = self.pos
        else:
            reward=-20
            done=True
        next_obs=self.get_obs()
        return next_obs,[self.speed_x,self.speed_z], reward,done


    def imgCallback(self, image_msg):
        if self.counter % 3 != 0:
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
        if random.random() > 0.8:
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

    def cbFollowLane(self, desired_center):
        center = desired_center.data
        self.cur_center=center

    def getOdometry(self, odom):
        self.pos = [odom.pose.pose.position.x,odom.pose.pose.position.y]