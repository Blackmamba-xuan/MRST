Support for ROS Developer
===========================

One of the main characteristics of our platform is we intergrate the communication and control of ROS. User can get the sensor data and control the robot in the ROSEntity class.

.. highlight:: sh
::

    class ROSEntity(object):

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


    def step(self,action):
        behavior_msg = UInt8()
        behavior_msg.data = np.uint8(action)
        self.pub_lane_behavior.publish(behavior_msg)

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