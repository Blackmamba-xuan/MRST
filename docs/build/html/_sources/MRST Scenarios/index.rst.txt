MRST Scenarios
==============

In this project, we provide diverse scenarios including single vehicle lane tracking and multi-vehicle cooperation.

Single-vehicle Lane Tracking
-----------------------------

.. figure:: Media/lane_tracking.gif
   :alt: lane tracking
   :width: 4.5in
   :height: 2.8in
   :align: center

In this scenario, the task of the vehicle is learn to tracking the lane in a short time period.

**State Space:** The vehicle's state includes its' position and front_view image. User can customize the state space in the **get_obs()** function.

**Action Space(discrete):** Choose whether to keep current speed, speed up, slow down or change to another lane.

**Action Space(continuous):** linear speed and angular speed.

**Reward Function:** 

.. figure:: Media/lane_tracking_reward_function.png
   :alt: multi-vehicle scenarios
   :align: center

Where d_lateral is the deviation to the center of the lane

Multi-vehicle Cooperation
-----------------------------

Instead of single-vehicle task, we provides several multi-vehicle cooperation tasks such as **cooperative lane change** and **multiv-vehicle intersection cooperation**.

.. figure:: Media/scenarios.png
   :alt: multi-vehicle scenarios
   :align: center

**Cooperative Lane Change:** A two-lane traffic road with four vehicles. One of the vehicle in right lane is randomly set to stopped in order to simulate the traffic accident or congestion. The vehicle behind is required to successfully perform lane change without collision with others.

**Intersection coordination:** We also consider several unprotected four-way intersections allowing bi-directional traffic as well as U-turns. Coordination and social behaviors are expected to learn in order to avoid collision. Two different intersection with different number of vehicles are tested in this experiments.

**Roundabout Coordination:** A four-way roundabout with two lanes. Each vehicle is assigned with a started lane and destination lane.This environment also includes merge and split junctions.

**T-junction:** T-junction is a common route in city road, where the vehicles need to perform cooperative left turn with others. We initialize 21 vehicles in this scenario.

**Simple Cross-lane:** A simple version of the intersection with 4 vehicles. We initialize the vehicles with fast speed in this scenario.