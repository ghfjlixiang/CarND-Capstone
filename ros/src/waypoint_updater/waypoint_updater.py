#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math

from std_msgs.msg import Int32
from scipy.spatial import KDTree
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 40 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 10     # Max acceleration is 10 m/s^2
MAX_JERK = 10      # Max jerk is 10 m/s^3

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.stopline_wp_idx = -1
        
        #rospy.spin()
        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if not None in(self.pose, self.base_waypoints, self.waypoints_tree):
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoints_tree.query([x,y], 1)[1]

        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])

        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def decelerate_waypoints(self, waypoints, stop_idx):
        wps = []
        for i,wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            dist = self.distance(waypoints, i, stop_idx)
            linear_vel = math.sqrt(2*MAX_DECEL*dist)
            if linear_vel < 1.0:
                linear_vel = 0
            p.twist.twist.linear.x = min(linear_vel, wp.twist.twist.linear.x)
            wps.append(p)
        rospy.loginfo("decelerate_waypoints: %s",stop_idx)
        return wps

    def generate_lane(self):
        lane = Lane()

        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS

        # rospy.loginfo("closest_idx: {0} farthest_idx: {1} stopline_wp_idx: {2}".format(closest_idx, farthest_idx, self.stopline_wp_idx))
        lane_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]
        if (self.stopline_wp_idx == -1) or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = lane_waypoints
        else:
			# Get stop idx which the car must stop in front of the stopline
            stopline_idx = max(self.stopline_wp_idx-closest_idx-10, 0)
            lane.waypoints = self.decelerate_waypoints(lane_waypoints, stopline_idx)

        return lane

    def publish_waypoints(self):
        # lane = Lane()
        # lane.header = self.base_waypoints.header
        # lane.waypoints = self.base_waypoints.waypoints[closest_idx:(closest_idx + LOOKAHEAD_WPS)]
        # self.final_waypoints_pub.publish(lane)

        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        # pass
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[Waypoint.pose.pose.position.x, Waypoint.pose.pose.position.y] for Waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)
            rospy.loginfo("base_waypoints size: {0}".format(len(self.waypoints_2d)))

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        # pass
        rospy.loginfo("traffic_waypoint topic message: %s",msg)
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
