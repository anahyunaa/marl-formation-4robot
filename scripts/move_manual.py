#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from math import sqrt


class MultiRobotController:
    def __init__(self, robot_names):
        self.robot_names = robot_names

        # Publisher & Subscriber container
        self.publishers = {}
        self.positions = {}  # { 'r1': (x, y), ... }

        # Init publishers & subscribers
        for name in self.robot_names:
            cmd_topic = f"/{name}/cmd_vel"
            odom_topic = f"/{name}/odom"

            self.publishers[name] = rospy.Publisher(
                cmd_topic, Twist, queue_size=10
            )

            rospy.Subscriber(
                odom_topic,
                Odometry,
                self.odom_callback,
                callback_args=name
            )

            self.positions[name] = (0.0, 0.0)

        rospy.loginfo("MultiRobotController initialized")

        # Pre-create message
        self.cmd_msg = Twist()
        self.cmd_msg.linear.x = 0.5
        self.cmd_msg.angular.z = 0.0

        # Target condition
        self.max_distance = 5.0

        # Rate
        self.rate = rospy.Rate(10)

        # kasih waktu subscriber connect
        rospy.sleep(1.0)

    def odom_callback(self, msg, robot_name):
        # ONLY update data, jangan taruh logic berat di sini
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.positions[robot_name] = (x, y)

    def compute_distance(self, pos):
        x, y = pos
        return sqrt(x**2 + y**2)

    def main_loop(self):
        while not rospy.is_shutdown():
            stop_all = False

            # cek kondisi berhenti
            for name in self.robot_names:
                dist = self.compute_distance(self.positions[name])
                if dist >= self.max_distance:
                    stop_all = True
                    break

            # publish command
            for name, pub in self.publishers.items():
                if stop_all:
                    stop_msg = Twist()  # default = 0
                    pub.publish(stop_msg)
                else:
                    pub.publish(self.cmd_msg)

            # debug ringan (1x per detik)
            rospy.loginfo_throttle(1, f"Positions: {self.positions}")

            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node("multi_robot_controller")

    robots = ["r1", "r2", "r3", "r4"]

    controller = MultiRobotController(robots)
    controller.main_loop()
