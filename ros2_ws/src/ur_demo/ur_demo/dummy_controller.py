#!/usr/bin/env python3
"""Dummy controller that sends a single trajectory goal to the UR3e arm.

This is the Python equivalent of the ``ros2 action send_goal ...`` command in
the README. It sends one FollowJointTrajectory goal moving the arm to a target
pose, then waits for the result and exits.

Run it inside the container with either:

    ros2 run ur_demo dummy_controller

or directly, without building:

    python3 src/ur_demo/ur_demo/dummy_controller.py
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

# Target joint positions (radians) and how long to take getting there.
TARGET_POSITIONS = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
TIME_FROM_START_SEC = 5

ACTION_NAME = '/scaled_joint_trajectory_controller/follow_joint_trajectory'


class DummyController(Node):
    def __init__(self):
        super().__init__('dummy_controller')
        self._client = ActionClient(self, FollowJointTrajectory, ACTION_NAME)

    def send_goal(self):
        self.get_logger().info(f'Waiting for action server {ACTION_NAME} ...')
        self._client.wait_for_server()

        point = JointTrajectoryPoint()
        point.positions = TARGET_POSITIONS
        point.time_from_start = Duration(sec=TIME_FROM_START_SEC, nanosec=0)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        goal.trajectory.points = [point]

        self.get_logger().info(f'Sending goal: {TARGET_POSITIONS}')
        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal was rejected by the controller.')
            return

        self.get_logger().info('Goal accepted, waiting for result ...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        self.get_logger().info(f'Done. error_code={result.error_code}')


def main(args=None):
    rclpy.init(args=args)
    node = DummyController()
    try:
        node.send_goal()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
