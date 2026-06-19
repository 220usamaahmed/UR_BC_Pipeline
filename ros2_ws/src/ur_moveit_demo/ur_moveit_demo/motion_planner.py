#!/usr/bin/env python3
"""
Motion Planner — plans and executes joint-space goals through MoveIt.

WHAT THIS TEACHES
-----------------
This node talks directly to the /move_group action server.  Every high-level
MoveIt library is a wrapper around this same action.  Talking to it directly
shows you exactly what information MoveIt needs and what it gives back.

WHY server_is_ready() INSTEAD OF wait_for_server()
----------------------------------------------------
ActionClient.wait_for_server() requires the DDS layer to discover BOTH the
action server's service endpoints (send_goal, cancel_goal, get_result) AND
its topic endpoints (feedback, status).

In some Docker configurations, topic-endpoint DDS discovery is unreliable —
services are found immediately but topics are never discovered.  The result:
wait_for_server() blocks forever even though the server is running and
accepting goals.

server_is_ready() just checks the current discovered state; it never blocks.
Pairing it with rclpy.spin_once() ensures pending DDS events are processed
each iteration.  Once the server is ready, rclpy.spin_until_future_complete()
handles all callback spinning for goal sending and result waiting.

THE /move_group ACTION
-----------------------
Action type: moveit_msgs/action/MoveGroup

  Goal
  ├── MotionPlanRequest   ← what I want
  │     ├── group_name                the joint group to plan for
  │     ├── goal_constraints[]        target configuration(s)
  │     ├── allowed_planning_time     seconds the planner may search
  │     └── max_velocity_scaling_factor, etc.
  └── PlanningOptions
        └── plan_only     False = plan AND execute; True = plan only

  Result (returned after execution completes)
  ├── error_code          1 = SUCCESS
  ├── planned_trajectory  the joint trajectory that was executed
  └── planning_time       seconds the planner actually used

WAYPOINTS
---------
home → left → raised → right → home

Run after scene_builder:

    ros2 run ur_moveit_demo motion_planner
"""

import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
)


PLANNING_GROUP = 'ur_manipulator'

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

WAYPOINTS = {
    'home':   [ 0.00, -1.57,  1.57, -1.57, -1.57,  0.0],
    'left':   [ 1.57, -1.00,  1.00, -1.57, -1.57,  0.0],
    'raised': [ 0.00, -1.00,  0.50, -1.00, -1.57,  0.0],
    'right':  [-1.57, -1.00,  1.00, -1.57, -1.57,  0.0],
}

WAYPOINT_ORDER = ['home', 'left', 'raised', 'right', 'home']

# With use_fake_hardware the controller reports a trajectory "done" a moment
# before move_group's state monitor settles and the previous controller goal
# fully terminates.  Firing the next goal inside that window occasionally trips
# move_group's execution manager, which returns generic FAILURE (99999) almost
# instantly.  A short settle pause avoids it; a couple of retries absorb the
# rare one that still slips through.
SETTLE_SEC = 0.5
MAX_ATTEMPTS = 3


class MotionPlanner(Node):
    def __init__(self):
        super().__init__('motion_planner')
        # The move_group NODE serves the MoveGroup action under the name
        # 'move_action' (MoveIt's move_group::MOVE_ACTION). The action is NOT
        # named '/move_group' — that's just the node name. Confirm with:
        #   ros2 action list   ->   /move_action
        self._client = ActionClient(self, MoveGroup, 'move_action')

    def run(self):
        # ------------------------------------------------------------------
        # Wait for the /move_group action server
        #
        # We poll server_is_ready() and call spin_once() each iteration so
        # that DDS discovery events are processed.  This is more reliable
        # than wait_for_server() in Docker where topic-endpoint discovery
        # can fail (see module docstring).
        # ------------------------------------------------------------------
        self.get_logger().info('Waiting for /move_group action server ...')
        while not self._client.server_is_ready():
            rclpy.spin_once(self, timeout_sec=0.5)
        self.get_logger().info('Connected to move_group.')

        for name in WAYPOINT_ORDER:
            angles = WAYPOINTS[name]
            self.get_logger().info(f'\n{"=" * 52}')
            self.get_logger().info(f'Waypoint : {name}')
            self.get_logger().info(f'Angles   : {angles}')

            if not self._move_to_with_retries(name, angles):
                self.get_logger().error(f'Stopping after failure at "{name}".')
                return

            # Let move_group's current-state monitor catch up before the next
            # goal so we don't race the just-finished execution (see SETTLE_SEC).
            time.sleep(SETTLE_SEC)

        self.get_logger().info('All waypoints reached.')

    def _move_to_with_retries(self, name: str, angles: list) -> bool:
        """Plan+execute one waypoint, retrying transient move_group failures."""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if self._move_to(name, angles):
                return True
            if attempt < MAX_ATTEMPTS:
                self.get_logger().warn(
                    f'"{name}" failed (attempt {attempt}/{MAX_ATTEMPTS}); '
                    f'letting state settle and retrying ...'
                )
                time.sleep(SETTLE_SEC)
        return False

    def _move_to(self, name: str, angles: list) -> bool:

        # ------------------------------------------------------------------
        # 1. MotionPlanRequest — describe the planning problem
        # ------------------------------------------------------------------
        request = MotionPlanRequest()
        request.group_name = PLANNING_GROUP
        request.allowed_planning_time = 5.0
        request.num_planning_attempts = 10
        request.max_velocity_scaling_factor = 0.5
        request.max_acceleration_scaling_factor = 0.5

        # ------------------------------------------------------------------
        # 2. Goal constraints — target joint configuration
        #
        # Each JointConstraint says "this joint must reach this angle within
        # ±tolerance".  Bundling them in a Constraints object lets you mix
        # joint, position, and orientation constraints if needed.
        # ------------------------------------------------------------------
        goal_constraints = Constraints()
        for joint_name, angle in zip(JOINT_NAMES, angles):
            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = angle
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            goal_constraints.joint_constraints.append(jc)
        request.goal_constraints = [goal_constraints]

        # ------------------------------------------------------------------
        # 3. PlanningOptions — plan AND execute in one call
        # ------------------------------------------------------------------
        options = PlanningOptions()
        options.plan_only = False

        goal_msg = MoveGroup.Goal()
        goal_msg.request = request
        goal_msg.planning_options = options

        # ------------------------------------------------------------------
        # 4. Send the goal
        #
        # send_goal_async() returns a Future that completes when the server
        # accepts or rejects the goal.  spin_until_future_complete() spins
        # the node's callback queue until that Future resolves.
        # ------------------------------------------------------------------
        self.get_logger().info('Sending to /move_group ...')
        goal_future = self._client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, goal_future)

        goal_handle = goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by move_group.')
            return False

        self.get_logger().info('Planning and executing ...')

        # ------------------------------------------------------------------
        # 5. Wait for the result
        #
        # get_result_async() returns another Future that completes when the
        # action finishes (after planning + full trajectory execution).
        # spin_until_future_complete() keeps spinning for however long that
        # takes — typically several seconds per waypoint.
        # ------------------------------------------------------------------
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result

        # MoveItErrorCodes: 1 = SUCCESS.  Full list: moveit_msgs/MoveItErrorCodes.msg
        if result.error_code.val == 1:
            pts = len(result.planned_trajectory.joint_trajectory.points)
            self.get_logger().info(
                f'Done.  Trajectory: {pts} points, '
                f'planning time: {result.planning_time:.2f} s'
            )
            return True

        self.get_logger().error(
            f'move_group error code: {result.error_code.val} '
            f'(see moveit_msgs/MoveItErrorCodes.msg)'
        )
        return False


def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanner()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
