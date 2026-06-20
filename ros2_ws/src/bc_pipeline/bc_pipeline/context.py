#!/usr/bin/env python3
"""
Execution context — the MoveIt capability layer shared by every Step.

WHY A CONTEXT OBJECT
--------------------
Each Step describes *intent* ("go to checkpoint home", "slide 10 cm along
−Z").  The reusable *mechanics* — talking to move_group, reading the EEF pose
via TF, computing and running a Cartesian path — live here, once, so the Step
classes stay tiny and a new step type only writes the new behaviour.

The context owns:
  • the three MoveIt clients (move_action, execute_trajectory,
    compute_cartesian_path),
  • a TF buffer/listener for reading the live EEF pose,
  • the resolved robot/planning settings and the checkpoint table.

All readiness/spin patterns mirror the ur_moveit_demo examples: we use
server_is_ready() + spin_once() rather than wait_for_server(), which can block
forever in Docker when topic-endpoint DDS discovery is unreliable.
"""

import rclpy
import rclpy.duration
import rclpy.time
from rclpy.action import ActionClient

import tf2_ros
from geometry_msgs.msg import Point, Pose, Quaternion
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
)
from moveit_msgs.srv import GetCartesianPath


class Context:
    def __init__(self, node, config: dict):
        self.node = node
        self.logger = node.get_logger()

        robot = config['robot']
        self.planning_group = robot['planning_group']
        self.joint_names = list(robot['joint_names'])
        self.eef_link = robot['eef_link']
        self.base_frame = robot['base_frame']

        planning = config['planning']
        self.velocity_scaling = float(planning['velocity_scaling'])
        self.accel_scaling = float(planning['accel_scaling'])
        self.planning_time = float(planning['planning_time'])

        self.checkpoints = config['checkpoints']

        # move_group serves the MoveGroup action under the name 'move_action'
        # (NOT '/move_group', which is the node name).
        self.move_client = ActionClient(node, MoveGroup, 'move_action')
        self.exec_client = ActionClient(node, ExecuteTrajectory, 'execute_trajectory')
        self.cartesian_client = node.create_client(
            GetCartesianPath, 'compute_cartesian_path'
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)

    # ── readiness ─────────────────────────────────────────────────────────────

    def wait_for_servers(self):
        """Block (while spinning) until all MoveIt endpoints are discovered."""
        self.logger.info('Waiting for MoveIt servers …')
        while not self.move_client.server_is_ready():
            rclpy.spin_once(self.node, timeout_sec=0.5)
        while not self.exec_client.server_is_ready():
            rclpy.spin_once(self.node, timeout_sec=0.5)
        while not self.cartesian_client.wait_for_service(timeout_sec=0.5):
            self.logger.info('  compute_cartesian_path not ready yet …')
        self.logger.info('All servers ready.')

    # ── joint-space plan + execute (used by Checkpoint) ────────────────────────

    def plan_and_execute_joints(self, angles: list) -> bool:
        request = MotionPlanRequest()
        request.group_name = self.planning_group
        request.allowed_planning_time = self.planning_time
        request.num_planning_attempts = 10
        request.max_velocity_scaling_factor = self.velocity_scaling
        request.max_acceleration_scaling_factor = self.accel_scaling

        gc = Constraints()
        for name, angle in zip(self.joint_names, angles):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(angle)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            gc.joint_constraints.append(jc)
        request.goal_constraints = [gc]

        options = PlanningOptions()
        options.plan_only = False

        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options = options

        future = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, future)
        handle = future.result()
        if not handle.accepted:
            self.logger.error('Goal rejected by move_group.')
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)
        result = result_future.result().result

        if result.error_code.val == 1:
            return True
        self.logger.error(
            f'move_group error code: {result.error_code.val} '
            f'(see moveit_msgs/MoveItErrorCodes.msg)'
        )
        return False

    # ── live EEF pose via TF (used by OrientationLockCheckpoint) ────────────────

    def get_eef_pose(self) -> Pose | None:
        """Return the current eef_link pose in base_frame, or None on timeout."""
        deadline = self.node.get_clock().now() + rclpy.duration.Duration(seconds=3.0)
        while self.node.get_clock().now() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.base_frame, self.eef_link, rclpy.time.Time()
                )
                t = tf.transform.translation
                r = tf.transform.rotation
                pose = Pose()
                pose.position = Point(x=t.x, y=t.y, z=t.z)
                pose.orientation = Quaternion(x=r.x, y=r.y, z=r.z, w=r.w)
                return pose
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                pass
        self.logger.error(
            f'TF lookup {self.base_frame} → {self.eef_link} timed out after 3 s.'
        )
        return None

    # ── Cartesian plan + execute (used by OrientationLockCheckpoint) ────────────

    def execute_cartesian(
        self, label: str, waypoints: list, max_step: float, min_fraction: float
    ) -> bool:
        req = GetCartesianPath.Request()
        req.header.frame_id = self.base_frame
        req.header.stamp = self.node.get_clock().now().to_msg()
        req.group_name = self.planning_group
        req.link_name = self.eef_link
        req.waypoints = waypoints
        req.max_step = max_step
        req.jump_threshold = 0.0   # disabled — safe for short straight slides
        req.avoid_collisions = True

        future = self.cartesian_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()

        self.logger.info(
            f'  {label}: fraction {response.fraction:.1%}, '
            f'error {response.error_code.val}'
        )
        if response.fraction < min_fraction:
            self.logger.error(
                f'Only {response.fraction:.1%} of "{label}" reachable '
                f'(min {min_fraction:.1%}).'
            )
            return False

        exec_goal = ExecuteTrajectory.Goal()
        exec_goal.trajectory = response.solution

        exec_future = self.exec_client.send_goal_async(exec_goal)
        rclpy.spin_until_future_complete(self.node, exec_future)
        exec_handle = exec_future.result()
        if not exec_handle.accepted:
            self.logger.error('ExecuteTrajectory goal rejected.')
            return False

        result_future = exec_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)
        result = result_future.result().result

        if result.error_code.val == 1:
            return True
        self.logger.error(f'Execution error code: {result.error_code.val}')
        return False
