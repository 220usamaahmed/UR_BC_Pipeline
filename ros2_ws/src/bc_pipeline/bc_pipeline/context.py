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
from ecpmi_gripper.srv import GripperControl
from geometry_msgs.msg import Point, Pose, Quaternion
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    OrientationConstraint,
    PlanningOptions,
    PositionConstraint,
)
from moveit_msgs.srv import GetCartesianPath, GetPositionFK
from shape_msgs.msg import SolidPrimitive


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
        # Forward kinematics: maps a set of joint angles to the EEF pose, so a
        # joint-space checkpoint can be combined with a Cartesian offset.
        self.fk_client = node.create_client(GetPositionFK, 'compute_fk')

        # ecpmi_gripper's suction_gripper_controller (real hardware only — see
        # its README). Not waited on in wait_for_servers(): most sequences run
        # against the mock driver, which never brings this service up, so
        # readiness is checked lazily the first time a Gripper step runs.
        self.gripper_client = node.create_client(GripperControl, 'gripper_control')

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
        while not self.fk_client.wait_for_service(timeout_sec=0.5):
            self.logger.info('  compute_fk not ready yet …')
        self.logger.info('All servers ready.')

    # ── joint-space plan + execute (used by Checkpoint) ────────────────────────

    def plan_and_execute_joints(self, angles: list) -> bool:
        """Plan + run a free motion to a joint-space goal."""
        gc = Constraints()
        for name, angle in zip(self.joint_names, angles):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(angle)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            gc.joint_constraints.append(jc)
        return self._plan_and_execute([gc])

    # ── pose plan + execute (used by Checkpoint with a cartesian_offset) ────────

    def plan_and_execute_pose(self, pose: Pose) -> bool:
        """Plan + run a free motion to a Cartesian pose goal (position + orientation).

        Unlike execute_cartesian (a straight-line slide from the current pose),
        this is an ordinary collision-aware joint-space plan whose goal is
        expressed as a pose — used to reach a checkpoint's pose plus an offset in
        one motion, without first stopping at the un-offset checkpoint.
        """
        gc = Constraints()

        # Constrain the EEF origin to a tiny sphere around the target point.
        pc = PositionConstraint()
        pc.header.frame_id = self.base_frame
        pc.link_name = self.eef_link
        pc.weight = 1.0
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.001]   # 1 mm tolerance
        region = BoundingVolume()
        region.primitives.append(sphere)
        region.primitive_poses.append(
            Pose(position=pose.position, orientation=Quaternion(w=1.0))
        )
        pc.constraint_region = region
        gc.position_constraints.append(pc)

        # Hold the EEF at the target orientation.
        oc = OrientationConstraint()
        oc.header.frame_id = self.base_frame
        oc.link_name = self.eef_link
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = 0.01
        oc.absolute_y_axis_tolerance = 0.01
        oc.absolute_z_axis_tolerance = 0.01
        oc.weight = 1.0
        gc.orientation_constraints.append(oc)

        return self._plan_and_execute([gc])

    def _plan_and_execute(self, goal_constraints: list) -> bool:
        """Send a MoveGroup goal with the given goal constraints, then block."""
        request = MotionPlanRequest()
        request.group_name = self.planning_group
        request.allowed_planning_time = self.planning_time
        request.num_planning_attempts = 10
        request.max_velocity_scaling_factor = self.velocity_scaling
        request.max_acceleration_scaling_factor = self.accel_scaling
        request.goal_constraints = goal_constraints

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

    # ── forward kinematics (used by Checkpoint with a cartesian_offset) ─────────

    def compute_fk(self, angles: list) -> Pose | None:
        """Return the eef_link pose in base_frame for the given joint angles.

        Lets a joint-space checkpoint be turned into a Cartesian target: we ask
        move_group where those angles put the end effector, then offset that
        pose.  Returns None on service failure.
        """
        req = GetPositionFK.Request()
        req.header.frame_id = self.base_frame
        req.fk_link_names = [self.eef_link]
        req.robot_state.joint_state.name = list(self.joint_names)
        req.robot_state.joint_state.position = [float(a) for a in angles]

        future = self.fk_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()

        if response is None or response.error_code.val != 1 or not response.pose_stamped:
            code = None if response is None else response.error_code.val
            self.logger.error(f'compute_fk failed (error code {code}).')
            return None
        return response.pose_stamped[0].pose

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

    # ── suction gripper (used by Gripper) ────────────────────────────────────

    def call_gripper(self, command: str, timeout_sec: float = 5.0) -> tuple:
        """Call ecpmi_gripper's gripper_control service. Returns (success, message)."""
        if not self.gripper_client.wait_for_service(timeout_sec=timeout_sec):
            return False, 'gripper_control service not available (is ecpmi_gripper running?)'

        request = GripperControl.Request()
        request.command = command

        future = self.gripper_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout_sec)
        response = future.result()
        if response is None:
            return False, 'gripper_control service call timed out'
        return response.success, response.message
