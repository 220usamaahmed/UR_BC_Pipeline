#!/usr/bin/env python3
"""
Constrained Cartesian Motion — drawer-pull demo.

Moves the end effector in a straight line while keeping its orientation
fixed, the way a suction-cup gripper must behave when pulling open a drawer:
the face of the gripper stays parallel to the drawer surface throughout.

CONCEPT
-------
Joint-space planning (motion_planner.py) finds ANY collision-free path from
A to B.  The planner may swing the end effector through intermediate poses
that change its orientation — fine for pick-and-place, but for a suction
pull that would break the seal or tilt the load.

Cartesian path planning solves this:
  • You supply a list of target poses (position + orientation) in world space.
  • MoveIt interpolates between them in Cartesian space, calling IK at each step.
  • All waypoints share the same orientation  →  orientation is constant.
  • max_step (metres) controls how densely IK is called along the line.

WHY NOT move_action WITH OrientationConstraint?
-----------------------------------------------
You can add an OrientationConstraint to a MoveGroup goal, but OMPL (the
default planner) treats constraints as guidance, not guarantees.  Plans that
deviate slightly are still accepted as long as the goal is reached.

GetCartesianPath is the hard guarantee: orientation is locked by construction
because you supply every intermediate waypoint yourself.

MOTION SEQUENCE
---------------
  1. Joint-space move to "approach" (arm in front of imaginary drawer).
  2. TF2 lookup → read the actual tool0 pose so we know its orientation.
  3. Derive the pull direction = −Z of tool0 in the world frame.
     That direction is perpendicular to the gripper face, pointing away from
     the drawer — the correct axis for pulling.
  4. Build waypoints: positions step along that axis, orientation fixed.
  5. GetCartesianPath service  →  joint trajectory.
  6. ExecuteTrajectory action  →  run the pull.
  7. Reverse the waypoints and execute  →  push back / close the drawer.

SERVICES / ACTIONS
------------------
  move_action               moveit_msgs/action/MoveGroup         (approach)
  /compute_cartesian_path   moveit_msgs/srv/GetCartesianPath     (plan)
  /execute_trajectory       moveit_msgs/action/ExecuteTrajectory (execute)

Run inside the container:
    ros2 run ur_moveit_demo constrained_cartesian
"""

import math
import time

import rclpy
import rclpy.duration
import rclpy.time
from rclpy.action import ActionClient
from rclpy.node import Node

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


# ── robot constants ──────────────────────────────────────────────────────────

PLANNING_GROUP = 'ur_manipulator'
EEF_LINK       = 'tool0'      # UR3e end-effector link
BASE_FRAME     = 'base_link'  # planning / world frame

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

# Joint angles that position tool0 in front of the robot at a comfortable
# height — imagine the gripper just making contact with a drawer handle.
# Adjust these if the approach pose collides with your scene geometry.
APPROACH_JOINTS = [0.00, -1.57, 1.57, -1.57, -1.57, 0.0]

# ── motion parameters ────────────────────────────────────────────────────────

PULL_DISTANCE_M = 0.10   # how far to pull the drawer (metres)

# Cartesian interpolation resolution.  MoveIt calls IK once per MAX_STEP_M
# metres along the path.  Smaller values give smoother trajectories at the cost
# of more IK calls and a longer planning time.
MAX_STEP_M = 0.005

# Reject the plan if fewer than this fraction of waypoints are reachable.
# A value below 1.0 tolerates tiny gaps at the workspace boundary.
MIN_FRACTION = 0.95

SETTLE_SEC   = 0.5   # pause between moves so the state monitor can catch up
MAX_ATTEMPTS = 3


class ConstrainedCartesian(Node):
    def __init__(self):
        super().__init__('constrained_cartesian')

        self._move_client = ActionClient(self, MoveGroup, 'move_action')
        self._exec_client = ActionClient(self, ExecuteTrajectory, 'execute_trajectory')
        self._cartesian_client = self.create_client(
            GetCartesianPath, 'compute_cartesian_path'
        )

        # TF2 buffer + listener so we can read the real tool0 pose after
        # the joint-space move settles, rather than relying on hard-coded FK.
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

    # ── server readiness ─────────────────────────────────────────────────────

    def _wait_for_servers(self):
        self.get_logger().info('Waiting for MoveIt servers …')

        # Use the same server_is_ready() + spin_once() pattern as
        # motion_planner.py to avoid blocking on TF topic discovery in Docker.
        while not self._move_client.server_is_ready():
            rclpy.spin_once(self, timeout_sec=0.5)
        while not self._exec_client.server_is_ready():
            rclpy.spin_once(self, timeout_sec=0.5)
        while not self._cartesian_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().info('  /compute_cartesian_path not ready yet …')

        self.get_logger().info('All servers ready.')

    # ── joint-space move (identical pattern to motion_planner.py) ────────────

    def _move_to_joints(self, label: str, angles: list) -> bool:
        self.get_logger().info(f'\nMoving to "{label}" …')

        request = MotionPlanRequest()
        request.group_name                      = PLANNING_GROUP
        request.allowed_planning_time           = 5.0
        request.num_planning_attempts           = 10
        request.max_velocity_scaling_factor     = 0.5
        request.max_acceleration_scaling_factor = 0.5

        gc = Constraints()
        for name, angle in zip(JOINT_NAMES, angles):
            jc = JointConstraint()
            jc.joint_name      = name
            jc.position        = angle
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            gc.joint_constraints.append(jc)
        request.goal_constraints = [gc]

        options = PlanningOptions()
        options.plan_only = False

        goal_msg = MoveGroup.Goal()
        goal_msg.request          = request
        goal_msg.planning_options = options

        future = self._move_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('Goal rejected by move_group.')
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.error_code.val == 1:
            self.get_logger().info(f'Reached "{label}".')
            return True
        self.get_logger().error(f'move_group error code: {result.error_code.val}')
        return False

    # ── TF lookup for EEF pose ────────────────────────────────────────────────

    def _get_eef_pose(self) -> Pose | None:
        """
        Return the current tool0 pose in the base_link frame via TF2.

        We spin briefly in a loop so the TF broadcaster (which runs in a
        background thread) has time to publish the updated transform after the
        joint-space move completes.
        """
        deadline = self.get_clock().now() + rclpy.duration.Duration(seconds=3.0)
        while self.get_clock().now() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                tf = self._tf_buffer.lookup_transform(
                    BASE_FRAME, EEF_LINK, rclpy.time.Time()
                )
                t = tf.transform.translation
                r = tf.transform.rotation
                pose = Pose()
                pose.position    = Point(x=t.x, y=t.y, z=t.z)
                pose.orientation = Quaternion(x=r.x, y=r.y, z=r.z, w=r.w)
                return pose
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                pass

        self.get_logger().error(
            f'TF lookup {BASE_FRAME} → {EEF_LINK} timed out after 3 s.'
        )
        return None

    # ── Cartesian waypoint builder ────────────────────────────────────────────

    def _build_pull_waypoints(
        self, start: Pose, distance: float, step: float
    ) -> list[Pose]:
        """
        Return evenly-spaced Cartesian waypoints along the pull axis,
        all sharing start's orientation (the orientation lock).

        PULL DIRECTION
        --------------
        tool0's local +Z axis is its approach direction — the axis along which
        the gripper presses into a surface.  Pulling the drawer means moving in
        the opposite direction: −Z of tool0 expressed in the world frame.

        To rotate the local +Z unit vector [0, 0, 1] into the world frame we
        apply the quaternion sandwich product  v_world = q ⊗ [0,0,1] ⊗ q*.
        For a unit quaternion q = (qx, qy, qz, qw) this simplifies to:

            x_world = 2·(qx·qz + qw·qy)
            y_world = 2·(qy·qz − qw·qx)
            z_world = 1 − 2·(qx² + qy²)

        We then negate the result to get the pull direction.
        """
        qx = start.orientation.x
        qy = start.orientation.y
        qz = start.orientation.z
        qw = start.orientation.w

        # tool0 +Z expressed in the world (base_link) frame
        ax = 2.0 * (qx * qz + qw * qy)
        ay = 2.0 * (qy * qz - qw * qx)
        az = 1.0 - 2.0 * (qx * qx + qy * qy)

        # Normalise (already unit if q is unit, but guard against float drift)
        mag = math.sqrt(ax * ax + ay * ay + az * az)
        ax /= mag
        ay /= mag
        az /= mag

        # Pull = move AWAY from the drawer → negate the approach direction
        pull_x, pull_y, pull_z = -ax, -ay, -az

        self.get_logger().info(
            f'Pull direction (world frame): '
            f'({pull_x:.3f}, {pull_y:.3f}, {pull_z:.3f})'
        )

        num_steps = max(2, round(distance / step))
        waypoints: list[Pose] = []
        for i in range(num_steps + 1):
            t = (i / num_steps) * distance
            pose = Pose()
            pose.position = Point(
                x=start.position.x + pull_x * t,
                y=start.position.y + pull_y * t,
                z=start.position.z + pull_z * t,
            )
            # Identical orientation at every step — this is what locks the gripper
            pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
            waypoints.append(pose)
        return waypoints

    # ── Cartesian plan + execute ──────────────────────────────────────────────

    def _execute_cartesian(self, label: str, waypoints: list[Pose]) -> bool:
        """
        Call GetCartesianPath to get a joint trajectory, then execute it.

        GetCartesianPath
        ----------------
        Service type: moveit_msgs/srv/GetCartesianPath

          Request
          ├── group_name        which planning group to use
          ├── link_name         the link that must pass through the waypoints
          ├── waypoints[]       desired Cartesian poses (position + orientation)
          ├── max_step          max Cartesian distance between consecutive IK solves
          ├── jump_threshold    max joint-space jump allowed between consecutive
          │                     IK solutions (0.0 disables the check — safe for
          │                     short, straight paths in a stable configuration)
          └── avoid_collisions  run collision checking at each IK solution

          Response
          ├── solution          the computed RobotTrajectory
          ├── fraction          portion of the path that was reachable (0.0 – 1.0)
          └── error_code        1 = SUCCESS

        fraction < 1.0 means IK failed at some point along the path (e.g. the
        arm cannot reach that part of the line from the current configuration).
        Adjust APPROACH_JOINTS or PULL_DISTANCE_M if fraction is consistently low.

        ExecuteTrajectory
        -----------------
        Action type: moveit_msgs/action/ExecuteTrajectory

          Goal
          └── trajectory   the RobotTrajectory returned by GetCartesianPath

          Result
          └── error_code   1 = SUCCESS
        """
        self.get_logger().info(
            f'\nComputing Cartesian path for "{label}" '
            f'({len(waypoints)} waypoints) …'
        )

        req = GetCartesianPath.Request()
        req.header.frame_id  = BASE_FRAME
        req.header.stamp     = self.get_clock().now().to_msg()
        req.group_name       = PLANNING_GROUP
        req.link_name        = EEF_LINK
        req.waypoints        = waypoints
        req.max_step         = MAX_STEP_M
        req.jump_threshold   = 0.0   # disabled — safe for a short straight pull
        req.avoid_collisions = True

        future = self._cartesian_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        self.get_logger().info(
            f'  fraction: {response.fraction:.1%}   '
            f'error: {response.error_code.val}'
        )

        if response.fraction < MIN_FRACTION:
            self.get_logger().error(
                f'Only {response.fraction:.1%} of "{label}" is reachable — '
                f'adjust APPROACH_JOINTS or PULL_DISTANCE_M and retry.'
            )
            return False

        self.get_logger().info('  Executing trajectory …')
        exec_goal            = ExecuteTrajectory.Goal()
        exec_goal.trajectory = response.solution

        exec_future = self._exec_client.send_goal_async(exec_goal)
        rclpy.spin_until_future_complete(self, exec_future)
        exec_handle = exec_future.result()
        if not exec_handle.accepted:
            self.get_logger().error('ExecuteTrajectory goal rejected.')
            return False

        result_future = exec_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.error_code.val == 1:
            self.get_logger().info(f'"{label}" complete.')
            return True
        self.get_logger().error(f'Execution error code: {result.error_code.val}')
        return False

    # ── main sequence ─────────────────────────────────────────────────────────

    def run(self):
        self._wait_for_servers()

        # 1. Move to the approach configuration with regular joint-space planning.
        if not self._move_to_joints('approach', APPROACH_JOINTS):
            return
        time.sleep(SETTLE_SEC)

        # 2. Read the actual EEF pose so we know the orientation to lock.
        start_pose = self._get_eef_pose()
        if start_pose is None:
            return
        self.get_logger().info(
            f'EEF pose  '
            f'pos=({start_pose.position.x:.3f}, '
            f'{start_pose.position.y:.3f}, '
            f'{start_pose.position.z:.3f})  '
            f'ori=({start_pose.orientation.x:.3f}, '
            f'{start_pose.orientation.y:.3f}, '
            f'{start_pose.orientation.z:.3f}, '
            f'{start_pose.orientation.w:.3f})'
        )

        # 3. Build waypoints: fixed orientation, positions along −Z of tool0.
        pull_waypoints = self._build_pull_waypoints(
            start_pose, PULL_DISTANCE_M, MAX_STEP_M
        )

        # 4. Pull (open the drawer).
        if not self._execute_cartesian('pull', pull_waypoints):
            return
        time.sleep(SETTLE_SEC)

        # 5. Push back (close the drawer) by reversing the same waypoints.
        push_waypoints = list(reversed(pull_waypoints))
        if not self._execute_cartesian('push back', push_waypoints):
            return

        self.get_logger().info('\nDrawer open-and-close demo complete.')


def main(args=None):
    rclpy.init(args=args)
    node = ConstrainedCartesian()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
