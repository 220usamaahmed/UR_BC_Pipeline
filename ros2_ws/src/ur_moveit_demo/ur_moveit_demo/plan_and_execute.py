import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions
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

SETTLE_SEC = 0.5
MAX_ATTEMPTS = 3


class PlanAndExecute(Node):
    def __init__(self):
        super().__init__('sequence_runner')
        self._plan_client = ActionClient(self, MoveGroup, 'move_action')
        self._exec_client = ActionClient(self, ExecuteTrajectory, 'execute_trajectory')

    def run(self):
        self.get_logger().info('Waiting for /move_group action server...')
        while not self._plan_client.server_is_ready():
            rclpy.spin_once(self, timeout_sec=0.5)
        self.get_logger().info('Connected to move_group.')

        for name in WAYPOINT_ORDER:
            angles = WAYPOINTS[name]
            self.get_logger().info(f'\n{"=" * 52}')
            self.get_logger().info(f"Waypoint : {name}")
            self.get_logger().info(f"Angles : {angles}")

            if not self._move_to_with_retries(name, angles):
                self.get_logger().error(f'Stopping after fairure at "{name}".')
                return

            time.sleep(SETTLE_SEC)

        self.get_logger().info('All waypoints reached.')

    def _move_to_with_retries(self, name: str, angles: list) -> bool:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            trajectory = self._plan_to(name, angles)
            if trajectory is None:
                self.get_logger().error(f'Could not plan to "{name}".')
            else:
                self._print_plan(trajectory)
                input('\nPress Enter to execute (Ctrl-C to abort)...')
                if self._execute(trajectory):
                    return True
                self.get_logger().error(f'Execution of "{name}" failed.')

            if attempt < MAX_ATTEMPTS:
                self.get_logger().warn(
                    f'"{name}" failed (attempt {attempt}/{MAX_ATTEMPTS}); '
                    f'letting state settle and retrying ...'
                )
                time.sleep(SETTLE_SEC)
        return False

    def _plan_to(self, name, angles):
        request = MotionPlanRequest()
        request.group_name = PLANNING_GROUP
        request.allowed_planning_time = 5.0
        request.num_planning_attempts = 10
        request.max_velocity_scaling_factor = 0.5
        request.max_acceleration_scaling_factor = 0.5

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

        options = PlanningOptions()
        options.plan_only = True

        goal_msg = MoveGroup.Goal()
        goal_msg.request = request
        goal_msg.planning_options = options

        goal_future = self._plan_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, goal_future)
        goal_handle = goal_future.result()
        if not goal_handle.accepted:
            return None

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.error_code.val != 1:
            self.get_logger().error(f'Plan failed: {result.error_code.val}')
            return None

        return result.planned_trajectory

    def _print_plan(self, trajectory):
        jt = trajectory.joint_trajectory
        self.get_logger().info(f'Joints: {jt.joint_names}')
        self.get_logger().info(f'Points: {len(jt.points)}')
        for i, pt in enumerate(jt.points):
            t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            angles = [f'{p:+.3f}' for p in pt.positions]
            self.get_logger().info(f'  [{i:02d}] t={t:5.2f}s  {angles}')

    def _execute(self, trajectory):
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory          # the RobotTrajectory from the plan
        goal_future = self._exec_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_future)
        goal_handle = goal_future.result()
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        code = result.error_code.val
        self.get_logger().info(f'Execute error code: {code}')

        return result_future.result().result.error_code.val == 1


def main(args=None):
    rclpy.init(args=args)
    node = PlanAndExecute()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
