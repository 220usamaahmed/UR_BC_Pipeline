import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from moveit_msgs.action import MoveGroup
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


class SequenceRunner(Node):
    def __init__(self):
        super().__init__('sequence_runner')
        self._client = ActionClient(self, MoveGroup, 'move_action')

    def run(self):
        self.get_logger().info('Waiting for /move_group action server...')
        while not self._client.server_is_ready():
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
        options.plan_only = False

        goal_msg = MoveGroup.Goal()
        goal_msg.request = request
        goal_msg.planning_options = options

        self.get_logger().info('Sending to /move_group ...')
        goal_future = self._client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, goal_future)

        goal_handle = goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by move_group.')
            return False

        self.get_logger().info('Planning and executing ...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result

        if result.error_code.val == 1:
            pts = len(result.planned_trajectory.joint_trajectory.points)
            self.get_logger().info(
                f'Done. Trajectory: {pts} points, '
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
    node = SequenceRunner()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
