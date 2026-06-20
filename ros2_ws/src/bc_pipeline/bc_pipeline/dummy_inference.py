#!/usr/bin/env python3
"""
Dummy Inference — simulate a chunked action-inference loop, no model required.

We don't have a trained policy yet, so the deltas saved in a run's
training_trajectory.npz stand in for what a policy would output.  This node
reproduces the *shape* of a real inference deployment:

    observe → emit a chunk of actions → execute → "think" → repeat

THE LOOP (per chunk)
--------------------
  1. Observe : read a fresh /joint_states; reorder by name to the npz layout.
  2. Accumulate the chunk's position deltas onto that measured seed to get
     absolute waypoints (deltas are relative; the controller needs absolute).
  3. Execute : send the waypoints as ONE FollowJointTrajectory goal to the
     scaled_joint_trajectory_controller — no MoveIt, no collision checking.
     Waypoint k is timed at (k+1)*dt, dt = 1/rate_hz (0.1 s for 10 Hz data),
     so a 10-step chunk runs in ~1.0 s.
  4. Think : pause pause_sec seconds (the arm rests — intended stop-and-go,
     representing inference latency between action chunks).

Re-observing each chunk mirrors closed-loop action chunking: the policy always
acts relative to where the arm actually is now.

PARAMS
------
  run        : path to a run directory or directly to a .npz file
  chunk_size : actions per inference step (default 10)
  pause_sec  : simulated "model thinking" pause between chunks (default 2.0)

USAGE
-----
    ros2 run bc_pipeline dummy_inference \\
        --ros-args -p run:=/root/ros2_ws/runs/drawer_2026-06-20_14-14-03
"""

import math
import os
import sys

import numpy as np

import rclpy
import rclpy.duration
from rclpy.action import ActionClient
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ACTION_NAME = '/scaled_joint_trajectory_controller/follow_joint_trajectory'
NPZ_NAME = 'training_trajectory.npz'

# Time allowed to drive from wherever the arm is to the recorded start pose.
# Generous because that move can be large; the chunks themselves run at dt.
START_MOVE_SEC = 5.0


def resolve_npz(run_path: str) -> str:
    """Accept a run directory or a .npz file; return the .npz path."""
    if not run_path:
        raise SystemExit("Set the 'run' parameter to a run directory or .npz file.")
    candidate = os.path.join(run_path, NPZ_NAME) if os.path.isdir(run_path) else run_path
    if not os.path.isfile(candidate):
        raise SystemExit(f"Trajectory file not found: {candidate}")
    return candidate


class DummyInference(Node):
    def __init__(self):
        super().__init__('dummy_inference')

        self.declare_parameter('run', '')
        self.declare_parameter('chunk_size', 10)
        self.declare_parameter('pause_sec', 2.0)

        self.chunk_size = self.get_parameter('chunk_size').value
        self.pause_sec = self.get_parameter('pause_sec').value

        npz_path = resolve_npz(self.get_parameter('run').value)
        data = np.load(npz_path)
        self.deltas = data['deltas']                       # (N-1, J)
        self.start_pose = data['positions'][0]             # first recorded sample
        self.joint_names = [str(n) for n in data['joint_names']]
        rate_hz = float(data['rate_hz'])
        self.dt = 1.0 / rate_hz
        self.get_logger().info(
            f"Loaded {npz_path}: {len(self.deltas)} deltas, "
            f"{len(self.joint_names)} joints @ {rate_hz:.0f} Hz (dt={self.dt:.3f}s)."
        )

        self._client = ActionClient(self, FollowJointTrajectory, ACTION_NAME)

        # Latest /joint_states sample, refreshed each observe().
        self._latest_js = None
        self.create_subscription(JointState, '/joint_states', self._on_joint_states, 10)

    # ── observation ─────────────────────────────────────────────────────────────

    def _on_joint_states(self, msg: JointState):
        self._latest_js = msg

    def observe(self) -> np.ndarray | None:
        """Read a fresh /joint_states and return positions in npz joint order."""
        self._latest_js = None   # force a genuinely new sample
        deadline = self.get_clock().now() + rclpy.duration.Duration(seconds=5.0)
        while self._latest_js is None and self.get_clock().now() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self._latest_js is None:
            self.get_logger().error('Timed out waiting for /joint_states.')
            return None

        name_to_pos = dict(zip(self._latest_js.name, self._latest_js.position))
        if not all(n in name_to_pos for n in self.joint_names):
            self.get_logger().error(
                f'/joint_states is missing one of {self.joint_names}.'
            )
            return None
        return np.array([name_to_pos[n] for n in self.joint_names])

    # ── execution ───────────────────────────────────────────────────────────────

    def move_to_start(self) -> bool:
        """Drive the arm to the recorded start pose (positions[0]) before inference.

        Without this, accumulating deltas from the current pose would trace the
        recorded SHAPE but from the wrong origin.  Seeding the arm at the first
        sample makes the reconstructed trajectory match the recording.
        """
        self.get_logger().info(
            f'Moving to recorded start pose over {START_MOVE_SEC:.1f}s …'
        )
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in self.start_pose]
        point.time_from_start = rclpy.duration.Duration(seconds=START_MOVE_SEC).to_msg()
        traj.points = [point]
        return self._send_trajectory(traj)

    def execute_chunk(self, seed: np.ndarray, chunk_deltas: np.ndarray) -> bool:
        """Accumulate deltas onto seed and send them as one trajectory goal."""
        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        pos = seed.copy()
        for k, delta in enumerate(chunk_deltas):
            pos = pos + delta
            point = JointTrajectoryPoint()
            point.positions = [float(x) for x in pos]
            t = (k + 1) * self.dt
            point.time_from_start = rclpy.duration.Duration(seconds=t).to_msg()
            traj.points.append(point)

        return self._send_trajectory(traj)

    def _send_trajectory(self, traj: JointTrajectory) -> bool:
        """Send one FollowJointTrajectory goal and wait for the result."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        handle = send_future.result()
        if not handle.accepted:
            self.get_logger().error('Trajectory goal rejected by the controller.')
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.error_code == 0:
            return True
        self.get_logger().error(
            f'Trajectory execution failed: error_code={result.error_code} '
            f'({result.error_string})'
        )
        return False

    # ── main loop ───────────────────────────────────────────────────────────────

    def _wait_for_server(self):
        self.get_logger().info(f'Waiting for {ACTION_NAME} …')
        while not self._client.server_is_ready():
            rclpy.spin_once(self, timeout_sec=0.5)
        self.get_logger().info('Controller ready.')

    def _sleep(self, seconds: float):
        end = self.get_clock().now() + rclpy.duration.Duration(seconds=seconds)
        while self.get_clock().now() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def run(self) -> int:
        self._wait_for_server()

        # Start at the recorded first sample so the delta replay is faithful.
        if not self.move_to_start():
            return 1

        n = len(self.deltas)
        num_chunks = math.ceil(n / self.chunk_size)
        for ci, start in enumerate(range(0, n, self.chunk_size), 1):
            chunk = self.deltas[start:start + self.chunk_size]
            self.get_logger().info(
                f'\n{"=" * 52}\n[chunk {ci}/{num_chunks}] '
                f'observe → {len(chunk)} actions'
            )

            seed = self.observe()
            if seed is None:
                return 1
            if not self.execute_chunk(seed, chunk):
                return 1

            self.get_logger().info(f'Thinking ({self.pause_sec:.1f}s) …')
            self._sleep(self.pause_sec)

        self.get_logger().info(f'\nInference complete: {num_chunks} chunk(s) executed.')
        return 0


def main(args=None):
    rclpy.init(args=args)
    node = DummyInference()
    code = 1
    try:
        code = node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
