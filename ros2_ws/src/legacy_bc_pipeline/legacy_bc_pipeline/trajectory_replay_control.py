#!/usr/bin/env python3
"""Replay actions or joint references from a legacy behavior-cloning dataset.

The node intentionally does not move on startup.  Call
``/start_trajectory_replay`` after checking the parameters and robot workspace.
By default ``dry_run`` is true, so the complete replay is only logged.
"""

import csv
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rclpy
import torch
from ecpmi_gripper.srv import GripperControl
from pymoveit2 import MoveIt2
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger


class TrajectoryReplayControl(Node):
    """Safely replay a stored trajectory through the legacy MoveIt interface."""

    def __init__(self) -> None:
        super().__init__("trajectory_replay_control")

        self.declare_parameter("trajectory_path", "")
        self.declare_parameter("replay_mode", "actions")
        self.declare_parameter("dry_run", True)
        self.declare_parameter("chunk_size", 10)
        self.declare_parameter("action_scale", 150.0)
        self.declare_parameter("start_index", 0)
        self.declare_parameter("stop_index", -1)
        self.declare_parameter("move_to_recorded_start", True)
        self.declare_parameter("initial_pose_tolerance_rad", 0.10)
        self.declare_parameter("tracking_error_limit_rad", 0.05)
        self.declare_parameter("max_chunk_delta_rad", 0.20)
        self.declare_parameter("goal_tolerance_rad", 0.001)
        self.declare_parameter("velocity_scaling", 0.15)
        self.declare_parameter("acceleration_scaling", 0.15)
        self.declare_parameter("replay_gripper", False)
        self.declare_parameter("csv_log_path", "")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("gripper_service", "/gripper_control")
        self.declare_parameter(
            "joint_names",
            [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
        )
        self.declare_parameter("joint_lower_limits_rad", [-6.283] * 6)
        self.declare_parameter("joint_upper_limits_rad", [6.283] * 6)

        self._trajectory_path = str(self.get_parameter("trajectory_path").value)
        self._mode = str(self.get_parameter("replay_mode").value)
        self._dry_run = bool(self.get_parameter("dry_run").value)
        self._chunk_size = int(self.get_parameter("chunk_size").value)
        self._action_scale = float(self.get_parameter("action_scale").value)
        self._start_index = int(self.get_parameter("start_index").value)
        self._requested_stop_index = int(self.get_parameter("stop_index").value)
        self._move_to_start = bool(
            self.get_parameter("move_to_recorded_start").value
        )
        self._initial_tolerance = float(
            self.get_parameter("initial_pose_tolerance_rad").value
        )
        self._tracking_limit = float(
            self.get_parameter("tracking_error_limit_rad").value
        )
        self._max_chunk_delta = float(
            self.get_parameter("max_chunk_delta_rad").value
        )
        self._goal_tolerance = float(
            self.get_parameter("goal_tolerance_rad").value
        )
        self._joint_names = [
            str(name) for name in self.get_parameter("joint_names").value
        ]
        self._lower_limits = np.asarray(
            self.get_parameter("joint_lower_limits_rad").value, dtype=np.float64
        )
        self._upper_limits = np.asarray(
            self.get_parameter("joint_upper_limits_rad").value, dtype=np.float64
        )
        self._replay_gripper = bool(
            self.get_parameter("replay_gripper").value
        )
        self._csv_log_path = str(self.get_parameter("csv_log_path").value)

        self._validate_parameters()
        self._actions, self._recorded_joints = self._load_trajectory()
        self._stop_index = (
            len(self._actions)
            if self._requested_stop_index < 0
            else min(self._requested_stop_index, len(self._actions))
        )
        if self._start_index >= self._stop_index:
            raise ValueError(
                f"Invalid replay range [{self._start_index}, {self._stop_index})"
            )

        self._latest_joints: Optional[np.ndarray] = None
        self._name_to_index: Optional[Dict[str, int]] = None
        self._state_lock = threading.Lock()
        self._replay_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._previous_gripper_state: Optional[bool] = None

        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_states_topic").value),
            self._on_joint_state,
            10,
        )
        self._start_service = self.create_service(
            Trigger, "/start_trajectory_replay", self._start_replay
        )
        self._stop_service = self.create_service(
            Trigger, "/stop_trajectory_replay", self._stop_replay
        )

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self._joint_names,
            base_link_name="base_link",
            end_effector_name="tool0",
            group_name="ur_manipulator",
            use_move_group_action=True,
        )
        self.moveit2.max_velocity = float(
            self.get_parameter("velocity_scaling").value
        )
        self.moveit2.max_acceleration = float(
            self.get_parameter("acceleration_scaling").value
        )
        self._gripper_client = self.create_client(
            GripperControl, str(self.get_parameter("gripper_service").value)
        )

        self.get_logger().info(
            f"Loaded {len(self._actions)} samples from {self._trajectory_path}; "
            f"mode={self._mode}, range=[{self._start_index}, "
            f"{self._stop_index}), dry_run={self._dry_run}."
        )
        self.get_logger().warn(
            "Replay is idle. Inspect the workspace, then call "
            "/start_trajectory_replay. Set dry_run:=false to permit motion."
        )

    def _validate_parameters(self) -> None:
        if not self._trajectory_path:
            raise ValueError("trajectory_path must name a legacy .pt file")
        if self._mode not in {"actions", "recorded_positions"}:
            raise ValueError(
                "replay_mode must be 'actions' or 'recorded_positions'"
            )
        if self._chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self._action_scale <= 0.0:
            raise ValueError("action_scale must be positive")
        if self._start_index < 0:
            raise ValueError("start_index cannot be negative")
        if len(self._joint_names) != 6:
            raise ValueError("joint_names must contain exactly six names")
        if self._lower_limits.shape != (6,) or self._upper_limits.shape != (6,):
            raise ValueError("joint limit parameters must each contain six values")
        if np.any(self._lower_limits >= self._upper_limits):
            raise ValueError("each lower joint limit must be below its upper limit")

    def _load_trajectory(self):
        path = Path(self._trajectory_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Trajectory does not exist: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict) or "actions" not in data:
            raise ValueError("Trajectory must be a dictionary containing 'actions'")
        if "obs" not in data or "joints" not in data["obs"]:
            raise ValueError("Trajectory must contain 'obs/joints'")

        actions = np.asarray(data["actions"], dtype=np.float64)
        joints = np.asarray(data["obs"]["joints"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(
                f"Expected actions with shape (N, 7), got {actions.shape}"
            )
        if joints.ndim != 2 or joints.shape[1] != 6:
            raise ValueError(
                f"Expected obs/joints with shape (N, 6), got {joints.shape}"
            )
        if len(actions) != len(joints):
            raise ValueError("actions and obs/joints must have equal lengths")
        if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(joints)):
            raise ValueError("Trajectory actions and joints must be finite")
        return actions, joints

    def _on_joint_state(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return
        name_to_index = {name: i for i, name in enumerate(msg.name)}
        if not all(name in name_to_index for name in self._joint_names):
            return
        joints = np.asarray(
            [msg.position[name_to_index[name]] for name in self._joint_names],
            dtype=np.float64,
        )
        with self._state_lock:
            self._name_to_index = name_to_index
            self._latest_joints = joints

    def _measured_joints(self) -> Optional[np.ndarray]:
        with self._state_lock:
            if self._latest_joints is None:
                return None
            return self._latest_joints.copy()

    def _start_replay(self, _request, response):
        if self._replay_thread is not None and self._replay_thread.is_alive():
            response.success = False
            response.message = "Replay is already running."
            return response
        if not self._dry_run and self._measured_joints() is None:
            response.success = False
            response.message = "No complete joint-state message has been received."
            return response

        self._stop_event.clear()
        self._replay_thread = threading.Thread(
            target=self._run_replay, name="trajectory-replay", daemon=True
        )
        self._replay_thread.start()
        response.success = True
        response.message = (
            "Dry-run started; no commands will be sent."
            if self._dry_run
            else "Trajectory replay started."
        )
        return response

    def _stop_replay(self, _request, response):
        self._stop_event.set()
        if not self._dry_run:
            self.moveit2.cancel_execution()
        response.success = True
        response.message = "Stop requested."
        return response

    def _run_replay(self) -> None:
        log_file = None
        writer = None
        try:
            if self._csv_log_path:
                log_path = Path(self._csv_log_path).expanduser()
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = log_path.open("w", newline="", encoding="utf-8")
                writer = csv.writer(log_file)
                writer.writerow(
                    [
                        "start_index",
                        "end_index",
                        "mode",
                        *[f"target_{name}" for name in self._joint_names],
                        *[f"measured_{name}" for name in self._joint_names],
                        *[f"tracking_error_{name}" for name in self._joint_names],
                        "max_abs_tracking_error",
                        "max_abs_reference_error",
                    ]
                )

            if not self._prepare_start_pose():
                return

            dry_run_joints = self._recorded_joints[self._start_index].copy()
            for start in range(
                self._start_index, self._stop_index, self._chunk_size
            ):
                if self._stop_event.is_set():
                    self.get_logger().warn("Replay stopped by request.")
                    return
                end = min(start + self._chunk_size, self._stop_index)
                chunk = self._actions[start:end]
                measured_before = (
                    dry_run_joints.copy()
                    if self._dry_run
                    else self._measured_joints()
                )
                if measured_before is None:
                    self._abort("No joint state is available before execution.")
                    return

                integrated_target = measured_before + np.sum(
                    chunk[:, :6] / self._action_scale, axis=0
                )
                reference_index = min(end, len(self._recorded_joints) - 1)
                recorded_target = self._recorded_joints[reference_index]
                target = (
                    integrated_target
                    if self._mode == "actions"
                    else recorded_target.copy()
                )
                commanded_delta = target - measured_before
                reference_error = integrated_target - recorded_target

                if np.max(np.abs(commanded_delta)) > self._max_chunk_delta:
                    self._abort(
                        f"Chunk [{start}, {end}) requests delta "
                        f"{commanded_delta}; limit is "
                        f"{self._max_chunk_delta:.4f} rad."
                    )
                    return
                if np.any(target < self._lower_limits) or np.any(
                    target > self._upper_limits
                ):
                    self._abort(
                        f"Chunk [{start}, {end}) target violates joint limits: "
                        f"{target}"
                    )
                    return

                if self._dry_run:
                    measured_after = target.copy()
                    dry_run_joints = target.copy()
                else:
                    self.moveit2.move_to_configuration(
                        target.tolist(),
                        self._joint_names,
                        tolerance=self._goal_tolerance,
                    )
                    if not self.moveit2.wait_until_executed():
                        self._abort(
                            f"MoveIt failed for chunk [{start}, {end})."
                        )
                        return
                    measured_after = self._measured_joints()
                    if measured_after is None:
                        self._abort("Joint state disappeared after execution.")
                        return

                tracking_error = measured_after - target
                max_tracking_error = float(np.max(np.abs(tracking_error)))
                max_reference_error = float(np.max(np.abs(reference_error)))
                self.get_logger().info(
                    f"chunk=[{start},{end}) target={target} "
                    f"measured={measured_after} tracking_error={tracking_error} "
                    f"integrated_vs_recorded_max={max_reference_error:.6f}"
                )
                if writer is not None:
                    writer.writerow(
                        [
                            start,
                            end,
                            self._mode,
                            *target,
                            *measured_after,
                            *tracking_error,
                            max_tracking_error,
                            max_reference_error,
                        ]
                    )
                    log_file.flush()

                if (
                    not self._dry_run
                    and max_tracking_error > self._tracking_limit
                ):
                    self._abort(
                        f"Tracking error {max_tracking_error:.4f} rad exceeds "
                        f"limit {self._tracking_limit:.4f} rad."
                    )
                    return
                if self._replay_gripper and not self._dry_run:
                    self._set_gripper_state(bool(chunk[-1, 6] > 0.5))

            self.get_logger().info("Trajectory replay completed.")
        except Exception as exc:  # noqa: BLE001
            self._abort(f"Replay failed: {exc}")
        finally:
            if log_file is not None:
                log_file.close()

    def _prepare_start_pose(self) -> bool:
        recorded_start = self._recorded_joints[self._start_index]
        measured = self._measured_joints()
        if self._dry_run:
            self.get_logger().info(
                f"Dry-run start pose would be {recorded_start}."
            )
            return True
        if measured is None:
            self._abort("No joint state is available.")
            return False

        initial_error = float(np.max(np.abs(measured - recorded_start)))
        if initial_error <= self._initial_tolerance:
            return True
        if not self._move_to_start:
            self._abort(
                f"Robot is {initial_error:.4f} rad from the recorded start; "
                f"limit is {self._initial_tolerance:.4f} rad."
            )
            return False

        self.get_logger().warn(
            f"Moving to recorded start pose; current maximum difference is "
            f"{initial_error:.4f} rad."
        )
        self.moveit2.move_to_configuration(
            recorded_start.tolist(),
            self._joint_names,
            tolerance=self._goal_tolerance,
        )
        if not self.moveit2.wait_until_executed():
            self._abort("MoveIt failed while moving to the recorded start pose.")
            return False
        measured = self._measured_joints()
        if measured is None:
            self._abort("No joint state after moving to the start pose.")
            return False
        remaining_error = float(np.max(np.abs(measured - recorded_start)))
        if remaining_error > self._tracking_limit:
            self._abort(
                f"Start-pose error {remaining_error:.4f} rad exceeds tracking "
                f"limit {self._tracking_limit:.4f} rad."
            )
            return False
        return True

    def _set_gripper_state(self, desired: bool) -> None:
        if desired == self._previous_gripper_state:
            return
        if not self._gripper_client.wait_for_service(timeout_sec=0.5):
            raise RuntimeError("Gripper service is unavailable")
        request = GripperControl.Request()
        request.command = "grip" if desired else "blow"
        self._gripper_client.call_async(request)
        if desired:
            # Match the legacy controller's suction sequence: energize the
            # vacuum briefly, then release the valve while retaining suction.
            time.sleep(1.0)
            release = GripperControl.Request()
            release.command = "release"
            self._gripper_client.call_async(release)
        self._previous_gripper_state = desired

    def _abort(self, message: str) -> None:
        self._stop_event.set()
        if not self._dry_run:
            self.moveit2.cancel_execution()
        self.get_logger().error(message)

    def destroy_node(self):
        self._stop_event.set()
        if not self._dry_run:
            self.moveit2.cancel_execution()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = TrajectoryReplayControl()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
