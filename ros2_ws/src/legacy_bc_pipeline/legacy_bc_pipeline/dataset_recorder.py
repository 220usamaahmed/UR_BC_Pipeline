"""Record synchronized robot state, commands, and depth images to NumPy files."""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import rclpy
from control_msgs.msg import JointJog
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Int32, UInt8
from std_srvs.srv import Trigger


RED = "\033[31m"
RESET = "\033[0m"


@dataclass
class LatestMsg:
    stamp_sec: float
    msg: object


class DatasetRecorder(Node):
    def __init__(self) -> None:
        super().__init__("dataset_recorder")

        self.declare_parameter("sample_rate_hz", 15.0)
        self.declare_parameter("sync_tolerance_sec", 0.3)
        self.declare_parameter("output_dir", "/data/external/incorrect-data")
        self.declare_parameter("stop_output_dir", "")
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
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("joint_command_topic", "/servo_node/delta_joint_cmds_raw")
        self.declare_parameter("gripper_state_topic", "/gripper_state")
        self.declare_parameter("depth_topic", "/zed/zed_node/depth/depth_registered")
        self.declare_parameter("start_service", "/dataset_recorder/start")
        self.declare_parameter("stop_service", "/dataset_recorder/stop")

        self._sample_rate_hz = float(self.get_parameter("sample_rate_hz").value)
        self._sync_tolerance_sec = float(self.get_parameter("sync_tolerance_sec").value)
        self._output_dir = str(self.get_parameter("output_dir").value)
        self._joint_names = list(self.get_parameter("joint_names").value)
        self._joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self._joint_command_topic = str(self.get_parameter("joint_command_topic").value)
        self._gripper_state_topic = str(self.get_parameter("gripper_state_topic").value)
        self._depth_topic = str(self.get_parameter("depth_topic").value)
        self._start_service = str(self.get_parameter("start_service").value)
        self._stop_service = str(self.get_parameter("stop_service").value)

        self._latest_joint_state: Optional[LatestMsg] = None
        self._latest_joint_cmd: Optional[LatestMsg] = None
        self._latest_gripper_state: Optional[LatestMsg] = None
        self._latest_depth: Optional[LatestMsg] = None
        self._latest_segment_id = 0
        self._name_to_index: Optional[Dict[str, int]] = None

        self._recording = False
        self._session_dir: Optional[str] = None
        self._observations: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []
        self._timestamps: List[float] = []
        self._depth_frames: List[np.ndarray] = []

        self.create_subscription(JointState, self._joint_states_topic, self._on_joint_state, 10)
        self.create_subscription(JointJog, self._joint_command_topic, self._on_joint_command, 10)
        self.create_subscription(UInt8, self._gripper_state_topic, self._on_gripper_state, 10)
        self.create_subscription(Int32, "/current_segment", self._on_segment, 10)
        self.create_subscription(Image, self._depth_topic, self._on_depth, 10)
        self._start_srv = self.create_service(Trigger, self._start_service, self._on_start)
        self._stop_srv = self.create_service(Trigger, self._stop_service, self._on_stop)

        period = 1.0 / self._sample_rate_hz if self._sample_rate_hz > 0 else 0.0333
        self._timer = self.create_timer(period, self._sample)

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _on_joint_state(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return
        if self._name_to_index is None:
            self._name_to_index = {name: i for i, name in enumerate(msg.name)}
        self._latest_joint_state = LatestMsg(self._stamp_to_sec(msg.header.stamp), msg)

    def _on_joint_command(self, msg: JointJog) -> None:
        self._latest_joint_cmd = LatestMsg(self._stamp_to_sec(msg.header.stamp), msg)

    def _on_gripper_state(self, msg: UInt8) -> None:
        self._latest_gripper_state = LatestMsg(self._now_sec(), msg)

    def _on_segment(self, msg: Int32) -> None:
        self._latest_segment_id = msg.data

    def _on_depth(self, msg: Image) -> None:
        self._latest_depth = LatestMsg(self._stamp_to_sec(msg.header.stamp), msg)

    def _on_start(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self._recording:
            response.success = True
            response.message = "Already recording."
            return response

        self._observations.clear()
        self._actions.clear()
        self._timestamps.clear()
        self._depth_frames.clear()
        self._recording = True
        response.success = True
        response.message = "Recording..."
        self.get_logger().info(response.message)
        return response

    def _on_stop(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if not self._recording:
            response.success = True
            response.message = "Not recording."
            return response

        self._recording = False
        observations = (
            np.stack(self._observations) if self._observations else np.zeros((0, 10))
        )
        actions = np.stack(self._actions) if self._actions else np.zeros((0, 9))
        timestamps = np.asarray(self._timestamps, dtype=np.float64)
        depth_frames = (
            np.stack(self._depth_frames) if self._depth_frames else np.zeros((0, 0, 0))
        )

        stop_output_dir = str(self.get_parameter("stop_output_dir").value).strip()
        index_in_folder = 0
        while True:
            session_dir = os.path.join(
                self._output_dir, f"{stop_output_dir}_{index_in_folder:04d}"
            )
            if not os.path.exists(session_dir):
                break
            index_in_folder += 1

        self._session_dir = session_dir
        os.makedirs(self._session_dir, exist_ok=True)
        output_path = os.path.join(self._session_dir, "dataset.npz")
        np.savez(
            output_path,
            observations=observations,
            actions=actions,
            timestamps=timestamps,
            depth_frames=depth_frames,
        )

        self._depth_frames.clear()
        self._observations.clear()
        self._actions.clear()
        self._timestamps.clear()
        response.success = True
        response.message = f"Saved dataset to {output_path}"
        self.get_logger().info(response.message)
        return response

    def _sample(self) -> None:
        if not self._recording:
            return

        now_sec = self._now_sec()
        skip_reason = self._input_skip_reason(now_sec)
        if skip_reason is not None:
            self._print_skipped_sample(skip_reason)
            return

        joint_state = self._latest_joint_state.msg
        joint_cmd = self._latest_joint_cmd.msg
        gripper_state = self._latest_gripper_state.msg
        depth = self._latest_depth.msg
        observation = self._build_observation(
            joint_state, gripper_state, self._latest_segment_id
        )
        action = self._build_action(joint_cmd, gripper_state)
        try:
            depth_array = self._image_to_array(depth)
        except ValueError as exc:
            self._print_skipped_sample(f"{self._depth_topic}: {exc}")
            return

        self._observations.append(observation)
        self._actions.append(action)
        self._timestamps.append(now_sec)
        self._depth_frames.append(depth_array)

    def _input_skip_reason(self, now_sec: float) -> Optional[str]:
        inputs = (
            (self._joint_states_topic, self._latest_joint_state),
            (self._joint_command_topic, self._latest_joint_cmd),
            (self._gripper_state_topic, self._latest_gripper_state),
            (self._depth_topic, self._latest_depth),
        )
        missing_topics = [topic for topic, latest in inputs if latest is None]
        if self._name_to_index is None and self._joint_states_topic not in missing_topics:
            missing_topics.append(f"{self._joint_states_topic} (joint-name mapping)")
        if missing_topics:
            return "missing input: " + ", ".join(missing_topics)

        stale_topics = []
        for topic, latest in (
            (self._joint_states_topic, self._latest_joint_state),
            (self._joint_command_topic, self._latest_joint_cmd),
            (self._depth_topic, self._latest_depth),
        ):
            age_sec = abs(now_sec - latest.stamp_sec)
            if age_sec > self._sync_tolerance_sec:
                stale_topics.append(f"{topic} ({age_sec:.3f} s old)")
        if stale_topics:
            return "stale input: " + ", ".join(stale_topics)
        return None

    @staticmethod
    def _print_skipped_sample(reason: str) -> None:
        print(f"{RED}Skipping sample: {reason}{RESET}")

    def _build_observation(
        self, joint_state: JointState, gripper_state: UInt8, _segment_id: int
    ) -> np.ndarray:
        joints = self._extract_joint_vector(joint_state)
        gripper_onehot = self._gripper_onehot(gripper_state.data)
        return np.concatenate([joints, [0.0]]) # For recording failed trajectories

    def _build_action(self, joint_cmd: JointJog, gripper_state: UInt8) -> np.ndarray:
        velocities = self._extract_velocity_vector(joint_cmd)
        gripper_onehot = self._gripper_onehot(gripper_state.data)
        return np.concatenate([velocities, gripper_onehot])

    def _extract_joint_vector(self, joint_state: JointState) -> np.ndarray:
        positions = np.zeros(len(self._joint_names), dtype=np.float32)
        for i, name in enumerate(self._joint_names):
            index = self._name_to_index.get(name)
            if index is not None and index < len(joint_state.position):
                positions[i] = joint_state.position[index]
        return positions

    def _extract_velocity_vector(self, joint_cmd: JointJog) -> np.ndarray:
        velocities = np.zeros(len(self._joint_names), dtype=np.float32)
        name_to_index = {name: i for i, name in enumerate(joint_cmd.joint_names)}
        for i, name in enumerate(self._joint_names):
            index = name_to_index.get(name)
            if index is not None and index < len(joint_cmd.velocities):
                velocities[i] = joint_cmd.velocities[index]
        return velocities

    @staticmethod
    def _gripper_onehot(state: int) -> np.ndarray:
        onehot = np.zeros(3, dtype=np.float32)
        if state == 1:
            onehot[0] = 1.0
        elif state == 2:
            onehot[1] = 1.0
        elif state == 3:
            onehot[2] = 1.0
        return onehot

    def _image_to_array(self, msg: Image) -> np.ndarray:
        if msg.encoding == "32FC1":
            dtype = np.float32
        elif msg.encoding == "16UC1":
            dtype = np.uint16
        else:
            raise ValueError(f"unsupported depth encoding {msg.encoding!r}")

        expected_len = msg.height * msg.width * np.dtype(dtype).itemsize
        if len(msg.data) < expected_len:
            raise ValueError(
                f"depth buffer has {len(msg.data)} bytes; expected at least {expected_len}"
            )
        array = np.frombuffer(msg.data, dtype=dtype, count=msg.height * msg.width)
        return array.reshape((msg.height, msg.width))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DatasetRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
