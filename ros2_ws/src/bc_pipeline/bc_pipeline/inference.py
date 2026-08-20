#!/usr/bin/env python3
"""Run a behaviour-cloning policy from synchronized depth and joint history."""

from collections import deque
import sys
import traceback

import numpy as np
import rclpy
import rclpy.duration
import torch
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from bc_pipeline.inference_debug import (
    EVENT_TOPIC,
    MATCHED_TARGET_TOPIC,
    ModelObservation,
    PHASE_TOPIC,
    PLANNED_TARGET_TOPIC,
    PREPROCESS_VERSION,
    SELECTED_DEPTH_TOPIC,
    SELECTED_JOINT_TOPIC,
    ExecutionCandidate,
    depth_sha256,
    dumps_event,
    event_message,
    integrate_joint_deltas,
    interpolate_joint_positions,
    selected_observation_id,
)
from bc_pipeline.model import ConditionalDiffusionModel
from control_msgs.action import FollowJointTrajectory
from ecpmi_gripper.srv import GripperControl
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ACTION_NAME = '/scaled_joint_trajectory_controller/follow_joint_trajectory'
DEPTH_TOPIC = '/zed/zed_node/depth/depth_registered'
GRIPPER_SERVICE = 'gripper_control'
DEFAULT_JOINT_NAMES = [
    'shoulder_pan_joint', 'wrist_2_joint', 'wrist_3_joint',
    'wrist_1_joint', 'elbow_joint', 'shoulder_lift_joint',
]
# drawer_demo.yaml defines home as [-90, 0, -90, 0, 90, 0] degrees
# in [lift, elbow, wrist_1, wrist_2, wrist_3, pan] order. Remapped here
# into DEFAULT_JOINT_NAMES order.
DEFAULT_HOME_POSITION = np.deg2rad(
    [0.0, 0.0, 90.0, -90.0, 0.0, -90.0]
).tolist()


def stamp_nanoseconds(msg) -> int:
    """Return a ROS header stamp without losing nanosecond precision."""
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def decode_depth_image(msg: Image) -> np.ndarray:
    """Decode a 32FC1 or 16UC1 depth image into float32 metres."""
    if msg.encoding == '32FC1':
        dtype, scale = np.dtype(np.float32), 1.0
    elif msg.encoding == '16UC1':
        dtype, scale = np.dtype(np.uint16), 0.001
    else:
        raise ValueError(f'Unsupported depth image encoding: {msg.encoding!r}')
    if msg.is_bigendian:
        dtype = dtype.newbyteorder('>')
    row_stride = msg.step // dtype.itemsize
    frame = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(
        msg.height, row_stride)
    return (frame[:, :msg.width].astype(np.float32) * scale).copy()


def quantize_depth_upper_numpy_batch(
    depth_batch,
    step=0.1,
    min_value=None,
    max_value=None,
    preserve_zero=True,
):
    """
    Quantize a batch of depth images by rounding up to the nearest step.

    Examples:
        0.71 -> 0.8
        0.70 -> 0.7
        0.65 -> 0.7
        0.60 -> 0.6

    Supports shapes:
        (B, H, W)
        (B, 1, H, W)
        (B, T, 1, H, W)

    Args:
        depth_batch: numpy array
        step: quantization step
        min_value: optional minimum clipping value
        max_value: optional maximum clipping value
        preserve_zero: keep zero values as zero

    Returns:
        Quantized depth batch with the same shape.
    """

    depth_q = np.asarray(depth_batch).astype(np.float32).copy()

    if preserve_zero:
        zero_mask = depth_q == 0

    # Optional clipping
    if min_value is not None or max_value is not None:
        if min_value is None:
            min_value = np.min(depth_q)
        if max_value is None:
            max_value = np.max(depth_q)

        depth_q = np.clip(depth_q, min_value, max_value)

    # Small epsilon avoids changing exact bin values
    eps = 1e-6

    depth_q = np.ceil((depth_q - eps) / step) * step

    if preserve_zero:
        depth_q[zero_mask] = 0.0

    return depth_q


def preprocess_depth(depth: np.ndarray) -> np.ndarray:
    """Crop and sanitize one depth frame exactly as done during training."""
    if depth.shape[0] < 190 or depth.shape[1] < 590:
        raise ValueError(
            f'Depth image shape {depth.shape} is too small for crop '
            '[:190, 100:590]'
        )
    depth = depth[60:170, 230:440]
    depth = np.nan_to_num(depth, nan=10.0)
    depth = np.clip(depth, 0, 0.8)

    first_box_start_x=41
    first_box_start_y=0
    first_box_end_x=110
    first_box_end_y=65

    second_box_start_x=41
    second_box_start_y=140
    second_box_end_x=110
    second_box_end_y=210        

    object_start_x=33
    object_start_y=98
    object_end_x=60
    object_end_y=125    
    
    first_drawer_start_x=0
    first_drawer_start_y=0
    first_drawer_end_x=41
    first_drawer_end_y=65
        
    second_drawer_start_x=0
    second_drawer_start_y=140
    second_drawer_end_x=41
    second_drawer_end_y=210
    
    depth = quantize_depth_upper_numpy_batch(depth, step=0.05)             
    inbetween_region_first_box=depth[first_box_start_x:first_box_end_x,first_box_start_y:first_box_end_y]
    
    inbetween_depth_value_first_box=np.percentile(inbetween_region_first_box,10)
    depth[first_box_start_x:first_box_end_x,first_box_start_y:first_box_end_y]=inbetween_depth_value_first_box
    
    inbetween_region_second_box=depth[second_box_start_x:second_box_end_x,second_box_start_y:second_box_end_y]
    
    inbetween_depth_value_second_box=np.percentile(inbetween_region_second_box,10)
    depth[second_box_start_x:second_box_end_x,second_box_start_y:second_box_end_y]=inbetween_depth_value_second_box
    
    inbetween_region_first_drawer=depth[first_drawer_start_x:first_drawer_end_x,first_drawer_start_y:first_drawer_end_y]
    
    inbetween_depth_value_first_drawer=np.percentile(inbetween_region_first_drawer,10)
    depth[first_drawer_start_x:first_drawer_end_x,first_drawer_start_y:first_drawer_end_y]=inbetween_depth_value_first_drawer
    
    inbetween_region_second_drawer=depth[second_drawer_start_x:second_drawer_end_x,second_drawer_start_y:second_drawer_end_y]
    
    inbetween_depth_value_second_drawer=np.percentile(inbetween_region_second_drawer,10)
    depth[second_drawer_start_x:second_drawer_end_x,second_drawer_start_y:second_drawer_end_y]=inbetween_depth_value_second_drawer

    inbetween_region_object=depth[object_start_x:object_end_x,object_start_y:object_end_y]
    
    inbetween_depth_value_object=np.percentile(inbetween_region_object,10)
    depth[object_start_x:object_end_x,object_start_y:object_end_y]=inbetween_depth_value_object

    return depth.astype(np.float32, copy=False)


class Inference(Node):
    """Collect observations, invoke the policy, and execute action chunks."""

    def __init__(self):
        super().__init__('inference')
        self.declare_parameter('joint_names', DEFAULT_JOINT_NAMES)
        self.declare_parameter('observation_length', 5)
        self.declare_parameter('action_chunk_length', 10)
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('sync_tolerance_sec', 0.05)
        self.declare_parameter('joint_match_tolerance_rad', 0.1)
        self.declare_parameter('grip_release_delay_sec', 0.5)
        self.declare_parameter('home_position', DEFAULT_HOME_POSITION)
        self.declare_parameter('home_move_sec', 5.0)
        self.declare_parameter('checkpoint_path', '')
        self.declare_parameter('device', 'auto')
        self.declare_parameter('flow_steps', 100)
        self.declare_parameter('num_candidates', 10)
        self.declare_parameter('candidate_index', 0)
        self.declare_parameter('model_action_horizon', 20)
        self.declare_parameter('gripper_threshold', 0.5)
        self.declare_parameter('debug_enabled', False)
        self.declare_parameter('run_id', '')

        self.joint_names = list(self.get_parameter('joint_names').value)
        self.observation_length = int(
            self.get_parameter('observation_length').value)
        self.action_chunk_length = int(
            self.get_parameter('action_chunk_length').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.sync_tolerance = float(
            self.get_parameter('sync_tolerance_sec').value)
        self.joint_match_tolerance = float(
            self.get_parameter('joint_match_tolerance_rad').value)
        self.grip_release_delay = float(
            self.get_parameter('grip_release_delay_sec').value)
        self.home_position = [
            float(value)
            for value in self.get_parameter('home_position').value
        ]
        self.home_move_sec = float(
            self.get_parameter('home_move_sec').value)
        self.checkpoint_path = str(
            self.get_parameter('checkpoint_path').value)
        self.flow_steps = int(self.get_parameter('flow_steps').value)
        self.num_candidates = int(
            self.get_parameter('num_candidates').value)
        self.candidate_index = int(
            self.get_parameter('candidate_index').value)
        self.model_action_horizon = int(
            self.get_parameter('model_action_horizon').value)
        self.gripper_threshold = float(
            self.get_parameter('gripper_threshold').value)
        self.debug_enabled = bool(
            self.get_parameter('debug_enabled').value)
        requested_run_id = str(self.get_parameter('run_id').value)
        self.run_id = requested_run_id or (
            f'inference-{self.get_clock().now().nanoseconds}'
        )
        self._validate_parameters()
        self.dt = 1.0 / self.rate_hz
        self.sync_tolerance_ns = int(round(self.sync_tolerance * 1e9))
        self.device = self._resolve_device(
            str(self.get_parameter('device').value))
        self.model = self._load_model()

        buffer_size = max(20, int(self.rate_hz * 2))
        self._joint_buffer = deque(maxlen=buffer_size)
        self._depth_buffer = deque(maxlen=buffer_size)
        self._observations = deque(maxlen=self.observation_length)
        self._trajectory_active = False
        self._trajectory_result_future = None
        self._execution_start_ns = None
        self._execution_end_ns = None
        self._execution_joint_samples = []
        self._execution_depth_samples = []
        self._gripper_state = 0
        self._depth_encoding_error = None
        self._last_failure_reason = None

        self.create_subscription(
            JointState, '/joint_states', self._on_joint_state, 50)
        self.create_subscription(Image, DEPTH_TOPIC, self._on_depth, 10)
        self._trajectory_client = ActionClient(
            self, FollowJointTrajectory, ACTION_NAME)
        self._gripper_client = self.create_client(
            GripperControl, GRIPPER_SERVICE)

        self._event_publisher = None
        self._phase_publisher = None
        self._planned_target_publisher = None
        self._matched_target_publisher = None
        self._selected_joint_publisher = None
        self._selected_depth_publisher = None
        if self.debug_enabled:
            # run_start configures the separately launched canonicalizer. Keep
            # trace events so a subscriber that is still being discovered when
            # run_start is emitted can receive it instead of remaining
            # permanently unconfigured.
            event_qos = QoSProfile(
                depth=100,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._event_publisher = self.create_publisher(
                String, EVENT_TOPIC, event_qos
            )
            self._phase_publisher = self.create_publisher(
                String, PHASE_TOPIC, 20
            )
            self._planned_target_publisher = self.create_publisher(
                JointState, PLANNED_TARGET_TOPIC, 20
            )
            self._matched_target_publisher = self.create_publisher(
                JointState, MATCHED_TARGET_TOPIC, 20
            )
            self._selected_joint_publisher = self.create_publisher(
                JointState, SELECTED_JOINT_TOPIC, 20
            )
            self._selected_depth_publisher = self.create_publisher(
                Image, SELECTED_DEPTH_TOPIC, 20
            )
            self._emit_phase('startup')
            self._emit_event(
                'run_start',
                checkpoint_path=self.checkpoint_path,
                joint_names=self.joint_names,
                preprocessing_version=PREPROCESS_VERSION,
                parameters={
                    'observation_length': self.observation_length,
                    'action_chunk_length': self.action_chunk_length,
                    'rate_hz': self.rate_hz,
                    'sync_tolerance_sec': self.sync_tolerance,
                    'joint_match_tolerance_rad': self.joint_match_tolerance,
                    'model_action_horizon': self.model_action_horizon,
                    'candidate_index': self.candidate_index,
                    'num_candidates': self.num_candidates,
                    'flow_steps': self.flow_steps,
                    'device': str(self.device),
                    'home_position': self.home_position,
                    'home_move_sec': self.home_move_sec,
                    'gripper_threshold': self.gripper_threshold,
                    'grip_release_delay_sec': self.grip_release_delay,
                },
            )

    def _validate_parameters(self):
        if not self.joint_names:
            raise ValueError('joint_names must not be empty')
        if self.observation_length <= 0 or self.action_chunk_length <= 0:
            raise ValueError('observation and action lengths must be positive')
        if self.observation_length > self.action_chunk_length:
            raise ValueError(
                'observation_length must not exceed action_chunk_length'
            )
        if self.rate_hz <= 0.0:
            raise ValueError('rate_hz must be positive')
        if self.sync_tolerance < 0.0 or self.grip_release_delay < 0.0:
            raise ValueError('timing parameters must be non-negative')
        if self.joint_match_tolerance <= 0.0:
            raise ValueError('joint_match_tolerance_rad must be positive')
        if len(self.home_position) != len(self.joint_names):
            raise ValueError(
                'home_position must contain one value per joint name'
            )
        if self.home_move_sec <= 0.0:
            raise ValueError('home_move_sec must be positive')
        if not self.checkpoint_path:
            raise ValueError(
                "checkpoint_path is required; pass "
                "'--ros-args -p checkpoint_path:=/path/to/checkpoint.pt'"
            )
        if self.flow_steps <= 0 or self.num_candidates <= 0:
            raise ValueError('flow_steps and num_candidates must be positive')
        if not 0 <= self.candidate_index < self.num_candidates:
            raise ValueError(
                'candidate_index must be in [0, num_candidates)'
            )
        if self.model_action_horizon < self.action_chunk_length:
            raise ValueError(
                'model_action_horizon must be at least action_chunk_length'
            )

    def _now_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    @staticmethod
    def _set_header_stamp(msg, stamp_ns: int, frame_id: str = ''):
        msg.header.stamp.sec = int(stamp_ns // 1_000_000_000)
        msg.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
        msg.header.frame_id = frame_id

    def _emit_event(self, event_type: str, chunk_index=None, **payload):
        if self._event_publisher is None or not rclpy.ok():
            return
        message = String()
        message.data = dumps_event(event_message(
            self.run_id,
            event_type,
            self._now_ns(),
            chunk_index=chunk_index,
            **payload,
        ))
        self._event_publisher.publish(message)

    def _emit_phase(self, phase: str):
        if self._phase_publisher is None or not rclpy.ok():
            return
        message = String()
        message.data = phase
        self._phase_publisher.publish(message)

    def _publish_joint_debug(
        self, publisher, stamp_ns: int, positions, frame_id: str = ''
    ):
        if publisher is None:
            return
        message = JointState()
        self._set_header_stamp(message, stamp_ns, frame_id)
        message.name = list(self.joint_names)
        message.position = [float(value) for value in positions]
        publisher.publish(message)

    def _publish_depth_debug(
        self, stamp_ns: int, depth: np.ndarray, frame_id: str
    ):
        if self._selected_depth_publisher is None:
            return
        pixels = np.ascontiguousarray(depth, dtype='<f4')
        message = Image()
        self._set_header_stamp(message, stamp_ns, frame_id)
        message.height = pixels.shape[0]
        message.width = pixels.shape[1]
        message.encoding = '32FC1'
        message.is_bigendian = False
        message.step = pixels.shape[1] * pixels.dtype.itemsize
        message.data = pixels.tobytes()
        self._selected_depth_publisher.publish(message)

    def _resolve_device(self, requested: str) -> torch.device:
        if requested == 'auto':
            requested = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(requested)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA was requested but is not available')
        return device

    def _load_model(self) -> ConditionalDiffusionModel:
        self.get_logger().info(
            f'Loading model checkpoint {self.checkpoint_path} on {self.device}.'
        )
        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device
        )

        print(f"Checkpoint keys: {list(checkpoint.keys())}")

        state_dict = (
            checkpoint['model']
            if isinstance(checkpoint, dict) and 'model' in checkpoint
            else checkpoint
        )

        print(f"State dict keys: {list(state_dict.keys())}")
        

        model = ConditionalDiffusionModel()
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        model.requires_grad_(False)

        print(f"Model loaded successfully on {self.device}.")

        return model

    def _on_joint_state(self, msg: JointState):
        by_name = dict(zip(msg.name, msg.position))
        if not all(name in by_name for name in self.joint_names):
            return
        positions = np.asarray(
            [by_name[name] for name in self.joint_names], dtype=np.float32)
        sample_time_ns = stamp_nanoseconds(msg)
        self._joint_buffer.append((sample_time_ns, positions))
        if self._is_execution_timestamp(sample_time_ns):
            self._execution_joint_samples.append(
                (sample_time_ns, positions.copy())
            )

    def _on_depth(self, msg: Image):
        try:
            frame = preprocess_depth(decode_depth_image(msg))
        except ValueError as exc:
            message = str(exc)
            if message != self._depth_encoding_error:
                self.get_logger().error(message)
                self._depth_encoding_error = message
            return
        sample_time_ns = stamp_nanoseconds(msg)
        self._depth_buffer.append(
            (sample_time_ns, frame, msg.header.frame_id)
        )
        if self._is_execution_timestamp(sample_time_ns):
            self._execution_depth_samples.append(
                (sample_time_ns, frame.copy(), msg.header.frame_id)
            )

    def _is_execution_timestamp(self, sample_time_ns: int) -> bool:
        """Return whether a sensor sample belongs to the active trajectory."""
        if not self._trajectory_active or self._execution_start_ns is None:
            return False
        if sample_time_ns < self._execution_start_ns:
            return False
        return (
            self._execution_end_ns is None or
            sample_time_ns <= self._execution_end_ns
        )

    def _latest_synchronized_pair(self):
        """Pair the newest joint state with the closest acceptable depth."""
        if not self._joint_buffer or not self._depth_buffer:
            return None
        joint_time_ns, positions = self._joint_buffer[-1]
        depth_time_ns, depth, depth_frame_id = min(
            self._depth_buffer,
            key=lambda item: abs(item[0] - joint_time_ns),
        )
        if abs(depth_time_ns - joint_time_ns) > self.sync_tolerance_ns:
            return None
        return (
            positions.copy(),
            depth.copy(),
            float(self._gripper_state),
            joint_time_ns,
            depth_time_ns,
            depth_frame_id,
        )

    def _execution_candidates(self):
        """Return depth frames paired with joints at the camera timestamps."""
        if not self._execution_joint_samples or not self._execution_depth_samples:
            return []

        joint_samples = sorted(
            self._execution_joint_samples, key=lambda sample: sample[0]
        )
        depth_samples = sorted(
            self._execution_depth_samples, key=lambda sample: sample[0]
        )
        joint_times = np.asarray(
            [sample[0] for sample in joint_samples], dtype=np.int64
        )
        joint_positions = np.stack(
            [sample[1] for sample in joint_samples]
        ).astype(np.float64)

        candidates = []
        for depth_time_ns, depth, depth_frame_id in depth_samples:
            right = int(np.searchsorted(
                joint_times, depth_time_ns, side='left'
            ))
            position = None
            before_stamp_ns = None
            after_stamp_ns = None
            interpolation_alpha = None

            if 0 < right < len(joint_times):
                left = right - 1
                before_gap = depth_time_ns - joint_times[left]
                after_gap = joint_times[right] - depth_time_ns
                if (before_gap <= self.sync_tolerance_ns and
                        after_gap <= self.sync_tolerance_ns):
                    span = joint_times[right] - joint_times[left]
                    if span > 0.0:
                        position, interpolation_alpha = (
                            interpolate_joint_positions(
                                int(joint_times[left]),
                                joint_positions[left],
                                int(joint_times[right]),
                                joint_positions[right],
                                int(depth_time_ns),
                            )
                        )
                    else:
                        position = joint_positions[left].copy()
                    before_stamp_ns = int(joint_times[left])
                    after_stamp_ns = int(joint_times[right])

            if position is None:
                nearest = min(
                    range(len(joint_times)),
                    key=lambda index: abs(
                        joint_times[index] - depth_time_ns
                    ),
                )
                if (abs(joint_times[nearest] - depth_time_ns) >
                        self.sync_tolerance_ns):
                    continue
                position = joint_positions[nearest].copy()
                before_stamp_ns = int(joint_times[nearest])
                after_stamp_ns = int(joint_times[nearest])

            candidates.append(ExecutionCandidate(
                stamp_ns=int(depth_time_ns),
                positions=position,
                depth=depth.copy(),
                depth_frame_id=depth_frame_id,
                joint_before_stamp_ns=before_stamp_ns,
                joint_after_stamp_ns=after_stamp_ns,
                interpolation_alpha=interpolation_alpha,
            ))
        return candidates

    def _select_action_observations(
        self,
        chunk_index: int,
        desired_positions: np.ndarray,
        gripper_states: np.ndarray,
    ):
        """Match ordered camera/joint observations to the final action targets."""
        first_action = self.action_chunk_length - self.observation_length
        # Include the immediately preceding action as a matching-only anchor.
        # This keeps a position seen during the first half of the chunk from
        # being mistaken for one of the final actions if the path revisits it.
        match_start = max(0, first_action - 1)
        output_offset = first_action - match_start
        targets = np.asarray(
            desired_positions[match_start:], dtype=np.float64
        )
        candidates = self._execution_candidates()
        if len(candidates) < len(targets):
            detail = (
                'Not enough synchronized execution observations: got '
                f'{len(candidates)}, need {len(targets)} including the '
                'ordering anchor.'
            )
            self.get_logger().error(detail)
            self._emit_event(
                'selection_failed',
                chunk_index=chunk_index,
                reason='insufficient_candidates',
                detail=detail,
                candidate_count=len(candidates),
                required_candidate_count=len(targets),
            )
            return None

        candidate_positions = np.stack([
            candidate.positions for candidate in candidates
        ])
        absolute_error = np.abs(
            targets[:, np.newaxis, :] -
            candidate_positions[np.newaxis, :, :]
        )
        max_joint_error = np.max(absolute_error, axis=2)
        valid = max_joint_error <= self.joint_match_tolerance
        costs = np.mean(
            (absolute_error / self.joint_match_tolerance) ** 2,
            axis=2,
        )

        target_count, candidate_count = costs.shape
        accumulated = np.full(
            (target_count, candidate_count), np.inf, dtype=np.float64
        )
        parents = np.full(
            (target_count, candidate_count), -1, dtype=np.int64
        )
        accumulated[0, valid[0]] = costs[0, valid[0]]

        # Dynamic programming gives one globally best, strictly chronological
        # assignment. It prevents one camera frame from representing multiple
        # actions and prevents later targets from matching earlier observations.
        for target_index in range(1, target_count):
            for candidate_index in range(target_index, candidate_count):
                if not valid[target_index, candidate_index]:
                    continue
                previous = accumulated[
                    target_index - 1, :candidate_index
                ]
                if not np.any(np.isfinite(previous)):
                    continue
                parent = int(np.argmin(previous))
                accumulated[target_index, candidate_index] = (
                    previous[parent] + costs[target_index, candidate_index]
                )
                parents[target_index, candidate_index] = parent

        final_candidate = int(np.argmin(accumulated[-1]))
        if not np.isfinite(accumulated[-1, final_candidate]):
            closest_errors = np.min(max_joint_error, axis=1)
            detail = (
                'Could not match an ordered observation to the anchor and '
                'every final action '
                'within joint_match_tolerance_rad='
                f'{self.joint_match_tolerance:.4f}. Closest per-target '
                'maximum joint errors were: '
                + np.array2string(closest_errors, precision=5)
            )
            self.get_logger().error(detail)
            self._emit_event(
                'selection_failed',
                chunk_index=chunk_index,
                reason='joint_tolerance',
                detail=detail,
                candidate_count=len(candidates),
                closest_max_joint_errors_rad=closest_errors.tolist(),
                joint_match_tolerance_rad=self.joint_match_tolerance,
            )
            return None

        selected_indices = [final_candidate]
        for target_index in range(target_count - 1, 0, -1):
            final_candidate = int(
                parents[target_index, final_candidate]
            )
            selected_indices.append(final_candidate)
        selected_indices.reverse()

        observations = []
        trace_observations = []
        output_indices = selected_indices[output_offset:]
        for target_offset, candidate_index in enumerate(output_indices):
            candidate = candidates[candidate_index]
            action_index = first_action + target_offset
            error = max_joint_error[
                output_offset + target_offset, candidate_index
            ]
            desired = targets[output_offset + target_offset]
            observation_id = selected_observation_id(
                chunk_index, action_index + 1
            )
            self.get_logger().debug(
                f'Matched action {action_index + 1} to depth frame at '
                f'{candidate.stamp_ns} ns with maximum joint error '
                f'{error:.6f} rad.'
            )
            observation = ModelObservation(
                observation_id=observation_id,
                stamp_ns=candidate.stamp_ns,
                positions=candidate.positions.astype(
                    np.float32, copy=True
                ),
                depth=candidate.depth.copy(),
                gripper_state=float(gripper_states[action_index]),
                depth_frame_id=candidate.depth_frame_id,
                joint_before_stamp_ns=candidate.joint_before_stamp_ns,
                joint_after_stamp_ns=candidate.joint_after_stamp_ns,
                interpolation_alpha=candidate.interpolation_alpha,
            )
            signed_error = (
                observation.positions.astype(np.float64) - desired
            )
            error = float(np.max(np.abs(signed_error)))
            observations.append(observation)
            self._publish_joint_debug(
                self._selected_joint_publisher,
                observation.stamp_ns,
                observation.positions,
                candidate.depth_frame_id,
            )
            self._publish_joint_debug(
                self._matched_target_publisher,
                observation.stamp_ns,
                desired,
                candidate.depth_frame_id,
            )
            self._publish_depth_debug(
                observation.stamp_ns,
                observation.depth,
                candidate.depth_frame_id,
            )
            trace_observations.append({
                'observation_id': observation_id,
                'action_index': action_index + 1,
                'selected_depth_stamp_ns': observation.stamp_ns,
                'selected_depth_frame_id': candidate.depth_frame_id,
                'selected_depth_sha256': depth_sha256(observation.depth),
                'joint_before_stamp_ns': candidate.joint_before_stamp_ns,
                'joint_after_stamp_ns': candidate.joint_after_stamp_ns,
                'interpolation_alpha': candidate.interpolation_alpha,
                'positions': observation.positions.tolist(),
                'desired_positions': desired.tolist(),
                'signed_error_rad': signed_error.tolist(),
                'max_abs_error_rad': error,
                'gripper_state': int(gripper_states[action_index]),
            })
        self._emit_event(
            'selection_complete',
            chunk_index=chunk_index,
            candidate_count=len(candidates),
            joint_match_tolerance_rad=self.joint_match_tolerance,
            observations=trace_observations,
        )
        return observations

    def _bootstrap_observations(self) -> bool:
        self._emit_phase('bootstrap')
        self.get_logger().info(
            'Waiting for the first synchronized joint/depth observation ...')
        deadline = self.get_clock().now() + rclpy.duration.Duration(seconds=10.0)
        pair = None
        while rclpy.ok() and self.get_clock().now() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            pair = self._latest_synchronized_pair()
            if pair is not None:
                break
        if pair is None:
            self.get_logger().error(
                'Timed out waiting for synchronized joint and depth data.')
            self._emit_event(
                'bootstrap_failed', reason='sensor_timeout'
            )
            return False
        observation = ModelObservation(
            observation_id='bootstrap-000001',
            stamp_ns=pair[4],
            positions=pair[0].copy(),
            depth=pair[1].copy(),
            gripper_state=pair[2],
            depth_frame_id=pair[5],
            joint_before_stamp_ns=pair[3],
            joint_after_stamp_ns=pair[3],
            interpolation_alpha=None,
        )
        for _ in range(self.observation_length):
            self._observations.append(observation)
        self._publish_joint_debug(
            self._selected_joint_publisher,
            observation.stamp_ns,
            observation.positions,
            observation.depth_frame_id,
        )
        self._publish_depth_debug(
            observation.stamp_ns,
            observation.depth,
            observation.depth_frame_id,
        )
        self._emit_event(
            'bootstrap_complete',
            observation_id=observation.observation_id,
            repeated_count=self.observation_length,
            selected_depth_stamp_ns=observation.stamp_ns,
            selected_depth_frame_id=observation.depth_frame_id,
            selected_depth_sha256=depth_sha256(observation.depth),
            joint_stamp_ns=observation.joint_before_stamp_ns,
            positions=observation.positions.tolist(),
            gripper_state=int(observation.gripper_state),
        )
        self.get_logger().info(
            f'Initial observation repeated {self.observation_length} times.')
        return True

    def _model_inputs(self):
        joints = np.stack([
            np.append(sample.positions, sample.gripper_state)
            for sample in self._observations
        ]).astype(np.float32)
        depth = np.stack([
            sample.depth for sample in self._observations
        ])
        return joints, depth

    @torch.inference_mode()
    def run_model(self, joint_history: np.ndarray, depth_history: np.ndarray):
        """Run the trained policy.

        Integrate the learned flow with the Heun method used by the reference
        implementation, then convert the selected candidate into ten joint
        deltas and ten binary gripper states.
        """
        self.get_logger().info(
            'Model input columns: ' +
            ', '.join([*self.joint_names, 'gripper']) +
            '\nModel joint/gripper input:\n' +
            np.array2string(
                joint_history,
                precision=2,
                suppress_small=False,
                separator=', ',
            )
        )
        depth = torch.from_numpy(depth_history).unsqueeze(0).unsqueeze(2)
        observations = torch.from_numpy(joint_history).unsqueeze(0)
        depth = depth.to(device=self.device, dtype=torch.float32)
        observations = observations.to(
            device=self.device, dtype=torch.float32
        )
        depth = depth.repeat(self.num_candidates, 1, 1, 1, 1)
        observations = observations.repeat(self.num_candidates, 1, 1)

        actions = torch.randn(
            (
                self.num_candidates,
                self.model_action_horizon,
                7,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        self.get_logger().info(
            'Model input tensors: '
            f'depth={tuple(depth.shape)}, '
            f'observations={tuple(observations.shape)}, '
            f'actions={tuple(actions.shape)}, '
            f'device={self.device}'
        )
        step_size = 1.0 / self.flow_steps
        for step in range(self.flow_steps):
            time = torch.full(
                (self.num_candidates,),
                step * step_size,
                device=self.device,
                dtype=torch.float32,
            )
            velocity = self.model(
                depth, observations, actions, time
            )
            predicted_actions = actions + step_size * velocity
            next_time = torch.full(
                (self.num_candidates,),
                (step + 1) * step_size,
                device=self.device,
                dtype=torch.float32,
            )
            next_velocity = self.model(
                depth, observations, predicted_actions, next_time
            )
            actions = actions + 0.5 * step_size * (
                velocity + next_velocity
            )

        selected = actions[
            self.candidate_index, :self.action_chunk_length
        ].cpu().numpy() / 20.0
        # Training multiplied all action targets by 20. Undo that
        # normalization above; the first six values are then joint deltas in
        # radians and the seventh is the gripper state.
        joint_deltas = selected[:, :len(self.joint_names)]
        gripper_states = (
            selected[:, 6] > self.gripper_threshold
        ).astype(np.uint8)
        model_output = np.column_stack((joint_deltas, gripper_states))
        self.get_logger().info(
            'Model output columns: ' +
            ', '.join([
                *(f'{name}_delta' for name in self.joint_names),
                'gripper',
            ]) +
            '\nModel joint-delta/gripper output:\n' +
            np.array2string(
                model_output,
                precision=6,
                suppress_small=False,
                separator=', ',
            )
        )
        return joint_deltas, gripper_states

    def _validate_model_output(self, output):
        if output is None:
            return None
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise ValueError(
                'run_model must return (joint_deltas, gripper_states) or None')
        deltas = np.asarray(output[0], dtype=np.float64)
        states = np.asarray(output[1])
        expected = (self.action_chunk_length, len(self.joint_names))
        if deltas.shape != expected:
            raise ValueError(
                f'joint_deltas shape is {deltas.shape}; expected {expected}')
        if states.shape != (self.action_chunk_length,):
            raise ValueError(
                f'gripper_states shape is {states.shape}; expected '
                f'({self.action_chunk_length},)')
        if not np.all(np.isfinite(deltas)):
            raise ValueError('joint_deltas contains NaN or infinity')
        if not np.all(np.isin(states, (0, 1))):
            raise ValueError('gripper_states must contain only 0 or 1')
        return deltas, states.astype(np.uint8)

    def _current_joint_position(self):
        if not self._joint_buffer:
            return None
        return self._joint_buffer[-1][1].astype(np.float64, copy=True)

    def _build_trajectory(self, seed, deltas):
        trajectory = JointTrajectory()
        trajectory.joint_names = list(self.joint_names)
        position = seed.copy()
        for index, delta in enumerate(deltas):
            position += delta
            point = JointTrajectoryPoint()
            point.positions = position.tolist()
            point.time_from_start = rclpy.duration.Duration(
                seconds=(index + 1) * self.dt).to_msg()
            trajectory.points.append(point)
        return trajectory

    def _call_gripper(self, command: str) -> bool:
        if not self._gripper_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(
                f'{GRIPPER_SERVICE} unavailable; cannot send {command}.')
            return False
        request = GripperControl.Request()
        request.command = command
        future = self._gripper_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        response = future.result()
        if response is None or not response.success:
            detail = 'timed out' if response is None else response.message
            self.get_logger().error(f'Gripper {command} failed: {detail}')
            return False
        self.get_logger().info(f'Gripper -> {command}')
        return True

    def _sleep_while_spinning(self, seconds: float):
        deadline = self.get_clock().now() + rclpy.duration.Duration(
            seconds=seconds)
        while rclpy.ok() and self.get_clock().now() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)

    def _apply_gripper_state(self, state: int) -> bool:
        if state == self._gripper_state:
            return True
        if self._gripper_state == 0 and state == 1:
            if not self._call_gripper('grip'):
                return False
            # The policy state changes at grip time; release only stops the
            # suction command and does not change this logical state.
            self._gripper_state = state
            self._sleep_while_spinning(self.grip_release_delay)
            if not self._call_gripper('release'):
                return False
        elif not self._call_gripper('blow'):
            return False
        self._gripper_state = state
        return True

    def _fail_chunk(self, chunk_index: int, reason: str, **payload) -> bool:
        self._last_failure_reason = reason
        self._emit_event(
            'execution_result',
            chunk_index=chunk_index,
            status='failed',
            reason=reason,
            **payload,
        )
        return False

    def execute_chunk(self, chunk_index, deltas, gripper_states) -> bool:
        self._last_failure_reason = None
        seed = self._current_joint_position()
        if seed is None:
            self.get_logger().error('No joint state available for action seed.')
            return self._fail_chunk(chunk_index, 'missing_action_seed')
        desired_positions = integrate_joint_deltas(seed, deltas)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = self._build_trajectory(seed, deltas)
        try:
            send_future = self._trajectory_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_future)
            handle = send_future.result()
            if handle is None or not handle.accepted:
                self.get_logger().error('Trajectory goal was rejected.')
                return self._fail_chunk(chunk_index, 'goal_rejected')

            result_future = handle.get_result_async()
            self._trajectory_result_future = result_future
            start_time = self.get_clock().now()
            self._execution_start_ns = int(start_time.nanoseconds)
            self._execution_end_ns = None
            self._execution_joint_samples = []
            self._execution_depth_samples = []
            self._trajectory_active = True
            planned_stamps_ns = [
                self._execution_start_ns + int(round(
                    (index + 1) * self.dt * 1e9
                ))
                for index in range(self.action_chunk_length)
            ]
            for stamp_ns, positions in zip(
                    planned_stamps_ns, desired_positions):
                self._publish_joint_debug(
                    self._planned_target_publisher,
                    stamp_ns,
                    positions,
                )
            self._emit_phase('execution')
            self._emit_event(
                'execution_start',
                chunk_index=chunk_index,
                execution_start_ns=self._execution_start_ns,
                seed_positions=seed.tolist(),
                desired_positions=desired_positions.tolist(),
                joint_deltas=np.asarray(deltas).tolist(),
                gripper_states=np.asarray(gripper_states).astype(int).tolist(),
                planned_target_stamps_ns=planned_stamps_ns,
            )
            gripper_index = 0
            while rclpy.ok() and not result_future.done():
                rclpy.spin_once(self, timeout_sec=0.01)
                elapsed = (
                    self.get_clock().now() - start_time).nanoseconds * 1e-9
                while (gripper_index < self.action_chunk_length and
                       elapsed >= (gripper_index + 1) * self.dt):
                    if not self._apply_gripper_state(
                            int(gripper_states[gripper_index])):
                        handle.cancel_goal_async()
                        return self._fail_chunk(
                            chunk_index,
                            'gripper_command_failed',
                            action_index=gripper_index + 1,
                        )
                    gripper_index += 1

            self._execution_end_ns = self._now_ns()

            wrapper = result_future.result()
            if wrapper is None:
                self.get_logger().error('Trajectory result was unavailable.')
                return self._fail_chunk(
                    chunk_index, 'trajectory_result_unavailable'
                )
            result = wrapper.result
            if result.error_code != 0:
                self.get_logger().error(
                    f'Trajectory failed: {result.error_code} '
                    f'({result.error_string})')
                return self._fail_chunk(
                    chunk_index,
                    'trajectory_failed',
                    controller_error_code=int(result.error_code),
                    controller_error_string=result.error_string,
                    execution_start_ns=self._execution_start_ns,
                    execution_end_ns=self._execution_end_ns,
                )
            while gripper_index < self.action_chunk_length:
                if not self._apply_gripper_state(
                        int(gripper_states[gripper_index])):
                    return self._fail_chunk(
                        chunk_index,
                        'gripper_command_failed',
                        action_index=gripper_index + 1,
                    )
                gripper_index += 1

            # Sensor messages captured during the trajectory may arrive after
            # its result. Continue spinning briefly, but the timestamp window
            # prevents post-trajectory samples from entering the candidates.
            observation_deadline = (
                self.get_clock().now() +
                rclpy.duration.Duration(
                    seconds=max(0.25, 2 * self.sync_tolerance + 2 * self.dt)
                )
            )
            while (rclpy.ok() and self.get_clock().now() < observation_deadline):
                rclpy.spin_once(self, timeout_sec=0.01)
                joint_caught_up = (
                    self._joint_buffer and
                    self._joint_buffer[-1][0] >= self._execution_end_ns
                )
                depth_caught_up = (
                    self._depth_buffer and
                    self._depth_buffer[-1][0] >= self._execution_end_ns
                )
                if joint_caught_up and depth_caught_up:
                    break

            self._emit_event(
                'execution_result',
                chunk_index=chunk_index,
                status='succeeded',
                controller_error_code=int(result.error_code),
                execution_start_ns=self._execution_start_ns,
                execution_end_ns=self._execution_end_ns,
                captured_joint_count=len(self._execution_joint_samples),
                captured_depth_count=len(self._execution_depth_samples),
            )
            self._emit_phase('selection')
            observations = self._select_action_observations(
                chunk_index, desired_positions, gripper_states
            )
            if observations is None:
                self._last_failure_reason = 'selection_failed'
                return False
            self._observations.extend(observations)
            return True
        finally:
            self._trajectory_active = False
            self._trajectory_result_future = None
            self._execution_start_ns = None
            self._execution_end_ns = None
            self._execution_joint_samples = []
            self._execution_depth_samples = []

    def _wait_for_controller(self):
        self.get_logger().info(f'Waiting for {ACTION_NAME} ...')
        while rclpy.ok() and not self._trajectory_client.server_is_ready():
            rclpy.spin_once(self, timeout_sec=0.5)
        return rclpy.ok()

    def _move_home(self) -> bool:
        """Move to drawer_demo.yaml's home pose before starting inference."""
        trajectory = JointTrajectory()
        trajectory.joint_names = list(self.joint_names)
        point = JointTrajectoryPoint()
        point.positions = list(self.home_position)
        point.time_from_start = rclpy.duration.Duration(
            seconds=self.home_move_sec
        ).to_msg()
        trajectory.points = [point]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        self.get_logger().info(
            f'Moving to home position over {self.home_move_sec:.1f} seconds.'
        )
        send_future = self._trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('Home trajectory goal was rejected.')
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapper = result_future.result()
        if wrapper is None:
            self.get_logger().error('Home trajectory result was unavailable.')
            return False
        result = wrapper.result
        if result.error_code != 0:
            self.get_logger().error(
                f'Home trajectory failed: {result.error_code} '
                f'({result.error_string})'
            )
            return False
        self.get_logger().info('Home position reached.')
        return True

    def _finish_run(self, status: str, reason: str, exit_code: int) -> int:
        self._emit_phase('stopped')
        self._emit_event(
            'run_end',
            status=status,
            reason=reason,
            exit_code=exit_code,
        )
        if self.debug_enabled and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
        return exit_code

    def run(self) -> int:
        if not self._wait_for_controller():
            return self._finish_run('failed', 'controller_unavailable', 1)
        if not self._move_home():
            return self._finish_run('failed', 'home_move_failed', 1)
        if not self._bootstrap_observations():
            return self._finish_run('failed', 'bootstrap_failed', 1)
        chunk_number = 0
        while rclpy.ok():
            chunk_number += 1
            inference_start_ns = self._now_ns()
            self._emit_phase('inference')
            input_observation_ids = [
                observation.observation_id
                for observation in self._observations
            ]
            input_depth_hashes = [
                depth_sha256(observation.depth)
                for observation in self._observations
            ]
            joint_history, depth_history = self._model_inputs()
            try:
                output = self._validate_model_output(
                    self.run_model(joint_history, depth_history))
            except Exception as exc:
                self.get_logger().error(f'Model inference failed: {exc}')
                self.get_logger().error(traceback.format_exc())
                self._emit_event(
                    'inference_failed',
                    chunk_index=chunk_number,
                    inference_start_ns=inference_start_ns,
                    inference_end_ns=self._now_ns(),
                    input_observation_ids=input_observation_ids,
                    error=str(exc),
                )
                return self._finish_run('failed', 'inference_failed', 1)
            if output is None:
                self.get_logger().warning(
                    'run_model returned None; implement it to produce actions.')
                self._emit_event(
                    'inference_failed',
                    chunk_index=chunk_number,
                    inference_start_ns=inference_start_ns,
                    inference_end_ns=self._now_ns(),
                    input_observation_ids=input_observation_ids,
                    error='run_model returned None',
                )
                return self._finish_run('stopped', 'no_model_output', 0)
            inference_end_ns = self._now_ns()
            self._emit_event(
                'inference_complete',
                chunk_index=chunk_number,
                inference_start_ns=inference_start_ns,
                inference_end_ns=inference_end_ns,
                inference_duration_ns=(
                    inference_end_ns - inference_start_ns
                ),
                input_observation_ids=input_observation_ids,
                input_depth_sha256=input_depth_hashes,
                model_input_joint_gripper=joint_history.tolist(),
                joint_deltas=output[0].tolist(),
                gripper_states=output[1].astype(int).tolist(),
            )
            self.get_logger().info(
                f'Executing policy action chunk {chunk_number}.')
            try:
                chunk_succeeded = self.execute_chunk(
                    chunk_number, *output
                )
            except Exception as exc:
                self.get_logger().error(
                    f'Action chunk {chunk_number} failed: {exc}'
                )
                self.get_logger().error(traceback.format_exc())
                self._last_failure_reason = 'execution_exception'
                self._emit_event(
                    'execution_result',
                    chunk_index=chunk_number,
                    status='failed',
                    reason='execution_exception',
                    error=str(exc),
                )
                chunk_succeeded = False
            if not chunk_succeeded:
                return self._finish_run(
                    'failed',
                    self._last_failure_reason or 'chunk_failed',
                    1,
                )
        return self._finish_run('stopped', 'rclpy_shutdown', 0)


def main(args=None):
    rclpy.init(args=args)
    node = None
    code = 1
    try:
        node = Inference()
        code = node.run()
    except (ValueError, KeyboardInterrupt) as exc:
        if node is not None and not isinstance(exc, KeyboardInterrupt):
            node.get_logger().error(str(exc))
        if node is not None:
            reason = (
                'keyboard_interrupt'
                if isinstance(exc, KeyboardInterrupt)
                else 'startup_error'
            )
            node._finish_run('failed', reason, 1)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
