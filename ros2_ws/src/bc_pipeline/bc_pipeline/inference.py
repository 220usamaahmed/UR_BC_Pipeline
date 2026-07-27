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

from bc_pipeline.model import ConditionalDiffusionModel
from control_msgs.action import FollowJointTrajectory
from ecpmi_gripper.srv import GripperControl
from sensor_msgs.msg import Image, JointState
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


def stamp_seconds(msg) -> float:
    """Return a ROS header stamp as floating-point seconds."""
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


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
        self.declare_parameter('grip_release_delay_sec', 0.5)
        self.declare_parameter('home_position', DEFAULT_HOME_POSITION)
        self.declare_parameter('home_move_sec', 5.0)
        self.declare_parameter('checkpoint_path', '')
        self.declare_parameter('device', 'auto')
        self.declare_parameter('flow_steps', 100)
        self.declare_parameter('num_candidates', 10)
        self.declare_parameter('candidate_index', 0)
        self.declare_parameter('model_action_horizon', 20)
        self.declare_parameter('model_output_scale', 2.0)
        self.declare_parameter('joint_delta_divisor', 150.0)
        self.declare_parameter('gripper_threshold', 0.5)

        self.joint_names = list(self.get_parameter('joint_names').value)
        self.observation_length = int(
            self.get_parameter('observation_length').value)
        self.action_chunk_length = int(
            self.get_parameter('action_chunk_length').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.sync_tolerance = float(
            self.get_parameter('sync_tolerance_sec').value)
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
        self.model_output_scale = float(
            self.get_parameter('model_output_scale').value)
        self.joint_delta_divisor = float(
            self.get_parameter('joint_delta_divisor').value)
        self.gripper_threshold = float(
            self.get_parameter('gripper_threshold').value)
        self._validate_parameters()
        self.dt = 1.0 / self.rate_hz
        self.device = self._resolve_device(
            str(self.get_parameter('device').value))
        self.model = self._load_model()

        buffer_size = max(20, int(self.rate_hz * 2))
        self._joint_buffer = deque(maxlen=buffer_size)
        self._depth_buffer = deque(maxlen=buffer_size)
        self._observations = deque(maxlen=self.observation_length)
        self._trajectory_active = False
        self._trajectory_result_future = None
        self._gripper_state = 0
        self._depth_encoding_error = None

        self.create_subscription(
            JointState, '/joint_states', self._on_joint_state, 50)
        self.create_subscription(Image, DEPTH_TOPIC, self._on_depth, 10)
        self.create_timer(self.dt, self._observation_tick)
        self._trajectory_client = ActionClient(
            self, FollowJointTrajectory, ACTION_NAME)
        self._gripper_client = self.create_client(
            GripperControl, GRIPPER_SERVICE)

    def _validate_parameters(self):
        if not self.joint_names:
            raise ValueError('joint_names must not be empty')
        if self.observation_length <= 0 or self.action_chunk_length <= 0:
            raise ValueError('observation and action lengths must be positive')
        if self.rate_hz <= 0.0:
            raise ValueError('rate_hz must be positive')
        if self.sync_tolerance < 0.0 or self.grip_release_delay < 0.0:
            raise ValueError('timing parameters must be non-negative')
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
        if self.joint_delta_divisor == 0.0:
            raise ValueError('joint_delta_divisor must not be zero')

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
        self._joint_buffer.append((stamp_seconds(msg), positions))

    def _on_depth(self, msg: Image):
        try:
            frame = preprocess_depth(decode_depth_image(msg))
        except ValueError as exc:
            message = str(exc)
            if message != self._depth_encoding_error:
                self.get_logger().error(message)
                self._depth_encoding_error = message
            return
        self._depth_buffer.append((stamp_seconds(msg), frame))

    def _latest_synchronized_pair(self):
        """Pair the newest joint state with the closest acceptable depth."""
        if not self._joint_buffer or not self._depth_buffer:
            return None
        joint_time, positions = self._joint_buffer[-1]
        depth_time, depth = min(
            self._depth_buffer, key=lambda item: abs(item[0] - joint_time))
        if abs(depth_time - joint_time) > self.sync_tolerance:
            return None
        return positions.copy(), depth.copy(), float(self._gripper_state)

    def _observation_tick(self):
        # After bootstrap, history changes only while a goal is active.
        if (not self._trajectory_active or
                (self._trajectory_result_future is not None and
                 self._trajectory_result_future.done())):
            return
        pair = self._latest_synchronized_pair()
        if pair is not None:
            self._observations.append(pair)

    def _bootstrap_observations(self) -> bool:
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
            return False
        for _ in range(self.observation_length):
            self._observations.append(
                (pair[0].copy(), pair[1].copy(), pair[2])
            )
        self.get_logger().info(
            f'Initial observation repeated {self.observation_length} times.')
        return True

    def _model_inputs(self):
        joints = np.stack([
            np.append(sample[0], sample[2]) for sample in self._observations
        ]).astype(np.float32)
        depth = np.stack([sample[1] for sample in self._observations])
        return joints, depth

    @torch.inference_mode()
    def run_model(self, joint_history: np.ndarray, depth_history: np.ndarray):
        """Run the trained policy.

        Integrate the learned flow with the Heun method used by the reference
        implementation, then convert the selected candidate into ten joint
        deltas and ten binary gripper states.
        """
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
        ].cpu().numpy()
        selected *= self.model_output_scale
        joint_deltas = selected[:, :len(self.joint_names)]
        joint_deltas /= self.joint_delta_divisor
        gripper_states = (
            selected[:, 6] > self.gripper_threshold
        ).astype(np.uint8)
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

    def execute_chunk(self, deltas, gripper_states) -> bool:
        seed = self._current_joint_position()
        if seed is None:
            self.get_logger().error('No joint state available for action seed.')
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = self._build_trajectory(seed, deltas)
        try:
            send_future = self._trajectory_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_future)
            handle = send_future.result()
            if handle is None or not handle.accepted:
                self.get_logger().error('Trajectory goal was rejected.')
                return False
            self._trajectory_active = True

            result_future = handle.get_result_async()
            self._trajectory_result_future = result_future
            start_time = self.get_clock().now()
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
                        return False
                    gripper_index += 1

            wrapper = result_future.result()
            if wrapper is None:
                self.get_logger().error('Trajectory result was unavailable.')
                return False
            result = wrapper.result
            if result.error_code != 0:
                self.get_logger().error(
                    f'Trajectory failed: {result.error_code} '
                    f'({result.error_string})')
                return False
            while gripper_index < self.action_chunk_length:
                if not self._apply_gripper_state(
                        int(gripper_states[gripper_index])):
                    return False
                gripper_index += 1
            return True
        finally:
            self._trajectory_active = False
            self._trajectory_result_future = None

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

    def run(self) -> int:
        if not self._wait_for_controller():
            return 1
        if not self._move_home():
            return 1
        if not self._bootstrap_observations():
            return 1
        chunk_number = 0
        while rclpy.ok():
            chunk_number += 1
            try:
                output = self._validate_model_output(
                    self.run_model(*self._model_inputs()))
            except Exception as exc:
                self.get_logger().error(f'Model inference failed: {exc}')
                self.get_logger().error(traceback.format_exc())
                return 1
            if output is None:
                self.get_logger().warning(
                    'run_model returned None; implement it to produce actions.')
                return 0
            self.get_logger().info(
                f'Executing policy action chunk {chunk_number}.')
            if not self.execute_chunk(*output):
                return 1
        return 0


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
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
