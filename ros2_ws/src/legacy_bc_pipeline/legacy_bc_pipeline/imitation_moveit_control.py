import math
from random import random
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence

import numpy as np
import rclpy
import torch
from bc_pipeline.model import CriticConfig, QTransformer
from ecpmi_gripper.srv import GripperControl
from pymoveit2 import MoveIt2
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import SetBool

from legacy_bc_pipeline.model import ConditionalDiffusionModel

np.set_printoptions(suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)


s = 103
np.random.seed(s)
torch.manual_seed(s)
    
@dataclass
class LatestMsg:
    stamp_sec: float
    msg: object
    
@dataclass
class Observation:
    joints: np.ndarray
    depth: np.ndarray
    
    
class ImitationMoveitControl(Node):
    
    def __init__(self):
        super().__init__('imitation_moveit_control')
        
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
        
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("depth_topic", "/zed/zed_node/depth/depth_registered")
        self.declare_parameter("obs_window", 5)
        self.declare_parameter("control_rate_hz", 15.0)
        self.declare_parameter("joint_names", [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ])
        self.declare_parameter("gripper_state_service", "/set_observation_gripper_state")
        self.declare_parameter("gripper_service", "/gripper_control")
        self.declare_parameter(
            "checkpoint_path",
            "/data/external/marvin_weights/"
            "flow_matching_manually_processed_depth_images_ddp_epoch_1000.pt",
        )
        self.declare_parameter(
            "critic_checkpoint_path", 
            "/data/external/marvin_weights/"
            "ckpt_without_normalization_wit_MC_uncertainty_discount_factor_999_top_30_percent_all_skills_with_incorrect_trajs_epoch_1000_modified_propagation_left_drawer_ep_700.pt"
        )
        self.declare_parameter("num_predicted_actions", 20)
        
        self._joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self._depth_topic = str(self.get_parameter("depth_topic").value)
        self._obs_window = int(self.get_parameter("obs_window").value)
        self._control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self._joint_names = [str(name) for name in self.get_parameter("joint_names").value]
        self._latest_joint_state: Optional[LatestMsg] = None
        self._latest_depth: Optional[LatestMsg] = None
        self._name_to_index: Optional[Dict[str, int]] = None
        self._obs_queue: Deque[Observation] = deque(maxlen=self._obs_window)
        self._inference_future: Optional[Future] = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._gripper_state_service = str(
            self.get_parameter("gripper_state_service").value
        )
        self._gripper_service = str(self.get_parameter("gripper_service").value)
        self._checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        self._critic_checkpoint_path = str(
            self.get_parameter("critic_checkpoint_path").value
        )
        self._num_predicted_actions = int(
            self.get_parameter("num_predicted_actions").value
        )
        if not self._critic_checkpoint_path:
            raise ValueError(
                "critic_checkpoint_path is required; pass '--ros-args -p "
                "critic_checkpoint_path:=/path/to/critic.pt'"
            )
        if self._num_predicted_actions < 1:
            raise ValueError("num_predicted_actions must be at least 1")
        self._critic_config = CriticConfig()
        if self._obs_window != self._critic_config.hist_len:
            raise ValueError(
                f"obs_window must be {self._critic_config.hist_len} for the critic"
            )
        
        self.joint_observations = deque(maxlen=1000)
        self.depth_observations = deque(maxlen=30)
        
        self.motion_start_time = None
        self.motion_end_time = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
        
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self._joint_names,
            base_link_name="base_link",
            end_effector_name="tool0",
            group_name="ur_manipulator",
            use_move_group_action=True,
        )
        
        print("Initializing Robot Framework...")
        time.sleep(2.0)
        
        def to_rad(waypoint_deg):
            return [math.radians(angle_deg) for angle_deg in waypoint_deg]
        
        home = [-0.00, -90.00, 0.00, -90.00, 0.00, 90.00]
        home = to_rad(home)
        
        self.moveit2.move_to_configuration(home, self._joint_names, tolerance=0.005)
        self.moveit2.wait_until_executed()
        
        print("Robot Framework initialized. Subscribing to topics and starting control loop...")
        
        self.create_subscription(
            JointState, self._joint_states_topic, self._on_joint_state, 10
        )
        self.create_subscription(Image, self._depth_topic, self._on_depth, 10)
        
        self._gripper_client = self.create_client(GripperControl, self._gripper_service)
        
        print(f"Subscribed to {self._joint_states_topic} and {self._depth_topic}")
        
        self._gripper_state_srv = self.create_service(
            SetBool, self._gripper_state_service, self._set_observation_gripper_state
        )
        
        period = 1.0 / self._control_rate_hz if self._control_rate_hz > 0 else 0.0667
        self._control_timer = self.create_timer(period, self._control_step)
        
        self.executing_actions = False
        self._gripper_sequence_active = False
        self._next_checkpoint: List[float] | None = None
        self._prev_gripper_state: bool | None = None
        self._next_gripper_state: bool | None = None
        self._gripper_release_timer = None
        
        # Initial gripper state, set to 1 for placing models
        self._obs_gripper_state = 0.0

        
    def load_model(self):
        self.flow_matching_policy = ConditionalDiffusionModel()
        check_point = torch.load(
            self._checkpoint_path,
            map_location=self.device,
        )
        print("Loading flow matching policy from checkpoint")
        
        self.flow_matching_policy.load_state_dict(check_point['model'])
        self.flow_matching_policy.to(self.device)
        self.flow_matching_policy.eval()
        for p in self.flow_matching_policy.parameters():
            p.requires_grad_(False)

        self.critic_model = QTransformer(
            d_vis=self._critic_config.d_vis,
            d_nonvis=self._critic_config.d_nonvis,
            d_act=self._critic_config.d_act,
            d_model=self._critic_config.d_model,
            n_heads=self._critic_config.n_heads,
            n_layers=self._critic_config.n_layers,
            dropout=self._critic_config.dropout,
            hist_len=self._critic_config.hist_len,
            horizon=self._critic_config.horizon,
        )
        critic_checkpoint = torch.load(
            self._critic_checkpoint_path,
            map_location=self.device,
        )
        critic_state = (
            critic_checkpoint["q_state_dict"]
            if isinstance(critic_checkpoint, dict)
            and "q_state_dict" in critic_checkpoint
            else critic_checkpoint
        )
        self.critic_model.load_state_dict(critic_state)
        self.critic_model.to(self.device)
        self.critic_model.eval()
        for parameter in self.critic_model.parameters():
            parameter.requires_grad_(False)
        
        
    def _stamp_to_sec(self, stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _extract_joint_vector(self, joint_state: JointState) -> np.ndarray:
        positions = np.zeros(len(self._joint_names), dtype=np.float32)
        for i, name in enumerate(self._joint_names):
            idx = self._name_to_index.get(name)
            if idx is None or idx >= len(joint_state.position):
                continue
            positions[i] = joint_state.position[idx]
        return positions
    
    def _image_to_array(self, msg: Image) -> Optional[np.ndarray]:
        if msg.encoding == "32FC1":
            dtype = np.float32
        elif msg.encoding == "16UC1":
            dtype = np.uint16
        else:
            self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
            return None

        expected_len = msg.height * msg.width * np.dtype(dtype).itemsize
        if len(msg.data) < expected_len:
            self.get_logger().warn("Depth image buffer is smaller than expected.")
            return None

        array = np.frombuffer(msg.data, dtype=dtype, count=msg.height * msg.width)
        return array.reshape((msg.height, msg.width))

    def _on_joint_state(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return
        if self._name_to_index is None:
            self._name_to_index = {name: i for i, name in enumerate(msg.name)}
        self._latest_joint_state = LatestMsg(self._stamp_to_sec(msg.header.stamp), msg)
        self.joint_observations.append((self._latest_joint_state.stamp_sec, self._extract_joint_vector(msg)))

    def _on_depth(self, msg: Image) -> None:
        self._latest_depth = LatestMsg(self._stamp_to_sec(msg.header.stamp), msg)
        self.depth_observations.append((self._latest_depth.stamp_sec, self._image_to_array(msg)))

    def _set_observation_gripper_state(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        self._obs_gripper_state = 1.0 if request.data else 0.0
        
        if self._obs_gripper_state == 0.0:
            self._next_gripper_state = False
            self._execute_gripper_action_if_ready()
        
        response.success = True
        response.message = (
            f"Observation gripper state set to {self._obs_gripper_state:.1f}."
        )
        return response

    def _control_step(self):        
        if len(self.joint_observations) == 0 or len(self.depth_observations) == 0:
            self.get_logger().debug("Waiting for initial observations...")
            return
        
        if not self.executing_actions:
            if self.motion_start_time is None and self.motion_end_time is None:

                self.motion_start_time = self.joint_observations[0][0]
                self.motion_start_joint_state = self.joint_observations[0][1]
                self.motion_end_time = self.joint_observations[-1][0]
                self.motion_end_joint_state = self.joint_observations[-1][1]
            
            self._maybe_start_inference()
            self._maybe_collect_inference()
            self._execute_waypoint_if_ready()
            self._execute_gripper_action_if_ready()
        
    def _maybe_start_inference(self):
        if self._inference_future is not None:
            return
        
        assert self.motion_start_time is not None and self.motion_end_time is not None, "Motion start and end times must be set before starting inference."
        
        start_time = self.motion_start_time
        end_time = self.motion_end_time
        print(f"Motion start time in inference: {start_time}, Motion end time: {end_time}")
        
        duration = end_time - start_time
        time_delta = duration / 10
        
        start_joint_pos= min(self.joint_observations, key=lambda obs: abs(obs[0] - start_time))[1]
        end_joint_pos= min(self.joint_observations, key=lambda obs: abs(obs[0] - end_time))[1]
        t = np.linspace(0, 1, 10)[:, None]   # shape (10, 1)
        out = (1 - t) * start_joint_pos + t * end_joint_pos

        obs_counter=5
        self._obs_queue.clear()
        for t in np.arange(start_time + 5 * time_delta, end_time, time_delta):
            closest_joint_obs = min(self.joint_observations, key=lambda obs: abs(obs[0] - t))
            closest_depth_obs = min(self.depth_observations, key=lambda obs: abs(obs[0] - t))
            
            if obs_counter >= 9:
                obs_counter=9
            joints = out[obs_counter]
            obs_counter += 1

            joints = np.append(joints, self._obs_gripper_state)
            
            depth = closest_depth_obs[1]
           
            depth = depth[60:170, 230:440] 
            
            depth = np.nan_to_num(depth, nan=10.0)
            depth = np.clip(depth, 0, 0.8)

                        
            self._obs_queue.append(Observation(joints, depth))
            
        observations = list(self._obs_queue)[-self._obs_window:]
        self._inference_future = self._executor.submit(self._run_model, observations)
        
    def _maybe_collect_inference(self):
        if self._inference_future is None:
            return
        if not self._inference_future.done():
            return
        
        print("Model inference completed, collecting results...")

        try:
            actions = self._inference_future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Model inference failed: {exc}")
            actions = None
        finally:
            self._inference_future = None

        if actions is None:
            return
        
        actions = np.array(actions)
        actions = actions[:10, :]
        # joint_actions = actions[:, :6] / 150.0
        joint_actions = actions[:, :6] / 100.0
        gripper_actions = actions[:, 6]
        
        self._next_gripper_state = (gripper_actions[5:] > 0.5).any()
        
        print("Model Actions --------------")
        print(actions)
        print(f"Gripper actions: {gripper_actions}, Next gripper state: {self._next_gripper_state}")
        print("----------------------------")
        
        print(f"Model inference completed. Actions shape: {actions.shape}")
        
        joint_deltas = joint_actions
        self._next_checkpoint = np.sum(joint_deltas, axis=0) + self.joint_observations[-1][1][:6]
        
    def _execute_gripper_action_if_ready(self):
        if self._next_gripper_state is None:
            return
        
        if self._gripper_sequence_active:
            return
        
        if self._next_gripper_state == self._prev_gripper_state:
            return
        
        self._prev_gripper_state = self._next_gripper_state
        
        if self._next_gripper_state:
            self._start_grip_sequence()
        else:
            self._send_gripper_command("blow")
            self._obs_gripper_state = 0.0
            
    def _start_grip_sequence(self) -> None:
        self._gripper_sequence_active = True
        
        self._obs_gripper_state = 1.0
        
        if not self._send_gripper_command("grip"):
            self._prev_gripper_state = None
            return

        if self._gripper_release_timer is not None:
            self._gripper_release_timer.cancel()
        self._gripper_release_timer = self.create_timer(1.0, self._release_grip)

    def _release_grip(self) -> None:
        if self._gripper_release_timer is not None:
            self._gripper_release_timer.cancel()
            self._gripper_release_timer = None

        self._send_gripper_command("release")
        self._gripper_sequence_active = False
        
    def _send_gripper_command(self, command: str) -> bool:
        if not self._gripper_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn(
                f"Waiting for gripper service at {self._gripper_service}"
            )
            self._gripper_sequence_active = False
            return False

        request = GripperControl.Request()
        request.command = command
        self._gripper_future = self._gripper_client.call_async(request)
        return True
        
    def _execute_waypoint_if_ready(self):        
        if self.executing_actions:
            return
            
        if self._next_checkpoint is None:
            return
        
        print(f"Executing actions")

        checkpoint = self._next_checkpoint
        
        self._next_checkpoint = None

        self.executing_actions = True

        def move_to_checkpoint():            
            self.get_logger().info("Moving to next checkpoint...")
            self.motion_start_time = time.time()
            print("start time before motion: ", self.motion_start_time)
            print("joints before motion: ", self.joint_observations[-1][1][:6] * 180.0 / np.pi)
            print("checkpoint: ", checkpoint * 180.0 / np.pi)
            self.moveit2.move_to_configuration(checkpoint, self._joint_names, tolerance=0.001)
            self.get_logger().info("Waiting for movement to complete...")
            
            if not self.moveit2.wait_until_executed():
                self.get_logger().error("Failed to execute movement to checkpoint.")
            self.motion_end_time = time.time()
            print("end time after motion: ", self.motion_end_time)
            joints_after_motion = self.joint_observations[-1][1][:6] * 180.0 / np.pi
            print(f"Reached checkpoint after action. Current joint state: {joints_after_motion}")
            print(f"Difference from checkpoint: {(joints_after_motion - checkpoint * 180.0 / np.pi)}")
            self.executing_actions = False

            self.get_logger().info("Movement to checkpoint completed.")

        threading.Thread(target=move_to_checkpoint).start()

    def quantize_depth_upper_numpy_batch(
        self,
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
        
        
    @torch.no_grad()
    def _run_model(self, observations: Sequence[Observation]) -> List[List[float]]:                
        device = self.device
        
        depth_images = torch.stack([torch.from_numpy(obs.depth).unsqueeze(0) for obs in observations], dim=0)#.unsqueeze(0)  # Shape: (obs_window, 1, H, W)
        non_visual_obs = torch.stack([torch.from_numpy(obs.joints) for obs in observations], dim=0)  # Shape: (obs_window, num_joints)
        
        print("Model input:")
        print(non_visual_obs)

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
        
        depth_images=np.array(depth_images.detach().cpu()).astype(np.float32)
        depth_images=np.squeeze(depth_images, axis=1)
        
        for i in range(depth_images.shape[0]):    
            depth_images[i] = self.quantize_depth_upper_numpy_batch(
            depth_images[i],
            step=0.05
            )  
                
            inbetween_region_first_box=depth_images[i][first_box_start_x:first_box_end_x,first_box_start_y:first_box_end_y]
            
            inbetween_depth_value_first_box=np.percentile(inbetween_region_first_box,10)
            depth_images[i][first_box_start_x:first_box_end_x,first_box_start_y:first_box_end_y]=inbetween_depth_value_first_box
            print("inbetween_depth_value_first_box frame {}== {}" .format(i, inbetween_depth_value_first_box))
            

            inbetween_region_second_box=depth_images[i][second_box_start_x:second_box_end_x,second_box_start_y:second_box_end_y]
            
            inbetween_depth_value_second_box=np.percentile(inbetween_region_second_box,10)
            depth_images[i][second_box_start_x:second_box_end_x,second_box_start_y:second_box_end_y]=inbetween_depth_value_second_box
            print("inbetween_depth_value_second_box frame {}== {}" .format(i, inbetween_depth_value_second_box))
            
            inbetween_region_first_drawer=depth_images[i][first_drawer_start_x:first_drawer_end_x,first_drawer_start_y:first_drawer_end_y]
            
            inbetween_depth_value_first_drawer=np.percentile(inbetween_region_first_drawer,10)
            depth_images[i][first_drawer_start_x:first_drawer_end_x,first_drawer_start_y:first_drawer_end_y]=inbetween_depth_value_first_drawer
            print("inbetween_depth_value_first_drawer frame {}== {}" .format(i, inbetween_depth_value_first_drawer))

            inbetween_region_second_drawer=depth_images[i][second_drawer_start_x:second_drawer_end_x,second_drawer_start_y:second_drawer_end_y]
            
            inbetween_depth_value_second_drawer=np.percentile(inbetween_region_second_drawer,10)
            depth_images[i][second_drawer_start_x:second_drawer_end_x,second_drawer_start_y:second_drawer_end_y]=inbetween_depth_value_second_drawer
            print("inbetween_depth_value_second_drawer frame {}== {}" .format(i, inbetween_depth_value_second_drawer))
            
            inbetween_region_object=depth_images[i][object_start_x:object_end_x,object_start_y:object_end_y]
            
            inbetween_depth_value_object=np.percentile(inbetween_region_object,10)
            depth_images[i][object_start_x:object_end_x,object_start_y:object_end_y]=inbetween_depth_value_object
            print("inbetween_depth_value_object frame {}== {}" .format(i, inbetween_depth_value_object))
                            
        num_predicted_actions = self._num_predicted_actions
        # num_predicted_actions = 10
        action_sequence_length = 20
        num_steps = 100
        action_dim = 7
        
        non_visual_obs = non_visual_obs.to(device=device, dtype=torch.float32)
        
        depth_images = np.expand_dims(depth_images, axis=1)
        depth_images = torch.from_numpy(depth_images).to(device=device, dtype=torch.float32)
        depth_images = depth_images.repeat(num_predicted_actions, 1,1,1,1)
      
        non_visual_obs = non_visual_obs.repeat(num_predicted_actions, 1,1)     
        
        x = torch.randn((num_predicted_actions, action_sequence_length, action_dim), device=device, dtype=torch.float32)
        
        dt = 1.0 / num_steps
        
        for k in range(num_steps):
            t_k = k * dt
            t_k_tensor = torch.full(
                (num_predicted_actions,),
                t_k,
                device=device,
                dtype=torch.float32,
            )
            
            v_k = self.flow_matching_policy(depth_images, non_visual_obs, x, t_k_tensor) 
                        
            x_pred = x + dt * v_k   # Add some noise to the predicted action for better exploration
            
            t_k1 = (k + 1) * dt
            t_k1_tensor = torch.full(
                (num_predicted_actions,),
                t_k1,
                device=device,
                dtype=torch.float32,
            )
            
            v_k1 = self.flow_matching_policy(depth_images, non_visual_obs, x_pred, t_k1_tensor)
            
            x = x + 0.5 * dt * (v_k + v_k1)
        depth_features = self.flow_matching_policy.depth_encoder(
            depth_images.reshape(
                num_predicted_actions * self._obs_window,
                *depth_images.shape[-3:],
            )
        ).reshape(num_predicted_actions, self._obs_window, -1)
        q_values = self.critic_model(depth_features, non_visual_obs, x)

        self.get_logger().info("All actions:")
        for i in range(num_predicted_actions):
            self.get_logger().info(
                f"Action {i}: {np.array2string(x[i].cpu().numpy()[:, :7], formatter={'float_kind': lambda f: f'{f:.5f}'}, separator=', ')}; Q value: {q_values[i].item():.5f}"
            )

        selected_idx = int(torch.argmax(q_values).item())
        # selected_idx = int(input(f"Select an action index (0-{num_predicted_actions - 1}): "))
        # selected_idx = np.random.randint(0, num_predicted_actions)
        # selected_idx = 0

        self.get_logger().info(
            f"Critic Q values: {np.array2string(q_values.cpu().numpy(), formatter={'float_kind': lambda f: f'{f:.5f}'}, separator=', ')}; selected trajectory "
            f"{selected_idx}"
        )

        actions = x[selected_idx].cpu().numpy()[:, :7]
        executed_actions = actions[:10, :]
        self.get_logger().info("chosen action  : {}".format(executed_actions))

        input("Press Enter to continue...")
        
        return executed_actions.tolist()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImitationMoveitControl()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    

if __name__ == "__main__":
    main()
