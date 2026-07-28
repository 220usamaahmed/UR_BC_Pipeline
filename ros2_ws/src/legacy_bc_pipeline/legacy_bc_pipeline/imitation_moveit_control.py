import math
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence

import numpy as np
import rclpy
import torch
from ecpmi_gripper.srv import GripperControl
from pymoveit2 import MoveIt2
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import SetBool

from legacy_bc_pipeline.model import ConditionalDiffusionModel

np.set_printoptions(suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)


np.random.seed(1231)
torch.manual_seed(1231)
    
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
            "/home/shokry/ur3e-trajectories/weights_june_2/"
            "flow_matching_manually_processed_depth_images_with_percentile_"
            "masks_corrected_box_pos_all_skills_marvin_ep_3950.pt",
        )
        
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
       # print("---- Latest joint observation: ", self.joint_observations[-1])

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
        
        # print("---- Latest joint observation: ", self.joint_observations[-1] if self.joint_observations else "None")
        
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
    #    print("time delta: ", time_delta)
        
        
        
        
        start_joint_pos= min(self.joint_observations, key=lambda obs: abs(obs[0] - start_time))[1]
        end_joint_pos= min(self.joint_observations, key=lambda obs: abs(obs[0] - end_time))[1]
     #   print("start joint pos: ", start_joint_pos)
      #  print("end joint pos: ", end_joint_pos)
        t = np.linspace(0, 1, 10)[:, None]   # shape (10, 1)
        out = (1 - t) * start_joint_pos + t * end_joint_pos
     #   print("interpolated joint positions: ", out)

        obs_counter=5
        self._obs_queue.clear()
        for t in np.arange(start_time + 5 * time_delta, end_time, time_delta):
            closest_joint_obs = min(self.joint_observations, key=lambda obs: abs(obs[0] - t))
            closest_depth_obs = min(self.depth_observations, key=lambda obs: abs(obs[0] - t))
            
       #     print(f"{t}, {closest_joint_obs[0]}, {closest_depth_obs[0]}")
        #    print("joint == ", closest_joint_obs[1])
            
            # if abs(closest_joint_obs[0] - t) > time_delta or abs(closest_depth_obs[0] - t) > time_delta:
            #     self.get_logger().warn(f"No close observation found for time {t:.2f}. Skipping this timestamp.")
            #     continue
            
         #   joints = closest_joint_obs[1]
            if obs_counter >= 9:
                obs_counter=9
            joints = out[obs_counter]
            obs_counter += 1
          #  print("Closest joint observation: ", joints)
            joints = np.append(joints, self._obs_gripper_state)
            
            depth = closest_depth_obs[1]
           # depth = depth[:190, 100:590]
           
            depth = depth[ 60: 170  ,  230 : 440 ] 
            
            depth = np.nan_to_num(depth, nan=10.0)
            depth = np.clip(depth, 0, 0.8)

                        
            self._obs_queue.append(Observation(joints, depth))
            
           
            
        #t = torch.linspace(0, 1, steps=10).unsqueeze(1)   # shape [10, 1]
        #out = (1 - t) * self.motion_start_joint_state + t * self.motion_end_joint_state
       # observations = out[-self._obs_window:]
       # print("start joint state: ", self.motion_start_joint_state)
       # print("end joint state: ", self.motion_end_joint_state)
       # print("interpolated joint states: ", out[-self._obs_window:])

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
       # joint_actions = actions[:, :6] * np.pi / 180.0
        joint_actions = actions[:, :6] /150.0
        gripper_actions = actions[:, 6]
        
        self._next_gripper_state = (gripper_actions[5:] > 0.5).any()
        
        print("Model Actions --------------")
        print(actions)
        print(f"Gripper actions: {gripper_actions}, Next gripper state: {self._next_gripper_state}")
        print("----------------------------")
        
        print(f"Model inference completed. Actions shape: {actions.shape}")
        
        # Integrate actions over time to get the actual joint positions to execute
        # joint_deltas = joint_actions / self._control_rate_hz
        joint_deltas = joint_actions
    #    print("joint actions in execution == ", joint_actions)
        # self._next_checkpoint = np.sum(joint_deltas, axis=0) + self._obs_queue[-1].joints[:6]
        self._next_checkpoint = np.sum(joint_deltas, axis=0) + self.joint_observations[-1][1][:6]
        
      #  print(f"Actions: {np.sum(joint_deltas, axis=0)*180.0/np.pi}")
      #  print(f"Current joint state: {self._obs_queue[-1].joints[:6]}")
     #   print(f"Current joint state: {self.joint_observations[-1][1][:6]*180.0/np.pi}")
     #   print(f"Next checkpoint: {self._next_checkpoint*180.0/np.pi}")
        
    def _execute_gripper_action_if_ready(self):
        if self._next_gripper_state is None:
            # print("No gripper action ready for execution.")
            return
        
        if self._gripper_sequence_active:
            # print("Gripper sequence already active, waiting for it to complete before executing next gripper action.")
            return
        
        if self._next_gripper_state == self._prev_gripper_state:
            # print("Gripper state has not changed since last execution, skipping gripper command.")
            return
        
        self._prev_gripper_state = self._next_gripper_state
        
        if self._next_gripper_state:
            self._start_grip_sequence()
        else:
            self._send_gripper_command("blow")
            self._obs_gripper_state = 0.0
            
    def _start_grip_sequence(self) -> None:
        # print("Starting gripper sequence: GRIP")
        
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
        
        # print(f"Sending gripper command: {command}")
        
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
            # print("No checkpoint ready for execution.")
            return
        
        print(f"Executing actions")

        checkpoint = self._next_checkpoint
        
        self._next_checkpoint = None

        self.executing_actions = True

        def move_to_checkpoint():            
            self.get_logger().info("Moving to next checkpoint...")
            self.motion_start_time = time.time()
            print("start time before motion: ", self.motion_start_time)
            print("joints before motion: ", self.joint_observations[-1][1][:6]*180.0/np.pi)
            print("checkpoint: ", checkpoint*180.0/np.pi)
            self.moveit2.move_to_configuration(checkpoint, self._joint_names, tolerance=0.001)
            self.get_logger().info("Waiting for movement to complete...")
            
            if not self.moveit2.wait_until_executed():
                self.get_logger().error("Failed to execute movement to checkpoint.")
            # time.sleep(2)
            self.motion_end_time = time.time()
            print("end time after motion: ", self.motion_end_time)
            joints_after_motion = self.joint_observations[-1][1][:6]*180.0/np.pi
            print(f"Reached checkpoint after action. Current joint state: {joints_after_motion}")
            print(f"Difference from checkpoint: {(joints_after_motion - checkpoint*180.0/np.pi)}")
            self.executing_actions = False
            
            # Dummy motion
            # self.motion_start_time = time.time()
            # time.sleep(2.0)
            # self.motion_end_time = time.time()
            # self.executing_actions = False
            
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
      #  non_visual_obs_1 = torch.tensor([-3.9,-98.5,-1.04,-87.38 , 3.6,87.77,0.0])*math.pi/180.0
       # non_visual_obs = non_visual_obs_1.unsqueeze(0).repeat(5,1)
        
        print("Model input:")
        print(non_visual_obs)



        '''
        first_box_start_x=51
        first_box_start_y=0
        first_box_end_x=110
        first_box_end_y=55
       
        second_box_start_x=55
        second_box_start_y=150
        second_box_end_x=110
        second_box_end_y=230        
        
        object_start_x=33
        object_start_y=98
        object_end_x=60
        object_end_y=125  
        
        depth_images=np.array(depth_images.detach().cpu()).astype(np.float32)
        print("depth image shape: ", depth_images.shape)
        depth_images=np.squeeze(depth_images, axis=1)
        print("depth image shape after squeeze: ", depth_images.shape)
        
            
        depth_images = self.quantize_depth_upper_numpy_batch(
        depth_images,
        step=0.05
        )    
        inbetween_region_first_box=depth_images[:,first_box_start_x+5:,:first_box_end_y-11]
        inbetween_depth_value_first_box=np.mean(inbetween_region_first_box,axis=(1, 2),keepdims=True)
        depth_images[:,first_box_start_x+5:,:first_box_end_y-11]=inbetween_depth_value_first_box
        
        inbetween_region_second_box=depth_images[:,second_box_start_x+10:,second_box_start_y+30:]
        inbetween_depth_value_second_box=inbetween_depth_value_first_box#np.mean(inbetween_region_second_box)
        depth_images[:,second_box_start_x+10:,second_box_start_y+30:]=inbetween_depth_value_second_box        
                
        depth_images[:,first_box_start_x-10:first_box_start_x+5 ,first_box_start_y:first_box_end_y+10] = inbetween_depth_value_first_box
        depth_images[:,first_box_start_x+4:first_box_end_x ,first_box_end_y-10:first_box_end_y+10] = inbetween_depth_value_first_box
        
        depth_images[:,second_box_start_x-10:second_box_start_x+10 ,second_box_start_y-10:] = inbetween_depth_value_second_box
        depth_images[:,second_box_start_x-10: ,second_box_start_y-10:second_box_start_y+30] = inbetween_depth_value_second_box
    
        
        depth_value_box_region_first_image=np.min(depth_images[:,object_start_x:object_end_x , object_start_y: object_end_y],axis=(1, 2),keepdims=True)
        depth_images[:,object_start_x:object_end_x , object_start_y: object_end_y]=depth_value_box_region_first_image
        '''
        
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
     #   print("depth image shape: ", depth_images.shape)
        depth_images=np.squeeze(depth_images, axis=1)
      #  print("depth image shape after squeeze: ", depth_images.shape)
        
        for i in range(depth_images.shape[0]):    
            depth_images[i] = self.quantize_depth_upper_numpy_batch(
            depth_images[i],
            step=0.05
            )  
                
            inbetween_region_first_box=depth_images[i][first_box_start_x:first_box_end_x,first_box_start_y:first_box_end_y]
            #  print("inbetween_region_first_box shape == " , inbetween_region_first_box.shape)
            
            inbetween_depth_value_first_box=np.percentile(inbetween_region_first_box,10)
            depth_images[i][first_box_start_x:first_box_end_x,first_box_start_y:first_box_end_y]=inbetween_depth_value_first_box
            print("inbetween_depth_value_first_box frame {}== {}" .format(i, inbetween_depth_value_first_box))
            

            inbetween_region_second_box=depth_images[i][second_box_start_x:second_box_end_x,second_box_start_y:second_box_end_y]
            #   print("inbetween_region_second_box shape == " , inbetween_region_second_box.shape)
            
            inbetween_depth_value_second_box=np.percentile(inbetween_region_second_box,10)
            depth_images[i][second_box_start_x:second_box_end_x,second_box_start_y:second_box_end_y]=inbetween_depth_value_second_box
            print("inbetween_depth_value_second_box frame {}== {}" .format(i, inbetween_depth_value_second_box))
            
            inbetween_region_first_drawer=depth_images[i][first_drawer_start_x:first_drawer_end_x,first_drawer_start_y:first_drawer_end_y]
            #  print("inbetween_region_first_drawer shape == " , inbetween_region_first_drawer.shape)
            
            inbetween_depth_value_first_drawer=np.percentile(inbetween_region_first_drawer,10)
            depth_images[i][first_drawer_start_x:first_drawer_end_x,first_drawer_start_y:first_drawer_end_y]=inbetween_depth_value_first_drawer
            print("inbetween_depth_value_first_drawer frame {}== {}" .format(i, inbetween_depth_value_first_drawer))
            


            inbetween_region_second_drawer=depth_images[i][second_drawer_start_x:second_drawer_end_x,second_drawer_start_y:second_drawer_end_y]
            # print("inbetween_region_second_drawer shape == " , inbetween_region_second_drawer.shape)
            
            inbetween_depth_value_second_drawer=np.percentile(inbetween_region_second_drawer,10)
            depth_images[i][second_drawer_start_x:second_drawer_end_x,second_drawer_start_y:second_drawer_end_y]=inbetween_depth_value_second_drawer
            print("inbetween_depth_value_second_drawer frame {}== {}" .format(i, inbetween_depth_value_second_drawer))
            

            inbetween_region_object=depth_images[i][object_start_x:object_end_x,object_start_y:object_end_y]
            #print("inbetween_region_object shape == " , inbetween_region_object.shape)
            
            inbetween_depth_value_object=np.percentile(inbetween_region_object,10)
            depth_images[i][object_start_x:object_end_x,object_start_y:object_end_y]=inbetween_depth_value_object
            print("inbetween_depth_value_object frame {}== {}" .format(i, inbetween_depth_value_object))
                            


        
        num_predicted_actions = 1
        action_sequence_length = 20
        num_steps = 100
        action_dim = 7
            
       # depth_images = depth_images.to(device=device, dtype=torch.float32)
       
       # depth_images_qunat = depth_images_qunat.to(device=device, dtype=torch.float32)
       
       
        non_visual_obs = non_visual_obs.to(device=device, dtype=torch.float32)
        #non_visual_obs [...,:6]*=0.0
        
      #  depth_images = depth_images.repeat(num_predicted_actions, 1,1,1,1)
        depth_images = np.expand_dims(depth_images, axis=1)
        depth_images = torch.from_numpy(depth_images).to(device=device, dtype=torch.float32)
        depth_images = depth_images.repeat(num_predicted_actions, 1,1,1,1)
        
        # Save depth images for debugging
        # for i in range(depth_images.shape[0]):
        #     depth_image_np = depth_images[i, 0].cpu().numpy()
        #     depth_image_uint16 = (depth_image_np*255).astype(np.uint8)  # Convert to millimeters and uint16
        #     print("shape to save", depth_image_uint16.shape, np.min(depth_image_uint16), np.max(depth_image_uint16))
        #     # image shape (1, 110, 210)
        #     depth_image_uint16 = np.squeeze(depth_image_uint16, axis=0)  # Remove channel dimension
        #     cv2.imwrite(f"debug_depth_image_{i}.png", depth_image_uint16)
        #     break
        
      
      
        non_visual_obs = non_visual_obs.repeat(num_predicted_actions, 1,1)     
        
        x = torch.randn((num_predicted_actions, action_sequence_length, action_dim), device=device, dtype=torch.float32)
        
        dt = 1.0 / num_steps
        
        for k in range(num_steps):
            t_k = k * dt
            t_k_tensor = torch.tensor(t_k, device=device, dtype=torch.float32).unsqueeze(0)
            
            v_k = self.flow_matching_policy(depth_images, non_visual_obs, x, t_k_tensor) 
                        
            x_pred = x + dt * v_k   # Add some noise to the predicted action for better exploration
            
            t_k1 = (k + 1) * dt
            t_k1_tensor = torch.tensor(t_k1, device=device, dtype=torch.float32).unsqueeze(0)
            
            v_k1 = self.flow_matching_policy(depth_images, non_visual_obs, x_pred, t_k1_tensor)
            
            x = x + 0.5 * dt * (v_k + v_k1)

        
        
        #actions = x.squeeze(0).cpu().numpy()[:, :7]
        # print("inferred actions == " )
        # for i in range(num_predicted_actions):
        #     print(f"Action sequence {i}:")
        #     print(x[i])#/150.0)
        #     print("------------------")
        
       # actions = x[].cpu().numpy()[:, :7]
        # random_idx = random.randint(0, num_predicted_actions - 1)
        # idx = int(input(f"Select action sequence to execute (0-{num_predicted_actions - 1}): ").strip() or random_idx)
        idx = 0
        
        actions = x[idx].cpu().numpy()[:, :7]
        # print("Selected action sequence: ", actions / 150.0)
      #  if self.flow_matching_policy.inference_step > 3:
        executed_actions = actions[:10, :]*2.0
        # joint_actions = executed_actions[:, :6] * np.pi / 180.0
        # executed_actions[:, :6] = executed_actions[:, :6] / 150.0
        print("chosen action  : ", executed_actions/2.0) 
        # input("Press Enter to execute the above action sequence...")
        # print("")
        
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
