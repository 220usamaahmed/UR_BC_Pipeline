"""Execute a configurable sequence of joint, gripper, and recorder steps."""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import rclpy
from control_msgs.msg import JointJog
from ecpmi_gripper.srv import GripperControl
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32, UInt8
from std_srvs.srv import Trigger


JointWaypoint = List[float]


@dataclass
class Step:
    """One operation in the hard-coded trajectory."""

    kind: str
    waypoint: Optional[JointWaypoint] = None
    gripper_command: Optional[str] = None
    wait_sec: float = 0.0
    output_dir: Optional[str] = None
    segment_id: Optional[int] = None


class TrajectoryControl(Node):
    def __init__(self) -> None:
        super().__init__("trajectory_control")

        self.declare_parameter("command_topic", "/servo_node/delta_joint_cmds")
        self.declare_parameter("command_topic_raw", "/servo_node/delta_joint_cmds_raw")
        self.declare_parameter("control_period", 0.01)
        self.declare_parameter("auto_start_servo", True)
        self.declare_parameter("start_servo_service", "/servo_node/start_servo")
        self.declare_parameter("gripper_service", "/gripper_control")
        self.declare_parameter("gripper_state_topic", "/gripper_state")
        self.declare_parameter("recorder_start_service", "/dataset_recorder/start")
        self.declare_parameter("recorder_stop_service", "/dataset_recorder/stop")
        self.declare_parameter("record", True)
        self.declare_parameter("k_p_joint", 4.0)
        self.declare_parameter("max_joint_speed", 1.0)
        self.declare_parameter("joint_tolerance", 0.01)
        self.declare_parameter("min_joint_speed", 0.01)
        self.declare_parameter("velocity_noise_std", 0.1)

        self._command_topic = str(self.get_parameter("command_topic").value)
        self._command_topic_raw = str(self.get_parameter("command_topic_raw").value)
        self._control_period = float(self.get_parameter("control_period").value)
        self._auto_start_servo = bool(self.get_parameter("auto_start_servo").value)
        self._start_servo_service = str(self.get_parameter("start_servo_service").value)
        self._gripper_service = str(self.get_parameter("gripper_service").value)
        self._gripper_state_topic = str(self.get_parameter("gripper_state_topic").value)
        self._recorder_start_service = str(
            self.get_parameter("recorder_start_service").value
        )
        self._recorder_stop_service = str(
            self.get_parameter("recorder_stop_service").value
        )
        self._record_enabled = bool(self.get_parameter("record").value)
        self._max_joint_speed = float(self.get_parameter("max_joint_speed").value)
        self._joint_tolerance = float(self.get_parameter("joint_tolerance").value)
        self._velocity_noise_std = float(
            self.get_parameter("velocity_noise_std").value
        )

        self._joint_names = [
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "shoulder_pan_joint",
        ]
        self._steps = self._make_steps()
        self._current_step_index = 0
        self._completed = False
        self._current_joints: Optional[List[float]] = None
        self._name_to_index: Optional[Dict[str, int]] = None
        self._waiting_until_sec: Optional[float] = None
        self._advance_after_wait = False
        self._gripper_future: Optional[rclpy.task.Future] = None
        self._gripper_wait_sec = 0.0
        self._gripper_state = 0.0
        self._current_segment_id = 0
        self._recorder_stop_future: Optional[rclpy.task.Future] = None
        self._recorder_stopping = False

        self._max_error = 0.0
        self._current_max_error = 0.0
        self._apply_deviation = False
        self._additive_noise = [0.0] * len(self._joint_names)
        self._noise_direction = [1] * len(self._joint_names)

        self._joint_cmd_pub = self.create_publisher(JointJog, self._command_topic, 10)
        self._joint_cmd_raw_pub = self.create_publisher(
            JointJog, self._command_topic_raw, 10
        )
        self._gripper_state_pub = self.create_publisher(
            UInt8, self._gripper_state_topic, 10
        )
        self._segment_pub = self.create_publisher(Int32, "/current_segment", 10)
        self._joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, 10
        )
        self._start_servo_client = self.create_client(Trigger, self._start_servo_service)
        self._gripper_client = self.create_client(GripperControl, self._gripper_service)

        self._recorder_start_client = None
        self._recorder_stop_client = None
        self._recorder_param_client = None
        if self._record_enabled:
            self._recorder_start_client = self.create_client(
                Trigger, self._recorder_start_service
            )
            self._recorder_stop_client = self.create_client(
                Trigger, self._recorder_stop_service
            )
            self._recorder_param_client = self.create_client(
                SetParameters, "/dataset_recorder/set_parameters"
            )

        self._start_servo_timer = None
        if self._auto_start_servo:
            self._start_servo_timer = self.create_timer(1.0, self._try_start_servo)
        self._recorder_start_timer = None
        if self._record_enabled:
            self._recorder_start_timer = self.create_timer(
                1.0, self._try_start_recorder
            )
        self._control_timer = self.create_timer(
            self._control_period, self._control_step
        )

        self._publish_segment(self._current_segment_id)
        self._publish_gripper_state(self._gripper_state)
        self.get_logger().info(
            f"TrajectoryControl ready. Publishing joint commands on "
            f"{self._command_topic} now"
        )

    @staticmethod
    def _make_steps() -> List[Step]:
        """Return the trajectory steps. Populate this list for each experiment."""

        def to_rad(waypoint_deg: JointWaypoint) -> JointWaypoint:
            return [math.radians(angle_deg) for angle_deg in waypoint_deg]

        home = [-90.00, 0.00, -90.00, 0.00, 90.00, -0.00]
        home_with_noise = np.array(home) + np.random.uniform(-6.0, 6.0, size=len(home))

        # gripping_prepare = [-110.04, -66.69, -92.83, 89.51, 103.23, -59.30]
        # gripping = [-111.31, -81.61, -76.64, 89.52, 103.23, -59.33]

        # gripping_prepare = [-72.48, 74.21, -91.82, -89.61, 77.11, 65.42]
        # gripping = [-71.01, 84.58, -103.67, -89.62, 77.16, 65.44]

        # gripping_prepare = [-108.49, -73.86, -87.56, 90.06, 91.45, -68.61]
        # gripping = [-109.92, -83.64, -76.34, 90.03, 91.46, -68.63]

        # gripping_prepare = [-74.02, 77.06, -93.12, -89.61, 92.68, 75.54]
        # gripping = [-72.48, 86.67, -104.30, -89.62, 92.73, 75.56]

        # gripping_prepare = [-124.01, -42.06, -103.83, 90.08, 84.71, -67.05]
        # gripping = [-122.68, -64.41, -82.81, 90.09, 84.74, -67.06]

        # gripping_prepare = [-71.02, 66.09, -85.17, -89.60, 92.93, 79.19]
        # gripping = [-69.84, 82.69, -102.95, -89.61, 93.01, 79.22]

        # gripping_prepare = [-109.33, -64.95, -95.61, 90.01, 82.37, -62.37]
        # gripping = [-110.74, -83.66, -75.49, 90.03, 82.38, -62.40]

        gripping_prepare = [-107.25, -77.34, -85.32, 90.01, 70.47, -75.42]
        gripping = [-108.66, -85.41, -75.83, 90.02, 70.47, -75.44]

        ### Open Right

        return [
            Step(kind="waypoint", waypoint=to_rad(home_with_noise)),
            
            Step(kind="recorder_start"),
            Step(kind="wait", wait_sec=1.0),
            
            Step(kind="reset_noise"),
            Step(kind="waypoint", waypoint=to_rad(gripping_prepare)),
            Step(kind="waypoint", waypoint=to_rad(gripping)),
            
            Step(kind="gripper", gripper_command="grip", wait_sec=0.3),
            Step(kind="gripper", gripper_command="release", wait_sec=0.1),
            
            Step(kind="reset_noise"),
            Step(kind="waypoint", waypoint=to_rad(home)),
            
            Step(kind="recorder_stop", output_dir="/data/external/incorrect-data/pick/pick"),
        ]

        ### Open Left

        grip = [-125.47, -79.93, -155.61, -33.94, 89.73, -35.28]

        w1 = [-119.03, -90.79, -151.31, -28.66, 89.85, -30.02]
        w2 = [-113.14, -100.13, -148.06, -22.59, 90.07, -23.97]
        w3 = [-106.15, -110.38, -145.54, -13.27, 90.85, -14.67]
        w4 = [-102.65, -115.16, -146.02, -6.86, 92.63, -8.27]

        high = [-115.91, -53.29, -191.04, -33.92, 87.69, -32.14]
        high_with_noise = np.array(high) + np.random.uniform(-6.0, 6.0, size=len(high))

        high2 = [-96.32, -78.83, -184.07, -16.90, 87.12, -15.18]
        high2_with_noise = np.array(high2) + np.random.uniform(-10.0, 10.0, size=len(high2))

        right = [-104.67, -104.12, -151.18, -46.42, 90.00, -45.15]
        right_with_noise = np.array(right) + np.random.uniform(-10.0, 10.0, size=len(right))

        right2 = [-85.38, -125.55, -148.85, -29.47, 89.80, -28.27]
        right2_with_noise = np.array(right2) + np.random.uniform(-10.0, 10.0, size=len(right2))

        return [
            Step(kind="waypoint", waypoint=to_rad(home)),
            Step(kind="wait", wait_sec=1.0),
            
            Step(kind="recorder_start"),
            Step(kind="wait", wait_sec=1.0),
            
            Step(kind="reset_noise"),
            Step(kind="waypoint", waypoint=to_rad(right_with_noise.tolist())),
            
            # Step(kind="waypoint", waypoint=to_rad(grip)),
            Step(kind="gripper", gripper_command="grip", wait_sec=0.3),
            Step(kind="gripper", gripper_command="release", wait_sec=1.0),

            Step(kind="waypoint", waypoint=to_rad(right2_with_noise.tolist())),
            # Step(kind="waypoint", waypoint=to_rad(w1)),
            # Step(kind="waypoint", waypoint=to_rad(w2)),
            # Step(kind="waypoint", waypoint=to_rad(w3)),
            # Step(kind="waypoint", waypoint=to_rad(w4)),
            
            Step(kind="gripper", gripper_command="blow", wait_sec=0.1),
            
            Step(kind="reset_noise"),
            Step(kind="waypoint", waypoint=to_rad(home)),
            
            Step(kind="recorder_stop", output_dir="/data/external/incorrect-data/open_left/open_left")
        ]

    def _publish_gripper_state(self, state: int) -> None:
        print(f"Publishing gripper state: {state}")

        msg = UInt8()
        msg.data = int(state)
        self._gripper_state_pub.publish(msg)

    def _publish_segment(self, segment_id: int) -> None:
        msg = Int32()
        msg.data = int(segment_id)
        self._segment_pub.publish(msg)

    def _set_gripper_state(self, state: int) -> None:
        if self._gripper_state != state:
            self._gripper_state = state
            self._publish_gripper_state(state)

    def _try_start_recorder(self) -> None:
        if self._recorder_start_client is None:
            return
        if not self._recorder_start_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn(
                f"Waiting for dataset recorder start service at "
                f"{self._recorder_start_service}"
            )
        # Recorder start is intentionally controlled by explicit trajectory steps.

    def _handle_start_recorder(self, future: rclpy.task.Future) -> None:
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Failed to call recorder start: {exc}")
            return
        if response.success:
            self.get_logger().info("Dataset recorder started.")
            self._publish_gripper_state(self._gripper_state)
            if self._recorder_start_timer is not None:
                self._recorder_start_timer.cancel()
        else:
            self.get_logger().warn(f"Recorder start failed: {response.message}")

    def _try_start_servo(self) -> None:
        if not self._auto_start_servo:
            return
        if not self._start_servo_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn(
                f"Waiting for MoveIt Servo start service at {self._start_servo_service}"
            )
            return
        future = self._start_servo_client.call_async(Trigger.Request())
        future.add_done_callback(self._handle_start_servo)

    def _handle_start_servo(self, future: rclpy.task.Future) -> None:
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Failed to call start_servo: {exc}")
            return
        if response.success:
            self.get_logger().info("MoveIt Servo started.")
            if self._start_servo_timer is not None:
                self._start_servo_timer.cancel()
        else:
            self.get_logger().warn(f"MoveIt Servo start failed: {response.message}")

    def _joint_state_callback(self, msg: JointState) -> None:
        if self._name_to_index is None:
            self._name_to_index = {name: i for i, name in enumerate(msg.name)}
            missing = [name for name in self._joint_names if name not in self._name_to_index]
            if missing:
                self.get_logger().warn(
                    f"JointState is missing joints: {missing}. Waypoint tracking may fail."
                )

        joints = []
        for name in self._joint_names:
            index = self._name_to_index.get(name)
            if index is None or index >= len(msg.position):
                return
            joints.append(msg.position[index])
        self._current_joints = joints

    def _abort_with_error(self, message: str) -> None:
        self.get_logger().error(message)
        self._finish_and_shutdown()

    def _finish_and_shutdown(self) -> None:
        self._completed = True
        self._publish_joint_command([0.0] * len(self._joint_names))
        self._set_gripper_state(0)
        for timer in (
            self._start_servo_timer,
            self._recorder_start_timer,
            self._control_timer,
        ):
            if timer is not None:
                timer.cancel()
        if self._record_enabled:
            self._request_recorder_stop()
        else:
            rclpy.shutdown()

    def _request_recorder_stop(self) -> None:
        if self._recorder_stop_client is None:
            self.get_logger().warn(
                "Recorder stop client not initialized. Shutting down anyway."
            )
            rclpy.shutdown()
            return
        if not self._recorder_stop_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn(
                f"Recorder stop service unavailable at {self._recorder_stop_service}. "
                "Shutting down anyway."
            )
            rclpy.shutdown()
            return
        self._recorder_stop_future = self._recorder_stop_client.call_async(
            Trigger.Request()
        )
        self._recorder_stop_future.add_done_callback(self._on_recorder_stopped)

    def _on_recorder_stopped(self, future: rclpy.task.Future) -> None:
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Dataset recorder stopped.")
                self._current_step_index += 1
                self._recorder_stopping = False
            else:
                self.get_logger().warn(f"Recorder stop failed: {response.message}")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Failed to stop recorder: {exc}")

    def _control_step(self) -> None:
        if not self._steps or self._completed:
            self._publish_joint_command([0.0] * len(self._joint_names))
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        if self._waiting_until_sec is not None:
            if now_sec < self._waiting_until_sec:
                self._publish_joint_command([0.0] * len(self._joint_names))
                return
            self._waiting_until_sec = None
            if self._advance_after_wait:
                self._advance_after_wait = False
                self._current_step_index += 1

        if self._current_step_index >= len(self._steps):
            self.get_logger().info("All steps complete. Shutting down.")
            self._finish_and_shutdown()
            return

        step = self._steps[self._current_step_index]
        if step.kind == "reset_noise":
            self._reset_noise()
            self._current_step_index += 1
            return
        if step.kind == "hold":
            input("Trajectory complete. Press Enter to continue...")
            self._current_step_index += 1
            return
        if step.kind == "recorder_start":
            self._handle_recorder_start_step()
            return
        if step.kind == "recorder_stop":
            self._handle_recorder_stop_step()
            return
        if step.kind == "wait":
            print("-" * 20)
            print(f"Waiting for {step.wait_sec} seconds...")
            print("-" * 20)
            self._handle_wait_step(step)
            return
        if step.kind == "start-segment":
            self._current_segment_id = step.segment_id or 0
            self._publish_segment(self._current_segment_id)
            self._current_step_index += 1
            return
        if step.kind == "gripper":
            self._handle_gripper_step(step)
            return
        if step.kind != "waypoint" or step.waypoint is None:
            self._abort_with_error("Invalid step configuration; stopping node.")
            return

        self._publish_segment(self._current_segment_id)
        if self._current_joints is None:
            self._publish_joint_command([0.0] * len(self._joint_names))
            return
        if len(step.waypoint) != len(self._joint_names):
            self._abort_with_error(
                "Waypoint length does not match number of joints; stopping node."
            )
            return

        errors = [
            math.atan2(math.sin(goal - current), math.cos(goal - current))
            for current, goal in zip(self._current_joints, step.waypoint)
        ]
        max_error = max(abs(error) for error in errors)
        self._max_error = max(self._max_error, max_error)
        self._current_max_error = max_error

        if max_error < self._joint_tolerance:
            self.get_logger().info(
                f"Reached joint waypoint {self._current_step_index + 1}/{len(self._steps)}"
            )
            if step.wait_sec > 0.0:
                self._waiting_until_sec = now_sec + step.wait_sec
                self._advance_after_wait = True
                self._publish_joint_command([0.0] * len(self._joint_names))
                return
            self._apply_deviation = False
            self._current_step_index += 1
            return

        speed_scale = 0.8 if self._apply_deviation else 1.0
        velocities = []
        for error in errors:
            speed = abs(error) / max_error * self._max_joint_speed * speed_scale
            velocity = math.copysign(speed, error)
            if abs(velocity) > self._max_joint_speed > 0.0:
                velocity = math.copysign(self._max_joint_speed, velocity)
            velocities.append(velocity)
        self._publish_joint_command(velocities)

    def _reset_noise(self) -> None:
        print("Resetting velocity noise...")
        self._additive_noise = [
            random.uniform(0.1, 0.60) for _ in self._joint_names
        ]
        self._noise_direction = [random.choice([-1, 1]) for _ in self._joint_names]
        self._apply_deviation = True

    def _handle_gripper_step(self, step: Step) -> None:
        self._publish_joint_command([0.0] * len(self._joint_names))
        command_map = {"grip": 1, "release": 2, "blow": 3}
        self._set_gripper_state(command_map.get(step.gripper_command or "", 0))

        if self._gripper_future is None:
            if not self._gripper_client.wait_for_service(timeout_sec=0.1):
                self.get_logger().warn(
                    f"Waiting for gripper service at {self._gripper_service}"
                )
                return
            request = GripperControl.Request()
            request.command = step.gripper_command or ""
            self.get_logger().info(f"Starting gripper task: {request.command or 'unknown'}")
            self._gripper_wait_sec = step.wait_sec
            self._gripper_future = self._gripper_client.call_async(request)
            return
        if not self._gripper_future.done():
            return

        try:
            response = self._gripper_future.result()
        except Exception as exc:  # noqa: BLE001
            self._abort_with_error(f"Gripper service call failed: {exc}")
            return
        finally:
            self._gripper_future = None
        if not response.success:
            self._abort_with_error(f"Gripper command failed: {response.message}")
            return

        self.get_logger().info(
            f"Finished gripper task: {step.gripper_command or 'unknown'}"
        )
        if self._gripper_wait_sec > 0.0:
            self._waiting_until_sec = (
                self.get_clock().now().nanoseconds / 1e9 + self._gripper_wait_sec
            )
            self._advance_after_wait = True
        else:
            self._current_step_index += 1

    def _handle_recorder_start_step(self) -> None:
        if not self._record_enabled:
            self._current_step_index += 1
            return
        if self._recorder_start_client is None:
            self._abort_with_error("Recorder start client not initialized; stopping node.")
            return
        if not self._recorder_start_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn(
                f"Waiting for dataset recorder start service at "
                f"{self._recorder_start_service}"
            )
            return
        future = self._recorder_start_client.call_async(Trigger.Request())
        future.add_done_callback(self._handle_start_recorder)
        self._current_step_index += 1

    def _handle_recorder_stop_step(self) -> None:
        if self._recorder_stopping:
            return
        self._recorder_stopping = True
        if not self._record_enabled:
            self._current_step_index += 1
            return

        step = self._steps[self._current_step_index]
        if step.output_dir and self._recorder_param_client is not None:
            if self._recorder_param_client.wait_for_service(timeout_sec=0.1):
                parameter = Parameter(
                    "stop_output_dir", Parameter.Type.STRING, step.output_dir
                )
                request = SetParameters.Request()
                request.parameters = [parameter.to_parameter_msg()]
                self._recorder_param_client.call_async(request)
        self._request_recorder_stop()

    def _handle_wait_step(self, step: Step) -> None:
        if step.wait_sec <= 0.0:
            self._current_step_index += 1
            return
        if self._waiting_until_sec is None:
            self._waiting_until_sec = (
                self.get_clock().now().nanoseconds / 1e9 + step.wait_sec
            )
            self._advance_after_wait = True
        self._publish_joint_command([0.0] * len(self._joint_names))

    def _publish_joint_command(self, velocities: List[float]) -> None:
        noisy_velocities = self._apply_velocity_noise(velocities)

        # Both topics intentionally receive the noisy command for compatibility.
        # The raw topic can be changed to `velocities` if raw actions are needed later.
        self._joint_cmd_raw_pub.publish(self._build_joint_jog(noisy_velocities))
        self._joint_cmd_pub.publish(self._build_joint_jog(noisy_velocities))

    def _build_joint_jog(self, velocities: List[float]) -> JointJog:
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = self._joint_names
        msg.velocities = velocities
        msg.displacements = []
        msg.duration = 0.0
        return msg

    def _apply_velocity_noise(self, velocities: List[float]) -> List[float]:
        if self._velocity_noise_std <= 0.0 or not self._apply_deviation:
            return velocities

        progress = self._current_max_error / self._max_error if self._max_error > 0 else 0.0
        progress = 1.0 - min(max(progress, 0.0), 1.0)
        curve_position = np.pow(
            progress, np.log(0.5) / np.log(self._additive_noise)
        )
        noise_scale = np.pow(np.sin(np.pi * curve_position), 4)
        print(f"X={progress:.2f}, n={np.round(noise_scale, 3)}")

        noisy_velocities = []
        for index, velocity in enumerate(velocities):
            noisy_velocity = velocity + (
                0.2
                * self._max_joint_speed
                * noise_scale[index]
                * self._noise_direction[index]
            )
            if abs(noisy_velocity) > self._max_joint_speed > 0.0:
                noisy_velocity = math.copysign(self._max_joint_speed, noisy_velocity)
            noisy_velocities.append(noisy_velocity)
        return noisy_velocities


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrajectoryControl()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
