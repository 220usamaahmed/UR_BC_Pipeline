#!/usr/bin/env python3
"""
Gripper — drives the ECPMi suction gripper via ecpmi_gripper's
`gripper_control` service (see ros2_ws/src/ecpmi_gripper).

Requires the real robot with the UR driver's IO controller active and
`ros2 launch ecpmi_gripper suction_gripper.launch.py` running — the mock
driver doesn't expose tool digital outputs, so this step fails fast (and
aborts the sequence) if the service isn't up.

YAML
----
  - type: Gripper
    action: grip      # grip | release | blow
    settle: 0.5        # optional — seconds to wait after the service call
"""

import rclpy
import rclpy.duration

from .base import Step, StepConfigError, register

VALID_ACTIONS = ('grip', 'release', 'blow')


@register('Gripper')
class Gripper(Step):
    def validate(self):
        raw = self._require('action')
        if not isinstance(raw, str) or raw.lower() not in VALID_ACTIONS:
            raise StepConfigError(
                f"Gripper action must be one of {VALID_ACTIONS}; got {raw!r}."
            )
        self.action = raw.lower()
        self.settle = float(self.cfg.get('settle', 0.5))

    @property
    def label(self) -> str:
        return f"Gripper {self.action}"

    def execute(self) -> bool:
        self.ctx.logger.info(f"Gripper → {self.action}")
        success, message = self.ctx.call_gripper(self.action)
        if not success:
            self.ctx.logger.error(f"Gripper {self.action} failed: {message}")
            return False
        self.ctx.logger.info(f"Gripper {self.action}: {message}")

        end = self.ctx.node.get_clock().now() + rclpy.duration.Duration(
            seconds=self.settle
        )
        while self.ctx.node.get_clock().now() < end:
            rclpy.spin_once(self.ctx.node, timeout_sec=0.1)
        return True
