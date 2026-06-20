#!/usr/bin/env python3
"""
Wait — pause for a fixed number of seconds.

We sleep by spinning the node in short slices rather than time.sleep() so the
node keeps processing callbacks (TF, discovery) while it waits.

YAML
----
  - type: Wait
    duration: 2.0
"""

import rclpy
import rclpy.duration

from .base import Step, StepConfigError, register


@register('Wait')
class Wait(Step):
    def validate(self):
        self.duration = float(self._require('duration'))
        if self.duration < 0.0:
            raise StepConfigError("Wait duration must be non-negative.")

    @property
    def label(self) -> str:
        return f"Wait {self.duration} s"

    def execute(self) -> bool:
        self.ctx.logger.info(f"Waiting {self.duration} s …")
        end = self.ctx.node.get_clock().now() + rclpy.duration.Duration(
            seconds=self.duration
        )
        while self.ctx.node.get_clock().now() < end:
            rclpy.spin_once(self.ctx.node, timeout_sec=0.1)
        return True
