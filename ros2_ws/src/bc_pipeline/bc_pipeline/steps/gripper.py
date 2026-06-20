#!/usr/bin/env python3
"""
Gripper — placeholder for suction-gripper control.

For now this only logs the requested action and waits a short settle time, so
sequences can already include gripper steps in the right places.  When the real
suction hardware arrives, only execute() changes — the YAML schema below is the
final one, so existing sequences keep working unchanged.

YAML
----
  - type: Gripper
    action: on        # on | off
    settle: 0.5       # optional — seconds to wait after switching

NOTE: YAML parses bare on/off as booleans (YAML 1.1), so we accept both the
strings 'on'/'off' and booleans true/false.
"""

import rclpy
import rclpy.duration

from .base import Step, StepConfigError, register


@register('Gripper')
class Gripper(Step):
    def validate(self):
        raw = self._require('action')
        if isinstance(raw, bool):
            self.action = 'on' if raw else 'off'   # YAML coerced on/off → bool
        elif isinstance(raw, str) and raw.lower() in ('on', 'off'):
            self.action = raw.lower()
        else:
            raise StepConfigError(
                f"Gripper action must be 'on' or 'off'; got {raw!r}."
            )
        self.settle = float(self.cfg.get('settle', 0.5))

    @property
    def label(self) -> str:
        return f"Gripper {self.action}"

    def execute(self) -> bool:
        self.ctx.logger.info(f"Gripper → {self.action} (placeholder, no hardware yet)")
        end = self.ctx.node.get_clock().now() + rclpy.duration.Duration(
            seconds=self.settle
        )
        while self.ctx.node.get_clock().now() < end:
            rclpy.spin_once(self.ctx.node, timeout_sec=0.1)
        return True
