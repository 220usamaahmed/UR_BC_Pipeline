#!/usr/bin/env python3
"""
Checkpoint — joint-space move to a named checkpoint.

The checkpoint name must exist in the config's 'checkpoints' table; an unknown
name is a hard error at load time (fail fast, before any motion).  Joint values
live only in the table so they stay reusable across many steps.

CARTESIAN OFFSET
----------------
An optional 'cartesian_offset' makes the target the checkpoint's *pose* shifted
by a vector in metres, reached in a single motion — a fused Checkpoint +
OrientationLockCheckpoint rather than two separate steps:

  - type: Checkpoint
    checkpoint: approach
    cartesian_offset:
      direction: [0, 0, -1]   # base_frame vector (need not be unit; normalised)
      distance: 0.10          # metres

Because the checkpoint is defined in joint space but the offset is in metres, we
ask MoveIt for the forward kinematics of the checkpoint angles (the pose those
angles produce), add the offset to that pose's position, keep its orientation,
and plan one free motion straight to that pose goal.  The arm therefore goes
*directly* to the offset pose; it never stops at the un-offset checkpoint.

The direction is interpreted in robot.base_frame (the world frame), so e.g.
direction [0,0,1] with distance 0.1 means "the checkpoint pose lifted 10 cm".
"""

import math

from .base import Step, StepConfigError, register


@register('Checkpoint')
class Checkpoint(Step):
    def validate(self):
        self.name = self._require('checkpoint')
        if self.name not in self.ctx.checkpoints:
            raise StepConfigError(
                f"Checkpoint '{self.name}' is not defined in the checkpoints "
                f"table. Defined: {sorted(self.ctx.checkpoints)}."
            )
        self.angles = self.ctx.checkpoints[self.name]
        self.offset = self._parse_offset(self.cfg.get('cartesian_offset'))

    def _parse_offset(self, raw):
        """Validate the optional cartesian_offset → (unit_direction, distance) or None."""
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise StepConfigError(
                "Checkpoint cartesian_offset must be a mapping with 'direction' "
                f"and 'distance'; got {raw!r}."
            )
        direction = raw.get('direction')
        if not isinstance(direction, list) or len(direction) != 3:
            raise StepConfigError(
                "Checkpoint cartesian_offset.direction must be a list of 3 "
                f"numbers; got {direction!r}."
            )
        norm = math.sqrt(sum(float(d) ** 2 for d in direction))
        if norm == 0.0:
            raise StepConfigError(
                "Checkpoint cartesian_offset.direction must be non-zero."
            )
        if 'distance' not in raw:
            raise StepConfigError(
                "Checkpoint cartesian_offset is missing 'distance'."
            )
        distance = float(raw['distance'])
        unit = [float(d) / norm for d in direction]
        return unit, distance

    @property
    def label(self) -> str:
        if self.offset is None:
            return f"Checkpoint '{self.name}'"
        unit, distance = self.offset
        return f"Checkpoint '{self.name}' + offset {distance} m along {unit}"

    def execute(self) -> bool:
        if self.offset is None:
            self.ctx.logger.info(f"Moving to checkpoint '{self.name}' → {self.angles}")
            return self.ctx.plan_and_execute_joints(self.angles)

        unit, distance = self.offset
        pose = self.ctx.compute_fk(self.angles)
        if pose is None:
            return False

        pose.position.x += unit[0] * distance
        pose.position.y += unit[1] * distance
        pose.position.z += unit[2] * distance

        self.ctx.logger.info(
            f"Moving to checkpoint '{self.name}' + offset {distance} m along "
            f"{unit} → pose target "
            f"({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})"
        )
        return self.ctx.plan_and_execute_pose(pose)
