#!/usr/bin/env python3
"""
OrientationLockCheckpoint — slide the EEF in a straight line, orientation fixed.

The end effector keeps its current orientation while translating along a
direction for a given distance.  This is what a suction gripper needs when
pulling a drawer: the gripper face stays parallel to the surface throughout.

Orientation is locked *by construction*: we hand MoveIt's GetCartesianPath a
list of waypoints that all share the start orientation, so it can never drift
(unlike an OrientationConstraint, which OMPL treats as mere guidance).

YAML
----
  - type: OrientationLockCheckpoint
    frame: tool        # 'tool' = axis is in the EEF frame; 'base' = world frame
    axis: [0, 0, -1]   # direction to slide (need not be unit; it's normalised)
    distance: 0.10     # metres
    max_step: 0.005    # optional — Cartesian IK resolution
    min_fraction: 0.95 # optional — reject plan if less is reachable

frame: tool with axis [0,0,-1] reproduces the drawer-pull (−Z of tool0).
frame: base with axis [0,0,1] means "lift 'distance' metres straight up".
"""

import math

from geometry_msgs.msg import Point, Pose, Quaternion

from .base import Step, StepConfigError, register


@register('OrientationLockCheckpoint')
class OrientationLockCheckpoint(Step):
    def validate(self):
        self.frame = self.cfg.get('frame', 'tool')
        if self.frame not in ('tool', 'base'):
            raise StepConfigError(
                f"OrientationLockCheckpoint frame must be 'tool' or 'base'; "
                f"got '{self.frame}'."
            )
        axis = self._require('axis')
        if not isinstance(axis, list) or len(axis) != 3:
            raise StepConfigError(
                f"OrientationLockCheckpoint axis must be a list of 3 numbers; "
                f"got {axis!r}."
            )
        norm = math.sqrt(sum(float(a) ** 2 for a in axis))
        if norm == 0.0:
            raise StepConfigError("OrientationLockCheckpoint axis must be non-zero.")
        self.axis = [float(a) / norm for a in axis]   # unit direction

        self.distance = float(self._require('distance'))
        self.max_step = float(self.cfg.get('max_step', 0.005))
        self.min_fraction = float(self.cfg.get('min_fraction', 0.95))

    @property
    def label(self) -> str:
        return f"OrientationLock({self.frame} {self.axis}, {self.distance} m)"

    def execute(self) -> bool:
        start = self.ctx.get_eef_pose()
        if start is None:
            return False

        q = start.orientation
        if self.frame == 'tool':
            dx, dy, dz = _rotate_vector(q, self.axis)
        else:
            dx, dy, dz = self.axis

        self.ctx.logger.info(
            f"Sliding {self.distance} m along ({dx:.3f}, {dy:.3f}, {dz:.3f}) "
            f"in base frame, orientation locked."
        )

        waypoints = self._build_waypoints(start, (dx, dy, dz))
        return self.ctx.execute_cartesian(
            self.label, waypoints, self.max_step, self.min_fraction
        )

    def _build_waypoints(self, start: Pose, direction: tuple) -> list:
        """Evenly-spaced poses along direction, all sharing start's orientation."""
        dx, dy, dz = direction
        q = start.orientation
        num_steps = max(2, round(self.distance / self.max_step))
        waypoints = []
        for i in range(num_steps + 1):
            t = (i / num_steps) * self.distance
            pose = Pose()
            pose.position = Point(
                x=start.position.x + dx * t,
                y=start.position.y + dy * t,
                z=start.position.z + dz * t,
            )
            # Identical orientation at every step — this is the orientation lock.
            pose.orientation = Quaternion(x=q.x, y=q.y, z=q.z, w=q.w)
            waypoints.append(pose)
        return waypoints


def _rotate_vector(q, v) -> tuple:
    """
    Rotate a 3-vector v by quaternion q (x, y, z, w) → vector in the world frame.

    Uses the efficient form  v' = v + 2w·(u×v) + 2·u×(u×v)  where u = (qx,qy,qz).
    For an EEF-frame axis this expresses it in the base frame.
    """
    ux, uy, uz = q.x, q.y, q.z
    w = q.w
    vx, vy, vz = v

    # t = 2 * (u × v)
    tx = 2.0 * (uy * vz - uz * vy)
    ty = 2.0 * (uz * vx - ux * vz)
    tz = 2.0 * (ux * vy - uy * vx)

    # v' = v + w·t + (u × t)
    rx = vx + w * tx + (uy * tz - uz * ty)
    ry = vy + w * ty + (uz * tx - ux * tz)
    rz = vz + w * tz + (ux * ty - uy * tx)
    return rx, ry, rz
