#!/usr/bin/env python3
"""
Shared obstacle definitions — the single source of truth for both:

  • scene_builder   — puts these into MoveIt's planning scene (what the
                      planner AVOIDS), and
  • obstacle_markers — publishes these as Foxglove markers (what you SEE).

Keeping the geometry in one place guarantees the visualization matches reality.
If these two ever drift apart, your Foxglove view is lying to you about what the
planner knows.

Both MoveIt's SolidPrimitive BOX and visualization_msgs/Marker CUBE use full
side lengths (not half-extents) and a centre pose, so one spec feeds both.
"""

# All geometry is defined in this frame.  For UR robots base_link is the
# mounting surface; z=0 is the base.
WORLD_FRAME = 'base_link'

OBSTACLES = [
    {
        'id': 'table',
        'size': (1.2, 1.2, 0.02),        # x, y, z side lengths [m]
        'position': (0.0, 0.0, -0.01),   # centre; top face flush with z=0
        'color': (0.6, 0.6, 0.6, 0.8),   # grey (r, g, b, a) — markers only
    },
    {
        'id': 'wall',
        'size': (0.02, 1.0, 1.0),        # thin in X, wide in Y and Z
        'position': (0.4, 0.0, 0.5),     # ~arm-length in front of the robot
        'color': (0.8, 0.2, 0.2, 0.6),   # semi-transparent red — markers only
    },
]
