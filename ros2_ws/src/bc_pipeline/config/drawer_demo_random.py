"""
Same experiment as drawer_demo.yaml, but the drawer-pull distance is randomised
per run (data augmentation for BC) and the push-back is derived from it with
plain arithmetic, so the drawer always ends up back at its start depth.

This is the .py convention: a module-level CONFIG dict, built with whatever
Python you need (math, random, loops, ...). It's executed once by the launch
file, which resolves it into a concrete YAML before any node reads it — see
record_sequence.launch.py.
"""

import random

# Metres. Same range the hand-authored demo used as its single fixed value.
MOVE_FORWARD_DISTANCE = 0.1

# Push-back happens in two steps in the original demo (0.10 then 0.04, a
# gripper "blow" in between) — keep that ratio so the drawer still lands back
# at its start depth regardless of how far MOVE_FORWARD_DISTANCE was randomised.
PULL_BACK_1 = round(MOVE_FORWARD_DISTANCE * (0.12 / 0.14), 4)
PULL_BACK_2 = round(MOVE_FORWARD_DISTANCE - PULL_BACK_1, 4)

# Add 5 degree noise to home checkpoint joints
HOME_BASE = [-90.00, 0.00, -90.00, 0.00, 90.00, -0.00]
# HOME_NOISY = [round(angle + random.uniform(-5, 5), 2) for angle in HOME_BASE]
HOME_NOISY = HOME_BASE  # No noise for now, to make the demo more repeatable

# Random cartesian offset in x direction (±3 cm max)
APPROACH_OFFSET_X = round(random.uniform(-0.03, 0.03), 4)
APPROACH_OFFSET_X = 0.0  # No offset for now, to make the demo more repeatable

CONFIG = {
    'robot': {
        'planning_group': 'ur_manipulator',
        'eef_link': 'tool0',
        'base_frame': 'base_link',
        'joint_names': [
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
            'shoulder_pan_joint',
        ],
    },
    'planning': {
        'velocity_scaling': 0.2,
        'accel_scaling': 0.5,
        'planning_time': 5.0,
    },
    'checkpoints': {
        'home': HOME_NOISY,
        'approach': [-108.18, -107.64, -150.28, -5.93, 94.92, -7.26],
    },
    'steps': [
        {'type': 'Checkpoint', 'checkpoint': 'home'},
        {'type': 'Checkpoint', 'checkpoint': 'approach', 'cartesian_offset': {'direction': [1, 0, 0], 'distance': APPROACH_OFFSET_X}},
        {'type': 'OrientationLockCheckpoint', 'frame': 'tool', 'axis': [0, 0, 1],
         'distance': MOVE_FORWARD_DISTANCE},
        {'type': 'Wait', 'duration': 1.0},
        {'type': 'Gripper', 'action': 'grip'},
        {'type': 'Wait', 'duration': 0.2},
        {'type': 'Gripper', 'action': 'release'},
        {'type': 'OrientationLockCheckpoint', 'frame': 'tool', 'axis': [0, 0, -1],
         'distance': PULL_BACK_1},
        {'type': 'Gripper', 'action': 'blow'},
        {'type': 'OrientationLockCheckpoint', 'frame': 'tool', 'axis': [0, 0, -1],
         'distance': PULL_BACK_2},
        {'type': 'Checkpoint', 'checkpoint': 'home'},
    ],
    'obstacles': [
        {'id': 'table', 'size': [1.2, 1.2, 0.02], 'position': [0.0, 0.0, -0.01],
         'color': [0.6, 0.6, 0.6, 0.8]},
        {'id': 'wall', 'size': [0.02, 1.0, 1.0], 'position': [0.4, 0.0, 0.5],
         'color': [0.8, 0.2, 0.2, 0.6]},
        {'id': 'back-wall', 'size': [0.6, 0.02, 1.0], 'position': [0.0, -0.1, 0.5],
         'color': [0.8, 0.2, 0.2, 0.6]},
    ],
    # 'recording': {
    #     'bag_uri': 'runs/drawer_random',
    #     'topics': ['/joint_states', '/tf', '/tf_static', '/robot_description'],
    # },
}
