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
PULL_DISTANCE = round(random.uniform(0.10, 0.18), 4)

# Push-back happens in two steps in the original demo (0.10 then 0.04, a
# gripper "blow" in between) — keep that ratio so the drawer still lands back
# at its start depth regardless of how far PULL_DISTANCE was randomised.
PUSH_BACK_1 = round(PULL_DISTANCE * (0.10 / 0.14), 4)
PUSH_BACK_2 = round(PULL_DISTANCE - PUSH_BACK_1, 4)

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
        'home': [-90.00, 0.00, -90.00, 0.00, 90.00, -0.00],
        'approach': [-108.18, -107.64, -150.28, -5.93, 94.92, -7.26],
    },
    'steps': [
        {'type': 'Checkpoint', 'checkpoint': 'home'},
        {'type': 'Checkpoint', 'checkpoint': 'approach'},
        {'type': 'OrientationLockCheckpoint', 'frame': 'tool', 'axis': [0, 0, 1],
         'distance': PULL_DISTANCE},
        {'type': 'Wait', 'duration': 1.0},
        {'type': 'Gripper', 'action': 'grip'},
        {'type': 'Wait', 'duration': 0.2},
        {'type': 'Gripper', 'action': 'release'},
        {'type': 'OrientationLockCheckpoint', 'frame': 'tool', 'axis': [0, 0, -1],
         'distance': PUSH_BACK_1},
        {'type': 'Gripper', 'action': 'blow'},
        {'type': 'OrientationLockCheckpoint', 'frame': 'tool', 'axis': [0, 0, -1],
         'distance': PUSH_BACK_2},
        {'type': 'Checkpoint', 'checkpoint': 'home'},
    ],
    'obstacles': [
        {'id': 'table', 'size': [1.2, 1.2, 0.02], 'position': [0.0, 0.0, -0.01],
         'color': [0.6, 0.6, 0.6, 0.8]},
        {'id': 'wall', 'size': [0.02, 1.0, 1.0], 'position': [0.4, 0.0, 0.5],
         'color': [0.8, 0.2, 0.2, 0.6]},
    ],
    'recording': {
        'bag_uri': 'runs/drawer_random',
        'topics': ['/joint_states', '/tf', '/tf_static', '/robot_description'],
    },
}
