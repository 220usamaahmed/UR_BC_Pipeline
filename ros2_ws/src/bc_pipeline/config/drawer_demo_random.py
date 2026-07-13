"""
Same experiment as drawer_demo.yaml, but in Python format for data augmentation.

This is the .py convention: a module-level CONFIG dict, built with whatever
Python you need (math, random, loops, ...). It's executed once by the launch
file, which resolves it into a concrete YAML before any node reads it — see
record_sequence.launch.py.
"""

import math
import random

HOME_BASE = [-90.00, 0.00, -90.00, 0.00, 90.00, -0.00]
HOME_NOISY = [round(angle + random.uniform(-2, 2), 2) for angle in HOME_BASE]

# Random offset within a 2cm radius circle (uniform distribution)
APPROACH_ANGLE = random.uniform(0, 2 * math.pi)
APPROACH_DISTANCE = math.sqrt(random.uniform(0, 1)) * 0.02
APPROACH_DIRECTION = [math.cos(APPROACH_ANGLE), 0, math.sin(APPROACH_ANGLE)]

# Randomized push-back: move back 12cm + random up to 1cm, then complete to 14cm total
PULL_BACK_1 = round(0.12 + random.uniform(0, 0.01), 4)
PULL_BACK_2 = round(0.14 - PULL_BACK_1, 4)

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
        'accel_scaling': 0.2,
        'planning_time': 5.0,
    },
    'checkpoints': {
        'home': HOME_NOISY,
        'approach': [-108.18, -107.64, -150.28, -5.93, 94.92, -7.26],
    },
    'steps': [
        {'type': 'Checkpoint', 'checkpoint': 'home'},
        {'type': 'Checkpoint', 'checkpoint': 'approach', 'cartesian_offset': {'direction': APPROACH_DIRECTION, 'distance': APPROACH_DISTANCE}},
        {'type': 'OrientationLockCheckpoint', 'frame': 'tool', 'axis': [0, 0, 1],
         'distance': 0.14},
        # {'type': 'Wait', 'duration': 0.2},
        {'type': 'Gripper', 'action': 'grip'},
        # {'type': 'Wait', 'duration': 0.2},
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
        {'id': 'drawer', 'size': [0.6, 0.4, 0.08], 'position': [0.0, 0.30, 0.05],
         'color': [0.2, 0.5, 0.8, 0.6]},
    ],
    # 'recording': {
    #     'bag_uri': 'runs/drawer',
    #     'topics': [
    #         '/joint_states',
    #         '/tf',
    #         '/tf_static',
    #         '/robot_description',
    #         '/zed/zed_node/depth/depth_registered',
    #     ],
    # },
}
