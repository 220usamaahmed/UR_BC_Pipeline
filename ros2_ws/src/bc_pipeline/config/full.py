"""
This is the .py convention: a module-level CONFIG dict, built with whatever
Python you need (math, random, loops, ...). It's executed once by the launch
file, which resolves it into a concrete YAML before any node reads it — see
record_sequence.launch.py.
"""

import math
import random

HOME_BASE = [-90.00, 0.00, -90.00, 0.00, 90.00, -0.00]
HOME_NOISY_1 = [round(angle + random.uniform(-6, 6), 2) for angle in HOME_BASE]
HOME_NOISY_2 = [round(angle + random.uniform(-6, 6), 2) for angle in HOME_BASE]
HOME_NOISY_3 = [round(angle + random.uniform(-6, 6), 2) for angle in HOME_BASE]

# Random offset within a 2cm radius circle (uniform distribution)
# APROACH = [-108.18, -107.64, -150.28, -5.93, 94.92, -7.26] # Left
APROACH = [-70.59, 105.89, -33.46, 5.91, 82.44, 6.08] # Right

APPROACH_ANGLE = random.uniform(0, 2 * math.pi)
APPROACH_DISTANCE = math.sqrt(random.uniform(0, 1)) * 0.03
APPROACH_DIRECTION = [math.cos(APPROACH_ANGLE), 0, math.sin(APPROACH_ANGLE)]

# Randomized push-back: move back 12cm + random up to 1cm, then complete to 14cm total
PULL_BACK_1 = round(0.12 + random.uniform(0, 0.01), 4)
PULL_BACK_2 = round(0.14 - PULL_BACK_1, 4)

PICK = [-120.90, -67.13, -81.87, 90.08, 88.17, -72.87]
PICK_APPROACH_OFFSET = random.uniform(0.02, 0.06)

# PLACE = [-124.29, -26.15, -108.33, 85.12, 94.66, -44.52] # Left
PLACE = [-63.90, 44.47, -86.37, -83.70, 90.23, 42.25] # Right
PLACE_NOISY = [round(angle + random.uniform(-1, 1), 2) for angle in PLACE]

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
        'max_ik_joint_deviation': 1.5708,
    },
    'checkpoints': {
        'home': HOME_BASE,
        'home_noisy_1': HOME_NOISY_1,
        'home_noisy_2': HOME_NOISY_2,
        'home_noisy_3': HOME_NOISY_3,
        'approach': APROACH,
        'pick': PICK,
        'place_noisy': PLACE_NOISY,
    },
    'steps': [
        # Reset to home
        {'type': 'Wait', 'duration': 1.0, 'ignore': True},
        {'type': 'Checkpoint', 'checkpoint': 'home_noisy_1', 'ignore': True},

        # Open drawer
        {'type': 'Checkpoint', 'checkpoint': 'approach', 'ignore': False, 'cartesian_offset': {'direction': APPROACH_DIRECTION, 'distance': APPROACH_DISTANCE}},
        {'type': 'OrientationLockCheckpoint', 'frame': 'tool', 'axis': [0, 0, 1],
         'distance': 0.14, 'ignore': False},
        {'type': 'Gripper', 'action': 'grip', 'ignore': False},
        {'type': 'Gripper', 'action': 'release', 'ignore': False},
        {'type': 'OrientationLockCheckpoint', 'frame': 'tool', 'axis': [0, 0, -1],
         'distance': PULL_BACK_1, 'ignore': False},
        {'type': 'Gripper', 'action': 'blow', 'ignore': False},
        {'type': 'OrientationLockCheckpoint', 'frame': 'tool', 'axis': [0, 0, -1],
         'distance': PULL_BACK_2, 'ignore': False},
        {'type': 'Checkpoint', 'checkpoint': 'home_noisy_2', 'ignore': False},

        # Pick up object
        {'type': 'Checkpoint', 'checkpoint': 'pick', 'ignore': False, 'cartesian_offset': {'direction': [0, 0, -1], 'distance': -PICK_APPROACH_OFFSET}},
        {'type': 'OrientationLockCheckpoint', 'frame': 'tool', 'axis': [0, 0, 1],
            'distance': PICK_APPROACH_OFFSET, 'ignore': False},
        {'type': 'Gripper', 'action': 'grip', 'ignore': False},
        {'type': 'Gripper', 'action': 'release', 'ignore': False},
        {'type': 'Checkpoint', 'checkpoint': 'home_noisy_3', 'ignore': False},

        # Place object back
        {'type': 'Checkpoint', 'checkpoint': 'place_noisy', 'ignore': False},
        {'type': 'Gripper', 'action': 'blow', 'ignore': False},
        {'type': 'Checkpoint', 'checkpoint': 'home', 'ignore': False},

        {'type': 'Wait', 'duration': 1.0, 'ignore': True},
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
    'recording': {
        'bag_uri': '/root/ros2_ws/trajectories/full/full',
        'topics': [
            '/joint_states',
            '/tf',
            '/tf_static',
            '/robot_description',
            '/zed/zed_node/depth/depth_registered',
        ],
    },
}
