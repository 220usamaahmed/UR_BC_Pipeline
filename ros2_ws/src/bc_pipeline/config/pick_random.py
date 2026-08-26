"""
Same experiment as drawer_demo.yaml, but in Python format for data augmentation.

This is the .py convention: a module-level CONFIG dict, built with whatever
Python you need (math, random, loops, ...). It's executed once by the launch
file, which resolves it into a concrete YAML before any node reads it — see
record_sequence.launch.py.
"""

import random

HOME_BASE = [-90.00, 0.00, -90.00, 0.00, 90.00, -0.00]
HOME_NOISY = [round(angle + random.uniform(-5, 5), 2) for angle in HOME_BASE]

# Pick Center
PICK = [-120.90, -67.13, -81.87, 90.08, 88.17, -72.87]
# [-59.57, 67.02, -97.55, -89.58, 88.09, 72.35]

PICK_APPROACH_OFFSET = random.uniform(0.02, 0.06)

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
        'home_noisy': HOME_NOISY,
        'pick': PICK,
    },
    'steps': [
        {'type': 'Checkpoint', 'checkpoint': 'home_noisy', 'ignore': True},
        {'type': 'Checkpoint', 'checkpoint': 'pick', 'ignore': False, 'cartesian_offset': {'direction': [0, 0, -1], 'distance': -PICK_APPROACH_OFFSET}},
        {'type': 'OrientationLockCheckpoint', 'frame': 'tool', 'axis': [0, 0, 1],
         'distance': PICK_APPROACH_OFFSET, 'ignore': False},
        {'type': 'Gripper', 'action': 'grip', 'ignore': False},
        {'type': 'Gripper', 'action': 'release', 'ignore': False},
        {'type': 'Checkpoint', 'checkpoint': 'home', 'ignore': False},
    ],
    'obstacles': [
        {'id': 'table', 'size': [1.2, 1.2, 0.02], 'position': [0.0, 0.0, -0.01],
         'color': [0.6, 0.6, 0.6, 0.8]},
        {'id': 'wall', 'size': [0.02, 1.0, 1.0], 'position': [0.4, 0.0, 0.5],
         'color': [0.8, 0.2, 0.2, 0.6]},
        {'id': 'side-wall', 'size': [0.02, 0.5, 0.5], 'position': [0.15, 0.25, 0.8],
         'color': [0.8, 0.2, 0.5, 0.6]},
        {'id': 'back-wall', 'size': [0.6, 0.02, 1.0], 'position': [0.0, -0.1, 0.5],
         'color': [0.8, 0.2, 0.2, 0.6]},
        {'id': 'drawer', 'size': [0.6, 0.4, 0.08], 'position': [0.0, 0.30, 0.05],
         'color': [0.2, 0.5, 0.8, 0.6]},
    ],
    'recording': {
        'bag_uri': '/root/ros2_ws/trajectories/pick/pick',
        'topics': [
            '/joint_states',
            '/tf',
            '/tf_static',
            '/robot_description',
            '/zed/zed_node/depth/depth_registered',
        ],
    },
}
