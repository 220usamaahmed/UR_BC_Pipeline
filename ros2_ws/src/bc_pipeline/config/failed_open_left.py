"""
This is the .py convention: a module-level CONFIG dict, built with whatever
Python you need (math, random, loops, ...). It's executed once by the launch
file, which resolves it into a concrete YAML before any node reads it — see
record_sequence.launch.py.
"""

import math
import random

HOME_BASE = [-90.00, 0.00, -90.00, 0.00, 90.00, -0.00]
HOME_NOISY = [round(angle + random.uniform(-6, 6), 2) for angle in HOME_BASE]

APPROACH = [-108.18, -107.64, -150.28, -5.93, 94.92, -7.26] # Left
APPROACH_RANDOM = [round(angle + random.uniform(-2, 2), 2) for angle in APPROACH]

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
        'approach_random': APPROACH_RANDOM,
    },
    'steps': [
        {'type': 'Wait', 'duration': 1.0, 'ignore': True},
        {'type': 'Checkpoint', 'checkpoint': 'home_noisy', 'ignore': True},
        {'type': 'Checkpoint', 'checkpoint': 'approach_random', 'ignore': False},

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
        'bag_uri': '/root/ros2_ws/trajectories/open_left_failed/open_left_failed',
        'topics': [
            '/joint_states',
            '/tf',
            '/tf_static',
            '/robot_description',
            '/zed/zed_node/depth/depth_registered',
        ],
    },
}
