"""
Launch the scene publisher, the sequence runner, and a bag recorder together.

Assumes the UR driver + MoveIt (move_group) + Foxglove bridge are ALREADY
running in another terminal — this launch only adds the data-collection layer.

FLOW
----
  scene_publisher  starts immediately (applies obstacles, latches markers, spins)
  recorder         starts immediately (ros2 bag record)
  sequence_runner  starts ~2 s later, so the bag is open and the planning scene
                   is populated before any motion begins
  on runner exit   Shutdown() SIGINTs the recorder, which flushes + closes the
                   bag cleanly — regardless of whether the run succeeded

CONFIG
------
All three read the SAME config file.  The 'config' launch argument is a filename
in bc_pipeline/config/ (or an absolute path):

    ros2 launch bc_pipeline record_sequence.launch.py config:=drawer_demo.yaml

BAG NAMING
----------
recording.bag_uri from the config is a base path; a timestamp suffix is appended
so every run is unique and nothing is ever overwritten, e.g.
    runs/drawer  ->  runs/drawer_2026-06-20_14-03-21
"""

import datetime
import os

import yaml
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

STARTUP_DELAY_SEC = 2.0


def _resolve_config_path(config_arg: str) -> str:
    if os.path.isabs(config_arg):
        return config_arg
    return os.path.join(
        get_package_share_directory('bc_pipeline'), 'config', config_arg
    )


def launch_setup(context, *args, **kwargs):
    config_path = _resolve_config_path(
        LaunchConfiguration('config').perform(context)
    )

    with open(config_path) as f:
        config = yaml.safe_load(f)
    recording = config['recording']

    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    bag_uri = f"{recording['bag_uri']}_{stamp}"

    recorder = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '--output', bag_uri, *recording['topics']],
        output='screen',
    )

    scene = Node(
        package='bc_pipeline',
        executable='scene_publisher',
        output='screen',
        parameters=[{'config': config_path}],
    )

    runner = Node(
        package='bc_pipeline',
        executable='sequence_runner',
        output='screen',
        parameters=[{'config': config_path}],
    )

    # Delay the runner so the recorder has the bag open and the scene is applied
    # before any motion produces messages.
    delayed_runner = TimerAction(period=STARTUP_DELAY_SEC, actions=[runner])

    # When the runner exits (all steps done, or aborted), stop everything.
    # Shutdown() SIGINTs the recorder, which flushes + closes the bag cleanly.
    stop_on_done = RegisterEventHandler(
        OnProcessExit(target_action=runner, on_exit=[Shutdown()])
    )

    return [scene, recorder, delayed_runner, stop_on_done]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'config',
            default_value='drawer_demo.yaml',
            description='Config YAML filename in bc_pipeline/config/ (or an absolute path).',
        ),
        OpaqueFunction(function=launch_setup),
    ])
