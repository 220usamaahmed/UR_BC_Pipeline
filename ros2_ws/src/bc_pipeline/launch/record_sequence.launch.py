"""
Launch the scene publisher, the sequence runner, and a bag recorder together.

Assumes the UR driver + MoveIt (move_group) + Foxglove bridge are ALREADY
running in another terminal — this launch only adds the data-collection layer.

FLOW
----
  scene_publisher  starts immediately (applies obstacles, latches markers, spins)
  recorder         starts immediately (ros2 bag record) — ONLY if the config has
                   a 'recording' section; otherwise skipped with a warning
  sequence_runner  starts ~2 s later, so the bag is open and the planning scene
                   is populated before any motion begins
  on runner exit   Shutdown() SIGINTs the recorder, which flushes + closes the
                   bag cleanly — regardless of whether the run succeeded

CONFIG
------
All three read the SAME config file.  The 'config' launch argument is a filename
in bc_pipeline/config/ (or an absolute path), and may be a .yaml file (plain
data) or a .py file (a module-level CONFIG dict, executed — lets an author use
math/random to build the config). Either way this file resolves the source
into a concrete dict EXACTLY ONCE here, dumps it to a plain resolved YAML file,
and points scene_publisher + sequence_runner at that resolved file instead of
the original path. This matters for .py sources: if each node re-executed the
Python independently, any randomness in it would give them different values
and they'd silently drift apart, which is exactly what "SAME config file" is
supposed to prevent.

    ros2 launch bc_pipeline record_sequence.launch.py config:=drawer_demo.yaml

BAG NAMING
----------
recording.bag_uri from the config is a base path; a timestamp suffix is appended
so every run is unique and nothing is ever overwritten, e.g.
    runs/drawer  ->  runs/drawer_2026-06-20_14-03-21

When recording is enabled, the resolved config is also saved as
runs/drawer_2026-06-20_14-03-21.yaml (a sibling of the bag directory, not
inside it — ros2 bag record refuses to start if its output directory already
exists) so a run with randomised values stays reproducible/debuggable later.
"""

import datetime
import os
import shutil
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory

from bc_pipeline.config_loader import resolve_source

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
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

    # Resolve the source (.yaml data, or .py CONFIG executed) into a plain dict
    # exactly once, then dump it to its own resolved YAML file. scene_publisher
    # and sequence_runner both read THAT file rather than config_path, so a .py
    # source with randomness in it can't give the two nodes different values.
    config = resolve_source(config_path)

    resolved_fd, resolved_path = tempfile.mkstemp(
        prefix='bc_pipeline_resolved_', suffix='.yaml'
    )
    with os.fdopen(resolved_fd, 'w') as f:
        yaml.safe_dump(config, f)

    # Recording is optional. If the config has no 'recording' section, skip the
    # bag recorder entirely and warn — the sequence still runs, just unrecorded.
    recording = config.get('recording')

    scene = Node(
        package='bc_pipeline',
        executable='scene_publisher',
        output='screen',
        parameters=[{'config': resolved_path}],
    )

    runner = Node(
        package='bc_pipeline',
        executable='sequence_runner',
        output='screen',
        parameters=[{'config': resolved_path}],
    )

    # Delay the runner so the recorder has the bag open and the scene is applied
    # before any motion produces messages.
    delayed_runner = TimerAction(period=STARTUP_DELAY_SEC, actions=[runner])

    # When the runner exits (all steps done, or aborted), stop everything.
    # Shutdown() SIGINTs the recorder, which flushes + closes the bag cleanly.
    stop_on_done = RegisterEventHandler(
        OnProcessExit(target_action=runner, on_exit=[Shutdown()])
    )

    actions = [scene, delayed_runner, stop_on_done]

    if recording is None:
        actions.append(LogInfo(
            msg='[record_sequence] WARNING: no "recording" section in config — '
                'recording is DISABLED; the sequence will run without saving a bag.'
        ))
    else:
        stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        bag_uri = f"{recording['bag_uri']}_{stamp}"
        topics = list(recording['topics']) + ['/bc_pipeline/current_step']
        recorder = ExecuteProcess(
            cmd=['ros2', 'bag', 'record',
                 '--output', bag_uri, *topics],
            output='screen',
        )
        actions.append(recorder)

        # Save the exact resolved config next to the bag (a sibling file, not
        # inside the bag directory — ros2 bag record errors if its output
        # directory already exists), so a run with randomised values stays
        # reproducible/debuggable later.
        resolved_copy = f"{bag_uri}.yaml"
        os.makedirs(os.path.dirname(resolved_copy) or '.', exist_ok=True)
        shutil.copyfile(resolved_path, resolved_copy)
        actions.append(LogInfo(
            msg=f'[record_sequence] Resolved config saved to {resolved_copy}'
        ))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'config',
            default_value='drawer_demo.yaml',
            description='Config filename (.yaml or .py) in bc_pipeline/config/ (or an absolute path).',
        ),
        OpaqueFunction(function=launch_setup),
    ])
