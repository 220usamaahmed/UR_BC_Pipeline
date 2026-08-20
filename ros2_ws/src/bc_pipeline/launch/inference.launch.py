"""Launch policy inference and an indexed MCAP debug recording together."""

import datetime
import os

from ament_index_python.packages import get_package_share_directory

from bc_pipeline.inference_debug import (
    CANONICAL_JOINT_TOPIC,
    EVENT_TOPIC,
    MATCHED_TARGET_TOPIC,
    PHASE_TOPIC,
    PLANNED_TARGET_TOPIC,
    SELECTED_DEPTH_TOPIC,
    SELECTED_JOINT_TOPIC,
)
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
DEPTH_TOPIC = '/zed/zed_node/depth/depth_registered'
CONTROLLER_STATE_TOPIC = (
    '/scaled_joint_trajectory_controller/controller_state'
)


def launch_setup(context, *args, **kwargs):
    checkpoint_path = LaunchConfiguration('checkpoint_path').perform(context)
    if not checkpoint_path:
        raise RuntimeError('checkpoint_path launch argument is required')

    bag_base = LaunchConfiguration('bag_uri').perform(context)
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    bag_uri = f'{bag_base}_{stamp}'
    run_id = os.path.basename(os.path.normpath(bag_uri))
    os.makedirs(os.path.dirname(bag_uri) or '.', exist_ok=True)

    package_share = get_package_share_directory('bc_pipeline')
    mcap_config = os.path.join(
        package_share, 'config', 'mcap_writer_options.yaml'
    )
    params_file = LaunchConfiguration('params_file').perform(context)
    parameters = []
    if params_file:
        if not os.path.isfile(params_file):
            raise RuntimeError(f'params_file does not exist: {params_file}')
        parameters.append(params_file)
    parameters.append({
        'checkpoint_path': checkpoint_path,
        'debug_enabled': True,
        'run_id': run_id,
    })

    topics = [
        '/joint_states',
        DEPTH_TOPIC,
        CONTROLLER_STATE_TOPIC,
        EVENT_TOPIC,
        PHASE_TOPIC,
        CANONICAL_JOINT_TOPIC,
        PLANNED_TARGET_TOPIC,
        MATCHED_TARGET_TOPIC,
        SELECTED_JOINT_TOPIC,
        SELECTED_DEPTH_TOPIC,
    ]
    recorder = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '--storage', 'mcap',
            '--storage-config-file', mcap_config,
            '--output', bag_uri,
            *topics,
        ],
        output='screen',
    )
    inference = Node(
        package='bc_pipeline',
        executable='inference',
        output='screen',
        parameters=parameters,
    )
    canonicalizer = Node(
        package='bc_pipeline',
        executable='joint_state_canonicalizer',
        output='screen',
    )
    delayed_inference = TimerAction(
        period=STARTUP_DELAY_SEC,
        actions=[inference],
    )
    stop_on_inference_exit = RegisterEventHandler(
        OnProcessExit(target_action=inference, on_exit=[Shutdown()])
    )
    stop_on_recorder_exit = RegisterEventHandler(
        OnProcessExit(target_action=recorder, on_exit=[Shutdown()])
    )
    stop_on_canonicalizer_exit = RegisterEventHandler(
        OnProcessExit(target_action=canonicalizer, on_exit=[Shutdown()])
    )

    return [
        LogInfo(msg=f'[inference] Recording run {run_id} to {bag_uri}'),
        LogInfo(msg=(
            '[inference] After recording, export the JSON trace with: '
            f'python3 /root/ros2_ws/src/processing/'
            f'export_inference_trace.py {bag_uri}'
        )),
        recorder,
        canonicalizer,
        delayed_inference,
        stop_on_inference_exit,
        stop_on_recorder_exit,
        stop_on_canonicalizer_exit,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'checkpoint_path',
            default_value='',
            description='Absolute path to the model checkpoint (required).',
        ),
        DeclareLaunchArgument(
            'bag_uri',
            default_value='/root/ros2_ws/runs/inference',
            description='Base output path; a timestamp suffix is appended.',
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value='',
            description='Optional ROS parameter YAML for the inference node.',
        ),
        OpaqueFunction(function=launch_setup),
    ])
