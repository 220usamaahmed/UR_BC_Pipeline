"""
Launch the motion planner and a ROS2 bag recorder together.

The recorder starts first, waits 2 seconds, then the planner runs.
When the planner process exits (naturally, after all waypoints are done),
OnProcessExit fires Shutdown() which sends SIGINT to every remaining
process — including the bag recorder, which flushes and closes the bag
cleanly before dying.

Usage:
    ros2 launch ur_moveit_demo record_motion.launch.py

Output:
    motion_data/   ← directory containing the .mcap bag file
"""

from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, Shutdown, TimerAction
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node


def generate_launch_description():

    recorder = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '--output', 'motion_data',
            '/joint_states',
            '/tf',
            '/tf_static',
        ],
        output='screen',
    )

    # Keep a reference to the Node action so OnProcessExit can target it.
    planner = Node(
        package='ur_moveit_demo',
        executable='motion_planner',
        output='screen',
    )

    # Delay the planner so the recorder has time to open the bag file before
    # any joint state messages arrive.
    delayed_planner = TimerAction(period=2.0, actions=[planner])

    # When the planner exits (all waypoints done), shut down the whole launch.
    # Shutdown() sends SIGINT to all remaining processes; ros2 bag record
    # handles SIGINT by flushing and closing the bag cleanly.
    stop_on_done = RegisterEventHandler(
        OnProcessExit(
            target_action=planner,
            on_exit=[Shutdown()],
        )
    )

    return LaunchDescription([
        recorder,
        delayed_planner,
        stop_on_done,
    ])
