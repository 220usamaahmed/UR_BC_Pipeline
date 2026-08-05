"""Launch the legacy dataset recorder and trajectory controller."""

from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


CONTROLLER_START_DELAY_SEC = 1.0


def generate_launch_description() -> LaunchDescription:
    dataset_recorder = Node(
        package="legacy_bc_pipeline",
        executable="dataset_recorder",
        name="dataset_recorder",
        output="screen",
    )

    trajectory_control = Node(
        package="legacy_bc_pipeline",
        executable="trajectory_control",
        name="trajectory_control",
        output="screen",
    )

    # Give the recorder time to advertise its services before the controller starts.
    delayed_trajectory_control = TimerAction(
        period=CONTROLLER_START_DELAY_SEC,
        actions=[trajectory_control],
    )

    return LaunchDescription(
        [
            dataset_recorder,
            delayed_trajectory_control,
        ]
    )
