#!/usr/bin/env python3
"""
Sequence Runner — executes a list of Steps from the experiment config.

Reads the 'config' ROS parameter, loads + validates the YAML, builds every Step
(which validates its own args up front), waits for MoveIt, then runs the steps
in order.

FAILURE BEHAVIOUR
-----------------
First failure aborts the whole sequence and the process exits non-zero.  No
retries.  A non-zero exit flags the run as suspect so the recorded bag can be
discarded from the dataset.  The launch file's OnProcessExit still fires
Shutdown() either way, so the recorder flushes and closes the bag cleanly.

Exit codes:
  0  all steps completed
  1  a step failed
  2  bad config / startup error

Run via the launch file, or standalone:
    ros2 run bc_pipeline sequence_runner --ros-args -p config:=/abs/path.yaml
"""

import sys

import rclpy
from rclpy.node import Node

from bc_pipeline.config_loader import load_config
from bc_pipeline.context import Context
from bc_pipeline.steps import build_step


def run(node: Node) -> int:
    node.declare_parameter('config', '')
    config_path = node.get_parameter('config').value

    try:
        config = load_config(config_path)
        ctx = Context(node, config)
        steps = [build_step(entry, ctx) for entry in config['steps']]
    except Exception as exc:   # ConfigError, StepConfigError, or anything else
        node.get_logger().error(f'Config error: {exc}')
        return 2

    ctx.wait_for_servers()

    total = len(steps)
    node.get_logger().info(f'Running {total} step(s).')
    for i, step in enumerate(steps, 1):
        node.get_logger().info(f'\n{"=" * 52}\n[{i}/{total}] {step.label}')
        ctx.publish_current_step(step.label)
        if not step.execute():
            node.get_logger().error(
                f'Step {i}/{total} ({step.label}) failed — aborting sequence.'
            )
            return 1

    node.get_logger().info(f'\nAll {total} step(s) completed.')
    return 0


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('sequence_runner')
    code = 2
    try:
        code = run(node)
    finally:
        node.destroy_node()
        # Guard in case a SIGINT-triggered handler already shut the context down.
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
