#!/usr/bin/env python3
"""Republish joint states in the inference model's canonical joint order."""

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from bc_pipeline.inference_debug import (
    CANONICAL_JOINT_TOPIC,
    EVENT_TOPIC,
    validate_event,
)


class JointStateCanonicalizer(Node):
    """Keep high-rate plotting independent of the blocking inference loop."""

    def __init__(self):
        super().__init__('joint_state_canonicalizer')
        self.joint_names = None
        self.publisher = self.create_publisher(
            JointState, CANONICAL_JOINT_TOPIC, qos_profile_sensor_data
        )
        self.create_subscription(String, EVENT_TOPIC, self._on_event, 20)
        self.create_subscription(
            JointState,
            '/joint_states',
            self._on_joint_state,
            qos_profile_sensor_data,
        )

    def _on_event(self, message: String):
        try:
            event = validate_event(json.loads(message.data))
        except (ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f'Ignoring invalid trace event: {exc}')
            return
        if event['event'] != 'run_start':
            return
        joint_names = event.get('joint_names')
        if not isinstance(joint_names, list) or not all(
                isinstance(name, str) and name for name in joint_names):
            self.get_logger().error(
                'run_start event has no valid joint_names list'
            )
            return
        self.joint_names = list(joint_names)
        self.get_logger().info(
            'Canonical joint order: ' + ', '.join(self.joint_names)
        )

    def _on_joint_state(self, message: JointState):
        if self.joint_names is None:
            return
        by_name = dict(zip(message.name, message.position))
        if not all(name in by_name for name in self.joint_names):
            return
        canonical = JointState()
        canonical.header = message.header
        canonical.name = list(self.joint_names)
        canonical.position = [
            float(by_name[name]) for name in self.joint_names
        ]
        self.publisher.publish(canonical)


def main(args=None):
    rclpy.init(args=args)
    node = JointStateCanonicalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
