#!/usr/bin/env python3
"""
Scene Publisher — obstacles into MoveIt AND into Foxglove, from one config.

Merges what ur_moveit_demo split across scene_builder.py and
obstacle_markers.py:

  • pushes the obstacles into MoveIt's planning scene (what the planner AVOIDS),
    via the /apply_planning_scene service, and
  • publishes the same geometry as a latched MarkerArray (what you SEE) so a
    Foxglove 3D panel can render exactly what the planner knows.

Both use full side lengths and a centre pose, so one obstacle spec feeds both.

Obstacles are read from the shared experiment config (the 'obstacles' section);
the frame is robot.base_frame.  Only BOX/CUBE is supported for now — dims come
from a 'size' list so other primitives are easy to add later.

LIFECYCLE
---------
On startup it applies the scene once and publishes the markers once, then spins.
The markers use TRANSIENT_LOCAL (latched) QoS so a Foxglove panel opened later
still receives them — which is also why this node must keep running rather than
exit after publishing.

    ros2 run bc_pipeline scene_publisher --ros-args -p config:=/abs/path.yaml
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile

from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from bc_pipeline.config_loader import load_config

DEFAULT_COLOR = (0.6, 0.6, 0.6, 0.8)   # grey — used when an obstacle omits color


class ScenePublisher(Node):
    def __init__(self):
        super().__init__('scene_publisher')

        self.declare_parameter('config', '')
        config = load_config(self.get_parameter('config').value)

        self.obstacles = config['obstacles']
        self.world_frame = config['robot']['base_frame']

        self._apply_scene()
        self._publish_markers()

    # ── MoveIt planning scene ──────────────────────────────────────────────────

    def _apply_scene(self):
        client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self.get_logger().info('Waiting for /apply_planning_scene service …')
        client.wait_for_service()

        scene = PlanningScene()
        scene.is_diff = True   # merge into the existing scene, don't replace it
        scene.world.collision_objects = [
            self._make_collision_object(spec) for spec in self.obstacles
        ]

        request = ApplyPlanningScene.Request()
        request.scene = scene

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None and future.result().success:
            names = ', '.join(f"'{spec['id']}'" for spec in self.obstacles)
            self.get_logger().info(f'Planning scene updated. Objects: {names or "(none)"}.')
        else:
            self.get_logger().error('ApplyPlanningScene returned failure.')

    def _make_collision_object(self, spec: dict) -> CollisionObject:
        obj = CollisionObject()
        obj.header.frame_id = self.world_frame
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.id = spec['id']
        obj.operation = CollisionObject.ADD

        shape = SolidPrimitive()
        shape.type = SolidPrimitive.BOX
        shape.dimensions = [float(v) for v in spec['size']]   # full side lengths
        obj.primitives = [shape]

        px, py, pz = spec['position']
        pose = Pose()
        pose.position = Point(x=float(px), y=float(py), z=float(pz))
        pose.orientation = Quaternion(w=1.0)
        obj.primitive_poses = [pose]
        return obj

    # ── Foxglove markers (latched) ─────────────────────────────────────────────

    def _publish_markers(self):
        qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,   # latched
        )
        self._marker_pub = self.create_publisher(MarkerArray, '/collision_markers', qos)
        self._marker_pub.publish(self._build_markers())
        self.get_logger().info(
            f'Published {len(self.obstacles)} obstacle marker(s) on '
            f'/collision_markers (latched).'
        )

    def _build_markers(self) -> MarkerArray:
        array = MarkerArray()
        for index, spec in enumerate(self.obstacles):
            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'obstacles'
            marker.id = index
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            sx, sy, sz = spec['size']
            marker.scale = Vector3(x=float(sx), y=float(sy), z=float(sz))

            px, py, pz = spec['position']
            marker.pose = Pose(
                position=Point(x=float(px), y=float(py), z=float(pz)),
                orientation=Quaternion(w=1.0),
            )

            r, g, b, a = spec.get('color', DEFAULT_COLOR)
            marker.color = ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))
            array.markers.append(marker)
        return array


def main(args=None):
    rclpy.init(args=args)
    try:
        node = ScenePublisher()
    except Exception as exc:
        # Surface config errors clearly instead of a bare traceback.
        print(f'[scene_publisher] startup failed: {exc}')
        rclpy.shutdown()
        raise
    try:
        rclpy.spin(node)   # keep latched markers alive for late subscribers
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
