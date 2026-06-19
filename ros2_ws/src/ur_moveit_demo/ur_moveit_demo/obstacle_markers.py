#!/usr/bin/env python3
"""
Obstacle Markers — publishes the collision obstacles as visualization markers.

WHY THIS NODE EXISTS
--------------------
scene_builder puts the table and wall into MoveIt's planning scene so the
planner avoids them.  That scene is a moveit_msgs/PlanningScene, which Foxglove
cannot render.  Foxglove DOES understand visualization_msgs/Marker.

This node publishes the SAME geometry (imported from obstacles.py) as a
MarkerArray, so Foxglove shows you exactly what the planner is avoiding.

LATCHED (TRANSIENT_LOCAL) QoS
-----------------------------
The obstacles are static, so we publish once with TRANSIENT_LOCAL durability —
a "latched" topic.  Any subscriber that connects later, including Foxglove when
you open the 3D panel, immediately receives the last MarkerArray without us
republishing on a timer.  The node then just spins to stay alive and keep
serving that latched message; if it exits, the latched data goes away.

This is also why `ros2 topic echo /monitored_planning_scene` appeared to hang
earlier: that topic is transient-local too, and a default echo subscriber's QoS
doesn't match it, so it never receives anything.

USAGE
-----
Run it in its own terminal (or background it), then add /collision_markers to a
Foxglove 3D panel:

    ros2 run ur_moveit_demo obstacle_markers
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile

from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from ur_moveit_demo.obstacles import OBSTACLES, WORLD_FRAME


class ObstacleMarkers(Node):
    def __init__(self):
        super().__init__('obstacle_markers')

        # TRANSIENT_LOCAL = latched: late-joining subscribers (Foxglove) still
        # receive the last published sample.  KEEP_LAST/depth=1 retains it.
        qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._pub = self.create_publisher(MarkerArray, '/collision_markers', qos)

        self._pub.publish(self._build_markers())
        self.get_logger().info(
            f"Published {len(OBSTACLES)} obstacle marker(s) on /collision_markers "
            f"(latched).  Add this topic to a Foxglove 3D panel."
        )

    def _build_markers(self) -> MarkerArray:
        array = MarkerArray()
        for index, spec in enumerate(OBSTACLES):
            marker = Marker()
            marker.header.frame_id = WORLD_FRAME
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'obstacles'
            marker.id = index              # unique per marker within the namespace
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            sx, sy, sz = spec['size']
            marker.scale = Vector3(x=sx, y=sy, z=sz)

            px, py, pz = spec['position']
            marker.pose = Pose(
                position=Point(x=px, y=py, z=pz),
                orientation=Quaternion(w=1.0),
            )

            r, g, b, a = spec['color']
            marker.color = ColorRGBA(r=r, g=g, b=b, a=a)

            array.markers.append(marker)
        return array


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleMarkers()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
