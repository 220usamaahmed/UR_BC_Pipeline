#!/usr/bin/env python3
"""
Scene Builder — adds collision objects to the MoveIt planning scene.

WHAT THIS TEACHES
-----------------
MoveIt maintains a "Planning Scene": an internal model of the world that
includes the robot itself and any additional obstacles you describe.  The
planner only knows about things you put in that scene — sensors are one way,
but for a mock setup you describe geometry manually.

HOW COLLISION OBJECTS WORK
---------------------------
A CollisionObject message says:
  • id         — a unique name so you can update or remove it later
  • header     — which coordinate frame the geometry is defined in
  • primitives — one or more shapes (BOX, SPHERE, CYLINDER, CONE)
  • poses       — where each shape sits in that frame
  • operation  — ADD, REMOVE, MOVE, or APPEND

The most reliable way to inject objects is the /apply_planning_scene
service.  Unlike publishing to a topic, the service call blocks until
move_group confirms the scene has been updated.

WHAT WE ADD
-----------
  table — a flat slab sitting at the robot's base (z = 0)
  wall  — a vertical slab in front of the robot (+X direction)

The wall is placed at x=0.4 m, roughly arm-length away, so the motion
planner has to route around it when moving to certain poses.

Run this node first, before motion_planner:

    ros2 run ur_moveit_demo scene_builder
"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Pose, Quaternion
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive

# Geometry lives in obstacles.py so the planning scene (here) and the Foxglove
# markers (obstacle_markers.py) always describe the same boxes.
from ur_moveit_demo.obstacles import OBSTACLES, WORLD_FRAME


class SceneBuilder(Node):
    """Adds a fixed set of collision objects to the MoveIt planning scene."""

    def __init__(self):
        super().__init__('scene_builder')

        # /apply_planning_scene is a service provided by move_group.
        # Calling it is synchronous: we block until move_group confirms the
        # update, so we know the objects are in the scene before we exit.
        self._client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self.get_logger().info('Waiting for /apply_planning_scene service ...')
        self._client.wait_for_service()
        self.get_logger().info('Service ready.')

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(self):
        """Publish all objects in a single service call."""

        # is_diff=True means "merge this into the existing scene" rather than
        # replacing the whole scene from scratch.  Always use this unless you
        # deliberately want to wipe everything.
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [self._make_box(spec) for spec in OBSTACLES]

        request = ApplyPlanningScene.Request()
        request.scene = scene

        future = self._client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result().success:
            names = ', '.join(f"'{spec['id']}'" for spec in OBSTACLES)
            self.get_logger().info(f'Planning scene updated.  Objects added: {names}.')
        else:
            self.get_logger().error('ApplyPlanningScene returned failure.')

    # ------------------------------------------------------------------
    # Object builders
    # ------------------------------------------------------------------

    def _make_box(self, spec: dict) -> CollisionObject:
        """
        Build one BOX collision object from a shared obstacle spec.

        A SolidPrimitive BOX takes full side lengths [x, y, z]; the pose is the
        box centre, expressed in WORLD_FRAME.  Both fields come straight from
        obstacles.py, the same data obstacle_markers.py uses for visualization.
        """
        obj = CollisionObject()
        obj.header.frame_id = WORLD_FRAME
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.id = spec['id']
        obj.operation = CollisionObject.ADD

        shape = SolidPrimitive()
        shape.type = SolidPrimitive.BOX
        shape.dimensions = list(spec['size'])   # [x_size, y_size, z_size] in metres
        obj.primitives = [shape]

        px, py, pz = spec['position']
        pose = Pose()
        pose.position = Point(x=px, y=py, z=pz)
        pose.orientation = Quaternion(w=1.0)
        obj.primitive_poses = [pose]

        sx, sy, sz = spec['size']
        self.get_logger().info(
            f"  {spec['id']}: {sx}×{sy}×{sz} m box at ({px}, {py}, {pz})"
        )
        return obj


def main(args=None):
    rclpy.init(args=args)
    node = SceneBuilder()
    try:
        node.build()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
