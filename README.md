# ROS 2 + UR3e Mock Hardware on macOS

This repo runs the Universal Robots driver with **mock hardware** inside a
Docker container (amd64). A `docker-compose.yml` builds the image, starts the
UR3e simulator, MoveIt `move_group`, and the Foxglove bridge automatically,
and mounts a workspace folder so you can write ROS 2 nodes on your host and
run them in the container.

## What runs inside the container

When you start the container the entrypoint launches three things:

| Process | What it is |
|---|---|
| `ur_moveit.launch.py` | UR3e driver (mock hardware) + `move_group` (MoveIt planner) |
| `foxglove_bridge` | WebSocket bridge, reachable at `ws://localhost:8765` |

`move_group` is the central MoveIt node.  It holds the robot model, owns the
planning scene (the world model with obstacles), and exposes action/service
interfaces that your nodes use to plan and execute motion.

---

## Start everything

```bash
docker compose up --build
```

To stop:

```bash
docker compose down
```

Open a shell in the running container:

```bash
docker compose exec ur_sim bash
```

The `./ros2_ws` folder on your host is mounted at `/root/ros2_ws` in the
container.  Write code on the host, run it inside.

---

## Demo 1 — dummy_controller (no MoveIt, direct trajectory)

This is the simplest possible way to move the arm: you specify joint angles
and a duration, and the controller executes it with no collision checking.

### Build and run

```bash
# Inside the container
cd /root/ros2_ws
colcon build --packages-select ur_demo
source install/setup.bash
ros2 run ur_demo dummy_controller
```

### What it does

Sends a single `FollowJointTrajectory` goal directly to the
`scaled_joint_trajectory_controller`.  No path planning, no obstacle
awareness.  The controller interpolates from the current pose to the target
joint angles in 5 seconds.

---

## Demo 2 — ur_moveit_demo (MoveIt path planning with obstacles)

This demo has two nodes that work together: one that populates the planning
scene with obstacles, and one that moves the arm through a waypoint sequence
while avoiding those obstacles.

### Architecture

```
scene_builder
  └─ calls /apply_planning_scene service
       └─ move_group adds 'table' and 'wall' to its world model

motion_planner
  └─ connects to move_group via MoveItPy
  └─ for each waypoint:
       set start state → set goal state → plan (OMPL) → execute
            ↓
       move_group checks every candidate path against the planning scene
            ↓
       sends collision-free trajectory to scaled_joint_trajectory_controller
```

### Build

```bash
# Inside the container
cd /root/ros2_ws
colcon build --packages-select ur_moveit_demo
source install/setup.bash
```

### Step 1 — Add obstacles to the planning scene

```bash
ros2 run ur_moveit_demo scene_builder
```

This node calls the `/apply_planning_scene` service once and exits.  After it
runs, `move_group` knows about two obstacles:

| Name | Shape | Position | Purpose |
|---|---|---|---|
| `table` | 1.2×1.2×0.02 m box | z = −0.01 m (flush with robot base) | Prevents the arm from swinging below its mount |
| `wall` | 0.02×1.0×1.0 m box | x = 0.4 m, z = 0.5 m | Blocks the forward direction; forces the planner to route around it |

You can verify the scene was applied by checking the move_group logs or
using RViz (if started separately).

### Step 2 — Run the motion planner

```bash
ros2 run ur_moveit_demo motion_planner
```

The arm visits five poses in order:

```
home → left → raised → right → home
```

For each leg, the node:
1. Updates the start state to the robot's current joint positions
2. Specifies the goal as a target joint configuration
3. Calls MoveIt's OMPL planner — this is where collision checking happens
4. Executes the resulting trajectory on the controller

Watch the logs: each leg prints how many trajectory points the planner
generated.  Legs that require routing around the wall will show a curved
multi-point trajectory; clear legs may be almost straight.

### Running both steps in one go

Open two terminals in the container:

```bash
# Terminal 1
ros2 run ur_moveit_demo scene_builder && ros2 run ur_moveit_demo motion_planner
```

Or sequentially in one terminal — `scene_builder` exits as soon as the
service call returns, so chaining with `&&` works fine.

---

## Understanding the package ownership

| Component | Owner | Role |
|---|---|---|
| `control_msgs` | ros2_control community | Defines `FollowJointTrajectory` action interface |
| `moveit_msgs` | MoveIt community | Defines `CollisionObject`, `PlanningScene`, `ApplyPlanningScene` |
| `controller_manager` | ros2_control community | Loads and manages controller plugins |
| `ScaledJointTrajectoryController` | Universal Robots | Executes trajectories, respects speed slider |
| `move_group` | MoveIt community | Plans collision-free paths, owns the planning scene |
| `ur_moveit_config` | Universal Robots | SRDF + kinematics config that teaches MoveIt the UR3e geometry |
| `moveit_py` | MoveIt community | Python API used by `motion_planner.py` |

---

## Manual reference commands

These are run from a shell inside the container.

### Check which controllers are active

```bash
ros2 control list_controllers
```

### Inspect the planning scene

```bash
ros2 service call /get_planning_scene moveit_msgs/srv/GetPlanningScene \
  "{components: {components: 1023}}"
```

### Send a goal directly (no MoveIt, no collision checking)

```bash
ros2 action send_goal /scaled_joint_trajectory_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory "{
    trajectory: {
      joint_names: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint,
                    wrist_1_joint, wrist_2_joint, wrist_3_joint],
      points: [
        {
          positions: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
          time_from_start: {sec: 5, nanosec: 0}
        }
      ]
    }
  }"
```
