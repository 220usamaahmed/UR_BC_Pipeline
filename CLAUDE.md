# UR_Sim — Behaviour-Cloning Data Pipeline

## What this project is for

The goal is to **collect behaviour-cloning (BC) datasets** on a UR3e arm by
running hard-coded, deterministic demonstrations and recording them.

Each demonstration is a single YAML file (an "experiment") that describes the
robot, the environment (collision obstacles), an ordered sequence of motion
steps, and what to record. Running that config drives the arm through the
scripted trajectory while a rosbag captures the joint states. Post-processing
turns each recorded run into a training array of position deltas, and a
placeholder inference node replays those deltas to mimic how a learned policy
would later be deployed.

So the lifecycle is:

```
  YAML config ──▶ scripted run on the arm ──▶ rosbag (a "run")
                                                   │
                              make_training_trajectory.py
                                                   │
                                          training_trajectory.npz
                                                   │
                                  dummy_inference (stand-in for a real policy)
```

The two packages under `ros2_ws/src/` other than `bc_pipeline` and the
`processing/` scripts are throwaway demos (`ur_demo`, `ur_moveit_demo`) —
**ignore them.** All real work lives in:

- `ros2_ws/src/bc_pipeline/` — the recording pipeline (a ROS 2 / ament_python package)
- `ros2_ws/src/processing/` — plain Python scripts to inspect runs and build training data

## Environment

Everything runs in Docker (ROS 2 Humble, amd64). The host `./ros2_ws` is
mounted at `/root/ros2_ws` in the container — **edit on the host, build/run
inside the container.**

```bash
docker compose up --build          # start UR3e mock hardware + move_group + foxglove_bridge
docker compose exec ur_sim bash    # shell into the running container
```

The entrypoint already runs the UR driver (mock hardware), MoveIt `move_group`,
and the Foxglove bridge (`ws://localhost:8765`). The bc_pipeline assumes these
are already up. Foxglove layout is in `foxglove-layout.json`.

Build + source inside the container:

```bash
cd /root/ros2_ws
colcon build --packages-select bc_pipeline
source install/setup.bash
```

## Recording a demonstration

```bash
ros2 launch bc_pipeline record_sequence.launch.py config:=drawer_demo.yaml
```

`config:=` is a filename in `bc_pipeline/config/` or an absolute path. The
launch file (`launch/record_sequence.launch.py`) starts three things from the
**same** config:

1. `scene_publisher` — applies obstacles to MoveIt's planning scene and latches
   `MarkerArray` markers for Foxglove. Keeps running so late Foxglove
   subscribers still get the (TRANSIENT_LOCAL) markers.
2. `ros2 bag record` — records `recording.topics` to `recording.bag_uri` plus a
   timestamp suffix (e.g. `runs/drawer_2026-06-20_14-14-03`), so runs never
   overwrite each other. Skipped (with a warning) if the config omits the
   `recording` section — the sequence then runs unrecorded.
3. `sequence_runner` — starts ~2 s later (so the bag is open and scene applied
   first), then executes the steps in order.

**Failure semantics:** the first failing step aborts the sequence and the runner
exits non-zero (1 = step failed, 2 = bad config/startup, 0 = success). A
non-zero exit flags the run as suspect so its bag can be discarded. Either way,
the runner's exit triggers `Shutdown()`, which SIGINTs the recorder so the bag
is flushed and closed cleanly.

## Processing a run

Run these **inside the container** (reading bags needs ROS 2 message defs to
deserialize CDR):

```bash
cd /root/ros2_ws/src/processing
python3 inspect_run.py /root/ros2_ws/runs/drawer_2026-06-20_14-14-03
python3 make_training_trajectory.py /root/ros2_ws/runs/drawer_2026-06-20_14-14-03
```

- `inspect_run.py` — summarizes a bag (topics, counts, approx rate, message
  field shapes). Degrades to a metadata-only summary if ROS isn't sourced.
- `make_training_trajectory.py` — reads `/joint_states`, reorders columns **by
  joint name** (so a publisher reordering can't corrupt columns), downsamples to
  a uniform 10 Hz grid, computes per-step deltas, and writes
  `<run>/training_trajectory.npz` with keys: `joint_names (J,)`,
  `timestamps (N,)`, `positions (N,J)`, `deltas (N-1,J)`, `rate_hz`.

## Simulated inference

```bash
ros2 run bc_pipeline dummy_inference --ros-args \
  -p run:=/root/ros2_ws/runs/drawer_2026-06-20_14-14-03
```

`dummy_inference.py` stands in for a trained policy we don't have yet. It
reproduces the *shape* of closed-loop action chunking: move to the recorded
start pose, then per chunk → observe a fresh `/joint_states`, accumulate the
chunk's deltas onto that measured seed to get absolute waypoints, send them as
one `FollowJointTrajectory` goal directly to the
`scaled_joint_trajectory_controller` (no MoveIt / no collision checking), then
"think" (pause). Params: `run`, `chunk_size` (10), `pause_sec` (2.0).

## Architecture of bc_pipeline

```
sequence_runner.py   — entry node: load config, build steps, wait for MoveIt, run in order
config_loader.py     — reads + structurally validates the experiment YAML (fail fast)
context.py           — shared MoveIt capability layer (the reusable "mechanics")
scene_publisher.py   — obstacles → MoveIt planning scene + Foxglove markers
dummy_inference.py   — policy stand-in (replays npz deltas)
steps/
  base.py            — Step ABC, @register decorator, STEP_REGISTRY, build_step factory
  checkpoint.py      — Checkpoint: joint-space move to a named checkpoint
                       (optional cartesian_offset → go straight to the
                       checkpoint pose shifted by a metre vector)
  orientation_lock.py— OrientationLockCheckpoint: straight Cartesian slide, orientation fixed
  wait.py            — Wait: pause
  gripper.py         — Gripper: placeholder (logs + settle delay, no hardware yet)
```

Design split: a **Step** expresses *intent* ("go home", "slide 10 cm along
−Z"); the reusable *mechanics* (talking to `move_group`, reading the EEF pose
via TF, computing/executing Cartesian paths) live once in `Context`. New step
types stay tiny because they reuse `Context`.

`Context` owns the MoveIt clients (`move_action`, `execute_trajectory`,
`compute_cartesian_path`) and a TF buffer. It uses `server_is_ready()` +
`spin_once()` rather than `wait_for_server()`, which can hang under Docker when
DDS discovery is flaky.

## The config schema

One config drives every node (see `config/drawer_demo.yaml`). A config is
either a `.yaml` file (plain data) or a `.py` file that defines a
module-level `CONFIG` dict — the `.py` form is executed, so it can use
`math`/`random`/loops to build the config (e.g. randomised offsets for data
augmentation; see `config/drawer_demo_random.py`). Either way it must
describe the same sections below.

`record_sequence.launch.py` resolves the source into a concrete dict exactly
ONCE (`config_loader.resolve_source()`), dumps it to a plain resolved YAML
file, and points `scene_publisher` + `sequence_runner` at that resolved file
instead of the original — this is what keeps "every node reads the SAME
config" true even when the source is a `.py` file with randomness in it (two
independent `exec`s would otherwise give the two nodes different values). If
recording is enabled, the resolved YAML is also saved as a sibling of the bag
directory (`<bag_uri>_<timestamp>.yaml`), so a run with randomised values
stays reproducible/debuggable later.

Sections, all validated up front in `config_loader.py`:

- `robot` — `planning_group`, `eef_link`, `base_frame`, `joint_names` (the
  canonical joint order; checkpoints must have one value each).
- `planning` — `velocity_scaling`, `accel_scaling`, `planning_time`.
- `checkpoints` — named joint-position lists in **degrees** (`config_loader`
  converts them to radians for the controller), reused by `Checkpoint` steps.
- `steps` — the ordered sequence; each entry needs a `type` matching a
  registered step.
- `obstacles` — optional boxes (`id`, `size` = full side lengths, `position` =
  centre, optional `color` for Foxglove only), in `robot.base_frame`.
- `recording` — optional; `bag_uri` (base path; timestamp appended) and `topics`
  to record. Omit the whole section to run without recording (launch warns).

**Validation philosophy:** structure is checked eagerly before anything moves;
each Step validates its own args in `validate()` at construction time, also
before any motion. A typo should never surface halfway through a motion with the
recorder already running.

## Adding a new step type

The step framework is a self-registering plugin registry — there is **no central
if/elif** to edit.

1. Create a module under `bc_pipeline/steps/`, e.g. `my_step.py`.
2. Subclass `Step` and decorate it with `@register("MyType")` (the string is the
   YAML `type`).
3. Implement `validate()` (parse + check `self.cfg`, raise `StepConfigError` on
   bad args) and `execute()` (do the work, return `True`/`False`). Optionally
   override the `label` property for nicer logs.
4. Add `from . import my_step` to `steps/__init__.py` so the `@register`
   decorator actually runs (importing the module is what registers it).

Inside a step you have:
- `self.cfg` — this step's YAML dict. Use `self._require('key')` for required
  keys (raises a `StepConfigError` naming the step) and `self.cfg.get('key',
  default)` for optional ones.
- `self.ctx` — the shared `Context`: `ctx.logger`, `ctx.node`, `ctx.joint_names`,
  `ctx.checkpoints`, and the mechanics `ctx.plan_and_execute_joints(angles)`,
  `ctx.get_eef_pose()`, `ctx.execute_cartesian(label, waypoints, max_step,
  min_fraction)`.

If a step needs a capability that doesn't exist yet (e.g. real gripper I/O), add
it as a method on `Context` so it's reusable, rather than putting ROS plumbing
in the step.

Example skeleton:

```python
# bc_pipeline/steps/my_step.py
from .base import Step, StepConfigError, register

@register('MyType')
class MyType(Step):
    def validate(self):
        self.amount = float(self._require('amount'))
        if self.amount < 0:
            raise StepConfigError("MyType amount must be non-negative.")

    @property
    def label(self) -> str:
        return f"MyType {self.amount}"

    def execute(self) -> bool:
        self.ctx.logger.info(f"Doing MyType with {self.amount}")
        return True   # False aborts the whole sequence
```

Then `from . import my_step` in `steps/__init__.py`, rebuild, and use
`- type: MyType` in a config.

## Conventions / gotchas

- Joints are always matched **by name**, never by position, when reading
  `/joint_states` (in both `dummy_inference` and `make_training_trajectory`) —
  publishers can reorder columns.
- YAML 1.1 parses bare `on`/`off` as booleans; the `Gripper` step accepts both
  the strings and the booleans.
- `Gripper` is a placeholder — only `execute()` will change when suction
  hardware arrives; the YAML schema is already final.
- `OrientationLockCheckpoint` locks orientation *by construction* (every Cartesian
  waypoint shares the start orientation) rather than via an OMPL
  OrientationConstraint, which the planner treats only as guidance.
- Nodes guard `rclpy.shutdown()` with `rclpy.ok()` because the launch
  `Shutdown()`/SIGINT path may have already torn the context down.
- `obstacles` currently supports BOX/CUBE only; `size` is full side lengths.
- Downsample rate is fixed at 10 Hz (`RATE_HZ` in `make_training_trajectory.py`);
  `dummy_inference` reads `rate_hz` back from the npz to time its waypoints.
