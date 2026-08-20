# Inference Debug Visualization — Usage and Handoff

This document describes the inference observation-matching and debugging work
currently implemented in this repository. It is intended both as operating
instructions on the Robot PC and as context for continuing development after
the current changes are committed.

## Current status

The implementation is complete at the static/unit-test level, but it has not
yet been built or exercised in ROS, Foxglove, or on the physical robot.

Host-safe verification completed:

- All changed Python files compile.
- Ruff passes for the changed Python files.
- `foxglove-layout.json` is valid JSON.
- Seven ROS-independent inference-debug tests pass.
- `git diff --check` passes.

The first Robot-PC run should therefore be treated as integration testing, not
as a validated production run.

## What changed

### Observation matching

`bc_pipeline/inference.py` now captures sensor data only inside the accepted
trajectory's execution timestamp window. It builds candidates around depth
frames and interpolates the high-frequency joint stream at each depth timestamp.

For actions 6–10, it chooses five distinct, chronologically ordered candidates
that minimize joint-position error against the absolute desired positions. The
preceding action is used as an ordering-only anchor. Every selected joint must
remain within `joint_match_tolerance_rad`, which defaults to `0.1` radians.

The selected joint value is therefore usually a synthetic interpolated value,
not one original `JointState` message. Its trace records both bracketing joint
timestamps and the interpolation coefficient.

### Debug recording

When `debug_enabled` is true, inference publishes these topics:

| Topic | Type | Purpose |
|---|---|---|
| `/bc_pipeline/inference/events` | `std_msgs/String` | Versioned JSON lifecycle and trace events |
| `/bc_pipeline/inference/phase` | `std_msgs/String` | `startup`, `bootstrap`, `inference`, `execution`, `selection`, `stopped` |
| `/bc_pipeline/inference/action_targets_planned` | `sensor_msgs/JointState` | Ten absolute targets at estimated planned timestamps |
| `/bc_pipeline/inference/action_targets_matched` | `sensor_msgs/JointState` | Targets 6–10 at the timestamps of their matched observations |
| `/bc_pipeline/inference/selected_joint_observations` | `sensor_msgs/JointState` | Interpolated joint inputs actually retained for the next inference |
| `/bc_pipeline/inference/selected_depth` | `sensor_msgs/Image` | Exact processed `32FC1` depth arrays supplied to the model |

`joint_state_canonicalizer.py` runs as a separate process and publishes:

| Topic | Type | Purpose |
|---|---|---|
| `/bc_pipeline/inference/joint_states_canonical` | `sensor_msgs/JointState` | Full-rate joint data reordered into the model's joint order |

The separate process is important: model inference blocks the inference node's
manual spin loop, so republishing canonical joints inside that node would
undersample a 400–500 Hz source.

The bag also records the original `/joint_states`, registered depth, and
`/scaled_joint_trajectory_controller/controller_state`.

### Trace identity and provenance

ROS 2 messages do not have a universal message ID. The trace instead uses:

- Integer nanosecond source timestamps.
- Stable IDs such as `chunk-000001-action-06`.
- A SHA-256 hash of the exact processed depth shape and float32 pixels.
- The joint timestamps immediately before and after the depth frame.
- The interpolation coefficient, or the identical nearest-joint timestamps
  when interpolation was unavailable.
- Actual, desired, and signed-error joint arrays.
- Maximum absolute per-joint error, gripper state, candidate count, and matcher
  tolerance.

Model-inference events record the five observation IDs and depth hashes used as
input, the joint/gripper input array, inference duration, joint deltas, and
gripper outputs. Execution events record the seed, all ten absolute targets,
planned timestamps, controller result, and capture counts.

## Preparing the Robot PC

The Docker image must be rebuilt once because `Dockerfile` now installs
`ros-humble-rosbag2-storage-mcap`.

```bash
cd /path/to/UR_Sim
docker compose up --build
```

In another terminal, enter the running container and rebuild the ROS package:

```bash
docker compose exec ur_sim bash
cd /root/ros2_ws
colcon build --packages-select bc_pipeline --symlink-install
source install/setup.bash
```

For the real robot, use the project's normal real-hardware compose/start
workflow and complete the existing physical safety checks before starting
inference.

## Recording an inference run

Start inference through the new combined launch file:

```bash
ros2 launch bc_pipeline inference.launch.py \
  checkpoint_path:=/root/ros2_ws/checkpoints/model.pt
```

Optional arguments:

```bash
ros2 launch bc_pipeline inference.launch.py \
  checkpoint_path:=/root/ros2_ws/checkpoints/model.pt \
  bag_uri:=/root/ros2_ws/runs/inference \
  params_file:=/root/ros2_ws/inference_params.yaml
```

- `checkpoint_path` is required.
- `bag_uri` is a base path; the launch appends a timestamp.
- `params_file` is optional and must be a ROS parameter YAML for the inference
  node. Launch-provided `checkpoint_path`, `debug_enabled`, and `run_id` take
  precedence.

The launch starts the MCAP recorder and canonicalizer first, waits two seconds,
then starts inference. Inference or recorder exit shuts down the entire launch,
allowing the bag to close cleanly. The terminal prints the exact bag path.

MCAP uses native indexed Zstd/Fast compression from
`config/mcap_writer_options.yaml`. Do not add rosbag file-level compression on
top of it because that would remove direct indexed access for Foxglove.

## Exporting the JSON trace

Run the exporter inside the sourced ROS container after the bag has closed:

```bash
cd /root/ros2_ws/src/processing
python3 export_inference_trace.py \
  /root/ros2_ws/runs/inference_<timestamp>
```

It writes:

```text
<bag>/inference_trace.json
```

The JSON contains run events and ordered chunk events, but does not duplicate
the high-frequency joints or images already stored in MCAP. It intentionally
preserves incomplete and failed chunks.

The exporter also reports:

- Header-stamp to bag-receive-time offsets for raw and debug sensor topics.
- Whether every selected-depth timestamp/hash reference matches an exact
  `selected_depth` image stored in the bag.
- Run completion and final status derived from `run_end`; an interrupted bag
  without that event is marked `incomplete`.

## Using Foxglove

1. Open the `.mcap` file from the timestamped bag directory in Foxglove.
2. Import the repository's `foxglove-layout.json`.
3. Select the **Inference Debug** tab.

The tab includes:

- Canonically ordered measured joint traces.
- Ten planned absolute model targets.
- Selected joint observations.
- Optional matched targets at the selected timestamps. These are disabled by
  default in the legend; enable them when inspecting position error.
- Raw and selected depth-frame timing points.
- Inference/execution/selection phase transitions.
- Raw registered-depth and exact selected-depth image panels.
- Controller desired-versus-actual traces.
- Raw JSON trace events.

Debug joint topics use header timestamps. Planned targets show their estimated
controller checkpoint time. Matched targets use the selected depth timestamp,
so their vertical distance from the selected observation represents position
error without hiding timing lag.

Selected processed images are chosen retrospectively after a chunk finishes.
Their ROS headers retain the original depth timestamp, but their bag receive
time is later, during selection. If the built-in Image panel's playback behavior
is confusing, use the raw-depth panel at the selected source timestamp and the
JSON/hash verification as the authoritative correlation. A custom Foxglove
panel can address this later if necessary.

## First Robot-PC acceptance run

Use a short run of at least two successful chunks. Preserve the terminal output
even if the run fails.

Check the bag before drawing conclusions from model performance:

```bash
ros2 bag info /root/ros2_ws/runs/inference_<timestamp>
python3 /root/ros2_ws/src/processing/inspect_run.py \
  /root/ros2_ws/runs/inference_<timestamp>
```

Confirm:

- Raw `/joint_states` remains approximately 400–500 Hz.
- Registered depth remains approximately 30 FPS.
- Canonical joints remain close to the raw joint rate during inference pauses.
- `/scaled_joint_trajectory_controller/controller_state` exists under that
  exact topic name on the Robot PC.
- Every completed inference event has ten deltas and every successful selection
  has five observations corresponding to actions 6–10.
- Selected source timestamps lie inside the corresponding execution window.
- Selected observation timestamps increase strictly within each chunk.
- `selected_depth_verification.all_matched` is true in the exported JSON.
- Planned, matched, and selected points appear in the Foxglove joint plot.
- Zooming to millisecond scale reveals individual high-frequency samples and
  remains responsive.

If matching fails, inspect `selection_failed` in the JSON before increasing the
tolerance. Its candidate count and closest per-target maximum errors distinguish
missing sensor data from genuine tracking error.

## Known unverified areas

- No ROS package build or launch validation has been performed yet.
- MCAP plugin availability and Humble writer-option compatibility must be
  confirmed after rebuilding the Robot-PC Docker image.
- The exact controller-state topic name must be confirmed against the installed
  UR controller configuration.
- Foxglove panel configuration may require minor adjustment for the installed
  Foxglove version, particularly array plotting and selected-image playback.
- The `0.1` radian joint tolerance is only an initial configurable default. Do
  not tune it until controller tracking, timestamp offsets, and candidate counts
  have been inspected.
- The planned target timestamps begin at action-goal acceptance and are timing
  estimates; controller state is the authoritative source for actual controller
  desired/actual timing.

## Information needed to continue this feature

After the first Robot-PC test, retain or provide all of the following:

1. The Git commit hash containing this implementation.
2. The exact inference launch command and any parameter YAML used.
3. The model checkpoint path or immutable model identifier.
4. The complete timestamped MCAP bag directory.
5. The generated `inference_trace.json`.
6. Complete launch/inference terminal output from startup through shutdown.
7. `ros2 bag info` and `inspect_run.py` output.
8. The result of:

   ```bash
   ros2 topic info /scaled_joint_trajectory_controller/controller_state -v
   ```

9. A screenshot or exported copy of the Foxglove Inference Debug layout if any
   panels fail to render as expected.
10. A short description of the physical behavior: whether the robot completed
    each chunk, visibly lagged or overshot, stopped unexpectedly, or produced
    degraded model behavior.
11. For any problematic chunk, its one-based chunk number and the relevant
    `inference_complete`, `execution_start`, `execution_result`, and
    `selection_complete`/`selection_failed` events from the JSON.

With those artifacts, continued work can distinguish among controller tracking,
clock alignment, observation matching, depth preprocessing, recorder QoS, and
Foxglove-only visualization problems without repeating the initial diagnosis.

## Files involved

- `ros2_ws/src/bc_pipeline/bc_pipeline/inference.py`
- `ros2_ws/src/bc_pipeline/bc_pipeline/inference_debug.py`
- `ros2_ws/src/bc_pipeline/bc_pipeline/joint_state_canonicalizer.py`
- `ros2_ws/src/bc_pipeline/launch/inference.launch.py`
- `ros2_ws/src/bc_pipeline/config/mcap_writer_options.yaml`
- `ros2_ws/src/processing/export_inference_trace.py`
- `ros2_ws/src/bc_pipeline/test/test_inference_debug.py`
- `foxglove-layout.json`
- `Dockerfile`

Suggested commit message:

```text
Add inference trace recording and Foxglove debugging
```
