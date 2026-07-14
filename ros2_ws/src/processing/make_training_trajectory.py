#!/usr/bin/env python3
"""
make_training_trajectory.py — turn a recorded run into a training array.

Pipeline:
  1. Read every /joint_states message from the bag (positions only).
  2. Reorder each message's positions into one consistent joint layout
     (looked up by joint name, so a publisher reordering can't corrupt columns).
  3. Downsample onto a uniform grid (nearest sample to each grid time), so dt
     is constant and the deltas are comparable.
  4. Compute deltas between subsequent samples: delta[i] = pos[i+1] - pos[i].
  5. If present, read /zed/zed_node/depth/depth_registered and resample it
     onto the *same* grid (nearest depth frame to each grid tick), so every
     row of `depth` is the observation paired with that row of `positions`.
  6. If present, read /bc_pipeline/current_step and sync step labels to the
     same grid, so steps[i] is the step being executed at timestamps[i].
  7. Save everything to <run>/training_trajectory.npz.

The .npz holds (keys):
  joint_names : (J,)      column labels for the position/delta matrices
  timestamps  : (N,)      seconds from the start of the run, spaced 1/rate_hz
  positions   : (N, J)    downsampled joint positions
  deltas      : (N-1, J)  position change between subsequent samples
  rate_hz     : scalar    the downsample rate
  depth       : (N, H, W) float32, metres — only present if the depth topic
                was recorded; row i is synced to positions[i] (same grid tick)
  steps       : (N,)      step labels (str) — only present if the current_step
                topic was recorded; row i is the step at timestamps[i]

Must run inside the container (ROS 2 sourced) — reading the bag needs the
sensor_msgs definitions to deserialize the CDR payloads.

USAGE
-----
    python3 make_training_trajectory.py /root/ros2_ws/runs/drawer_2026-06-20_14-14-03

    # Batch process all runs in a parent directory
    python3 make_training_trajectory.py /root/ros2_ws/runs --batch

    # Force reprocessing even if output already exists
    python3 make_training_trajectory.py /root/ros2_ws/runs --batch --ignore-existing
"""

import argparse
import os
import sys

import numpy as np

# Reuse the bag-locating + metadata helpers from inspect_run.py (same folder).
from inspect_run import find_bag_dir, load_metadata

JOINT_STATE_TYPE = 'sensor_msgs/msg/JointState'
DEPTH_IMAGE_TYPE = 'sensor_msgs/msg/Image'
CURRENT_STEP_TYPE = 'std_msgs/msg/String'
DEPTH_TOPIC = '/zed/zed_node/depth/depth_registered'
CURRENT_STEP_TOPIC = '/bc_pipeline/current_step'
RATE_HZ = 20.0
OUTPUT_NAME = 'training_trajectory.npz'


def find_joint_states_topic(meta: dict) -> dict:
    """Pick the JointState topic from the bag metadata (prefer /joint_states)."""
    candidates = [t for t in meta['topics'] if t['type'] == JOINT_STATE_TYPE]
    if not candidates:
        raise SystemExit(f"No {JOINT_STATE_TYPE} topic found in this bag.")
    for t in candidates:
        if t['name'] == '/joint_states':
            return t
    return candidates[0]


def find_depth_topic(meta: dict):
    """Return the depth topic's metadata entry, or None if it wasn't recorded."""
    for t in meta['topics']:
        if t['name'] == DEPTH_TOPIC:
            if t['type'] != DEPTH_IMAGE_TYPE:
                raise SystemExit(
                    f"{DEPTH_TOPIC} has unexpected type {t['type']!r} "
                    f"(expected {DEPTH_IMAGE_TYPE!r}).")
            return t
    return None


def find_current_step_topic(meta: dict):
    """Return the current step topic's metadata entry, or None if it wasn't recorded."""
    for t in meta['topics']:
        if t['name'] == CURRENT_STEP_TOPIC:
            if t['type'] != CURRENT_STEP_TYPE:
                raise SystemExit(
                    f"{CURRENT_STEP_TOPIC} has unexpected type {t['type']!r} "
                    f"(expected {CURRENT_STEP_TYPE!r}).")
            return t
    return None


def read_joint_positions(bag_dir: str, storage_id: str, topic: str):
    """Read all messages of `topic`, returning (joint_names, times, positions)."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage_id),
        rosbag2_py.ConverterOptions('cdr', 'cdr'),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

    msg_cls = get_message(JOINT_STATE_TYPE)
    joint_names = None
    times, rows = [], []

    while reader.has_next():
        _topic, data, _stamp = reader.read_next()
        msg = deserialize_message(data, msg_cls)

        # Fix the column layout from the first message that carries names.
        if joint_names is None:
            if not msg.name:
                continue
            joint_names = list(msg.name)

        name_to_pos = dict(zip(msg.name, msg.position))
        if not all(n in name_to_pos for n in joint_names):
            continue  # message missing one of our joints — skip it

        rows.append([name_to_pos[n] for n in joint_names])
        times.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

    if not rows:
        raise SystemExit(f"No usable messages on {topic}.")
    return joint_names, np.asarray(times), np.asarray(rows)


def decode_depth_image(msg) -> np.ndarray:
    """Decode one sensor_msgs/Image depth frame to a (H, W) float32 array, metres."""
    if msg.encoding == '32FC1':
        dtype = np.dtype(np.float32)
    elif msg.encoding == '16UC1':
        dtype = np.dtype(np.uint16)
    else:
        raise SystemExit(f"Unsupported depth image encoding: {msg.encoding!r}")
    if msg.is_bigendian:
        dtype = dtype.newbyteorder('>')

    row_stride = msg.step // dtype.itemsize  # step may include row padding
    frame = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(msg.height, row_stride)
    frame = frame[:, :msg.width].astype(np.float32)
    if msg.encoding == '16UC1':
        frame /= 1000.0  # millimetres -> metres
    return frame


def read_depth_images(bag_dir: str, storage_id: str, topic: str):
    """Read all depth frames of `topic`, returning (times, frames (N, H, W))."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage_id),
        rosbag2_py.ConverterOptions('cdr', 'cdr'),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

    msg_cls = get_message(DEPTH_IMAGE_TYPE)
    times, frames = [], []

    while reader.has_next():
        _topic, data, _stamp = reader.read_next()
        msg = deserialize_message(data, msg_cls)
        frames.append(decode_depth_image(msg))
        times.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

    if not frames:
        raise SystemExit(f"No usable messages on {topic}.")
    return np.asarray(times), np.stack(frames, axis=0)


def read_current_steps(bag_dir: str, storage_id: str, topic: str):
    """Read all step messages, returning (times, step_labels)."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage_id),
        rosbag2_py.ConverterOptions('cdr', 'cdr'),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

    msg_cls = get_message(CURRENT_STEP_TYPE)
    times, steps = [], []

    while reader.has_next():
        _topic, data, stamp = reader.read_next()
        msg = deserialize_message(data, msg_cls)
        steps.append(msg.data)
        times.append(stamp * 1e-9)

    if not steps:
        raise SystemExit(f"No usable messages on {topic}.")
    return np.asarray(times), np.asarray(steps)




def nearest_indices(times: np.ndarray, abs_grid: np.ndarray) -> np.ndarray:
    """For each grid time, the index into (ascending) `times` of the closest sample."""
    right = np.clip(np.searchsorted(times, abs_grid), 1, len(times) - 1)
    left = right - 1
    pick_left = (abs_grid - times[left]) <= (times[right] - abs_grid)
    return np.where(pick_left, left, right)


def downsample(times: np.ndarray, positions: np.ndarray, rate_hz: float):
    """Resample onto a uniform grid by picking the nearest sample to each tick."""
    order = np.argsort(times)          # ensure ascending time
    times, positions = times[order], positions[order]

    t0 = times[0]
    grid = np.arange(0.0, times[-1] - t0, 1.0 / rate_hz)   # seconds from start
    abs_grid = t0 + grid

    nearest = nearest_indices(times, abs_grid)
    return grid, positions[nearest], abs_grid


def process_trajectory(save_kwargs: dict) -> dict:
    """Process trajectory arrays by removing the first step if it's a Checkpoint to home.

    Removes all samples that belong to the first step only if it is a Checkpoint
    step with target "home". Otherwise returns the data unchanged.
    """
    processed = save_kwargs.copy()

    # Only process if steps are available
    if 'steps' not in processed or len(processed['steps']) == 0:
        return processed

    first_step = processed['steps'][0]

    # Only remove first step if it's a Checkpoint to home (case-insensitive)
    first_step_lower = first_step.lower()
    if not ('checkpoint' in first_step_lower and 'home' in first_step_lower):
        return processed

    # Find indices where step is NOT the first step
    keep_mask = processed['steps'] != first_step

    # Apply mask to all position-aligned arrays
    processed['positions'] = processed['positions'][keep_mask]
    processed['timestamps'] = processed['timestamps'][keep_mask]
    processed['steps'] = processed['steps'][keep_mask]

    if 'depth' in processed:
        processed['depth'] = processed['depth'][keep_mask]
    if 'is_gripping' in processed:
        processed['is_gripping'] = processed['is_gripping'][keep_mask]

    # Recompute deltas from the filtered positions
    if len(processed['positions']) > 1:
        processed['deltas'] = np.diff(processed['positions'], axis=0)
    else:
        processed['deltas'] = np.array([]).reshape(0, processed['positions'].shape[1])

    return processed


def build_is_gripping(down_steps: np.ndarray) -> np.ndarray:
    """Build a boolean array indicating gripper state from synced step labels.

    Uses the synced step labels (down_steps) to ensure alignment with the grid.
    Starts as False. 'grip' in label sets to True, 'blow' sets to False.
    """
    is_gripping = np.zeros(len(down_steps), dtype=bool)
    current_gripping = False

    for i, label in enumerate(down_steps):
        label_lower = label.lower()
        # Check 'blow' first because "Gripper" contains "grip"
        if 'blow' in label_lower:
            current_gripping = False
        elif 'grip' in label_lower:
            current_gripping = True
        # 'release' is ignored
        is_gripping[i] = current_gripping

    return is_gripping


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag', help='Bag directory or a storage file inside it.')
    parser.add_argument('--batch', action='store_true',
                        help='Treat bag as a parent path; scan subdirs and process all runs '
                             'that don\'t yet have training_trajectory.npz.')
    parser.add_argument('--ignore-existing', action='store_true',
                        help='Reprocess even if training_trajectory.npz already exists.')
    parser.add_argument('--render', action='store_true',
                        help='Generate visualization after processing.')
    args = parser.parse_args()

    if args.batch:
        process_batch(args.bag, args.ignore_existing, args.render)
    else:
        process_single(args.bag, args.render)


def process_single(bag_path, render=False):
    bag_dir = find_bag_dir(bag_path)
    meta = load_metadata(bag_dir)
    topic = find_joint_states_topic(meta)

    joint_names, times, positions = read_joint_positions(
        bag_dir, meta['storage_id'], topic['name'])
    print(f"Read {len(positions)} {topic['name']} messages "
          f"({positions.shape[1]} joints): {joint_names}")

    grid, down_positions, abs_grid = downsample(times, positions, RATE_HZ)
    deltas = np.diff(down_positions, axis=0)
    print(f"Downsampled to {RATE_HZ:.0f} Hz: {down_positions.shape[0]} samples "
          f"over {grid[-1]:.2f} s; deltas {deltas.shape}.")

    save_kwargs = dict(
        joint_names=np.asarray(joint_names),
        timestamps=grid,
        positions=down_positions,
        deltas=deltas,
        rate_hz=np.asarray(RATE_HZ),
    )

    depth_topic = find_depth_topic(meta)
    if depth_topic is None:
        print(f"No {DEPTH_TOPIC} topic in this bag — skipping depth.")
    else:
        depth_times, depth_frames = read_depth_images(
            bag_dir, meta['storage_id'], depth_topic['name'])
        order = np.argsort(depth_times)
        depth_times, depth_frames = depth_times[order], depth_frames[order]
        print(f"Read {len(depth_frames)} {depth_topic['name']} messages "
              f"({depth_frames.shape[1]}x{depth_frames.shape[2]}).")

        # Sync to the *same* grid ticks used for the joint positions, so
        # depth[i] and positions[i] are the observation/state for one tick.
        depth_idx = nearest_indices(depth_times, abs_grid)
        down_depth = depth_frames[depth_idx]
        offsets = np.abs(depth_times[depth_idx] - abs_grid)
        print(f"Synced depth to the joint grid: mean offset {offsets.mean():.3f} s, "
              f"max offset {offsets.max():.3f} s.")
        save_kwargs['depth'] = down_depth

    step_topic = find_current_step_topic(meta)
    if step_topic is None:
        print(f"No {CURRENT_STEP_TOPIC} topic in this bag — skipping steps and gripper state.")
    else:
        step_times, step_labels = read_current_steps(
            bag_dir, meta['storage_id'], step_topic['name'])
        order = np.argsort(step_times)
        step_times, step_labels = step_times[order], step_labels[order]
        print(f"Read {len(step_labels)} {step_topic['name']} messages.")

        # Sync steps to the same grid: each grid tick gets the current step label.
        step_idx = nearest_indices(step_times, abs_grid)
        down_steps = step_labels[step_idx]
        offsets = np.abs(step_times[step_idx] - abs_grid)
        print(f"Synced steps to the joint grid: mean offset {offsets.mean():.3f} s, "
              f"max offset {offsets.max():.3f} s.")
        save_kwargs['steps'] = down_steps

        # Infer gripper state from synced step labels (ensures alignment)
        is_gripping = build_is_gripping(down_steps)
        n_gripping = np.sum(is_gripping)
        print(f"Built is_gripping: {n_gripping}/{len(is_gripping)} grid points with gripper active.")

        save_kwargs['is_gripping'] = is_gripping

    # Process trajectory (e.g., remove first step)
    save_kwargs = process_trajectory(save_kwargs)

    out_path = os.path.join(bag_dir, OUTPUT_NAME)
    np.savez(out_path, **save_kwargs)
    print(f"Saved {out_path}")

    if render:
        try:
            import visualize_run
            visualize_run.process_single(bag_dir, draw_every=20)
        except Exception as e:
            print(f"Warning: failed to render visualization: {e}")


def process_batch(parent_path, ignore_existing=False, render=False):
    """Scan parent_path for subdirectories and process runs without training_trajectory.npz."""
    if not os.path.isdir(parent_path):
        raise SystemExit(f"{parent_path} is not a directory.")

    subdirs = sorted([
        os.path.join(parent_path, d)
        for d in os.listdir(parent_path)
        if os.path.isdir(os.path.join(parent_path, d))
    ])

    if not subdirs:
        print(f"No subdirectories found in {parent_path}")
        return

    to_process = []
    for subdir in subdirs:
        out_path = os.path.join(subdir, OUTPUT_NAME)
        if os.path.exists(out_path) and not ignore_existing:
            print(f"Skip {subdir} (already has {OUTPUT_NAME})")
        else:
            to_process.append(subdir)

    if not to_process:
        print(f"All {len(subdirs)} subdirectories already processed.")
        return

    print(f"Found {len(to_process)}/{len(subdirs)} runs to process.\n")

    for i, run_path in enumerate(to_process, 1):
        print(f"\n[{i}/{len(to_process)}] Processing {os.path.basename(run_path)}...")
        try:
            process_single(run_path, render)
        except Exception as e:
            print(f"ERROR processing {run_path}: {e}")


if __name__ == '__main__':
    main()
