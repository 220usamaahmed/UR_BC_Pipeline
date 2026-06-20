#!/usr/bin/env python3
"""
make_training_trajectory.py — turn a recorded run into a training array.

Pipeline:
  1. Read every /joint_states message from the bag (positions only).
  2. Reorder each message's positions into one consistent joint layout
     (looked up by joint name, so a publisher reordering can't corrupt columns).
  3. Downsample onto a uniform 10 Hz grid (nearest sample to each grid time),
     so dt is constant and the deltas are comparable.
  4. Compute deltas between subsequent samples: delta[i] = pos[i+1] - pos[i].
  5. Save everything to <run>/training_trajectory.npz.

The .npz holds (keys):
  joint_names : (J,)    column labels for the position/delta matrices
  timestamps  : (N,)    seconds from the start of the run, spaced 1/10 s
  positions   : (N, J)  downsampled joint positions
  deltas      : (N-1, J) position change between subsequent samples
  rate_hz     : scalar  the downsample rate (10.0)

Must run inside the container (ROS 2 sourced) — reading the bag needs the
sensor_msgs definitions to deserialize the CDR payloads.

USAGE
-----
    python3 make_training_trajectory.py /root/ros2_ws/runs/drawer_2026-06-20_14-14-03
"""

import argparse
import os

import numpy as np

# Reuse the bag-locating + metadata helpers from inspect_run.py (same folder).
from inspect_run import find_bag_dir, load_metadata

JOINT_STATE_TYPE = 'sensor_msgs/msg/JointState'
RATE_HZ = 10.0
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


def downsample(times: np.ndarray, positions: np.ndarray, rate_hz: float):
    """Resample onto a uniform grid by picking the nearest sample to each tick."""
    order = np.argsort(times)          # ensure ascending time
    times, positions = times[order], positions[order]

    t0 = times[0]
    grid = np.arange(0.0, times[-1] - t0, 1.0 / rate_hz)   # seconds from start
    abs_grid = t0 + grid

    right = np.clip(np.searchsorted(times, abs_grid), 1, len(times) - 1)
    left = right - 1
    pick_left = (abs_grid - times[left]) <= (times[right] - abs_grid)
    nearest = np.where(pick_left, left, right)
    return grid, positions[nearest]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag', help='Bag directory or a storage file inside it.')
    args = parser.parse_args()

    bag_dir = find_bag_dir(args.bag)
    meta = load_metadata(bag_dir)
    topic = find_joint_states_topic(meta)

    joint_names, times, positions = read_joint_positions(
        bag_dir, meta['storage_id'], topic['name'])
    print(f"Read {len(positions)} {topic['name']} messages "
          f"({positions.shape[1]} joints): {joint_names}")

    grid, down_positions = downsample(times, positions, RATE_HZ)
    deltas = np.diff(down_positions, axis=0)
    print(f"Downsampled to {RATE_HZ:.0f} Hz: {down_positions.shape[0]} samples "
          f"over {grid[-1]:.2f} s; deltas {deltas.shape}.")

    out_path = os.path.join(bag_dir, OUTPUT_NAME)
    np.savez(
        out_path,
        joint_names=np.asarray(joint_names),
        timestamps=grid,
        positions=down_positions,
        deltas=deltas,
        rate_hz=np.asarray(RATE_HZ),
    )
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
