#!/usr/bin/env python3
"""Print timing summaries for recorded trajectories below a parent folder."""

import argparse
import os
import sys

import numpy as np


NPZ_NAME = "dataset.npz"
TARGET_FRAME_RATE_HZ = 15.0
FRAME_RATE_TOLERANCE_HZ = 0.1
RED = "\033[31m"
RESET = "\033[0m"


def find_npz_files(parent: str) -> list[str]:
    """Recursively find all recorded trajectory files below ``parent``."""
    resolved = os.path.abspath(os.path.expanduser(parent))
    if not os.path.isdir(resolved):
        raise SystemExit(f"Path is not a directory: {parent}")

    matches = []
    for directory, directory_names, filenames in os.walk(resolved):
        directory_names.sort()
        if NPZ_NAME in filenames:
            matches.append(os.path.join(directory, NPZ_NAME))
    return sorted(matches)


def summarize(npz_path: str) -> tuple[int, float, float, float]:
    """Return frame count, time span, average delta, and average frame rate."""
    with np.load(npz_path) as data:
        if "timestamps" not in data.files:
            raise ValueError("missing timestamps array")
        timestamps = np.asarray(data["timestamps"], dtype=np.float64)

    if timestamps.ndim != 1:
        raise ValueError(f"timestamps must be one-dimensional; got {timestamps.shape}")

    frame_count = len(timestamps)
    if frame_count == 0:
        return 0, 0.0, 0.0, 0.0
    if frame_count == 1:
        return 1, 0.0, 0.0, 0.0

    deltas = np.diff(timestamps)
    time_span = float(timestamps[-1] - timestamps[0])
    average_delta = float(np.mean(deltas))
    average_frame_rate = 1.0 / average_delta if average_delta > 0.0 else 0.0
    return frame_count, time_span, average_delta, average_frame_rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", help="Parent directory containing recorded trajectories")
    args = parser.parse_args()

    npz_files = find_npz_files(args.parent)
    if not npz_files:
        print(f"No {NPZ_NAME} files found below {args.parent}")
        return

    parent = os.path.abspath(os.path.expanduser(args.parent))
    failures = 0
    unexpected_frame_rates = []
    for npz_path in npz_files:
        trajectory = os.path.relpath(os.path.dirname(npz_path), parent)
        try:
            frames, time_span, average_delta, average_frame_rate = summarize(npz_path)
        except (OSError, ValueError, KeyError) as exc:
            print(f"{trajectory}: ERROR: {exc}", file=sys.stderr)
            failures += 1
            continue

        print(
            f"{trajectory}: frames={frames}, time_span={time_span:.6f} s, "
            f"average_delta={average_delta:.6f} s, "
            f"average_frame_rate={average_frame_rate:.3f} Hz"
        )
        if not np.isclose(
            average_frame_rate,
            TARGET_FRAME_RATE_HZ,
            rtol=0.0,
            atol=FRAME_RATE_TOLERANCE_HZ,
        ):
            unexpected_frame_rates.append((trajectory, average_frame_rate))

    if unexpected_frame_rates:
        print(
            f"\n{RED}Trajectories outside {TARGET_FRAME_RATE_HZ:.1f} ± "
            f"{FRAME_RATE_TOLERANCE_HZ:.1f} Hz:{RESET}"
        )
        for trajectory, average_frame_rate in unexpected_frame_rates:
            print(f"{RED}{trajectory}: {average_frame_rate:.3f} Hz{RESET}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
