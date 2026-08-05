#!/usr/bin/env python3
"""Create a static PNG or animated GIF from a recorded trajectory.

In single mode, the input may be either a trajectory directory or the NPZ file
itself. In batch mode, every ``dataset.npz`` below the input directory is
processed unless its visualization already exists. Results are saved as
``trajectory_visualization.png`` or ``trajectory_visualization.gif`` beside
each NPZ file.
"""

import argparse
import os
import sys
from typing import Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch


NPZ_NAME = "dataset.npz"
OUTPUT_NAME = "trajectory_visualization.png"
GIF_OUTPUT_NAME = "trajectory_visualization.gif"
JOINT_COUNT = 6
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)
GRIPPER_COLOR = "tab:orange"


def find_npz(path: str) -> str:
    """Resolve a trajectory directory or NPZ path to an NPZ file."""
    resolved = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(resolved):
        return resolved
    candidate = os.path.join(resolved, NPZ_NAME)
    if os.path.isfile(candidate):
        return candidate
    raise SystemExit(f"No {NPZ_NAME} found at or in {path}")


def find_npz_files(parent: str) -> list[str]:
    """Recursively find trajectory NPZ files below a parent directory."""
    resolved = os.path.abspath(os.path.expanduser(parent))
    if not os.path.isdir(resolved):
        raise SystemExit(f"Batch path is not a directory: {parent}")

    matches = []
    for directory, directory_names, filenames in os.walk(resolved):
        directory_names.sort()
        if NPZ_NAME in filenames:
            matches.append(os.path.join(directory, NPZ_NAME))
    return sorted(matches)


def validate_arrays(data: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, ...]:
    """Load and validate the arrays needed by the visualization."""
    required = ("observations", "actions", "timestamps", "depth_frames")
    missing = [name for name in required if name not in data.files]
    if missing:
        raise SystemExit(f"Dataset is missing required arrays: {', '.join(missing)}")

    observations = data["observations"]
    actions = data["actions"]
    timestamps = data["timestamps"]
    depth_frames = data["depth_frames"]

    if observations.ndim != 2 or observations.shape[1] <= JOINT_COUNT:
        raise SystemExit(
            f"Expected observations shaped (N, >{JOINT_COUNT}); got {observations.shape}"
        )
    if actions.ndim != 2 or actions.shape[1] <= JOINT_COUNT:
        raise SystemExit(
            f"Expected actions shaped (N, >{JOINT_COUNT}); got {actions.shape}"
        )
    if timestamps.ndim != 1:
        raise SystemExit(f"Expected one-dimensional timestamps; got {timestamps.shape}")
    if depth_frames.ndim != 3 or len(depth_frames) == 0:
        raise SystemExit(
            f"Expected non-empty depth frames shaped (N, H, W); got {depth_frames.shape}"
        )

    sample_count = len(timestamps)
    if len(observations) != sample_count or len(actions) != sample_count:
        raise SystemExit(
            "Observations, actions, and timestamps must have the same sample count."
        )
    return observations, actions, timestamps, depth_frames


def gripper_is_on(gripper_values: np.ndarray) -> np.ndarray:
    """Build the latched gripper state from scalar or one-hot commands.

    Grip (state 1) turns the state on, blow (state 3) turns it off, and release
    (state 2) leaves it unchanged.
    """
    if gripper_values.ndim != 2 or gripper_values.shape[1] == 0:
        raise ValueError("Expected at least one gripper column")

    if gripper_values.shape[1] >= 3:
        grip_commands = np.isclose(gripper_values[:, 0], 1.0)
        blow_commands = np.isclose(gripper_values[:, 2], 1.0)
    else:
        grip_commands = np.isclose(gripper_values[:, 0], 1.0)
        blow_commands = np.isclose(gripper_values[:, 0], 3.0)

    active = np.zeros(len(gripper_values), dtype=bool)
    is_on = False
    for index, (grip, blow) in enumerate(zip(grip_commands, blow_commands)):
        if grip:
            is_on = True
        elif blow:
            is_on = False
        active[index] = is_on
    return active


def add_gripper_regions(ax, time: np.ndarray, active: np.ndarray) -> None:
    """Shade every contiguous interval in which gripper state one is active."""
    if len(time) == 0 or not np.any(active):
        return

    transitions = np.diff(active.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    typical_step = np.median(np.diff(time)) if len(time) > 1 else 0.0

    for start, stop in zip(starts, stops):
        left = time[start]
        right = time[stop] if stop < len(time) else time[-1] + typical_step
        ax.axvspan(left, right, color=GRIPPER_COLOR, alpha=0.2, linewidth=0)


def plot_trajectory(npz_path: str, output_path: str) -> None:
    """Render and save the trajectory overview."""
    with np.load(npz_path) as data:
        observations, actions, timestamps, depth_frames = validate_arrays(data)

        time = timestamps.astype(np.float64) - float(timestamps[0])
        observation_joints = observations[:, :JOINT_COUNT]
        action_joints = actions[:, :JOINT_COUNT]
        observation_gripper = gripper_is_on(observations[:, JOINT_COUNT:])
        action_gripper = gripper_is_on(actions[:, JOINT_COUNT:])
        last_depth = np.asarray(depth_frames[-1], dtype=np.float32)

    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.05, 1.7))
    depth_ax = fig.add_subplot(grid[:, 0])
    observation_ax = fig.add_subplot(grid[0, 1])
    action_ax = fig.add_subplot(grid[1, 1], sharex=observation_ax)

    finite_depth = last_depth[np.isfinite(last_depth)]
    if finite_depth.size:
        vmin, vmax = np.percentile(finite_depth, (1.0, 99.0))
        if vmin == vmax:
            vmin, vmax = float(finite_depth.min()), float(finite_depth.max())
        image = depth_ax.imshow(last_depth, cmap="viridis", vmin=vmin, vmax=vmax)
    else:
        image = depth_ax.imshow(last_depth, cmap="viridis")
    depth_ax.set_title("Last depth frame")
    depth_ax.set_xlabel("pixel x")
    depth_ax.set_ylabel("pixel y")
    fig.colorbar(image, ax=depth_ax, label="depth")

    for index, name in enumerate(JOINT_NAMES):
        observation_ax.plot(time, observation_joints[:, index], label=name, linewidth=1.2)
        action_ax.plot(time, action_joints[:, index], label=name, linewidth=1.2)

    add_gripper_regions(observation_ax, time, observation_gripper)
    add_gripper_regions(action_ax, time, action_gripper)

    gripper_patch = Patch(
        facecolor=GRIPPER_COLOR,
        alpha=0.2,
        label="gripper on",
    )
    handles, labels = observation_ax.get_legend_handles_labels()
    observation_ax.legend(
        handles + [gripper_patch],
        labels + [gripper_patch.get_label()],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize="small",
    )

    observation_ax.set_title("Joint angle observations")
    observation_ax.set_ylabel("angle (rad)")
    observation_ax.grid(alpha=0.25)
    observation_ax.tick_params(labelbottom=False)

    action_ax.set_title("Joint actions")
    action_ax.set_xlabel("time from recording start (s)")
    action_ax.set_ylabel("action")
    action_ax.grid(alpha=0.25)

    fig.suptitle(os.path.basename(os.path.dirname(npz_path)))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def animate_trajectory(
    npz_path: str,
    output_path: str,
    skip_frames: int,
    fps: float,
) -> None:
    """Render a depth animation with synchronized chart cursors."""
    if skip_frames <= 0:
        raise ValueError("skip_frames must be greater than zero")
    if fps <= 0:
        raise ValueError("fps must be greater than zero")

    with np.load(npz_path) as data:
        observations, actions, timestamps, depth_frames = validate_arrays(data)
        if len(depth_frames) != len(timestamps):
            raise ValueError(
                "Depth frames and timestamps must have the same sample count for GIF mode."
            )

        time = timestamps.astype(np.float64) - float(timestamps[0])
        observation_joints = observations[:, :JOINT_COUNT]
        action_joints = actions[:, :JOINT_COUNT]
        observation_gripper = gripper_is_on(observations[:, JOINT_COUNT:])
        action_gripper = gripper_is_on(actions[:, JOINT_COUNT:])
        depth_frames = np.asarray(depth_frames)

    frame_indices = np.arange(0, len(time), skip_frames, dtype=int)
    if frame_indices[-1] != len(time) - 1:
        frame_indices = np.append(frame_indices, len(time) - 1)

    # Estimate stable color limits from downsampled pixels in rendered frames.
    depth_sample = depth_frames[frame_indices, ::4, ::4]
    finite_depth = depth_sample[np.isfinite(depth_sample)]
    if finite_depth.size:
        vmin, vmax = np.percentile(finite_depth, (1.0, 99.0))
        if vmin == vmax:
            vmin, vmax = float(finite_depth.min()), float(finite_depth.max())
    else:
        vmin = vmax = None

    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.05, 1.7))
    depth_ax = fig.add_subplot(grid[:, 0])
    observation_ax = fig.add_subplot(grid[0, 1])
    action_ax = fig.add_subplot(grid[1, 1], sharex=observation_ax)

    image = depth_ax.imshow(
        depth_frames[frame_indices[0]],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        animated=True,
    )
    depth_ax.set_xlabel("pixel x")
    depth_ax.set_ylabel("pixel y")
    fig.colorbar(image, ax=depth_ax, label="depth")

    for index, name in enumerate(JOINT_NAMES):
        observation_ax.plot(time, observation_joints[:, index], label=name, linewidth=1.2)
        action_ax.plot(time, action_joints[:, index], label=name, linewidth=1.2)

    add_gripper_regions(observation_ax, time, observation_gripper)
    add_gripper_regions(action_ax, time, action_gripper)
    gripper_patch = Patch(facecolor=GRIPPER_COLOR, alpha=0.2, label="gripper on")
    handles, labels = observation_ax.get_legend_handles_labels()
    observation_ax.legend(
        handles + [gripper_patch],
        labels + [gripper_patch.get_label()],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize="small",
    )

    observation_ax.set_title("Joint angle observations")
    observation_ax.set_ylabel("angle (rad)")
    observation_ax.grid(alpha=0.25)
    observation_ax.tick_params(labelbottom=False)
    action_ax.set_title("Joint actions")
    action_ax.set_xlabel("time from recording start (s)")
    action_ax.set_ylabel("action")
    action_ax.grid(alpha=0.25)

    observation_cursor = observation_ax.axvline(time[0], color="black", linewidth=1.5)
    action_cursor = action_ax.axvline(time[0], color="black", linewidth=1.5)
    fig.suptitle(os.path.basename(os.path.dirname(npz_path)))

    def update(sample_index: int):
        current_time = time[sample_index]
        image.set_data(depth_frames[sample_index])
        depth_ax.set_title(
            f"Depth frame {sample_index + 1}/{len(depth_frames)} "
            f"({current_time:.2f} s)"
        )
        observation_cursor.set_xdata([current_time, current_time])
        action_cursor.set_xdata([current_time, current_time])
        return image, observation_cursor, action_cursor

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=1000.0 / fps,
        blit=False,
    )
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)


def plot_batch(
    parent: str,
    output_name: str,
    ignore_existing: bool,
    gif: bool,
    skip_frames: int,
    fps: float,
) -> int:
    """Plot missing visualizations for every trajectory below ``parent``."""
    npz_files = find_npz_files(parent)
    if not npz_files:
        print(f"No {NPZ_NAME} files found below {parent}")
        return 0

    created = 0
    skipped = 0
    failed = 0
    for npz_path in npz_files:
        output_path = os.path.join(os.path.dirname(npz_path), output_name)
        if os.path.exists(output_path) and not ignore_existing:
            print(f"Skipping existing visualization: {output_path}")
            skipped += 1
            continue

        try:
            if gif:
                animate_trajectory(npz_path, output_path, skip_frames, fps)
            else:
                plot_trajectory(npz_path, output_path)
        except Exception as exc:  # noqa: BLE001 - keep processing other trajectories
            print(f"Failed to visualize {npz_path}: {exc}", file=sys.stderr)
            failed += 1
            continue

        print(f"Saved visualization to {output_path}")
        created += 1

    print(
        f"Batch complete: {created} created, {skipped} skipped, {failed} failed."
    )
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trajectory",
        help="Trajectory directory/NPZ, or a parent directory with --batch",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Recursively process missing visualizations below the parent directory",
    )
    parser.add_argument(
        "--ignore-existing",
        action="store_true",
        help="In batch mode, recreate visualizations even when they already exist",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Create an animated GIF instead of a static PNG",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=20,
        metavar="N",
        help="In GIF mode, render every Nth recorded frame (default: 20)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=4.0,
        help="GIF playback rate in frames per second (default: 4)",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output filename saved beside the NPZ (default depends on mode)",
    )
    args = parser.parse_args()
    output_name = args.output_name or (GIF_OUTPUT_NAME if args.gif else OUTPUT_NAME)

    if args.batch:
        raise SystemExit(
            plot_batch(
                args.trajectory,
                output_name,
                args.ignore_existing,
                args.gif,
                args.skip_frames,
                args.fps,
            )
        )

    npz_path = find_npz(args.trajectory)
    output_path = os.path.join(os.path.dirname(npz_path), output_name)
    if args.gif:
        animate_trajectory(npz_path, output_path, args.skip_frames, args.fps)
    else:
        plot_trajectory(npz_path, output_path)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
