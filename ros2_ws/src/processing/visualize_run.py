#!/usr/bin/env python3
"""
visualize_run.py — animate a run's depth frames alongside its joint traces.

Reads the training_trajectory.npz written by make_training_trajectory.py and
renders a synced side-by-side animation: the depth frame on the left, each
joint's position trace on the right with a moving cursor marking "now". One
video frame per saved sample, played back at rate_hz so the video runs at the
same speed as the recording.

If the .npz contains step information, the joint position graph shows colored
background rectangles to mark which step was active during each time period,
making it easy to correlate sensor data and joint motions with specific steps.

Only needs numpy + matplotlib (no ROS), so this runs fine on the host inside
the repo's .venv:

USAGE
-----
    .venv/bin/python3 ros2_ws/src/processing/visualize_run.py \\
        ros2_ws/runs/drawer_2026-06-20_14-14-03

    # Batch visualize all runs with training data
    .venv/bin/python3 ros2_ws/src/processing/visualize_run.py \\
        ros2_ws/runs --batch

    # Force recreate visualization even if it exists
    .venv/bin/python3 ros2_ws/src/processing/visualize_run.py \\
        ros2_ws/runs --batch --ignore-existing

Saves <run>/visualization.mp4 next to the .npz (falls back to an animated
.gif if the `ffmpeg` binary isn't on PATH).
"""

import argparse
import os
import shutil

import matplotlib
matplotlib.use('Agg')  # no display available when run headless / over SSH

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

NPZ_NAME = 'training_trajectory.npz'
OUTPUT_STEM = 'visualization'


def _simplify_step_label(label: str) -> str:
    """Strip offset/along details from step labels, keeping just the step type and name."""
    # Remove everything after "offset" or "along" keywords
    for keyword in [' offset', ' along']:
        if keyword in label:
            label = label[:label.index(keyword)]
    return label.strip()


def find_npz(path: str) -> str:
    """Return the path to training_trajectory.npz given a run dir or the file itself."""
    if os.path.isfile(path):
        return path
    candidate = os.path.join(path, NPZ_NAME)
    if os.path.isfile(candidate):
        return candidate
    raise SystemExit(f"No {NPZ_NAME} found at or in {path} — run "
                      f"make_training_trajectory.py on this run first.")


def build_step_rectangles(ax, timestamps, steps, step_color_map, grip_ymin=None, grip_ymax=None):
    """Draw background rectangles for each step transition, avoiding gripping area."""
    if steps is None or len(steps) == 0:
        return []

    # Find step transitions: where the step label changes
    step_changes = [0]  # first step always starts at time 0
    for i in range(1, len(steps)):
        if steps[i] != steps[i-1]:
            step_changes.append(i)
    step_changes.append(len(steps))  # end marker

    ymin, ymax = ax.get_ylim()
    # If gripping rectangles exist, only draw steps above them
    if grip_ymin is not None and grip_ymax is not None:
        step_ymin = grip_ymax
        step_ymax = ymax
    else:
        step_ymin = ymin
        step_ymax = ymax

    rectangles = []
    for i in range(len(step_changes) - 1):
        start_idx = step_changes[i]
        end_idx = step_changes[i + 1]
        step_label = steps[start_idx]

        x_start = timestamps[start_idx]
        x_end = timestamps[end_idx - 1]

        rect = mpatches.Rectangle(
            (x_start, step_ymin),
            x_end - x_start,
            step_ymax - step_ymin,
            linewidth=0,
            edgecolor='none',
            facecolor=step_color_map[step_label],
            alpha=0.25,
            zorder=0,
        )
        ax.add_patch(rect)
        rectangles.append((rect, step_label))

    return rectangles


def build_gripping_rectangles(ax, timestamps, is_gripping):
    """Draw rectangles at the bottom of the plot where is_gripping is True (no overlap with step rects).

    Returns the gripping area bounds (grip_ymin, grip_ymax) so step rectangles can avoid this space.
    """
    if is_gripping is None or len(is_gripping) == 0:
        return [], None, None

    # Find gripping regions: where is_gripping changes
    gripping_changes = [0] if is_gripping[0] else []
    for i in range(1, len(is_gripping)):
        if is_gripping[i] != is_gripping[i-1]:
            gripping_changes.append(i)
    if is_gripping[-1]:
        gripping_changes.append(len(is_gripping))

    ymin, ymax = ax.get_ylim()
    # Reserve bottom 20% of plot for gripping indicator
    grip_height = (ymax - ymin) * 0.20
    grip_ymin = ymin
    grip_ymax = grip_ymin + grip_height

    rectangles = []
    gripping_color = (1.0, 0.65, 0.0)  # orange

    i = 0
    while i < len(gripping_changes):
        start_idx = gripping_changes[i]
        if is_gripping[start_idx]:  # only draw if gripping starts
            if i + 1 < len(gripping_changes):
                end_idx = gripping_changes[i + 1]
            else:
                end_idx = len(is_gripping)

            x_start = timestamps[start_idx]
            x_end = timestamps[end_idx - 1]

            rect = mpatches.Rectangle(
                (x_start, grip_ymin),
                x_end - x_start,
                grip_height,
                linewidth=0,
                edgecolor='none',
                facecolor=gripping_color,
                alpha=0.6,
                zorder=1,
            )
            ax.add_patch(rect)
            rectangles.append(rect)
            i += 2  # skip to next pair
        else:
            i += 1

    return rectangles, grip_ymin, grip_ymax


def build_animation(data: dict, draw_every: int):
    timestamps = data['timestamps']
    positions = data['positions']
    deltas = data['deltas']
    delta_times = timestamps[:-1]  # deltas[i] = positions[i+1] - positions[i]
    joint_names = [str(n) for n in data['joint_names']]
    depth = data.get('depth')
    steps = data.get('steps')
    is_gripping = data.get('is_gripping')
    n_frames = len(timestamps)

    has_depth = depth is not None

    frame_indices = np.arange(0, len(timestamps), draw_every)

    fig = plt.figure(figsize=(12, 6) if has_depth else (6, 6))
    if has_depth:
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])
        ax_depth = fig.add_subplot(gs[:, 0])
        ax_joints = fig.add_subplot(gs[0, 1])
        ax_deltas = fig.add_subplot(gs[1, 1], sharex=ax_joints)
    else:
        gs = fig.add_gridspec(2, 1)
        ax_joints = fig.add_subplot(gs[0, 0])
        ax_deltas = fig.add_subplot(gs[1, 0], sharex=ax_joints)

    im = None
    if has_depth:
        depth_clean = np.where(np.isfinite(depth), depth, np.nan)
        vmin, vmax = np.nanmin(depth_clean), np.nanmax(depth_clean)
        cmap = matplotlib.colormaps['viridis'].copy()
        cmap.set_bad('black')
        im = ax_depth.imshow(depth_clean[0], cmap=cmap, vmin=vmin, vmax=vmax)
        ax_depth.set_title('Depth (m)')
        ax_depth.set_xticks([])
        ax_depth.set_yticks([])
        fig.colorbar(im, ax=ax_depth, fraction=0.046, pad=0.04)

    # Build step color map first (used for both rectangles and legend)
    step_color_map = {}
    if steps is not None:
        unique_steps = []
        seen = set()
        for step in steps:
            if step not in seen:
                unique_steps.append(step)
                seen.add(step)
        colors = plt.cm.Set3(np.linspace(0, 1, max(len(unique_steps), 3)))
        step_color_map = {step: colors[i % len(colors)] for i, step in enumerate(unique_steps)}

    # Draw gripping and step background rectangles
    # Gripping goes first to determine the reserved space
    gripping_rectangles = []
    grip_ymin, grip_ymax = None, None
    if is_gripping is not None:
        gripping_rectangles, grip_ymin, grip_ymax = build_gripping_rectangles(ax_joints, timestamps, is_gripping)

    step_rectangles = []
    if steps is not None:
        step_rectangles = build_step_rectangles(ax_joints, timestamps, steps, step_color_map, grip_ymin, grip_ymax)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for j, name in enumerate(joint_names):
        color = colors[j % len(colors)]
        ax_joints.plot(timestamps, positions[:, j], label=name, linewidth=1, color=color)
        ax_deltas.plot(delta_times, deltas[:, j], linewidth=1, color=color)

    joint_cursor = ax_joints.axvline(timestamps[0], color='black', linewidth=1.5)
    delta_cursor = ax_deltas.axvline(delta_times[0], color='black', linewidth=1.5)

    ax_joints.set_ylabel('position (rad)')
    ax_joints.set_title('Joint positions')

    # Build legend with joint names and step regions (positioned outside the plot)
    joint_handles, joint_labels = ax_joints.get_legend_handles_labels()
    handles = joint_handles
    labels = list(joint_labels)

    if steps is not None and len(step_color_map) > 0:
        # Count occurrences of each unique step
        step_occurrence_count = {}
        for step in steps:
            step_occurrence_count[step] = step_occurrence_count.get(step, 0) + 1

        # Create patches with occurrence numbers for legend (in order they appear)
        step_patches = []
        step_labels_legend = []
        for step in step_color_map.keys():
            simplified = _simplify_step_label(step)
            occurrence_num = step_occurrence_count[step]
            if occurrence_num > 1:
                label = f"{simplified} ({occurrence_num}x)"
            else:
                label = simplified
            patch = mpatches.Patch(facecolor=step_color_map[step],
                                  label=label, alpha=0.25)
            step_patches.append(patch)
            step_labels_legend.append(label)

        handles = handles + step_patches
        labels = labels + step_labels_legend

    if is_gripping is not None and np.any(is_gripping):
        gripping_patch = mpatches.Patch(facecolor=(1.0, 0.65, 0.0),
                                        label='Gripping', alpha=0.25)
        handles.append(gripping_patch)
        labels.append('Gripping')

    ax_joints.legend(handles=handles, labels=labels, loc='upper left',
                    bbox_to_anchor=(1.02, 1), fontsize='small', frameon=True)

    ax_joints.tick_params(labelbottom=False)

    ax_deltas.set_xlabel('time (s)')
    ax_deltas.set_ylabel('delta (rad)')
    ax_deltas.set_title('Joint deltas')
    fig.tight_layout()

    last_delta_idx = len(delta_times) - 1

    def update(i):
        artists = [joint_cursor, delta_cursor]
        joint_cursor.set_xdata([timestamps[i], timestamps[i]])
        di = min(i, last_delta_idx)
        delta_cursor.set_xdata([delta_times[di], delta_times[di]])
        if has_depth:
            im.set_data(depth_clean[i])
            artists.append(im)
        return artists

    anim = FuncAnimation(fig, update, frames=frame_indices, blit=False)
    return fig, anim


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', help='Run directory, or a training_trajectory.npz path.')
    parser.add_argument('--output', help='Output video path (default: <run>/visualization.mp4)')
    parser.add_argument('--draw_every', type=int, default=20,
                         help='Render every Nth sample, to speed up rendering (default 20).')
    parser.add_argument('--batch', action='store_true',
                        help='Treat run as a parent path; scan subdirs and visualize all runs '
                             'that have training_trajectory.npz but not yet visualization.mp4/.gif.')
    parser.add_argument('--ignore-existing', action='store_true',
                        help='Recreate visualization even if it already exists.')
    args = parser.parse_args()
    if args.draw_every < 1:
        raise SystemExit("--draw_every must be >= 1")

    if args.batch:
        process_batch(args.run, args.draw_every, args.ignore_existing)
    else:
        process_single(args.run, args.draw_every, args.output)


def process_single(run_path, draw_every, output_path=None):
    npz_path = find_npz(run_path)
    run_dir = os.path.dirname(npz_path)
    data = np.load(npz_path)
    rate_hz = float(data['rate_hz'])
    fps = rate_hz / draw_every

    n_samples = data['positions'].shape[0]
    n_rendered = len(range(0, n_samples, draw_every))
    print(f"Loaded {npz_path}: {n_samples} samples "
          f"{'with' if 'depth' in data.files else 'without'} depth; "
          f"rendering {n_rendered} of them (every {draw_every}) at {fps:.2f} fps.")

    fig, anim = build_animation(data, draw_every)

    use_ffmpeg = shutil.which('ffmpeg') is not None
    if output_path:
        out_path = output_path
    else:
        ext = 'mp4' if use_ffmpeg else 'gif'
        out_path = os.path.join(run_dir, f'{OUTPUT_STEM}.{ext}')

    if use_ffmpeg:
        anim.save(out_path, writer='ffmpeg', fps=fps, dpi=120)
    else:
        print("ffmpeg not found on PATH — falling back to an animated GIF.")
        anim.save(out_path, writer=PillowWriter(fps=fps), dpi=120)

    plt.close(fig)
    print(f"Saved {out_path}")


def process_batch(parent_path, draw_every, ignore_existing=False):
    """Scan parent_path for runs with training_trajectory.npz and visualize those without visualization.mp4/.gif."""
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
        npz_path = os.path.join(subdir, NPZ_NAME)
        if not os.path.isfile(npz_path):
            print(f"Skip {subdir} (no {NPZ_NAME})")
            continue

        # Check for either .mp4 or .gif visualization
        mp4_path = os.path.join(subdir, f'{OUTPUT_STEM}.mp4')
        gif_path = os.path.join(subdir, f'{OUTPUT_STEM}.gif')
        if (os.path.exists(mp4_path) or os.path.exists(gif_path)) and not ignore_existing:
            print(f"Skip {subdir} (already visualized)")
        else:
            to_process.append(subdir)

    if not to_process:
        print(f"All subdirectories either have no {NPZ_NAME} or are already visualized.")
        return

    print(f"Found {len(to_process)} runs to visualize.\n")

    for i, run_path in enumerate(to_process, 1):
        print(f"\n[{i}/{len(to_process)}] Visualizing {os.path.basename(run_path)}...")
        try:
            process_single(run_path, draw_every)
        except Exception as e:
            print(f"ERROR visualizing {run_path}: {e}")


if __name__ == '__main__':
    main()
