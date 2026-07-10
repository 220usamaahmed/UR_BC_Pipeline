#!/usr/bin/env python3
"""
visualize_run.py — animate a run's depth frames alongside its joint traces.

Reads the training_trajectory.npz written by make_training_trajectory.py and
renders a synced animation: the depth frame on the left, joint positions top
right and joint deltas bottom right, each with a moving cursor marking "now".
--draw_every subsamples which saved samples get rendered (default 20, i.e.
every 20th) to speed up rendering; playback fps is scaled down to match, so
the video's duration still matches the real recording — it's just choppier.

Only needs numpy + matplotlib (no ROS), so this runs fine on the host inside
the repo's .venv:

USAGE
-----
    .venv/bin/python3 ros2_ws/src/processing/visualize_run.py \\
        ros2_ws/runs/drawer_2026-06-20_14-14-03

Saves <run>/visualization.mp4 next to the .npz (falls back to an animated
.gif if the `ffmpeg` binary isn't on PATH).
"""

import argparse
import os
import shutil

import matplotlib
matplotlib.use('Agg')  # no display available when run headless / over SSH

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

NPZ_NAME = 'training_trajectory.npz'
OUTPUT_STEM = 'visualization'


def find_npz(path: str) -> str:
    """Return the path to training_trajectory.npz given a run dir or the file itself."""
    if os.path.isfile(path):
        return path
    candidate = os.path.join(path, NPZ_NAME)
    if os.path.isfile(candidate):
        return candidate
    raise SystemExit(f"No {NPZ_NAME} found at or in {path} — run "
                      f"make_training_trajectory.py on this run first.")


def build_animation(data: dict, draw_every: int):
    timestamps = data['timestamps']
    positions = data['positions']
    deltas = data['deltas']
    delta_times = timestamps[:-1]  # deltas[i] = positions[i+1] - positions[i]
    joint_names = [str(n) for n in data['joint_names']]
    depth = data.get('depth')
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

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for j, name in enumerate(joint_names):
        color = colors[j % len(colors)]
        ax_joints.plot(timestamps, positions[:, j], label=name, linewidth=1, color=color)
        ax_deltas.plot(delta_times, deltas[:, j], linewidth=1, color=color)

    joint_cursor = ax_joints.axvline(timestamps[0], color='black', linewidth=1.5)
    delta_cursor = ax_deltas.axvline(delta_times[0], color='black', linewidth=1.5)

    ax_joints.set_ylabel('position (rad)')
    ax_joints.set_title('Joint positions')
    ax_joints.legend(loc='upper right', fontsize='small')
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
    args = parser.parse_args()
    if args.draw_every < 1:
        raise SystemExit("--draw_every must be >= 1")

    npz_path = find_npz(args.run)
    run_dir = os.path.dirname(npz_path)
    data = np.load(npz_path)
    rate_hz = float(data['rate_hz'])
    fps = rate_hz / args.draw_every

    n_samples = data['positions'].shape[0]
    n_rendered = len(range(0, n_samples, args.draw_every))
    print(f"Loaded {npz_path}: {n_samples} samples "
          f"{'with' if 'depth' in data.files else 'without'} depth; "
          f"rendering {n_rendered} of them (every {args.draw_every}) at {fps:.2f} fps.")

    fig, anim = build_animation(data, args.draw_every)

    use_ffmpeg = shutil.which('ffmpeg') is not None
    if args.output:
        out_path = args.output
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


if __name__ == '__main__':
    main()
