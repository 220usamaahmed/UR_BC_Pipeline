#!/usr/bin/env python3
"""
visualize_run.py — animate a run's depth frames alongside its joint traces.

Reads the training_trajectory.npz written by make_training_trajectory.py and
renders a synced side-by-side animation: the depth frame on the left, each
joint's position trace on the right with a moving cursor marking "now". One
video frame per saved sample, played back at rate_hz so the video runs at the
same speed as the recording.

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


def build_animation(data: dict):
    timestamps = data['timestamps']
    positions = data['positions']
    joint_names = [str(n) for n in data['joint_names']]
    depth = data.get('depth')
    n_frames = len(timestamps)

    has_depth = depth is not None
    fig, axes = plt.subplots(
        1, 2 if has_depth else 1, figsize=(12 if has_depth else 6, 5))
    ax_joints = axes[1] if has_depth else axes

    im = None
    if has_depth:
        ax_depth = axes[0]
        depth_clean = np.where(np.isfinite(depth), depth, np.nan)
        vmin, vmax = np.nanmin(depth_clean), np.nanmax(depth_clean)
        cmap = matplotlib.colormaps['viridis'].copy()
        cmap.set_bad('black')
        im = ax_depth.imshow(depth_clean[0], cmap=cmap, vmin=vmin, vmax=vmax)
        ax_depth.set_title('Depth (m)')
        ax_depth.set_xticks([])
        ax_depth.set_yticks([])
        fig.colorbar(im, ax=ax_depth, fraction=0.046, pad=0.04)

    for j, name in enumerate(joint_names):
        ax_joints.plot(timestamps, positions[:, j], label=name, linewidth=1)
    cursor = ax_joints.axvline(timestamps[0], color='black', linewidth=1.5)
    ax_joints.set_xlabel('time (s)')
    ax_joints.set_ylabel('position (rad)')
    ax_joints.set_title('Joint positions')
    ax_joints.legend(loc='upper right', fontsize='small')
    fig.tight_layout()

    def update(i):
        artists = [cursor]
        cursor.set_xdata([timestamps[i], timestamps[i]])
        if has_depth:
            im.set_data(depth_clean[i])
            artists.append(im)
        return artists

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    return fig, anim


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', help='Run directory, or a training_trajectory.npz path.')
    parser.add_argument('--output', help='Output video path (default: <run>/visualization.mp4)')
    args = parser.parse_args()

    npz_path = find_npz(args.run)
    run_dir = os.path.dirname(npz_path)
    data = np.load(npz_path)
    rate_hz = float(data['rate_hz'])

    print(f"Loaded {npz_path}: {data['positions'].shape[0]} samples "
          f"{'with' if 'depth' in data.files else 'without'} depth.")

    fig, anim = build_animation(data)

    use_ffmpeg = shutil.which('ffmpeg') is not None
    if args.output:
        out_path = args.output
    else:
        ext = 'mp4' if use_ffmpeg else 'gif'
        out_path = os.path.join(run_dir, f'{OUTPUT_STEM}.{ext}')

    if use_ffmpeg:
        anim.save(out_path, writer='ffmpeg', fps=rate_hz, dpi=120)
    else:
        print("ffmpeg not found on PATH — falling back to an animated GIF.")
        anim.save(out_path, writer=PillowWriter(fps=rate_hz), dpi=120)

    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
