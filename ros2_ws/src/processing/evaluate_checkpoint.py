#!/usr/bin/env python3
"""Evaluate a flow-matching checkpoint on a saved training trajectory.

For each chunk, five observations condition the policy and the following
twenty recorded deltas are compared with the twenty predicted actions.  The
observation window then advances by twenty samples.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter


OBSERVATION_LENGTH = 5
ACTION_HORIZON = 20
ACTION_DIM = 7


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="PyTorch checkpoint (.pt)")
    parser.add_argument(
        "trajectory",
        type=Path,
        help="training_trajectory.npz, or the directory containing it",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("checkpoint_evaluation.gif"),
        help="output animation (default: checkpoint_evaluation.gif)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device, for example cpu or cuda (default: auto)",
    )
    parser.add_argument(
        "--flow-steps",
        type=int,
        default=100,
        help="number of Heun integration steps (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for the initial flow noise (default: 0)",
    )
    parser.add_argument(
        "--model-output-scale",
        type=float,
        default=2.0,
        help="scale applied to raw model actions (default: 2.0)",
    )
    parser.add_argument(
        "--joint-delta-divisor",
        type=float,
        default=150.0,
        help="divisor converting model joint outputs to radians (default: 150)",
    )
    return parser.parse_args()


def resolve_trajectory(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "training_trajectory.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory not found: {path}")
    return path


def import_model():
    """Import model.py from the sibling bc_pipeline ROS package."""
    package_root = Path(__file__).resolve().parents[1] / "bc_pipeline"
    sys.path.insert(0, str(package_root))
    from bc_pipeline.model import ConditionalDiffusionModel

    return ConditionalDiffusionModel


def preprocess_depth(depth_batch: np.ndarray) -> np.ndarray:
    """Apply the same crop and manual depth processing as online inference."""
    if depth_batch.ndim != 3:
        raise ValueError(f"Expected depth shape (N,H,W), got {depth_batch.shape}")
    if depth_batch.shape[1] < 170 or depth_batch.shape[2] < 440:
        raise ValueError(f"Depth frames are too small: {depth_batch.shape[1:]}")

    depth = depth_batch[:, 60:170, 230:440].astype(np.float32, copy=True)
    depth = np.nan_to_num(depth, nan=10.0)
    depth = np.clip(depth, 0.0, 0.8)
    zero_mask = depth == 0
    depth = np.ceil((depth - 1e-6) / 0.05) * 0.05
    depth[zero_mask] = 0.0

    # (row start, row end, column start, column end)
    regions = (
        (41, 110, 0, 65),
        (41, 110, 140, 210),
        (0, 41, 0, 65),
        (0, 41, 140, 210),
        (33, 60, 98, 125),
    )
    for row0, row1, col0, col1 in regions:
        region = depth[:, row0:row1, col0:col1]
        fill = np.percentile(region, 10, axis=(1, 2))

        print(f"Filling region {row0}:{row1},{col0}:{col1} with {fill}")

        depth[:, row0:row1, col0:col1] = fill[:, None, None]
    return depth


def load_model(checkpoint_path: Path, device: torch.device):
    model_class = import_model()
    checkpoint = torch.load(
        checkpoint_path.expanduser().resolve(), map_location=device
    )
    state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    # Also accept checkpoints saved directly from DistributedDataParallel.
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value
                      for key, value in state_dict.items()}
    model = model_class(
        context_length=OBSERVATION_LENGTH,
        action_horizon=ACTION_HORIZON,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval().requires_grad_(False)
    return model


@torch.inference_mode()
def predict_actions(model, depth, observations, device, flow_steps, generator):
    depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(2).to(device)
    observation_tensor = torch.from_numpy(observations).unsqueeze(0).to(device)
    actions = torch.randn(
        (1, ACTION_HORIZON, ACTION_DIM),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    step_size = 1.0 / flow_steps
    for step in range(flow_steps):
        time = torch.full((1,), step * step_size, device=device)
        velocity = model(
            depth_tensor, observation_tensor, actions, time
        )
        predicted = actions + step_size * velocity
        next_time = torch.full((1,), (step + 1) * step_size, device=device)
        next_velocity = model(
            depth_tensor, observation_tensor, predicted, next_time
        )
        actions += 0.5 * step_size * (velocity + next_velocity)
    return actions[0].cpu().numpy()


def evaluate(args, model, data, device):
    positions = np.asarray(data["positions"], dtype=np.float32)
    deltas = np.asarray(data["deltas"], dtype=np.float32)
    depth_frames = data["depth"]
    gripping = (
        np.asarray(data["is_gripping"], dtype=np.float32)
        if "is_gripping" in data.files
        else np.zeros(len(positions), dtype=np.float32)
    )
    if positions.shape[1] != 6 or deltas.shape[1] != 6:
        raise ValueError("The model expects exactly six robot joints")
    if len(depth_frames) != len(positions) or len(gripping) != len(positions):
        raise ValueError("positions, depth, and is_gripping lengths must match")

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    indices, actual_chunks, predicted_chunks = [], [], []
    current_observation_indices, current_depth_frames = [], []
    max_start = len(positions) - OBSERVATION_LENGTH

    for start in range(0, max_start + 1, ACTION_HORIZON):
        target_start = start + OBSERVATION_LENGTH - 1
        target_end = target_start + ACTION_HORIZON
        if target_end > len(deltas):
            break

        observations = np.column_stack(
            (positions[start:start + OBSERVATION_LENGTH],
             gripping[start:start + OBSERVATION_LENGTH])
        ).astype(np.float32)
        depth = preprocess_depth(
            depth_frames[start:start + OBSERVATION_LENGTH]
        )
        raw_prediction = predict_actions(
            model, depth, observations, device, args.flow_steps, generator
        )
        predicted = (
            raw_prediction[:, :6]
            * args.model_output_scale
            / args.joint_delta_divisor
        )
        indices.append(np.arange(target_start, target_end))
        actual_chunks.append(deltas[target_start:target_end])
        predicted_chunks.append(predicted)
        current_observation_indices.append(start + OBSERVATION_LENGTH - 1)
        current_depth_frames.append(depth[-1])
        print(
            f"Evaluated observations {start}:{start + OBSERVATION_LENGTH} "
            f"against deltas {target_start}:{target_end}"
        )

    if not indices:
        raise ValueError("Trajectory is too short for one 5-to-20 evaluation")
    return (
        np.concatenate(indices),
        np.concatenate(actual_chunks),
        np.concatenate(predicted_chunks),
        np.asarray(current_observation_indices),
        np.stack(current_depth_frames),
    )


def animate_comparison(
    indices,
    actual,
    predicted,
    positions,
    timestamps,
    current_observation_indices,
    current_depth_frames,
    joint_names,
    rate_hz,
    output,
):
    fig = plt.figure(figsize=(15, 15))
    grid = fig.add_gridspec(4, 2, height_ratios=(1.25, 1, 1, 1))
    depth_ax = fig.add_subplot(grid[0, 0])
    observation_ax = fig.add_subplot(grid[0, 1])
    delta_axes = [
        fig.add_subplot(grid[row, col])
        for row in range(1, 4)
        for col in range(2)
    ]

    depth_artist = depth_ax.imshow(
        current_depth_frames[0],
        cmap="viridis",
        vmin=0.0,
        vmax=0.8,
        animated=True,
    )
    depth_ax.set_title(
        f"Current depth (observation {current_observation_indices[0]})"
    )
    depth_ax.set_xlabel("image column")
    depth_ax.set_ylabel("image row")
    fig.colorbar(depth_artist, ax=depth_ax, label="depth (m)", shrink=0.85)

    observation_x = (
        timestamps if len(timestamps) == len(positions)
        else np.arange(len(positions))
    )
    observation_x_label = (
        "time (s)" if len(timestamps) == len(positions)
        else "trajectory observation index"
    )
    for joint in range(positions.shape[1]):
        name = (
            joint_names[joint]
            if joint < len(joint_names)
            else f"joint_{joint}"
        )
        observation_ax.plot(
            observation_x, positions[:, joint], linewidth=1, label=str(name)
        )
    observation_cursor = observation_ax.axvline(
        observation_x[current_observation_indices[0]],
        color="black",
        linewidth=2,
    )
    observation_ax.set_title("Joint observations")
    observation_ax.set_xlabel(observation_x_label)
    observation_ax.set_ylabel("position (rad)")
    observation_ax.grid(alpha=0.25)
    observation_ax.legend(fontsize=7, ncol=2)

    delta_cursors = []
    for joint, ax in enumerate(delta_axes):
        name = joint_names[joint] if joint < len(joint_names) else f"joint_{joint}"
        ax.plot(indices, actual[:, joint], label="recorded", linewidth=1.4)
        ax.plot(indices, predicted[:, joint], label="model", linewidth=1.1)
        delta_cursors.append(
            ax.axvline(indices[0], color="black", linewidth=1.5)
        )
        ax.set_title(str(name))
        ax.set_ylabel("delta (rad)")
        ax.grid(alpha=0.25)
    delta_axes[0].legend()
    delta_axes[-2].set_xlabel("trajectory delta index")
    delta_axes[-1].set_xlabel("trajectory delta index")
    title = fig.suptitle("Recorded trajectory deltas vs model predictions")
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    def update(frame):
        chunk = frame // ACTION_HORIZON
        observation_index = current_observation_indices[chunk]
        depth_artist.set_data(current_depth_frames[chunk])
        depth_ax.set_title(
            f"Current depth (observation {observation_index})"
        )
        observation_cursor.set_xdata(
            [observation_x[observation_index]] * 2
        )
        for cursor in delta_cursors:
            cursor.set_xdata([indices[frame]] * 2)
        title.set_text(
            "Recorded trajectory deltas vs model predictions "
            f"(delta {indices[frame]})"
        )
        return [
            depth_artist,
            observation_cursor,
            *delta_cursors,
            title,
        ]

    interval_ms = 1000.0 / rate_hz if rate_hz > 0 else 50.0
    animation = FuncAnimation(
        fig,
        update,
        frames=len(indices),
        interval=interval_ms,
        blit=False,
    )
    output = output.expanduser().resolve()
    if output.suffix.lower() != ".gif":
        output = output.with_suffix(".gif")
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(
        output,
        writer=PillowWriter(fps=rate_hz if rate_hz > 0 else 20.0),
        dpi=100,
    )
    plt.close(fig)
    return output


def main():
    args = parse_args()
    if args.flow_steps <= 0:
        raise ValueError("--flow-steps must be positive")
    if args.joint_delta_divisor == 0:
        raise ValueError("--joint-delta-divisor must be non-zero")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    trajectory_path = resolve_trajectory(args.trajectory)
    model = load_model(args.checkpoint, device)
    with np.load(trajectory_path, mmap_mode="r") as data:
        (
            indices,
            actual,
            predicted,
            current_observation_indices,
            current_depth_frames,
        ) = evaluate(args, model, data, device)
        joint_names = data["joint_names"].tolist()
        positions = np.asarray(data["positions"])
        timestamps = np.asarray(data["timestamps"])
        rate_hz = float(data["rate_hz"])
    output = animate_comparison(
        indices,
        actual,
        predicted,
        positions,
        timestamps,
        current_observation_indices,
        current_depth_frames,
        joint_names,
        rate_hz,
        args.output,
    )
    rmse = np.sqrt(np.mean((predicted - actual) ** 2, axis=0))
    print(f"Saved comparison animation to {output}")
    print("Per-joint RMSE (rad):", np.array2string(rmse, precision=6))


if __name__ == "__main__":
    main()
