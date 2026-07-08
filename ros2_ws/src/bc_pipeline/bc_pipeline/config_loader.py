#!/usr/bin/env python3
"""
Config loader — reads and validates the single experiment config.

One config file describes a whole experiment:

  robot       — which arm/group/links to plan for
  planning    — default planning effort + speed
  checkpoints — named joint positions reused by Checkpoint steps
  steps       — the ordered sequence to execute
  obstacles   — collision geometry (planning scene + Foxglove markers)
  recording   — which topics to record and where to write the bag

Every node (sequence_runner, scene_publisher) and the launch file load the
SAME file via a 'config' ROS parameter, so they can never drift apart.

SOURCE FORMATS
---------------
A config is authored as either:

  .yaml  — plain data, parsed with yaml.safe_load (as always).
  .py    — a Python module defining a module-level `CONFIG` dict.  It is
           executed, so it can use math, random, loops, etc. to build the
           dict (e.g. randomised offsets for data augmentation).  It must
           still describe the exact same schema a YAML file would.

The launch file resolves whichever format is authored into a concrete,
already-executed dict exactly ONCE, dumps it to a plain YAML file, and points
every node at that resolved file — so two nodes independently loading a
config with randomness in it can never see different values, and the
resolved values get saved alongside the run for reproducibility. See
resolve_source() below and record_sequence.launch.py.

VALIDATION PHILOSOPHY
---------------------
Validate structure here, eagerly, before anything moves.  A typo in a joint
name or a missing section should raise immediately — never halfway through a
motion with the recorder already running.  Step-specific argument validation
lives in each Step subclass (also at construction time, before execution).
"""

import importlib.util
import math
import os

import yaml


class ConfigError(Exception):
    """Raised when the experiment config is missing or malformed."""


def resolve_source(path: str) -> dict:
    """Parse a config file into a plain dict — no validation, no unit
    conversion.  Handles both source formats (see module docstring).

    This is the only place that needs to know a config might be Python
    instead of YAML; load_config() and everything downstream just sees a
    dict, exactly as before.
    """
    if not path:
        raise ConfigError(
            "No config path given. Set the 'config' ROS parameter to a config file."
        )
    if not os.path.isfile(path):
        raise ConfigError(f"Config file not found: {path}")

    if path.endswith('.py'):
        config = _exec_python_config(path)
    else:
        with open(path) as f:
            config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ConfigError(f"Config root must be a mapping; got {type(config).__name__}.")
    return config


def _exec_python_config(path: str) -> dict:
    spec = importlib.util.spec_from_file_location('bc_pipeline_experiment_config', path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"Could not load Python config '{path}'.")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ConfigError(f"Error executing Python config '{path}': {exc}") from exc

    if not hasattr(module, 'CONFIG'):
        raise ConfigError(
            f"Python config '{path}' must define a module-level 'CONFIG' dict."
        )
    return module.CONFIG


def load_config(path: str) -> dict:
    """Read, parse, and structurally validate the experiment config."""
    config = resolve_source(path)

    _validate_robot(config)
    _validate_planning(config)
    _validate_checkpoints(config)
    _validate_steps(config)
    _validate_obstacles(config)
    _validate_recording(config)

    return config


# ── section validators ───────────────────────────────────────────────────────


def _require_section(config: dict, key: str, kind: type):
    if key not in config:
        raise ConfigError(f"Config is missing the required '{key}' section.")
    value = config[key]
    if not isinstance(value, kind):
        raise ConfigError(
            f"'{key}' must be a {kind.__name__}; got {type(value).__name__}."
        )
    return value


def _validate_robot(config: dict):
    robot = _require_section(config, 'robot', dict)
    for key in ('planning_group', 'eef_link', 'base_frame', 'joint_names'):
        if key not in robot:
            raise ConfigError(f"robot section is missing '{key}'.")
    if not isinstance(robot['joint_names'], list) or not robot['joint_names']:
        raise ConfigError("robot.joint_names must be a non-empty list.")


def _validate_planning(config: dict):
    planning = _require_section(config, 'planning', dict)
    for key in ('velocity_scaling', 'accel_scaling', 'planning_time'):
        if key not in planning:
            raise ConfigError(f"planning section is missing '{key}'.")


def _validate_checkpoints(config: dict):
    # Checkpoint joint values are authored in DEGREES (more human-readable than
    # radians).  We validate and convert them to radians in place here, so every
    # downstream consumer (Context, Checkpoint steps, MoveIt) keeps working in
    # the radians the controller expects without having to know about the unit.
    checkpoints = _require_section(config, 'checkpoints', dict)
    n = len(config['robot']['joint_names'])
    for name, angles in checkpoints.items():
        if not isinstance(angles, list) or len(angles) != n:
            raise ConfigError(
                f"checkpoint '{name}' must be a list of {n} joint values "
                f"(one per robot.joint_names); got {angles!r}."
            )
        if not all(isinstance(a, (int, float)) and not isinstance(a, bool)
                   for a in angles):
            raise ConfigError(
                f"checkpoint '{name}' joint values must all be numbers "
                f"(degrees); got {angles!r}."
            )
        checkpoints[name] = [math.radians(a) for a in angles]


def _validate_steps(config: dict):
    steps = _require_section(config, 'steps', list)
    if not steps:
        raise ConfigError("steps section is empty — nothing to execute.")
    for i, entry in enumerate(steps):
        if not isinstance(entry, dict) or 'type' not in entry:
            raise ConfigError(
                f"step {i} must be a mapping with a 'type' key; got {entry!r}."
            )


def _validate_obstacles(config: dict):
    # Obstacles are optional — an empty scene is valid.
    obstacles = config.get('obstacles', [])
    if not isinstance(obstacles, list):
        raise ConfigError("obstacles must be a list.")
    config['obstacles'] = obstacles
    for i, spec in enumerate(obstacles):
        if not isinstance(spec, dict):
            raise ConfigError(f"obstacle {i} must be a mapping; got {spec!r}.")
        for key in ('id', 'size', 'position'):
            if key not in spec:
                raise ConfigError(f"obstacle {i} is missing '{key}'.")
        if len(spec['size']) != 3 or len(spec['position']) != 3:
            raise ConfigError(
                f"obstacle '{spec.get('id', i)}' size and position must each "
                f"have 3 values."
            )


def _validate_recording(config: dict):
    # Recording is optional — omit the 'recording' section to run a sequence
    # without capturing a bag (the launch file skips the recorder and warns).
    if 'recording' not in config:
        return
    recording = config['recording']
    if not isinstance(recording, dict):
        raise ConfigError(
            f"'recording' must be a dict; got {type(recording).__name__}."
        )
    if 'bag_uri' not in recording:
        raise ConfigError("recording section is missing 'bag_uri'.")
    if not isinstance(recording.get('topics'), list) or not recording['topics']:
        raise ConfigError("recording.topics must be a non-empty list.")
