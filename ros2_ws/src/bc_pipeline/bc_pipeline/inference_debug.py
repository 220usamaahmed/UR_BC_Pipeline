"""Pure helpers for recording and exporting inference debug traces."""

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Optional

import numpy as np


TRACE_SCHEMA_VERSION = 1
EVENT_TOPIC = '/bc_pipeline/inference/events'
PHASE_TOPIC = '/bc_pipeline/inference/phase'
CANONICAL_JOINT_TOPIC = '/bc_pipeline/inference/joint_states_canonical'
PLANNED_TARGET_TOPIC = '/bc_pipeline/inference/action_targets_planned'
MATCHED_TARGET_TOPIC = '/bc_pipeline/inference/action_targets_matched'
SELECTED_JOINT_TOPIC = '/bc_pipeline/inference/selected_joint_observations'
SELECTED_DEPTH_TOPIC = '/bc_pipeline/inference/selected_depth'
PREPROCESS_VERSION = 'drawer-depth-v1'


@dataclass
class ExecutionCandidate:
    """One depth frame with a joint position aligned to its source stamp."""

    stamp_ns: int
    positions: np.ndarray
    depth: np.ndarray
    depth_frame_id: str
    joint_before_stamp_ns: int
    joint_after_stamp_ns: int
    interpolation_alpha: Optional[float]


@dataclass
class ModelObservation:
    """One synchronized observation retained in the model history."""

    observation_id: str
    stamp_ns: int
    positions: np.ndarray
    depth: np.ndarray
    gripper_state: float
    depth_frame_id: str
    joint_before_stamp_ns: int
    joint_after_stamp_ns: int
    interpolation_alpha: Optional[float]


def depth_sha256(depth: np.ndarray) -> str:
    """Hash the exact little-endian float32 pixels sent to the model."""
    canonical = np.ascontiguousarray(depth, dtype='<f4')
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype='<u8').tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def selected_observation_id(chunk_index: int, action_index: int) -> str:
    """Return the stable human-readable ID for one selected action result."""
    if chunk_index <= 0 or action_index <= 0:
        raise ValueError('chunk and action indices must be positive')
    return f'chunk-{chunk_index:06d}-action-{action_index:02d}'


def integrate_joint_deltas(seed: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Convert a chunk of joint deltas into absolute controller targets."""
    seed = np.asarray(seed, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.ndim != 2 or seed.shape != (deltas.shape[1],):
        raise ValueError('seed and joint deltas have incompatible shapes')
    return seed + np.cumsum(deltas, axis=0)


def interpolate_joint_positions(
    before_stamp_ns: int,
    before_positions: np.ndarray,
    after_stamp_ns: int,
    after_positions: np.ndarray,
    target_stamp_ns: int,
) -> tuple[np.ndarray, Optional[float]]:
    """Interpolate joint positions at one timestamp without float timestamps."""
    if after_stamp_ns < before_stamp_ns:
        raise ValueError('joint interpolation stamps are out of order')
    if not before_stamp_ns <= target_stamp_ns <= after_stamp_ns:
        raise ValueError('target stamp is outside the interpolation interval')
    before = np.asarray(before_positions, dtype=np.float64)
    after = np.asarray(after_positions, dtype=np.float64)
    if before.shape != after.shape:
        raise ValueError('joint interpolation positions have different shapes')
    span_ns = after_stamp_ns - before_stamp_ns
    if span_ns == 0:
        return before.copy(), None
    alpha = (target_stamp_ns - before_stamp_ns) / span_ns
    return before + alpha * (after - before), float(alpha)


def event_message(
    run_id: str,
    event_type: str,
    event_time_ns: int,
    *,
    chunk_index: Optional[int] = None,
    **payload: Any,
) -> dict:
    """Build one versioned JSON-safe event envelope."""
    event = {
        'schema_version': TRACE_SCHEMA_VERSION,
        'run_id': str(run_id),
        'event': str(event_type),
        'event_time_ns': int(event_time_ns),
    }
    if chunk_index is not None:
        event['chunk_index'] = int(chunk_index)
    event.update(payload)
    return event


def dumps_event(event: dict) -> str:
    """Serialize an event deterministically and reject non-finite numbers."""
    return json.dumps(
        event,
        allow_nan=False,
        separators=(',', ':'),
        sort_keys=True,
    )


def validate_event(event: Any) -> dict:
    """Validate the stable envelope fields used by the exporter."""
    if not isinstance(event, dict):
        raise ValueError('trace event must be a JSON object')
    if event.get('schema_version') != TRACE_SCHEMA_VERSION:
        raise ValueError(
            'unsupported trace schema_version '
            f"{event.get('schema_version')!r}; expected {TRACE_SCHEMA_VERSION}"
        )
    for field in ('run_id', 'event'):
        if not isinstance(event.get(field), str) or not event[field]:
            raise ValueError(f'trace event requires a non-empty {field!r}')
    event_time_ns = event.get('event_time_ns')
    if not isinstance(event_time_ns, int) or event_time_ns < 0:
        raise ValueError('trace event requires a non-negative event_time_ns')
    chunk_index = event.get('chunk_index')
    if chunk_index is not None and (
        not isinstance(chunk_index, int) or chunk_index <= 0
    ):
        raise ValueError('chunk_index must be a positive integer')
    return event


def assemble_trace(records: Iterable[tuple[int, dict]]) -> dict:
    """Group validated bag events by run and chunk without losing failures."""
    ordered = []
    for bag_receive_time_ns, raw_event in records:
        event = dict(validate_event(raw_event))
        event['bag_receive_time_ns'] = int(bag_receive_time_ns)
        ordered.append(event)
    ordered.sort(key=lambda event: (
        event['bag_receive_time_ns'], event['event_time_ns']
    ))
    if not ordered:
        raise ValueError('bag contains no inference trace events')

    run_ids = sorted({event['run_id'] for event in ordered})
    if len(run_ids) != 1:
        raise ValueError(
            'expected one run_id in an inference bag; found '
            + ', '.join(run_ids)
        )

    run_events = []
    chunks = {}
    for event in ordered:
        chunk_index = event.get('chunk_index')
        if chunk_index is None:
            run_events.append(event)
        else:
            chunks.setdefault(chunk_index, []).append(event)

    run_end = next(
        (event for event in reversed(run_events)
         if event['event'] == 'run_end'),
        None,
    )
    return {
        'schema_version': TRACE_SCHEMA_VERSION,
        'run_id': run_ids[0],
        'complete': run_end is not None,
        'status': run_end.get('status') if run_end else 'incomplete',
        'run_events': run_events,
        'chunks': [
            {'chunk_index': index, 'events': chunks[index]}
            for index in sorted(chunks)
        ],
    }
