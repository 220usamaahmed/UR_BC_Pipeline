"""Unit tests for ROS-independent inference trace helpers."""

import json

import numpy as np
import pytest

from bc_pipeline.inference_debug import (
    TRACE_SCHEMA_VERSION,
    assemble_trace,
    depth_sha256,
    dumps_event,
    event_message,
    integrate_joint_deltas,
    interpolate_joint_positions,
    selected_observation_id,
)


def test_integrate_joint_deltas_builds_absolute_targets():
    seed = np.array([1.0, -1.0])
    deltas = np.array([[0.1, 0.2], [0.3, -0.4]])
    targets = integrate_joint_deltas(seed, deltas)
    np.testing.assert_allclose(targets, [[1.1, -0.8], [1.4, -1.2]])


def test_interpolation_preserves_integer_timestamp_provenance():
    positions, alpha = interpolate_joint_positions(
        1_000_000_000,
        np.array([0.0, 2.0]),
        1_000_000_010,
        np.array([1.0, 4.0]),
        1_000_000_004,
    )
    np.testing.assert_allclose(positions, [0.4, 2.8])
    assert alpha == pytest.approx(0.4)


def test_depth_hash_includes_shape_and_exact_float32_pixels():
    pixels = np.array([[0.0, 0.05], [0.1, 0.8]], dtype=np.float32)
    assert depth_sha256(pixels) == depth_sha256(pixels.copy())
    assert depth_sha256(pixels) != depth_sha256(pixels.reshape(1, 4))


def test_selected_observation_id_is_stable_and_one_based():
    assert selected_observation_id(12, 6) == 'chunk-000012-action-06'
    with pytest.raises(ValueError):
        selected_observation_id(0, 6)


def test_event_serialization_rejects_non_finite_numbers():
    event = event_message('run', 'run_start', 123, value=float('nan'))
    with pytest.raises(ValueError):
        dumps_event(event)


def test_assemble_trace_preserves_failed_and_incomplete_chunks():
    events = [
        (10, event_message('run', 'run_start', 9)),
        (20, event_message(
            'run', 'inference_complete', 19, chunk_index=1
        )),
        (30, event_message(
            'run', 'selection_failed', 29, chunk_index=1,
            reason='joint_tolerance',
        )),
    ]
    trace = assemble_trace(events)
    assert trace['schema_version'] == TRACE_SCHEMA_VERSION
    assert trace['complete'] is False
    assert trace['status'] == 'incomplete'
    assert trace['chunks'][0]['chunk_index'] == 1
    assert [
        event['event'] for event in trace['chunks'][0]['events']
    ] == ['inference_complete', 'selection_failed']


def test_complete_trace_uses_run_end_status():
    events = [
        (1, event_message('run', 'run_start', 1)),
        (2, event_message('run', 'run_end', 2, status='failed')),
    ]
    trace = assemble_trace(events)
    assert trace['complete'] is True
    assert trace['status'] == 'failed'
    json.loads(json.dumps(trace, allow_nan=False))
