"""Tests for ROS-independent training-trajectory processing helpers."""

import numpy as np

from make_training_trajectory import decode_step_messages, filter_ignored_samples


def test_decode_step_messages_extracts_labels_and_ignore_flags():
    messages = np.asarray([
        '{"label": "Wait 1.0 s", "ignore": true}',
        '{"label": "Checkpoint \'place\'", "ignore": false}',
        'legacy step label',
    ])

    labels, ignored = decode_step_messages(messages)

    assert labels.tolist() == ['Wait 1.0 s', "Checkpoint 'place'", 'legacy step label']
    assert ignored.tolist() == [True, False, False]


def test_filter_ignored_samples_removes_every_aligned_array_and_recomputes_deltas():
    data = {
        'joint_names': np.asarray(['joint_a', 'joint_b']),
        'timestamps': np.asarray([0.0, 0.1, 0.2, 0.3]),
        'positions': np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        'deltas': np.asarray([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        'depth': np.asarray([[[0]], [[1]], [[2]], [[3]]]),
        'steps': np.asarray(['ignored_a', 'kept_a', 'ignored_b', 'kept_b']),
        'is_gripping': np.asarray([False, True, True, False]),
        'rate_hz': np.asarray(20.0),
    }

    result = filter_ignored_samples(data, np.asarray([True, False, True, False]))

    np.testing.assert_allclose(result['timestamps'], [0.1, 0.3])
    np.testing.assert_allclose(result['positions'], [[1.0, 1.0], [3.0, 3.0]])
    np.testing.assert_allclose(result['deltas'], [[2.0, 2.0]])
    assert result['steps'].tolist() == ['kept_a', 'kept_b']
    assert result['depth'].reshape(-1).tolist() == [1, 3]
    assert result['is_gripping'].tolist() == [True, False]
    assert result['joint_names'].tolist() == ['joint_a', 'joint_b']
