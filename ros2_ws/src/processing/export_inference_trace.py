#!/usr/bin/env python3
"""Export versioned inference debug events from a rosbag2 run to JSON."""

import argparse
import json
import os
import tempfile

import numpy as np

from bc_pipeline.inference_debug import (
    EVENT_TOPIC,
    SELECTED_DEPTH_TOPIC,
    assemble_trace,
    depth_sha256,
)

from inspect_run import find_bag_dir, load_metadata


OUTPUT_NAME = 'inference_trace.json'
OFFSET_TOPICS = {
    '/joint_states',
    '/zed/zed_node/depth/depth_registered',
    '/bc_pipeline/inference/joint_states_canonical',
    '/bc_pipeline/inference/selected_joint_observations',
    '/bc_pipeline/inference/selected_depth',
}


def _open_reader(bag_dir: str, storage_id: str, topics: list[str]):
    import rosbag2_py

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage_id),
        rosbag2_py.ConverterOptions('cdr', 'cdr'),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    return reader


def read_events(bag_dir: str, storage_id: str) -> list[tuple[int, dict]]:
    """Read and decode every JSON trace event with its bag receive time."""
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import String

    reader = _open_reader(bag_dir, storage_id, [EVENT_TOPIC])
    records = []
    while reader.has_next():
        topic, data, bag_stamp_ns = reader.read_next()
        if topic != EVENT_TOPIC:
            continue
        message = deserialize_message(data, String)
        try:
            event = json.loads(message.data)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f'invalid JSON on {EVENT_TOPIC} at {bag_stamp_ns}: {exc}'
            ) from exc
        records.append((int(bag_stamp_ns), event))
    return records


def read_clock_offsets(
    bag_dir: str, storage_id: str, topic_types: dict[str, str]
) -> dict:
    """Summarize bag-receive minus header-stamp latency per sensor topic."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    selected_types = {
        topic: type_name
        for topic, type_name in topic_types.items()
        if topic in OFFSET_TOPICS
    }
    if not selected_types:
        return {}
    classes = {
        topic: get_message(type_name)
        for topic, type_name in selected_types.items()
    }
    offsets = {topic: [] for topic in selected_types}
    reader = _open_reader(
        bag_dir, storage_id, sorted(selected_types)
    )
    while reader.has_next():
        topic, data, bag_stamp_ns = reader.read_next()
        message = deserialize_message(data, classes[topic])
        header = getattr(message, 'header', None)
        if header is None:
            continue
        header_stamp_ns = (
            header.stamp.sec * 1_000_000_000 + header.stamp.nanosec
        )
        offsets[topic].append(int(bag_stamp_ns) - header_stamp_ns)

    summary = {}
    for topic, values in offsets.items():
        if not values:
            continue
        summary[topic] = {
            'count': len(values),
            'mean_offset_ns': int(round(sum(values) / len(values))),
            'min_offset_ns': min(values),
            'max_offset_ns': max(values),
            'mean_abs_offset_ns': int(round(
                sum(abs(value) for value in values) / len(values)
            )),
        }
    return summary


def read_selected_depth_hashes(
    bag_dir: str, storage_id: str
) -> set[tuple[int, str]]:
    """Return source stamp/hash identities for recorded processed frames."""
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image

    identities = set()
    reader = _open_reader(bag_dir, storage_id, [SELECTED_DEPTH_TOPIC])
    while reader.has_next():
        topic, data, _bag_stamp_ns = reader.read_next()
        if topic != SELECTED_DEPTH_TOPIC:
            continue
        message = deserialize_message(data, Image)
        if message.encoding != '32FC1':
            raise ValueError(
                f'{SELECTED_DEPTH_TOPIC} uses unexpected encoding '
                f'{message.encoding!r}'
            )
        dtype = np.dtype(np.float32)
        if message.is_bigendian:
            dtype = dtype.newbyteorder('>')
        row_stride = message.step // dtype.itemsize
        pixels = np.frombuffer(bytes(message.data), dtype=dtype).reshape(
            message.height, row_stride
        )[:, :message.width].astype(np.float32)
        stamp_ns = (
            message.header.stamp.sec * 1_000_000_000 +
            message.header.stamp.nanosec
        )
        identities.add((stamp_ns, depth_sha256(pixels)))
    return identities


def expected_selected_depths(trace: dict) -> list[dict]:
    """Extract every processed-depth reference from assembled trace events."""
    expected = []
    for event in trace['run_events']:
        if event['event'] == 'bootstrap_complete':
            expected.append({
                'observation_id': event['observation_id'],
                'stamp_ns': event['selected_depth_stamp_ns'],
                'sha256': event['selected_depth_sha256'],
            })
    for chunk in trace['chunks']:
        for event in chunk['events']:
            if event['event'] != 'selection_complete':
                continue
            for observation in event['observations']:
                expected.append({
                    'observation_id': observation['observation_id'],
                    'stamp_ns': observation['selected_depth_stamp_ns'],
                    'sha256': observation['selected_depth_sha256'],
                })
    return expected


def verify_selected_depths(
    trace: dict, recorded: set[tuple[int, str]]
) -> dict:
    """Compare trace references with the exact selected image payloads."""
    expected = expected_selected_depths(trace)
    missing = [
        reference for reference in expected
        if (reference['stamp_ns'], reference['sha256']) not in recorded
    ]
    return {
        'expected_count': len(expected),
        'recorded_unique_count': len(recorded),
        'matched_count': len(expected) - len(missing),
        'all_matched': not missing,
        'missing': missing,
    }


def export_trace(bag_path: str, output_path: str = '') -> str:
    """Export one bag and return the written JSON path."""
    bag_dir = find_bag_dir(bag_path)
    metadata = load_metadata(bag_dir)
    topic_types = {
        topic['name']: topic['type'] for topic in metadata['topics']
    }
    if EVENT_TOPIC not in topic_types:
        raise ValueError(f'bag does not contain required topic {EVENT_TOPIC}')

    trace = assemble_trace(read_events(
        bag_dir, metadata['storage_id']
    ))
    trace['bag'] = {
        'path': os.path.abspath(bag_dir),
        'storage_id': metadata['storage_id'],
        'duration_ns': int(round(metadata['duration_s'] * 1e9)),
        'message_count': metadata['message_count'],
        'header_to_bag_receive_offsets': read_clock_offsets(
            bag_dir, metadata['storage_id'], topic_types
        ),
    }
    if SELECTED_DEPTH_TOPIC in topic_types:
        trace['bag']['selected_depth_verification'] = (
            verify_selected_depths(
                trace,
                read_selected_depth_hashes(
                    bag_dir, metadata['storage_id']
                ),
            )
        )
    else:
        trace['bag']['selected_depth_verification'] = {
            'all_matched': False,
            'error': f'missing topic {SELECTED_DEPTH_TOPIC}',
        }

    destination = output_path or os.path.join(bag_dir, OUTPUT_NAME)
    destination = os.path.abspath(destination)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix='.inference_trace_',
        suffix='.json',
        dir=os.path.dirname(destination),
    )
    try:
        with os.fdopen(descriptor, 'w') as output:
            json.dump(
                trace,
                output,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            output.write('\n')
        os.replace(temporary_path, destination)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'bag', help='Bag directory or an MCAP/DB3 file inside it.'
    )
    parser.add_argument(
        '--output', default='', help='Override the output JSON path.'
    )
    args = parser.parse_args()
    destination = export_trace(args.bag, args.output)
    print(f'Saved {destination}')


if __name__ == '__main__':
    main()
