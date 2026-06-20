#!/usr/bin/env python3
"""
inspect_run.py — summarize the contents of one recorded run (a rosbag2 bag).

A first look at the data before we build any training pipeline.  It prints, for
every channel (topic) in the bag:

  • the message type,
  • how many messages were recorded (and an approximate rate in Hz),
  • the "shape" of the payload — the fields and, for arrays, their lengths —
    inferred from the first message of that topic.

TWO LEVELS OF DETAIL
--------------------
Channel names, types, and counts come from the bag's metadata.yaml, so this
script prints them anywhere (host or container) with only PyYAML.

The per-field SHAPE requires deserializing a sample CDR message, which needs the
ROS 2 message definitions.  Run inside the container (with ROS 2 sourced) to get
that extra detail; on a plain host you still get the channel summary.

USAGE
-----
    python3 inspect_run.py /path/to/runs/drawer_2026-06-20_14-14-03

The path may be the bag directory (the folder containing metadata.yaml) or a
storage file inside it (.db3 / .mcap).
"""

import argparse
import array
import os
import re
import sys

import yaml


# ── locating + reading bag metadata (no ROS needed) ──────────────────────────


def find_bag_dir(path: str) -> str:
    """Return the bag directory given a directory or a file inside it."""
    if os.path.isdir(path):
        bag_dir = path
    elif os.path.isfile(path):
        bag_dir = os.path.dirname(os.path.abspath(path))
    else:
        sys.exit(f"Path does not exist: {path}")

    if not os.path.isfile(os.path.join(bag_dir, 'metadata.yaml')):
        sys.exit(f"No metadata.yaml found in {bag_dir} — is this a rosbag2 bag?")
    return bag_dir


def load_metadata(bag_dir: str) -> dict:
    """Parse metadata.yaml into a small, convenient summary dict."""
    with open(os.path.join(bag_dir, 'metadata.yaml')) as f:
        info = yaml.safe_load(f)['rosbag2_bagfile_information']

    topics = []
    for entry in info['topics_with_message_count']:
        meta = entry['topic_metadata']
        topics.append({
            'name': meta['name'],
            'type': meta['type'],
            'count': entry['message_count'],
        })

    return {
        'storage_id': info['storage_identifier'],
        'duration_s': info['duration']['nanoseconds'] / 1e9,
        'message_count': info['message_count'],
        'topics': sorted(topics, key=lambda t: t['name']),
    }


# ── shape introspection of a deserialized message (needs ROS) ─────────────────


def _is_array_type(ftype: str) -> bool:
    return ftype.startswith('sequence<') or bool(re.search(r'\[\d*\]$', ftype))


def describe(msg, indent: str = '    ', depth: int = 0, max_depth: int = 5):
    """Recursively print a ROS message's fields and array lengths."""
    if not hasattr(msg, 'get_fields_and_field_types'):
        return
    for fname, ftype in msg.get_fields_and_field_types().items():
        value = getattr(msg, fname)
        line = f"{indent}{fname}: {ftype}"

        if _is_array_type(ftype):
            try:
                n = len(value)
            except TypeError:
                n = '?'
            print(f"{line}  (len={n})")
            if isinstance(n, int) and n > 0 and depth < max_depth:
                first = value[0]
                if hasattr(first, 'get_fields_and_field_types'):
                    print(f"{indent}  [0]:")
                    describe(first, indent + '    ', depth + 1, max_depth)
                elif n <= 12:
                    # short scalar array — the values themselves are informative
                    vals = list(value) if isinstance(value, (array.array, list)) \
                        else [value[i] for i in range(n)]
                    print(f"{indent}    = {vals}")
        elif hasattr(value, 'get_fields_and_field_types'):
            print(line)
            if depth < max_depth:
                describe(value, indent + '    ', depth + 1, max_depth)
        else:
            # scalar — show short values inline
            shown = repr(value)
            print(f"{line} = {shown}" if len(shown) <= 60 else line)


def read_samples(bag_dir: str, storage_id: str, type_map: dict) -> dict:
    """Return {topic_name: first deserialized message} using rosbag2_py."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        ),
    )

    classes = {name: get_message(t) for name, t in type_map.items()}
    samples: dict = {}
    while reader.has_next() and len(samples) < len(type_map):
        name, data, _stamp = reader.read_next()
        if name in classes and name not in samples:
            samples[name] = deserialize_message(data, classes[name])
    return samples


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag', help='Bag directory or a storage file inside it.')
    args = parser.parse_args()

    bag_dir = find_bag_dir(args.bag)
    meta = load_metadata(bag_dir)

    print(f"Run: {bag_dir}")
    print(f"  storage   : {meta['storage_id']}")
    print(f"  duration  : {meta['duration_s']:.2f} s")
    print(f"  messages  : {meta['message_count']} across {len(meta['topics'])} channel(s)")
    print()

    # Try to deserialize a sample per topic for shape info. Degrade gracefully
    # to a metadata-only summary if ROS 2 isn't available (e.g. on the host).
    type_map = {t['name']: t['type'] for t in meta['topics']}
    samples = None
    try:
        samples = read_samples(bag_dir, meta['storage_id'], type_map)
    except ImportError:
        print("(ROS 2 not found — showing channels only. Run inside the "
              "container with ROS 2 sourced to see field shapes.)\n")

    for t in meta['topics']:
        rate = t['count'] / meta['duration_s'] if meta['duration_s'] > 0 else 0.0
        print(f"{t['name']}")
        print(f"  type  : {t['type']}")
        print(f"  count : {t['count']} msgs  (~{rate:.1f} Hz)")
        if samples is not None:
            sample = samples.get(t['name'])
            if sample is None:
                print("  shape : (no messages recorded)")
            else:
                print("  shape :")
                describe(sample, indent='    ')
        print()


if __name__ == '__main__':
    main()
