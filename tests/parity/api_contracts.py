# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select existing API behavior tests, without expanding connector/model cells."""

import argparse
import hashlib
import json
from pathlib import Path

from contract import HERE, manifest

ROOT = HERE.parents[1]


def groups(data, gate='local', selected=None):
    return [
        group
        for group in data['api_contracts']
        if group['gate'] == gate and (not selected or group['id'] in selected)
    ]


def validate(data):
    assert (
        hashlib.sha256(
            (ROOT / 'tests/golden/public_api.json').read_bytes()
        ).hexdigest()
        == data['public_api_snapshot_sha256']
    ), 'public surface changed: update parity API mappings'
    seen = set()
    for group in data['api_contracts']:
        assert group['id'] not in seen, 'duplicate API contract'
        seen.add(group['id'])
        assert set(group['apis']) >= {'python', 'java'}
        for tests in group['tests'].values():
            for node in tests:
                path = ROOT / node.split('::')[0]
                assert path.is_file(), node
                if '::' in node:
                    assert (
                        'def ' + node.split('::')[1] + '(' in path.read_text()
                    ), node
    return seen


class Outcomes:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def pytest_runtest_logreport(self, report):
        if report.failed:
            self.failed += 1
        elif report.skipped:
            self.skipped += 1
        elif report.when == 'call' and report.passed:
            self.passed += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--language', choices=['python', 'java', 'rust'], default='python'
    )
    parser.add_argument(
        '--gate', choices=['local', 'native', 'large-memory'], default='local'
    )
    parser.add_argument('--group', action='append')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    data = manifest()
    validate(data)
    selected = groups(data, args.gate, args.group)
    nodes = sorted(
        {
            node
            for group in selected
            for node in group['tests'].get(args.language, [])
        }
    )
    if not args.run:
        print(
            json.dumps(
                {'groups': [group['id'] for group in selected], 'tests': nodes},
                indent=2,
            )
        )
        return 0 if nodes else 1
    if args.language != 'python' or args.out is None:
        parser.error(
            '--run uses installed Python pytest and requires --out; Java/Rust reuse their existing test runners'
        )
    import pytest

    outcomes = Outcomes()
    status = pytest.main(
        [*(str(ROOT / node) for node in nodes), '-q'], plugins=[outcomes]
    )
    complete = (
        bool(nodes)
        and status == 0
        and outcomes.passed > 0
        and outcomes.skipped == 0
    )
    args.out.write_text(
        json.dumps(
            {
                'gate': args.gate,
                'groups': [group['id'] for group in selected],
                'status': 'passed'
                if complete
                else 'incomplete'
                if outcomes.skipped
                else 'failed',
                'passed': outcomes.passed,
                'failed': outcomes.failed,
                'skipped': outcomes.skipped,
                'pytest_exit': int(status),
            },
            indent=2,
        )
        + '\n'
    )
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
