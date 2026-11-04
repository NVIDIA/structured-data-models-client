# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select existing API behavior tests, without expanding connector/model cells."""

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from ledger import python_symbols, validate_api_inventory

from contract import HERE, manifest

ROOT = HERE.parents[1]


def groups(data, gate='local', selected=None):
    if gate == 'live':
        return [
            {
                'id': record['api_contract'],
                'tests': {
                    'java' if '#' in record['test'] else 'python': [
                        record['test']
                    ]
                },
            }
            for record in data['live_api_tests']
            if not selected or record['api_contract'] in selected
        ]
    return [
        group
        for group in data['api_contracts']
        if group['gate'] == gate and (not selected or group['id'] in selected)
    ]


def validate(data):
    assert (
        hashlib.sha256(
            (ROOT / 'tests/golden/public_api.json')
            .read_bytes()
            .replace(b'\r\n', b'\n')
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
                path = ROOT / node.split('::')[0].split('#')[0]
                assert path.is_file(), node
                if '::' in node:
                    scope = ast.parse(path.read_text()).body
                    for name in node.split('::')[1:]:
                        matches = [
                            item
                            for item in scope
                            if getattr(item, 'name', None) == name
                        ]
                        assert len(matches) == 1, node
                        scope = getattr(matches[0], 'body', [])
    validate_live_targets(data, seen)
    inventory = data['api_inventory']
    validate_api_inventory(
        python_symbols(
            json.loads((ROOT / 'tests/golden/public_api.json').read_text())
        ),
        inventory['python'],
    )
    contracts = {group['id']: group for group in data['api_contracts']}
    exceptions = set()
    for exception in data['api_exceptions']:
        key = (exception['capability'], exception['language'])
        assert key not in exceptions, ('duplicate applicability exception', key)
        assert key[0] in contracts and key[1] in data['languages'], key
        assert exception['reason'].strip(), key
        assert not contracts[key[0]]['tests'].get(key[1]), (
            'stale exception',
            key,
        )
        exceptions.add(key)
    for capability, group in contracts.items():
        for language in data['languages']:
            assert (
                group['tests'].get(language)
                or (capability, language) in exceptions
            ), (capability, language)
    for language, mappings in inventory.items():
        for symbol, record in mappings.items():
            capability = record['capability']
            assert capability in contracts, (symbol, capability)
            assert contracts[capability]['tests'].get(language), (
                symbol,
                language,
            )
    return seen


def validate_live_targets(data, capabilities):
    """Resolve every live registry entry without connecting to any service."""
    seen = set()
    for record in data['live_api_tests']:
        node = record['test']
        identity = (record['api_contract'], node)
        assert identity not in seen, ('duplicate live requirement', identity)
        seen.add(identity)
        assert record['api_contract'] in capabilities, identity
        assert record['evidence_level'] == 'live-required', identity
        path, separator, name = node.partition('#' if '#' in node else '::')
        source = ROOT / path
        assert separator and name and source.is_file(), node
        if separator == '::':
            scope = ast.parse(source.read_text()).body
            for part in name.split('::'):
                matches = [
                    item
                    for item in scope
                    if getattr(item, 'name', None) == part
                ]
                assert len(matches) == 1, node
                scope = getattr(matches[0], 'body', [])
        else:
            assert (
                type(record.get('invocations', 1)) is int
                and record.get('invocations', 1) > 0
            ), node
            assert re.search(
                r'\bvoid\s+' + re.escape(name) + r'\s*\(', source.read_text()
            ), node


class Outcomes:
    def __init__(self, live_owned=()):
        self.live_owned = frozenset(live_owned)
        self.live_owned_tests = []
        self.skipped_tests = []
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.expected = []
        self.completed = []
        self.deselected = []

    def pytest_collection_modifyitems(self, items):
        # Whole-module local ownership must not pull explicitly registered live
        # methods into an offline gate. All other skips remain failures.
        retained = []
        for item in items:
            if (
                item.nodeid.split('[', 1)[0].replace('\\', '/')
                in self.live_owned
            ):
                self.live_owned_tests.append(item.nodeid)
            else:
                retained.append(item)
        items[:] = retained

    def pytest_collection_finish(self, session):
        self.expected = [item.nodeid for item in session.items]

    def pytest_deselected(self, items):
        self.deselected.extend(item.nodeid for item in items)

    @property
    def complete(self):
        return (
            bool(self.expected)
            and len(self.expected) == len(set(self.expected))
            and len(self.completed) == len(set(self.completed))
            and set(self.expected) == set(self.completed)
            and not self.deselected
            and not self.failed
            and not self.skipped
        )

    def pytest_runtest_logreport(self, report):
        if report.failed:
            self.failed += 1
        elif report.skipped:
            self.skipped += 1
            self.skipped_tests.append(report.nodeid)
        elif report.when == 'call' and report.passed:
            self.passed += 1
            self.completed.append(report.nodeid)


def rust_run(nodes, output):
    """Use Cargo/libtest directly and reconcile its collected test identities."""
    targets = {}
    for node in nodes:
        path, _, name = node.partition('::')
        targets.setdefault(path, set()).add(name)
    evidence = []
    for path, required in sorted(targets.items()):
        source = Path(path)
        command = [
            'cargo',
            'test',
            '--locked',
            '-p',
            source.parts[2],
            '--test',
            source.stem,
            '--color',
            'never',
        ]
        listed = subprocess.run(
            [*command, '--', '--list'],
            cwd=ROOT / 'rust',
            text=True,
            capture_output=True,
            check=True,
        )
        expected = [
            line.removesuffix(': test')
            for line in listed.stdout.splitlines()
            if line.endswith(': test')
        ]
        if not expected or not (required - {''}) <= set(expected):
            raise ValueError(f'missing required Rust tests: {path}: {required}')
        ran = subprocess.run(
            command,
            cwd=ROOT / 'rust',
            text=True,
            capture_output=True,
            check=False,
        )
        completed = re.findall(r'^test (.+) \.\.\. ok$', ran.stdout, re.M)
        complete = (
            ran.returncode == 0
            and len(expected) == len(set(expected))
            and len(completed) == len(set(completed))
            and set(expected) == set(completed)
        )
        evidence.append(
            {
                'target': path,
                'expected_tests': expected,
                'completed_tests': completed,
                'status': 'passed' if complete else 'failed',
            }
        )
        output.write_text(json.dumps(evidence, indent=2) + '\n')
        if not complete:
            raise ValueError(
                f'Rust test evidence incomplete for {path}:\n{ran.stdout}\n{ran.stderr}'
            )
    return 0 if evidence else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--language', choices=['python', 'java', 'rust'], default='python'
    )
    parser.add_argument(
        '--gate',
        choices=['local', 'native', 'large-memory', 'in-process', 'live'],
        default='local',
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
    if args.gate == 'native':
        nodes = sorted(
            set(nodes)
            | {
                node
                for record in data['type_contracts']
                for node in record.get('tests', {}).get(args.language, [])
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
    if not nodes:
        parser.error('no tests selected for this language and gate')
    if args.run and args.language == 'rust' and args.out is not None:
        return rust_run(nodes, args.out)
    if args.language != 'python' or args.out is None:
        parser.error(
            '--run uses installed Python pytest and requires --out; Java/Rust reuse their existing test runners'
        )
    # Pytest fixtures also import repository-owned tests.parity helpers.
    sys.path.insert(0, str(ROOT))
    import pytest

    outcomes = Outcomes(
        live_owned=(
            record['test']
            for record in data['live_api_tests']
            if '::' in record['test']
        )
        if args.gate == 'local'
        else ()
    )
    status = pytest.main(
        [
            *(str(ROOT / node) for node in nodes),
            '-q',
            '-ra',
            *(['-m', '', '-o', 'addopts='] if args.gate == 'live' else []),
        ],
        plugins=[outcomes],
    )
    complete = bool(nodes) and status == 0 and outcomes.complete
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
                'expected_tests': outcomes.expected,
                'completed_tests': outcomes.completed,
                'deselected_tests': outcomes.deselected,
                'live_owned_tests': outcomes.live_owned_tests,
                'skipped_tests': outcomes.skipped_tests,
            },
            indent=2,
        )
        + '\n'
    )
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
