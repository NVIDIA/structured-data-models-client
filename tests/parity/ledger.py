# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inventory and change checks for the existing compatibility manifest."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def python_symbols(snapshot):
    """Include exported objects and their public members, not only __all__."""
    symbols = set()
    for module, surface in snapshot.items():
        for name in surface['__all__']:
            qualified = f'{module}.{name}'
            symbols.add(qualified)
            member = surface['members'][name]
            for attribute in member.get('public_attributes', []):
                symbols.add(f'{qualified}.{attribute}')
            for method in member.get('public_methods', {}):
                symbols.add(f'{qualified}.{method}')
    return symbols


def validate_api_inventory(actual, mappings):
    """Require explicit per-symbol mappings, without wildcard escape hatches."""
    declared = set(mappings)
    missing = set(actual) - declared
    stale = declared - set(actual)
    if missing or stale:
        raise ValueError(
            f'API mappings differ: missing={sorted(missing)}, stale={sorted(stale)}'
        )
    for symbol, record in mappings.items():
        if not record.get('capability'):
            raise ValueError(f'API has no capability: {symbol}')


def validate_coverage(data, root):
    """Promises survive deletion or narrowing of executable scenarios."""
    scenarios = {case['id']: case for case in data['scenarios']}
    covered = set()
    identifiers = set()
    apis = {group['id'] for group in data['api_contracts']}
    for capability in data['capabilities']:
        identifier = capability['id']
        if identifier in identifiers:
            raise ValueError(f'duplicate capability: {identifier}')
        identifiers.add(identifier)
        links = capability.get('api_contracts', [])
        if not links or not set(links) <= apis:
            raise ValueError(f'capability needs valid API links: {identifier}')
        for required in capability['requirements']:
            name = required['scenario']
            if name not in scenarios:
                raise ValueError(f'missing required scenario: {name}')
            case = scenarios[name]
            covered.add(name)
            fixtures = {case['fixture']} | {
                variant['fixture']
                for variant in case.get('variants', {}).values()
                if 'fixture' in variant
            }
            hashes = {
                path: hashlib.sha256(
                    (root / 'tests/parity' / path).read_bytes()
                ).hexdigest()
                for path in fixtures
            }
            if required.get('fixture_hashes') != hashes:
                raise ValueError(f'coverage fixture changed: {name}')
            for dimension in ('languages', 'models', 'connectors'):
                actual = (
                    data['languages']
                    if dimension == 'languages'
                    else case[dimension]
                )
                if set(required[dimension]) != set(actual):
                    raise ValueError(
                        f'coverage scope differs: {name}/{dimension}'
                    )
            for field in (
                'expected',
                'expected_by_model',
                'expected_by_language',
            ):
                if required.get(field, {}) != case.get(field, {}):
                    raise ValueError(
                        f'coverage expectation differs: {name}/{field}'
                    )
    if covered != scenarios.keys():
        raise ValueError(
            f'scenarios need capability mapping: {sorted(scenarios.keys() - covered)}'
        )
    for record in data['type_contracts']:
        if 'vectors' in record:
            if (
                hashlib.sha256(
                    (root / record['vectors']).read_bytes()
                ).hexdigest()
                != record['vectors_sha256']
            ):
                raise ValueError(
                    f'type vector contract changed: {record["id"]}'
                )
            vectors = json.loads((root / record['vectors']).read_text())[
                'vectors'
            ]
            if set(record['required_vectors']) != {
                vector['id'] for vector in vectors
            }:
                raise ValueError(
                    f'type vector inventory differs: {record["id"]}'
                )
        for name in record.get('scenarios', []):
            if name not in covered:
                raise ValueError(f'type coverage missing scenario: {name}')
        for tests in record.get('tests', {}).values():
            for test in tests:
                if not (root / test.split('::')[0]).is_file():
                    raise ValueError(f'type test missing: {test}')


def changed_records(before, after):
    """Diff promises, independently of which scenarios remain executable."""
    changes = set()
    for section in (
        'api_contracts',
        'scenarios',
        'capabilities',
        'type_contracts',
    ):
        previous = {record['id']: record for record in before.get(section, [])}
        current = {record['id']: record for record in after.get(section, [])}
        for identifier in previous.keys() | current.keys():
            if previous.get(identifier) != current.get(identifier):
                changes.add(f'{section}/{identifier}')
    for section in (
        'languages',
        'models',
        'connectors',
        'deferrals',
        'input_paths',
        'api_inventory',
        'api_exceptions',
        'live_api_tests',
        'public_api_snapshot_sha256',
    ):
        if before.get(section) != after.get(section):
            changes.add(section)
    return changes


def validate_changes(before, after, base_commit):
    """Decisions are tied to the reviewed merge base, not reusable blanket waivers."""
    decisions = {}
    for decision in after.get('contract_changes', []):
        if decision.get('base_commit') != base_commit:
            continue
        key = decision['record']
        if key in decisions:
            raise ValueError(f'duplicate contract decision: {key}')
        if not decision.get('reason', '').strip():
            raise ValueError(f'contract decision needs a reason: {key}')
        decisions[key] = decision
    missing = changed_records(before, after) - decisions.keys()
    if missing:
        raise ValueError(f'contract changes need decisions: {sorted(missing)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--base',
        required=True,
        help='Target Git ref; compare against its merge base',
    )
    parser.add_argument('--java-inventory', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]

    def git(*arguments):
        return subprocess.run(
            ['git', *arguments],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()

    from api_contracts import validate

    from contract import manifest

    data = manifest()
    validate(data)
    validate_coverage(data, root)
    validate_api_inventory(
        json.loads(args.java_inventory.read_text()),
        data['api_inventory']['java'],
    )
    base = git('merge-base', args.base, 'HEAD')
    before = json.loads(git('show', f'{base}:tests/parity/manifest.json'))
    validate_changes(before, data, base)
    print(
        json.dumps(
            {
                'base_commit': base,
                'python_symbols': len(data['api_inventory']['python']),
                'java_symbols': len(data['api_inventory']['java']),
                'status': 'passed',
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
