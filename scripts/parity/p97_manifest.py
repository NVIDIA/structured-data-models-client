# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify actual fixture catalog bytes without recording or replaying outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def document(raw):
    def pairs(items):
        result = {}
        for name, value in items:
            if name in result:
                raise ValueError('duplicate JSON key')
            result[name] = value
        return result

    def constant(value):
        raise ValueError(f'nonfinite JSON constant: {value}')

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def fixture_path(root, name):
    relative = Path(name)
    if (
        not name
        or relative.is_absolute()
        or '..' in relative.parts
        or '\\' in name
    ):
        raise ValueError('invalid fixture path')
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or any(
        parent.is_symlink()
        for parent in (path, *path.parents)
        if parent.is_relative_to(root)
    ):
        raise ValueError('fixture path escapes source root')
    return path


def semantic_cases(name, value):
    if name.startswith('tests/parity/requests/') and name.endswith(
        '/calls.json'
    ):
        return [
            f'requests/{value["scenario"]}/call-{ordinal}'
            for ordinal in range(len(value['calls']))
        ]
    if name.startswith('tests/parity/sampling/') and name.endswith(
        '/boundary.json'
    ):
        return [
            f'sampling/{value["scenario"]}/call-{ordinal}'
            for ordinal in range(len(value['calls']))
        ]
    if value.get('format') in {'p97-public-v2', 'p97-generators-v1'}:
        return [case['id'] for case in value['cases']]
    if name.startswith('tests/parity/dialects/'):
        return [
            f'{name}::call-{ordinal}' for ordinal in range(len(value['calls']))
        ]
    if name.startswith('tests/parity/transcripts/') and name.endswith(
        '/transcript.json'
    ):
        return [name]
    if name.startswith(('tests/parity/fixtures/', 'tests/parity/pql/')):
        return [f'{name}::{case["id"]}' for case in value['cases']]
    if name == 'tests/golden/wire_payloads.json':
        return [f'{name}::{identity}' for identity in value]
    sections = {
        'rust/crates/pql/tests/accept_quirks.json': ('queries', 'dates'),
        'rust/crates/pql/tests/custom_profiles.json': ('profiles', 'options'),
    }
    if name in sections:
        return [
            f'{name}::{section}/{case["id"]}'
            for section in sections[name]
            for case in value[section]
        ]
    raise ValueError(f'no semantic identity producer for {name}')


def manifest(root, catalog, policy):
    records = catalog['files']
    names = {record['path'] for record in records}
    if len(names) != len(records):
        raise ValueError('duplicate catalog fixture')
    dispositions = policy['fixture_dispositions']
    pickle_contract = policy['pickle_contract']
    pickles = {record['path']: record for record in pickle_contract['pickles']}
    owners = set()
    result = []
    for record in sorted(records, key=lambda item: item['path']):
        name = record['path']
        raw = fixture_path(root, name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != record['sha256']:
            raise ValueError(f'catalog fixture bytes changed: {name}')
        declared = dispositions.get(
            name, {'kind': 'contract', 'artifact_assertion': None}
        )
        kind = declared['kind']
        if kind not in {'contract', 'scaffold'}:
            raise ValueError('invalid fixture classification')
        dependencies = set(record['depends_on'])
        assertion = declared['artifact_assertion']
        identities = []
        if name.endswith('.pkl'):
            binding = pickles[name]
            if (
                digest != binding['sha256']
                or assertion != f'P9.pickle/{binding["fixture_identity"]}'
            ):
                raise ValueError('pickle assertion changed')
            dependencies.add(pickle_contract['pickle_expected_path'])
        elif name.endswith('.json') and assertion is None:
            value = document(raw)
            identities = semantic_cases(name, value)
            if value.get('format') in {'p97-public-v2', 'p97-generators-v1'}:
                for case in value['cases']:
                    dependencies.update(case['depends_on'])
        elif not name.endswith(('.arrow', '.json')):
            raise ValueError(f'unsupported fixture format: {name}')
        if any(
            type(identity) is not str or not identity for identity in identities
        ):
            raise ValueError('invalid semantic identity')
        if len(set(identities)) != len(identities) or owners.intersection(
            identities
        ):
            raise ValueError('duplicate semantic owner')
        owners.update(identities)
        if name in dependencies or not dependencies <= names:
            raise ValueError('missing or self fixture dependency')
        result.append(
            {
                'path': name,
                'sha256': digest,
                'kind': kind,
                'depends_on': sorted(dependencies),
                'semantic_cases': identities,
                'artifact_assertion': assertion,
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    arguments = parser.parse_args()
    generated = manifest(
        arguments.root.resolve(),
        document(arguments.catalog.read_bytes()),
        document(arguments.policy.read_bytes()),
    )
    raw = (
        json.dumps(generated, indent=2, sort_keys=True, allow_nan=False) + '\n'
    ).encode()
    with arguments.out.open('xb') as destination:
        destination.write(raw)
    print(
        json.dumps(
            {
                'files': len(generated),
                'sha256': hashlib.sha256(raw).hexdigest(),
                'measurement_performed': False,
            }
        )
    )


if __name__ == '__main__':
    main()
