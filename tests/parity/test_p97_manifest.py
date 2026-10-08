# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'p97_manifest',
    Path(__file__).resolve().parents[2] / 'scripts/parity/p97_manifest.py',
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixtures(root):
    documents = {
        'tests/parity/requests/demo/calls.json': {
            'scenario': 'demo',
            'calls': [{}, {}],
        },
        'tests/parity/requests/tables/input.arrow': b'actual dependency bytes',
        'tests/rust_core/switch/pickles/expected.json': {'sample': {}},
        'tests/rust_core/switch/pickles/sample.pkl': b'opaque pickle bytes',
    }
    records = []
    for name, value in documents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = value if isinstance(value, bytes) else json.dumps(value).encode()
        path.write_bytes(raw)
        records.append(
            {
                'path': name,
                'sha256': hashlib.sha256(raw).hexdigest(),
                'depends_on': ['tests/parity/requests/tables/input.arrow']
                if name.endswith('calls.json')
                else [],
            }
        )
    pickle_name = 'tests/rust_core/switch/pickles/sample.pkl'
    dispositions = {
        record['path']: {'kind': 'contract', 'artifact_assertion': None}
        for record in records
    }
    dispositions[pickle_name] = {
        'kind': 'scaffold',
        'artifact_assertion': 'P9.pickle/sample',
    }
    dispositions['tests/rust_core/switch/pickles/expected.json'] = {
        'kind': 'scaffold',
        'artifact_assertion': 'P9.pickle',
    }
    return {'files': records}, {
        'fixture_dispositions': dispositions,
        'pickle_contract': {
            'pickle_expected_path': 'tests/rust_core/switch/pickles/expected.json',
            'pickles': [
                {
                    'path': pickle_name,
                    'fixture_identity': 'sample',
                    'sha256': hashlib.sha256(
                        documents[pickle_name]
                    ).hexdigest(),
                }
            ],
        },
    }


def test_manifest_preserves_dependency_leaves_and_artifacts(tmp_path):
    catalog, policy = fixtures(tmp_path)
    generated = MODULE.manifest(tmp_path, catalog, policy)
    assert generated == MODULE.manifest(tmp_path, catalog, policy)
    indexed = {record['path']: record for record in generated}
    assert indexed['tests/parity/requests/demo/calls.json'][
        'semantic_cases'
    ] == ['requests/demo/call-0', 'requests/demo/call-1']
    assert (
        indexed['tests/parity/requests/tables/input.arrow']['depends_on'] == []
    )
    assert (
        indexed['tests/parity/requests/tables/input.arrow']['semantic_cases']
        == []
    )
    assert indexed['tests/rust_core/switch/pickles/sample.pkl'][
        'depends_on'
    ] == ['tests/rust_core/switch/pickles/expected.json']


@pytest.mark.parametrize(
    'mutation',
    [
        'bytes',
        'duplicate',
        'missing_dependency',
        'self_dependency',
        'escape',
        'pickle',
    ],
)
def test_manifest_rejects_mutated_catalog(tmp_path, mutation):
    catalog, policy = fixtures(tmp_path)
    record = catalog['files'][0]
    if mutation == 'bytes':
        (tmp_path / record['path']).write_bytes(b'changed')
    elif mutation == 'duplicate':
        catalog['files'].append(record)
    elif mutation == 'missing_dependency':
        record['depends_on'] = ['absent.arrow']
    elif mutation == 'self_dependency':
        record['depends_on'] = [record['path']]
    elif mutation == 'escape':
        record['path'] = '../outside.json'
    else:
        policy['pickle_contract']['pickles'][0]['sha256'] = '0' * 64
    with pytest.raises(ValueError):
        MODULE.manifest(tmp_path, catalog, policy)


@pytest.mark.parametrize(
    ('name', 'value', 'expected'),
    [
        (
            'tests/parity/fixtures/example.json',
            {'cases': [{'id': 'one'}]},
            ['tests/parity/fixtures/example.json::one'],
        ),
        (
            'tests/parity/dialects/sqlite.json',
            {'calls': [{}]},
            ['tests/parity/dialects/sqlite.json::call-0'],
        ),
        (
            'tests/parity/frozen-public/records/example.json',
            {'format': 'p97-public-v2', 'cases': [{'id': 'public-one'}]},
            ['public-one'],
        ),
        (
            'tests/parity/frozen-generators/records/example.json',
            {'format': 'p97-generators-v1', 'cases': [{'id': 'stream-one'}]},
            ['stream-one'],
        ),
        (
            'tests/parity/sampling/demo/boundary.json',
            {'scenario': 'demo', 'calls': [{}]},
            ['sampling/demo/call-0'],
        ),
    ],
)
def test_semantic_identity_families(name, value, expected):
    assert MODULE.semantic_cases(name, value) == expected


def test_duplicate_json_key_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        MODULE.document('{"cases": [], "cases": []}')


@pytest.mark.parametrize('duplicate', [False, True])
def test_generated_dependencies_and_unique_ownership(tmp_path, duplicate):
    catalog, policy = fixtures(tmp_path)
    dependency = 'tests/parity/requests/tables/input.arrow'
    for ordinal in range(2):
        name = f'tests/parity/frozen-generators/records/{ordinal}.json'
        value = {
            'format': 'p97-generators-v1',
            'cases': [
                {
                    'id': 'same' if duplicate else f'case-{ordinal}',
                    'depends_on': [dependency],
                }
            ],
        }
        raw = json.dumps(value).encode()
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        catalog['files'].append(
            {
                'path': name,
                'sha256': hashlib.sha256(raw).hexdigest(),
                'depends_on': [],
            }
        )
    if duplicate:
        with pytest.raises(ValueError, match='duplicate semantic owner'):
            MODULE.manifest(tmp_path, catalog, policy)
    else:
        generated = MODULE.manifest(tmp_path, catalog, policy)
        owned = [
            record
            for record in generated
            if record['path'].startswith('tests/parity/frozen-generators/')
        ]
        assert len(owned) == 2
        assert all(record['depends_on'] == [dependency] for record in owned)


@pytest.mark.parametrize(
    ('name', 'value', 'expected'),
    [
        (
            'tests/parity/transcripts/memory/example/transcript.json',
            {},
            ['tests/parity/transcripts/memory/example/transcript.json'],
        ),
        (
            'tests/golden/wire_payloads.json',
            {'one': {}},
            ['tests/golden/wire_payloads.json::one'],
        ),
        (
            'rust/crates/pql/tests/accept_quirks.json',
            {'queries': [{'id': 'one'}], 'dates': [{'id': 'two'}]},
            [
                'rust/crates/pql/tests/accept_quirks.json::queries/one',
                'rust/crates/pql/tests/accept_quirks.json::dates/two',
            ],
        ),
    ],
)
def test_remaining_semantic_identity_families(name, value, expected):
    assert MODULE.semantic_cases(name, value) == expected
