# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generated evidence views preserve declaration, identity and execution boundaries."""

import json

import pytest
from developer_views import execution_evidence, render
from jsonschema import ValidationError
from test_contract import one, paired

COMMIT = 'a' * 40


def observations(tmp_path, *, live=True, mutation=None):
    data = one()
    _, records = paired(data)
    for record in records.values():
        record['artifact'].update(
            commit=COMMIT, package_hash='b' * 64, native_hash='c' * 64
        )
        if not live:
            record['execution'] = 'local'
            record['semantic']['lifecycle'] = ['materialize']
            record['semantic']['results'] = []
    if mutation:
        mutation(next(iter(records.values())))
    path = tmp_path / 'observations.jsonl'
    path.write_text(
        ''.join(json.dumps(record) + '\n' for record in records.values())
    )
    return data, path


@pytest.mark.parametrize('live', [False, True])
def test_receipts_show_actual_status_separately_from_declared_support(
    tmp_path, live
):
    data, path = observations(tmp_path, live=live)
    evidence = execution_evidence(data, COMMIT, [(path, live)])
    assert {row['status'] for row in evidence['scenarios']} == {
        'live passed' if live else 'local passed'
    }
    text = render(data, evidence)
    assert '| supported |' in text
    assert COMMIT in text and 'b' * 64 in text
    assert ('model-v1' in text) is True
    assert 'live passed' not in text if not live else 'live passed' in text


@pytest.mark.parametrize(
    'field,value',
    [('commit', 'd' * 40), ('native_hash', ''), ('package_hash', '')],
)
def test_mismatched_artifacts_cannot_supply_pass_evidence(
    tmp_path, field, value
):
    data, path = observations(
        tmp_path,
        mutation=lambda record: record['artifact'].__setitem__(field, value),
    )
    with pytest.raises((ValueError, ValidationError)):
        execution_evidence(data, COMMIT, [(path, True)])


def test_stub_or_missing_model_identity_cannot_become_live_pass(tmp_path):
    data, path = observations(
        tmp_path,
        mutation=lambda record: record.__setitem__(
            'endpoint', {'kind': 'stub'}
        ),
    )
    with pytest.raises(ValidationError):
        execution_evidence(data, COMMIT, [(path, True)])


def test_live_api_aggregate_pass_needs_actual_completed_identity(tmp_path):
    data = one()
    data['live_api_tests'] = [
        {'api_contract': 'reports', 'test': 'test_reports.py::test_view'}
    ]
    path = tmp_path / 'receipt.json'
    receipt = {
        'candidate_commit': COMMIT,
        'requirements': data['live_api_tests'],
        'artifacts': {
            'python': {
                'commit': COMMIT,
                'package_hash': 'b' * 64,
                'native_hash': 'c' * 64,
                'contract': 70,
            }
        },
        'models': {
            name: {
                'version': '27',
                'transport': {
                    'pretrained': 'in-process',
                    'gradients': 'http',
                }.get(name, 'nim'),
            }
            for name in ('tabular', 'relational', 'pretrained', 'gradients')
        },
        'results': {
            'python': {
                'status': 'passed',
                'exit_code': 0,
                'required_methods': ['test_reports.py::test_view'],
                'completed_tests': [],
            }
        },
    }
    path.write_text(json.dumps(receipt))
    assert (
        execution_evidence(data, COMMIT, api_receipts=[path])['api_tests'][0][
            'status'
        ]
        == 'unavailable'
    )
    receipt['results']['python']['completed_tests'] = [
        'test_reports.py::test_view[case]'
    ]
    path.write_text(json.dumps(receipt))
    assert (
        execution_evidence(data, COMMIT, api_receipts=[path])['api_tests'][0][
            'status'
        ]
        == 'live passed'
    )
    receipt['artifacts']['python']['commit'] = 'd' * 40
    path.write_text(json.dumps(receipt))
    with pytest.raises((ValueError, ValidationError)):
        execution_evidence(data, COMMIT, api_receipts=[path])


@pytest.mark.parametrize(
    'models',
    [
        {'case_routes': {}},
        {
            name: {'version': 27, 'transport': 'nim'}
            for name in ('tabular', 'relational', 'pretrained', 'gradients')
        },
        {
            name: {'version': '27', 'transport': 'unknown'}
            for name in ('tabular', 'relational', 'pretrained', 'gradients')
        },
    ],
)
def test_malformed_model_identity_rejected(models):
    from developer_views import model_identities

    with pytest.raises(ValueError, match='model identity'):
        model_identities(models)


def test_source_equivalent_java_preserves_execution_identity():
    from developer_views import artifact_identity, identity_text

    artifact = dict(
        execution='source-equivalent',
        source_commit=COMMIT,
        tree_clean=True,
        native_hash='b' * 64,
        contract=70,
        verified_reference_jar_hash='c' * 64,
    )
    assert artifact_identity(artifact, COMMIT)[0] == 'source-equivalent'
    assert 'verified_reference_jar_hash=' in identity_text(artifact)
    assert 'package_hash=' not in identity_text(artifact)
    for key, value in [
        ('tree_clean', False),
        ('source_commit', 'd' * 40),
        ('contract', True),
        ('verified_reference_jar_hash', ''),
    ]:
        with pytest.raises(ValueError, match='artifact identity'):
            artifact_identity({**artifact, key: value}, COMMIT)


@pytest.mark.parametrize(
    'field', ['verified_reference_jar_hash', 'native_hash']
)
def test_source_equivalent_java_must_match_recorded_artifact(tmp_path, field):
    data, path = observations(tmp_path)
    test = 'ExampleTest#testView'
    data['live_api_tests'] = [{'api_contract': 'reports', 'test': test}]
    receipt = {
        'candidate_commit': COMMIT,
        'requirements': data['live_api_tests'],
        'artifacts': {
            'java': {
                'execution': 'source-equivalent',
                'source_commit': COMMIT,
                'tree_clean': True,
                'verified_reference_jar_hash': 'b' * 64,
                'native_hash': 'c' * 64,
                'contract': 1,
            }
        },
        'models': {
            name: {'version': '27', 'transport': transport}
            for name, transport in [
                ('tabular', 'nim'),
                ('relational', 'nim'),
                ('pretrained', 'in-process'),
                ('gradients', 'http'),
            ]
        },
        'results': {
            'java': {
                'status': 'passed',
                'exit_code': 0,
                'required_methods': [test],
                'completed_tests': [test],
            }
        },
    }
    # Match the independently supplied observation's contract, not a fixed guess.
    records = [json.loads(line) for line in path.read_text().splitlines()]
    receipt['artifacts']['java']['contract'] = next(
        record['artifact']['contract']
        for record in records
        if record['language'] == 'java'
    )
    receipt_path = tmp_path / 'api.json'
    receipt_path.write_text(json.dumps(receipt))
    result = execution_evidence(data, COMMIT, [(path, True)], [receipt_path])
    assert result['api_tests'][0]['status'] == 'live passed'
    receipt['artifacts']['java'][field] = 'd' * 64
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='differs from observations'):
        execution_evidence(data, COMMIT, [(path, True)], [receipt_path])
