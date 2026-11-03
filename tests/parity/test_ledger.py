# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Missing APIs and deleted promises cannot disappear with their tests."""

import copy

import pytest
from ledger import python_symbols, validate_api_inventory, validate_changes


def test_inventory_includes_methods_and_properties():
    snapshot = {
        'client': {
            '__all__': ['Schema'],
            'members': {
                'Schema': {
                    'public_attributes': ['tables'],
                    'public_methods': {'read': '()'},
                },
            },
        }
    }
    actual = python_symbols(snapshot)
    assert actual == {
        'client.Schema',
        'client.Schema.tables',
        'client.Schema.read',
    }
    mapping = {symbol: {'capability': 'schema'} for symbol in actual}
    validate_api_inventory(actual, mapping)
    with pytest.raises(ValueError, match='missing'):
        validate_api_inventory(actual | {'client.new_api'}, mapping)
    with pytest.raises(ValueError, match='stale'):
        validate_api_inventory(actual - {'client.Schema.read'}, mapping)


def test_deleting_capability_and_its_scenario_requires_both_decisions():
    before = {
        'capabilities': [{'id': 'predict', 'languages': ['python', 'java']}],
        'scenarios': [{'id': 'predict-live'}],
    }
    after = {'capabilities': [], 'scenarios': []}
    with pytest.raises(ValueError, match='capabilities/predict'):
        validate_changes(before, after, 'base')
    after['contract_changes'] = [
        {
            'base_commit': 'base',
            'record': record,
            'reason': 'Explicit retirement',
        }
        for record in ('capabilities/predict', 'scenarios/predict-live')
    ]
    validate_changes(before, after, 'base')
    with pytest.raises(ValueError, match='need decisions'):
        validate_changes(before, after, 'different-base')


def test_narrowing_language_support_requires_a_decision():
    before = {
        'capabilities': [{'id': 'predict', 'languages': ['python', 'java']}]
    }
    after = copy.deepcopy(before)
    after['capabilities'][0]['languages'].remove('java')
    with pytest.raises(ValueError, match='capabilities/predict'):
        validate_changes(before, after, 'base')


def test_required_scenarios_are_independent_of_runner_selection():
    from ledger import validate_coverage

    from contract import HERE, manifest

    data = manifest()
    validate_coverage(data, HERE.parents[1])
    missing = copy.deepcopy(data)
    missing['scenarios'].pop(0)
    with pytest.raises(ValueError, match='missing required scenario'):
        validate_coverage(missing, HERE.parents[1])
    narrowed = copy.deepcopy(data)
    narrowed['scenarios'][0]['connectors'].pop()
    with pytest.raises(ValueError, match='coverage scope differs'):
        validate_coverage(narrowed, HERE.parents[1])
    wrong = copy.deepcopy(data)
    wrong['scenarios'][0]['expected'] = {'status': 'refused'}
    with pytest.raises(ValueError, match='coverage expectation differs'):
        validate_coverage(wrong, HERE.parents[1])


def test_type_fixture_changes_need_contract_updates():
    from ledger import validate_coverage

    from contract import HERE, manifest

    data = manifest()
    data['capabilities'][0]['requirements'][0]['fixture_hashes'] = {}
    with pytest.raises(ValueError, match='coverage fixture changed'):
        validate_coverage(data, HERE.parents[1])
    data = manifest()
    data['type_contracts'][0]['vectors_sha256'] = 'stale'
    with pytest.raises(ValueError, match='type vector contract changed'):
        validate_coverage(data, HERE.parents[1])


def test_language_exceptions_are_required_and_cannot_be_stale():
    from api_contracts import validate

    from contract import manifest

    data = manifest()
    data['api_exceptions'] = []
    with pytest.raises(AssertionError):
        validate(data)
    data = manifest()
    data['api_exceptions'].append(
        {
            'capability': 'sessions',
            'language': 'java',
            'reason': 'Not applicable',
        }
    )
    with pytest.raises(AssertionError, match='stale exception'):
        validate(data)
