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


def test_developer_contracts_match_current_operations_and_owners():
    from ledger import validate_developer_contracts

    from contract import HERE, manifest

    validate_developer_contracts(manifest(), HERE.parents[1])


@pytest.mark.parametrize(
    ('mutate', 'message'),
    [
        (lambda group: group.pop('ownership'), 'ownership'),
        (
            lambda group: group['ownership'].update(operations=['invented']),
            'unknown native operation',
        ),
        (
            lambda group: group['ownership'].update(operations=[]),
            'host-only reason',
        ),
        (
            lambda group: group['scope'].pop('scenarios'),
            'applicability missing',
        ),
        (
            lambda group: group['apis'].pop('java'),
            'language applicability missing',
        ),
    ],
)
def test_developer_contract_missing_information_fails(mutate, message):
    from ledger import validate_developer_contracts

    from contract import HERE, manifest

    data = manifest()
    mutate(data['api_contracts'][0])
    with pytest.raises(ValueError, match=message):
        validate_developer_contracts(data, HERE.parents[1])


def test_changed_signatures_need_direct_assertions_but_private_edits_do_not():
    from ledger import validate_changed_symbols

    before = {'api_inventory': {'python': {}, 'java': {}}}
    after = copy.deepcopy(before)
    old = {
        'client': {
            '__all__': ['read'],
            'members': {'read': {'signature': '(source)'}},
        }
    }
    new = copy.deepcopy(old)
    validate_changed_symbols(before, after, old, new)
    new['client']['members']['read']['signature'] = '(source, options=None)'
    after['api_inventory']['python']['client.read'] = {'capability': 'readers'}
    with pytest.raises(ValueError, match='explicit behavior assertions'):
        validate_changed_symbols(before, after, old, new)
    after['api_inventory']['python']['client.read']['assertions'] = [
        {'test': 'test_read', 'behavior': 'Options preserve source ownership'}
    ]
    validate_changed_symbols(before, after, old, new)
    after['api_inventory']['java']['Reader#read(Source, Options)'] = {
        'capability': 'readers'
    }
    with pytest.raises(ValueError, match='java/Reader'):
        validate_changed_symbols(before, after, old, new)


def test_assertion_must_name_a_group_test_and_observable_behavior():
    from ledger import validate_developer_contracts

    from contract import HERE, manifest

    data = manifest()
    symbol, record = next(iter(data['api_inventory']['python'].items()))
    record['assertions'] = [{'test': 'missing', 'behavior': 'Expected output'}]
    with pytest.raises(ValueError, match='needs a group test'):
        validate_developer_contracts(data, HERE.parents[1])
    group = next(
        g for g in data['api_contracts'] if g['id'] == record['capability']
    )
    record['assertions'][0] = {
        'test': group['tests']['python'][0],
        'behavior': '',
    }
    with pytest.raises(ValueError, match='needs a description'):
        validate_developer_contracts(data, HERE.parents[1])


def test_disposable_view_does_not_turn_declarations_into_passes():
    from developer_views import render

    from contract import manifest

    text = render(manifest())
    assert 'absent evidence is untested' in text
    assert (
        'existing group coverage; no direct symbol assertion declared' in text
    )
    assert 'gradient-explanations' in text


@pytest.mark.parametrize('axis', ['scenarios', 'input_paths', 'type_contracts'])
def test_stale_structured_scope_reference_fails(axis):
    from ledger import validate_developer_contracts

    from contract import HERE, manifest

    data = manifest()
    data['api_contracts'][0]['scope'][axis]['refs'].append('missing')
    with pytest.raises(ValueError, match='stale scope reference'):
        validate_developer_contracts(data, HERE.parents[1])


def test_existing_scenario_cannot_be_attached_to_unrelated_group():
    from ledger import validate_developer_contracts

    from contract import HERE, manifest

    data = manifest()
    group = next(
        group
        for group in data['api_contracts']
        if group['id'] == 'release-metadata'
    )
    group['scope']['scenarios']['refs'] = [data['scenarios'][0]['id']]
    with pytest.raises(ValueError, match='scope scenario linkage differs'):
        validate_developer_contracts(data, HERE.parents[1])


def test_input_route_must_link_to_group_or_scenario_entry_points():
    from ledger import validate_developer_contracts

    from contract import HERE, manifest

    data = manifest()
    group = next(
        group
        for group in data['api_contracts']
        if group['id'] == 'release-metadata'
    )
    group['scope']['input_paths']['refs'] = ['frames']
    with pytest.raises(ValueError, match='scope input route not linked'):
        validate_developer_contracts(data, HERE.parents[1])


def test_type_journey_requires_linked_group_scenario():
    from ledger import validate_developer_contracts

    from contract import HERE, manifest

    data = manifest()
    group = next(
        group
        for group in data['api_contracts']
        if group['id'] == 'release-metadata'
    )
    group['scope']['type_contracts']['refs'] = ['reader-model-type-journeys']
    with pytest.raises(ValueError, match='type journey not linked'):
        validate_developer_contracts(data, HERE.parents[1])


@pytest.mark.parametrize('change', ['added', 'signature'])
def test_first_adoption_requires_direct_python_assertions(change):
    from ledger import validate_introduction

    old = {
        'client': {
            '__all__': ['read'],
            'members': {'read': {'signature': '(source)'}},
        }
    }
    new = copy.deepcopy(old)
    if change == 'added':
        new['client']['__all__'].append('write')
        new['client']['members']['write'] = {'signature': '(source)'}
        symbol = 'client.write'
    else:
        new['client']['members']['read']['signature'] = '(source, options=None)'
        symbol = 'client.read'
    data = {
        'api_inventory': {'python': {symbol: {}}, 'java': {}},
        'contract_changes': [
            {
                'base_commit': 'base',
                'record': 'api_inventory/java',
                'reason': 'Baseline contains no Java artifact.',
            }
        ],
    }
    with pytest.raises(ValueError, match='explicit behavior assertions'):
        validate_introduction(data, old, new, 'base', baseline_has_java=False)
    data['api_inventory']['python'][symbol]['assertions'] = [
        {
            'test': 'test_public.py::test_options',
            'behavior': 'Independent expected options are retained.',
        }
    ]
    validate_introduction(data, old, new, 'base', baseline_has_java=False)
    with pytest.raises(ValueError, match='Java baseline'):
        validate_introduction(data, old, new, 'base', baseline_has_java=True)


def test_first_adoption_removal_needs_actual_baseline_decision():
    from ledger import validate_introduction

    old = {
        'client': {
            '__all__': ['read'],
            'members': {'read': {'signature': '()'}},
        }
    }
    new = {'client': {'__all__': [], 'members': {}}}
    data = {
        'api_inventory': {'python': {}, 'java': {}},
        'contract_changes': [
            {
                'base_commit': 'base',
                'record': 'api_inventory/java',
                'reason': 'New Java host.',
            }
        ],
    }
    with pytest.raises(ValueError, match='removal needs'):
        validate_introduction(data, old, new, 'base', baseline_has_java=False)
    data['contract_changes'].append(
        {
            'base_commit': 'different',
            'record': 'api_inventory/python/client.read',
            'reason': 'Reviewed replacement.',
        }
    )
    with pytest.raises(ValueError, match='removal needs'):
        validate_introduction(data, old, new, 'base', baseline_has_java=False)
    data['contract_changes'][-1]['base_commit'] = 'base'
    validate_introduction(data, old, new, 'base', baseline_has_java=False)


def test_assertion_child_node_must_exist_in_selected_file(tmp_path):
    from ledger import validate_assertion_node

    (tmp_path / 'test_public.py').write_text('def test_read(): pass\n')
    validate_assertion_node(
        'test_public.py::test_read', ['test_public.py'], tmp_path
    )
    with pytest.raises(ValueError, match='missing test'):
        validate_assertion_node(
            'test_public.py::test_missing', ['test_public.py'], tmp_path
        )
    with pytest.raises(ValueError, match='group test'):
        validate_assertion_node(
            'other.py::test_read', ['test_public.py'], tmp_path
        )


def test_assertion_helper_is_not_a_collected_test(tmp_path):
    from ledger import validate_assertion_node

    (tmp_path / 'test_public.py').write_text('def helper(): pass\n')
    with pytest.raises(ValueError, match='collected test'):
        validate_assertion_node(
            'test_public.py::helper', ['test_public.py'], tmp_path
        )


def test_retired_comparison_annotations_are_not_current_support_promises():
    before = {
        'scenarios': [
            {
                'id': 'predict',
                'models': ['relational', 'tabular'],
                'expected': {'status': 'supported'},
                'baseline': {'commit': 'retired-client'},
                'differences': ['retired-comparison-waiver'],
            }
        ],
    }
    after = copy.deepcopy(before)
    after['scenarios'][0].pop('baseline')
    after['scenarios'][0].pop('differences')
    validate_changes(before, after, 'base')
    # Normalize both sides; legacy metadata never grants an approval either.
    validate_changes(after, before, 'base')
    for mutation in ('models', 'expected', 'deleted'):
        changed = copy.deepcopy(after)
        if mutation == 'models':
            changed['scenarios'][0]['models'] = ['relational']
        elif mutation == 'expected':
            changed['scenarios'][0]['expected'] = {'status': 'refused'}
        else:
            changed['scenarios'] = []
        with pytest.raises(ValueError, match='scenarios/predict'):
            validate_changes(before, changed, 'base')
