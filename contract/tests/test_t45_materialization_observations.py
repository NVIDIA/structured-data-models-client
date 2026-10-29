# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual SQL entry ownership is required before a common materialize role.

Synthetic records below are independently authored observer DTO fixtures.
They are neither frozen captures nor a claim that an SDK executed them.
"""

from __future__ import annotations

import copy
import json

import pyarrow as pa
import pytest
from jsonschema.exceptions import ValidationError
from test_t44_materialization_raw_journal import _bytes, _validate
from test_t44_prediction_observation import reader
from test_t136_forecast_capture import RUNNER


def _task():
    return {
        'stage': 'task',
        'request': {
            'provider': {'provider': 'sql', 'dialect': 'duckdb'},
            'schema': 1,
            'query': 2,
            'indices': 3,
            'values': 5,
            'seed': None,
            'anchor_time': None,
            'anchor_entity': False,
            'options': [],
            'most_parts': 10,
        },
    }


def _terminal():
    return {
        'value': 'request',
        'data': {
            'state': {'version': 2, 'payload': 11, 'query': 12},
            'query': None,
        },
    }


def _input_reads():
    return [
        _bytes(1, b'{"tables":[],"links":[]}'),
        _bytes(2, b'"PREDICT events.value FOR EACH events.id"'),
        {'attachment': 3, 'scalars': [{'t': 'i64', 'v': '7'}]},
        _bytes(5, b'{"num_context_samples":2}'),
    ]


def actual_materialization():
    """One actual Task/result/late state-read shape, with no inferred owners."""
    return {
        'inputs': [],
        'fetches': [],
        'accepted': None,
        'described': None,
        'materialized': None,
        'predicted': None,
        'error': None,
        'materialization_capture_capabilities': {'cancellation': True},
        'materialization_entries': [
            {
                'ordinal': 0,
                'parent': None,
                'prediction_parent': None,
                'runs': [0],
                'completion': 'returned',
                'terminal': {'run': 0},
                'failure': None,
            }
        ],
        'materialization_guards': [],
        'materialization_preparations': [
            {
                'run': 0,
                'entry': 0,
                'operation': {'op': 'prepare_materialize', 'input': 9},
                'input': _bytes(9, json.dumps(_task()).encode()),
                'input_reads': _input_reads(),
                'owner_binds': [],
                'owner_takes': [],
                'output_reads': [
                    _bytes(11, b'{"context":{},"predict":{}}'),
                    _bytes(12, b'"resolved query"'),
                ],
                'origin': None,
                'completion': 'done',
                'result': {'op': 'prepare_materialize', 'value': 10},
                'value_reads': [_bytes(10, json.dumps(_terminal()).encode())],
                'fetches': [],
                'error': None,
                'events': [
                    {'event': 'done'},
                    {'event': 'value_taken', 'attachment': 10},
                ],
            }
        ],
    }


def test_complete_actual_task_and_late_state_reads_supply_only_a_comparison_role():
    core = actual_materialization()
    before = copy.deepcopy(core)
    observed = reader().view(core, ['prepare_materialize'])
    assert observed['operations'] == ['materialize']
    assert observed['inputs'] == [
        {
            'op': 'materialize',
            'provider': _task()['request']['provider'],
            'schema': {'tables': [], 'links': []},
            'query': 'PREDICT events.value FOR EACH events.id',
            'indices': [{'t': 'i64', 'v': '7'}],
            'anchor_time': None,
            'seed': None,
        }
    ]
    assert observed['input_positions'] == [0]
    assert observed['materialized'] == {
        'version': 2,
        'query': 'resolved query',
        'payload': {'context': {}, 'predict': {}},
    }
    assert observed['materializations'][0]['native'] is True
    assert observed['error'] is None
    assert core == before
    assert core['materialized'] is None
    assert core['inputs'] == []
    _validate(core)


@pytest.mark.parametrize(
    'change',
    [
        'no_terminal',
        'wrong_run',
        'entry_owner',
        'missing_task_schema',
        'missing_indices',
        'missing_payload',
        'missing_query',
        'repeated_value',
        'result_and_error',
        'value_digest',
        'output_digest',
        'outside_entry',
    ],
)
def test_incomplete_or_contradictory_materialization_evidence_never_earns_a_role(
    change,
):
    core = actual_materialization()
    entry = core['materialization_entries'][0]
    record = core['materialization_preparations'][0]
    if change == 'no_terminal':
        entry['terminal'] = None
    elif change == 'wrong_run':
        entry['terminal']['run'] = 1
    elif change == 'entry_owner':
        record['entry'] = 1
    elif change == 'missing_task_schema':
        record['input_reads'].pop(0)
    elif change == 'missing_indices':
        record['input_reads'].pop(2)
    elif change == 'missing_payload':
        record['output_reads'].pop(0)
    elif change == 'missing_query':
        record['output_reads'].pop()
    elif change == 'repeated_value':
        record['value_reads'].append(copy.deepcopy(record['value_reads'][0]))
    elif change == 'result_and_error':
        record['error'] = {'kind': 'data_source', 'code': 'PROVIDER_ERROR'}
    elif change == 'value_digest':
        record['value_reads'][0]['sha256'] = '0' * 64
    elif change == 'output_digest':
        record['output_reads'][0]['sha256'] = '0' * 64
    else:
        entry['runs'] = []
    with pytest.raises(reader().ObservationError):
        reader().view(core, ['prepare_materialize'])


def test_actual_late_takes_remain_with_the_returned_run_after_entry_cleanup():
    capture = RUNNER._Capture(native_cancellations_observed=True)
    entry = capture.enter_materialization()
    task = _task()
    attachments = {
        1: b'{"tables":[],"links":[]}',
        2: b'"query"',
        3: pa.table({'indices': pa.array([7], type=pa.int64())}),
        5: b'{}',
        9: json.dumps(task).encode(),
    }
    capture.started(
        json.dumps(
            {'operation': {'op': 'prepare_materialize', 'input': 9}}
        ).encode(),
        attachments,
    )
    owner = capture._run(None)
    capture.done({'op': 'prepare_materialize', 'value': 10}, run=owner)
    capture.taken(10, json.dumps(_terminal()).encode(), run=owner)
    capture.leave_materialization(entry, None, terminal=owner)
    with pytest.raises(reader().ObservationError):
        reader().view(capture.core(), capture.operations)
    capture.taken(11, b'{"context":{},"predict":{}}', run=owner)
    capture.taken(12, b'"resolved query"', run=owner)
    observed = reader().view(capture.core(), capture.operations)
    assert observed['materialized']['query'] == 'resolved query'
    assert capture.core()['materialized'] is None
    assert (
        capture.core()['materialization_entries'][0]['completion'] == 'returned'
    )
    assert capture._materialization_stack == []
    assert capture._materialization_owners == {}
    _validate(capture.core())


@pytest.mark.parametrize(
    'change', ['extra_entry', 'bad_origin', 'unknown_bind', 'bad_scalar_tail']
)
def test_closed_journal_schema_rejects_malformed_extensions(change):
    core = actual_materialization()
    record = core['materialization_preparations'][0]
    if change == 'extra_entry':
        core['materialization_entries'][0]['callback'] = 'run'
    elif change == 'bad_origin':
        record['origin'] = {'run': 0, 'first': 1}
    elif change == 'unknown_bind':
        record['owner_binds'] = [
            {
                'attachment': 4,
                'producer': {'run': 0, 'attachment': 4, 'origin': 'same_table'},
            }
        ]
    else:
        record['input_reads'][2]['scalars'].extend(
            [{'t': 'i64', 'v': '7'}] * 12000 + [False]
        )
    with pytest.raises(ValidationError):
        _validate(core)


def test_new_journals_preserve_existing_long_list_admission_proof():
    schema = json.loads(
        (RUNNER.HERE.parent / 'observation.schema.json').read_text()
    )
    RUNNER._long_lists_held(schema)


def test_cancelled_original_read_cannot_be_written_as_a_completed_legacy_replay():
    capture = RUNNER._Capture(native_cancellations_observed=True)
    capture.inputs = [
        {
            'op': 'describe_source',
            'provider': {},
            'source': {},
            'check_keys': False,
        }
    ]
    # This independently authored cancellation carries no reply. It cannot
    # stand in for a provider-error reply to an original materialize Run.
    capture.operations = ['prepare_materialize']
    capture.materialization_preparations = actual_materialization()[
        'materialization_preparations'
    ]
    record = capture.materialization_preparations[0]
    record['completion'] = 'cancelled'
    record['result'] = None
    record['value_reads'] = []
    record['output_reads'] = []
    record['events'] = [{'event': 'cancelled'}]
    with pytest.raises(RUNNER.RunnerFailure, match='cancelled|unanswered'):
        RUNNER._replay_core(capture)


def _started(capture, value, attachments=None):
    capture.started(
        json.dumps(
            {'operation': {'op': 'prepare_materialize', 'input': 9}}
        ).encode(),
        {9: json.dumps(value).encode(), **(attachments or {})},
    )
    return capture._run(None)


def _returned(capture, owner, kind, data, opaque=None):
    capture.done({'op': 'prepare_materialize', 'value': 10}, run=owner)
    capture.taken(
        10, json.dumps({'value': kind, 'data': data}).encode(), run=owner
    )
    if opaque is not None:
        capture.taken(data['owner'], opaque, run=owner)


def _owned_chain(capture, *, foreign=False):
    entry = capture.enter_materialization()
    task = _started(
        capture,
        _task(),
        {
            1: b'{"tables":[],"links":[]}',
            2: b'"query"',
            3: pa.table({'indices': pa.array([7], type=pa.int64())}),
            5: b'{}',
        },
    )
    source = {
        'source': 'source',
        'identity': 'source identity',
        'owner': 4,
        'target': {'source': {'source': ['events']}, 'column': 'value'},
        'tables': [
            {
                'source': {'source': ['events']},
                'read': {'source': ['events']},
                'columns': ['value'],
            }
        ],
    }
    source_owner, population_owner = object(), object()
    _returned(capture, task, 'source', source, source_owner)
    rebound = {**source, 'owner': 2}
    population = _started(
        capture,
        {'stage': 'population', 'source': rebound},
        {2: object() if foreign else source_owner},
    )
    population_data = {
        'source': 'source',
        'identity': 'population identity',
        'owner': 8,
        'rows': 0,
        'batch_rows': 10,
        'target': source['target'],
    }
    _returned(
        capture, population, 'population', population_data, population_owner
    )
    terminal = _started(
        capture,
        {
            'stage': 'request',
            'population': {**population_data, 'owner': 3},
            'batches': [],
            'values': None,
        },
        {3: population_owner},
    )
    _returned(capture, terminal, 'request', _terminal()['data'])
    capture.leave_materialization(entry, None, terminal=terminal)
    capture.taken(11, b'{"context":{},"predict":{}}', run=terminal)
    capture.taken(12, b'"resolved query"', run=terminal)
    return entry, task, population, terminal


def test_cross_run_proof_uses_actual_opaque_takes_and_rebinds_with_reused_local_ids():
    capture = RUNNER._Capture(native_cancellations_observed=True)
    _, task, population, terminal = _owned_chain(capture)
    records = capture.core()['materialization_preparations']
    assert records[0]['owner_takes'] == [{'attachment': 4, 'origin': 'result'}]
    assert records[1]['owner_binds'] == [
        {
            'attachment': 2,
            'producer': {
                'run': task.ordinal,
                'attachment': 4,
                'origin': 'result',
            },
        }
    ]
    assert records[2]['owner_binds'] == [
        {
            'attachment': 3,
            'producer': {
                'run': population.ordinal,
                'attachment': 8,
                'origin': 'result',
            },
        }
    ]
    assert reader().view(capture.core(), capture.operations)['operations'] == [
        'materialize'
    ]
    assert capture._materialization_owners == {}
    _validate(capture.core())


def test_same_json_and_local_id_without_actual_owner_identity_cannot_prove_a_chain():
    capture = RUNNER._Capture(native_cancellations_observed=True)
    _owned_chain(capture, foreign=True)
    assert (
        capture.core()['materialization_preparations'][1]['owner_binds'] == []
    )
    with pytest.raises(reader().ObservationError):
        reader().view(capture.core(), capture.operations)


def test_actual_owner_producer_kind_and_description_are_checked_before_a_common_role():
    for change in ('kind', 'identity'):
        capture = RUNNER._Capture(native_cancellations_observed=True)
        _owned_chain(capture)
        core = copy.deepcopy(capture.core())
        terminal = core['materialization_preparations'][2]
        if change == 'kind':
            terminal['owner_binds'][0]['producer'].update(run=0, attachment=4)
        else:
            changed = terminal['input']['parsed']
            changed['population']['identity'] = 'unobserved replacement'
            terminal['input'] = _bytes(9, json.dumps(changed).encode())
        with pytest.raises(reader().ObservationError):
            reader().view(core, capture.operations)


@pytest.mark.parametrize(
    'change', ['missing_done', 'value_before_done', 'duplicate_core_error']
)
def test_materialization_lifecycle_facts_must_match_the_actual_product(change):
    core = actual_materialization()
    record = core['materialization_preparations'][0]
    if change == 'missing_done':
        record['events'] = [record['events'][1]]
    elif change == 'value_before_done':
        record['events'].reverse()
    else:
        record.update(
            completion='error',
            result=None,
            value_reads=[],
            output_reads=[],
            error={'kind': 'data_source', 'code': 'PROVIDER_ERROR'},
            events=[{'event': 'core_error'}, {'event': 'core_error'}],
        )
        core['error'] = record['error']
        core['materialization_entries'][0].update(
            completion='raised',
            terminal=None,
            failure={'run': 0, 'relation': 'selected'},
        )
    with pytest.raises(reader().ObservationError):
        reader().view(core, ['prepare_materialize'])


def test_separate_entries_with_reused_ids_keep_distinct_resource_producers():
    capture = RUNNER._Capture(native_cancellations_observed=True)
    _owned_chain(capture)
    _, second_task, _, _ = _owned_chain(capture)
    second_population = capture.core()['materialization_preparations'][4]
    assert (
        second_population['owner_binds'][0]['producer']['run']
        == second_task.ordinal
    )
    assert second_task.ordinal == 3
    observed = reader().view(capture.core(), capture.operations)
    assert observed['operations'] == ['materialize', 'materialize']
    assert len(observed['materializations']) == 2
    assert capture._materialization_owners == {}


def test_actual_pass_through_guard_has_native_selection_without_a_core_error():
    core = actual_materialization()
    entry = core['materialization_entries'][0]
    entry.update(
        runs=[0, 1],
        completion='raised',
        terminal=None,
        failure={'run': 1, 'relation': 'native'},
    )
    source = {
        'source': 'source',
        'identity': 'source identity',
        'owner': 4,
        'target': {'source': {'source': ['events']}, 'column': 'value'},
        'tables': [
            {
                'source': {'source': ['events']},
                'read': {'source': ['events']},
                'columns': ['value'],
            }
        ],
    }
    core['materialization_preparations'][0].update(
        value_reads=[
            _bytes(10, json.dumps({'value': 'source', 'data': source}).encode())
        ],
        owner_takes=[{'attachment': 4, 'origin': 'result'}],
        output_reads=[],
    )
    facts = {
        'builtin': {'ancestry': []},
        'key_error': False,
        'structured_data_error': True,
        'relational_error': False,
        'deepest_module': None,
        'message_casing': {'mapping': 'full_root', 'unicode_version': '14.0.0'},
    }
    guard_input = {
        'stage': 'assess_failure',
        'origin': 'engine',
        'failure': facts,
        'signature': None,
        'window_outran': None,
    }
    assessment = {
        'origin': 'engine',
        'failure': facts,
        'signature': None,
        'category': 'pass_through',
        'part': None,
    }
    core['materialization_guards'] = [
        {
            'run': 1,
            'entry': 0,
            'operation': {'op': 'prepare_frames', 'input': 1},
            'input': _bytes(1, json.dumps(guard_input).encode()),
            'completion': 'done',
            'result': {'op': 'prepare_frames', 'value': 20},
            'value_reads': [
                _bytes(
                    20,
                    json.dumps(
                        {'value': 'failure_assessment', 'data': assessment}
                    ).encode(),
                )
            ],
            'error': None,
            'events': [
                {'event': 'done'},
                {'event': 'value_taken', 'attachment': 20},
            ],
            'origin': None,
        }
    ]
    observed = reader().view(core, ['prepare_materialize', 'prepare_frames'])
    assert observed['operations'] == ['materialize']
    assert observed['materializations'][0]['failure'] == {
        'run': 1,
        'relation': 'native',
    }
    assert observed['materializations'][0]['native'] is True
    assert observed['materialized'] is None
    assert observed['error'] is None
    _validate(core)
    core['materialization_guards'][0]['value_reads'][0] = _bytes(
        20,
        json.dumps(
            {
                'value': 'failure_assessment',
                'data': {**assessment, 'category': 'caller_data'},
            }
        ).encode(),
    )
    with pytest.raises(reader().ObservationError):
        reader().view(core, ['prepare_materialize', 'prepare_frames'])
