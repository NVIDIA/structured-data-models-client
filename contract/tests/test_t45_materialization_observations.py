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
    with pytest.raises(RUNNER.RunnerFailure):
        RUNNER._replay_core(capture)
