# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual materialization preparation events retain their own Run owners."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from test_t136_forecast_capture import RUNNER


def _bytes(attachment, raw):
    return {
        'attachment': attachment,
        'base64': base64.b64encode(raw).decode('ascii'),
        'sha256': hashlib.sha256(raw).hexdigest(),
        'parsed': json.loads(raw),
    }


def _start(capture, raw=b'{ "stage" : "population", "source" : {} }'):
    operation = {'op': 'prepare_materialize', 'input': 1}
    opaque = object()
    attachments = {1: raw, 9: opaque}
    before = attachments.copy()
    capture.started(json.dumps({'operation': operation}).encode(), attachments)
    assert attachments == before
    assert attachments[9] is opaque
    return capture._run(None)


def _records(capture):
    return capture.core()['materialization_preparations']


def _validate(core):
    document = json.loads(
        (Path(__file__).parents[1] / 'observation.schema.json').read_text()
    )
    document['required'] = ['core']
    Draft202012Validator(document).validate({'core': core})


def test_raw_start_keeps_actual_envelope_bytes_and_ordinal_without_a_role():
    capture = RUNNER._Capture()
    raw = b'{ "stage" : "population", "source" : {} }'
    owner = _start(capture, raw)
    assert _records(capture) == [
        {
            'run': 0,
            'operation': {'op': 'prepare_materialize', 'input': 1},
            'input': _bytes(1, raw),
            'completion': 'active',
            'result': None,
            'value_reads': [],
            'error': None,
            'events': [],
        }
    ]
    assert capture.operations == ['prepare_materialize']
    assert capture.inputs == []
    assert capture.materialized is None
    assert owner.completion == 'unobserved'
    assert capture.core()['materialization_capture_capabilities'] == {
        'cancellation': False,
    }
    assert capture.core()['capture_capabilities'] == {
        'model_replies': True,
        'native_error_control': True,
    }
    _validate(capture.core())


@pytest.mark.parametrize(
    'kind',
    [
        'source',
        'population',
        'column_meaning',
        'label_groups',
        'label_batch',
        'request',
    ],
)
def test_done_is_independent_of_actual_value_reads_and_unknown_opaque_products(
    kind,
):
    capture = RUNNER._Capture()
    owner = _start(capture)
    result = {'op': 'prepare_materialize', 'value': 2}
    data = (
        {'state': {'version': 2, 'payload': 3, 'query': 4}}
        if (kind == 'request')
        else {'owner': 3}
    )
    raw = json.dumps({'value': kind, 'data': data}, indent=2).encode()
    capture.done(result, run=owner)
    record = _records(capture)[0]
    assert record['completion'] == 'done'
    assert record['result'] == result
    assert record['value_reads'] == []
    capture.taken(9, raw, run=owner)
    capture.taken(2, object(), run=owner)
    assert record['value_reads'] == []
    capture.taken(2, raw, run=owner)
    assert record['value_reads'] == [_bytes(2, raw)]
    assert record['events'] == [
        {'event': 'done'},
        {'event': 'value_taken', 'attachment': 2},
    ]
    assert capture.materialized is None
    assert capture.inputs == []
    # Observing repeated returned bytes preserves the actual event sequence.
    capture.taken(2, raw, run=owner)
    assert record['value_reads'] == [_bytes(2, raw), _bytes(2, raw)]
    assert record['events'][-1] == {'event': 'value_taken', 'attachment': 2}
    _validate(capture.core())


def test_reused_attachment_ids_and_interleaved_starts_keep_exact_run_ownership():
    capture = RUNNER._Capture()
    first = _start(capture)
    second = _start(capture)
    result = {'op': 'prepare_materialize', 'value': 2}
    capture.done(result, run=first)
    capture.done(result, run=second)
    one = b'{"value":"source","data":{"identity":"one"}}'
    two = b'{"value":"source","data":{"identity":"two"}}'
    capture.taken(2, two, run=second)
    capture.taken(2, one, run=first)
    records = _records(capture)
    assert [record['run'] for record in records] == [0, 1]
    assert records[0]['value_reads'] == [_bytes(2, one)]
    assert records[1]['value_reads'] == [_bytes(2, two)]
    assert capture.operations == ['prepare_materialize', 'prepare_materialize']
    _validate(capture.core())


class _CoreError(Exception):
    kind = 'data_source'
    code = 'PROVIDER_ERROR'

    def __str__(self):
        raise AssertionError('capture read a private error message')


def test_only_an_actual_core_error_gets_a_core_error_record():
    capture = RUNNER._Capture()
    owner = _start(capture)
    capture.failed(_CoreError(), run=owner)
    [record] = _records(capture)
    assert record['completion'] == 'error'
    assert record['error'] == {'kind': 'data_source', 'code': 'PROVIDER_ERROR'}
    assert record['result'] is None
    assert record['value_reads'] == []
    assert record['events'] == [{'event': 'core_error'}]
    _validate(capture.core())


def test_private_source_failure_operand_is_not_an_exported_error_summary():
    capture = RUNNER._Capture()
    raw = json.dumps(
        {
            'stage': 'source_failure',
            'source': {'owner': 9},
            'read': {
                'kind': 'neighbors',
                'source': ['catalog', 'schema', 'buys'],
                'columns': ['buy_id', 'item_id'],
            },
            'error': {
                'kind': 'data_source',
                'retryable': False,
                'sqlstate': '08006',
                'driver_message': 'private driver diagnostic',
            },
        },
        indent=2,
    ).encode()
    owner = _start(capture, raw)
    capture.failed(_CoreError(), run=owner)
    core = capture.core()
    [record] = core['materialization_preparations']
    assert record['input'] == _bytes(1, raw)
    assert (
        record['error']
        == core['error']
        == {
            'kind': 'data_source',
            'code': 'PROVIDER_ERROR',
        }
    )
    _validate(core)


class _UnreadFailure(Exception):
    def __str__(self):
        raise AssertionError('capture formatted an arbitrary failure')

    def __getattribute__(self, name):
        if name in ('kind', 'code', 'control', 'attachments'):
            raise AssertionError('capture classified an arbitrary failure')
        return super().__getattribute__(name)


@pytest.mark.parametrize(
    'failure',
    [
        None,
        _UnreadFailure(),
        KeyboardInterrupt(),
        SystemExit(3),
    ],
)
def test_actual_cancel_is_transparent_and_records_its_own_run(failure):
    capture = RUNNER._Capture(native_cancellations_observed=True)
    first = _start(capture)
    _start(capture)
    calls = []
    token = object()

    def cancel():
        calls.append('cancel')
        if failure is not None:
            raise failure
        return token

    handle = RUNNER._Handle(
        SimpleNamespace(cancel=cancel),
        SimpleNamespace(CoreError=_CoreError),
        capture,
        first,
    )
    if failure is None:
        assert handle.cancel() is token
    else:
        with pytest.raises(type(failure)) as caught:
            handle.cancel()
        assert caught.value is failure
    assert calls == ['cancel']
    records = _records(capture)
    event = 'cancelled' if failure is None else 'cancel_failed'
    assert records[0]['completion'] == event
    assert records[0]['events'] == [{'event': event}]
    assert records[1]['completion'] == 'active'
    assert all(record['error'] is None for record in records)
    assert all(record['result'] is None for record in records)
    assert capture.core()['error'] is None
    assert capture.core()['materialization_capture_capabilities'] == {
        'cancellation': True,
    }
    _validate(capture.core())


def test_completion_and_cancel_events_preserve_actual_order_without_filling_values():
    capture = RUNNER._Capture(native_cancellations_observed=True)
    owner = _start(capture)
    capture.done({'op': 'prepare_materialize', 'value': 2}, run=owner)
    handle = RUNNER._Handle(
        SimpleNamespace(cancel=lambda: None),
        object(),
        capture,
        owner,
    )
    handle.cancel()
    [record] = _records(capture)
    assert record['events'] == [{'event': 'done'}, {'event': 'cancelled'}]
    assert record['completion'] == 'cancelled'
    assert record['result'] == {'op': 'prepare_materialize', 'value': 2}
    assert record['value_reads'] == []
    _validate(capture.core())


@pytest.mark.parametrize(
    'path',
    [
        ('materialization_preparations', 0, 'extra'),
        ('materialization_preparations', 0, 'operation', 'extra'),
        ('materialization_preparations', 0, 'input', 'extra'),
        ('materialization_preparations', 0, 'result', 'extra'),
        ('materialization_preparations', 0, 'value_reads', 0, 'extra'),
        ('materialization_preparations', 0, 'events', 0, 'extra'),
        ('materialization_capture_capabilities', 'extra'),
    ],
)
def test_raw_journal_schema_closes_every_observation_wrapper(path):
    capture = RUNNER._Capture()
    owner = _start(capture)
    capture.done({'op': 'prepare_materialize', 'value': 2}, run=owner)
    capture.taken(2, b'{"value":"source","data":{}}', run=owner)
    core = copy.deepcopy(capture.core())
    _validate(core)
    parent = core
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = True
    with pytest.raises(ValidationError):
        _validate(core)


def test_legacy_and_sparse_captures_do_not_claim_new_observations():
    capture = RUNNER._Capture(
        model_replies_observed=False,
        native_error_control_observed=False,
    )
    core = capture.core()
    assert 'materialization_preparations' not in core
    assert 'materialization_capture_capabilities' not in core
    _start(capture)
    core = capture.core()
    assert core['materialization_capture_capabilities'] == {
        'cancellation': False
    }
    assert core['capture_capabilities'] == {
        'model_replies': False,
        'native_error_control': False,
    }
    _validate(core)


def test_full_boundary_and_its_reset_advertise_the_installed_cancel_hook():
    boundary = RUNNER._Boundary()
    try:
        assert boundary.capture.native_cancellations_observed is True
        assert boundary.reset().native_cancellations_observed is True
    finally:
        boundary.close()
