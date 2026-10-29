# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared prediction products belong to checked logical operation positions."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import warnings
from pathlib import Path

import pytest
from test_t44_prediction_observation import (
    actual_observation,
    legacy_observation,
    reader,
    two_observed_predictions,
)
from test_t136_forecast_capture import RUNNER, SERVE, _module

ROOT = Path(__file__).parents[2]
COMPARE = _module('t44_prefix_compare', ROOT / 'scripts/rust_core/compare.py')


def observation(core, operations):
    return {
        'identity': {'operations': operations, 'contract': 35},
        'core': core,
    }


def compare_prefix(short, long):
    report = COMPARE.Report(paths=(('metadata', 'request_id'),))
    COMPARE._core_agree(
        report, {'id': 't44-prediction-prefix'}, short, long, stop='predict'
    )
    return report.failures


def replace_body(call, raw):
    encoded = json.dumps(raw).encode()
    call['input_body_base64'] = base64.b64encode(encoded).decode('ascii')
    call['input_body_sha256'] = hashlib.sha256(encoded).hexdigest()
    call['body'] = RUNNER._payload(encoded)['parsed']
    call['body_sha256'] = hashlib.sha256(encoded).hexdigest()


def mutate_product(core, ordinal, changed):
    """Change one owned event consistently, without touching another event."""
    entry = core['predictions'][ordinal]
    call = entry['calls'][0]
    result = entry['results'][0]
    if changed in ('body', 'call-order', 'outputs'):
        raw = json.loads(base64.b64decode(call['input_body_base64']))
        if changed == 'body':
            raw['metadata']['kept'] = 'changed shared call'
        elif changed == 'call-order':
            raw['predict']['instance_table']['rows'].reverse()
            ids = [row[1] for row in result['result']['rows']][::-1]
            for row, value in zip(result['result']['rows'], ids, strict=True):
                row[1] = value
        else:
            raw['output']['fields'] = ['prediction', 'explanation']
        replace_body(call, raw)
    if changed == 'arrow':
        result['result']['rows'][0][2] = {'f64_bits': '4045000000000000'}
    elif changed == 'model':
        call['scope']['model'] = 'another-endpoint-model'
        result['data']['predicted']['model'] = 'another-endpoint-model'
    elif changed == 'endpoint-id':
        call['scope']['request_id'] = 'another-endpoint-response'
        result['data']['predicted']['request_id'] = 'another-endpoint-response'
    elif changed == 'metadata':
        result['metadata']['endpoint'] = 'changed shared result metadata'
    elif changed == 'outputs':
        result['data']['predicted']['outputs'] = ['prediction', 'explanation']
        result['input']['input']['bound']['named'] = True
        result['input']['input']['bound']['outputs'] = [
            'prediction',
            'explanation',
        ]
    # The independently recorded native result and its later finish input
    # retain the same scope. A valid change must survive all local bindings
    # before it can test the common-prefix comparator itself.
    result['input']['input']['aggregation']['parts'][0]['data']['scope'] = (
        copy.deepcopy(call['scope'])
    )
    result['input']['input']['aggregation']['parts'][0]['attribution'][
        'scope'
    ] = copy.deepcopy(call['scope'])
    for record in core['inputs']:
        if record.get('entry') != ordinal:
            continue
        if record['input']['stage'] == 'call_relational':
            record['result']['scope'] = copy.deepcopy(call['scope'])
            if changed == 'outputs':
                record['input']['input']['call']['named'] = True
                record['input']['input']['call']['outputs'] = [
                    'prediction',
                    'explanation',
                ]
        elif record['run'] == result['run']:
            record['input'] = copy.deepcopy(result['input'])
            record['result'] = copy.deepcopy(result['data'])
    if ordinal == len(core['predictions']) - 1:
        predicted = result['data']['predicted']
        core['predicted'] = {
            'body': copy.deepcopy(call['body']),
            'model': predicted['model'],
            'request_id': predicted['request_id'],
            'metadata': copy.deepcopy(result['metadata']),
            'result': copy.deepcopy(result['result']),
        }


def test_events_use_logical_positions_and_an_identical_shared_prefix_passes():
    short_core, short_ops = actual_observation()
    long_core, long_ops = two_observed_predictions()
    assert [
        event['position']
        for event in reader().view(long_core, long_ops)['events']
    ] == [2, 3]
    assert COMPARE.shared_operations(
        observation(short_core, short_ops),
        observation(long_core, long_ops),
        'predict',
    ) == ['describe_source', 'materialize', 'predict']
    assert (
        compare_prefix(
            observation(short_core, short_ops), observation(long_core, long_ops)
        )
        == []
    )


@pytest.mark.parametrize(
    'changed',
    [
        'body',
        'call-order',
        'arrow',
        'model',
        'endpoint-id',
        'metadata',
        'outputs',
    ],
)
def test_changed_first_shared_product_is_not_hidden_by_unchanged_final_global(
    changed,
):
    short_core, short_ops = actual_observation()
    long_core, long_ops = two_observed_predictions()
    final_global = copy.deepcopy(long_core['predicted'])
    mutate_product(long_core, 0, changed)
    assert long_core['predicted'] == final_global
    reader().view(long_core, long_ops)
    assert compare_prefix(
        observation(short_core, short_ops), observation(long_core, long_ops)
    )


@pytest.mark.parametrize('changed', ['body', 'arrow', 'metadata', 'outputs'])
def test_valid_longer_only_prediction_changes_remain_outside_the_prefix(
    changed,
):
    short_core, short_ops = actual_observation()
    long_core, long_ops = two_observed_predictions()
    first = copy.deepcopy(long_core['predictions'][0])
    mutate_product(long_core, 1, changed)
    assert long_core['predictions'][0] == first
    reader().view(long_core, long_ops)
    assert (
        compare_prefix(
            observation(short_core, short_ops), observation(long_core, long_ops)
        )
        == []
    )


def two_partial_products(core, operations):
    """Two actually observed finishes, with only one selected by _carry."""
    entry = core['predictions'][0]
    for record in core['inputs']:
        if record.get('run', -1) >= 5:
            record['run'] += 1
    for other in core['predictions'][1:]:
        other['runs'] = [run + 1 for run in other['runs']]
        for call in other['calls']:
            call['run'] += 1
        for result in other['results']:
            result['run'] += 1
    second = copy.deepcopy(entry['results'][0])
    second['run'] = 5
    second['metadata']['selection'] = 'different partial result'
    second['result']['rows'][0][2] = {'f64_bits': '4045000000000000'}
    core['inputs'].insert(
        5,
        {
            'op': 'prepare_request',
            'run': 5,
            'entry': 0,
            'input': copy.deepcopy(second['input']),
            'result': copy.deepcopy(second['data']),
            'error': None,
        },
    )
    entry['runs'].append(5)
    entry['results'].append(second)
    entry.update(
        completion='raised',
        final=None,
        carried={'run': 4, 'relation': 'selected'},
    )
    if len(core['predictions']) == 1:
        core['predicted'] = None
    operations.insert(5, 'prepare_request')


def test_changed_carried_selection_is_compared_with_identical_internal_products():
    short_core, short_ops = actual_observation()
    long_core, long_ops = two_observed_predictions()
    two_partial_products(short_core, short_ops)
    two_partial_products(long_core, long_ops)
    short = observation(short_core, short_ops)
    long = observation(long_core, long_ops)
    assert compare_prefix(short, long) == []
    untouched = copy.deepcopy(long_core['predicted'])
    products = copy.deepcopy(long_core['predictions'][0]['results'])
    long_core['predictions'][0]['carried']['run'] = 5
    assert long_core['predicted'] == untouched
    assert long_core['predictions'][0]['results'] == products
    reader().view(long_core, long_ops)
    assert compare_prefix(short, long)


def test_edit_and_unrelated_preparation_do_not_shift_events_by_input_count():
    core, operations = two_observed_predictions()
    for record in core['inputs']:
        if record.get('op') == 'prepare_request':
            record['run'] += 2
    for entry in core['predictions']:
        entry['runs'] = [run + 2 for run in entry['runs']]
        for call in entry['calls']:
            call['run'] += 2
        for result in entry['results']:
            result['run'] += 2
    generic = {
        'op': 'prepare_request',
        'run': 3,
        'entry': None,
        'input': {'stage': 'model_catalogue'},
        'result': ['kumo-relational'],
        'error': None,
    }
    core['inputs'].insert(2, generic)
    operations[2:2] = ['edit_schema', 'prepare_request']
    checked = reader().view(core, operations)
    assert checked['operations'] == [
        'describe_source',
        'materialize',
        'edit_schema',
        'prepare_request',
        'predict',
        'predict',
    ]
    assert [event['position'] for event in checked['events']] == [4, 5]
    assert generic in checked['inputs']


@pytest.fixture(scope='module')
def actual_legacy_capture():
    """Run the real legacy Predict producer, then a native version refusal."""
    import duckdb
    import pandas as pd
    from sdm_client import (
        PredictiveQuery,
        Schema,
        SDMClient,
        StructuredDataError,
    )
    from sdm_client._rust import codec
    from sdm_client._rust.predict import _send
    from sdm_client._rust.run import run
    from sdm_connectors import DuckDBSource

    with pytest.MonkeyPatch.context() as monkeypatch:
        serving = _module(
            't44_prefix_serving_fixture',
            ROOT
            / 'packages/structured-data-models-client/tests/test_t313_serving_route.py',
        )
        serving._old.__wrapped__(monkeypatch)
        monkeypatch.setenv(serving._routing.ENVIRONMENT, serving.MATERIALIZE)
        with duckdb.connect() as connection, warnings.catch_warnings():
            for statement in serving._shop():
                connection.execute(statement)
            warnings.simplefilter('ignore')
            schema = Schema.from_source(DuckDBSource(connection))
            # Explanation is supported for exactly one scored entity.
            # Keep the actual public materializer and the explicit output.
            request = PredictiveQuery(serving.QUERY, schema).materialize(
                indices=[0], anchor_time=pd.Timestamp('2024-03-01')
            )
        monkeypatch.setenv(serving._routing.ENVIRONMENT, '')
        assert request._core is not None
        state = {'version': request._core.version, 'payload': 1, 'query': 2}
        attached = {
            1: codec.attachment(request.payloads[0]),
            2: codec.attachment(request.query),
        }

        def legacy_prepare(*args, **kwargs):
            raise AssertionError('legacy native producer called SDK _prepare')

        monkeypatch.setattr(SDMClient, '_prepare', legacy_prepare)
        boundary = RUNNER._Boundary()
        capture = boundary.reset()
        try:
            with (
                SERVE(explanation=True) as (url, received),
                SDMClient(url, max_retries=0) as client,
            ):
                operation = {
                    'op': 'predict',
                    'state': state,
                    'model': 'kumo-relational',
                    'request_id': 'sdm-000000000001',
                    'outputs': ['prediction', 'explanation'],
                    'default_outputs': False,
                    'inference_asked': [],
                }
                finished = run(
                    operation,
                    attached,
                    model=lambda body: _send(client, 'kumo-relational', body),
                )
                assert finished.result['op'] == 'predict'
                finished.take(finished.result['result'])
                metadata = json.loads(
                    finished.take(finished.result['metadata'])
                )
                refused = {
                    **operation,
                    'request_id': 'sdm-000000000002',
                    'state': {**state, 'version': state['version'] + 1000},
                }
                with pytest.raises(StructuredDataError) as caught:
                    run(
                        refused,
                        attached,
                        model=lambda body: _send(
                            client, 'kumo-relational', body
                        ),
                    )
                assert len(received.requests) == 1
        finally:
            boundary.close()
        assert capture.operations == ['predict', 'predict']
        core = copy.deepcopy(capture.core())
        return core, capture.operations.copy(), metadata, caught.value.code


def test_actual_legacy_runs_keep_success_and_early_error_separately(
    actual_legacy_capture,
):
    core, operations, metadata, error_code = actual_legacy_capture
    first, second = core['legacy_predictions']
    assert [first['run'], second['run']] == [0, 1]
    assert first['completion'] == 'done'
    assert first['error'] is first['answered'] is None
    assert first['result']['metadata'] == metadata
    assert first['result']['result'] is not None
    assert len(first['result']['result']['rows']) == 1
    assert first['result']['outputs'] == ['prediction', 'explanation']
    assert any(
        column['name'] == 'explanation'
        for column in first['result']['result']['columns']
    )
    assert second == {
        'run': 1,
        'completion': 'error',
        'result': None,
        'error': core['error'],
        'answered': None,
    }
    assert second['error']['code'] == error_code
    assert core['predicted'] is None
    checked = reader().view(core, operations)
    assert [event['position'] for event in checked['events']] == [0, 1]
    assert checked['events'][0]['result'] is not None
    assert checked['events'][1]['result'] is None
    assert checked['events'][1]['partial']['answered'] is None


@pytest.mark.parametrize(
    'changed',
    [
        'missing-journal',
        'duplicate-run',
        'reversed-order',
        'foreign-run',
        'wrong-operation',
    ],
)
def test_legacy_journal_cannot_borrow_another_run_or_the_latest_global(
    actual_legacy_capture, changed
):
    original, operations, _, _ = actual_legacy_capture
    core = copy.deepcopy(original)
    reader().view(core, operations)
    if changed == 'missing-journal':
        core['predicted'] = copy.deepcopy(
            core['legacy_predictions'][0]['result']
        )
        core['error'] = None
        core.pop('legacy_predictions')
    elif changed == 'duplicate-run':
        core['legacy_predictions'][1]['run'] = 0
    elif changed == 'reversed-order':
        core['legacy_predictions'].reverse()
    elif changed == 'foreign-run':
        core['legacy_predictions'][0]['run'] = 2
    else:
        operations = ['describe_source', 'predict']
    with pytest.raises(reader().ObservationError):
        reader().view(core, operations)


def test_historical_single_predict_can_use_only_its_sole_snapshot():
    core, operations = legacy_observation()
    checked = reader().view(core, operations)
    [event] = checked['events']
    assert event['position'] == 2
    assert event['result'] is not None
    assert event['partial']['answered'] is None


def test_missing_first_legacy_result_stays_unobserved_and_fails_common_prefix(
    actual_legacy_capture,
):
    original, operations, _, _ = actual_legacy_capture
    short = copy.deepcopy(original)
    short['inputs'] = short['inputs'][:1]
    short['legacy_predictions'] = short['legacy_predictions'][:1]
    short['predicted'] = copy.deepcopy(short['legacy_predictions'][0]['result'])
    short['error'] = None
    long = copy.deepcopy(original)
    assert (
        compare_prefix(
            observation(short, operations[:1]), observation(long, operations)
        )
        == []
    )
    long['legacy_predictions'][0]['result'] = None
    checked = reader().view(long, operations)
    assert checked['events'][0]['result'] is None
    assert compare_prefix(
        observation(short, operations[:1]), observation(long, operations)
    )


def test_shared_prefix_cannot_cut_a_parent_prediction_from_its_native_child():
    short_core, short_ops = actual_observation()
    long_core, long_ops = two_observed_predictions()
    # The later native work belongs to a prediction invoked while its
    # parent was still active, not a later independent public call.
    long_core['predictions'][1]['parent'] = 0
    long_core['predicted'] = copy.deepcopy(short_core['predicted'])
    checked = reader().view(long_core, long_ops)
    assert [event['position'] for event in checked['events']] == [2, 3]
    short, long = (
        observation(short_core, short_ops),
        observation(long_core, long_ops),
    )
    assert COMPARE.shared_operations(short, long, 'predict') is None
    assert compare_prefix(short, long)
