# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paid native observations remain independently checked after failure."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_t136_forecast_capture import RUNNER, _ask, _module
from test_t44_prediction_observation import (
    actual_observation,
    legacy_observation,
    reader,
)

ROOT = Path(__file__).parents[2]
COMPARE = _module('t44_partial_compare', ROOT / 'scripts/rust_core/compare.py')
CHECK = _module(
    't44_partial_conservation', ROOT / 'contract/runners/capture_check.py'
)


def partial_observation():
    core, operations = actual_observation()
    entry = core['predictions'][0]
    failed = copy.deepcopy(entry['calls'][0])
    failed.update(run=5, request_id='sdm-second-failed', scope=None)
    failed['body']['metadata']['request_id'] = failed['request_id']

    # This seed observes two actual independent JSON bodies. It does not
    # claim these hand-authored DTOs were executed by an SDK.
    # Decode the existing number-token snapshot back only for this fixture
    # producer, not in the capture implementation.
    def native(value):
        if isinstance(value, dict):
            if set(value) == {'number'}:
                return int(value['number'])
            return {key: native(item) for key, item in value.items()}
        if isinstance(value, list):
            return [native(item) for item in value]
        return value

    raw = json.dumps(native(failed['body'])).encode()
    failed['input_body_base64'] = base64.b64encode(raw).decode('ascii')
    failed['input_body_sha256'] = failed['body_sha256'] = hashlib.sha256(
        raw
    ).hexdigest()
    failed['body'] = json.loads(
        raw,
        parse_int=lambda value: {'number': value},
        parse_float=lambda value: {'number': value},
    )
    error = {'kind': 'transport', 'code': 'TRANSPORT_ERROR'}
    started = copy.deepcopy(core['inputs'][3])
    started.update(run=5, result=None, error=error)
    started['input']['input']['request_id'] = failed['request_id']
    core['inputs'].append(started)
    entry.update(
        completion='raised',
        final=None,
        failure={'run': 5, 'relation': 'selected'},
    )
    entry['runs'].append(5)
    entry['calls'].append(failed)
    core.update(predicted=None, error=error)
    operations.append('prepare_request')
    return core, operations


def wrapped(core, operations, *, runner='python-core'):
    calls = core.get('predictions', [{}])[0].get('calls', [])
    return {
        'case': 'partial-proof',
        'identity': {
            'operations': operations,
            'contract': 35,
            'runner': runner,
        },
        'outcome': {
            'result': 'refusal',
            'stage': 'predict',
            'code': 'TRANSPORT_ERROR',
            'error': 'NimRequestError',
        },
        'core': core,
        'public': {
            'payloads': [
                {'parsed': copy.deepcopy(call['body'])}
                for call in calls
                if call['body'] is not None
            ]
        },
        'driver': {
            'capture': {'incomplete': [], 'statements': [], 'bindings': []},
            'query_log': {'statements': 0, 'bindings': 0, 'differing': []},
        },
    }


def legacy_partial(actual):
    core, operations = legacy_observation()
    core.update(predicted=None, error=copy.deepcopy(actual['error']))
    entry = actual['predictions'][0]
    core['model_calls'] = [
        {
            'run': 2,
            'body': copy.deepcopy(call['body']),
            'body_sha256': call['body_sha256'],
        }
        for call in entry['calls']
    ]
    final = entry['results'][-1]
    core['answered'] = {
        'rows': final['data']['answer_details']['rows_answered'],
        'result': copy.deepcopy(final['result']),
        'model': final['data']['predicted']['model'],
        'request_id': final['data']['predicted']['request_id'],
        'metadata': copy.deepcopy(final['metadata']),
        'outputs': final['data']['predicted']['outputs'],
    }
    return core, operations


def test_failed_prediction_keeps_checked_paid_calls_and_partial_result():
    core, operations = partial_observation()
    view = reader().view(core, operations)
    assert core['predicted'] is None
    assert len(view['calls']) == 2
    assert view['partial']['observed'] is True
    assert view['partial']['answered'] == {
        'rows': 2,
        'result': core['predictions'][0]['results'][-1]['result'],
        'model': core['predictions'][0]['results'][-1]['data']['predicted'][
            'model'
        ],
        'request_id': core['predictions'][0]['results'][-1]['data'][
            'predicted'
        ]['request_id'],
        'metadata': core['predictions'][0]['results'][-1]['metadata'],
        'outputs': core['predictions'][0]['results'][-1]['data']['predicted'][
            'outputs'
        ],
    }
    assert len(view['partial']['results']) == 1
    assert (
        view['partial']['results'][0]['metadata']
        == core['predictions'][0]['results'][0]['metadata']
    )


@pytest.mark.parametrize(
    'tamper', [False, True], ids=['untampered', 'endpoint-body-changed']
)
def test_failure_body_is_bound_to_independently_received_request(tamper):
    core, operations = partial_observation()
    observation = wrapped(core, operations)
    if tamper:
        observation['public']['payloads'][0]['parsed']['model'] = (
            'not-the-core-body'
        )
    report = COMPARE.Report()
    COMPARE._core_payload(
        report, {'id': 'partial-proof'}, 'python-core', observation, None
    )
    assert bool(report.failures) is tamper
    assert bool(CHECK.complete(observation)) is tamper


@pytest.mark.parametrize(
    'tamper',
    [
        None,
        'body',
        'arrow',
        'rows',
        'metadata',
        'model',
        'endpoint-id',
        'outputs',
        'missing-calls',
        'missing-partial',
    ],
)
def test_failed_new_and_legacy_hosts_compare_actual_common_paid_observations(
    tamper,
):
    actual, operations = partial_observation()
    legacy, old_operations = legacy_partial(actual)
    if tamper == 'body':
        legacy['model_calls'][0]['body']['model'] = 'different-core-body'
    elif tamper == 'arrow':
        legacy['answered']['result']['rows'][0][2] = {
            'f64_bits': '0000000000000000'
        }
    elif tamper == 'rows':
        legacy['answered']['rows'] += 1
    elif tamper == 'metadata':
        legacy['answered']['metadata']['endpoint'] = 'changed native metadata'
    elif tamper == 'model':
        legacy['answered']['model'] = 'changed native model'
    elif tamper == 'endpoint-id':
        legacy['answered']['request_id'] = 'changed endpoint id'
    elif tamper == 'outputs':
        legacy['answered']['outputs'] = []
    elif tamper == 'missing-calls':
        del legacy['model_calls']
    elif tamper == 'missing-partial':
        del legacy['answered']
    report = COMPARE.Report()
    COMPARE._core_agree(
        report,
        {'id': 'partial-proof'},
        wrapped(actual, operations),
        wrapped(legacy, old_operations, runner='rust-host'),
    )
    assert bool(report.failures) is (tamper is not None)


@pytest.mark.parametrize(
    'changed', ['metadata', 'model', 'endpoint-id', 'arrow']
)
def test_two_new_hosts_compare_genuinely_observed_partial_products(changed):
    left, operations = partial_observation()
    right = copy.deepcopy(left)
    final = right['predictions'][0]['results'][-1]
    if changed == 'metadata':
        final['metadata']['endpoint'] = 'changed partial metadata'
    elif changed == 'model':
        final['data']['predicted']['model'] = 'changed partial model'
        right['inputs'][4]['result'] = copy.deepcopy(final['data'])
    elif changed == 'endpoint-id':
        final['data']['predicted']['request_id'] = 'changed partial endpoint-id'
        right['inputs'][4]['result'] = copy.deepcopy(final['data'])
    else:
        final['result']['rows'][0][2] = {'f64_bits': '0000000000000000'}
    report = COMPARE.Report()
    COMPARE._core_agree(
        report,
        {'id': 'partial-proof'},
        wrapped(left, operations),
        wrapped(right, operations),
    )
    assert report.failures


def test_actual_paid_forecast_failure_exposes_native_partial_arrow(monkeypatch):
    observed = _ask(monkeypatch, (), capture=True, fail_call=1)
    assert observed['error'] is not None
    capture = observed['capture']
    view = reader().view(capture.core(), capture.operations)
    assert len(view['calls']) == 2
    assert view['partial']['observed'] is True
    assert view['partial']['answered'] is not None
    assert view['partial']['answered']['rows'] == 2
    assert len(view['partial']['answered']['result']['rows']) == 2
    assert (
        view['partial']['results'][-1]['result']
        == view['partial']['answered']['result']
    )


def test_legacy_capture_retains_failed_call_journal_without_predict_done():
    core, _operations = legacy_partial(partial_observation()[0])
    capture = RUNNER._Capture()
    state = core['inputs'][-1]['state']
    capture.started(
        json.dumps(
            {
                'operation': {
                    'op': 'predict',
                    'model': 'kumo-relational',
                    'request_id': 'a',
                    'state': {
                        'version': 35,
                        'query': 1,
                        'payload': 2,
                        'batches': [{'request_id': 'b', 'payload': 3}],
                    },
                }
            }
        ).encode(),
        {
            1: json.dumps(state['query']).encode(),
            2: b'{"predict":{"instance_table":{"rows":[[1]]}}}',
            3: b'{"predict":{"instance_table":{"rows":[[2]]}}}',
        },
    )
    for attachment, request_id in [(10, 'a'), (11, 'b')]:
        capture.stepped({'step': 'call_model', 'body': attachment})
        capture.taken(
            attachment,
            json.dumps({'metadata': {'request_id': request_id}}).encode(),
        )
    assert capture.predicted is None
    assert [
        item['body']['metadata']['request_id']
        for item in capture.core()['model_calls']
    ] == ['a', 'b']


def test_unobserved_legacy_failure_is_distinct_from_observed_empty():
    core, operations = legacy_observation()
    core.update(
        predicted=None, error={'kind': 'transport', 'code': 'TRANSPORT_ERROR'}
    )
    unknown = reader().view(core, operations)
    assert unknown['calls'] is None
    assert unknown['partial'] == {
        'observed': False,
        'answered': None,
        'results': None,
    }
    core.update(model_calls=[], answered=None)
    empty = reader().view(core, operations)
    assert empty['calls'] == []
    assert empty['partial'] == {
        'observed': True,
        'answered': None,
        'results': None,
    }


@pytest.mark.parametrize('bad', ['wrong-run', 'negative-count', 'extra-field'])
def test_legacy_partial_products_remain_closed_native_evidence(bad):
    core, operations = legacy_partial(partial_observation()[0])
    if bad == 'wrong-run':
        core['model_calls'][0]['run'] = 0
    elif bad == 'negative-count':
        core['answered']['rows'] = -1
    else:
        core['answered']['next_operation'] = 'not-a-product'
    with pytest.raises(reader().ObservationError):
        reader().view(core, operations)
