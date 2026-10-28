# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E35 applies to a proved serving request; receipts remain actual wire facts."""

from __future__ import annotations

import copy
import hashlib
import json
import warnings
from pathlib import Path

import pytest
from test_t136_forecast_capture import EN, RUNNER, SERVE, _module
from test_t44_prediction_observation import reader
from test_t44_prediction_serving_capture import CHECK

ROOT = Path(__file__).parents[2]
SERVING_PATH = '/serving-endpoints/fixture/invocations'


def envelope(request_id, *, amount='1'):
    return {
        'dataframe_records': [
            {
                'request_json': (
                    '{"metadata":{"request_id":'
                    + json.dumps(request_id)
                    + ',"kept":"value"},"amount":'
                    + amount
                    + ',"nested":{"text":"é😀","items":[true,null]}}'
                )
            }
        ]
    }


def test_serving_mask_is_only_the_existing_e35_value_rule():
    first, second = (
        envelope('independent-first'),
        envelope('independent-second'),
    )
    untouched = copy.deepcopy((first, second))
    assert first != second
    assert CHECK.masked_serving(first) == CHECK.masked_serving(second)
    inner = json.loads(
        first['dataframe_records'][0]['request_json'],
        parse_int=lambda token: {'number': token},
    )
    assert CHECK.masked_serving(first) == CHECK.masked(inner)
    assert CHECK.masked_serving(first, paths=()) != CHECK.masked_serving(
        second, paths=()
    )
    assert (first, second) == untouched


@pytest.mark.parametrize(
    'changed',
    [
        'absent-id',
        'number-id',
        'null-id',
        'object-id',
        'metadata',
        'nested-value',
        'number-token',
        'signed-zero',
    ],
)
def test_e35_keeps_id_presence_type_and_every_other_nested_value(changed):
    first = envelope(
        'first', amount='-0.0' if changed == 'signed-zero' else '1'
    )
    second = envelope(
        'second', amount='-0.0' if changed == 'signed-zero' else '1'
    )
    text = second['dataframe_records'][0]['request_json']
    if changed == 'absent-id':
        text = text.replace('"request_id":"second",', '')
    elif changed == 'number-id':
        text = text.replace('"request_id":"second"', '"request_id":1')
    elif changed == 'null-id':
        text = text.replace('"request_id":"second"', '"request_id":null')
    elif changed == 'object-id':
        text = text.replace(
            '"request_id":"second"', '"request_id":{"kept":true}'
        )
    elif changed == 'metadata':
        text = text.replace('"kept":"value"', '"kept":"changed"')
    elif changed == 'nested-value':
        text = text.replace('"items":[true,null]', '"items":[false,null]')
    elif changed == 'number-token':
        text = text.replace('"amount":1', '"amount":1.0')
    else:
        text = text.replace('"amount":-0.0', '"amount":0.0')
    second['dataframe_records'][0]['request_json'] = text
    assert CHECK.masked_serving(first) != CHECK.masked_serving(second)


@pytest.mark.parametrize('path', [SERVING_PATH, '/v1/predictions'])
def test_physical_serving_path_is_required_before_decoding_request_json(path):
    left = {
        'payloads': [
            {'method': 'POST', 'path': path, 'parsed': envelope('left')}
        ]
    }
    right = {
        'payloads': [
            {'method': 'POST', 'path': path, 'parsed': envelope('right')}
        ]
    }
    if path == SERVING_PATH:
        assert CHECK._payloads(left) == CHECK._payloads(right)
    else:
        assert CHECK._payloads(left) != CHECK._payloads(right)


def full_routes():
    from sdm_client import _routing

    return ','.join([*_routing.S0, *EN.core_routes({})])


def forbid_legacy_prepare(monkeypatch):
    from sdm_client import SDMClient

    def legacy(*args, **kwargs):
        raise AssertionError('core fixture called legacy SDMClient._prepare')

    monkeypatch.setattr(SDMClient, '_prepare', legacy)


def test_actual_serving_sdk_retry_is_one_attempt_with_two_identical_receipts(
    monkeypatch,
):
    serving = _module(
        't44_capture_serving_route_controls',
        ROOT
        / 'packages/structured-data-models-client/tests/test_t313_serving_route.py',
    )
    serving._old.__wrapped__(monkeypatch)
    request = serving._materialized(monkeypatch, unicode_feature=True)
    forbid_legacy_prepare(monkeypatch)
    busy = (
        429,
        {'error_code': 'REQUEST_LIMIT_EXCEEDED', 'message': 'slow down'},
        {'Retry-After': '0'},
    )

    def ask(captured):
        boundary = RUNNER._Boundary() if captured else None
        capture = boundary.reset() if boundary else None
        try:
            monkeypatch.setenv('SDM_CORE_ROUTES', full_routes())
            with serving._workspace(busy, serving._served) as (
                workspace,
                received,
            ):
                with serving._client(workspace) as client:
                    result = client.predict(
                        request, model=serving.KumoRelational()
                    )
            return result, received, capture
        finally:
            if boundary:
                boundary.close()

    plain, plain_received, _ = ask(False)
    result, received, capture = ask(True)
    assert result.to_pandas().equals(plain.to_pandas())
    assert result.items == plain.items
    assert result.metadata == plain.metadata
    assert len(received) == len(plain_received) == 2
    assert received[0][1] == received[1][1]
    assert plain_received[0][1] == plain_received[1][1]
    assert all(
        path == serving.INVOCATIONS
        for path, _, _ in [*received, *plain_received]
    )
    assert [
        CHECK.masked_serving(json.loads(body)) for _, body, _ in received
    ] == [
        CHECK.masked_serving(json.loads(body)) for _, body, _ in plain_received
    ]
    [entry] = capture.core()['predictions']
    assert len(entry['calls']) == 1
    checked = reader().view(capture.core(), capture.operations)
    [attempt] = checked['attempts']
    assert attempt['reply'] == {'outcome': 'response', 'status': 200}
    assert attempt['model_endpoint']['endpoint'] == 'serving_endpoint'
    assert attempt['body_sha256'] == hashlib.sha256(received[0][1]).hexdigest()
    receipts = [RUNNER._payload(body) for _, body, _ in received]
    reader().check_receipts(checked['attempts'], receipts)
    with pytest.raises(reader().ObservationError):
        reader().check_receipts(checked['attempts'], [])


def pre_send_failure(monkeypatch, *, captured, raise_at):
    from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient

    recorder = _module(
        't44_serving_multicall_record',
        ROOT / 'contract/tools/multicall_record.py',
    )
    [(case, _)] = RUNNER.load_cases(
        [ROOT / 'contract/cases/s1/multicall.rows-per-call-3.json']
    )
    monkeypatch.setenv('SDM_CORE_ROUTES', full_routes())
    forbid_legacy_prepare(monkeypatch)
    connection, cleanup, _ = RUNNER._open(case, RUNNER.load_bases(), 'duckdb')
    boundary = None
    capture = None
    retained = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(RUNNER._source('duckdb', connection))
            request = PredictiveQuery(case['input']['pql'], schema).materialize(
                **RUNNER._materialize_arguments(case['input']['materialize'])
            )
        boundary = RUNNER._Boundary() if captured else None
        capture = boundary.reset() if boundary else None
        with (
            SERVE() as (url, received),
            SDMClient(url, max_retries=0) as client,
        ):
            recorder._raising(client, raise_at)
            original = client._transport.predict_body

            def send(*args, **kwargs):
                try:
                    return original(*args, **kwargs)
                except ValueError as error:
                    retained.append(error)
                    raise

            monkeypatch.setattr(client._transport, 'predict_body', send)
            with pytest.raises(ValueError) as caught:
                client.predict(request, model=KumoRelational())
            error = caught.value
            [native] = retained
            assert error is native
            assert type(error) is ValueError
            assert str(error) == recorder.RAISED_TEXT
            details = dict(getattr(error, 'details', None) or {})
            answered = details.pop('answered', None)
            return {
                'error': error,
                'capture': capture,
                'details': details,
                'answered': answered,
                'partial': recorder.projection(answered),
                'receipts': [
                    RUNNER._payload(body) for _, _, body in received.requests
                ],
            }
    finally:
        if boundary:
            boundary.close()
        cleanup()


@pytest.mark.parametrize(
    'raise_at', [0, 1], ids=['before-first-send', 'after-paid-call']
)
def test_actual_pre_send_value_error_preserves_paid_prefix_and_zero_receipt(
    monkeypatch, raise_at
):
    plain = pre_send_failure(monkeypatch, captured=False, raise_at=raise_at)
    captured = pre_send_failure(monkeypatch, captured=True, raise_at=raise_at)
    assert type(captured['error']) is type(plain['error'])
    assert captured['error'].args == plain['error'].args
    assert captured['details'] == plain['details']
    assert captured['partial'] == plain['partial']
    assert len(captured['receipts']) == len(plain['receipts']) == raise_at
    if raise_at:
        assert captured['details']['calls_answered'] == 1
        assert captured['details']['rows_answered'] == 3
        assert captured['answered'].items == plain['answered'].items
        assert captured['answered'].model == plain['answered'].model
        assert captured['answered'].request_id == plain['answered'].request_id
        assert captured['answered'].outputs == plain['answered'].outputs
        assert captured['answered'].metadata == plain['answered'].metadata
        assert (
            captured['answered']
            .to_pandas()
            .equals(plain['answered'].to_pandas())
        )
    else:
        assert captured['answered'] is plain['answered'] is None
    capture = captured['capture']
    checked = reader().view(capture.core(), capture.operations)
    assert len(checked['attempts']) == raise_at + 1
    assert checked['attempts'][-1]['reply'] == {'outcome': 'transport_error'}
    assert all(
        attempt['reply'] == {'outcome': 'response', 'status': 200}
        for attempt in checked['attempts'][:-1]
    )
    reader().check_receipts(checked['attempts'], captured['receipts'])
