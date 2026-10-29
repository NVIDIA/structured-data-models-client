# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual serving bodies remain distinct from the encoded native request."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_t44_prediction_observation import actual_observation, reader
from test_t136_forecast_capture import RUNNER, _module

ROOT = Path(__file__).parents[2]
CHECK = _module(
    't44_serving_capture_check', ROOT / 'contract/runners/capture_check.py'
)


def served_observation(input_text=None, posted_text=None):
    core, operations = actual_observation()
    call = core['predictions'][0]['calls'][0]
    request_id = call['request_id']
    if input_text is None:
        input_text = (
            '{ "model":"kumo-relational", "metadata":{'
            '"request_id":"' + request_id + '","duplicate":1,"duplicate":2},'
            '"text":"\\u00e9\\ud83d\\ude00","float":-0.0,'
            '"actual_object":{"number":"1"},"actual_number":1 }'
        )
    if posted_text is None:
        posted_text = input_text.replace('{ ', '{').replace(' }', '}')
    wire = json.dumps(
        {'dataframe_records': [{'request_json': posted_text}]}
    ).encode()
    call['input_body_base64'] = base64.b64encode(input_text.encode()).decode(
        'ascii'
    )
    call['input_body_sha256'] = hashlib.sha256(input_text.encode()).hexdigest()
    call['body'] = json.loads(wire)
    call['body_sha256'] = hashlib.sha256(wire).hexdigest()
    core['inputs'][3]['input']['input']['model_endpoint'] = {
        'endpoint': 'serving_endpoint',
        'name': 'fixture',
        'max_request_bytes': 16000000,
    }
    core['predicted']['body'] = copy.deepcopy(call['body'])
    return core, operations


def test_serving_common_reader_binds_original_bytes_without_equating_wire_hash():
    core, operations = served_observation()
    call = core['predictions'][0]['calls'][0]
    assert call['input_body_sha256'] != call['body_sha256']
    assert 'metadata' not in call['body']
    assert reader().view(core, operations)['operations'][-1] == 'predict'


@pytest.mark.parametrize(
    'tamper',
    [
        'id',
        'order',
        'number-token',
        'number-object',
        'extra-row',
        'extra-field',
        'bad-base64',
        'input-hash',
        'direct-mode',
    ],
)
def test_serving_reader_rejects_changed_binding_without_rebuilding_body(tamper):
    core, operations = served_observation()
    call = core['predictions'][0]['calls'][0]
    inner = call['body']['dataframe_records'][0]['request_json']
    if tamper == 'id':
        inner = inner.replace(call['request_id'], 'another-call')
    elif tamper == 'order':
        inner = inner.replace(
            '"duplicate":1,"duplicate":2', '"duplicate":2,"duplicate":1'
        )
    elif tamper == 'number-token':
        inner = inner.replace('"float":-0.0', '"float":0.0')
    elif tamper == 'number-object':
        inner = inner.replace(
            '"actual_number":1', '"actual_number":{"number":"1"}'
        )
    elif tamper == 'extra-row':
        call['body']['dataframe_records'].append({'request_json': inner})
    elif tamper == 'extra-field':
        call['body']['dataframe_records'][0]['extra'] = None
    elif tamper == 'bad-base64':
        call['input_body_base64'] = '!!!'
    elif tamper == 'input-hash':
        call['input_body_sha256'] = '0' * 64
    elif tamper == 'direct-mode':
        core['inputs'][3]['input']['input']['model_endpoint'] = None
    call['body']['dataframe_records'][0]['request_json'] = inner
    core['predicted']['body'] = copy.deepcopy(call['body'])
    with pytest.raises(reader().ObservationError):
        reader().view(core, operations)


@pytest.mark.parametrize('limit', [None, 1], ids=['success', 'pre-call-limit'])
def test_actual_serving_prediction_is_unchanged_under_capture(
    monkeypatch, limit
):
    from sdm_client import _routing

    serving = _module(
        't44_capture_serving_route_controls',
        ROOT
        / 'packages/structured-data-models-client/tests/test_t313_serving_route.py',
    )
    serving._old.__wrapped__(monkeypatch)

    def legacy_prepare(*args, **kwargs):
        raise AssertionError('core serving fixture called legacy _prepare')

    monkeypatch.setattr(serving.SDMClient, '_prepare', legacy_prepare)
    request = serving._materialized(monkeypatch, unicode_feature=True)
    # Fixture helpers own only their original three routes. The actual new
    # producer requires the full existing preparation inventory as well.
    from test_t136_forecast_capture import EN

    routes = ','.join(list(_routing.S0) + EN.core_routes({}))

    def ask(captured):
        boundary = RUNNER._Boundary() if captured else None
        observed = boundary.reset() if boundary else None
        try:
            result, received = serving._predict(
                monkeypatch, request, routes, max_request_bytes=limit
            )
            return result, received, observed
        finally:
            if boundary:
                boundary.close()

    plain, plain_received, _ = ask(False)
    captured, received, observed = ask(True)
    assert observed is not None
    if limit is None:
        assert not isinstance(plain, Exception), plain
        assert not isinstance(captured, Exception), captured
        assert captured.to_pandas().equals(plain.to_pandas())
        assert captured.items == plain.items
        assert captured.metadata == plain.metadata
        assert all(
            path == serving.INVOCATIONS
            for path, _, _ in [*plain_received, *received]
        )
        plain_body = json.loads(plain_received[0][1])
        captured_body = json.loads(received[0][1])
        plain_id = json.loads(
            plain_body['dataframe_records'][0]['request_json']
        )['metadata']['request_id']
        captured_id = json.loads(
            captured_body['dataframe_records'][0]['request_json']
        )['metadata']['request_id']
        assert plain_id != captured_id
        assert [
            CHECK.masked_serving(json.loads(body)) for _, body, _ in received
        ] == [
            CHECK.masked_serving(json.loads(body))
            for _, body, _ in plain_received
        ]
        [entry] = observed.core()['predictions']
        [call] = entry['calls']
        assert call['body_sha256'] == hashlib.sha256(received[0][1]).hexdigest()
        assert call['body'] == json.loads(received[0][1])
        assert call['input_body_sha256'] != call['body_sha256']
        assert (
            hashlib.sha256(
                base64.b64decode(call['input_body_base64'])
            ).hexdigest()
            == call['input_body_sha256']
        )
    else:
        assert type(captured) is type(plain)
        assert (captured.code, captured.message, captured.details) == (
            plain.code,
            plain.message,
            plain.details,
        )
        assert received == plain_received == []
        [entry] = observed.core()['predictions']
        [call] = entry['calls']
        assert call['body'] is None
    checked = reader().view(observed.core(), observed.operations)
    reader().check_receipts(
        checked['attempts'],
        [
            {
                'parsed': json.loads(body),
                'sha256': hashlib.sha256(body).hexdigest(),
            }
            for _, body, _ in received
        ],
    )


def test_direct_body_snapshot_cannot_change_while_retaining_the_observed_hash():
    core, operations = actual_observation()
    call = core['predictions'][0]['calls'][0]
    call['body']['model'] = 'changed-after-the-hash'
    core['predicted']['body'] = copy.deepcopy(call['body'])
    with pytest.raises(reader().ObservationError):
        reader().view(core, operations)
