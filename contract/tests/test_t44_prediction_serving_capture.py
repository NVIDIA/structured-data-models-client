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
from test_t136_forecast_capture import _module

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


def test_direct_body_snapshot_cannot_change_while_retaining_the_observed_hash():
    core, operations = actual_observation()
    call = core['predictions'][0]['calls'][0]
    call['body']['model'] = 'changed-after-the-hash'
    core['predicted']['body'] = copy.deepcopy(call['body'])
    with pytest.raises(reader().ObservationError):
        reader().view(core, operations)
