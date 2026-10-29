# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical observation serialization preserves strict native body binding."""

from __future__ import annotations

import base64
import copy
import hashlib
import json

import pytest
from test_t44_prediction_observation import actual_observation, reader
from test_t44_prediction_prefix import compare_prefix, observation
from test_t44_prediction_serving_capture import served_observation


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


@pytest.mark.parametrize('serving', [False, True])
def test_complete_native_prediction_survives_sorted_jsonl_roundtrip(serving):
    core, operations = served_observation() if serving else actual_observation()
    before = copy.deepcopy(core)
    encoded = canonical(core)
    assert reader().view(encoded, operations) == reader().view(core, operations)
    assert core == before
    for original, restored in zip(
        core['predictions'][0]['calls'],
        encoded['predictions'][0]['calls'],
        strict=True,
    ):
        assert restored['input_body_base64'] == original['input_body_base64']
        assert restored['input_body_sha256'] == original['input_body_sha256']
        assert restored['body_sha256'] == original['body_sha256']


@pytest.mark.parametrize(
    'tamper',
    [
        'input-hash',
        'posted-hash',
        'missing-input',
        'body',
        'number-token',
        'list-order',
        'raw-order',
    ],
)
def test_direct_roundtrip_still_requires_actual_bytes_and_exact_snapshot(
    tamper,
):
    core, operations = actual_observation()
    core = canonical(core)
    call = core['predictions'][0]['calls'][0]
    if tamper == 'input-hash':
        call['input_body_sha256'] = '0' * 64
    elif tamper == 'posted-hash':
        call['body_sha256'] = '0' * 64
    elif tamper == 'missing-input':
        del call['input_body_base64']
    elif tamper == 'body':
        call['body']['model'] = 'changed-model'
    elif tamper == 'number-token':
        call['body']['context']['instance_table']['rows'][0][0] = {
            'number': '0.0'
        }
    elif tamper == 'list-order':
        call['body']['predict']['instance_table']['rows'].reverse()
    else:
        raw = json.loads(base64.b64decode(call['input_body_base64']))
        changed = json.dumps(raw, sort_keys=True).encode()
        call['input_body_base64'] = base64.b64encode(changed).decode('ascii')
        call['input_body_sha256'] = hashlib.sha256(changed).hexdigest()
        assert call['input_body_sha256'] != call['body_sha256']
    core['predicted']['body'] = copy.deepcopy(call['body'])
    with pytest.raises(reader().ObservationError):
        reader().view(core, operations)


@pytest.mark.parametrize(
    'tamper', ['member-order', 'duplicate-order', 'number-object']
)
def test_serving_roundtrip_keeps_ordered_actual_inner_json(tamper):
    core, operations = served_observation()
    core = canonical(core)
    call = core['predictions'][0]['calls'][0]
    text = call['body']['dataframe_records'][0]['request_json']
    if tamper == 'member-order':
        text = text.replace(
            '"actual_object":{"number":"1"},"actual_number":1',
            '"actual_number":1,"actual_object":{"number":"1"}',
        )
    elif tamper == 'duplicate-order':
        text = text.replace(
            '"duplicate":1,"duplicate":2', '"duplicate":2,"duplicate":1'
        )
    else:
        text = text.replace(
            '"actual_number":1', '"actual_number":{"number":"1"}'
        )
    call['body']['dataframe_records'][0]['request_json'] = text
    core['predicted']['body'] = copy.deepcopy(call['body'])
    with pytest.raises(reader().ObservationError):
        reader().view(core, operations)


def test_only_canonical_dto_object_keys_have_unordered_meaning():
    first = {'z': {'number': '-0.0'}, 'a': [{'number': '1'}, {'number': '2'}]}
    sorted_copy = canonical(first)
    assert not reader()._same_snapshot(first, sorted_copy)
    assert reader()._same_jsonl_snapshot(first, sorted_copy)
    for changed in (
        {'z': {'number': '0.0'}, 'a': first['a']},
        {'z': first['z'], 'a': list(reversed(first['a']))},
        {'z': first['z'], 'a': [{'number': '1.0'}, {'number': '2'}]},
    ):
        assert not reader()._same_jsonl_snapshot(first, changed)
    assert not reader()._same_jsonl_snapshot({'value': True}, {'value': 1})


def test_probability_column_order_survives_canonical_observation_comparison():
    core, operations = actual_observation()
    table = core['predictions'][0]['results'][0]['result']
    table['columns'].extend(
        {'name': name, 'type': 'float64'}
        for name in ('probability_class-b', 'probability_class-a')
    )
    for row in table['rows']:
        row.extend(
            [{'f64_bits': '3fd0000000000000'}, {'f64_bits': '3fe8000000000000'}]
        )
    core['predicted']['result'] = copy.deepcopy(table)
    encoded = canonical(core)
    reader().view(encoded, operations)
    assert (
        compare_prefix(
            observation(core, operations), observation(encoded, operations)
        )
        == []
    )
    changed = copy.deepcopy(encoded)
    changed_table = changed['predictions'][0]['results'][0]['result']
    changed_table['columns'][-2:] = reversed(changed_table['columns'][-2:])
    for row in changed_table['rows']:
        row[-2:] = reversed(row[-2:])
    changed['predicted']['result'] = copy.deepcopy(changed_table)
    reader().view(changed, operations)
    assert compare_prefix(
        observation(encoded, operations), observation(changed, operations)
    )
