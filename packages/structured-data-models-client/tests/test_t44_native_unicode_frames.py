# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless native Unicode inspection, keys, words, wire and result values."""

from __future__ import annotations

import copy
import json
import struct
import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import requests

from sdm_client import KumoRelational, PredictiveQuery, SDMClient, Schema

from test_t44_text_frames import (
    Scripted,
    core as core,
    dt3_no_time_body,
    p3e2_warning_codes,
    shop,
    text,
)


def reference(case):
    return json.loads(
        Path(__file__)
        .with_name('t44_native_unicode_journeys_reference.json')
        .read_text()
    )['records'][case]


def native_value(value):
    """Construct only the actual native values recorded by the producer.

    The ordinary JSON receipt cannot distinguish an explicit surrogate pair
    from one supplementary scalar, so the independent producer records all
    native Unicode points and numeric bits alongside its untouched bodies.
    """
    kind = value['kind']
    if kind == 'unicode':
        return ''.join(map(chr, value['points']))
    if kind == 'null':
        return None
    if kind == 'boolean':
        return value['value']
    if kind == 'integer':
        return int(value['digits'])
    if kind == 'float':
        return struct.unpack('>d', bytes(value['bits']))[0]
    if kind == 'array':
        return [native_value(item) for item in value['items']]
    assert kind == 'object'
    return {
        native_value(member['key']): native_value(member['value'])
        for member in value['members']
    }


@pytest.mark.parametrize(
    'case',
    ['categorical', 'text', 'key', 'pair-categorical', 'pair-text', 'pair-key'],
)
def test_native_unicode_keeps_full_inspection_wire_and_result_meaning(
    core, case
):
    recorded = reference(case)
    expected = native_value(recorded['inspection_native_values'])
    sent = native_value(recorded['sent_native_values'])
    wire = [
        json.loads(body) for body in recorded['actual_transport_ascii_bodies']
    ]
    notices = recorded['warnings']
    if core:
        # Only the registered DT-3 marker omission and P3E-2 warning codes
        # differ from the complete, untouched frozen capture.
        for body in [*expected, *sent, *wire]:
            dt3_no_time_body(body)
        notices = p3e2_warning_codes(notices)
    paired = case.startswith('pair-')
    kind = case.removeprefix('pair-')
    frames = shop('python')
    if kind == 'key':
        special = ['\ud83d\ude3a', '😺'] if paired else ['\ud800', '\udfff']
        keys = ['1', '2', '3', '4', '5', '6', *special]
        frames['users']['user_id'] = text(keys, 'python')
        frames['orders']['user_id'] = text(
            [key for key in keys for _ in range(8)], 'python'
        )
        supplied = special
    else:
        if paired and kind == 'text':
            values = ['A' * 298 + '\ud83d\ude3ax', 'A' * 298 + '😺x']
        elif paired:
            values = ['\ud83d\ude3a', '😺']
        else:
            values = ['\ud800 ΟΣ', '\udfff İ']
        frames['users'].loc[6, 'note'], frames['users'].loc[7, 'note'] = values
        supplied = [7, 8]
    scripted = Scripted('age')
    native_sent = []
    actual_wire = []
    ids = [body['metadata']['request_id'] for body in sent]
    with (
        pd.option_context(
            'future.infer_string', False, 'mode.string_storage', 'pyarrow'
        ),
        warnings.catch_warnings(record=True) as caught,
        patch('sdm_client.adapters.relational._request_id', side_effect=ids),
    ):
        warnings.simplefilter('always')
        schema = Schema.from_data(
            frames,
            primary_keys={'users': 'user_id', 'orders': 'order_id'},
            time_columns={},
        )
        if kind == 'text':
            schema['users']['note'].stype = 'text'
        query = PredictiveQuery(
            'PREDICT users.age FOR EACH users.user_id', schema
        )
        request = query.materialize(
            indices=supplied,
            num_context_samples=3,
            num_neighbors=[3],
            num_hops=1,
            random_seed=7,
        )
        assert type(request.payloads) is tuple
        assert list(request.payloads) == expected
        if core:
            assert request._core is not None
        with SDMClient('http://localhost:1', max_retries=0) as client:
            original_predict = client._transport.predict

            def observe_predict(payload):
                native_sent.append(copy.deepcopy(payload))
                return original_predict(payload)

            def local_post(url, *, data, **kwargs):
                assert url.endswith('/v1/predictions')
                assert type(data) is bytes
                assert data.isascii()
                actual_wire.append(json.loads(data))
                response = requests.Response()
                response.status_code = 200
                response._content = json.dumps(
                    scripted.predict(json.loads(data)), allow_nan=False
                ).encode('utf-8')
                response._content_consumed = True
                return response

            with (
                patch.object(client._transport, 'predict', observe_predict),
                patch.object(client._transport._session, 'post', local_post),
            ):
                result = client.predict(
                    request, model=KumoRelational()
                ).to_pandas()
        assert result.to_dict(orient='split') == native_value(
            recorded['result_native_values']
        )
        assert result.merge(
            frames['users'], left_on='id', right_on='user_id'
        ).to_dict(orient='split') == native_value(
            recorded['merged_native_values']
        )
    assert native_sent == sent
    assert actual_wire == wire
    assert [
        {'type': type(item.message).__name__, 'message': str(item.message)}
        for item in caught
    ] == notices
    if paired:
        native = request.payloads[0]['predict']['related_tables']['users']
        encoded = actual_wire[0]['predict']['related_tables']['users']
        column = 'user_id' if kind == 'key' else 'note'
        at = native['columns'].index(column)
        native_values = [row[at] for row in native['rows']]
        wire_values = [row[at] for row in encoded['rows']]
        if kind == 'text':
            native_values = [value[0] for value in native_values]
            wire_values = [value[0] for value in wire_values]
            assert [len(value) for value in native_values] == [300, 300]
            assert [len(value) for value in wire_values] == [299, 300]
            assert list(map(ord, native_values[0][-2:])) == [0xD83D, 0xDE3A]
            assert list(map(ord, native_values[1][-2:])) == [0x1F63A, ord('x')]
        else:
            assert [list(map(ord, value)) for value in native_values] == [
                [0xD83D, 0xDE3A],
                [0x1F63A],
            ]
            assert [list(map(ord, value)) for value in wire_values] == [
                [0x1F63A],
                [0x1F63A],
            ]
