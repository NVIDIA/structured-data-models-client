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
import numpy as np
import pytest
import requests

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    SDMClient,
    Schema,
    StructuredDataError,
)

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


@pytest.mark.parametrize('storage', ['python', 'pyarrow', 'utf8'])
def test_native_key_conversion_failure_keeps_its_guard_and_untouched_neighbors(
    core, storage
):
    expected = json.loads(
        Path(__file__)
        .with_name('t44_native_key_conversion_failures_reference.json')
        .read_text()
    )['records'][storage]
    frames = shop('python')
    keys = [str(number) for number in range(1, 9)]
    frames['users']['user_id'] = text(keys, storage)
    frames['orders']['user_id'] = text(
        [key for key in keys for _ in range(8)], storage
    )
    with (
        pd.option_context('future.infer_string', False),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter('always')
        schema = Schema.from_data(
            frames,
            primary_keys={'users': 'user_id', 'orders': 'order_id'},
            time_columns={},
        )
        query = PredictiveQuery(
            'PREDICT users.age FOR EACH users.user_id', schema
        )
        with pytest.raises(StructuredDataError) as failure:
            query.materialize(
                indices=['7', '\ud800'],
                num_context_samples=3,
                num_neighbors=[3],
                num_hops=1,
                random_seed=7,
            )
    assert {
        'stage': 'materialize',
        'type': type(failure.value).__name__,
        'code': failure.value.code,
        'message': str(failure.value),
    } == expected['error']
    notices = expected['warnings']
    if core:
        notices = p3e2_warning_codes(notices)
        state = query._relational_materializer._core_sampler_state['held'][
            'frame_checkpoint'
        ]
        neighbors = json.loads(state['metadata'])['neighbors']
        # Independent native MT initialization: these failures happen before
        # either a reset to the caller's seed7 or the first neighbor draw.
        initial = np.random.RandomState(123).get_state()
        assert neighbors == {
            'words': initial[1].tolist(),
            'position': int(initial[2]),
        }
    assert [
        {'type': type(item.message).__name__, 'message': str(item.message)}
        for item in caught
    ] == notices
    assert not [
        event
        for event in expected['native_trace']
        if event['event'] == 'local-sample-enter'
    ]


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


@pytest.mark.parametrize('case', ['front', 'beyond-cap'])
def test_native_text_conversion_precedes_character_cap_after_sampling(
    core, case
):
    recorded = json.loads(
        Path(__file__)
        .with_name('t44_native_text_conversion_failures_reference.json')
        .read_text()
    )['records'][case]
    invalid = '\ud800' if case == 'front' else 'A' * 300 + '\ud800'
    frames = shop('python')
    frames['orders']['ordered_at'] = pd.date_range(
        '2024-01-01', periods=64
    ).as_unit('ns')
    values = ['valid'] * 48 + [invalid] * 16
    frames['orders']['note'] = text(values, 'python')
    options = dict(
        indices=[7, 8],
        num_context_samples=3,
        num_neighbors=[3],
        num_hops=1,
        random_seed=7,
    )
    notices = recorded['warnings']
    if core:
        notices = p3e2_warning_codes(notices)
    switched = []
    original_show = warnings.showwarning

    def switch_at_hops(
        message, category, filename, lineno, file=None, line=None
    ):
        if str(message) == notices[0]['message']:
            assert pd.options.future.infer_string is False
            assert pd.options.mode.string_storage == 'python'
            pd.options.future.infer_string = True
            pd.options.mode.string_storage = 'pyarrow'
            switched.append(str(message))
        original_show(message, category, filename, lineno, file=file, line=line)

    def question():
        schema = Schema.from_data(
            frames,
            primary_keys={'users': 'user_id', 'orders': 'order_id'},
            time_columns={'orders': 'ordered_at'},
        )
        schema['orders']['note'].stype = 'text'
        return PredictiveQuery(
            'PREDICT users.age FOR EACH users.user_id', schema
        )

    with (
        pd.option_context(
            'future.infer_string', False, 'mode.string_storage', 'python'
        ),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter('always')
        query = question()
        with patch.object(warnings, 'showwarning', switch_at_hops):
            with pytest.raises(StructuredDataError) as failure:
                query.materialize(**options)
    assert {
        'stage': 'materialize',
        'type': type(failure.value).__name__,
        'code': failure.value.code,
        'message': str(failure.value),
    } == recorded['error']
    assert switched == [notices[0]['message']]
    assert [
        {'type': type(item.message).__name__, 'message': str(item.message)}
        for item in caught
    ] == notices
    assert not recorded['sent']
    assert [event['event'] for event in recorded['trace']] == [
        'schema-ready',
        'query-ready',
        'warning',
        'hops-option-switch',
        'sample-enter',
        'sample-exit',
        'normalize-enter',
        'astype-str-enter',
        'normalize-error',
    ]
    if core:
        # A successful full-root conversion over the same immutable data has
        # exactly the same graph walk. This compares real continuing state,
        # independently of the producer's unrelated NumPy global state.
        failed = query._relational_materializer._core_sampler_state['held'][
            'frame_checkpoint'
        ]
        after_failure = json.loads(failed['metadata'])['neighbors']
        with (
            pd.option_context(
                'future.infer_string', False, 'mode.string_storage', 'python'
            ),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter('ignore')
            control = question()
            control.materialize(**options)
        checkpoint = control._relational_materializer._core_sampler_state[
            'held'
        ]['frame_checkpoint']
        assert after_failure == json.loads(checkpoint['metadata'])['neighbors']
        initial = np.random.RandomState(123).get_state()
        assert after_failure != {
            'words': initial[1].tolist(),
            'position': int(initial[2]),
        }
