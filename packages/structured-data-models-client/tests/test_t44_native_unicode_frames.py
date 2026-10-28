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
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import requests
from test_t42_arrow_decimal_frames import _decimal_error_graph
from test_t44_native_text_population import (
    CONTEXT_ANSWERS,
    HOPS,
    SOURCE_POSITIONS,
    TEXT_UNREAD,
    expected_core,
    points,
)
from test_t44_text_frames import (
    Scripted,
    dt3_no_time_body,
    p3e2_warning_codes,
    shop,
    text,
)
from test_t44_text_frames import (
    core as core,
)

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Schema,
    SDMClient,
    StructuredDataError,
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
            original_encode = client._transport.encode_prediction

            def observe_encode(payload):
                native_sent.append(copy.deepcopy(payload))
                return original_encode(payload)

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
                patch.object(
                    client._transport, 'encode_prediction', observe_encode
                ),
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


def _observe_text_conversion(monkeypatch, *, core_route, case):
    """Observe the existing public journey on one actual native profile."""
    with monkeypatch.context() as route:
        core.__wrapped__(SimpleNamespace(param=core_route), route)
        frames = shop('python')
        frames['orders']['ordered_at'] = pd.date_range(
            '2024-01-01', periods=64
        ).as_unit('ns')
        invalid = '\ud800' if case == 'front' else 'A' * 300 + '\ud800'
        frames['orders']['note'] = text(
            ['valid'] * 48 + [invalid] * 16, 'python'
        )
        options = dict(
            indices=[7, 8],
            num_context_samples=3,
            num_neighbors=[3],
            num_hops=1,
            random_seed=7,
        )
        events, conversions, switched = [], [], []
        sampled = False
        original_astype = pd.Series.astype

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

        def observed_astype(series, dtype, *args, **kwargs):
            item = None
            if series.name == 'note' and dtype is str:
                item = {
                    'input': points(series),
                    'index': series.index.tolist(),
                    'sampled': sampled,
                    'dtype': str(series.dtype),
                    'storage': getattr(series.dtype, 'storage', None),
                    'infer_string': pd.options.future.infer_string,
                    'option_storage': pd.options.mode.string_storage,
                }
                conversions.append(item)
                events.append('astype-str-enter')
            try:
                result = original_astype(series, dtype, *args, **kwargs)
            except Exception as error:
                if item is not None:
                    item.update(
                        outcome='error', error=_decimal_error_graph(error)
                    )
                    events.append('astype-str-error')
                raise
            if item is not None:
                item.update(
                    outcome='success',
                    result=points(result),
                    result_dtype=str(result.dtype),
                    result_storage=getattr(result.dtype, 'storage', None),
                )
                events.append('astype-str-exit')
            return result

        with (
            pd.option_context(
                'future.infer_string', False, 'mode.string_storage', 'python'
            ),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter('always')
            query = question()
            events.extend(['schema-ready', 'query-ready'])
            original_show = warnings.showwarning

            def switch_at_hops(
                message, category, filename, lineno, file=None, line=None
            ):
                nonlocal sampled
                events.append('warning')
                if str(message) == (
                    '[HOPS_IGNORED] ' + HOPS if core_route else HOPS
                ):
                    assert pd.options.future.infer_string is False
                    assert pd.options.mode.string_storage == 'python'
                    pd.options.future.infer_string = True
                    pd.options.mode.string_storage = 'pyarrow'
                    sampled = True
                    switched.append(True)
                    events.append('hops-option-switch')
                original_show(
                    message, category, filename, lineno, file=file, line=line
                )

            with (
                patch.object(warnings, 'showwarning', switch_at_hops),
                patch.object(pd.Series, 'astype', observed_astype),
            ):
                try:
                    request = query.materialize(**options)
                except StructuredDataError as error:
                    observed = {
                        'outcome': 'error',
                        'error': _decimal_error_graph(error),
                        'details': copy.deepcopy(error.details),
                    }
                    events.append('materialize-error')
                else:
                    observed = {
                        'outcome': 'success',
                        'payloads': copy.deepcopy(request.payloads),
                        'context_rows': request.context_rows,
                        'predict_rows': request.predict_rows,
                        'nothing_to_score': request.nothing_to_score,
                    }
                    events.append('materialize-complete')
            observed.update(
                warnings=[
                    {
                        'type': type(item.message).__name__,
                        'category': item.category.__module__
                        + '.'
                        + item.category.__qualname__,
                        'message': str(item.message),
                    }
                    for item in caught
                ],
                events=events,
                conversions=conversions,
                switched=switched,
                options_after={
                    'infer_string': pd.options.future.infer_string,
                    'storage': pd.options.mode.string_storage,
                },
            )
            if core_route:
                checkpoint = query._relational_materializer._core_sampler_state[
                    'held'
                ]['frame_checkpoint']
                observed['neighbors'] = json.loads(checkpoint['metadata'])[
                    'neighbors'
                ]
                # Same immutable data and graph walk with ordinary full-root
                # native conversion, as in the original failure control.
                with (
                    pd.option_context(
                        'future.infer_string',
                        False,
                        'mode.string_storage',
                        'python',
                    ),
                    warnings.catch_warnings(),
                ):
                    warnings.simplefilter('ignore')
                    control = question()
                    control.materialize(**options)
                checkpoint = (
                    control._relational_materializer._core_sampler_state[
                        'held'
                    ]['frame_checkpoint']
                )
                observed['control_neighbors'] = json.loads(
                    checkpoint['metadata']
                )['neighbors']
        return observed


@pytest.mark.parametrize('core_route', [False, True], ids=['original', 'core'])
@pytest.mark.parametrize('case', ['front', 'beyond-cap'])
def test_native_text_conversion_precedes_character_cap_after_sampling(
    monkeypatch, core_route, case
):
    recorded = json.loads(
        Path(__file__)
        .with_name('t44_native_text_conversion_failures_reference.json')
        .read_text()
    )['records'][case]
    # The historical producer's failure and schedule remain pinned evidence.
    # The same native operation can succeed on another installed profile.
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
    original = _observe_text_conversion(
        monkeypatch, core_route=False, case=case
    )
    actual = (
        _observe_text_conversion(monkeypatch, core_route=True, case=case)
        if core_route
        else original
    )
    if core_route:
        actual = copy.deepcopy(actual)
        neighbors = actual.pop('neighbors')
        assert neighbors == actual.pop('control_neighbors')
        initial = np.random.RandomState(123).get_state()
        assert neighbors != {
            'words': initial[1].tolist(),
            'position': int(initial[2]),
        }
        assert actual == expected_core(original)
    assert original['switched'] == [True]
    assert original['options_after'] == {
        'infer_string': True,
        'storage': 'pyarrow',
    }
    conversions = original['conversions']
    assert [len(item['input']) for item in conversions] == [6, 7, 64, 64, 9]
    assert [item['sampled'] for item in conversions] == [False] * 4 + [True]
    assert all(item['outcome'] == 'success' for item in conversions[:-1])
    final = conversions[-1]
    invalid = '\ud800' if case == 'front' else 'A' * 300 + '\ud800'
    assert final['input'] == [
        list(map(ord, 'valid' if source < 48 else invalid))
        for source in SOURCE_POSITIONS
    ]
    assert final['index'] == list(range(9))
    assert (
        final['infer_string'] is True and final['option_storage'] == 'pyarrow'
    )
    assert final['storage'] == 'python'
    if case == 'beyond-cap':
        assert any(
            len(value) == 301 and value[-1] == 0xD800
            for value in final['input']
        )
    prefix = [
        'schema-ready',
        'query-ready',
        *[
            event
            for _ in range(4)
            for event in ('astype-str-enter', 'astype-str-exit')
        ],
        'warning',
        'hops-option-switch',
        'astype-str-enter',
    ]
    if original['outcome'] == 'error':
        error = original['error']
        assert {
            'stage': 'materialize',
            'type': error['type'].rsplit('.', 1)[-1],
            'code': error['code'],
            'message': error['text'],
        } == recorded['error']
        assert final['outcome'] == 'error'
        assert original['events'] == prefix + [
            'astype-str-error',
            'materialize-error',
        ]
        assert [
            {'type': item['type'], 'message': item['message']}
            for item in original['warnings']
        ] == recorded['warnings']
    else:
        assert original['outcome'] == final['outcome'] == 'success'
        assert final['result'] == final['input']
        assert original['context_rows'] == 1 and original['predict_rows'] == 2
        assert original['nothing_to_score'] is False
        assert original['warnings'] == [
            {
                'type': 'UserWarning',
                'category': 'builtins.UserWarning',
                'message': HOPS,
            },
            {
                'type': 'SDMWarning',
                'category': 'sdm_client.core.warnings.SDMWarning',
                'message': TEXT_UNREAD,
            },
            {
                'type': 'SDMWarning',
                'category': 'sdm_client.core.warnings.SDMWarning',
                'message': CONTEXT_ANSWERS,
            },
        ]
        assert original['events'] == prefix + [
            'astype-str-exit',
            'warning',
            'warning',
            'materialize-complete',
        ]
        for side in ('context', 'predict'):
            orders = original['payloads'][0][side]['related_tables']['orders']
            note = orders['columns'].index('note')
            values = [row[note] for row in orders['rows']]
            assert values == (
                [['valid']] * 3
                if side == 'context'
                else [[invalid[:300].lower()]] * 6
            )
