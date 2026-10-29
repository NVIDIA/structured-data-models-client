# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public native Text operands retain source-row compaction before conversion."""

from __future__ import annotations

import copy
import json
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from test_t42_arrow_decimal_frames import _decimal_error_graph
from test_t44_native_profile_observations import (
    OptionEffects,
    product_notices,
    semantic_events,
)
from test_t44_text_frames import core as select_core
from test_t44_text_frames import p3e2_warning_codes, shop, text

from sdm_client import PredictiveQuery, Schema, StructuredDataError

HOPS = "Received custom 'num_neighbors' option; ignoring custom 'num_hops=1' option"
TEXT_UNREAD = (
    'orders.note is typed text, which this model does not read, so the request '
    'is answered without it and it contributes nothing to the prediction. '
    'Type a column categorical if its values are labels rather than prose, '
    'or leave it out to say so.'
)
CONTEXT_ANSWERS = (
    '2 of the rows sampled as context are the rows being scored, at one moment '
    'or another, carrying the value being asked for. Rows that carry the '
    'answer to the question being asked are not sent as context, so these do '
    'not go into it.'
)
# Original public marker values from the independently sampled temporal shop.
# These identify source rows, independently of text values or projection facts.
SOURCE_POSITIONS = [5, 6, 7, 53, 54, 55, 61, 62, 63]
OPTIONS = dict(
    indices=[7, 8],
    num_context_samples=3,
    num_neighbors=[3],
    num_hops=1,
    random_seed=7,
)


def constructor_notice(column):
    # The schema infers categorical before this fixture declares Text.
    return (
        f'orders.{column} holds 64 distinct values in the '
        '64 rows that have one, so the rows ran out before '
        'the values did and the count is a measure of the frame '
        'rather than of the column. It is read as categorical here '
        'because 64 is at most 64; on more rows the same '
        'column could pass that and be read as text, which the '
        'models treat differently. If this frame is a sample, say '
        'what the column is with '
        f"schema['orders'][{column!r}].stype = 'categorical' (or "
        "'text'), so that it does not change under a larger one."
    )


def points(values):
    # Preserve every actual native code point, including NUL and non-BMP text.
    return [
        None if value is None or value is pd.NA else list(map(ord, value))
        for value in values
    ]


def frames_for(case, *, later=False):
    frames = shop('python')
    orders = frames['orders']
    orders['ordered_at'] = pd.date_range('2024-01-01', periods=64).as_unit('ns')
    if case == 'front':
        values = [f'ΑΒ É<BR/>😺 ROW {at}' for at in range(64)]
    elif case == 'beyond-cap':
        values = ['A' * 300 + f' ΑΒ 😺 ROW {at}' for at in range(64)]
    else:
        assert case == 'equal-text'
        values = ['ΑΒ É <br/>😺'] * 64
    orders['note'] = text(values, 'python')
    if later:
        orders['note_later'] = text(
            [f'LATER ΑΒ {at}' for at in range(64)], 'python'
        )
    return frames


def dtype_observation(series):
    return {
        'type': f'{type(series.dtype).__module__}.{type(series.dtype).__qualname__}',
        'name': str(series.dtype),
        'storage': getattr(series.dtype, 'storage', None),
    }


def observe(monkeypatch, *, core, case, later_failure=False):
    frames = frames_for(case, later=later_failure)
    original_note = frames['orders']['note'].copy(deep=True)
    original_later = (
        frames['orders']['note_later'].copy(deep=True)
        if later_failure
        else None
    )
    conversions = []
    events = []
    sampled = False
    option_effects = OptionEffects()

    def read_inference():
        with option_effects.at('option-read'):
            return pd.options.future.infer_string

    retained = ValueError('native later Text conversion failed')
    original_astype = pd.Series.astype
    result = {}

    def observed_astype(series, dtype, *args, **kwargs):
        selected = series.name in ('note', 'note_later') and dtype is str
        item = None
        if selected:
            item = {
                'column': series.name,
                'sampled': sampled,
                'input': points(series),
                'index': series.index.tolist(),
                'dtype': dtype_observation(series),
                'infer_string': read_inference(),
                'storage': pd.options.mode.string_storage,
            }
            conversions.append(item)
            events.append((series.name, 'astype-enter'))
        try:
            if selected and sampled and series.name == 'note' and later_failure:
                # Conversion has already received its immutable operands.
                # The later column must still use its retained source data.
                frames['orders'].loc[:, 'note_later'] = 'MUTATED caller'
                events.append(('note', 'caller-mutation'))
            if selected and sampled and series.name == 'note_later':
                assert later_failure
                raise retained
            converted = original_astype(series, dtype, *args, **kwargs)
        except Exception as error:
            if item is not None:
                item.update(outcome='error', error=_decimal_error_graph(error))
                events.append((series.name, 'astype-error'))
            raise
        if item is not None:
            item.update(
                outcome='success',
                result=points(converted),
                result_dtype=dtype_observation(converted),
            )
            events.append((series.name, 'astype-exit'))
        return converted

    with (
        monkeypatch.context() as route,
        pd.option_context(
            'future.infer_string', False, 'mode.string_storage', 'python'
        ),
        warnings.catch_warnings(record=True) as caught,
    ):
        select_core.__wrapped__(SimpleNamespace(param=core), route)
        warnings.simplefilter('always')
        schema = Schema.from_data(
            frames,
            primary_keys={'users': 'user_id', 'orders': 'order_id'},
            time_columns={'orders': 'ordered_at'},
        )
        schema['orders']['note'].stype = 'text'
        if later_failure:
            schema['orders']['note_later'].stype = 'text'
        query = PredictiveQuery(
            'PREDICT users.age FOR EACH users.user_id', schema
        )
        events.extend(['schema-ready', 'query-ready'])
        original_show = warnings.showwarning

        def switch_at_hops(
            message, category, filename, lineno, file=None, line=None
        ):
            nonlocal sampled
            option_effects.warning(
                message, category, filename, len(events), occurrence=len(caught)
            )
            events.append('warning')
            expected = '[HOPS_IGNORED] ' + HOPS if core else HOPS
            if str(message) == expected:
                assert sampled is False
                assert read_inference() is False
                assert pd.options.mode.string_storage == 'python'
                with option_effects.at('option-write'):
                    pd.options.future.infer_string = True
                pd.options.mode.string_storage = 'pyarrow'
                sampled = True
                events.append('hops-option-switch')
            original_show(
                message, category, filename, lineno, file=file, line=line
            )

        with (
            patch.object(warnings, 'showwarning', switch_at_hops),
            patch.object(pd.Series, 'astype', observed_astype),
        ):
            try:
                request = query.materialize(**OPTIONS)
            except StructuredDataError as error:
                assert later_failure
                result.update(
                    outcome='error',
                    error=_decimal_error_graph(error),
                    details=copy.deepcopy(error.details),
                    retained_cause=error.__cause__ is retained,
                )
                events.append('materialize-error')
            else:
                assert not later_failure
                result.update(
                    outcome='success',
                    payloads=copy.deepcopy(list(request.payloads)),
                    context_rows=request.context_rows,
                    predict_rows=request.predict_rows,
                    nothing_to_score=request.nothing_to_score,
                )
                events.append('materialize-complete')
        with patch.object(warnings, 'showwarning', switch_at_hops):
            result['options_after'] = {
                'infer_string': read_inference(),
                'storage': pd.options.mode.string_storage,
            }
        result['warnings'] = [
            {
                'type': type(item.message).__name__,
                'category': f'{item.category.__module__}.{item.category.__qualname__}',
                'message': str(item.message),
            }
            for item in caught
        ]
        result['option_effects'] = option_effects.snapshot(caught)
        if core:
            held = query._relational_materializer._core_sampler_state['held']
            result['neighbors'] = json.loads(
                held['frame_checkpoint']['metadata']
            )['neighbors']
    pd.testing.assert_series_equal(frames['orders']['note'], original_note)
    if later_failure:
        assert (
            frames['orders']['note_later'].tolist() == ['MUTATED caller'] * 64
        )
        result['later_source'] = points(original_later)
    result.update(conversions=conversions, events=events)
    return result


def assert_population(observed, case):
    note = [
        item for item in observed['conversions'] if item['column'] == 'note'
    ]
    # All four native graph-building conversions precede the warning mutation.
    # The last conversion must see sorted source positions, never fifteen
    # expanded occurrences and never text-content deduplication.
    assert [len(item['input']) for item in note] == [6, 7, 64, 64, 9]
    assert [item['sampled'] for item in note] == [False] * 4 + [True]
    assert all(item['outcome'] == 'success' for item in note)
    final = note[-1]
    source = frames_for(case)['orders']['note']
    assert final['input'] == points(source.iloc[SOURCE_POSITIONS])
    assert final['result'] == final['input']
    assert final['index'] == list(range(9))
    assert final['dtype']['storage'] == 'python'
    assert final['infer_string'] is True
    assert final['storage'] == 'pyarrow'
    if case == 'equal-text':
        assert len({tuple(value) for value in final['input']}) == 1
        assert len(final['input']) == 9


def assert_public_source_rows(observed, case):
    assert observed['outcome'] == 'success'
    assert observed['context_rows'] == 1
    assert observed['predict_rows'] == 2
    assert observed['nothing_to_score'] is False
    assert observed['options_after'] == {
        'infer_string': True,
        'storage': 'pyarrow',
    }
    assert len(observed['payloads']) == 1
    body = observed['payloads'][0]
    source_positions = []
    for side in ('context', 'predict'):
        table = body[side]['related_tables']['orders']
        marker = table['columns'].index('marker')
        note = table['columns'].index('note')
        for row in table['rows']:
            at = row[marker]
            source_positions.append(at)
            if case == 'front':
                assert row[note] == ['αβ', 'é', '😺', 'row', str(at)]
            elif case == 'beyond-cap':
                assert row[note] == ['a' * 300]
            else:
                assert row[note] == ['αβ', 'é', '😺']
    # Independently recover the source rows from public marker values.
    assert sorted(source_positions) == SOURCE_POSITIONS


def expected_core(original):
    expected = copy.deepcopy(original)
    expected['warnings'] = p3e2_warning_codes(expected['warnings'])
    for notice in expected['warnings']:
        if notice['message'] == '[HOPS_IGNORED] ' + HOPS:
            # The registered P3E-2 selected warning is a native subclass of
            # UserWarning; compare its exact class as well as its exact code.
            assert notice['category'] == 'builtins.UserWarning'
            notice['category'] = 'sdm_client._rust.errors.UserWarning'
        elif notice['message'] == TEXT_UNREAD:
            assert notice['category'] == 'sdm_client.core.warnings.SDMWarning'
            notice['message'] = '[COLUMNS_UNREAD] ' + TEXT_UNREAD
    return expected


@pytest.mark.parametrize('case', ['front', 'beyond-cap', 'equal-text'])
def test_native_text_uses_source_population_before_conversion(
    monkeypatch, case
):
    original = observe(monkeypatch, core=False, case=case)
    actual = observe(monkeypatch, core=True, case=case)
    neighbors = actual.pop('neighbors')
    assert neighbors['position'] > 0
    assert actual == expected_core(original)
    assert_population(original, case)
    assert_public_source_rows(original, case)
    constructor_notices = (
        [] if case == 'equal-text' else [constructor_notice('note')]
    )
    assert [notice['message'] for notice in product_notices(original)] == [
        *constructor_notices,
        HOPS,
        TEXT_UNREAD,
        CONTEXT_ANSWERS,
    ]
    assert semantic_events(original) == [
        'schema-ready',
        'query-ready',
        *[
            ('note', event)
            for _ in range(4)
            for event in ('astype-enter', 'astype-exit')
        ],
        'warning',
        'hops-option-switch',
        ('note', 'astype-enter'),
        ('note', 'astype-exit'),
        'warning',
        'warning',
        'materialize-complete',
    ]
    assert_population(actual, case)
    assert_public_source_rows(actual, case)


def test_later_native_text_failure_retains_compact_operands_and_draws(
    monkeypatch,
):
    original = observe(
        monkeypatch, core=False, case='front', later_failure=True
    )
    actual = observe(monkeypatch, core=True, case='front', later_failure=True)
    neighbors = actual.pop('neighbors')
    assert actual == expected_core(original)
    assert_population(original, 'front')
    assert original['outcome'] == 'error'
    assert original['retained_cause'] is True
    assert [notice['message'] for notice in product_notices(original)] == [
        constructor_notice('note'),
        constructor_notice('note_later'),
        HOPS,
    ]
    later = [
        item
        for item in original['conversions']
        if item['column'] == 'note_later' and item['sampled']
    ]
    assert len(later) == 1
    assert later[0]['input'] == [
        original['later_source'][at] for at in SOURCE_POSITIONS
    ]
    assert later[0]['outcome'] == 'error'
    assert later[0]['error'] == {
        'type': 'builtins.ValueError',
        'code': None,
        'text': 'native later Text conversion failed',
        'cause': None,
        'context': 'same-as-cause',
        'suppress': False,
    }
    assert semantic_events(original)[-5:] == [
        ('note', 'caller-mutation'),
        ('note', 'astype-exit'),
        ('note_later', 'astype-enter'),
        ('note_later', 'astype-error'),
        'materialize-error',
    ]
    assert_population(actual, 'front')
    control = observe(monkeypatch, core=True, case='front')
    assert neighbors == control['neighbors']
