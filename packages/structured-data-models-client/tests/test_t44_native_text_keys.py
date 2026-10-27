# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen public text-key conversions and native physical Text refusals."""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from sdm_client import PredictiveQuery, Schema, StructuredDataError

from test_t44_text_frames import (
    core as core,
    dt3_no_time_body,
    inspection,
    p3e2_warning_codes,
    predict,
    question,
    shop,
    text,
)


def reference(family):
    cell = 'pandas2' if int(pd.__version__.split('.')[0]) == 2 else 'pandas3'
    return json.loads(
        Path(__file__)
        .with_name(f't44_native_text_{family}_{cell}_reference.json')
        .read_text()
    )['records']


KEY_CASES = [
    f'{storage}-{variant}'
    for storage in ('python', 'pyarrow', 'utf8')
    for variant in (
        'lexical',
        'special',
        'null-key',
        'duplicate-key',
        'integer-entities',
        'float-entities',
        'boolean-entities',
        'null-entity',
        'literal-na-null-entity',
        'literal-nan-null-entity',
    )
] + ['largeutf8-lexical', 'largeutf8-null-entity']


@pytest.mark.parametrize('case', KEY_CASES)
def test_text_keys_preserve_native_conversion_and_full_public_results(
    core, case
):
    expected = copy.deepcopy(reference('keys')[case])
    storage = case.split('-', 1)[0]
    frames = shop(storage)
    keys = expected['input_keys']
    frames['users']['user_id'] = text(keys, storage)
    frames['orders']['user_id'] = text(
        [key for key in keys for _ in range(8)], storage
    )
    if core:
        for body in expected.get('inspection', {}).get('payloads', []):
            dt3_no_time_body(body)
        for body in expected['sent']:
            dt3_no_time_body(body)
        expected['warnings'] = p3e2_warning_codes(expected['warnings'])
    stage = 'schema'

    def gather():
        nonlocal stage
        asked = question(frames)
        stage = 'materialize'
        return asked.materialize(
            indices=expected['supplied_entities'],
            num_context_samples=3,
            num_neighbors=[3],
            num_hops=1,
            random_seed=7,
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if 'error' in expected:
            with pytest.raises(StructuredDataError) as failure:
                gather()
            assert {
                'stage': stage,
                'type': type(failure.value).__name__,
                'code': failure.value.code,
                'message': str(failure.value),
            } == expected['error']
        else:
            request = gather()
            assert inspection(request) == expected['inspection']
            if core:
                assert request._core is not None
            predict(request, 'age', frames, expected)
    assert [
        {'type': type(item.message).__name__, 'message': str(item.message)}
        for item in caught
    ] == expected['warnings']


def numeric(kind):
    if kind == 'numpy-int64':
        return pd.Series(np.arange(1, 9), dtype='int64')
    if kind == 'numpy-bool':
        return pd.Series([True, False] * 4, dtype=bool)
    if kind in ('numpy-float32', 'numpy-float64'):
        return pd.Series(
            [1.0, float('nan'), -0.0, 1e-20, 1e20, 6.25, float('nan'), 8.5],
            dtype=kind.removeprefix('numpy-'),
        )
    dtype = pa.int64() if kind == 'arrow-int64' else pa.float64()
    values = (
        [1, None, -3, 4, 5, 6, 7, 8]
        if kind == 'arrow-int64'
        else [1.0, None, -0.0, 1e-20, 1e20, 6.25, 7.0, 8.5]
    )
    return pd.Series(
        pd.arrays.ArrowExtensionArray(pa.array(values, type=dtype))
    )


NUMERIC_CASES = [
    f'{kind}-{storage}-True-False'
    for kind in (
        'numpy-int64',
        'numpy-float32',
        'numpy-float64',
        'numpy-bool',
        'arrow-int64',
        'arrow-float64',
    )
    for storage in ('python', 'pyarrow')
] + [
    'numpy-int64-python-False-False',
    'arrow-int64-python-True-True',
    'arrow-float64-python-True-True',
]


@pytest.mark.parametrize('case', NUMERIC_CASES)
def test_physical_numeric_text_keeps_actual_schema_and_native_graph_errors(
    core, case
):
    expected = reference('numeric')[case]
    kind, storage, metadata_edit, selected_null = case.rsplit('-', 3)
    note = numeric(kind)
    if selected_null == 'True':
        note.iloc[6] = pd.NA
    frames = shop()
    frames['users']['note'] = note
    frames['orders']['note'] = note.repeat(8).reset_index(drop=True)
    stage = 'schema'
    with (
        pd.option_context(
            'mode.string_storage',
            storage,
            'future.infer_string',
            int(pd.__version__.split('.')[0]) >= 3,
        ),
        warnings.catch_warnings(record=True) as caught,
        pytest.raises(StructuredDataError) as failure,
    ):
        warnings.simplefilter('always')
        schema = Schema.from_data(
            frames,
            primary_keys={'users': 'user_id', 'orders': 'order_id'},
            time_columns={},
        )
        for table in ('users', 'orders'):
            if metadata_edit == 'True':
                schema[table]['note'].dtype = 'string'
            schema[table]['note'].stype = 'text'
        schema.check_consistent()
        asked = PredictiveQuery(
            'PREDICT users.age FOR EACH users.user_id', schema
        )
        stage = 'materialize'
        asked.materialize(
            indices=[7, 8],
            num_context_samples=3,
            num_neighbors=[3],
            num_hops=1,
            random_seed=7,
        )
    assert {
        'stage': stage,
        'type': type(failure.value).__name__,
        'code': failure.value.code,
        'message': str(failure.value),
    } == expected['error']
    assert [
        {'type': type(item.message).__name__, 'message': str(item.message)}
        for item in caught
    ] == expected['warnings']
    assert not [
        event
        for event in expected['trace']
        if event['stage'] == 'normalize-enter'
    ]
