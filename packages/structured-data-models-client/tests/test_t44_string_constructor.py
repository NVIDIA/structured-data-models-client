# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declared native strings reach the unchanged Unicode public journeys."""

from __future__ import annotations

import warnings

import pandas as pd
import pyarrow as pa
import pytest
from test_t42_frames import PATHS, describe, projection

from sdm_client import Schema, StructuredDataError
from sdm_client._rust.frames import _convert, _native_dtype


@pytest.fixture(autouse=True)
def native_string_options():
    # Match the successful frozen Unicode journeys. With inferred strings,
    # pandas 3 can choose Arrow while the old date hint casts to str, which
    # rejects these points before either constructor is being compared.
    with pd.option_context('future.infer_string', False):
        yield


@pytest.mark.parametrize(
    'values',
    [
        ['ordinary', '', None, 'a\0b', '😺'],
        ['\ud800', None, '\udfff', '', '\ud83d\ude3a', '😺'],
        ['\ud83d\ude3a', '😺'],
        [f'{number}\ud800' for number in range(65)],
    ],
)
def test_declared_string_constructor_keeps_schema_validity_and_native_owner(
    monkeypatch, values
):
    frame = pd.DataFrame(
        {
            'before': range(len(values)),
            'value': pd.Series(values, dtype=pd.StringDtype(storage='python')),
            'after': range(len(values)),
        }
    )
    original = frame.copy(deep=True)
    old, old_notices = describe(monkeypatch, {'rows': frame}, core=False)
    new, new_notices = describe(monkeypatch, {'rows': frame}, core=True)
    assert projection(new) == projection(old)
    assert new_notices == old_notices
    assert new['rows'].data is frame
    assert new['rows']['value'].dtype == 'string'
    assert not new['rows']['value'].untyped
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize('variant', ['unique', 'repeat', 'null'])
def test_declared_string_keys_keep_pairs_scalars_and_native_diagnostics(
    monkeypatch, variant
):
    values = [f'key{number}' for number in range(98)] + ['\ud83d\ude3a', '😺']
    if variant == 'repeat':
        values[-1] = values[-2]
    elif variant == 'null':
        values[-1] = None
    frame = pd.DataFrame(
        {'user_id': pd.Series(values, dtype=pd.StringDtype(storage='python'))}
    )
    outcomes = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                schema = Schema.from_data(
                    {'users': frame}, primary_keys={'users': 'user_id'}
                )
            except StructuredDataError as error:
                outcome = ('error', error.code, str(error))
            else:
                outcome = ('schema', projection(schema))
                assert schema['users'].data is frame
            outcomes.append(
                (
                    outcome,
                    [
                        (type(item.message), str(item.message))
                        for item in caught
                    ],
                )
            )
    assert outcomes[1] == outcomes[0]
    assert outcomes[0][0][0] == ('error' if variant == 'null' else 'schema')


def test_constructor_export_retains_original_string_dtype_and_lossless_rows():
    values = ['\ud800', None, '\ud83d\ude3a', '😺', '']
    frame = pd.DataFrame(
        {'value': pd.Series(values, dtype=pd.StringDtype(storage='python'))}
    )
    rows, types = _convert('rows', frame)
    assert _native_dtype(types['value']) is frame['value'].dtype
    assert rows.column_names == ['value']
    records = rows.column(0).to_pylist()
    assert [record['kind'] for record in records] == [6, 1, 6, 6, 6]
    assert [record['text'] for record in records] == [
        None if value is None else str.encode(value, 'utf-8', 'surrogatepass')
        for value in values
    ]
    assert types['value'].descriptor == {
        'category': False,
        'meaning': {'dtype': 'string', 'untyped': False},
    }


def test_ordinary_string_export_keeps_existing_arrow_representation():
    series = pd.Series(
        ['ordinary', None, '😺'], dtype=pd.StringDtype(storage='python')
    )
    rows, types = _convert('rows', pd.DataFrame({'value': series}))
    assert rows.column(0).equals(
        pa.chunked_array([pa.array(series, from_pandas=True)])
    )
    assert types['value'] is series.dtype


def test_lossless_constructor_export_does_not_call_scalar_conversion_overrides():
    class NativeString(str):
        def __str__(self):
            raise AssertionError('constructor displayed a native string')

        def __iter__(self):
            raise AssertionError(
                'constructor called a scalar iterator override'
            )

        def encode(self, *args, **kwargs):
            raise AssertionError('constructor called a scalar encoder override')

    value = NativeString('\ud800')
    frame = pd.DataFrame(
        {'value': pd.Series([value], dtype=pd.StringDtype(storage='python'))}
    )
    rows, _ = _convert('rows', frame)
    assert rows.column(0).to_pylist()[0]['text'] == b'\xed\xa0\x80'
