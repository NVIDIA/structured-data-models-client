# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed source scalar construction consumes the core-selected meaning."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pyarrow as pa
import pytest

from sdm_client._rust.native_source import scalar_column


def _meaning(kind, physical=None):
    result = {'meaning': kind}
    if physical is not None:
        result['physical'] = {'type': physical}
    return result


@pytest.mark.parametrize(
    ('physical', 'values', 'dtype'),
    [
        ('int64', [2**53 + 1, None, -(2**63)], 'Int64'),
        ('float64', [1.5, None, float('nan')], 'Float64'),
        ('bool', [True, None, False], 'boolean'),
    ],
)
def test_raw_cell_scalars_keep_exact_wide_values_validity_and_original_owner(
    physical, values, dtype
):
    original = pd.Series(values, dtype=object, index=[8, 2, 4], name='actual')
    before = original.copy(deep=True)
    actual = scalar_column(original, _meaning('nullable', physical))
    pd.testing.assert_extension_array_equal(
        actual, pd.array(values, dtype=dtype)
    )
    pd.testing.assert_series_equal(original, before)


def test_native_stringification_retains_null_dict_list_and_actual_user_call_order():
    events = []

    class Observed:
        def __str__(self):
            events.append('render')
            return 'changed'

    original = pd.Series(
        [None, pd.NA, {'x': [1, None]}, [True, 'a'], Observed()], dtype=object
    )
    actual = scalar_column(original, _meaning('nullable', 'utf8'))
    assert events == ['render']
    assert actual[:2].isna().all()
    assert actual[2:].tolist() == [
        '{"x":[1,null]}',
        '[true,"a"]',
        'changed',
    ]
    assert type(actual.dtype) is pd.StringDtype
    assert type(original.iloc[4]) is Observed


def test_unchanged_source_is_the_exact_native_owner_without_new_scalar_reads():
    class Unread:
        def __str__(self):
            pytest.fail('an original source meaning must not stringify values')

    original = pd.Series([Unread()], dtype=object)
    assert scalar_column(original, _meaning('original')) is original


@pytest.mark.parametrize(
    ('physical', 'dtype'),
    [('int64', pa.int64()), ('float64', pa.float64()), ('utf8', pa.string())],
)
def test_decimal_cast_keeps_the_actual_native_result_and_complete_index(
    physical, dtype
):
    original = pd.Series(
        [Decimal('1'), None, Decimal('-3')],
        dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
        index=[7, 0, 6],
        name='key',
    )
    expected = original.astype(pd.ArrowDtype(dtype))
    actual = scalar_column(original, _meaning('decimal', physical))
    pd.testing.assert_series_equal(actual, expected)
    assert actual.dtype == pd.ArrowDtype(dtype)
    assert original.dtype == pd.ArrowDtype(pa.decimal128(38, 0))


def test_actual_decimal_cast_failure_is_retained_for_the_core_fallback_decision():
    failure = ArithmeticError('actual failed cast')
    events = []

    class Observed(pd.Series):
        def astype(self, dtype, *args, **kwargs):
            events.append(dtype)
            raise failure

    original = Observed([Decimal('1')], dtype=object)
    with pytest.raises(ArithmeticError) as raised:
        scalar_column(original, _meaning('decimal', 'int64'))
    assert raised.value is failure
    assert events == [pd.ArrowDtype(pa.int64())]
    assert original.iloc[0] == Decimal('1')


@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
@pytest.mark.parametrize('zone', [None, 'America/New_York'])
def test_core_selected_timestamp_constructs_actual_naive_nanosecond_series(
    unit, zone
):
    values = pd.Series(
        pa.array([0, None, 123], type=pa.timestamp(unit, tz=zone)),
        dtype=pd.ArrowDtype(pa.timestamp(unit, tz=zone)),
        index=[9, 3, 7],
        name='observed_time',
    )
    expected = pd.Series(
        values.to_numpy(), index=values.index, name=values.name
    )
    expected = pd.to_datetime(expected, unit='ns', errors='coerce')
    if isinstance(expected.dtype, pd.DatetimeTZDtype):
        expected = expected.dt.tz_convert('UTC').dt.tz_localize(None)
    expected = expected.astype('datetime64[ns]')
    actual = scalar_column(
        values,
        {
            'meaning': 'nullable',
            'physical': {'type': 'timestamp', 'unit': 'ns', 'zone': None},
        },
    )
    pd.testing.assert_series_equal(actual, expected)
    assert actual.dtype == 'datetime64[ns]'
    assert values.dtype == pd.ArrowDtype(pa.timestamp(unit, tz=zone))


def test_timestamp_boxing_failure_retains_actual_native_owner():
    failure = ValueError('actual source timestamp boxing failed')

    class Observed(pd.Series):
        def to_numpy(self, *args, **kwargs):
            raise failure

    values = Observed([0], dtype=pd.ArrowDtype(pa.timestamp('us')))
    with pytest.raises(ValueError) as raised:
        scalar_column(
            values,
            {
                'meaning': 'nullable',
                'physical': {'type': 'timestamp', 'unit': 'ns', 'zone': None},
            },
        )
    assert raised.value is failure


def test_core_selected_boxed_datetime_values_keep_index_nulls_and_utc():
    from datetime import datetime, timezone

    values = pd.Series(
        [datetime(2023, 1, 2, tzinfo=timezone.utc), None],
        dtype=object,
        index=[8, 2],
        name='actual',
    )
    actual = scalar_column(
        values,
        {
            'meaning': 'nullable',
            'physical': {'type': 'timestamp', 'unit': 'ns', 'zone': None},
        },
    )
    pd.testing.assert_series_equal(
        actual,
        pd.Series(
            [pd.Timestamp('2023-01-02'), pd.NaT],
            dtype='datetime64[ns]',
            index=[8, 2],
            name='actual',
        ),
    )
