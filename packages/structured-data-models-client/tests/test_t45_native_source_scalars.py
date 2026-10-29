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
    expected = pd.Series(
        pd.array(values, dtype=dtype), index=original.index, name=original.name
    )
    pd.testing.assert_series_equal(actual, expected)
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
    assert actual.iloc[:2].isna().all()
    assert actual.iloc[2:].tolist() == [
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
