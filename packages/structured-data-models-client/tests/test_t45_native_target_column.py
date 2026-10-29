# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual target conversions retain their values, positions and native owner."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sdm_client._rust.native_target import NativeTargetColumn


@pytest.mark.parametrize(
    ('dtype', 'values', 'plain_dtype'),
    [
        ('int8[pyarrow]', [1, None, 3], np.dtype('int64')),
        ('uint64[pyarrow]', [0, None, 2**64 - 1], np.dtype('int64')),
        ('bool[pyarrow]', [True, None, False], np.dtype('bool')),
        ('Int16', [1, None, 3], np.dtype('int64')),
        ('boolean', [True, None, False], np.dtype('bool')),
    ],
)
def test_plain_and_filtered_targets_keep_distinct_actual_storage(
    dtype, values, plain_dtype
):
    source = pd.Series(values, dtype=dtype, index=[8, 4, 9], name='id')
    before = source.copy(deep=True)
    plain = NativeTargetColumn.column(source, drop_missing=True)
    filtered = NativeTargetColumn.column(source, drop_missing=False)
    assert plain.values.dtype == plain_dtype
    assert filtered.values.dtype == source.dtype
    assert plain.positions.tolist() == [0, 2]
    assert filtered.positions.tolist() == [0, 1, 2]
    assert plain.present.tolist() == filtered.present.tolist() == [True, False, True]
    assert plain.values.name is None and plain.values.index.name is None
    assert filtered.values.name is None and filtered.values.index.name is None
    assert plain.values.index.tolist() == [0, 1]
    actual = filtered.filtered(np.array([False, True, True]))
    assert actual.positions.tolist() == [2]
    assert actual.values.tolist() == [values[2]]
    assert actual.values.dtype == source.dtype
    if dtype == 'uint64[pyarrow]':
        assert plain.values.tolist() == [0, -1]
        assert actual.values.tolist() == [2**64 - 1]
    pd.testing.assert_series_equal(source, before)


def test_native_float_conversion_precedes_missing_and_later_window_selection():
    events = []

    class Observed(pd.Series):
        @property
        def _constructor(self):
            return Observed

        def astype(self, dtype, *args, **kwargs):
            events.append(('cast', dtype, self.tolist()))
            result = super().astype(dtype, *args, **kwargs)
            result.iloc[0] = 17
            return result

        def notna(self):
            events.append(('present', self.tolist()))
            return super().notna()

    source = Observed([1.0, float('nan'), 2.0], name='raw')
    actual = NativeTargetColumn.column(source, drop_missing=True)
    assert events[0][0:2] == ('cast', 'float32')
    assert events[0][2][0] == 1 and events[0][2][2] == 2
    assert events[1][0] == 'present' and events[1][1][0] == 17
    assert actual.values.dtype == np.dtype('float32')
    assert actual.positions.tolist() == [0, 2]
    final = actual.windowed(np.array([True, False]))
    assert final.values.tolist() == [17]
    assert final.positions.tolist() == [0]
    assert source.iloc[0] == 1


def test_failed_float_cast_is_reached_before_any_missing_or_window_read():
    failure = ValueError('cast happens before excluding the final row')
    events = []

    class Refusing(pd.Series):
        @property
        def _constructor(self):
            return Refusing

        def astype(self, dtype, *args, **kwargs):
            events.append(('cast', dtype))
            raise failure

        def notna(self):
            pytest.fail('a failed cast must not observe missingness')

    with pytest.raises(ValueError) as raised:
        NativeTargetColumn.column(Refusing([1.0, 2.0]), drop_missing=True)
    assert raised.value is failure
    assert events == [('cast', 'float32')]


@pytest.mark.parametrize('drop_missing', [False, True])
@pytest.mark.parametrize(
    'source',
    [
        pd.Series([1, 2], dtype='int16[pyarrow]'),
        pd.Series([True, False], dtype='bool[pyarrow]'),
        pd.Series(['a', 'b'], dtype='string[pyarrow]'),
        pd.Series(['a', 'b'], dtype='category'),
        pd.Series([1, 2], dtype='Int16'),
        pd.Series([1, 2], dtype='int16'),
    ],
)
def test_group_restoration_uses_the_actual_derived_owner(source, drop_missing):
    column = NativeTargetColumn.column(source, drop_missing=drop_missing)
    if not drop_missing:
        column = column.filtered(np.array([True, True]))
    column = column.windowed(np.array([True, True]))
    with warnings.catch_warnings(record=True) as native_notices:
        warnings.simplefilter('always')
        try:
            expected = column.values.groupby([0, 0]).agg(list)
        except Exception as error:
            native_error = error
        else:
            native_error = None
    with warnings.catch_warnings(record=True) as actual_notices:
        warnings.simplefilter('always')
        try:
            actual = column.group_result([column.values.tolist()])
        except Exception as error:
            actual_error = error
        else:
            actual_error = None
    assert [(n.category, str(n.message)) for n in actual_notices] == [
        (n.category, str(n.message)) for n in native_notices
    ]
    if native_error is not None:
        assert type(actual_error) is type(native_error)
        assert str(actual_error) == str(native_error)
    else:
        assert actual_error is None
        pd.testing.assert_series_equal(
            pd.Series(actual), expected.reset_index(drop=True), check_names=False
        )


@pytest.mark.parametrize('drop_missing', [False, True])
@pytest.mark.parametrize('values', [[], [None, None]])
def test_empty_targets_preserve_actual_native_empty_conversion(values, drop_missing):
    column = NativeTargetColumn.column(
        pd.Series(values, dtype='Int64'), drop_missing=drop_missing
    )
    if not drop_missing:
        column = column.filtered(np.ones(len(values), dtype=bool))
    assert column.values.empty
    assert column.positions.tolist() == []
    assert len(column.windowed(np.array([], dtype=bool)).values) == 0
