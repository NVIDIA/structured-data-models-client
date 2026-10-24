# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from datetime import timedelta, timezone

import pandas as pd
import pytest
from sdm_relational_engine.rfm.base.utils import (
    Timestamp,
    shift_time,
    to_datetime,
)


def test_timestamp() -> None:
    assert Timestamp('2026-01-01') == pd.Timestamp('2026-01-01')

    with pytest.raises(TypeError, match='got NaT'):
        Timestamp(None)


def test_timestamp_normalizes_timezone() -> None:
    out = Timestamp('2026-01-01 12:00:00+00:00')
    assert out.tzinfo is None
    assert out == pd.Timestamp('2026-01-01 12:00:00')

    out = Timestamp(pd.Timestamp('2026-01-01 12:00:00+05:00'))
    assert out.tzinfo is None
    assert out == pd.Timestamp('2026-01-01 07:00:00')


def test_to_datetime_normalizes_timezone() -> None:
    ser = pd.Series(pd.to_datetime(['2026-01-01 12:00:00']))
    assert to_datetime(ser).iloc[0] == pd.Timestamp('2026-01-01 12:00:00')

    ser = pd.Series(pd.to_datetime(['2026-01-01 12:00:00+05:00']))
    out = to_datetime(ser)
    assert out.dtype == 'datetime64[ns]'
    assert out.iloc[0] == pd.Timestamp('2026-01-01 07:00:00')
    assert out.iloc[0] == Timestamp(ser.iloc[0])

    ser = pd.Series(pd.to_datetime(['2026-01-01 17:00:00'])).dt.tz_localize(
        timezone(timedelta(hours=-8))
    )
    assert to_datetime(ser).iloc[0] == pd.Timestamp('2026-01-02 01:00:00')


def test_to_datetime_warns_about_coerced_values() -> None:
    # Regression: unparseable values became NaT with no warning and no count.
    ser = pd.Series(['2026-01-01', 'not a date', 'nope', None])
    with pytest.warns(UserWarning, match='Could not parse 2 of 4') as record:
        out = to_datetime(ser, 'TS')
    assert out.isna().tolist() == [False, True, True, True]
    assert "'TS'" in str(record[0].message)
    assert 'not a date' in str(record[0].message)


def test_to_datetime_is_quiet_without_a_column_name() -> None:
    # Candidate probing parses columns that are not expected to be dates.
    ser = pd.Series(['2026-01-01', 'not a date'])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert to_datetime(ser).isna().tolist() == [False, True]


def test_to_datetime_does_not_warn_about_already_null_values() -> None:
    ser = pd.Series(['2026-01-01', None, pd.NaT])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert to_datetime(ser, 'TS').isna().tolist() == [False, True, True]


@pytest.mark.parametrize(
    'offset',
    [
        pd.DateOffset(),
        pd.DateOffset(days=1),
        pd.DateOffset(months=1, days=-31),
        -pd.DateOffset(days=2) * 3,
    ],
)
def test_shift_time_moves_by_an_offset_that_moves(
    offset: pd.DateOffset,
) -> None:
    time = pd.Timestamp('2020-02-01')
    assert shift_time(time, offset) == time + offset


@pytest.mark.parametrize(
    'offset',
    [pd.DateOffset(0), pd.DateOffset(days=0), pd.DateOffset(days=1) * 0],
)
def test_shift_time_leaves_the_earliest_time_where_it_is(
    offset: pd.DateOffset,
) -> None:
    assert shift_time(pd.Timestamp.min, offset) == pd.Timestamp.min
