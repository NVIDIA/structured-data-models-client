# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sdm_relational_engine.rfm.base.utils import (
    Timestamp,
    random_state,
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


@pytest.mark.parametrize('seed', [0, 1, 42, 2**31, 2**32 - 1])
def test_random_state_of_a_32_bit_seed_is_the_seed_itself(seed: int) -> None:
    expected = np.random.RandomState(seed).randint(0, 2**31, size=8)
    assert np.array_equal(
        random_state(seed).randint(0, 2**31, size=8), expected
    )


def test_random_state_of_a_larger_seed_uses_its_32_bit_words() -> None:
    seed = 2**32 + 7 + (3 << 64)
    words = np.array([7, 1, 3], dtype=np.uint32)
    expected = np.random.RandomState(words).randint(0, 2**31, size=8)
    assert np.array_equal(
        random_state(seed).randint(0, 2**31, size=8), expected
    )
    # Distinct seeds past 2**32 give distinct streams:
    assert not np.array_equal(
        random_state(2**32).randint(0, 2**31, size=8),
        random_state(2**33).randint(0, 2**31, size=8),
    )


def test_random_state_without_a_seed_is_unseeded() -> None:
    assert isinstance(random_state(None), np.random.RandomState)
