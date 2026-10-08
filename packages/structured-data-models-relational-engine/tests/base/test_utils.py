# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from predictive_query.schema.typing import Dtype
from sdm_relational_engine.rfm.base import SourceColumn
from sdm_relational_engine.rfm.base.utils import (
    Timestamp,
    random_state,
    sample_order_terms,
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


def _source_columns(dtypes: dict[str, Dtype | None]) -> list[SourceColumn]:
    return [
        SourceColumn(
            name=name,
            dtype=dtype,
            is_primary_key=False,
            is_unique_key=False,
            is_nullable=True,
        )
        for name, dtype in dtypes.items()
    ]


def test_sample_order_terms_hash_a_declared_key_alone() -> None:
    columns = _source_columns(
        {'s': Dtype.string, 'id': Dtype.int, 'v': Dtype.floatlist}
    )

    terms = sample_order_terms(columns, 'id', lambda name: name, 'H')

    # The key may not be enforced, so ties are ordered by the other scalars:
    assert terms == ['H(id)', 'id', 's']


def test_sample_order_terms_hash_a_few_cheap_columns_without_a_key() -> None:
    dtypes: dict[str, Dtype | None] = {f's{i}': Dtype.string for i in range(4)}
    dtypes |= {f'n{i}': Dtype.int for i in range(6)}
    dtypes |= {'v': Dtype.intlist, 'x': Dtype.unsupported, 'u': None}

    terms = sample_order_terms(
        _source_columns(dtypes), None, lambda name: name, 'H'
    )

    # Eight at most, numbers before strings, in source order; every scalar
    # orders the rows, and lists, unsupported and unknown types do neither:
    scalars = [f's{i}' for i in range(4)] + [f'n{i}' for i in range(6)]
    assert terms == ['H(s0, s1, n0, n1, n2, n3, n4, n5)', *scalars]


def test_sample_order_terms_of_a_table_without_scalars_are_empty() -> None:
    columns = _source_columns({'v': Dtype.floatlist, 'x': Dtype.unsupported})

    assert sample_order_terms(columns, None, lambda name: name, 'H') == []
