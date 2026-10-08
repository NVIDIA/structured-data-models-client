# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import os
from decimal import Decimal

import pandas as pd
import pytest
from predictive_query.schema.typing import Dtype, Stype

try:
    from sdm_relational_engine.rfm.backend.postgres import (
        PostgresTable,
        dialect,
    )
    from sdm_relational_engine.rfm.backend.postgres.table import (
        _sanitize_postgres_frame,
    )
except ImportError:
    if os.getenv('SDM_REQUIRE_POSTGRES'):
        raise
    pytest.skip("'postgres' extension not installed", allow_module_level=True)


@pytest.mark.parametrize(
    'sql_type,udt,scale,expected',
    [
        ('bigint', 'int8', None, Dtype.int),
        ('numeric', 'numeric', 0, Dtype.int),
        ('numeric', 'numeric', 2, Dtype.float),
        ('timestamp with time zone', 'timestamptz', None, Dtype.date),
        ('uuid', 'uuid', None, Dtype.string),
        ('array', '_text', None, Dtype.stringlist),
        ('array', '_int8', None, Dtype.intlist),
    ],
)
def test_postgres_type_mapping(sql_type, udt, scale, expected):
    assert PostgresTable._to_dtype(sql_type, udt, scale) == expected


def test_chunk_rows_keeps_every_value():
    rows = ['a' * 20, 'b' * 20, 'c' * 20]
    chunks = dialect.chunk_rows(rows, max_bytes=30)
    assert [value for chunk in chunks for value in chunk] == rows
    assert len(chunks) == 3


def test_time_serialization_preserves_precision_and_timezone():
    timestamp = pd.Timestamp('2025-01-02T03:04:05.123456-08:00')
    assert dialect.serialize_time(timestamp) == (
        '2025-01-02T03:04:05.123456-08:00'
    )
    assert dialect.serialize_time(pd.NaT) is None


def test_a_utc_bound_names_its_instant():
    r"""A naive bound means UTC: written with ``+00:00``, a ``timestamptz``
    column reads it the same under any session ``TimeZone``.
    """
    naive = pd.Timestamp('2025-01-02T03:04:05.123456')
    assert dialect.serialize_time(naive, utc=True) == (
        '2025-01-02T03:04:05.123456+00:00'
    )
    zoned = pd.Timestamp('2025-01-02T03:04:05-08:00')
    assert dialect.serialize_time(zoned, utc=True) == (
        '2025-01-02T11:04:05+00:00'
    )
    assert dialect.serialize_time(naive) == '2025-01-02T03:04:05.123456'
    assert dialect.serialize_time(pd.NaT, utc=True) is None


def test_sanitize_preserves_nullable_postgres_scalar_types():
    frame = pd.DataFrame(
        {
            'integer': [1, None, 3],
            'numeric_integer': [Decimal('1'), None, Decimal('3')],
            'numeric_float': [Decimal('1.25'), None, Decimal('3.75')],
            'flag': [True, None, False],
            'unchanged': ['a', None, 'c'],
        }
    )

    actual = _sanitize_postgres_frame(
        df=frame,
        dtype_dict={
            'integer': Dtype.int,
            'numeric_integer': Dtype.int,
            'numeric_float': Dtype.float,
            'flag': Dtype.bool,
            'unchanged': Dtype.string,
        },
        stype_dict={
            'integer': Stype.ID,
            'numeric_integer': Stype.numerical,
            'numeric_float': Stype.numerical,
            'flag': Stype.categorical,
            'unchanged': Stype.categorical,
        },
    )

    assert str(actual['integer'].dtype) == 'Int64'
    assert str(actual['numeric_integer'].dtype) == 'Int64'
    assert str(actual['numeric_float'].dtype) == 'Float64'
    assert str(actual['flag'].dtype) == 'boolean'
    assert actual['integer'].tolist() == [1, pd.NA, 3]
    assert actual['numeric_integer'].tolist() == [1, pd.NA, 3]
    assert actual['numeric_float'].tolist() == [1.25, pd.NA, 3.75]
    assert actual['flag'].tolist() == [True, pd.NA, False]
    assert actual['unchanged'].iloc[[0, 2]].tolist() == ['a', 'c']
    assert pd.isna(actual['unchanged'].iloc[1])


def _zone(**offset: int) -> datetime.timezone:
    return datetime.timezone(datetime.timedelta(**offset))


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        (
            '12:00:00+05:30',
            datetime.time(12, tzinfo=_zone(hours=5, minutes=30)),
        ),
        ('12:00:00+00', datetime.time(12, tzinfo=_zone())),
        ('23:59:59-12', datetime.time(23, 59, 59, tzinfo=_zone(hours=-12))),
        (
            '12:00:00.5-03:30',
            datetime.time(12, 0, 0, 500_000, _zone(hours=-3, minutes=-30)),
        ),
        (
            '00:00:00.000001+14',
            datetime.time(0, 0, 0, 1, _zone(hours=14)),
        ),
        (
            '12:00:00+05:53:28',
            datetime.time(12, tzinfo=_zone(hours=5, minutes=53, seconds=28)),
        ),
        (
            '12:00:00-05:53:28',
            datetime.time(12, tzinfo=_zone(hours=-5, minutes=-53, seconds=-28)),
        ),
        # PostgreSQL's end of day has no Python time; it keeps its text.
        ('24:00:00+00', '24:00:00+00'),
        ('24:00:00-15:59', '24:00:00-15:59'),
    ],
)
def test_a_timetz_is_read_back_as_a_time_with_its_zone(
    text: str, expected: datetime.time | str
) -> None:
    from sdm_relational_engine.rfm.backend.postgres.table import _timetz

    actual = _timetz(text)

    assert actual == expected
    assert type(actual) is type(expected)
    if isinstance(actual, datetime.time):
        # Aware times compare as instants; the zone itself must match too.
        assert actual.tzinfo == expected.tzinfo  # type: ignore[union-attr]
