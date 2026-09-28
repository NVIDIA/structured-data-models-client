# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest
from predictive_query.schema.typing import Dtype, Stype

try:
    from sdm_relational_engine.rfm.backend.postgres import (
        PostgresSampler,
        PostgresTable,
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


def test_json_cte_binds_payload_and_preserves_ordinality():
    sql = PostgresSampler._json_cte(
        'uuid', ('e', 's'), 'timestamp with time zone'
    )
    assert '%s::jsonb' in sql
    assert 'WITH ORDINALITY' in sql
    assert '::uuid AS __KUMO_ID__' in sql
    assert '::timestamp with time zone AS __KUMO_END_TIME__' in sql
    assert '::timestamp with time zone AS __KUMO_START_TIME__' in sql


def test_projection_order_follows_table_order():
    sampler = object.__new__(PostgresSampler)
    sampler._table_column_proj_dict = {
        'users': {'id': '"id"', 'created': '"created"', 'name': '"name"'}
    }
    assert sampler._projections('users', {'name', 'id'}) == ['"id"', '"name"']


def test_chunk_rows_keeps_every_value():
    sampler: Any = object.__new__(PostgresSampler)
    rows = ['a' * 20, 'b' * 20, 'c' * 20]
    chunks = sampler._chunk_rows(rows, max_bytes=30)
    assert [value for chunk in chunks for value in chunk] == rows
    assert len(chunks) == 3


def test_postgres_seed_is_stable_and_in_range():
    values = [
        PostgresSampler._postgres_seed(seed)
        for seed in (-(10**30), -1, 0, 1, 42, 10**30)
    ]
    assert all(-1.0 <= value <= 1.0 for value in values)
    assert PostgresSampler._postgres_seed(42) == values[4]
    assert PostgresSampler._postgres_seed(42) != (
        PostgresSampler._postgres_seed(2_000_043)
    )


def test_time_serialization_preserves_precision_and_timezone():
    timestamp = pd.Timestamp('2025-01-02T03:04:05.123456-08:00')
    assert PostgresSampler._serialize_time(timestamp) == (
        '2025-01-02T03:04:05.123456-08:00'
    )
    assert PostgresSampler._serialize_time(pd.NaT) is None


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
