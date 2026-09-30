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


def test_chunk_rows_keeps_every_value():
    sampler: Any = object.__new__(PostgresSampler)
    rows = ['a' * 20, 'b' * 20, 'c' * 20]
    chunks = sampler._chunk_rows(rows, max_bytes=30)
    assert [value for chunk in chunks for value in chunk] == rows
    assert len(chunks) == 3


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
