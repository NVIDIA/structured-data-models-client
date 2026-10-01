# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from predictive_query.schema.typing import Dtype

try:
    from sdm_relational_engine.rfm.backend.duckdb import DuckDBTable
except ImportError:
    pytest.skip("'duckdb' extension not installed", allow_module_level=True)


def test_to_dtype_scalars() -> None:
    # https://duckdb.org/docs/stable/sql/data_types/overview
    cases = [
        ('INTEGER', Dtype.int),
        ('BIGINT', Dtype.int),
        ('HUGEINT', Dtype.int),
        ('UINTEGER', Dtype.int),
        ('DOUBLE', Dtype.float),
        ('REAL', Dtype.float),
        ('DECIMAL(10,2)', Dtype.float),
        ('DECIMAL(10,0)', Dtype.int),
        ('VARCHAR', Dtype.string),
        ('BOOLEAN', Dtype.bool),
        ('DATE', Dtype.date),
        ('TIMESTAMP', Dtype.date),
        ('TIME', Dtype.time),
        ('BLOB', Dtype.binary),
    ]
    for dtype, expected in cases:
        assert DuckDBTable._to_dtype(dtype) == expected, dtype

    assert DuckDBTable._to_dtype(None) is None
    assert DuckDBTable._to_dtype('UNKNOWN_TYPE') is None


def test_to_dtype_lists_are_not_claimed_by_their_element_type() -> None:
    # DuckDB spells a list as '<element>[]', and every scalar test matches on
    # a substring, so the list branch used to be unreachable for all but one
    # input: 'INTEGER[]' came back as a scalar `int`, 'DOUBLE[]' as `float`.
    # The sibling Databricks mapping has always returned the list dtypes here
    # (see test_databricks_table.py), so this pins the two to one answer.
    cases = [
        ('INTEGER[]', Dtype.intlist),
        ('BIGINT[]', Dtype.intlist),
        ('UINTEGER[]', Dtype.intlist),
        ('DECIMAL(10,0)[]', Dtype.intlist),
        ('DOUBLE[]', Dtype.floatlist),
        ('DECIMAL(10,2)[]', Dtype.floatlist),
        ('VARCHAR[]', Dtype.stringlist),
    ]
    for dtype, expected in cases:
        assert DuckDBTable._to_dtype(dtype) == expected, dtype


def test_to_dtype_reports_a_list_it_cannot_represent_as_unsupported() -> None:
    # There is no boolean, temporal or nested list dtype on the wire, and
    # answering with the element's scalar dtype would describe the column as
    # something the model can read when it cannot.
    for dtype in ('BOOLEAN[]', 'TIMESTAMP[]', 'INTEGER[][]'):
        assert DuckDBTable._to_dtype(dtype) == Dtype.unsupported, dtype

    assert DuckDBTable._to_dtype('UNKNOWN_TYPE[]') is None
