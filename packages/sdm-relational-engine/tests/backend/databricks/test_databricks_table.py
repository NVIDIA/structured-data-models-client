# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from predictive_query.schema.typing import Dtype, Stype
from sdm_relational_engine.rfm.backend.local import LocalTable
from sdm_relational_engine.rfm.base import LocalExpression

try:
    from sdm_relational_engine.rfm.backend.databricks import (
        Connection,
        DatabricksTable,
    )
except ImportError:
    pytest.skip("'databricks' extension not installed", allow_module_level=True)


def test_to_dtype() -> None:
    # https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-datatypes
    cases = [
        ('tinyint', Dtype.int),
        ('smallint', Dtype.int),
        ('int', Dtype.int),
        ('integer', Dtype.int),
        ('bigint', Dtype.int),
        ('long', Dtype.int),
        ('decimal(10,0)', Dtype.int),
        ('decimal(10,2)', Dtype.float),
        ('numeric(20,4)', Dtype.float),
        ('float', Dtype.float),
        ('real', Dtype.float),
        ('double', Dtype.float),
        ('boolean', Dtype.bool),
        ('string', Dtype.string),
        ('varchar(100)', Dtype.string),
        ('char(10)', Dtype.string),
        ('binary', Dtype.binary),
        ('date', Dtype.date),
        ('timestamp', Dtype.date),
        ('timestamp_ntz', Dtype.date),
        ('timestamp_ltz', Dtype.date),
        ('array<int>', Dtype.intlist),
        ('array<bigint>', Dtype.intlist),
        ('array<double>', Dtype.floatlist),
        ('array<string>', Dtype.stringlist),
        ('array<array<int>>', Dtype.unsupported),
        ('map<string,int>', Dtype.unsupported),
        ('struct<a:int>', Dtype.unsupported),
        ('variant', Dtype.unsupported),
        ('interval day to second', Dtype.unsupported),
        ('void', Dtype.unsupported),
    ]
    for dtype, expected in cases:
        assert DatabricksTable._to_dtype(dtype) == expected, dtype

    assert DatabricksTable._to_dtype(None) is None
    assert DatabricksTable._to_dtype('UNKNOWN_TYPE') is None


def test_invalid_name(
    connection: Connection,
    catalog: str,
    schema: str,
) -> None:
    with pytest.raises(ValueError, match='does not exist'):
        DatabricksTable(
            connection, name='__no_such_table__', catalog=catalog, schema=schema
        )


def test_quoted_source_name(
    connection: Connection,
    catalog: str,
    schema: str,
) -> None:
    table = DatabricksTable(
        connection, name='customers', catalog=catalog, schema=schema
    )
    assert table.source_name == f'{catalog}.{schema}.customers'
    assert table._quoted_source_name == f'`{catalog}`.`{schema}`.`customers`'


def test_customers(
    connection: Connection,
    catalog: str,
    schema: str,
) -> None:
    table = DatabricksTable(
        connection,
        name='customers',
        catalog=catalog,
        schema=schema,
        columns=[
            'customer_id',
            'segment',
            'payment_terms_days',
            'onboarded_year',
        ],
    )

    # The ERP dataset declares no constraints, so no primary key is set until
    # it is inferred heuristically:
    assert table.primary_key is None
    table.infer_primary_key(verbose=False)
    assert table.primary_key is not None
    assert table.primary_key.name == 'customer_id'

    assert table['customer_id'].dtype == Dtype.string
    assert table['segment'].dtype == Dtype.string
    assert table['payment_terms_days'].dtype == Dtype.int
    assert table['onboarded_year'].dtype == Dtype.int


def test_dtypes(
    connection: Connection,
    catalog: str,
    schema: str,
) -> None:
    table = DatabricksTable(
        connection,
        name='order_lines',
        catalog=catalog,
        schema=schema,
        columns=['order_date', 'quantity', 'unit_price_usd', 'order_id'],
    )
    assert table['order_date'].dtype == Dtype.date
    assert table['order_date'].stype == Stype.timestamp
    assert table['quantity'].dtype == Dtype.int
    assert table['unit_price_usd'].dtype == Dtype.float
    assert table['order_id'].dtype == Dtype.string


def test_num_rows(
    connection: Connection,
    catalog: str,
    schema: str,
) -> None:
    table = DatabricksTable(
        connection, name='products', catalog=catalog, schema=schema
    )

    with connection.cursor() as cursor:
        cursor.execute(f'SELECT COUNT(*) FROM `{catalog}`.`{schema}`.products')
        row = cursor.fetchone()
        assert row is not None
        expected = row[0]

    assert expected > 0
    assert table._num_rows == expected


def test_column_expression(
    connection: Connection,
    catalog: str,
    schema: str,
) -> None:
    table = DatabricksTable(
        connection,
        name='ol',
        source_name='order_lines',
        catalog=catalog,
        schema=schema,
        columns=[
            dict(name='double_qty', expr='2 * quantity'),
            dict(name='line_total', expr='unit_price_usd * quantity'),
            dict(name='order_day', expr='order_date'),
        ],
    )
    table.infer_metadata(verbose=False)

    # No source information was queried:
    assert '_source_column_dict' not in table.__dict__
    assert '_source_sample_df' not in table.__dict__

    assert table.source_name == f'{catalog}.{schema}.order_lines'

    column = table['double_qty']
    assert not column.is_source
    assert isinstance(column.expr, LocalExpression)
    assert column.expr.value == '2 * quantity'
    assert column.dtype == Dtype.int

    column = table['line_total']
    assert column.dtype == Dtype.float

    column = table['order_day']
    assert column.dtype == Dtype.date
    assert column.stype == Stype.timestamp


def test_dtype_parity_with_local(
    connection: Connection,
    catalog: str,
    schema: str,
) -> None:
    table = DatabricksTable(
        connection,
        name='order_lines',
        catalog=catalog,
        schema=schema,
        columns=['order_id', 'quantity', 'unit_price_usd', 'order_date'],
    )

    local_table = LocalTable(table._source_sample_df, name='order_lines')
    for column in table.columns:
        local_column = local_table[column.name]
        assert column.dtype == local_column.dtype
        assert column.stype == local_column.stype
