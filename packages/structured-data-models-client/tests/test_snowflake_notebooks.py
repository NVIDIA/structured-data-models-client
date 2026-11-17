# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the Snowflake example's table layout and source-backed schema offline."""

import ast
import datetime
import json
import re
from pathlib import Path

import pytest
from sdm_connectors import DuckDBSource

from sdm_client import Relationship, Schema, SourceTable

NOTEBOOK = (
    Path(__file__).parents[3]
    / 'examples/snowflake/kumo_relational_quickstart.ipynb'
)


def test_uploaded_tables_join_and_omit_future_information():
    duckdb = pytest.importorskip('duckdb')
    pytest.importorskip('sdm_relational_engine')
    cells = json.loads(NOTEBOOK.read_text())['cells']
    sources = [''.join(cell['source']) for cell in cells]
    setup = next(
        source for source in sources if 'CREATE TABLE IF NOT EXISTS' in source
    )
    connection = duckdb.connect()
    try:
        for statement in re.findall(
            r'CREATE TABLE IF NOT EXISTS .*?;', setup, re.DOTALL
        ):
            statement = statement.replace('<database>.<schema>.', '')
            connection.execute(statement.replace('TIMESTAMP_NTZ', 'TIMESTAMP'))

        for table in (
            'Customer',
            'Product',
            'SalesOrderHeader',
            'SalesOrderDetail',
        ):
            columns = connection.execute(f'DESCRIBE "{table}"').fetchall()
            row = {}
            for name, kind, *_ in columns:
                if kind == 'TIMESTAMP':
                    row[name] = datetime.datetime(2024, 1, 1)
                elif kind in ('BIGINT', 'INTEGER', 'FLOAT', 'DOUBLE'):
                    row[name] = 1
                else:
                    row[name] = 'example'
            placeholders = ', '.join('?' for _ in row)
            connection.execute(
                f'INSERT INTO "{table}" VALUES ({placeholders})',
                list(row.values()),
            )

        view = next(
            source for source in sources if source.startswith('session.sql(')
        )
        sql_call = ast.parse(view).body[0].value.func.value
        sql = eval(
            compile(ast.Expression(sql_call.args[0]), '<view>', 'eval'),
            {'DATA_LOCATION': 'main'},
        )
        # DuckDB cursors do not share temporary views across connections.
        connection.execute(sql.replace('TEMPORARY ', ''))
        joined = connection.execute(
            'SELECT "SalesOrderDetailID", "CustomerID", "OrderDate" '
            'FROM "SDM_SalesOrderDetail"'
        ).fetchall()
        assert joined == [(1, 1, datetime.datetime(2024, 1, 1))]

        source = DuckDBSource(connection)
        namespace = {
            'session': object(),
            'DATABASE': 'DB',
            'SCHEMA': 'PUBLIC',
            'SnowflakeSource': lambda session, *, database, schema: source,
            'Schema': Schema,
            'Relationship': Relationship,
        }
        for marker in ('source = SnowflakeSource(', 'leave_out = {'):
            code = next(item for item in sources if item.startswith(marker))
            exec(compile(code, '<notebook>', 'exec'), namespace)
        schema = namespace['schema']
        schema.check_consistent()
        assert schema.source is source
        assert all(
            isinstance(table, SourceTable) for table in schema.tables.values()
        )
        assert set(schema.tables) == {
            'customers',
            'products',
            'sales_order_headers',
            'sales_order_details',
        }
        assert schema['sales_order_details'].time_column == 'OrderDate'
        assert schema['products']['Name'].stype == 'categorical'
        assert schema['sales_order_details']['OrderQty'].stype == 'numerical'
        for table in schema.tables.values():
            assert 'ModifiedDate' not in table.columns
        assert 'ShipDate' not in schema['sales_order_headers'].columns
        assert not {'SellStartDate', 'SellEndDate', 'DiscontinuedDate'} & set(
            schema['products'].columns
        )
    finally:
        connection.close()
