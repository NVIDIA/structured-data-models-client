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


def test_uploaded_tables_join_and_build_source_schema():
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

        context_setup = next(
            source for source in sources if source.startswith('session.sql(')
        )
        for statement in ast.parse(context_setup).body:
            sql_call = statement.value.func.value
            sql = eval(
                compile(ast.Expression(sql_call.args[0]), '<context>', 'eval'),
                {'DATA_LOCATION': 'main'},
            )
            # DuckDB cursors do not share temporary tables across connections.
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
        for marker in ('source = SnowflakeSource(', "schema['products']"):
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
    finally:
        connection.close()


def test_lowercase_namespace_matches_unquoted_snowflake_setup():
    cells = json.loads(NOTEBOOK.read_text())['cells']
    configuration = next(
        ''.join(cell['source'])
        for cell in cells
        if "DATABASE = '<database>'" in ''.join(cell['source'])
    )
    configuration = configuration.replace('<database>', 'example_db').replace(
        '<schema>', 'example_schema'
    )
    # Exercise the configuration without requiring Snowpark or a live session.
    assignments = ast.Module(
        body=[
            node
            for node in ast.parse(configuration).body
            if isinstance(node, ast.Assign)
        ],
        type_ignores=[],
    )
    namespace = {'get_active_session': lambda: object()}
    exec(compile(assignments, '<notebook>', 'exec'), namespace)
    assert namespace['DATABASE'] == 'EXAMPLE_DB'
    assert namespace['SCHEMA'] == 'EXAMPLE_SCHEMA'
    assert namespace['DATA_LOCATION'] == 'EXAMPLE_DB.EXAMPLE_SCHEMA'
