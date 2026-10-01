# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A semantic model, here a Databricks metric view, as a schema's source.

The connection is faked: it answers the metadata and preview queries a
metric view is read with, the way a SQL warehouse would. Sampling over a real
view is exercised against a workspace, not here: the samplers' SQL is
Databricks SQL, which nothing local runs.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import pandas as pd
import pytest

pytest.importorskip('sdm_relational_engine.relationallib')
pytest.importorskip('databricks.sql')
pa = pytest.importorskip('pyarrow')
pytest.importorskip('yaml')

from databricks.sql.client import Connection  # noqa: E402
from sdm_connectors import DatabricksSource, SemanticModel  # noqa: E402

from sdm_client import (  # noqa: E402
    Relationship,
    Schema,
    SourceTable,
    StructuredDataError,
)
from sdm_client.materializers import RelationalMaterializer  # noqa: E402

_CATALOG, _SCHEMA = 'cat', 'sch'

_VIEW = """
version: 0.1
source: cat.sch.order_lines
joins:
  - name: customers
    source: cat.sch.customers
    on: source.customer_id = customers.customer_id
  - name: products
    source: cat.sch.products
    on: products.product_id = source.product_id
dimensions:
  - name: line_id
    expr: source.line_id
  - name: ordered_at
    expr: source.ordered_at
  - name: quantity
    expr: source.quantity
  - name: discounted
    expr: source.discount > 0
  - name: segment
    expr: customers.segment
  - name: color
    expr: products.color
measures:
  - name: total_quantity
    expr: SUM(source.quantity)
"""

_TYPES = {
    'order_lines': {
        'line_id': 'INT',
        'customer_id': 'INT',
        'product_id': 'INT',
        'ordered_at': 'TIMESTAMP',
        'quantity': 'INT',
        'discount': 'DOUBLE',
    },
    'customers': {'customer_id': 'INT', 'segment': 'STRING'},
    'products': {'product_id': 'INT', 'color': 'STRING'},
}

_DATA = {
    'order_lines': pd.DataFrame(
        {
            'line_id': range(1, 9),
            'customer_id': [1, 1, 2, 2, 3, 3, 1, 2],
            'product_id': [10, 11, 10, 12, 11, 12, 12, 10],
            'ordered_at': pd.date_range('2024-01-01', periods=8, freq='7D'),
            'quantity': [1, 2, 1, 3, 1, 2, 4, 1],
            'discount': [0.0, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.1],
        }
    ),
    'customers': pd.DataFrame(
        {'customer_id': [1, 2, 3], 'segment': ['a', 'b', 'a']}
    ),
    'products': pd.DataFrame(
        {'product_id': [10, 11, 12], 'color': ['red', 'blue', 'red']}
    ),
}
_EXPRESSIONS = {
    'order_lines': {'discounted': lambda df: df['discount'] > 0},
    'customers': {},
    'products': {},
}

_DESCRIBE = [
    ('line_id', 'int', None),
    ('ordered_at', 'timestamp', None),
    ('quantity', 'int', None),
    ('discounted', 'boolean', None),
    ('segment', 'string', None),
    ('color', 'string', None),
    ('total_quantity', 'bigint measure', None),
    ('', '', None),
    ('# Detailed Table Information', '', None),
    ('Catalog', _CATALOG, None),
    ('Database', _SCHEMA, None),
    ('Type', 'METRIC_VIEW', None),
    ('View Text', _VIEW, None),
]


class _Cursor:
    def __init__(self) -> None:
        self._rows: list = []
        self._arrow: Any = None

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    @staticmethod
    def _bound(sql: str, parameters: Any, column: str) -> str | None:
        markers = re.findall(
            r'lower\((?:\w+\.)?(\w+)\) = lower\(\?\)'
            r'|(?:\w+\.)?(\w+) = lower\(\?\)'
            r'|(?:\w+\.)?(\w+) = \?',
            sql,
        )
        names = [a or b or c for a, b, c in markers]
        if column not in names or parameters is None:
            return None
        return str(parameters[names.index(column)]).lower()

    def execute(self, sql: str, parameters: Any = None) -> None:
        self._rows, self._arrow = [], None
        sql = sql.strip()
        if sql.upper().startswith('DESCRIBE TABLE EXTENDED'):
            self._rows = list(_DESCRIBE)
            return
        if 'information_schema.columns' in sql:
            schema = self._bound(sql, parameters, 'table_schema')
            table = self._bound(sql, parameters, 'table_name')
            types = _TYPES.get(table or '', {}) if schema == _SCHEMA else {}
            self._rows = [
                (name, dtype, 'YES', _CATALOG, _SCHEMA, table)
                for name, dtype in types.items()
            ]
            return
        if 'information_schema' in sql:
            return
        table_match = re.search(r'FROM `[^`]+`\.`[^`]+`\.`([^`]+)`', sql)
        assert table_match is not None, f'unexpected SQL: {sql}'
        name = table_match.group(1)
        if sql.upper().startswith('SELECT COUNT(*)'):
            self._rows = [(len(_DATA[name]),)]
            return
        frame = _DATA[name]
        projection = sql.split(' FROM ')[0]
        if ' AS `' in projection:
            columns = re.findall(r'AS `((?:[^`]|``)+)`', projection)
        else:
            columns = re.findall(r'`((?:[^`]|``)+)`', projection)
        data = {}
        for column in columns:
            if column in frame.columns:
                data[column] = frame[column]
            else:
                data[column] = _EXPRESSIONS[name][column](frame)
        self._arrow = pa.Table.from_pandas(
            pd.DataFrame(data), preserve_index=False
        )

    def fetchall(self) -> list:
        return self._rows

    def fetchone(self) -> tuple:
        return self._rows[0]

    def fetchall_arrow(self) -> Any:
        return self._arrow


class _Warehouse(Connection):
    def __init__(self) -> None:
        pass

    def cursor(self) -> _Cursor:  # type: ignore[override]
        return _Cursor()

    def close(self) -> None:
        pass

    def __del__(self) -> None:
        pass


@pytest.fixture
def source() -> Any:
    return DatabricksSource(_Warehouse(), catalog=_CATALOG, schema=_SCHEMA)


def _schema(source: Any, tables: Any = None, **options: Any) -> Schema:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return Schema.from_source(
            source.semantic_model('sales_mv'), tables, **options
        )


def test_the_model_names_the_tables_keys_and_relationships(
    source: Any,
) -> None:
    schema = _schema(source)

    assert isinstance(schema.source, SemanticModel)
    assert schema.source.name == 'sales_mv'
    assert schema.source.source is source
    assert set(schema.tables) == {'order_lines', 'customers', 'products'}
    assert all(isinstance(t, SourceTable) for t in schema.tables.values())
    assert schema['customers'].primary_key == 'customer_id'
    assert schema['products'].primary_key == 'product_id'
    assert schema['order_lines'].time_column == 'ordered_at'
    assert set(schema.relationships) == {
        Relationship('order_lines', 'customer_id', 'customers', 'customer_id'),
        Relationship('order_lines', 'product_id', 'products', 'product_id'),
    }
    assert schema['order_lines']['discounted'].dtype == 'bool'
    assert 'discount' not in schema['order_lines']
    assert not hasattr(schema, 'metric_view')


def test_tables_select_and_rename_among_the_model_s_own(source: Any) -> None:
    schema = _schema(source, {'lines': 'order_lines', 'buyers': 'customers'})

    assert set(schema.tables) == {'lines', 'buyers'}
    assert schema['lines'].source_name == 'order_lines'
    assert schema.relationships == [
        Relationship('lines', 'customer_id', 'buyers', 'customer_id')
    ]


def test_a_table_the_model_does_not_have_is_named(source: Any) -> None:
    with pytest.raises(StructuredDataError, match='not tables of'):
        _schema(source, {'orders': 'orders'})


def test_the_graph_sampled_is_the_model_cut_to_the_schema(
    source: Any,
) -> None:
    from sdm_relational_engine.rfm.base import DataBackend

    schema = _schema(source, {'lines': 'order_lines', 'buyers': 'customers'})
    del schema['lines']['quantity']

    graph = RelationalMaterializer(schema)._build_graph()

    assert graph.backend == DataBackend.DATABRICKS
    assert set(graph.tables) == {'lines', 'buyers'}
    assert 'quantity' not in graph['lines']
    assert str(graph['lines']['discounted'].expr) == 'discount > 0'
    assert {(e.src_table, e.fkey, e.dst_table) for e in graph.edges} == {
        ('lines', 'customer_id', 'buyers')
    }


def test_a_schema_built_by_hand_reads_the_model_it_names(source: Any) -> None:
    described = _schema(source)
    by_hand = Schema(
        {
            name: SourceTable(
                name,
                table.source_name,
                dict(table.columns),
                primary_key=table.primary_key,
                time_column=table.time_column,
            )
            for name, table in described.tables.items()
        },
        described.relationships,
        source=described.source,
    )

    def shape(schema: Schema) -> Any:
        graph = RelationalMaterializer(schema)._build_graph()
        return (
            {name: list(graph[name].columns) for name in graph.tables},
            {(e.src_table, e.fkey, e.dst_table) for e in graph.edges},
        )

    assert shape(by_hand) == shape(described)
