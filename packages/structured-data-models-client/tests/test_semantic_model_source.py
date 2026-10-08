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
from sdm_client._request_core import gather_graph  # noqa: E402

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
    # information_schema lists a metric view's measures as plain columns.
    'sales_mv': {
        'line_id': 'INT',
        'segment': 'STRING',
        'total_quantity': 'BIGINT',
    },
}
_TABLE_TYPES = {'sales_mv': 'METRIC_VIEW'}

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

# What a MIN/MAX over a whole table answers, where it is not the rows read.
_RANGES: dict[str, tuple[Any, Any, int]] = {}

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
                (
                    name,
                    dtype,
                    'YES',
                    _CATALOG,
                    _SCHEMA,
                    table,
                    _TABLE_TYPES.get(table or '', 'MANAGED'),
                )
                for name, dtype in types.items()
            ]
            return
        if 'information_schema' in sql:
            return
        table_match = re.search(r'FROM `[^`]+`\.`[^`]+`\.`([^`]+)`', sql)
        assert table_match is not None, f'unexpected SQL: {sql}'
        name = table_match.group(1)
        frame = _DATA[name]
        if sql.startswith('SELECT COUNT(*), MIN('):
            row: list[Any] = [len(frame)]
            for column in re.findall(r'MIN\(`((?:[^`]|``)+)`\)', sql):
                whole = (
                    frame[column].min(),
                    frame[column].max(),
                    int(frame[column].count()),
                )
                row.extend(_RANGES.get(column, whole))
            self._rows = [tuple(row)]
            return
        if sql.upper().startswith('SELECT COUNT(*)'):
            self._rows = [(len(frame),)]
            return
        # An inference sample selects the terms it is ordered by first, and
        # drops them by position:
        *order, projection = re.split(
            r' AS __kumo_order_\d+__, ', sql.split(' FROM ')[0]
        )
        if ' AS `' in projection:
            columns = re.findall(r'AS `((?:[^`]|``)+)`', projection)
        else:
            columns = re.findall(r'`((?:[^`]|``)+)`', projection)
        data = {f'__kumo_order_{i}__': 0 for i in range(len(order))}
        for column in columns:
            if column in frame.columns:
                data[column] = frame[column]
            else:
                data[column] = _EXPRESSIONS[name][column](frame)
        self._arrow = pa.Table.from_pandas(
            pd.DataFrame(data, index=frame.index), preserve_index=False
        )

    def fetchall_arrow(self) -> Any:
        if self._arrow is not None:
            return self._arrow
        width = len(self._rows[0]) if self._rows else 0
        return pa.table(
            {f'c{i}': [row[i] for row in self._rows] for i in range(width)}
        )


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


def test_a_metric_view_named_as_a_table_points_to_the_semantic_model(
    source: Any,
) -> None:
    r"""Reading its rows fails in Databricks on the measures, with an error
    that says nothing of what to do instead. The catalog knows what the
    relation is before any row is read; this fake would fail on such a read.
    """
    with pytest.raises(StructuredDataError) as info:
        Schema.from_source(source, {'m': 'sales_mv'})

    error = info.value
    assert error.code == 'INVALID_REQUEST'
    assert 'is a metric view' in error.message
    assert "source.semantic_model('sales_mv')" in error.message
    assert error.details == {'metric_view': 'sales_mv'}


def test_the_graph_sampled_is_the_model_cut_to_the_schema(
    source: Any,
) -> None:
    from sdm_relational_engine.rfm.base import DataBackend

    schema = _schema(source, {'lines': 'order_lines', 'buyers': 'customers'})
    del schema['lines']['quantity']

    graph = gather_graph(schema)
    tables = dict(graph['tables'])
    lines = {column['name']: column for column in tables['lines']['columns']}

    assert graph['backend'] == DataBackend.DATABRICKS.value
    assert set(tables) == {'lines', 'buyers'}
    assert 'quantity' not in lines
    assert str(lines['discounted']['expr']) == 'discount > 0'
    assert {tuple(edge) for edge in graph['edges']} == {
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
        graph = gather_graph(schema)
        return (
            {
                name: [column['name'] for column in table['columns']]
                for name, table in graph['tables']
            },
            {tuple(edge) for edge in graph['edges']},
        )

    assert shape(by_hand) == shape(described)


_UDF_DIMENSION = '  - name: tagged\n    expr: tag(source.line_id)\n'
_UDF_VIEW = _VIEW.replace('measures:', _UDF_DIMENSION + 'measures:')
_UDF_DESCRIBE = [
    ('View Text', _UDF_VIEW, None) if row[0] == 'View Text' else row
    for row in _DESCRIBE
]
_UDF_DESCRIBE.insert(6, ('tagged', 'string', None))


@pytest.fixture
def udf_view(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    r"""The view gains a dimension calling ``tag()``, a user-defined function.
    Returns what each read of the view was told about trusting it.
    """
    import json

    from sdm_core import _native

    monkeypatch.setitem(globals(), '_DESCRIBE', _UDF_DESCRIBE)
    monkeypatch.setitem(
        _EXPRESSIONS['order_lines'],
        'tagged',
        lambda df: 't' + df['line_id'].astype(str),
    )
    # What the core was told about trusting the view each time it read it:
    # describing the schema, and gathering one that kept nothing it read.
    seen: list[Any] = []
    describe = _native.schema_source_run
    gather = _native.gather_graph_run

    def described(backend: Any, source: Any, args: Any) -> Any:
        seen.append(json.loads(source)['trust_expressions'])
        return describe(backend, source, args)

    def gathered(state: Any, tables: Any) -> Any:
        held = json.loads(state)['schema']
        if not held.get('engine'):
            seen.append(held['source']['trust_expressions'])
        return gather(state, tables)

    monkeypatch.setattr(_native, 'schema_source_run', described)
    monkeypatch.setattr(_native, 'gather_graph_run', gathered)
    return seen


def test_a_udf_dimension_is_left_out_by_default(
    source: Any, udf_view: list[Any]
) -> None:
    with pytest.warns(UserWarning, match="calls 'tag'"):
        schema = Schema.from_source(source.semantic_model('sales_mv'))

    assert 'tagged' not in schema['order_lines']
    assert 'segment' in schema['customers']
    assert udf_view == [False]


def test_trusting_the_model_reaches_both_reads(
    source: Any, udf_view: list[Any]
) -> None:
    model = source.semantic_model('sales_mv', trust_expressions=True)
    assert model.trust_expressions
    assert 'trust_expressions=True' in repr(model)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        described = Schema.from_source(model)
    assert 'tagged' in described['order_lines']

    # A schema written by hand has read nothing yet, so gathering reads the
    # model again and must keep the same column.
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
        source=model,
    )
    graph = dict(gather_graph(by_hand)['tables'])
    tagged = {c['name']: c for c in graph['order_lines']['columns']}['tagged']

    assert str(tagged['expr']) == 'tag(line_id)'
    assert udf_view == [True, True]


# --- a model's time column is checked as a table's is -----------------------


@pytest.fixture
def one_order_date(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Every order line read holds one date."""
    lines = _DATA['order_lines'].assign(ordered_at=pd.Timestamp('2024-01-01'))
    monkeypatch.setitem(_DATA, 'order_lines', lines)


def _time_warnings(source: Any) -> tuple[Schema, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        schema = Schema.from_source(source.semantic_model('sales_mv'))
    said = [
        str(item.message)
        for item in caught
        if 'was not taken as the time column' in str(item.message)
    ]
    return schema, said


def test_a_model_s_constant_time_column_is_refused_and_named(
    source: Any, one_order_date: None
) -> None:
    schema, said = _time_warnings(source)

    assert schema['order_lines'].time_column is None
    assert said == [
        'Some time columns were inferred, or refused, on a guess about what '
        'a column holds:\n'
        '- order_lines.ordered_at was not taken as the time column because '
        'every row holds 2024-01-01 00:00:00. If it does say when a row '
        "became true, name it: time_columns={'order_lines': 'ordered_at'}."
    ]


def test_a_model_s_column_constant_only_in_the_rows_read_is_kept(
    source: Any, one_order_date: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sdm_relational_engine.rfm.backend.databricks import DatabricksTable

    # The rows read are capped at what the fake holds, so they may not be
    # the whole table.
    rows = len(_DATA['order_lines'])
    monkeypatch.setattr(DatabricksTable, '_NUM_SAMPLE_ROWS', rows)
    monkeypatch.setitem(
        _RANGES,
        'ordered_at',
        (pd.Timestamp('2024-01-01'), pd.Timestamp('2024-06-01'), rows),
    )

    schema, said = _time_warnings(source)

    assert schema['order_lines'].time_column == 'ordered_at'
    assert said == []
