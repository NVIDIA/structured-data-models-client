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
import sys
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
from sdm_relational_engine import (  # noqa: E402
    GraphConstructionError,
    TableNotFoundError,
    ViewConversionError,
)

from sdm_client import (  # noqa: E402
    Relationship,
    Schema,
    SDMWarning,
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
        if re.match(r'SELECT \d+ AS k, COUNT\(\*\) AS n', sql):
            # Each join's key: its rows, distinct values and missing values.
            self._rows = [
                self._key_count(part) for part in sql.split('\nUNION ALL\n')
            ]
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

    @staticmethod
    def _key_count(sql: str) -> tuple[int, int, int, int]:
        frame = _DATA[re.search(r'FROM `[^`]+`\.`[^`]+`\.`([^`]+)`', sql)[1]]
        values = frame[re.search(r'COUNT\(DISTINCT `(\w+)`\)', sql)[1]]
        return int(sql.split()[1]), len(values), values.nunique(), 0

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
    from sdm_relational_engine.rfm import Graph

    monkeypatch.setitem(globals(), '_DESCRIBE', _UDF_DESCRIBE)
    monkeypatch.setitem(
        _EXPRESSIONS['order_lines'],
        'tagged',
        lambda df: 't' + df['line_id'].astype(str),
    )
    seen: list[Any] = []
    read = Graph.from_databricks_metric_view.__func__  # type: ignore[attr-defined]

    def spy(cls: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get('trust_expressions'))
        return read(cls, *args, **kwargs)

    monkeypatch.setattr(Graph, 'from_databricks_metric_view', classmethod(spy))
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
    graph = RelationalMaterializer(by_hand)._build_graph()

    assert str(graph['order_lines']['tagged'].expr) == 'tag(line_id)'
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


@pytest.mark.parametrize(
    'named', [None, 'order_month'], ids=['engine-choice', 'named-otherwise']
)
def test_a_model_s_time_note_holds_for_the_time_column_the_schema_has(
    named: str | None, source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""The view has only the month of `ordered_at`, which starts earliest of
    any timestamp, so it was taken, and every window counted months. Naming
    the month as the time column still noted that `ordered_at` was taken,
    which the schema no longer held.
    """
    _month_only(monkeypatch)
    time_columns = None if named is None else {'order_lines': named}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        schema = Schema.from_source(
            source.semantic_model('sales_mv'), time_columns=time_columns
        )

    lines = schema['order_lines']
    assert 'order_month' in lines.columns
    taken = [n for n in schema.conversion_notes if 'time column' in n]
    if named is None:
        assert lines.time_column == 'ordered_at'
        said = '\n'.join(str(item.message) for item in caught)
        assert "time_columns={'order_lines': 'order_month'}" in said
        (note,) = taken
        assert "'ordered_at'" in note and "'order_month'" in note
    else:
        assert lines.time_column == 'order_month'
        assert taken == []
        assert schema.conversion_notes  # The measures' notes stay.


def test_notes_name_the_tables_as_the_schema_names_them(
    source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A note named the model's table, which a question asked of a schema
    that names it otherwise cannot name.
    """
    _month_only(monkeypatch)

    schema = _schema(source, {'lines': 'order_lines'})

    notes = schema.conversion_notes
    assert any('(lines.quantity,' in note for note in notes)
    assert any("table 'lines'" in note for note in notes)
    assert not any('order_lines' in note for note in notes)


def _month_only(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The view holds `ordered_at` only as its month, `order_month`."""
    view = _VIEW.replace(
        'name: ordered_at\n    expr: source.ordered_at',
        "name: order_month\n    expr: DATE_TRUNC('MONTH', source.ordered_at)",
    )
    described = {
        'View Text': ('View Text', view, None),
        'ordered_at': ('order_month', 'timestamp', None),
    }
    monkeypatch.setattr(
        sys.modules[__name__],
        '_DESCRIBE',
        [described.get(row[0], row) for row in _DESCRIBE],
    )
    monkeypatch.setitem(
        _EXPRESSIONS,
        'order_lines',
        {
            **_EXPRESSIONS['order_lines'],
            'order_month': lambda df: (
                df['ordered_at'].dt.to_period('M').dt.to_timestamp()
            ),
        },
    )


# --- what the engine says reaches the caller as the client says it ----------


def _with_view(monkeypatch: pytest.MonkeyPatch, view: str) -> None:
    monkeypatch.setattr(
        sys.modules[__name__],
        '_DESCRIBE',
        [
            ('View Text', view, None) if row[0] == 'View Text' else row
            for row in _DESCRIBE
        ],
    )


def test_what_a_model_loses_is_one_sdm_warning_at_the_caller(
    source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""The engine's own warning is no `SDMWarning`, so filtering those let
    it through, and it named this client's module rather than the call.
    """
    _with_view(monkeypatch, _VIEW + 'materialization:\n  mode: relaxed\n')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        Schema.from_source(source.semantic_model('sales_mv'))

    lost = [w for w in caught if 'cannot hold' in str(w.message)]
    assert len(lost) == 1
    assert lost[0].category is SDMWarning
    assert lost[0].filename == __file__
    message = str(lost[0].message)
    assert message.endswith(
        "so it is read in part:\n- Ignored the 'materialization' section "
        'since it cannot be represented in a graph'
    )
    assert not any(
        type(w.message).__name__ == 'ViewConversionWarning' for w in caught
    )


@pytest.mark.parametrize(
    'refused, said',
    [
        ('joined-filter', "references the joined table 'customers'"),
        ('query-source', 'Create a regular view with this query'),
        ('missing-source', "order_lines' does not exist"),
    ],
)
def test_a_model_the_engine_refuses_is_an_invalid_request(
    refused: str, said: str, source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A missing source table is a request about nothing, as it is for a
    source's own tables, but escaped as the engine's `TableNotFoundError`
    once only its view refusals were translated.
    """
    if refused == 'joined-filter':
        view = _VIEW.replace(
            'version: 0.1\n', "version: 0.1\nfilter: customers.segment = 'a'\n"
        )
        _with_view(monkeypatch, view)
    elif refused == 'query-source':
        view = _VIEW.replace(
            'source: cat.sch.order_lines',
            'source: SELECT * FROM cat.sch.order_lines',
        )
        _with_view(monkeypatch, view)
    else:
        monkeypatch.delitem(_TYPES, 'order_lines')

    with pytest.raises(StructuredDataError) as caught:
        _schema(source)

    assert caught.value.code == 'INVALID_REQUEST'
    assert said in caught.value.message
    assert caught.value.details == {'semantic_model': 'sales_mv'}
    assert isinstance(
        caught.value.__cause__, ViewConversionError | TableNotFoundError
    )


class _DriverError(Exception):
    r"""Stands in for a warehouse driver's own exception."""


def test_a_failure_that_is_not_a_refusal_is_not_an_invalid_request(
    source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""Every `ValueError`, a pandas or numpy failure inside the engine
    among them, became `INVALID_REQUEST`, telling the caller to change a
    model that has nothing wrong with it.
    """

    def _fails(frame: pd.DataFrame) -> Any:
        raise _DriverError('the warehouse went away')

    monkeypatch.setitem(_EXPRESSIONS, 'order_lines', {'discounted': _fails})

    with pytest.raises(GraphConstructionError) as caught:
        _schema(source)

    assert not isinstance(caught.value, StructuredDataError)
    assert isinstance(caught.value.__cause__, _DriverError)


def test_no_relationship_is_inferred_beside_a_model_s_joins(
    source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A snowflake view joins lines to orders and orders to customers, and
    the lines' own `customer_id` matched `customers` by name, a second path
    to customers the view never declared.
    """
    view = _VIEW.replace(
        '  - name: customers\n'
        '    source: cat.sch.customers\n'
        '    on: source.customer_id = customers.customer_id\n',
        '  - name: orders\n'
        '    source: cat.sch.orders\n'
        '    on: source.order_id = orders.order_id\n'
        '    joins:\n'
        '      - name: customers\n'
        '        source: cat.sch.customers\n'
        '        on: orders.customer_id = customers.customer_id\n',
    ).replace(
        'expr: customers.segment',
        'expr: orders.customers.segment\n'
        '  - name: customer_id\n'
        '    expr: source.customer_id',
    )
    _with_view(monkeypatch, view)
    lines = _DATA['order_lines']
    monkeypatch.setitem(
        _DATA, 'order_lines', lines.assign(order_id=lines['line_id'] % 4)
    )
    monkeypatch.setitem(
        _TYPES, 'order_lines', {**_TYPES['order_lines'], 'order_id': 'INT'}
    )
    monkeypatch.setitem(
        _DATA,
        'orders',
        pd.DataFrame({'order_id': range(4), 'customer_id': [1, 2, 3, 1]}),
    )
    monkeypatch.setitem(
        _TYPES, 'orders', {'order_id': 'INT', 'customer_id': 'INT'}
    )
    monkeypatch.setitem(_EXPRESSIONS, 'orders', {})

    schema = _schema(source)

    assert set(schema.relationships) == {
        Relationship('order_lines', 'order_id', 'orders', 'order_id'),
        Relationship('orders', 'customer_id', 'customers', 'customer_id'),
        Relationship('order_lines', 'product_id', 'products', 'product_id'),
    }
    # Asked for, it is added, and the model's own are replaced:
    asked = Relationship(
        'order_lines', 'customer_id', 'customers', 'customer_id'
    )
    added = _schema(source, relationships=[*schema.relationships, asked])
    assert set(added.relationships) == {*schema.relationships, asked}
    assert _schema(source, relationships=[asked]).relationships == [asked]


def test_a_dimension_no_query_can_name_is_left_out_and_named(
    source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""`order_lines.a.b` reads as a column `a` of a table `order_lines.a`,
    so a dimension named `a.b` refused the whole schema.
    """
    view = _VIEW.replace('name: discounted', 'name: "is.discounted"')
    _with_view(monkeypatch, view)
    monkeypatch.setitem(
        _EXPRESSIONS,
        'order_lines',
        {'is.discounted': lambda df: df['discount'] > 0},
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        schema = Schema.from_source(source.semantic_model('sales_mv'))

    assert 'is.discounted' not in schema['order_lines'].columns
    assert 'quantity' in schema['order_lines'].columns
    (said,) = [w for w in caught if "'is.discounted'" in str(w.message)]
    assert said.category is SDMWarning
    assert 'a predictive query cannot name a column holding a dot' in str(
        said.message
    )


def test_a_model_that_loses_nothing_converts_without_a_warning(
    source: Any,
) -> None:
    r"""A view with measures loses nothing, yet every one warned that it
    "is read in part", so a caller running with warnings as errors could
    read no such view. The aggregation each measure is stays reachable.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        # Register entry HOST-5: on an older pyarrow (14 and 15, on Databricks
        # Runtime 15.4 and 16.4) the old route's sampler converts with pandas
        # internals newer pandas deprecates; the runtime's warnings, not the
        # client's, so only those of pandas and pyarrow are let through.
        warnings.filterwarnings(
            'ignore', category=DeprecationWarning, module=r'(pandas|pyarrow)\.'
        )
        schema = Schema.from_source(source.semantic_model('sales_mv'))

    (note,) = schema.conversion_notes
    assert "'total_quantity' is SUM(order_lines.quantity, 0, 30, days)" in note
    assert schema.snapshot().conversion_notes == schema.conversion_notes
