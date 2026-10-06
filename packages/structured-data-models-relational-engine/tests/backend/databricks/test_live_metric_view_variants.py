# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Metric views written the ways people write them, converted live.

Each test creates one metric view over the AdventureWorks tables, converts it
with :meth:`Graph.from_databricks_metric_view`, checks the graph and its
conversion messages and notes, and drops the view and its helper views again,
whether or not the checks pass. Each variant changes the baseline star in the
ways one kind of view does.

Opt-in on top of the ``DATABRICKS_*`` variables of the other live tests:

- ``SDM_MV_TEST_SCHEMA``: ``catalog.schema``, or a schema of
  ``DATABRICKS_CATALOG``, that the tests may create and drop views in.
- ``SDM_MV_TEST_ORDER_LINES``, ``SDM_MV_TEST_ORDERS``,
  ``SDM_MV_TEST_CUSTOMERS`` and ``SDM_MV_TEST_PRODUCTS``: the AdventureWorks
  tables, resolved against ``DATABRICKS_CATALOG`` and ``DATABRICKS_SCHEMA``.
  They default to the names the example notebook loads; the tests skip when
  one cannot be read.

Predictions need an endpoint, so they are left to the example notebook.
"""

import os
import re
import uuid
import warnings
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from sdm_relational_engine.rfm import Edge, Graph, ViewConversionWarning

try:
    import yaml
    from sdm_connectors import quote_ident
    from sdm_connectors.backends.databricks import Connection
    from sdm_relational_engine.rfm.backend.databricks.metric_view import (
        parse_table_reference,
    )
except ImportError:
    pytest.skip("'databricks' extension not installed", allow_module_level=True)

# The conftest marks every test that reaches the warehouse through its
# fixtures; this says so up front as well.
pytestmark = pytest.mark.live_warehouse

_TABLES = {
    'order_lines': (
        'SDM_MV_TEST_ORDER_LINES',
        'adventureworks_sales_order_details',
    ),
    'orders': ('SDM_MV_TEST_ORDERS', 'adventureworks_sales_order_headers'),
    'customers': ('SDM_MV_TEST_CUSTOMERS', 'adventureworks_customers'),
    'products': ('SDM_MV_TEST_PRODUCTS', 'adventureworks_products'),
}

_LINE_DIMS = (
    'SalesOrderID',
    'CustomerID',
    'ProductID',
    'OrderQty',
    'UnitPrice',
    'LineTotal',
    'OrderDate',
)
_CUSTOMER_DIMS = ('TerritoryID', 'StoreID')
_PRODUCT_DIMS = ('Color', 'ListPrice', 'ProductLine')

# The notes every variant gets, since each has these measures: the PQL
# aggregation that matches each one.
_MEASURES = (
    r"'total_quantity' is SUM\(\S+\.OrderQty, 0, 30, days\)",
    r"'revenue' is SUM\(\S+\.LineTotal, 0, 30, days\)",
    r"'order_count' is COUNT_DISTINCT\(\S+\.SalesOrderID, 0, 30, days\)",
)


@dataclass(frozen=True)
class Expected:
    r"""One table of the graph: its key, its time column and its columns."""

    primary_key: str | None
    time_column: str | None
    columns: set[str]


@dataclass
class Views:
    r"""Creates metric views in the test schema and converts them; the
    fixture drops everything created when the test ends.
    """

    connection: 'Connection'
    schema: str  # quoted catalog.schema the views go in
    tables: dict[str, str]  # role to the quoted name of its table
    prefix: str
    created: list[str] = field(default_factory=list)

    @property
    def fact(self) -> str:
        r"""The graph's name for the order lines: its table's name."""
        parts = parse_table_reference(self.tables['order_lines'])
        assert parts is not None
        return parts[-1]

    def execute(self, sql: str) -> list[tuple]:
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
            return list(cursor.fetchall()) if cursor.description else []

    def columns(self, role: str) -> set[str]:
        with self.connection.cursor() as cursor:
            cursor.execute(f'SELECT * FROM {self.tables[role]} LIMIT 0')
            return {column[0] for column in cursor.description}

    def create_view(self, suffix: str, select: str) -> str:
        name = f'{self.schema}.{quote_ident(self.prefix + suffix, "`")}'
        self.created.append(name)
        self.execute(f'CREATE VIEW {name} AS {select}')
        return name

    def create_metric_view(self, document: dict[str, Any]) -> str:
        # A bare `on`, as people write it; YAML 1.1 reads that key as `true`.
        text = yaml.safe_dump(document, sort_keys=False, width=1000)
        text = text.replace("'on':", 'on:')
        name = f'{self.schema}.{quote_ident(self.prefix + "mv", "`")}'
        self.created.append(name)
        self.execute(
            f'CREATE VIEW {name} WITH METRICS LANGUAGE YAML AS $$\n{text}$$'
        )
        return name

    def read(self, name: str) -> Graph:
        r"""The view's graph, which warns once if, and only if, the view
        loses something.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            graph = Graph.from_databricks_metric_view(
                name, connection=self.connection, verbose=False
            )
        said = [w for w in caught if w.category is ViewConversionWarning]
        assert len(said) == (1 if graph.conversion_messages else 0), said
        return graph

    def convert(self, document: dict[str, Any]) -> Graph:
        return self.read(self.create_metric_view(document))


def _quoted(name: str, *defaults: str) -> str:
    r"""``name``, unquoted and dot-separated, with the parts it leaves out
    taken from ``defaults``, quoted for SQL.
    """
    parts = (*defaults, *name.split('.'))[-len(defaults) - 1 :]
    return '.'.join(quote_ident(part, '`') for part in parts)


@pytest.fixture(scope='session')
def mv_schema() -> str:
    # Requested before `connection`, so a run without it connects to nothing.
    schema = os.getenv('SDM_MV_TEST_SCHEMA')
    if not schema:
        pytest.skip('SDM_MV_TEST_SCHEMA names no schema to create views in')
    return schema


@pytest.fixture(scope='session')
def adventureworks(
    mv_schema: str, connection: 'Connection', catalog: str, schema: str
) -> dict[str, str]:
    tables = {}
    for role, (variable, default) in _TABLES.items():
        tables[role] = _quoted(os.getenv(variable) or default, catalog, schema)
        try:
            with connection.cursor() as cursor:
                cursor.execute(f'SELECT 1 FROM {tables[role]} LIMIT 0')
        except Exception as error:  # noqa: BLE001 - any failure means skip
            pytest.skip(f'Cannot read {tables[role]}: {error}')
    return tables


@pytest.fixture
def views(
    mv_schema: str,
    connection: 'Connection',
    catalog: str,
    adventureworks: dict[str, str],
    request: pytest.FixtureRequest,
) -> Generator[Views, None, None]:
    quoted = _quoted(mv_schema, catalog)
    test = re.sub(r'\W', '_', request.node.name.removeprefix('test_'))
    prefix = f'sdm_mvt_{uuid.uuid4().hex[:8]}_{test}_'
    created = Views(connection, quoted, adventureworks, prefix)
    try:
        yield created
    finally:
        for name in reversed(created.created):
            created.execute(f'DROP VIEW IF EXISTS {name}')


def _dim(name: str, expr: str, **metadata: Any) -> dict[str, Any]:
    return {'name': name, 'expr': expr, **metadata}


def _star(views: Views) -> dict[str, Any]:
    r"""The baseline: order lines joined to customers and products, plain
    dimensions and three simple measures, in version 0.1, which Databricks
    keeps as written.
    """
    return {
        'version': 0.1,
        'source': views.tables['order_lines'],
        'joins': [
            {
                'name': 'customers',
                'source': views.tables['customers'],
                'on': 'source.CustomerID = customers.CustomerID',
            },
            {
                'name': 'products',
                'source': views.tables['products'],
                'on': 'source.ProductID = products.ProductID',
            },
        ],
        'dimensions': [_dim(c, f'source.{c}') for c in _LINE_DIMS]
        + [_dim(c, f'customers.{c}') for c in _CUSTOMER_DIMS]
        + [_dim(c, f'products.{c}') for c in _PRODUCT_DIMS],
        'measures': [
            _dim('total_quantity', 'SUM(source.OrderQty)'),
            _dim('revenue', 'SUM(source.LineTotal)'),
            _dim('order_count', 'COUNT(DISTINCT source.SalesOrderID)'),
        ],
    }


def _star_tables(views: Views) -> dict[str, Expected]:
    return {
        views.fact: Expected(None, 'OrderDate', set(_LINE_DIMS)),
        'customers': Expected(
            'CustomerID', None, {'CustomerID', *_CUSTOMER_DIMS}
        ),
        'products': Expected('ProductID', None, {'ProductID', *_PRODUCT_DIMS}),
    }


def _star_edges(views: Views) -> set[Edge]:
    return {
        Edge(views.fact, 'CustomerID', 'customers'),
        Edge(views.fact, 'ProductID', 'products'),
    }


def _assert_graph(
    graph: Graph,
    tables: dict[str, Expected],
    edges: set[Edge],
    *,
    notes: Sequence[str] = _MEASURES,
    lost: Sequence[str] = (),
) -> None:
    r"""The graph has exactly ``tables`` and ``edges``, one conversion note
    matching each pattern in ``notes`` and one conversion message, saying
    what the view lost, matching each pattern in ``lost``, and no others.
    """
    got = {
        name: Expected(
            table.primary_key.name if table.primary_key else None,
            table.time_column.name if table.time_column else None,
            {column.name for column in table.columns},
        )
        for name, table in graph.tables.items()
    }
    assert got == tables
    assert set(graph.edges) == edges
    for said, patterns in (
        (tuple(map(str, graph.conversion_notes)), notes),
        (graph.conversion_messages, lost),
    ):
        assert len(said) == len(patterns), said
        for pattern in patterns:
            assert any(re.search(pattern, message) for message in said), (
                pattern,
                said,
            )


def test_star(views: Views) -> None:
    r"""Display names, synonyms, formats and comments change nothing, nor
    does the server re-serializing a 1.1 definition, a join ``using`` its
    key, or names written in another case, which Databricks matches without
    regard to case.
    """
    document = _star(views)
    document['version'] = 1.1
    document['comment'] = 'Order lines with semantic metadata'
    customers, products = document['joins']
    customers['on'] = 'source.CUSTOMERID = CUSTOMERS.CUSTOMERID'
    del products['on']
    products['using'] = ['ProductID']
    dimensions = {d['name']: d for d in document['dimensions']}
    dimensions['OrderQty'].update(
        display_name='Order quantity',
        comment='Units on the line',
        synonyms=['quantity', 'units'],
        format={'type': 'number'},
    )
    dimensions['OrderDate']['format'] = {
        'type': 'date',
        'date_format': 'year_month_day',
    }
    for name in _CUSTOMER_DIMS:
        dimensions[name]['expr'] = f'CUSTOMERS.{name.upper()}'
    document['measures'][1]['format'] = {
        'type': 'currency',
        'currency_code': 'USD',
    }

    graph = views.convert(document)
    tables = _star_tables(views)
    # The key no dimension names takes the spelling of the join's condition.
    tables['customers'] = Expected(
        'CUSTOMERID', None, {'CUSTOMERID', *_CUSTOMER_DIMS}
    )
    _assert_graph(graph, tables, _star_edges(views))


def test_wildcards(views: Views) -> None:
    r"""``alias.* EXCEPT (...)`` expands to the table's columns, as
    Databricks expands it.
    """
    stamps = {'rowguid', 'ModifiedDate'}
    product_dates = {'SellStartDate', 'SellEndDate', 'DiscontinuedDate'}
    document = _star(views)
    document['dimensions'] = [
        {'expr': 'source.* EXCEPT (rowguid, ModifiedDate)'},
        {'expr': 'customers.* EXCEPT (CustomerID, rowguid, ModifiedDate)'},
        {
            'expr': 'products.* EXCEPT (ProductID, SellStartDate, '
            'SellEndDate, DiscontinuedDate, rowguid, ModifiedDate)'
        },
    ]

    graph = views.convert(document)
    tables = {
        views.fact: Expected(
            None, 'OrderDate', views.columns('order_lines') - stamps
        ),
        'customers': Expected(
            'CustomerID', None, views.columns('customers') - stamps
        ),
        'products': Expected(
            'ProductID',
            None,
            views.columns('products') - stamps - product_dates,
        ),
    }
    _assert_graph(graph, tables, _star_edges(views))


def test_snowflake(views: Views) -> None:
    r"""Customers hang off the orders, not the lines; the lines keep their
    ``CustomerID`` dimension without a second path to customers.
    """
    document = _star(views)
    customers, products = document['joins']
    customers['on'] = 'orders.CustomerID = customers.CustomerID'
    document['joins'] = [
        {
            'name': 'orders',
            'source': views.tables['orders'],
            'on': 'source.SalesOrderID = orders.SalesOrderID',
            'joins': [customers],
        },
        products,
    ]
    document['dimensions'] = (
        [_dim(c, f'source.{c}') for c in _LINE_DIMS]
        + [
            _dim('PlacedOn', 'orders.OrderDate'),
            _dim('Status', 'orders.Status'),
        ]
        + [_dim(c, f'orders.customers.{c}') for c in _CUSTOMER_DIMS]
        + [_dim(c, f'products.{c}') for c in _PRODUCT_DIMS]
    )

    graph = views.convert(document)
    tables = _star_tables(views)
    tables['orders'] = Expected(
        'SalesOrderID',
        'PlacedOn',
        {'SalesOrderID', 'CustomerID', 'PlacedOn', 'Status'},
    )
    edges = {
        Edge(views.fact, 'SalesOrderID', 'orders'),
        Edge('orders', 'CustomerID', 'customers'),
        Edge(views.fact, 'ProductID', 'products'),
    }
    _assert_graph(graph, tables, edges)


def test_role_playing(views: Views) -> None:
    r"""One table joined twice becomes two tables, one per alias."""
    document = _star(views)
    document['joins'][0:1] = [
        {
            'name': alias,
            'source': views.tables['customers'],
            'on': f'source.CustomerID = {alias}.CustomerID',
        }
        for alias in ('buyer', 'account')
    ]
    document['dimensions'] = (
        [_dim(c, f'source.{c}') for c in _LINE_DIMS]
        + [_dim('BuyerTerritory', 'buyer.TerritoryID')]
        + [_dim('AccountStore', 'account.StoreID')]
        + [_dim(c, f'products.{c}') for c in _PRODUCT_DIMS]
    )

    graph = views.convert(document)
    tables = _star_tables(views)
    del tables['customers']
    tables['buyer'] = Expected(
        'CustomerID', None, {'CustomerID', 'BuyerTerritory'}
    )
    tables['account'] = Expected(
        'CustomerID', None, {'CustomerID', 'AccountStore'}
    )
    edges = {
        Edge(views.fact, 'CustomerID', 'buyer'),
        Edge(views.fact, 'CustomerID', 'account'),
        Edge(views.fact, 'ProductID', 'products'),
    }
    _assert_graph(graph, tables, edges)


def test_month_named_as_the_raw_date(views: Views) -> None:
    r"""A month named as the date it truncates keeps its name and value,
    and the raw date is added under a name of its own and taken.
    """
    document = _star(views)
    document['dimensions'] = [
        _dim('OrderDate', "DATE_TRUNC('MONTH', source.OrderDate)")
        if d['name'] == 'OrderDate'
        else d
        for d in document['dimensions']
    ]

    graph = views.convert(document)
    tables = _star_tables(views)
    tables[views.fact] = Expected(
        None, 'OrderDate_raw', {*_LINE_DIMS, 'OrderDate_raw'}
    )
    _assert_graph(
        graph,
        tables,
        _star_edges(views),
        notes=(
            *_MEASURES,
            r"Took 'OrderDate_raw' as the time column .*, since 'OrderDate' "
            r'truncates or casts it, .*\. It is the source column '
            r"'OrderDate', added under a name of its own because",
        ),
    )
    assert str(graph[views.fact]['OrderDate'].expr).startswith('DATE_TRUNC')


def test_measure_inputs_that_are_not_dimensions(views: Views) -> None:
    r"""The columns a measure aggregates join their tables, so a predictive
    query can aggregate them too, a cast one included. A dimension with a
    column's name that holds something else shadows it, so its measure gets
    no note and the column is not added.
    """
    document = _star(views)
    document['dimensions'] = [
        _dim('OrderQty', 'source.OrderQty * 2')
        if d['name'] == 'OrderQty'
        else d
        for d in document['dimensions']
        if d['name'] not in ('LineTotal', 'UnitPrice')
    ]
    document['measures'] += [
        _dim('avg_price', 'AVG(source.UnitPrice)'),
        _dim('total_cost', 'SUM(CAST(products.StandardCost AS DOUBLE))'),
    ]

    graph = views.convert(document)
    tables = _star_tables(views)
    tables['products'].columns.add('StandardCost')
    _assert_graph(
        graph,
        tables,
        _star_edges(views),
        notes=(
            *_MEASURES[1:],
            r"'avg_price' is AVG\(\S+\.UnitPrice, 0, 30, days\)",
        ),
        lost=[r"'OrderQty', which the measure 'total_quantity' reads"],
    )


def _assert_rows(views: Views, graph: Graph, condition: str) -> None:
    r"""The fact table holds the order lines ``condition`` keeps."""
    ((expected,),) = views.execute(
        f'SELECT COUNT(*) FROM {views.tables["order_lines"]} WHERE {condition}'
    )
    assert graph[views.fact]._get_num_rows() == expected


def test_filter_on_a_parameter(views: Views) -> None:
    r"""A parameter is read as its default, as a query of the view without
    arguments reads it.
    """
    document = _star(views)
    document['parameters'] = [
        {'name': 'min_qty', 'data_type': 'INT', 'default': 2}
    ]
    document['filter'] = 'source.OrderQty >= min_qty'

    graph = views.convert(document)
    _assert_graph(
        graph,
        _star_tables(views),
        _star_edges(views),
        notes=(*_MEASURES, r"Read the parameter 'min_qty' as its default '2'"),
    )
    _assert_rows(views, graph, 'OrderQty >= 2')


def test_joins_that_cannot_be_used(views: Views) -> None:
    r"""A join whose key repeats, which would make one entity of several
    rows, and a join to a table that is gone are left out, not the view.
    """
    customers = views.tables['customers']
    repeated = views.create_view(
        'customers',
        f'SELECT * FROM {customers} UNION ALL '
        f'SELECT * FROM {customers} WHERE CustomerID % 10 = 0',
    )
    gone = views.create_view(
        'products', f'SELECT * FROM {views.tables["products"]}'
    )
    document = _star(views)
    document['joins'][0]['source'] = repeated
    document['joins'][1]['source'] = gone
    name = views.create_metric_view(document)
    views.execute(f'DROP VIEW {gone}')

    graph = views.read(name)
    tables = {views.fact: _star_tables(views)[views.fact]}
    _assert_graph(
        graph,
        tables,
        set(),
        lost=[
            r"join 'customers'.* its key 'CustomerID' is not unique",
            r"join 'products'.* does not exist or you may not see it",
        ],
    )
