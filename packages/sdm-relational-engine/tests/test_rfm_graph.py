# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any

import pandas as pd
import pytest
from predictive_query.schema.definition import ColumnKey, ColumnKeyGroup
from sdm_relational_engine.rfm import Edge, Graph, LocalTable
from sdm_relational_engine.rfm.base import ColumnSpec


@pytest.fixture
def sample_dfs() -> dict[str, pd.DataFrame]:
    """Create sample DataFrames for testing."""
    df_dict = {}
    df_dict['users'] = pd.DataFrame(
        {
            'user_id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'created_at': pd.date_range('2023-01-01', periods=5),
            'is_active': [True, True, False, True, False],
        }
    )
    df_dict['orders'] = pd.DataFrame(
        {
            'order_id': [101, 102, 103, 104, 105],
            'user_id': [1, 2, 1, 3, 4],
            'product_id': [201, 202, 201, 203, 202],
            'amount': [100.0, 250.0, 75.0, 300.0, 150.0],
            'order_date': pd.date_range('2023-02-01', periods=5),
            'status': [
                'completed',
                'pending',
                'completed',
                'cancelled',
                'completed',
            ],
        }
    )
    df_dict['products'] = pd.DataFrame(
        {
            'product_id': [201, 202, 203],
            'name': ['Widget A', 'Widget B', 'Widget C'],
            'price': [10.0, 20.0, 30.0],
            'category': ['tools', 'tools', 'gadgets'],
        }
    )
    return df_dict


@pytest.fixture
def sample_tables(sample_dfs: dict[str, pd.DataFrame]) -> list[LocalTable]:
    """Create sample LocalTable objects for testing."""
    users_table = LocalTable(
        sample_dfs['users'],
        name='users',
        primary_key='user_id',
        time_column='created_at',
    )
    orders_table = LocalTable(
        sample_dfs['orders'],
        name='orders',
        primary_key='order_id',
        time_column='order_date',
    )
    products_table = LocalTable(
        sample_dfs['products'],
        name='products',
        primary_key='product_id',
    )
    return [users_table, orders_table, products_table]


def test_init_basic(sample_tables: list[LocalTable]) -> None:
    graph = Graph(sample_tables)

    assert len(graph.tables) == 3
    assert graph.has_table('users')
    assert isinstance(graph['users'], LocalTable)
    assert graph.has_table('orders')
    assert isinstance(graph['orders'], LocalTable)
    assert graph.has_table('products')
    assert isinstance(graph['products'], LocalTable)
    assert len(graph.edges) == 0


def test_init_with_edges(sample_tables: list[LocalTable]) -> None:
    graph = Graph(
        tables=sample_tables,
        edges=[('orders', 'user_id', 'users')],
    )

    assert len(graph.tables) == 3
    assert len(graph.edges) == 1
    assert graph.edges[0] == Edge('orders', 'user_id', 'users')


def test_duplicate_table_names(sample_dfs: dict[str, pd.DataFrame]) -> None:
    table1 = LocalTable(sample_dfs['users'], name='duplicate')
    table2 = LocalTable(sample_dfs['orders'], name='duplicate')

    with pytest.raises(KeyError, match='names must be globally unique'):
        Graph(tables=[table1, table2])


def test_from_data_basic(sample_dfs: dict[str, pd.DataFrame]) -> None:
    graph = Graph.from_data(sample_dfs)

    assert len(graph.tables) == 3
    assert graph.has_table('users')
    assert isinstance(graph['users'], LocalTable)
    assert graph.has_table('orders')
    assert isinstance(graph['orders'], LocalTable)
    assert graph.has_table('products')
    assert isinstance(graph['products'], LocalTable)

    for table in graph.tables.values():
        for col in table.columns:
            assert col.dtype is not None
            assert col.stype is not None

    assert graph['users'].primary_key is not None
    assert graph['users'].primary_key.name == 'user_id'
    assert graph['users'].time_column is not None
    assert graph['users'].time_column.name == 'created_at'
    assert graph['orders'].primary_key is not None
    assert graph['orders'].primary_key.name == 'order_id'
    assert graph['orders'].time_column is not None
    assert graph['orders'].time_column.name == 'order_date'
    assert graph['products'].primary_key is not None
    assert graph['products'].primary_key.name == 'product_id'
    assert not graph['products'].has_time_column()

    assert len(graph.edges) == 2
    assert Edge('orders', 'user_id', 'users') in graph.edges
    assert Edge('orders', 'product_id', 'products') in graph.edges


def test_from_data_with_edges(sample_dfs: dict[str, pd.DataFrame]) -> None:
    graph = Graph.from_data(
        sample_dfs,
        edges=[('orders', 'user_id', 'users')],
    )

    for table in graph.tables.values():
        for col in table.columns:
            assert col.dtype is not None
            assert col.stype is not None

    assert graph['users'].primary_key is not None
    assert graph['users'].primary_key.name == 'user_id'
    assert graph['users'].time_column is not None
    assert graph['users'].time_column.name == 'created_at'
    assert graph['orders'].primary_key is not None
    assert graph['orders'].primary_key.name == 'order_id'
    assert graph['orders'].time_column is not None
    assert graph['orders'].time_column.name == 'order_date'
    assert graph['products'].primary_key is not None
    assert graph['products'].primary_key.name == 'product_id'
    assert not graph['products'].has_time_column()

    assert len(graph.edges) == 1
    assert graph.edges[0] == Edge('orders', 'user_id', 'users')


def test_empty_graph() -> None:
    graph = Graph(tables=[])
    with pytest.raises(ValueError, match='At least one table needs to be'):
        graph.validate()

    graph = Graph.from_data({})
    with pytest.raises(ValueError, match='At least one table needs to be'):
        graph.validate()

    assert str(graph) == ('Graph(\n  tables=[],\n  edges=[],\n)')


def test_has_table(sample_tables: list[LocalTable]) -> None:
    graph = Graph(sample_tables)

    assert graph.has_table('users')
    assert 'users' in graph
    assert graph.has_table('orders')
    assert 'orders' in graph
    assert graph.has_table('products')
    assert 'products' in graph
    assert not graph.has_table('nonexistent')
    assert 'nonexistent' not in graph


def test_table_access(sample_tables: list[LocalTable]) -> None:
    graph = Graph(sample_tables)

    assert isinstance(graph.table('users'), LocalTable)
    assert graph.table('users').name == 'users'
    assert graph['users'].name == 'users'


def test_table_not_found(sample_tables: list[LocalTable]) -> None:
    graph = Graph(sample_tables)

    with pytest.raises(KeyError, match="'nonexistent' not found in graph"):
        graph.table('nonexistent')

    with pytest.raises(KeyError, match="'nonexistent' not found in graph"):
        graph['nonexistent']


def test_tables(sample_tables: list[LocalTable]) -> None:
    graph = Graph(sample_tables)

    assert isinstance(graph.tables, dict)
    assert len(graph.tables) == 3
    assert 'users' in graph.tables
    assert 'orders' in graph.tables
    assert 'products' in graph.tables


def test_link_basic(sample_tables: list[LocalTable]) -> None:
    graph = Graph(sample_tables)

    graph.link('orders', 'user_id', 'users')
    assert len(graph.edges) == 1
    assert graph.edges[0] == Edge('orders', 'user_id', 'users')


def test_link_from_table(sample_tables: list[LocalTable]) -> None:
    graph = Graph(sample_tables)

    graph.link(graph['orders'], 'user_id', graph['users'])
    assert len(graph.edges) == 1
    assert graph.edges[0] == Edge('orders', 'user_id', 'users')


def test_link_duplicate_edge(sample_tables: list[LocalTable]) -> None:
    graph = Graph(sample_tables)

    graph.link('orders', 'user_id', 'users')

    with pytest.raises(ValueError, match='Edge.* already exists'):
        graph.link('orders', 'user_id', 'users')


def test_link_nonexistent(sample_tables: list[LocalTable]) -> None:
    graph = Graph(sample_tables)

    with pytest.raises(ValueError, match="'nonexistent' does not exist"):
        graph.link('nonexistent', 'user_id', 'users')

    with pytest.raises(ValueError, match="'nonexistent' does not exist"):
        graph.link('orders', 'nonexistent', 'users')

    with pytest.raises(ValueError, match="'nonexistent' does not exist"):
        graph.link('orders', 'user_id', 'nonexistent')


def test_link_invalid_foreign_key(sample_tables: list[LocalTable]) -> None:
    graph = Graph(sample_tables)

    graph.link('orders', 'order_id', 'users')
    with pytest.raises(ValueError, match='Cannot treat the primary key'):
        graph.validate()

    with pytest.raises(ValueError, match='incompatible data type'):
        graph.link('users', 'created_at', 'products')


def test_unlink_basic(sample_tables: list[LocalTable]) -> None:
    graph = Graph(sample_tables)

    graph.link('orders', 'user_id', 'users')
    assert len(graph.edges) == 1

    graph.unlink('orders', 'user_id', 'users')
    assert len(graph.edges) == 0


def test_unlink_from_table(sample_tables: list[LocalTable]) -> None:
    graph = Graph(sample_tables)

    graph.link('orders', 'user_id', 'users')
    assert len(graph.edges) == 1

    graph.unlink(graph['orders'], 'user_id', graph['users'])
    assert len(graph.edges) == 0


def test_unlink_nonexistent_edge(sample_tables: list[LocalTable]) -> None:
    graph = Graph(sample_tables)

    with pytest.raises(ValueError, match='Edge.* is not present'):
        graph.unlink('orders', 'user_id', 'users')


def test_validate_success(sample_dfs: dict[str, pd.DataFrame]) -> None:
    graph = Graph.from_data(sample_dfs)

    assert graph.validate() == graph


def test_validate_compatible_data_type_families(
    sample_dfs: dict[str, pd.DataFrame],  #
) -> None:
    # Create data frames with different types but same family:
    users_df = sample_dfs['users']
    orders_df = sample_dfs['orders']
    orders_df['user_id'] = orders_df['user_id'].astype(float)

    users_table = LocalTable(users_df, 'users', primary_key='user_id')
    orders_table = LocalTable(orders_df, 'orders', primary_key='order_id')

    graph = Graph(tables=[users_table, orders_table])
    graph.link('orders', 'user_id', 'users')
    assert graph.validate() == graph


def test_validate_mismatched_data_type_families(
    sample_dfs: dict[str, pd.DataFrame],  #
) -> None:
    # Create data frames with different types but same family:
    users_df = sample_dfs['users']
    orders_df = sample_dfs['orders']
    orders_df['user_id'] = orders_df['user_id'].astype(str)

    users_table = LocalTable(users_df, 'users', primary_key='user_id')
    orders_table = LocalTable(orders_df, 'orders', primary_key='order_id')

    graph = Graph(tables=[users_table, orders_table])
    graph.link('orders', 'user_id', 'users')

    with pytest.raises(ValueError, match='have incompatible data types'):
        graph.validate()


def test_validate_single_table(sample_tables: list[LocalTable]) -> None:
    for table in sample_tables:
        graph = Graph([table])
        assert graph.validate() == graph


def test_hash(sample_dfs: dict[str, pd.DataFrame]) -> None:
    graph1 = Graph.from_data(sample_dfs)
    graph2 = Graph.from_data(sample_dfs)

    assert hash(graph1) == hash(graph2)

    graph1.unlink('orders', 'user_id', 'users')
    assert hash(graph1) != hash(graph2)


def test_repr(sample_tables: list[LocalTable]) -> None:
    graph = Graph(sample_tables)
    graph.link('orders', 'user_id', 'users')

    assert str(graph) == (
        'Graph(\n'
        '  tables=[\n'
        '    users,\n'
        '    orders,\n'
        '    products,\n'
        '  ],\n'
        '  edges=[\n'
        '    orders.user_id ⇔ users.user_id,\n'
        '  ],\n'
        ')'
    )


def test_copy(sample_tables: list[LocalTable]) -> None:
    graph = Graph(sample_tables)
    graph.link('orders', 'user_id', 'users')

    graph_copy = copy.deepcopy(graph)

    # Should be different objects
    assert graph is not graph_copy
    assert graph.tables is not graph_copy.tables
    assert graph.edges is not graph_copy.edges

    # But should have same content
    assert hash(graph) == hash(graph_copy)


def test_link_self_reference() -> None:
    df = pd.DataFrame(
        {
            'user_id': [1, 2, 3],
            'parent_user_id': [None, 1, 2],  # Self-referencing column
            'name': ['Alice', 'Bob', 'Charlie'],
        }
    )

    table = LocalTable(df, 'users', primary_key='user_id')
    graph = Graph(tables=[table]).infer_links()

    assert graph.edges == [Edge('users', 'parent_user_id', 'users')]


def test_to_api_graph_definition(sample_tables: list[LocalTable]) -> None:
    graph = Graph(sample_tables)
    graph.link('orders', 'user_id', 'users')
    graph.link('orders', 'product_id', 'products')
    graph_def = graph._to_api_graph_definition()
    assert set(graph_def.tables.keys()) == {'users', 'orders', 'products'}
    assert graph_def.col_groups == [
        ColumnKeyGroup(
            columns=(
                ColumnKey(table_name='orders', col_name='user_id'),
                ColumnKey(table_name='users', col_name='user_id'),
            )
        ),
        ColumnKeyGroup(
            columns=(
                ColumnKey(table_name='orders', col_name='product_id'),
                ColumnKey(table_name='products', col_name='product_id'),
            )
        ),
    ]


def test_to_mermaid_basic(sample_tables: list[LocalTable]) -> None:
    graph = Graph(sample_tables)
    graph.link('orders', 'user_id', 'users')
    graph.link('orders', 'product_id', 'products')

    mermaid = graph._to_mermaid(show_columns=False)
    assert mermaid.startswith('erDiagram')

    # Check tables are present:
    assert 'users {' in mermaid
    assert 'orders {' in mermaid
    assert 'products {' in mermaid

    # Check primary keys:
    assert 'user_id PK' in mermaid
    assert 'order_id PK' in mermaid
    assert 'product_id PK' in mermaid

    # Check foreign keys:
    assert 'user_id FK' in mermaid
    assert 'product_id FK' in mermaid

    # Check timestamps:
    assert 'timestamp created_at' in mermaid
    assert 'timestamp order_date' in mermaid

    # Check no feature columns:
    assert 'name' not in mermaid
    assert 'age' not in mermaid
    assert 'amount' not in mermaid

    # Check relationships:
    assert 'users o|--o{ orders : user_id' in mermaid
    assert 'products o|--o{ orders : product_id' in mermaid


def test_to_mermaid_with_columns(sample_tables: list[LocalTable]) -> None:
    graph = Graph(sample_tables)

    mermaid = graph._to_mermaid(show_columns=True)

    assert 'categorical name' in mermaid
    assert 'numerical age' in mermaid
    assert 'numerical amount' in mermaid


def test_to_mermaid_empty_graph() -> None:
    graph = Graph(tables=[])
    mermaid = graph._to_mermaid()

    assert mermaid == 'erDiagram'


def test_to_mermaid_no_edges(sample_tables: list[LocalTable]) -> None:
    graph = Graph(sample_tables)

    mermaid = graph._to_mermaid(show_columns=False)

    assert 'users {' in mermaid
    assert 'o|--o{' not in mermaid


# Regression tests.
def test_validate_primary_key_as_foreign_key_message(
    sample_dfs: dict[str, pd.DataFrame],  #
) -> None:
    users_table = LocalTable(
        sample_dfs['users'], 'users', primary_key='user_id'
    )
    orders_table = LocalTable(
        sample_dfs['orders'], 'orders', primary_key='user_id'
    )

    graph = Graph(tables=[users_table, orders_table])
    graph.link('orders', 'user_id', 'users')

    with pytest.raises(ValueError) as err:
        graph.validate()

    assert 'the link before proceeding.' in str(err.value)
    assert 'before before' not in str(err.value)


def test_validate_non_key_foreign_key_dtype_message(
    sample_dfs: dict[str, pd.DataFrame],  #
) -> None:
    users_table = LocalTable(
        sample_dfs['users'], 'users', primary_key='user_id'
    )
    orders_table = LocalTable(
        sample_dfs['orders'], 'orders', primary_key='order_id'
    )

    graph = Graph(tables=[users_table, orders_table])
    graph.link('orders', 'user_id', 'users')
    graph['orders'].remove_column('user_id')
    graph['orders'].add_column(
        ColumnSpec('user_id', dtype='date', stype='timestamp')
    )

    with pytest.raises(ValueError) as err:
        graph.validate()

    message = str(err.value)
    assert "(got 'date')" in message
    assert message.count('(') == message.count(')')


def test_validate_names_the_edge_left_behind_by_remove_column(
    sample_dfs: dict[str, pd.DataFrame],  #
) -> None:
    """``Table.remove_column`` has no back-reference to the graph, so dropping a
    linked foreign key leaves the edge behind. The column lookup in
    ``validate`` then raised ``KeyError`` -- the wrong type for a graph
    consistency failure, and uncatchable by the ``except ValueError`` the
    documented failure mode implies.
    """
    users_table = LocalTable(
        sample_dfs['users'], 'users', primary_key='user_id'
    )
    orders_table = LocalTable(
        sample_dfs['orders'], 'orders', primary_key='order_id'
    )

    graph = Graph(tables=[users_table, orders_table])
    graph.link('orders', 'user_id', 'users')
    graph['orders'].remove_column('user_id')

    with pytest.raises(ValueError) as err:
        graph.validate()

    message = str(err.value)
    assert "'orders'" in message
    assert "'user_id'" in message
    assert 'unlink()' in message


def test_empty_edges_suppress_inference_unlike_none(
    sample_dfs: dict[str, pd.DataFrame],  #
) -> None:
    r"""The documented divergence between ``edges=None`` and ``edges=[]``."""
    inferred = Graph.from_data(sample_dfs, verbose=False)
    assert len(inferred.edges) > 0

    assert Graph.from_data(sample_dfs, edges=[], verbose=False).edges == []


def test_empty_edges_suppress_catalog_foreign_keys_too(
    tmp_path: Any,  #
) -> None:
    """``edges=[]`` is documented as "these edges and no others". It suppressed
    the heuristic ``infer_links`` pass but not the one in ``Graph.__init__``
    that reads foreign keys straight out of the source catalog, so on any
    warehouse that declares them an explicit edge list still picked up edges
    the caller did not ask for -- and edges change what the model sees.
    """
    pytest.importorskip('adbc_driver_sqlite')

    import sqlite3

    path = tmp_path / 'fk.db'
    connection = sqlite3.connect(path)
    connection.executescript("""
    CREATE TABLE users  (user_id INTEGER PRIMARY KEY, age INTEGER);
    CREATE TABLE orders (order_id INTEGER PRIMARY KEY, user_id INTEGER,
                         price REAL,
                         FOREIGN KEY(user_id) REFERENCES users(user_id));
    """)
    connection.executemany(
        'INSERT INTO users VALUES (?,?)', [(i, 20 + i) for i in range(20)]
    )
    connection.executemany(
        'INSERT INTO orders VALUES (?,?,?)',
        [(i, i % 20, float(i)) for i in range(60)],
    )
    connection.commit()
    connection.close()

    declared = [('orders', 'user_id', 'users')]

    # `edges=None` still applies the catalog's foreign keys, as documented.
    assert [
        tuple(edge)
        for edge in Graph.from_sqlite(str(path), verbose=False).edges
    ] == declared

    for kwargs in ({'edges': []}, {'edges': [], 'infer_metadata': False}):
        graph = Graph.from_sqlite(str(path), verbose=False, **kwargs)
        assert list(graph.edges) == [], kwargs

    # An explicit list is honoured exactly, and is not duplicated by the
    # catalog pass naming the same edge.
    pinned = Graph.from_sqlite(str(path), edges=declared, verbose=False)
    assert [tuple(edge) for edge in pinned.edges] == declared


def test_a_graph_without_any_primary_key_is_valid_but_warns() -> None:
    r"""A partial conversion may legitimately yield a graph with no key.

    `from_databricks_metric_view` produces exactly this when every join in the
    view is rejected, so `validate()` cannot require a primary key without
    turning a safely-degraded conversion into a hard failure. The graph is
    reported twice instead: once as a warning naming the columns that could
    serve as the key, and again by the parser if a query names one as an
    entity.
    """
    left = pd.DataFrame({'n_id': range(5), 'value': range(5)})
    right = pd.DataFrame({'w_id': range(5), 'n_id': range(5)})

    with pytest.warns(UserWarning, match='pass `primary_key=` explicitly'):
        graph = Graph.from_data({'nulls': left, 'wide': right}, verbose=False)

    assert not any(graph[name].has_primary_key() for name in graph.tables)
    assert graph.validate() is graph


def test_column_specs_are_distinguishable() -> None:
    r"""Two specs for different columns are not the same spec.

    `ColumnSpec` was a dataclass with no annotated fields, so the generated
    `__eq__` compared an empty tuple and every instance compared equal, while
    `__repr__` rendered as `ColumnSpec()`. Its sibling `Column` already opts
    out of both.
    """
    alpha = ColumnSpec('alpha')
    beta = ColumnSpec('beta')

    assert alpha != beta
    assert alpha == alpha
    assert hash(alpha) == hash(alpha)
    assert repr(alpha) == 'ColumnSpec(name=alpha)'
    assert repr(ColumnSpec('gamma', dtype='int')) == (
        'ColumnSpec(name=gamma, dtype=int)'
    )
