# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Behaviour the SQL samplers must share.

Each test here covers a fix that had landed on some of the backends but not
all of them. They are written against the whole family, not one member, so a
future fix cannot be applied to one file and forgotten in the others.

``_BIG`` is 2**53 + 1, the first integer ``float64`` cannot represent, so a
column seeded from it detects the widening the moment it happens.
"""

import sqlite3
import uuid
import warnings
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
import sdm_relational_engine.rfm as rfm

pytest.importorskip('sdm_relational_engine.relationallib')

_BIG = 9007199254740993
_QUERY = 'PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.user_id'


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    num_users, num_orders = 24, 144
    rng = np.random.default_rng(0)
    external = [
        None if index % 5 == 0 else _BIG + 2 * index
        for index in range(num_orders)
    ]
    users = pd.DataFrame(
        {
            'user_id': np.arange(num_users, dtype='int64'),
            'age': np.arange(20, 20 + num_users, dtype='int64'),
        }
    )
    orders = pd.DataFrame(
        {
            'order_id': np.arange(num_orders, dtype='int64'),
            'user_id': np.tile(np.arange(num_users, dtype='int64'), 6),
            'date': pd.to_datetime('2025-01-01')
            + pd.to_timedelta(
                rng.integers(0, 90, num_orders).astype('int64'), unit='D'
            ),
            'amount': rng.uniform(5, 100, num_orders).round(2),
            'ext_id': pd.array(external, dtype='Int64'),
        }
    )
    return users, orders


def _write_duckdb(path: str, users: pd.DataFrame, orders: pd.DataFrame) -> None:
    duckdb = pytest.importorskip('duckdb')
    connection = duckdb.connect(path)
    connection.execute(
        'CREATE TABLE users (user_id BIGINT PRIMARY KEY, age BIGINT)'
    )
    connection.execute(
        'CREATE TABLE orders (order_id BIGINT PRIMARY KEY, user_id BIGINT, '
        'date TIMESTAMP, amount DOUBLE, ext_id BIGINT)'
    )
    connection.register('_users', users)
    connection.execute('INSERT INTO users SELECT * FROM _users')
    connection.register('_orders', orders)
    connection.execute('INSERT INTO orders SELECT * FROM _orders')
    connection.execute('CREATE INDEX orders_user_id ON orders(user_id)')
    connection.close()


def _write_sqlite(path: str, users: pd.DataFrame, orders: pd.DataFrame) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        'CREATE TABLE users (user_id BIGINT PRIMARY KEY, age BIGINT)'
    )
    connection.execute(
        'CREATE TABLE orders (order_id BIGINT PRIMARY KEY, user_id BIGINT, '
        'date TEXT, amount REAL, ext_id BIGINT)'
    )
    connection.executemany(
        'INSERT INTO users VALUES (?, ?)', users.values.tolist()
    )
    rows = orders.copy()
    rows['date'] = rows['date'].astype(str)
    rows['ext_id'] = (
        rows['ext_id'].astype(object).where(rows['ext_id'].notna(), None)
    )
    connection.executemany(
        'INSERT INTO orders VALUES (?, ?, ?, ?, ?)',
        [tuple(row) for row in rows.itertuples(index=False)],
    )
    connection.execute('CREATE INDEX orders_user_id ON orders(user_id)')
    connection.commit()
    connection.close()


def _write_postgres(
    dsn: str,
    schema: str,
    users: pd.DataFrame,
    orders: pd.DataFrame,
) -> None:
    psycopg = pytest.importorskip('psycopg')
    connection = psycopg.connect(dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f'CREATE SCHEMA "{schema}"')
            cursor.execute(
                f'CREATE TABLE "{schema}".users ('
                'user_id BIGINT PRIMARY KEY, age BIGINT)'
            )
            cursor.execute(
                f'CREATE TABLE "{schema}".orders ('
                'order_id BIGINT PRIMARY KEY, '
                'user_id BIGINT REFERENCES '
                f'"{schema}".users(user_id), '
                'date TIMESTAMP, amount DOUBLE PRECISION, ext_id BIGINT)'
            )
            cursor.executemany(
                f'INSERT INTO "{schema}".users VALUES (%s, %s)',
                [
                    (int(row.user_id), int(row.age))
                    for row in users.itertuples(index=False)
                ],
            )
            cursor.executemany(
                f'INSERT INTO "{schema}".orders VALUES (%s, %s, %s, %s, %s)',
                [
                    (
                        int(row.order_id),
                        int(row.user_id),
                        row.date.to_pydatetime(),
                        float(row.amount),
                        None if pd.isna(row.ext_id) else int(row.ext_id),
                    )
                    for row in orders.itertuples(index=False)
                ],
            )
            cursor.execute(
                f'CREATE INDEX orders_user_time_idx ON "{schema}".orders '
                '(user_id, date DESC, order_id)'
            )
        connection.commit()
    finally:
        connection.close()


def _graph(backend: str, path: str, schema: str | None = None) -> rfm.Graph:
    if backend == 'postgres':
        graph = rfm.Graph.from_postgres(path, schema=schema, verbose=False)
    else:
        graph = getattr(rfm.Graph, f'from_{backend}')(path, verbose=False)
    graph['orders'].remove_column('ext_id')
    graph['orders'].add_column({'name': 'ext_id', 'stype': 'numerical'})
    edges = {
        (edge.src_table, edge.fkey, edge.dst_table) for edge in graph.edges
    }
    if ('orders', 'user_id', 'users') not in edges:
        graph.link('orders', 'user_id', 'users')
    graph.validate()
    return graph


@pytest.fixture(params=['duckdb', 'sqlite', 'postgres'])
def sql_graph(request, tmp_path) -> Iterator[rfm.Graph]:
    backend = request.param
    users, orders = _frames()
    if backend == 'postgres':
        dsn = request.getfixturevalue('postgres_dsn')
        schema = f'sdm_parity_{uuid.uuid4().hex}'
        _write_postgres(dsn, schema, users, orders)
        graph = _graph(backend, dsn, schema)
        try:
            yield graph
        finally:
            assert graph._connection is not None
            graph._connection.close()
            graph._connection = None
            psycopg = pytest.importorskip('psycopg')
            connection = psycopg.connect(dsn)
            try:
                with connection.cursor() as cursor:
                    cursor.execute(f'DROP SCHEMA "{schema}" CASCADE')
                connection.commit()
            finally:
                connection.close()
        return

    driver = 'duckdb' if backend == 'duckdb' else 'adbc_driver_sqlite'
    pytest.importorskip(driver)
    path = str(tmp_path / f'db.{backend}')
    if backend == 'duckdb':
        _write_duckdb(path, users, orders)
    else:
        _write_sqlite(path, users, orders)
    graph = _graph(backend, path)
    try:
        yield graph
    finally:
        if graph._connection is not None:
            graph._connection.close()
            graph._connection = None


def _related_payload(graph: rfm.Graph) -> tuple[dict, dict]:
    model = rfm.KumoRelational(graph, verbose=False)
    task = model._get_task_table(
        model._parse_query(_QUERY), indices=[1, 2], random_seed=0
    )
    payload = model.materialize_task(task, random_seed=0, verbose=False)[
        0
    ].payload
    return (
        payload['schema']['related_tables']['orders']['columns'],
        payload['context']['related_tables']['orders'],
    )


@pytest.mark.filterwarnings(
    'ignore:.*does not support seeded random sampling.*:UserWarning'
)
def test_nullable_integer_column_survives_subgraph_traversal(sql_graph):
    """``_by_pkey``/``_by_fkey`` converted their Arrow result with a bare
    ``to_pandas()``, so one NULL widened the whole integer column to
    ``float64`` and rounded every value past 2**53, silently, because the
    declared dtype was rewritten to match the corrupted values.
    """
    schema, table = _related_payload(sql_graph)
    assert schema['ext_id']['dtype'] == 'int64'

    position = table['columns'].index('ext_id')
    values = [
        row[position] for row in table['rows'] if row[position] is not None
    ]
    assert len(values) > 0

    _, orders = _frames()
    source = {int(value) for value in orders['ext_id'].dropna()}
    assert all(int(value) in source for value in values)


def test_neighbor_order_breaks_ties_on_the_primary_key(sql_graph):
    """``_by_fkey`` keeps the first ``num_neighbors`` rows of each partition, so
    ordering on the time column alone lets the engine choose arbitrarily
    between rows sharing a timestamp. ``ORDER BY <fkey>`` cannot break that
    tie: the partition is keyed by the foreign key, so it holds one value.
    """
    sampler = rfm.KumoRelational(sql_graph, verbose=False)._sampler
    order_by = sampler._neighbor_order_by('orders', '"date"', '"user_id"')
    assert order_by.endswith('"order_id"')
    assert '"date" DESC' in order_by


def test_neighbor_order_of_a_keyless_table_breaks_ties_on_every_column(
    sql_graph,
):
    """A table with no primary key had nothing after the timestamp to order
    by, so rows sharing one came back in whatever order the engine read them.
    Every scalar column, in name order, makes the order total: rows that agree
    on all of them are indistinguishable to the model.
    """
    sql_graph['orders'].primary_key = None
    sampler = rfm.KumoRelational(sql_graph, verbose=False)._sampler
    refs = sampler.table_column_ref_dict['orders']
    order_by = sampler._neighbor_order_by(
        'orders', refs['date'], refs['user_id']
    )

    terms = [term.strip() for term in order_by.split(',')]
    assert terms[0] == f'{refs["date"]} DESC'
    expected = [ref for name, ref in sorted(refs.items()) if name != 'date']
    assert terms[1:] == expected


def test_projection_order_is_independent_of_set_iteration(sql_graph):
    """The projection list was built by iterating a ``set``, whose order depends
    on the per-process string hash seed.
    """
    sampler = rfm.KumoRelational(sql_graph, verbose=False)._sampler
    columns = {'amount', 'user_id', 'date', 'order_id'}
    expected = sampler._ordered_columns('orders', columns)

    assert expected == sampler._ordered_columns(
        'orders', set(reversed(sorted(columns)))
    )
    assert set(expected) == columns
    declared = list(sampler.table_column_proj_dict['orders'])
    assert expected == [name for name in declared if name in columns]


def test_unknown_column_cannot_reintroduce_set_ordering(sql_graph):
    sampler = rfm.KumoRelational(sql_graph, verbose=False)._sampler
    columns = {'amount', 'zzz_unknown', 'aaa_unknown'}
    ordered = sampler._ordered_columns('orders', columns)
    assert ordered == ['amount', 'aaa_unknown', 'zzz_unknown']


# The graph shapes a reviewer sampled on DuckDB and SQLite: a table reached
# by two paths, a column linking to two copies of one table, as a view
# joining a table twice makes, a chain and a star. Each is sampled from three
# of its tables.
_SHAPES = {
    'diamond': [
        ('orders', 'cid', 'customers'),
        ('customers', 'aid', 'addresses'),
        ('orders', 'aid', 'addresses'),
    ],
    'same-table-twice': [
        ('orders', 'cid', 'customers'),
        ('orders', 'cid', 'buyers'),
    ],
    'chain': [
        ('lines', 'oid', 'orders'),
        ('orders', 'cid', 'customers'),
        ('customers', 'aid', 'addresses'),
    ],
    'star': [
        ('orders', 'cid', 'customers'),
        ('orders', 'pid', 'products'),
        ('orders', 'sid', 'stores'),
    ],
}
_KEYS = {
    'addresses': 'aid',
    'buyers': 'cid',
    'customers': 'cid',
    'lines': 'lid',
    'orders': 'oid',
    'products': 'pid',
    'stores': 'sid',
}


def _shape_frames() -> dict[str, pd.DataFrame]:
    start = pd.Timestamp('2024-01-01')
    order, line = np.arange(40), np.arange(120)
    customers = pd.DataFrame(
        {'cid': np.arange(10), 'aid': np.arange(10) % 5, 'seg': ['x', 'y'] * 5}
    )
    frames = {
        'addresses': pd.DataFrame({'aid': np.arange(5), 'city': [*'abcde']}),
        'buyers': customers.copy(),
        'customers': customers,
        'lines': pd.DataFrame(
            {
                'lid': line,
                'oid': line % 40,
                'qty': line % 7,
                'ts': start + pd.to_timedelta(line % 40, 'D'),
            }
        ),
        'orders': pd.DataFrame(
            {
                'oid': order,
                'cid': order % 10,
                'aid': (order + 2) % 5,
                'pid': order % 6,
                'sid': order % 3,
                'ts': start + pd.to_timedelta(order, 'D'),
            }
        ),
        'products': pd.DataFrame(
            {'pid': np.arange(6), 'price': np.arange(6.0)}
        ),
        'stores': pd.DataFrame({'sid': np.arange(3), 'size': np.arange(3)}),
    }
    # A key is no feature, so a copy of it tells the sampled rows apart.
    for name, frame in frames.items():
        frame['tag'] = frame[_KEYS[name]]
    return frames


def _shape_graph(backend: str, path: str, edges: list) -> rfm.Graph:
    frames = _shape_frames()
    names = sorted({table for edge in edges for table in (edge[0], edge[2])})
    if backend == 'duckdb':
        duckdb = pytest.importorskip('duckdb')
        connection = duckdb.connect(path)
        for name in names:
            connection.register('_frame', frames[name])
            connection.execute(f'CREATE TABLE {name} AS SELECT * FROM _frame')
            connection.unregister('_frame')
        connection.close()
    elif backend == 'sqlite':
        pytest.importorskip('adbc_driver_sqlite')
        connection = sqlite3.connect(path)
        for name in names:
            frame = frames[name].astype(object)
            if 'ts' in frame:
                frame['ts'] = frames[name]['ts'].dt.strftime(
                    '%Y-%m-%d %H:%M:%S'
                )
            connection.execute(f'CREATE TABLE {name} ({", ".join(frame)})')
            connection.executemany(
                f'INSERT INTO {name} VALUES ({", ".join("?" * frame.shape[1])})',
                frame.values.tolist(),
            )
        connection.commit()
        connection.close()
    tables = [
        dict(
            name=name,
            primary_key=_KEYS[name],
            **({'time_column': 'ts'} if 'ts' in frames[name] else {}),
        )
        for name in names
    ]
    if backend == 'local':
        graph = rfm.Graph.from_data(
            {name: frames[name] for name in names},
            edges=edges,
            infer_metadata=False,
            verbose=False,
        )
        for spec in tables:
            graph[spec['name']].primary_key = spec['primary_key']
            if 'time_column' in spec:
                graph[spec['name']].time_column = spec['time_column']
        return graph
    return getattr(rfm.Graph, f'from_{backend}')(
        path, tables=tables, edges=edges, infer_metadata=False, verbose=False
    )


def _sampled(graph: rfm.Graph, entity: str) -> dict[str, tuple]:
    r"""Each table's nodes per hop, and which rows each entity reached."""
    from sdm_relational_engine.rfm.backend.duckdb import DuckDBSampler
    from sdm_relational_engine.rfm.backend.local.sampler import LocalSampler
    from sdm_relational_engine.rfm.backend.sqlite import SQLiteSampler

    backend = graph[entity].backend
    sampler_cls = {
        'duckdb': DuckDBSampler,
        'local': LocalSampler,
        'sqlite': SQLiteSampler,
    }[str(backend)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)  # Missing indices.
        sampler = sampler_cls(graph, verbose=False)
    subgraph = sampler.sample_subgraph(
        (entity,),
        pd.Series([0, 1, 2]),
        pd.Series(pd.to_datetime(['2025-01-01'] * 3)),
        [16, 16, 16],
    )
    sampled = {}
    for name, table in sorted(subgraph.tables.items()):
        keys = table.df['tag'].astype(int).to_numpy()
        if table.row is not None:  # Each node's row of `df`.
            keys = keys[table.row]
        sampled[name] = (
            # The SQL samplers stop at the first hop that reaches nothing.
            [
                int(n)
                for n in np.trim_zeros(np.asarray(table.num_sampled_nodes), 'b')
            ],
            sorted(
                zip(
                    np.asarray(table.batch).tolist(), keys.tolist(), strict=True
                )
            ),
        )
    return sampled


@pytest.mark.parametrize('backend', ['duckdb', 'sqlite'])
@pytest.mark.parametrize(
    'shape, entity',
    [
        (shape, entity)
        for shape, edges in _SHAPES.items()
        for entity in sorted({e[0] for e in edges} | {e[2] for e in edges})[:3]
    ],
)
def test_every_shape_samples_as_in_memory(
    tmp_path, backend: str, shape: str, entity: str
) -> None:
    r"""Two paths to a table sampled a hop too deep, and a column linking
    to two tables failed, on every SQL backend, while the in-memory
    sampler read both right. Each backend now reaches the same rows on each
    hop, from each table, as the in-memory sampler does.
    """
    edges = _SHAPES[shape]
    graph = _shape_graph(backend, str(tmp_path / f'db.{backend}'), edges)
    try:
        sampled = _sampled(graph, entity)
    finally:
        if graph._connection is not None:
            graph._connection.close()
            graph._connection = None

    assert sampled == _sampled(_shape_graph('local', '', edges), entity)
