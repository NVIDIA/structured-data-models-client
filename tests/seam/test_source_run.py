# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P3.run: every source runs SQL itself, and the engine only asks it to.

``Source._run(sql, params)`` returns a ``pyarrow.Table`` for every warehouse,
and each warehouse's quirks live behind it: PostgreSQL's per-read
transactions and savepoints, Snowflake's parameter style, the notebook
Spark session's way of running a statement.

The engine must reach no driver itself. Static checks cannot see a driver
method reached some other way, so predictions also run with every driver
method watched, and each call must come from ``sdm_connectors`` through
``_run``: SQLite and
PostgreSQL through their real drivers, Databricks, its Spark path, a metric
view, Snowflake and a semantic view through stub drivers.
"""

from __future__ import annotations

import ast
import inspect
import uuid
from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pytest
from _seam import (
    ENGINE_SRC,
    REPO,
    drop_postgres_schema,
    duckdb_database,
    postgres_dsn,
    postgres_schema,
    predict,
    sqlite_database,
    watch_postgres,
    watch_sqlite,
)


def _source_classes() -> list[type]:
    import sdm_connectors

    return [
        sdm_connectors.DatabricksSource,
        sdm_connectors.SnowflakeSource,
        sdm_connectors.PostgresSource,
        sdm_connectors.DuckDBSource,
        sdm_connectors.SQLiteSource,
        sdm_connectors.SemanticModel,
    ]


@pytest.mark.parametrize('index', range(6))
def test_every_source_has_an_internal_run(index: int) -> None:
    cls = _source_classes()[index]
    run = getattr(cls, '_run', None)
    assert callable(run), f'{cls.__name__} has no _run(sql, params)'
    inspect.signature(run).bind(object(), 'SELECT 1', None)
    inspect.signature(run).bind(object(), 'SELECT ?', [1])


@pytest.mark.parametrize('index', range(6))
def test_no_source_exposes_a_public_run(index: int) -> None:
    # The seam is internal (DV13): a public ``run`` would change the public
    # API snapshot, and making it public is a decision for later.
    cls = _source_classes()[index]
    assert 'run' not in dir(cls), (
        f'{cls.__name__} exposes a public run; the seam is _run'
    )


# Running SQL on each database.


def test_sqlite_run(tmp_path: Any) -> None:
    pytest.importorskip('adbc_driver_sqlite')
    from sdm_connectors import SQLiteSource

    source = SQLiteSource(sqlite_database(tmp_path / 'shop.sqlite'))
    table = source._run(
        'SELECT user_id, age FROM users WHERE user_id <= ? ORDER BY user_id',
        [3],
    )
    assert isinstance(table, pa.Table)
    assert table.column_names == ['user_id', 'age']
    assert table.column('user_id').to_pylist() == [1, 2, 3]
    assert source._run(
        'SELECT COUNT(*) AS n FROM orders', None
    ).to_pylist() == [{'n': 400}]


def test_duckdb_run(tmp_path: Any) -> None:
    pytest.importorskip('duckdb')
    from sdm_connectors import DuckDBSource

    source = DuckDBSource(duckdb_database(tmp_path / 'shop.duckdb'))
    table = source._run(
        'SELECT user_id, age FROM users WHERE user_id <= ? ORDER BY user_id',
        [3],
    )
    assert isinstance(table, pa.Table)
    assert table.column_names == ['user_id', 'age']
    assert table.column('user_id').to_pylist() == [1, 2, 3]
    assert source._run(
        'SELECT COUNT(*) AS n FROM orders', None
    ).to_pylist() == [{'n': 400}]


@pytest.fixture
def pg() -> Iterator[tuple[str, str]]:
    pytest.importorskip('psycopg')
    dsn = postgres_dsn()
    schema = f'p3_seam_{uuid.uuid4().hex[:10]}'
    postgres_schema(dsn, schema, autocommit=True).close()
    try:
        yield dsn, schema
    finally:
        drop_postgres_schema(dsn, schema)


def test_postgres_run_types_follow_the_declared_types(
    pg: tuple[str, str],
) -> None:
    import psycopg
    from sdm_connectors import PostgresSource

    dsn, schema = pg
    with psycopg.connect(dsn, autocommit=True) as connection:
        source = PostgresSource(connection, schema=schema)
        sql = (
            'SELECT x.a, x.b, x.c, x.d, x.e FROM (VALUES '
            '(1::bigint, 2.5::double precision, %s::text, true, '
            "TIMESTAMP '2024-01-02 03:04:05'), "
            '(NULL::bigint, NULL::double precision, NULL::text, NULL::boolean,'
            ' NULL::timestamp)) AS x(a, b, c, d, e)'
        )
        table = source._run(sql, ['seam'])
        assert isinstance(table, pa.Table)
        assert table.column_names == ['a', 'b', 'c', 'd', 'e']
        types = [table.schema.field(n).type for n in table.column_names]
        assert types[0] == pa.int64(), types
        assert types[1] == pa.float64(), types
        assert pa.types.is_string(types[2]) or pa.types.is_large_string(
            types[2]
        ), types
        assert types[3] == pa.bool_(), types
        assert pa.types.is_timestamp(types[4]), types
        assert table.column('a').to_pylist() == [1, None]
        assert table.column('c').to_pylist() == ['seam', None]

        empty = source._run(f'{sql} WHERE false', ['seam'])
        assert empty.num_rows == 0
        assert empty.schema.field('a').type == pa.int64()
        assert empty.schema.field('b').type == pa.float64()

        users = source._run(
            f'SELECT user_id FROM "{schema}".users WHERE user_id <= %s '
            f'ORDER BY user_id',
            [2],
        )
        assert users.column('user_id').to_pylist() == [1, 2]


def test_postgres_run_leaves_no_transaction_open(pg: tuple[str, str]) -> None:
    import psycopg
    from psycopg.pq import TransactionStatus
    from sdm_connectors import PostgresSource

    dsn, schema = pg
    with psycopg.connect(dsn, autocommit=False) as connection:
        source = PostgresSource(connection, schema=schema)
        assert connection.info.transaction_status == TransactionStatus.IDLE
        source._run(f'SELECT COUNT(*) AS n FROM "{schema}".users', None)
        assert connection.info.transaction_status == TransactionStatus.IDLE, (
            '_run left a transaction open, holding its locks until a commit'
        )


def test_postgres_run_keeps_the_caller_s_transaction(
    pg: tuple[str, str],
) -> None:
    import psycopg
    from psycopg.pq import TransactionStatus
    from sdm_connectors import PostgresSource

    dsn, schema = pg
    with psycopg.connect(dsn, autocommit=False) as connection:
        source = PostgresSource(connection, schema=schema)
        connection.execute('CREATE TEMP TABLE caller_scratch (x int)')
        assert connection.info.transaction_status == TransactionStatus.INTRANS

        source._run(f'SELECT COUNT(*) AS n FROM "{schema}".users', None)
        with pytest.raises(Exception):  # noqa: B017 - any driver error
            source._run('SELECT * FROM "no such table"', None)

        assert connection.info.transaction_status == TransactionStatus.INTRANS
        # The failed statement did not poison the caller's transaction:
        connection.execute('INSERT INTO caller_scratch VALUES (1)')
        connection.rollback()
        # And nothing committed it: rolling back removed the temp table.
        missing = connection.execute(
            "SELECT to_regclass('pg_temp.caller_scratch')"
        ).fetchone()
        assert missing == (None,), "_run committed the caller's transaction"


# The warehouses, over stub drivers.


@pytest.fixture
def warehouse() -> Any:
    pytest.importorskip('sdm_relational_engine.relationallib')
    pytest.importorskip('databricks.sql')
    pytest.importorskip('snowflake.connector')
    pytest.importorskip('yaml')
    import _warehouse

    return _warehouse


def test_databricks_run(warehouse: Any) -> None:
    for name in ('databricks', 'databricks-spark'):
        log = warehouse.Log()
        source = warehouse.source_for(name, log)
        table = source._run('SELECT COUNT(*) FROM `main`.`shop`.`users`', None)
        assert isinstance(table, pa.Table), name
        assert table.column(0).to_pylist() == [12], name
        table = source._run(
            'SELECT table_name FROM `main`.information_schema.tables '
            'WHERE table_schema = ?',
            ['shop'],
        )
        assert table.column(0).to_pylist() == ['orders', 'users'], name
        assert [s.params for s in log.statements] == [None, ['shop']], name


def test_snowflake_run_binds_with_qmark_and_restores_the_style(
    warehouse: Any,
) -> None:
    log = warehouse.Log()
    source = warehouse.source_for('snowflake', log)
    connection = source.connection
    assert connection._paramstyle == 'pyformat'
    table = source._run('SHOW TABLES LIKE ? IN SCHEMA "DB"."SHOP"', ['USERS'])
    assert isinstance(table, pa.Table)
    assert 'USERS' in table.column('name').to_pylist()
    assert log.statements[-1].paramstyle == 'qmark'
    assert connection._paramstyle == 'pyformat', (
        "_run left the caller's connection in another parameter style"
    )


@pytest.mark.parametrize(
    'name',
    [
        'databricks',
        'databricks-spark',
        'databricks-metric-view',
        'snowflake',
        'snowflake-semantic-view',
    ],
)
def test_warehouse_statements_come_from_the_connector(
    warehouse: Any, name: str
) -> None:
    result = warehouse.run(name)
    statements = result['statements']
    assert len(statements) >= 15, f'only {len(statements)} statements ran'
    direct = sorted(
        {f'{s.caller} via {s.entry}' for s in statements if s.entry != '_run'}
    )
    assert not direct, (
        f'{name}: statements reached the driver other than through '
        f'Source._run: {direct}'
    )
    if name.startswith('snowflake'):
        for s in statements:
            if s.params:
                assert s.paramstyle == 'qmark', s.sql
        assert result['connection']._paramstyle == 'pyformat'


# Local databases, through their real drivers.


def test_sqlite_prediction_reaches_the_driver_only_through_run(
    tmp_path: Any,
) -> None:
    pytest.importorskip('sdm_relational_engine.relationallib')
    pytest.importorskip('adbc_driver_sqlite')
    from sdm_connectors import SQLiteSource

    connection = sqlite_database(tmp_path / 'shop.sqlite', autocommit=True)
    with watch_sqlite() as log:
        predict(SQLiteSource(connection))
    assert len(log.statements) >= 10, 'the driver saw almost no statements'
    direct = log.not_through_run()
    assert not direct, (
        f'{len(direct)} driver call(s) the engine made other than through '
        f'_run:\n  '
        + '\n  '.join(
            f'{c.caller} via {c.entry}: {c.method} {c.sql!r:.80}'
            for c in direct
        )
    )


def test_postgres_prediction_reaches_the_driver_only_through_run(
    pg: tuple[str, str],
) -> None:
    pytest.importorskip('sdm_relational_engine.relationallib')
    import psycopg
    from psycopg.pq import TransactionStatus
    from sdm_connectors import PostgresSource

    dsn, schema = pg
    with psycopg.connect(dsn, autocommit=False) as connection:
        with watch_postgres() as log:
            predict(PostgresSource(connection, schema=schema))
        assert connection.info.transaction_status == TransactionStatus.IDLE
    assert len(log.statements) >= 10, 'the driver saw almost no statements'
    direct = log.not_through_run()
    assert not direct, (
        f'{len(direct)} driver call(s) the engine made other than through '
        f'_run:\n  '
        + '\n  '.join(
            f'{c.caller} via {c.entry}: {c.method} {c.sql!r:.80}'
            for c in direct
        )
    )


# What moved into the connectors stays there.


def _engine_trees() -> Iterator[tuple[str, ast.AST]]:
    for path in sorted(ENGINE_SRC.rglob('*.py')):
        yield path.relative_to(REPO).as_posix(), ast.parse(path.read_text())


def test_the_engine_handles_no_driver_quirk() -> None:
    found: list[str] = []
    for rel, tree in _engine_trees():
        for node in ast.walk(tree):
            line = getattr(node, 'lineno', 0)
            if isinstance(node, ast.Attribute) and node.attr == '_paramstyle':
                found.append(f'{rel}:{line}: sets the Snowflake paramstyle')
            if isinstance(node, ast.Name) and node.id == 'own_transaction':
                found.append(f'{rel}:{line}: opens a PostgreSQL transaction')
            if isinstance(node, ast.alias) and node.name == 'own_transaction':
                found.append(f'{rel}:{line}: imports own_transaction')
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ('transaction', 'rollback')
            ):
                found.append(f'{rel}:{line}: calls .{node.func.attr}()')
    assert not found, (
        'driver quirks still handled in the engine, rather than inside a '
        'connector _run:\n  ' + '\n  '.join(found)
    )
