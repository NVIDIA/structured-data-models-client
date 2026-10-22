# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The warehouse loader's own tests (``contract/runners/databases.py``,
T3.3; design, choice 1): each engine's type map, the statements loading a
case's tables into Databricks and Snowflake (every one fully qualified,
every value written as a literal the engine reads back), the types an
engine stores no column of, refused by name, and the Postgres loader
qualifying every name by its schema, so Lakebase's one test schema holds
case after case and a case table named as a catalog one loads. The
acceptance tests are the test author's.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import uuid
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = ROOT / 'contract/runners'


def _module(name: str) -> ModuleType:
    key = f'warehouse_loader_tests_{name}'
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, RUNNERS / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


databases = _module('databases')
runner = _module('run_python')

DATABRICKS = ('main', 'sdm_test')
SNOWFLAKE = ('SDM', 'SDM_TEST')


def _duck(*statements: str):
    import duckdb

    duck = duckdb.connect()
    for statement in statements:
        duck.execute(statement)
    return duck


def test_the_warehouse_columns_are_the_vector_file_s() -> None:
    vectors = json.loads((ROOT / 'contract/type-map-vectors.json').read_text())
    assert vectors['map'] == databases.TYPE_MAP
    for name, entry in databases.TYPE_MAP.items():
        for engine in databases.WAREHOUSE_ENGINES:
            assert engine in entry, (name, engine)
    assert set(databases.ENGINE) == {
        'databricks-sql',
        'databricks-spark',
        'snowflake',
        'lakebase',
    }


def test_nested_members_take_their_stored_forms_on_databricks() -> None:
    duck = _duck(
        'CREATE TABLE t (s STRUCT(a UINTEGER, t TIME), l UBIGINT[])',
        "INSERT INTO t VALUES ({a: 4294967295, t: '01:02:03'}, [18446744073709551615])",
    )
    data = databases._read(duck, 't')
    assert [
        databases.column_type(f.type, 'databricks', f't.{f.name}')
        for f in data.schema
    ] == ['STRUCT<`a`: BIGINT, `t`: STRING>', 'ARRAY<DECIMAL(20,0)>']
    statements, _ = databases.warehouse_statements(
        duck, 'databricks', DATABRICKS
    )
    insert = statements[-1]
    assert 'CAST(4294967295 AS BIGINT)' in insert
    assert f"CAST(X'{b'01:02:03'.hex().upper()}' AS STRING)" in insert
    assert "CAST('18446744073709551615' AS DECIMAL(20,0))" in insert


def _every_name_qualified(statements: list[str], quote: str, ns: tuple) -> None:
    prefix = '.'.join(f'{quote}{part}{quote}' for part in ns) + '.'
    for statement in statements:
        verb = statement.split(' ', 2)
        if verb[0] in ('CREATE', 'DROP', 'INSERT'):
            assert prefix in statement, statement
        for reference in re.findall(r'REFERENCES (\S+)', statement):
            assert reference.startswith(prefix), statement


def test_databricks_statements_qualify_every_name_and_keep_its_keys() -> None:
    duck = _duck(
        'CREATE TABLE users (id BIGINT PRIMARY KEY, email VARCHAR UNIQUE)',
        'CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id BIGINT '
        'REFERENCES users (id), ts TIMESTAMP, amount DECIMAL(12,2))',
        "INSERT INTO users VALUES (1, 'a''b\\c'), (2, NULL)",
        "INSERT INTO orders VALUES (10, 1, '2024-01-02 03:04:05.123456', "
        '12.50)',
    )
    statements, notes = databases.warehouse_statements(
        duck, 'databricks', DATABRICKS
    )
    _every_name_qualified(statements, '`', DATABRICKS)
    assert statements[:2] == [
        'DROP TABLE IF EXISTS `main`.`sdm_test`.`orders`',
        'DROP TABLE IF EXISTS `main`.`sdm_test`.`users`',
    ]
    create = [s for s in statements if s.startswith('CREATE')]
    assert create[0] == (
        'CREATE TABLE `main`.`sdm_test`.`users` (`id` BIGINT NOT NULL, '
        '`email` STRING, CONSTRAINT `users_pk` PRIMARY KEY (`id`))'
    )
    assert (
        'CONSTRAINT `orders_fk1` FOREIGN KEY (`user_id`) REFERENCES '
        in (create[1])
    )
    assert '`ts` TIMESTAMP_NTZ' in create[1]
    assert '`amount` DECIMAL(12,2)' in create[1]
    assert notes == ['users: UNIQUE (email) not stored']
    insert = [s for s in statements if s.startswith('INSERT')]
    # Text is written as its UTF-8 bytes, so no quote or backslash in a
    # value can end the literal.
    text = b"a'b\\c".hex().upper()
    assert f"CAST(X'{text}' AS STRING)" in insert[0]
    assert 'CAST(NULL AS STRING)' in insert[0]
    assert "CAST('12.5' AS DECIMAL(12,2))" in insert[1]
    assert "CAST('2024-01-02 03:04:05.123456' AS TIMESTAMP_NTZ)" in insert[1]


def test_snowflake_statements_take_text_constants_only() -> None:
    duck = _duck(
        'CREATE TABLE t (i INTEGER, f DOUBLE, s VARCHAR, z TIMESTAMPTZ, '
        'l INTEGER[], m MAP(VARCHAR, INTEGER), b BLOB, d DATE, tm TIME)',
        "INSERT INTO t VALUES (1, 'NaN', 'x''y', '2024-01-01 00:00:00+00', "
        "[1, NULL], MAP {'k': 2}, '\\x00\\xFF'::BLOB, '2024-02-03', "
        "'01:02:03.5')",
        'INSERT INTO t VALUES (NULL, NULL, NULL, NULL, NULL, NULL, NULL, '
        'NULL, NULL)',
    )
    statements, notes = databases.warehouse_statements(
        duck, 'snowflake', SNOWFLAKE
    )
    assert notes == []
    _every_name_qualified(statements, '"', SNOWFLAKE)
    create = next(s for s in statements if s.startswith('CREATE'))
    for declared in (
        '"i" NUMBER(38,0)',
        '"f" FLOAT',
        '"s" VARCHAR',
        '"z" TIMESTAMP_LTZ(6)',
        '"l" ARRAY',
        '"m" OBJECT',
        '"b" BINARY',
        '"tm" TIME(6)',
    ):
        assert declared in create, declared
    (insert,) = [s for s in statements if s.startswith('INSERT')]
    head, values = insert.split(' FROM VALUES ')
    assert 'HEX_DECODE_STRING(column3)' in head
    assert 'PARSE_JSON(HEX_DECODE_STRING(column5))::ARRAY' in head
    assert "TO_BINARY(column7, 'HEX')" in head
    # Only constants after VALUES, each one plain text.
    for constant in re.findall(r"'([^']*)'", values):
        assert re.fullmatch(r'[A-Za-z0-9+.: -]*', constant), constant
    assert "'NaN'" in values
    assert "'2024-01-01 00:00:00+00:00'" in values
    assert values.endswith(
        '(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)'
    )


def test_a_nanosecond_databricks_cannot_hold_is_refused_by_name() -> None:
    duck = _duck(
        'CREATE TABLE t (ts TIMESTAMP_NS)',
        "INSERT INTO t VALUES ('2024-01-01 00:00:00.123456789')",
    )
    with pytest.raises(databases.Unloadable, match=r't\.ts: a time past'):
        databases.warehouse_statements(duck, 'databricks', DATABRICKS)
    statements, _ = databases.warehouse_statements(duck, 'snowflake', SNOWFLAKE)
    assert any("'2024-01-01 00:00:00.123456789'" in s for s in statements)
    assert any('TIMESTAMP_NTZ(9)' in s for s in statements)


def test_a_column_name_delta_reads_as_syntax_maps_columns_by_name() -> None:
    duck = _duck('CREATE TABLE t ("a b" INTEGER)')
    statements, _ = databases.warehouse_statements(
        duck, 'databricks', DATABRICKS
    )
    create = next(s for s in statements if s.startswith('CREATE'))
    assert create.endswith(
        "TBLPROPERTIES ('delta.columnMapping.mode' = 'name')"
    )


def test_large_tables_are_inserted_in_bounded_batches() -> None:
    duck = _duck('CREATE TABLE t AS SELECT range AS i FROM range(1201)')
    statements, _ = databases.warehouse_statements(
        duck, 'databricks', DATABRICKS
    )
    inserts = [s for s in statements if s.startswith('INSERT')]
    assert len(inserts) == 3
    assert sum(s.count('CAST(') for s in inserts) == 1201


def test_load_runs_every_statement_through_the_runner() -> None:
    duck = _duck('CREATE TABLE t (i INTEGER)', 'INSERT INTO t VALUES (1)')
    ran: list[str] = []
    notes = databases.load(duck, 'databricks', ran.append, namespace=DATABRICKS)
    assert notes == []
    assert (
        ran == databases.warehouse_statements(duck, 'databricks', DATABRICKS)[0]
    )

    class Cursor:
        def execute(self, statement: str) -> None:
            ran.append(statement)

        def close(self) -> None:
            ran.append('closed')

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor()

    ran.clear()
    databases.load(duck, 'snowflake', Connection(), namespace=SNOWFLAKE)
    assert ran[1::2] == ['closed'] * (len(ran) // 2)
    with pytest.raises(ValueError, match='two parts'):
        databases.load(duck, 'snowflake', Connection(), namespace=('one',))


def _cases() -> list[Path]:
    return sorted((ROOT / 'contract/cases').glob('s1/**/*.json'))


def test_every_s1_case_is_written_or_named_on_each_warehouse() -> None:
    # Every S1 case's tables either load on each warehouse engine or are
    # refused by name: never an unnamed failure.
    bases = runner.load_bases()
    loaded = {'databricks': 0, 'snowflake': 0}
    for case, _ in runner.load_cases(_cases()):
        duck = runner._connection(case, bases)
        try:
            for engine, namespace in (
                ('databricks', DATABRICKS),
                ('snowflake', SNOWFLAKE),
            ):
                try:
                    databases.warehouse_statements(duck, engine, namespace)
                except databases.Unloadable as refused:
                    assert re.match(r'\S+\.\S+: ', str(refused)), refused
                    continue
                loaded[engine] += 1
        finally:
            duck.close()
    assert loaded['databricks'] > 250 and loaded['snowflake'] > 250, loaded


DSN = os.environ.get('SDM_POSTGRES_TEST_DSN')


@pytest.fixture
def database():
    r"""A new database on ``SDM_POSTGRES_TEST_DSN``'s server, or ``None``
    where it names none and ``SDM_REQUIRE_POSTGRES`` is not set: a test
    given ``None`` has nothing to hold and returns (H26)."""
    if not DSN:
        assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
            'SDM_REQUIRE_POSTGRES is set and no server is named'
        )
        yield None
        return
    import psycopg
    from psycopg.conninfo import make_conninfo

    name = f'sdm_loader_{uuid.uuid4().hex[:12]}'
    with psycopg.connect(DSN, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE {name}')
    try:
        yield make_conninfo(DSN, dbname=name)
    finally:
        with psycopg.connect(DSN, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE {name} WITH (FORCE)')


def test_a_case_table_named_as_a_catalog_one_loads(
    database: str | None,
) -> None:
    if database is None:
        return
    import psycopg

    duck = _duck(
        'CREATE TABLE pg_roles (id INTEGER PRIMARY KEY, name VARCHAR)',
        "INSERT INTO pg_roles VALUES (1, 'a'), (2, 'b')",
    )
    databases.load(duck, 'postgres', database)
    with psycopg.connect(database) as connection:
        rows = connection.execute(
            'SELECT id, name FROM public.pg_roles ORDER BY id'
        ).fetchall()
    assert rows == [(1, 'a'), (2, 'b')]


def test_lakebase_s_one_schema_holds_case_after_case(
    database: str | None,
) -> None:
    if database is None:
        return
    import psycopg

    with psycopg.connect(database, autocommit=True) as connection:
        connection.execute('CREATE SCHEMA sdm_test')
        connection.execute('CREATE VIEW sdm_test.kept AS SELECT 1 AS one')
    first = _duck(
        'CREATE TABLE users (id INTEGER PRIMARY KEY, s STRUCT(a INTEGER))',
        'CREATE TABLE orders (id INTEGER, u INTEGER REFERENCES users (id))',
        'INSERT INTO users VALUES (1, {a: 1})',
        'INSERT INTO orders VALUES (5, 1)',
    )
    second = _duck(
        'CREATE TABLE users (id VARCHAR PRIMARY KEY, s STRUCT(a VARCHAR))',
        "INSERT INTO users VALUES ('x', {a: 'y'})",
    )

    def listed(connection) -> list[tuple]:
        return connection.execute(
            'SELECT table_name FROM information_schema.tables '
            "WHERE table_schema = 'sdm_test' ORDER BY 1"
        ).fetchall()

    with psycopg.connect(database) as connection:
        schema = databases.CaseSchema('lakebase', connection, ('sdm_test',))
        schema.load(first)
        assert listed(connection) == [('kept',), ('orders',), ('users',)]
        schema.load(second)
        rows = connection.execute(
            'SELECT id, (s).a FROM sdm_test.users'
        ).fetchall()
        assert rows == [('x', 'y')]
        # The earlier case's tables are gone; the schema's own view stays.
        assert listed(connection) == [('kept',), ('users',)]
        schema.clear()
        assert listed(connection) == [('kept',)]
        types = connection.execute(
            'SELECT count(*) FROM pg_type t JOIN pg_namespace n '
            "ON n.oid = t.typnamespace WHERE n.nspname = 'sdm_test' "
            "AND t.typname LIKE 'sdm\\_%'"
        ).fetchone()
        assert types == (0,)


def test_a_warehouse_schema_drops_only_what_it_loaded() -> None:
    ran: list[str] = []
    schema = databases.CaseSchema('databricks', ran.append, DATABRICKS)
    schema.load(
        _duck('CREATE TABLE a (i INTEGER)', 'CREATE TABLE b (i INTEGER)')
    )
    ran.clear()
    schema.load(_duck('CREATE TABLE c (i INTEGER)'))
    drops = [s for s in ran if s.startswith('DROP')]
    assert drops[:2] == [
        'DROP TABLE IF EXISTS `main`.`sdm_test`.`b`',
        'DROP TABLE IF EXISTS `main`.`sdm_test`.`a`',
    ]
    ran.clear()
    schema.clear()
    assert ran == ['DROP TABLE IF EXISTS `main`.`sdm_test`.`c`']


class _Target:
    r"""The recording mode's WarehouseTarget, as run_python.py builds it."""

    def __init__(self, connection, names, view=None) -> None:
        self.connection = connection
        self.names = names
        self.view = view


def test_a_leg_s_schema_holds_one_case_at_a_time_through_its_target() -> None:
    ran: list[str] = []

    class Cursor:
        def execute(self, statement: str) -> None:
            ran.append(statement)

        def close(self) -> None:
            pass

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor()

    names = {'catalog': 'cat_t', 'schema': 'sch_t'}
    databases.load(
        _duck('CREATE TABLE a (i INTEGER)'),
        'databricks-sql',
        _Target(
            Connection(),
            names,
            {'kind': 'metric_view', 'name': 'mv', 'yaml': 'version: 1.1'},
        ),
    )
    assert ran[-1].startswith(
        'CREATE OR REPLACE VIEW `cat_t`.`sch_t`.`mv` WITH METRICS'
    )
    ran.clear()
    # The next case, on a new connection: the earlier case's view and
    # table are dropped first, then its own table is loaded.
    databases.load(
        _duck('CREATE TABLE b (i INTEGER)'),
        'databricks-sql',
        _Target(Connection(), names),
    )
    assert ran[:2] == [
        'DROP VIEW IF EXISTS `cat_t`.`sch_t`.`mv`',
        'DROP TABLE IF EXISTS `cat_t`.`sch_t`.`a`',
    ]
    assert any(s.startswith('CREATE TABLE `cat_t`.`sch_t`.`b`') for s in ran)

    class Spark:
        def sql(self, statement: str):
            ran.append(statement)
            return self

        def collect(self) -> list:
            return []

    ran.clear()
    databases.load(
        _duck('CREATE TABLE c (i INTEGER)'),
        'databricks-spark',
        _Target(Spark(), names),
    )
    assert any(s.startswith('CREATE TABLE `cat_t`.`sch_t`.`c`') for s in ran)
