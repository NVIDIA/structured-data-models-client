# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live write guard (scripts/live/write_guard.py; KO3-51): an allowlist
of the old client's reads and the loader's writes, the writes only in the
one test schema, enforced on every driver's entry points; everything else
never reaches a driver. Every bypass the round 4 reviews found is here,
refused."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wg = _load('t33_write_guard', ROOT / 'scripts/live/write_guard.py')
databases = _load('t33_guard_databases', ROOT / 'contract/runners/databases.py')

#: Made-up names; Snowflake stores unquoted names upper-cased.
SNOWFLAKE = wg.WriteGuard('snowflake', 'REVIEW_DB', 'LIVE_TESTS')
DATABRICKS = wg.WriteGuard('databricks', 'review-catalog', 'live_tests')
POSTGRES = wg.WriteGuard('postgres', None, 'live_tests')
GUARDS = {
    'snowflake': SNOWFLAKE,
    'databricks': DATABRICKS,
    'postgres': POSTGRES,
}

ALLOWED = {
    'snowflake': [
        'SELECT * FROM "OTHER_DB"."PUBLIC"."T"',
        "SELECT REPLACE(a, 'x', 'y') FROM OTHER_DB.S.T",
        'WITH TMP AS (SELECT 1 AS x) SELECT * FROM TMP',
        'SELECT SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW(?)',
        'SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()',
        "SHOW PARAMETERS LIKE 'TIMEZONE' IN SESSION",
        'SHOW TABLES IN SCHEMA "REVIEW_DB"."LIVE_TESTS"',
        'SHOW TABLES LIKE ? IN SCHEMA OTHER_DB.S',
        'SHOW IMPORTED KEYS IN TABLE OTHER_DB.S.T',
        'DESCRIBE TABLE OTHER_DB.S.T',
        'ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = 60',
        'CREATE TABLE "REVIEW_DB"."LIVE_TESTS"."users" ("ID" NUMBER(38,0) NOT NULL, '
        'PRIMARY KEY ("ID"))',
        'CREATE TABLE "REVIEW_DB"."LIVE_TESTS"."orders" ("U" NUMBER(38,0), '
        'FOREIGN KEY ("U") REFERENCES "REVIEW_DB"."LIVE_TESTS"."users" ("ID"))',
        'create or replace table review_db.live_tests.t (i int)',
        'INSERT INTO "REVIEW_DB"."LIVE_TESTS"."users" ("ID") SELECT '
        "CAST(column1 AS NUMBER(38,0)) FROM VALUES ('1'), (NULL)",
        'DROP TABLE IF EXISTS "REVIEW_DB"."LIVE_TESTS"."users"',
        'DROP SEMANTIC VIEW IF EXISTS "REVIEW_DB"."LIVE_TESTS".shop',
        'CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML(\'"REVIEW_DB"."LIVE_TESTS"\', '
        '$$name: shop\ntables:\n  - name: users\n    base_table:\n'
        '      database: REVIEW_DB\n      schema: LIVE_TESTS\n      table: \'"users"\'\n$$)',
    ],
    'databricks': [
        'SELECT * FROM `other`.`s`.`t`',
        'SELECT current_timezone() AS zone',
        'SHOW TABLES IN `review-catalog`.`live_tests`',
        'DESCRIBE TABLE EXTENDED `other`.s.t',
        'USE CATALOG `review-catalog`',
        'USE SCHEMA `live_tests`',
        'CREATE TABLE `review-catalog`.`live_tests`.`users` (`id` BIGINT)',
        'CREATE TABLE `review-catalog`.live_tests.t (`a b` INT) '
        "TBLPROPERTIES ('delta.columnMapping.mode' = 'name')",
        'INSERT INTO `review-catalog`.`live_tests`.`users` (`id`) VALUES '
        "(CAST(1 AS BIGINT)), (CAST(X'6869' AS STRING)), (DATE'2024-01-01')",
        'DROP TABLE IF EXISTS `review-catalog`.`live_tests`.`users`',
        'DROP VIEW IF EXISTS `review-catalog`.`live_tests`.`m`',
        'CREATE OR REPLACE VIEW `review-catalog`.`live_tests`.`m` WITH METRICS '
        'LANGUAGE YAML AS $$\nversion: 0.1\nsource: other.s.t\n$$',
    ],
    'postgres': [
        'SELECT current_schema()',
        'SELECT table_name FROM information_schema.tables WHERE table_schema = %s',
        'SELECT setseed(%s)',
        'CREATE TABLE "live_tests"."users" ("id" bigint NOT NULL)',
        'CREATE TYPE "live_tests"."users_tags" AS ("a" text, "b" bigint[])',
        'DROP TABLE IF EXISTS "live_tests"."users"',
        'DROP TYPE IF EXISTS "live_tests"."users_tags"',
        'COPY "live_tests"."users" FROM STDIN',
    ],
}

REFUSED = {
    'snowflake': [
        # Another schema or database, or not fully qualified.
        'CREATE TABLE OTHER_DB.LIVE_TESTS.T (i INT)',
        'CREATE TABLE REVIEW_DB.OTHER.T (i INT)',
        'CREATE TABLE "review_db"."live_tests".t (i INT)',
        'CREATE TABLE LIVE_TESTS.T (i INT)',
        'CREATE TABLE T (i INT)',
        'DROP TABLE OTHER_DB.S.T',
        'DROP TABLE IF EXISTS "users"',
        'INSERT INTO OTHER_DB.S.T VALUES (1)',
        'INSERT INTO T VALUES (1)',
        'CREATE TABLE REVIEW_DB.LIVE_TESTS.A (u INT REFERENCES OTHER_DB.S.B (i))',
        # Forms the guard does not admit.
        'INSERT ALL INTO REVIEW_DB.LIVE_TESTS.A INTO OTHER_DB.S.B SELECT 1',
        'CREATE TABLE REVIEW_DB.LIVE_TESTS.T AS SELECT * FROM OTHER_DB.S.T',
        'DELETE FROM REVIEW_DB.LIVE_TESTS.A',
        'TRUNCATE TABLE REVIEW_DB.LIVE_TESTS.A',
        'ALTER TABLE REVIEW_DB.LIVE_TESTS.A RENAME TO OTHER_DB.S.B',
        'DROP SCHEMA REVIEW_DB.LIVE_TESTS',
        'CREATE SCHEMA IF NOT EXISTS REVIEW_DB.LIVE_TESTS',
        'DROP TABLE REVIEW_DB.LIVE_TESTS.A CASCADE',
        'USE SCHEMA OTHER_DB.S; DROP TABLE OTHER_DB.S.T',
        'ALTER SESSION SET AUTOCOMMIT = FALSE',
        'BEGIN',
        'CALL OTHER_DB.S.PROC()',
        "EXECUTE IMMEDIATE 'DROP TABLE OTHER_DB.S.T'",
        "CREATE TABLE IDENTIFIER('REVIEW_DB.LIVE_TESTS.T') (i INT)",
        'WITH x AS (SELECT 1) INSERT INTO OTHER_DB.S.T SELECT * FROM x',
        # Codex r4: COPY's PURGE and a task's body.
        'COPY INTO REVIEW_DB.LIVE_TESTS.T FROM @OTHER_DB.PUBLIC.STAGE PURGE=TRUE',
        "CREATE TASK REVIEW_DB.LIVE_TESTS.T SCHEDULE = '1 MINUTE' AS "
        'DELETE FROM OTHER_DB.PUBLIC.T',
        'ALTER TASK REVIEW_DB.LIVE_TESTS.T RESUME',
        "SELECT SYSTEM$CANCEL_QUERY('outside-query-id')",
        # Codex r4: the CALL's schema argument as an expression.
        "CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML('REVIEW_DB.LIVE_TESTS' || "
        "'_OTHER', $$name: v$$)",
        # Claude r4: comments the engine reads differently.
        "SELECT 1 /**/// '\n; DROP TABLE OTHER.PROD.T --'",
        '-- a comment\nSELECT 1',
        'SELECT 1 // a comment',
        'SELECT 1 /* a comment */',
        # Claude r4: deferred execution and state-changing functions.
        "CREATE TASK REVIEW_DB.LIVE_TESTS.T WAREHOUSE = WH SCHEDULE = '1 MINUTE' "
        'AS DELETE FROM OTHER.PROD.T',
        "CREATE ALERT REVIEW_DB.LIVE_TESTS.A WAREHOUSE = WH SCHEDULE = '1 MINUTE' "
        'IF (EXISTS (SELECT 1)) THEN DROP TABLE OTHER.PROD.T',
        'CREATE PIPE REVIEW_DB.LIVE_TESTS.P AUTO_INGEST = TRUE AS COPY INTO '
        'OTHER.PROD.T FROM @REVIEW_DB.LIVE_TESTS.S',
        'CREATE PROCEDURE REVIEW_DB.LIVE_TESTS.P() RETURNS INT LANGUAGE SQL AS $$ SELECT 1 $$',
        "SELECT SYSTEM$PIPE_FORCE_RESUME('OTHER.PROD.P')",
        "SELECT SYSTEM$TASK_DEPENDENTS_ENABLE('OTHER.PROD.ROOT_TASK')",
        'SHOW TABLES ->> DROP TABLE OTHER.PROD.T',
        'SELECT 1 ->> SELECT 2',
        # Claude r4: the CALL's schema string, YAML name and tables.
        "CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML('\\tEVIEW_DB.LIVE_TESTS', $$name: v$$)",
        "CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML('REVIEW_DB.LIVE_TESTS', "
        '$$name: OTHER.PROD.V\ntables: []$$)',
        "CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML('REVIEW_DB.LIVE_TESTS', "
        '$$name: v\ntables:\n  - name: t\n    base_table:\n      database: OTHER\n'
        '      schema: PROD\n      table: T\n$$)',
        "CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML('REVIEW_DB.LIVE_TESTS', "
        '$$name: v$$, TRUE)',
        "CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML('OTHER_DB.S', $$name: v$$)",
        # Quoted names holding their own quote; a second statement.
        'DROP TABLE "REVIEW_DB"."LIVE_TESTS"."a""b"',
        'SELECT 1; SELECT 2',
        'SELECT 1;',
        "SELECT 'x\\'; DROP TABLE OTHER.PROD.T",
        "SELECT 'x; DROP TABLE OTHER_DB.S.T",
    ],
    'databricks': [
        'CREATE TABLE `other`.`live_tests`.t (i INT)',
        'CREATE TABLE "review-catalog"."live_tests".t (i INT)',
        'CREATE TABLE review-catalog.live_tests.t (i INT)',
        'DROP TABLE live_tests.t',
        "INSERT OVERWRITE DIRECTORY '/tmp/x' SELECT 1",
        "CREATE TABLE `review-catalog`.live_tests.t (i INT) LOCATION 's3://x'",
        'DROP SCHEMA `review-catalog`.live_tests CASCADE',
        # Claude r4: a raw string and nested comments.
        "WITH a AS (SELECT r'\\') INSERT INTO other.prod.t SELECT 1 --'",
        "WITH a AS (SELECT r'x') SELECT 1",
        "WITH a AS (SELECT 1) /* /* */ ' */ INSERT INTO other.prod.t SELECT 1 --'",
        # Claude r4 P2-1: OPTIONS (path ...).
        'CREATE TABLE `review-catalog`.live_tests.t USING DELTA OPTIONS '
        "(path 's3://bucket/other/prod/t') AS SELECT 1 a",
        'SELECT 1 |> WHERE TRUE',
        'MERGE INTO `review-catalog`.live_tests.t USING `other`.s.u ON TRUE '
        'WHEN MATCHED THEN DELETE',
        'CALL `review-catalog`.live_tests.p()',
    ],
    'postgres': [
        # Codex r4: more than one target, a second statement, CASCADE.
        'DROP TABLE IF EXISTS "live_tests"."a", "other"."b"',
        'DROP TABLE IF EXISTS "live_tests"."a"; DROP TABLE "other"."b"',
        'DROP TABLE IF EXISTS "live_tests"."a" CASCADE',
        'DROP TYPE IF EXISTS "live_tests"."t" CASCADE',
        'CREATE TABLE "live_tests"."t" (a int); DROP TABLE public.x',
        'DROP TABLE "outside"."t"',
        'CREATE TABLE "t" (a int)',
        'COPY "live_tests"."t" TO PROGRAM \'rm -rf /\'',
        'COPY "live_tests"."t" FROM \'/etc/passwd\'',
        'WITH x AS (DELETE FROM other.t RETURNING *) SELECT * FROM x',
        'SELECT * INTO other.t FROM live_tests.t',
        "SELECT set_config('search_path', 'other', false)",
        "SELECT nextval('other.seq')",
        'SELECT $a$ $$ $a$',
        "SELECT E'x\\'; DROP TABLE other.t; --'",
        "SELECT U&'d\\0061t\\+000061'",
        'CREATE TABLE "live_tests"."t" (a int DEFAULT nextval(\'other.s\'))',
    ],
}


@pytest.mark.parametrize(
    ('dialect', 'sql'),
    [(d, sql) for d, texts in ALLOWED.items() for sql in texts],
)
def test_the_admitted_reads_and_test_schema_writes_pass(
    dialect: str, sql: str
) -> None:
    GUARDS[dialect].check(sql)


@pytest.mark.parametrize(
    ('dialect', 'sql'),
    [(d, sql) for d, texts in REFUSED.items() for sql in texts],
)
def test_everything_else_is_refused(dialect: str, sql: str) -> None:
    with pytest.raises(wg.WriteRefused):
        GUARDS[dialect].check(sql)


def test_the_test_schema_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for dialect in ('', 'SNOWFLAKE_', 'POSTGRES_'):
        for part in ('DATABASE', 'SCHEMA'):
            monkeypatch.delenv(f'SDM_LIVE_{dialect}TEST_{part}', raising=False)
    with pytest.raises(wg.WriteRefused, match='no test schema'):
        wg.WriteGuard('snowflake')
    assert wg.from_environment() == {d: None for d in wg.DIALECTS}
    with pytest.raises(ValueError):
        wg.WriteGuard('mysql', 'A', 'B')
    monkeypatch.setenv('SDM_LIVE_TEST_DATABASE', 'REVIEW_DB')
    monkeypatch.setenv('SDM_LIVE_TEST_SCHEMA', 'LIVE_TESTS')
    wg.WriteGuard('snowflake').check('DROP TABLE REVIEW_DB.LIVE_TESTS.T')
    # A warehouse's own variables come first; Postgres needs no database.
    monkeypatch.setenv('SDM_LIVE_POSTGRES_TEST_SCHEMA', 'pg_tests')
    wg.WriteGuard('postgres').check('DROP TABLE pg_tests.t')
    with pytest.raises(wg.WriteRefused):
        wg.WriteGuard('postgres').check('DROP TABLE live_tests.t')


def test_every_case_the_loader_writes_passes_in_the_test_schema_only() -> None:
    # Every statement the loader writes for every S1 case passes in the
    # test schema, on each warehouse and on Lakebase; in another schema
    # every write is refused.
    runner = _load(
        't33_guard_run_python', ROOT / 'contract/runners/run_python.py'
    )
    bases = runner.load_bases()
    checked = 0
    for path in sorted((ROOT / 'contract/cases').glob('s1/**/*.json')):
        for case, _ in runner.load_cases([path]):
            duck = runner._connection(case, bases)
            try:
                for engine, namespace in (
                    ('snowflake', ('REVIEW_DB', 'LIVE_TESTS')),
                    ('databricks', ('review-catalog', 'live_tests')),
                ):
                    try:
                        ours, _ = databases.warehouse_statements(
                            duck, engine, namespace
                        )
                        theirs, _ = databases.warehouse_statements(
                            duck, engine, ('OTHER', 'S')
                        )
                    except databases.Unloadable:
                        continue
                    for statement in ours:
                        _admitted(GUARDS[engine], statement, case['id'])
                    for statement in theirs:
                        with pytest.raises(wg.WriteRefused):
                            GUARDS[engine].check(statement)
                    checked += 1
                for statement in _postgres_statements(duck, 'live_tests'):
                    _admitted(POSTGRES, statement, case['id'])
                for statement in _postgres_statements(duck, 'other'):
                    with pytest.raises(wg.WriteRefused):
                        POSTGRES.check(statement)
            finally:
                duck.close()
    assert checked > 500, checked


#: The cases whose names hold their own quote or a line break, which the
#: guard's one strict name pattern refuses (the owner's allowlist
#: decision): never recorded live.
QUOTE_CASES = ('name.backtick-', 'name.line-break-', 'name.double-quote-')


def _admitted(guard: Any, statement: str, case: str) -> None:
    try:
        guard.check(statement)
    except wg.WriteRefused as refused:
        assert case.startswith(QUOTE_CASES), (
            case,
            str(refused),
            statement[:200],
        )
        assert 'quoted name' in str(refused), refused


def _postgres_statements(duck: Any, schema: str) -> list[str]:
    r"""What the loader runs on Lakebase for ``duck``'s tables, recorded
    on a stand-in connection; an unloadable case writes none."""
    ran: list[str] = []

    class Copy:
        def __enter__(self) -> Copy:
            return self

        def __exit__(self, *exc: Any) -> None:
            pass

        def write_row(self, row: Any) -> None:
            pass

    class Cursor:
        def copy(self, statement: str) -> Copy:
            ran.append(statement)
            return Copy()

    class Connection:
        def execute(self, statement: str, *arguments: Any) -> Any:
            ran.append(statement)
            return self

        def cursor(self) -> Cursor:
            return Cursor()

        def commit(self) -> None:
            pass

    try:
        databases._load_postgres(duck, Connection(), schema, replace=True)
    except databases.Unloadable:
        return []
    return ran


def test_the_loaders_semantic_view_passes_and_its_drop() -> None:
    # The semantic view case's own CALL, as CaseSchema writes it.
    case = json.loads((ROOT / 'tests/t33-semantic-view-case.json').read_text())
    view = case['input']['view']
    ran: list[str] = []
    schema = databases.CaseSchema(
        'snowflake', ran.append, ('REVIEW_DB', 'LIVE_TESTS')
    )
    schema.tables = ['users', 'orders']
    yaml = (
        view['yaml']
        .replace('{catalog}', 'REVIEW_DB')
        .replace('{schema}', 'LIVE_TESTS')
    )
    schema.create_view('semantic_view', view['name'], yaml)
    schema.clear()
    assert ran[0].startswith('CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML(')
    for statement in ran:
        SNOWFLAKE.check(statement)
    # The same YAML naming another schema's tables is refused.
    elsewhere = ran[0].replace('database: REVIEW_DB', 'database: OTHER_DB')
    assert elsewhere != ran[0]
    with pytest.raises(wg.WriteRefused, match='outside the test schema'):
        SNOWFLAKE.check(elsewhere)


# The drivers.


def _fake_drivers(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    r"""Snowflake's, Snowpark's, Databricks SQL's and Spark's classes as
    small fakes under their modules' names, each recording what reaches
    it."""
    seen: list[str] = []

    class SnowflakeCursor:
        def execute(self, command: str, *a: Any, **k: Any) -> SnowflakeCursor:
            seen.append(command)
            return self

        def executemany(self, command: str, seq: Any) -> None:
            seen.append(command)

        def execute_async(self, command: str) -> dict[str, str]:
            seen.append(command)
            return {'queryId': 'q'}

    class DictCursor(SnowflakeCursor):
        pass

    class OwnCursor(SnowflakeCursor):
        def execute(self, command: str, *a: Any, **k: Any) -> OwnCursor:
            seen.append('own:' + command)
            return self

    class SnowflakeConnection:
        def cursor(self) -> SnowflakeCursor:
            return SnowflakeCursor()

        def execute_string(self, text: str) -> list[SnowflakeCursor]:
            seen.append(text)
            return [SnowflakeCursor()]

        def execute_stream(self, stream: Any) -> list[Any]:
            seen.append('stream')
            return []

    class Session:
        connection = SnowflakeConnection()

        def sql(self, query: str) -> str:
            seen.append(query)
            return query

    class Cursor:
        def execute(self, operation: str, parameters: Any = None) -> None:
            seen.append(operation)

    class SparkSession:
        def sql(self, sqlQuery: str, args: Any = None) -> str:
            seen.append(sqlQuery)
            return sqlQuery

    class DataFrame:
        @property
        def write(self) -> str:
            seen.append('write')
            return 'writer'

        def writeTo(self, table: str) -> None:
            seen.append('writeTo')

    class Catalog:
        def createTable(self, name: str) -> None:
            seen.append('createTable')

    for module, classes in {
        'snowflake.connector.cursor': [SnowflakeCursor, DictCursor, OwnCursor],
        'snowflake.connector.connection': [SnowflakeConnection],
        'snowflake.snowpark.session': [Session],
        'databricks.sql.client': [Cursor],
        'pyspark.sql.session': [SparkSession],
        'pyspark.sql.dataframe': [DataFrame],
        'pyspark.sql.catalog': [Catalog],
    }.items():
        fake = types.ModuleType(module)
        for cls in classes:
            setattr(fake, cls.__name__, cls)
        monkeypatch.setitem(sys.modules, module, fake)
    return {'seen': seen, **{c.__name__: c for c in (
        SnowflakeCursor, DictCursor, OwnCursor, SnowflakeConnection, Session,
        Cursor, SparkSession, DataFrame, Catalog,
    )}}  # fmt: skip


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch) -> Any:
    drivers = _fake_drivers(monkeypatch)
    for name in ('pyspark.sql.connect.session', 'pyspark.sql.connect.dataframe',
                 'pyspark.sql.connect.catalog'):  # fmt: skip
        monkeypatch.setitem(sys.modules, name, None)  # absent: not importable
    undo = wg.install(GUARDS)
    try:
        yield drivers
    finally:
        undo()


OUTSIDE = 'DROP TABLE OTHER_DB.PUBLIC.T'


def test_every_snowflake_entry_point_is_guarded(installed: Any) -> None:
    d, seen = installed, installed['seen']
    connection = d['SnowflakeConnection']()
    calls = [
        lambda: connection.cursor().execute(OUTSIDE),
        lambda: connection.cursor().execute(command=OUTSIDE),
        lambda: connection.cursor().executemany(OUTSIDE, [()]),
        lambda: connection.cursor().execute_async(OUTSIDE),
        lambda: d['DictCursor']().execute(OUTSIDE),
        lambda: d['OwnCursor']().execute(OUTSIDE),
        lambda: connection.execute_string(OUTSIDE),
        lambda: connection.execute_stream(None),
        # Codex r4 P0-2: execute_string's cursors are the driver's own, and
        # guarded as every cursor is.
        lambda: connection.execute_string('SELECT 1')[0].execute(OUTSIDE),
        # Codex r4 P0-1, Claude r4 P1-1: Snowpark's sql and its connection.
        lambda: d['Session']().sql(OUTSIDE),
        lambda: d['Session']().connection.cursor().execute(OUTSIDE),
    ]
    for call in calls:
        with pytest.raises(wg.WriteRefused):
            call()
    assert seen == ['SELECT 1']
    connection.cursor().execute('SELECT 1')
    d['OwnCursor']().execute('DROP TABLE REVIEW_DB.LIVE_TESTS.T')
    assert seen[1:] == ['SELECT 1', 'own:DROP TABLE REVIEW_DB.LIVE_TESTS.T']


def test_every_databricks_and_spark_entry_point_is_guarded(
    installed: Any,
) -> None:
    d, seen = installed, installed['seen']
    for call in (
        lambda: d['Cursor']().execute('DROP TABLE `other`.s.t'),
        lambda: d['SparkSession']().sql('DROP TABLE `other`.s.t'),
        lambda: d['SparkSession']().sql(sqlQuery='DROP TABLE `other`.s.t'),
        lambda: d['DataFrame']().write,
        lambda: d['DataFrame']().writeTo('`other`.s.t'),
        lambda: d['Catalog']().createTable('t'),
    ):
        with pytest.raises(wg.WriteRefused):
            call()
    assert seen == []
    d['SparkSession']().sql('USE CATALOG `review-catalog`')
    assert seen == ['USE CATALOG `review-catalog`']


def test_install_is_undone_and_a_dialect_with_no_schema_refuses_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d = _fake_drivers(monkeypatch)
    original = d['SnowflakeCursor'].__dict__['execute']
    undo = wg.install({'snowflake': None})
    try:
        with pytest.raises(wg.WriteRefused, match='no snowflake test schema'):
            d['SnowflakeCursor']().execute('SELECT 1')
        with pytest.raises(RuntimeError, match='installed already'):
            wg.install(GUARDS)
    finally:
        undo()
    assert d['SnowflakeCursor'].__dict__['execute'] is original
    assert 'execute' not in d['DictCursor'].__dict__
    d['SnowflakeCursor']().execute('SELECT 1')
    # A driver module whose class moved stops the install, undoing it.
    monkeypatch.setitem(
        sys.modules, 'databricks.sql.client', types.ModuleType('x')
    )
    with pytest.raises(RuntimeError, match='no class Cursor'):
        wg.install(GUARDS)
    assert d['SnowflakeCursor'].__dict__['execute'] is original


def test_psycopgs_own_classes_are_guarded() -> None:
    # Codex r4 P0-1/P0-4, Claude r4 P1-1: Lakebase's psycopg connection,
    # its cursors (a subclass's included) and COPY, on psycopg's real
    # classes; a refused statement never reaches them.
    import psycopg
    from psycopg import sql

    def bare(cls: type) -> Any:
        # Unconnected, and closed, so its finaliser has nothing to do.
        made = object.__new__(cls)
        made._closed = True
        return made

    undo = wg.install(GUARDS)
    try:
        outside = 'DROP TABLE "outside"."t"'
        composed = sql.SQL('DROP TABLE {}').format(
            sql.Identifier('outside', 't')
        )
        for call in (
            lambda: psycopg.Connection.execute(
                bare(psycopg.Connection), outside
            ),
            lambda: psycopg.Cursor.execute(bare(psycopg.Cursor), outside),
            lambda: psycopg.Cursor.execute(bare(psycopg.Cursor), composed),
            lambda: psycopg.ClientCursor.execute(
                bare(psycopg.ClientCursor), outside
            ),
            lambda: psycopg.ServerCursor.execute(
                bare(psycopg.ServerCursor), outside
            ),
            lambda: psycopg.Cursor.executemany(
                bare(psycopg.Cursor), outside, []
            ),
            lambda: psycopg.Cursor.copy(
                bare(psycopg.Cursor), 'COPY "outside"."t" FROM STDIN'
            ),
            lambda: psycopg.AsyncCursor.execute(
                bare(psycopg.AsyncCursor), outside
            ),
        ):
            with pytest.raises(wg.WriteRefused):
                call()
        # An admitted statement reaches the driver (which, unconnected, fails).
        with pytest.raises(Exception) as reached:
            psycopg.Cursor.execute(
                bare(psycopg.Cursor), 'DROP TABLE "live_tests"."t"'
            )
        assert not isinstance(reached.value, wg.WriteRefused)
    finally:
        undo()
    assert not hasattr(psycopg.Cursor.execute, '__sdm_write_guard__')
