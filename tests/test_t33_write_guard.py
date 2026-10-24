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
import re
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
capture_run = _load(
    't33_guard_capture_run', ROOT / 'scripts/live/capture_run.py'
)
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
        'create or replace table review_db.live_tests.t (i number(38, 0))',
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
    # test schema, on each warehouse and on Lakebase, but for the cases the
    # live plan leaves out by name on that leg (capture_run.NOT_LIVE, Claude
    # r9 P1-2), each refused for its quoted name; in another schema every
    # write is refused.
    runner = _load(
        't33_guard_run_python', ROOT / 'contract/runners/run_python.py'
    )
    bases = runner.load_bases()
    checked = 0
    for path in sorted((ROOT / 'contract/cases').glob('s1/**/*.json')):
        for case, _ in runner.load_cases([path]):
            duck = runner._connection(case, bases)
            try:
                for engine, namespace, leg in (
                    ('snowflake', ('REVIEW_DB', 'LIVE_TESTS'), 'snowflake'),
                    (
                        'databricks',
                        ('review-catalog', 'live_tests'),
                        'databricks-sql',
                    ),
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
                    _admitted(GUARDS[engine], ours, case['id'], leg)
                    for statement in theirs:
                        with pytest.raises(wg.WriteRefused):
                            GUARDS[engine].check(statement)
                    checked += 1
                _admitted(
                    POSTGRES,
                    _postgres_statements(duck, 'live_tests'),
                    case['id'],
                    'lakebase',
                )
                for statement in _postgres_statements(duck, 'other'):
                    with pytest.raises(wg.WriteRefused):
                        POSTGRES.check(statement)
            finally:
                duck.close()
    assert checked > 500, checked


def _admitted(guard: Any, statements: list[str], case: str, leg: str) -> None:
    r"""Every statement passes, unless ``case`` is left out of ``leg``'s
    live plan by name, when one at least is refused, for its quoted name."""
    refused = []
    for statement in statements:
        try:
            guard.check(statement)
        except wg.WriteRefused as refusal:
            assert 'quoted name' in str(refusal), (
                case,
                str(refusal),
                statement[:200],
            )
            refused.append(statement)
    assert bool(refused) == (
        leg in capture_run.NOT_LIVE.get(case, ((), ''))[0]
    ), (
        case,
        leg,
        refused[:1],
    )


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

    # As the connector (4.x) has them: every entry point on
    # SnowflakeCursorBase, SnowflakeCursor and DictCursor siblings below it
    # (Claude r5).
    class SnowflakeCursorBase:
        def execute(self, command: str, *a: Any, **k: Any) -> Any:
            seen.append(command)
            return self

        def executemany(self, command: str, seq: Any) -> None:
            seen.append(command)

        def execute_async(self, command: str) -> dict[str, str]:
            seen.append(command)
            return {'queryId': 'q'}

    class SnowflakeCursor(SnowflakeCursorBase):
        pass

    class DictCursor(SnowflakeCursorBase):
        pass

    class OwnCursor(SnowflakeCursor):
        def execute(self, command: str, *a: Any, **k: Any) -> OwnCursor:
            seen.append('own:' + command)
            return self

    class SnowflakeConnection:
        def cursor(self) -> SnowflakeCursor:
            # As the connector's: a cursor knows its connection.
            made = SnowflakeCursor()
            made.connection = self  # type: ignore[attr-defined]
            return made

        def execute_string(self, text: str) -> list[SnowflakeCursor]:
            seen.append(text)
            return [SnowflakeCursor()]

        def execute_stream(self, stream: Any) -> list[Any]:
            seen.append('stream')
            return []

        def cmd_query(
            self, sql: str, sequence_counter: int, request_id: Any
        ) -> Any:
            seen.append(sql)
            return {}

    class Session:
        _session_id = 42
        # What close sends to cancel the session's queries (Snowpark's
        # cancel_all, session.py).
        closing = 'select system$cancel_all_queries(42)'

        def __init__(self) -> None:
            # As Snowpark's: the session's server connection (_conn) holds
            # the connector's connection (_conn).
            self.connection = SnowflakeConnection()
            self._conn = types.SimpleNamespace(_conn=self.connection)

        def sql(self, query: str) -> str:
            seen.append(query)
            return query

        def cancel_all(self) -> None:
            self.connection.cursor().execute(self.closing)

        def close(self) -> None:
            self.cancel_all()

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
        'snowflake.connector.cursor': [
            SnowflakeCursorBase,
            SnowflakeCursor,
            DictCursor,
            OwnCursor,
        ],
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
        SnowflakeCursorBase, SnowflakeCursor, DictCursor, OwnCursor,
        SnowflakeConnection, Session,
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
        lambda: d['DictCursor']().executemany(OUTSIDE, [()]),
        lambda: d['DictCursor']().execute_async(OUTSIDE),
        lambda: d['OwnCursor']().execute(OUTSIDE),
        lambda: connection.execute_string(OUTSIDE),
        lambda: connection.execute_stream(None),
        # Claude r6 P3-2: the connection's public REST entry point.
        lambda: connection.cmd_query(OUTSIDE, 1, None),
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
    original = d['SnowflakeCursorBase'].__dict__['execute']
    undo = wg.install({'snowflake': None})
    try:
        with pytest.raises(wg.WriteRefused, match='no snowflake test schema'):
            d['SnowflakeCursor']().execute('SELECT 1')
        with pytest.raises(RuntimeError, match='installed already'):
            wg.install(GUARDS)
    finally:
        undo()
    assert d['SnowflakeCursorBase'].__dict__['execute'] is original
    assert 'execute' not in d['DictCursor'].__dict__
    d['SnowflakeCursor']().execute('SELECT 1')
    # A driver module whose class moved stops the install, undoing it.
    monkeypatch.setitem(
        sys.modules, 'databricks.sql.client', types.ModuleType('x')
    )
    with pytest.raises(RuntimeError, match='no class Cursor'):
        wg.install(GUARDS)
    assert d['SnowflakeCursorBase'].__dict__['execute'] is original


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


# Round 5: reads are an allowlist too, and Spark's writers are refused.

#: Each round 5 bypass, refused.
REFUSED_R5 = {
    'postgres': [
        # Codex r5 P0-1: a quoted or qualified function name.
        'SELECT pg_catalog."setval"(\'outside.production_id_seq\', 1, false)',
        'SELECT "nextval"(\'outside.production_id_seq\')',
        "SELECT pg_catalog.setval('outside.s', 1)",
        'SELECT "setval"(\'outside.s\', 1) FROM live_tests.t',
        # A function on no list.
        'SELECT pg_sleep(1)',
        'SELECT lo_unlink(1)',
        "SELECT set_config('search_path', 'x', false)",
    ],
    'snowflake': [
        # Codex r5 P0-2: a sequence advanced outside the test schema.
        'SELECT OTHER_DB.PUBLIC.PRODUCTION_ID_SEQ.NEXTVAL',
        'SELECT OTHER_DB.PUBLIC.PRODUCTION_ID_SEQ.nextval AS n',
        'INSERT INTO REVIEW_DB.LIVE_TESTS.T SELECT OTHER_DB.PUBLIC.PRODUCTION_ID_SEQ.NEXTVAL',
        'SELECT OTHER_DB.PUBLIC.S.GETNEXTVAL()',
        'SELECT * FROM TABLE(GETNEXTVAL(OTHER_DB.PUBLIC.S))',
        # Functions on no list.
        'SELECT UUID_STRING()',
        'SELECT SEQ8()',
        'SELECT SYSTEM$WAIT(1)',
        "SELECT SECRET('x')",
        'SELECT "LOWER"(\'A\')',
        'SELECT m!PREDICT(1)',
        'INSERT INTO REVIEW_DB.LIVE_TESTS.T SELECT UUID_STRING()',
        # Claude r5: a YAML key given twice, read differently by the engine.
        "CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML('REVIEW_DB.LIVE_TESTS', "
        '$$name: v\nname: OTHER.PROD.V\n$$)',
        "CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML('REVIEW_DB.LIVE_TESTS', "
        '$$name: v\ntables:\n  - name: t\n    base_table:\n      database: OTHER\n'
        '      database: REVIEW_DB\n      schema: LIVE_TESTS\n      table: T\n$$)',
    ],
    'databricks': [
        "SELECT http_request('x')",
        "SELECT secret('scope', 'key')",
        'SELECT `lower`(a) FROM `other`.s.t',
    ],
}

#: Reads the old client's generated SQL makes, admitted.
ALLOWED_R5 = {
    'postgres': [
        "SELECT a ->> 'k' FROM live_tests.t",
        'SELECT COUNT(*), MIN(a), MAX(a) FROM other.t',
        'SELECT format_type(a.atttypid, a.atttypmod) FROM pg_catalog.pg_attribute a',
        'SELECT ROW_NUMBER() OVER (PARTITION BY a ORDER BY b) AS n FROM other.t',
        'SELECT CAST(a AS numeric(20, 0)), a::numeric(20, 0) FROM other.t',
    ],
    'snowflake': [
        'SELECT COUNT(*) FROM "OTHER"."S"."T" SAMPLE ROW (5 ROWS)',
        'SELECT t.value FROM other.s.x, LATERAL FLATTEN(INPUT => PARSE_JSON(x.v)) t',
        "SELECT DATE_TRUNC('day', ts), DATEADD('day', 1, ts) FROM other.s.t",
        'INSERT INTO "REVIEW_DB"."LIVE_TESTS"."T" ("A") SELECT '
        "PARSE_JSON(HEX_DECODE_STRING(column1))::VARIANT FROM VALUES ('6869')",
    ],
    'databricks': [
        'SELECT * FROM `other`.s.t TABLESAMPLE (5 PERCENT) REPEATABLE (1)',
        "SELECT posexplode(from_json(v, 'array<int>')) FROM `other`.s.t",
        'SELECT current_catalog(), current_schema()',
    ],
}


@pytest.mark.parametrize(
    ('dialect', 'sql'),
    [(d, sql) for d, texts in REFUSED_R5.items() for sql in texts],
)
def test_round_5_bypasses_are_refused(dialect: str, sql: str) -> None:
    with pytest.raises(wg.WriteRefused):
        GUARDS[dialect].check(sql)


@pytest.mark.parametrize(
    ('dialect', 'sql'),
    [(d, sql) for d, texts in ALLOWED_R5.items() for sql in texts],
)
def test_the_old_clients_reads_pass(dialect: str, sql: str) -> None:
    GUARDS[dialect].check(sql)


def test_every_statement_the_old_client_recorded_on_postgres_passes() -> None:
    # Every statement the reference recorded on Postgres, the old client's
    # own generated SQL (contract/reference), passes the Postgres guard: the
    # read allowlist holds what the old client sends.
    guard = wg.WriteGuard('postgres', None, 'public')
    checked = 0
    for path in (ROOT / 'contract/reference').glob('*.jsonl'):
        for line in path.read_text(encoding='utf-8').splitlines():
            record = json.loads(line)
            if not record['case'].endswith('@postgres'):
                continue
            capture = (record.get('driver') or {}).get('capture') or {}
            for statement in capture.get('statements') or []:
                try:
                    guard.check(statement['sql'])
                except wg.WriteRefused as refused:
                    # A case left out of Lakebase's live plan, for its name.
                    case = record['case'].partition('@')[0]
                    assert 'lakebase' in capture_run.NOT_LIVE[case][0], refused
                    assert 'quoted name' in str(refused), refused
                checked += 1
    assert checked > 1000, checked


def test_a_quoted_function_never_reaches_psycopg() -> None:
    # Codex r5 P0-1, through the installed driver boundary.
    import psycopg

    undo = wg.install(GUARDS)
    try:
        cursor = object.__new__(psycopg.Cursor)
        cursor._closed = True
        with pytest.raises(wg.WriteRefused):
            psycopg.Cursor.execute(
                cursor,
                'SELECT pg_catalog."setval"(\'outside.production_id_seq\', 1, false)',
            )
    finally:
        undo()


def _fake_spark_writers(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    r"""Spark's and Delta's writer classes as small fakes under their
    modules' names, each recording a write that reaches it."""
    seen: list[str] = []

    def make(name: str, methods: tuple[str, ...]) -> type:
        def method(label: str) -> Any:
            def call(self: Any, *a: Any, **k: Any) -> Any:
                seen.append(label)
                return self

            return call

        namespace = {m: method(f'{name}.{m}') for m in methods}
        return type(name, (), namespace)

    writer = make(
        'DataFrameWriter', ('mode', 'saveAsTable', 'insertInto', 'save')
    )
    v2 = make('DataFrameWriterV2', ('using', 'create', 'append', 'overwrite'))
    stream = make('DataStreamWriter', ('toTable', 'start'))
    merge = make('MergeIntoWriter', ('whenMatched', 'merge'))
    delta = make('DeltaTable', ('delete', 'update', 'merge', 'vacuum'))
    delta.forName = classmethod(
        lambda cls, *a: seen.append('DeltaTable.forName')
    )
    builder = make('DeltaMergeBuilder', ('execute',))
    for module, classes in {
        'pyspark.sql.readwriter': [writer, v2],
        'pyspark.sql.streaming.readwriter': [stream],
        'pyspark.sql.merge': [merge],
        'delta.tables': [delta, builder],
    }.items():
        fake = types.ModuleType(module)
        for cls in classes:
            setattr(fake, cls.__name__, cls)
        monkeypatch.setitem(sys.modules, module, fake)
    return {
        'seen': seen,
        **{c.__name__: c for c in (writer, v2, stream, merge, delta, builder)},
    }


def test_spark_and_delta_writers_are_refused_however_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Codex r5 P0-3: a writer kept from before the install, or made after
    # it, writes nothing.
    w = _fake_spark_writers(monkeypatch)
    kept = w['DataFrameWriter']()
    undo = wg.install(GUARDS)
    try:
        for call in (
            lambda: kept.mode('overwrite').saveAsTable('other.prod.accounts'),
            lambda: kept.saveAsTable('other.prod.accounts'),
            lambda: w['DataFrameWriter']().insertInto('other.prod.accounts'),
            lambda: w['DataFrameWriterV2']().append(),
            lambda: w['DataStreamWriter']().toTable('other.prod.t'),
            lambda: w['MergeIntoWriter']().merge(),
            lambda: w['DeltaTable']().delete(),
            lambda: w['DeltaTable'].forName(None, 'other.prod.t'),
            lambda: w['DeltaMergeBuilder']().execute(),
        ):
            with pytest.raises(wg.WriteRefused):
                call()
        assert w['seen'] == []
    finally:
        undo()
    kept.saveAsTable('t')
    assert w['seen'] == ['DataFrameWriter.saveAsTable']


def test_spark_sql_with_substituted_arguments_is_refused(
    installed: Any,
) -> None:
    # Codex r5 P0-3, Claude's lead: PySpark formats keyword arguments into
    # the text after the guard reads it; bound args are not substituted.
    session = installed['SparkSession']()
    with pytest.raises(wg.WriteRefused):
        session.sql('SELECT * FROM {t}', t='`other`.s.t')
    with pytest.raises(wg.WriteRefused):
        session.sql('DROP TABLE {t}', t='`other`.s.t')
    session.sql('SELECT ? AS a', args=[1])
    assert installed['seen'] == ['SELECT ? AS a']


def test_a_driver_subclass_made_after_the_install_is_guarded(
    installed: Any,
) -> None:
    # Claude's lead: a subclass defined later, overriding an entry point.
    seen = installed['seen']

    class Later(installed['SnowflakeCursor']):
        def execute(self, command: str, *a: Any, **k: Any) -> Any:
            seen.append('later:' + command)
            return self

    with pytest.raises(wg.WriteRefused):
        Later().execute('DROP TABLE OTHER_DB.PUBLIC.T')
    Later().execute('SELECT 1')
    assert seen == ['later:SELECT 1']


def test_spark_connect_admits_only_checked_sql_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Claude r5 P1: every command a Connect session sends passes the
    # client's execution boundary; only a SQL command, its text checked,
    # goes through.
    seen: list[Any] = []

    class SparkConnectClient:
        def _execute_and_fetch_as_iterator(self, req: Any, *a: Any) -> Any:
            seen.append(req.plan.command.WhichOneof('command_type'))
            return iter(())

        def _execute(self, req: Any) -> None:
            seen.append(req.plan.command.WhichOneof('command_type'))

        def execute_command(self, command: Any) -> Any:
            return list(
                self._execute_and_fetch_as_iterator(
                    _Message(plan=_Message(command=command))
                )
            )

    def sql(text: str) -> Any:
        return _Message(
            sql_command=_Message(input=_Message(sql=_Message(query=text)))
        )

    fake = types.ModuleType('pyspark.sql.connect.client.core')
    fake.SparkConnectClient = SparkConnectClient
    monkeypatch.setitem(sys.modules, 'pyspark.sql.connect.client.core', fake)
    undo = wg.install(GUARDS)
    try:
        client = SparkConnectClient()
        for command in (
            _Message(write_operation=_Message(mode=1)),
            _Message(write_operation_v2=_Message(mode=1)),
            _Message(create_dataframe_view=_Message(name='v')),
            sql('DROP TABLE `other`.s.t'),
        ):
            with pytest.raises(wg.WriteRefused):
                client.execute_command(command)
            with pytest.raises(wg.WriteRefused):
                client._execute(_Message(plan=_Message(command=command)))
        client.execute_command(sql('SELECT 1'))
        assert seen == ['sql_command']
    finally:
        undo()


# Round 6: a table's columns are held to the function rule too.

REFUSED_R6 = {
    'postgres': [
        # Claude r6 P0: a CHECK calls a function on every INSERT.
        'CREATE TABLE live_tests.t (a int CHECK (prod.audit(a) > 0))',
        'CREATE TABLE live_tests.t (a int CHECK (audit(a) > 0))',
        'CREATE TABLE "live_tests"."t" ("a" int, CHECK (audit("a") > 0))',
        'CREATE TABLE live_tests.t (a int, CONSTRAINT c CHECK (lo_from_bytea(0, a::text::bytea) > 0))',
        'CREATE TABLE live_tests.t (a int GENERATED ALWAYS AS (audit(a)) STORED)',
        "CREATE TABLE live_tests.t (a oid DEFAULT lo_from_bytea(0, 'x'))",
        'CREATE TYPE live_tests.t AS (a int, b audit(1))',
        # Claude r6 P2-2: a function after AS or ::.
        'SELECT a::audit(1) FROM other.t',
        'SELECT CAST(a AS audit(1)) FROM other.t',
    ],
    'databricks': [
        'CREATE TABLE `review-catalog`.live_tests.t (a INT, CONSTRAINT c CHECK (prod_fn(a) > 0))',
        'CREATE TABLE `review-catalog`.live_tests.t (a INT GENERATED ALWAYS AS (prod_fn(a)))',
        'CREATE TABLE `review-catalog`.live_tests.t (a INT DEFAULT prod_fn(1))',
    ],
    'snowflake': [
        'CREATE TABLE REVIEW_DB.LIVE_TESTS.T (A INT DEFAULT OTHER.S.SEQ.NEXTVAL)',
        'CREATE TABLE REVIEW_DB.LIVE_TESTS.T (A INT AS (PROD_FN(1)))',
        'CREATE TABLE REVIEW_DB.LIVE_TESTS.T (A INT, CHECK (PROD_FN(A) > 0))',
        'SELECT CAST(A AS PROD_FN(1)) FROM OTHER.S.T',
    ],
}


@pytest.mark.parametrize(
    ('dialect', 'sql'),
    [(d, sql) for d, texts in REFUSED_R6.items() for sql in texts],
)
def test_round_6_column_functions_are_refused(dialect: str, sql: str) -> None:
    with pytest.raises(wg.WriteRefused):
        GUARDS[dialect].check(sql)


# Round 7: a table takes nothing from another object, and its columns only
# built-in types (or Postgres' composite types in the test schema).

REFUSED_R7 = {
    'postgres': [
        # Claude r7 P0: LIKE copies another table's defaults and checks.
        'CREATE TABLE live_tests.t (LIKE prod.orders INCLUDING ALL)',
        'CREATE TABLE live_tests.t (LIKE prod.orders INCLUDING DEFAULTS)',
        'CREATE TABLE live_tests.t (a int, LIKE prod.orders INCLUDING CONSTRAINTS)',
        'CREATE TABLE live_tests.t (a int) INHERITS (prod.orders)',
        'CREATE TABLE live_tests.t PARTITION OF prod.orders FOR VALUES IN (1)',
        'CREATE TABLE live_tests.t OF prod.row_type',
        # A domain or user type from another schema can carry a CHECK.
        'CREATE TABLE live_tests.t (a prod.positive_int)',
        'CREATE TABLE live_tests.t ("a" "prod"."positive_int")',
        'CREATE TABLE live_tests.t (a positive_int)',
        'CREATE TABLE live_tests.t (a public.positive_int[])',
        'CREATE TYPE live_tests.t AS (a prod.positive_int)',
    ],
    'databricks': [
        'CREATE TABLE `review-catalog`.live_tests.t (a prod_type)',
        'CREATE TABLE `review-catalog`.live_tests.t (a `prod`.`s`.`t`)',
        'CREATE TABLE `review-catalog`.live_tests.t (LIKE `other`.s.t)',
    ],
    'snowflake': [
        'CREATE TABLE REVIEW_DB.LIVE_TESTS.T (A OTHER.S.DOM)',
        'CREATE TABLE REVIEW_DB.LIVE_TESTS.T (A MYTYPE)',
        'CREATE TABLE REVIEW_DB.LIVE_TESTS.T LIKE OTHER.S.T',
    ],
}

ALLOWED_R7 = {
    'postgres': [
        'CREATE TABLE "live_tests"."t" ("a" "live_tests"."t_a", "b" bigint[], '
        '"c" double precision NOT NULL, "d" timestamp with time zone, '
        '"e" numeric(20, 0), "f" jsonb)',
    ],
    'databricks': [
        'CREATE TABLE `review-catalog`.`live_tests`.`t` (`a` STRUCT<`x`: INT, '
        '`y`: ARRAY<STRING>>, `b` MAP<STRING, BIGINT>, `c` DECIMAL(20,0) NOT NULL)',
    ],
}


@pytest.mark.parametrize(
    ('dialect', 'sql'),
    [(d, sql) for d, texts in REFUSED_R7.items() for sql in texts],
)
def test_round_7_tables_take_nothing_from_elsewhere(
    dialect: str, sql: str
) -> None:
    with pytest.raises(wg.WriteRefused):
        GUARDS[dialect].check(sql)


@pytest.mark.parametrize(
    ('dialect', 'sql'),
    [(d, sql) for d, texts in ALLOWED_R7.items() for sql in texts],
)
def test_the_loaders_column_types_pass(dialect: str, sql: str) -> None:
    GUARDS[dialect].check(sql)


def test_cmd_query_by_keyword_is_refused_by_the_rule(installed: Any) -> None:
    # Claude r7 P3-2: the statement given by keyword is read and judged.
    connection = installed['SnowflakeConnection']()
    with pytest.raises(wg.WriteRefused, match='outside the test schema'):
        connection.cmd_query(sql=OUTSIDE, sequence_counter=1, request_id=None)
    assert installed['seen'] == []


# Codex round 7: no exemption admits a qualified name or a call, names are
# plain ASCII, a cast is to a built-in type, and Spark's plans and global
# views are guarded.

REFUSED_C7 = {
    'postgres': [
        # P0-2: a qualified call whose last part looks like a type or keyword.
        'SELECT outside.text(7)',
        'SELECT outside.numeric(7)',
        'SELECT outside.by(7)',
        'SELECT pg_catalog.text(7)',
        # P0-2: an unreserved keyword or type called where a call can stand.
        'SELECT by(7)',
        'SELECT filter(7)',
        'SELECT text(7)',
        # P0-2: a name the tokenizer would split.
        'SELECT élower(7)',
        'SELECT * FROM other."tablé"',
        # P0-3: a cast to a domain or unknown type.
        'SELECT 7::outside.audit_domain',
        'SELECT CAST(7 AS outside.audit_domain)',
        'SELECT 7::audit_domain',
        'SELECT CAST(7 AS audit_domain)',
        'SELECT 7::"audit_domain"',
    ],
    'snowflake': [
        'SELECT OTHER.S.LOWER(1)',
        'SELECT 7::OTHER.S.DOM',
        'SELECT CAST(7 AS OTHER.S.DOM)',
        'SELECT CAST(7 AS MYTYPE)',
        'SELECT Ölower(1)',
    ],
    'databricks': [
        'SELECT `other`.s.lower(1)',
        'SELECT CAST(1 AS mytype)',
    ],
}


@pytest.mark.parametrize(
    ('dialect', 'sql'),
    [(d, sql) for d, texts in REFUSED_C7.items() for sql in texts],
)
def test_codex_round_7_calls_and_casts_are_refused(
    dialect: str, sql: str
) -> None:
    with pytest.raises(wg.WriteRefused):
        GUARDS[dialect].check(sql)


class _Message:
    r"""A protobuf message as the guard reads one: its set fields."""

    def __init__(self, **fields: Any) -> None:
        self.fields = fields
        for name, value in fields.items():
            setattr(self, name, value)

    def ListFields(self) -> list[tuple[Any, Any]]:  # noqa: N802 - protobuf's name
        return [
            (types.SimpleNamespace(name=k), v) for k, v in self.fields.items()
        ]

    def WhichOneof(self, group: str) -> str | None:  # noqa: N802
        return next(iter(self.fields), None)


def test_spark_connect_plans_are_guarded_at_their_common_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Codex r7 P0-4: to_table and every other plan path send a whole plan
    # past execute_command; a command in a plan, or a SQL relation's text,
    # is held to the rule there.
    seen: list[str] = []

    class SparkConnectClient:
        def _execute_and_fetch_as_iterator(
            self, req: Any, *a: Any, **k: Any
        ) -> Any:
            seen.append('fetch')
            return iter(())

        def _execute(self, req: Any) -> None:
            seen.append('execute')

        def to_table(self, plan: Any, observations: Any) -> Any:
            return list(
                self._execute_and_fetch_as_iterator(_Message(plan=plan))
            )

    def sql(text: str) -> Any:
        return _Message(sql=_Message(query=text))

    command = _Message(
        command=_Message(
            sql_command=_Message(input=sql('DROP TABLE `other`.s.t'))
        )
    )
    hidden = _Message(
        root=_Message(project=_Message(input=sql('DROP TABLE `other`.s.t')))
    )
    writer = _Message(command=_Message(write_operation=_Message(mode=1)))
    read = _Message(
        root=_Message(project=_Message(input=sql('SELECT * FROM `other`.s.t')))
    )
    fake = types.ModuleType('pyspark.sql.connect.client.core')
    fake.SparkConnectClient = SparkConnectClient
    monkeypatch.setitem(sys.modules, 'pyspark.sql.connect.client.core', fake)
    undo = wg.install(GUARDS)
    try:
        client = SparkConnectClient()
        for plan in (command, hidden, writer):
            with pytest.raises(wg.WriteRefused):
                client.to_table(plan, {})
            with pytest.raises(wg.WriteRefused):
                client._execute(_Message(plan=plan))
        client.to_table(read, {})
        assert seen == ['fetch']
    finally:
        undo()


def test_spark_global_and_temporary_views_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Codex r7 P0-5: views in Spark's application-wide namespace.
    seen: list[str] = []

    class DataFrame:
        def createOrReplaceGlobalTempView(self, name: str) -> None:  # noqa: N802
            seen.append(name)

        def createGlobalTempView(self, name: str) -> None:  # noqa: N802
            seen.append(name)

    class Catalog:
        def dropGlobalTempView(self, name: str) -> None:  # noqa: N802
            seen.append(name)

    for module, cls in (
        ('pyspark.sql.dataframe', DataFrame),
        ('pyspark.sql.catalog', Catalog),
    ):
        fake = types.ModuleType(module)
        setattr(fake, cls.__name__, cls)
        monkeypatch.setitem(sys.modules, module, fake)
    undo = wg.install(GUARDS)
    try:
        for call in (
            lambda: DataFrame().createOrReplaceGlobalTempView(
                'production_view'
            ),
            lambda: DataFrame().createGlobalTempView('production_view'),
            lambda: Catalog().dropGlobalTempView('production_view'),
        ):
            with pytest.raises(wg.WriteRefused):
                call()
        assert seen == []
    finally:
        undo()


def test_every_refusal_of_an_installed_guard_is_kept(installed: Any) -> None:
    # Claude r10 P1: a refusal the caller swallows (the old client's call
    # makes it the case's outcome) is still seen: install keeps each one.
    assert wg.REFUSALS == []
    try:
        installed['SnowflakeConnection']().cursor().execute(OUTSIDE)
    except wg.WriteRefused:
        pass  # what the old client's call does with it
    with pytest.raises(wg.WriteRefused):
        _ = installed['DataFrame']().write
    assert len(wg.REFUSALS) == 2
    assert 'outside the test schema' in wg.REFUSALS[0]


def test_the_old_clients_sampled_derived_table_read_passes() -> None:
    # C-1 (databricks-sql, rule.key-name-form): the old client samples a
    # renamed source as a derived table, `FROM (SELECT ...) TABLESAMPLE`.
    # That read passes; a write dressed with the same sample, and a
    # TABLESAMPLE after any other parenthesis, are still refused.
    read = (
        'SELECT `a`, `b`\n'
        'FROM (SELECT source.`id` AS `a`, source.`x` AS `b` '
        'FROM `other`.`s`.`t` AS source)\n'
        'TABLESAMPLE (100.0 PERCENT) REPEATABLE (42)\n'
        'WHERE `a` IS NOT NULL\nLIMIT 10000'
    )
    DATABRICKS.check(read)
    for sql in (
        'CREATE TABLE `other`.s.u AS SELECT * FROM (SELECT * FROM '
        '`other`.s.t) TABLESAMPLE (5 PERCENT)',
        'INSERT INTO `other`.s.u SELECT * FROM (SELECT * FROM '
        '`other`.s.t) TABLESAMPLE (5 PERCENT)',
        'SELECT * FROM `other`.s.t WHERE (a > 1) TABLESAMPLE (5 PERCENT)',
        'SELECT * FROM (VALUES (1)) TABLESAMPLE (5 PERCENT)',
    ):
        with pytest.raises(wg.WriteRefused):
            DATABRICKS.check(sql)


def test_the_old_clients_join_onto_a_derived_table_passes() -> None:
    # C-1 (databricks-sql, rule.key-name-form): the old client joins its
    # batch onto a renamed source as a derived table, `JOIN (SELECT ...)`.
    # That read passes; a write inside or around it, and a JOIN onto any
    # other parenthesis, are still refused.
    read = (
        'WITH TMP AS (\nSELECT * FROM `other`.`s`.`ids`\n)\n'
        'SELECT TMP.__KUMO_BATCH__ as __KUMO_BATCH__, `a`\nFROM TMP\n'
        'JOIN (SELECT source.`id` AS `a` FROM `other`.`s`.`t` AS source)\n'
        '  ON `a` = TMP.__KUMO_ID__'
    )
    DATABRICKS.check(read)
    for sql in (
        'INSERT INTO `other`.s.u SELECT * FROM `other`.s.v JOIN '
        '(SELECT * FROM `other`.s.t) ON 1 = 1',
        'CREATE TABLE `other`.s.u AS SELECT * FROM `other`.s.v JOIN '
        '(SELECT * FROM `other`.s.t) ON 1 = 1',
        'SELECT * FROM `other`.s.v JOIN (DELETE FROM `other`.s.t) ON 1 = 1',
        'SELECT * FROM `other`.s.v JOIN (VALUES (1)) ON 1 = 1',
    ):
        with pytest.raises(wg.WriteRefused):
            DATABRICKS.check(sql)


def test_the_old_clients_trusted_bitmap_sample_passes() -> None:
    # T3.20 second capture (view.metric.trusted-function): with
    # trust_expressions=True the old client samples the view's
    # bitmap_bucket_number dimension. That pure read passes; a write
    # calling it is still refused.
    DATABRICKS.check(
        'SELECT bitmap_bucket_number(source.`order_id`) AS `bucket`\n'
        'FROM `other`.`s`.`orders` AS source\nLIMIT 1000'
    )
    for sql in (
        'CREATE TABLE `other`.s.u AS SELECT bitmap_bucket_number(1) AS b',
        'INSERT INTO `other`.s.u SELECT bitmap_bucket_number(`id`) '
        'FROM `other`.s.t',
    ):
        with pytest.raises(wg.WriteRefused):
            DATABRICKS.check(sql)


def test_snowpark_cancels_its_own_queries_only_while_it_closes(
    installed: Any,
) -> None:
    # T3.20 second capture (snowflake.connect.snowpark): Session.close
    # sends select system$cancel_all_queries(<its id>). That one statement
    # passes while its session closes; sent at any other time, for another
    # session, or as anything else during close, it is refused.
    d, seen = installed, installed['seen']
    d['Session']().close()
    assert seen[-1] == 'select system$cancel_all_queries(42)'
    with pytest.raises(wg.WriteRefused):
        d['Session']().cancel_all()
    for sent in ('select system$cancel_all_queries(43)', OUTSIDE):
        session = d['Session']()
        session.closing = sent
        with pytest.raises(wg.WriteRefused):
            session.close()


def test_snowpark_close_admits_its_cancel_only_on_its_own_connection(
    installed: Any,
) -> None:
    # Codex T3.20 r1 P2: while a session closes, its cancel is admitted on
    # that session's own Snowflake connection alone. The same statement on
    # another Snowflake connection, a cursor of no connection, a Databricks
    # cursor or a Spark session is refused, during the close and outside it.
    d, seen = installed, installed['seen']
    statement = 'select system$cancel_all_queries(42)'
    other = d['SnowflakeConnection']()
    sends = [
        lambda: other.cursor().execute(statement),
        lambda: d['SnowflakeCursor']().execute(statement),
        lambda: d['Cursor']().execute(statement),
        lambda: d['SparkSession']().sql(statement),
    ]
    for send in sends:
        session = d['Session']()
        session.cancel_all = send  # type: ignore[method-assign]
        with pytest.raises(wg.WriteRefused):
            session.close()
        with pytest.raises(wg.WriteRefused):
            send()
    assert statement not in seen
    d['Session']().close()
    assert seen == [statement]
    # The recording mode hands Snowpark its capture of the connection,
    # which keeps the driver's connection as its _driver: the cursor the
    # capture makes is the driver connection's, and its cancel passes.
    session = d['Session']()
    capture = types.SimpleNamespace(
        _driver=session.connection, cursor=session.connection.cursor
    )
    session._conn = types.SimpleNamespace(_conn=capture)
    session.close()
    assert seen == [statement, statement]
    # The connector runs a cursor's statement through its connection's
    # cmd_query, which the guard checks too: the same connection passes.
    session = d['Session']()
    session.cancel_all = (  # type: ignore[method-assign]
        lambda: session.connection.cmd_query(statement, 1, None)
    )
    session.close()
    assert seen == [statement] * 3


def test_a_plain_view_is_admitted_only_in_the_loaders_setup_in_the_test_schema() -> (
    None
):
    # T3.20 second capture (snowflake.rowcount.view-as-table): the
    # loader's case setup creates a view over a case table in the test
    # schema. Admitted there; anywhere else, outside the setup, or over a
    # write, it is refused.
    view = (
        'CREATE VIEW REVIEW_DB.LIVE_TESTS.USERS_V AS '
        'SELECT * FROM REVIEW_DB.LIVE_TESTS.USERS'
    )
    with wg.setup():
        SNOWFLAKE.check(view)
        DATABRICKS.check(
            'CREATE VIEW `review-catalog`.live_tests.v AS '
            'SELECT * FROM `review-catalog`.live_tests.t'
        )
    refused = [
        (SNOWFLAKE, view, False),
        (
            SNOWFLAKE,
            'CREATE VIEW OTHER_DB.PUBLIC.USERS_V AS '
            'SELECT * FROM REVIEW_DB.LIVE_TESTS.USERS',
            True,
        ),
        (SNOWFLAKE, 'CREATE VIEW USERS_V AS SELECT 1', True),
        (
            SNOWFLAKE,
            'CREATE OR REPLACE VIEW REVIEW_DB.LIVE_TESTS.USERS_V AS SELECT 1',
            True,
        ),
        (
            SNOWFLAKE,
            'CREATE VIEW REVIEW_DB.LIVE_TESTS.V AS '
            'SELECT * FROM REVIEW_DB.LIVE_TESTS.T; DROP TABLE OTHER_DB.P.T',
            True,
        ),
    ]
    for guard, sql, in_setup in refused:
        with pytest.raises(wg.WriteRefused):
            if in_setup:
                with wg.setup():
                    guard.check(sql)
            else:
                guard.check(sql)


def test_a_temporary_table_is_admitted_only_in_the_loaders_lakebase_setup() -> (
    None
):
    # T3.20 second capture (lakebase.name.shadowed-by-temporary): the
    # loader's case setup creates a session-only table on the Lakebase
    # connection. Admitted there; outside the setup, in a schema, on
    # another dialect, or with anything else TEMPORARY, it is refused.
    table = 'CREATE TEMPORARY TABLE users (user_id BIGINT, note TEXT)'
    with wg.setup():
        POSTGRES.check(table)
    refused = [
        (POSTGRES, table, False),
        (POSTGRES, 'CREATE TEMPORARY TABLE public.users (i BIGINT)', True),
        (POSTGRES, 'CREATE TEMPORARY TABLE u AS SELECT 1', True),
        (POSTGRES, 'CREATE TEMPORARY VIEW v AS SELECT 1', True),
        (SNOWFLAKE, 'CREATE TEMPORARY TABLE users (i NUMBER)', True),
    ]
    for guard, sql, in_setup in refused:
        with pytest.raises(wg.WriteRefused):
            if in_setup:
                with wg.setup():
                    guard.check(sql)
            else:
                guard.check(sql)


#: The core's Postgres reads that bind a seed's time in the window column's
#: own type (sdm-data's seed CTE, T3.7): the table's row read once, the
#: seed's value put in place of the column (``jsonb_populate_record``).
_SEEDS = (
    'WITH "__sdm_seeds__" AS (\n'
    '  SELECT (r.value->>0)::bigint AS "key", (r.value->>1)::bigint AS "batch", '
    'w2."ts" AS "time"\n'
    '  FROM jsonb_array_elements(%s::jsonb) AS r(value)\n'
    '  CROSS JOIN (SELECT ROW(t.*)::"live_tests"."orders" AS base '
    'FROM "live_tests"."orders" AS t LIMIT 1) AS b\n'
    '  CROSS JOIN LATERAL jsonb_populate_record(b.base, '
    "jsonb_build_object('ts', r.value->2)) AS w2\n"
    ')\n'
    'SELECT s."batch" AS "__kumo_batch__", t."ts"\n'
    'FROM "__sdm_seeds__" s\n'
    'JOIN "live_tests"."orders" t\n'
    '  ON t."user_id" = s."key"\n'
    ' AND t."ts" <= s."time"'
)


def _seeding() -> Any:
    r"""A Postgres guard under which the loader made ``live_tests.orders``,
    the only table whose rows the core's seed-time read may convert."""
    guard = wg.WriteGuard('postgres', None, 'live_tests')
    guard.check(
        'CREATE TABLE "live_tests"."orders" (user_id BIGINT, ts TIMESTAMP)'
    )
    return guard


@pytest.mark.parametrize(
    'sql',
    [
        # Each shape inside a write, or changed past what the core writes.
        'INSERT INTO live_tests.t ' + _SEEDS,
        _SEEDS.replace('CROSS JOIN (SELECT ROW', 'CROSS JOIN (DELETE ROW'),
        _SEEDS.replace(
            '"live_tests"."orders" AS base', '"other"."orders" AS base'
        ),
        _SEEDS.replace('::"live_tests"."orders" AS base', '::orders AS base'),
        _SEEDS.replace('ROW(t.*)::', '(t.*)::'),
        _SEEDS.replace(
            '"live_tests"."orders" AS base', '"live_tests"."orders"."x" AS base'
        ),
        _SEEDS.replace(
            'LATERAL jsonb_populate_record', 'jsonb_populate_record'
        ),
        _SEEDS.replace(
            "jsonb_build_object('ts', r.value->2)",
            "jsonb_build_object('ts', nextval('s'))",
        ),
        _SEEDS.replace(
            "jsonb_build_object('ts', r.value->2)",
            "jsonb_build_object('ts', pg_sleep(1))",
        ),
        "SELECT jsonb_build_object('a', 1)",
        'SELECT x FROM t CROSS JOIN LATERAL jsonb_populate_record(NULL, '
        "jsonb_build_object('a', 1)) AS w INTO live_tests.copy",
    ],
)
def test_the_core_s_seed_time_reads_pass_on_postgres_as_reads_only(
    sql: str,
) -> None:
    guard = _seeding()
    guard.check(_SEEDS)
    with pytest.raises(wg.WriteRefused):
        guard.check(sql)
    # Only Postgres reads them so.
    with pytest.raises(wg.WriteRefused):
        DATABRICKS.check(_SEEDS)


#: A row of the loader's table, read by its alias, converted with a
#: seed's value.
_BY_ALIAS = (
    'SELECT w.ts FROM "live_tests"."orders" AS o '
    "CROSS JOIN LATERAL jsonb_populate_record(o, jsonb_build_object('ts', 1)) AS w"
)


def test_only_a_table_made_under_the_guard_is_converted_or_cast_to() -> None:
    guard = _seeding()
    guard.check(_BY_ALIAS)
    # A table of the test schema the guard did not see made, or one it saw
    # dropped, may hold a type whose checks call anything.
    fresh = wg.WriteGuard('postgres', None, 'live_tests')
    for sql in (_SEEDS, _BY_ALIAS):
        with pytest.raises(wg.WriteRefused, match='test schema table'):
            fresh.check(sql)
    guard.check('DROP TABLE live_tests.orders')
    with pytest.raises(wg.WriteRefused, match='test schema table'):
        guard.check(_SEEDS)


#: The seed read's two shapes, each a fragment the cases below change.
_LATERAL = "CROSS JOIN LATERAL jsonb_populate_record({base}, jsonb_build_object('ts', {value})) AS w"
_SEED_ROW = 'CROSS JOIN (SELECT ROW(t.*)::{type} AS base FROM {table} AS t LIMIT 1) AS b'


def _seed(
    base: str = 'b.base',
    value: str = '1',
    type_: str = '"live_tests"."orders"',
    table: str = '"live_tests"."orders"',
    head: str = 'SELECT w.ts FROM "live_tests"."orders" AS o ',
) -> str:
    return (
        head
        + _SEED_ROW.format(type=type_, table=table)
        + ' '
        + _LATERAL.format(base=base, value=value)
    )


@pytest.mark.parametrize(
    ('sql', 'reason'),
    [
        # The converted row: only one of the loader's table, by its alias or
        # the seed row of it; never another schema's, the catalog's, a
        # subquery's, a function's or a cast's (Codex r2 P1: a domain's
        # check runs in the conversion).
        (
            'SELECT p.x FROM information_schema.columns AS c '
            + _LATERAL.format(base='c', value='-1').replace(' AS w', ' AS p'),
            "a row converted that is not a test schema table's",
        ),
        (
            'SELECT 1 FROM other.t AS t '
            + _LATERAL.format(base='t', value='1'),
            "a row converted that is not a test schema table's",
        ),
        (
            'SELECT 1 FROM live_tests.other AS t '
            + _LATERAL.format(base='t', value='1'),
            "a row converted that is not a test schema table's",
        ),
        (
            'SELECT 1 FROM "live_tests"."orders" AS o '
            'CROSS JOIN (SELECT 1 AS base) AS b '
            + _LATERAL.format(base='b.base', value='1'),
            "a row converted that is not a test schema table's",
        ),
        (
            _seed(base="jsonb_build_object('ts', 1)"),
            "a row converted that is not a test schema table's",
        ),
        (
            _seed(base='NULL::"live_tests"."orders"'),
            "a row converted that is not a test schema table's",
        ),
        (
            _seed(base='pg_sleep(1)'),
            "a row converted that is not a test schema table's",
        ),
        (
            _seed(base='o.base'),
            "a row converted that is not a test schema table's",
        ),
        # The seed row read from another table than the one it is cast to.
        (
            _seed(table='other.t'),
            "a row converted that is not a test schema table's",
        ),
        # The cast: only to a table made under the guard.
        (
            'SELECT ROW(1)::live_tests.some_domain',
            'a row cast to a type not of a test schema table',
        ),
        (
            _seed(type_='other.orders'),
            'a row cast to a type not of a test schema table',
        ),
        # Calls and writes inside each shape.
        (_seed(value="nextval('s')"), 'NEXTVAL, a sequence'),
        (_seed(value='pg_sleep(1)'), 'PG_SLEEP(), a function on no list'),
        (
            _seed().replace('ROW(t.*)', "ROW(nextval('s'))"),
            'NEXTVAL, a sequence',
        ),
        (
            _seed().replace('ROW(t.*)', 'ROW(pg_sleep(1))'),
            'PG_SLEEP(), a function on no list',
        ),
        (
            _seed().replace('AS base FROM', 'AS base INTO live_tests.x FROM'),
            'INTO where no write may be',
        ),
        (
            'WITH x AS (' + _seed() + ') INSERT INTO live_tests.t SELECT 1',
            'INSERT where no write may be',
        ),
        # Each shape inside a write's SELECT, refused for itself.
        (
            'INSERT INTO live_tests.t SELECT 1 FROM "live_tests"."orders" AS o '
            + _LATERAL.format(base='o', value='1'),
            'JSONB_POPULATE_RECORD(), a function on no list',
        ),
        (
            'INSERT INTO live_tests.t SELECT ROW(o.*)::"live_tests"."orders" '
            'FROM "live_tests"."orders" AS o',
            'a cast to a type that is not built in',
        ),
        # Each outside its one place.
        (
            'SELECT 1 FROM "live_tests"."orders" AS o, '
            + _LATERAL.format(base='o', value='1').replace(
                'CROSS JOIN LATERAL ', ''
            ),
            'JSONB_POPULATE_RECORD(), a function on no list',
        ),
        (
            "SELECT jsonb_build_object('ts', 1)",
            'JSONB_BUILD_OBJECT(), a function on no list',
        ),
        # Cut short (Codex r2 P3).
        ('SELECT 1 FROM t CROSS JOIN (', 'JOIN(), a function on no list'),
    ],
)
def test_each_seed_read_shape_is_refused_where_it_is_not_the_core_s(
    sql: str, reason: str
) -> None:
    with pytest.raises(wg.WriteRefused, match=re.escape(reason)):
        _seeding().check(sql)


@pytest.mark.parametrize(
    ('sql', 'reason'),
    [
        (
            'SELECT 1 FROM t AS o ' + _LATERAL.format(base='o', value='1'),
            'JSONB_POPULATE_RECORD(), a function on no list',
        ),
        (
            'SELECT ROW(t.*)::live_tests.orders FROM t',
            'a cast to a type that is not built in',
        ),
    ],
)
def test_no_other_dialect_reads_a_seed_shape(sql: str, reason: str) -> None:
    for guard in (DATABRICKS, SNOWFLAKE):
        with pytest.raises(wg.WriteRefused, match=re.escape(reason)):
            guard.check(sql)
