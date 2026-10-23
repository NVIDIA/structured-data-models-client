# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live write guard (scripts/live/write_guard.py; KO3-51): a write
outside the one test schema never reaches the driver; reads pass."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load() -> Any:
    spec = importlib.util.spec_from_file_location(
        't33_write_guard', ROOT / 'scripts/live/write_guard.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wg = _load()

#: Made-up names; Snowflake stores unquoted names upper-cased.
SNOWFLAKE = wg.WriteGuard('snowflake', 'REVIEW_DB', 'LIVE_TESTS')
DATABRICKS = wg.WriteGuard('databricks', 'review-catalog', 'live_tests')

SNOWFLAKE_ALLOWED = [
    'SELECT * FROM OTHER_DB.PUBLIC.T',
    'select 1',
    "SELECT REPLACE(a, 'x', 'y') FROM OTHER_DB.S.T",
    'WITH x AS (SELECT 1) SELECT * FROM x',
    "SHOW PARAMETERS LIKE 'TIMEZONE' IN SESSION",
    'SHOW TABLES IN SCHEMA "REVIEW_DB"."LIVE_TESTS"',
    'DESCRIBE TABLE OTHER_DB.S.T',
    'USE SCHEMA OTHER_DB.S',
    "ALTER SESSION SET TIMEZONE = 'UTC'",
    'BEGIN',
    'COMMIT',
    'LIST @OTHER_DB.S.STAGE',
    'CREATE TABLE "REVIEW_DB"."LIVE_TESTS"."users" ("ID" NUMBER(38,0))',
    'create or replace table review_db.live_tests.t (i int)',
    'INSERT INTO "REVIEW_DB"."LIVE_TESTS"."users" ("ID") VALUES (1), (2)',
    'INSERT INTO "REVIEW_DB"."LIVE_TESTS"."users" ("ID") SELECT PARSE_JSON(column1) FROM VALUES (1)',
    'INSERT ALL INTO REVIEW_DB.LIVE_TESTS.A INTO REVIEW_DB.LIVE_TESTS.B SELECT 1',
    'DROP TABLE IF EXISTS "REVIEW_DB"."LIVE_TESTS"."users"',
    'DROP SEMANTIC VIEW IF EXISTS "REVIEW_DB"."LIVE_TESTS".shop',
    'ALTER TABLE REVIEW_DB.LIVE_TESTS.A RENAME TO REVIEW_DB.LIVE_TESTS.B',
    'ALTER TABLE REVIEW_DB.LIVE_TESTS.A SWAP WITH REVIEW_DB.LIVE_TESTS.B',
    'DELETE FROM REVIEW_DB.LIVE_TESTS.A WHERE 1 = 1',
    'UPDATE REVIEW_DB.LIVE_TESTS.A SET x = 1',
    'MERGE INTO REVIEW_DB.LIVE_TESTS.A USING OTHER_DB.S.B ON a.i = b.i '
    'WHEN MATCHED THEN UPDATE SET x = 1',
    'TRUNCATE TABLE IF EXISTS REVIEW_DB.LIVE_TESTS.A',
    'COPY INTO REVIEW_DB.LIVE_TESTS.A FROM @REVIEW_DB.LIVE_TESTS.UPLOADS',
    'PUT file:///tmp/a.csv @REVIEW_DB.LIVE_TESTS.UPLOADS',
    'REMOVE @REVIEW_DB.LIVE_TESTS.UPLOADS/a.csv',
    'GRANT SELECT ON TABLE REVIEW_DB.LIVE_TESTS.A TO ROLE R',
    "COMMENT ON TABLE REVIEW_DB.LIVE_TESTS.A IS 'DROP TABLE OTHER_DB.S.T'",
    'CREATE SCHEMA IF NOT EXISTS REVIEW_DB.LIVE_TESTS',
    'CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML(\'"REVIEW_DB"."LIVE_TESTS"\', '
    '$$\nname: shop\ntables: [{base_table: {database: OTHER_DB}}]\n$$)',
    '-- DROP TABLE OTHER_DB.S.T\nSELECT 1',
    '/* DROP TABLE OTHER_DB.S.T; */ SELECT 1;',
    "SELECT 'x; DROP TABLE OTHER_DB.S.T'",
    '// DROP TABLE OTHER_DB.S.T\nSELECT 1',
    'SELECT 1; SELECT 2;',
]

SNOWFLAKE_REFUSED = [
    # Another schema or database.
    'CREATE TABLE OTHER_DB.LIVE_TESTS.T (i INT)',
    'CREATE TABLE REVIEW_DB.OTHER.T (i INT)',
    'DROP TABLE OTHER_DB.S.T',
    'INSERT INTO OTHER_DB.S.T VALUES (1)',
    'INSERT ALL INTO REVIEW_DB.LIVE_TESTS.A INTO OTHER_DB.S.B SELECT 1',
    'ALTER TABLE REVIEW_DB.LIVE_TESTS.A RENAME TO OTHER_DB.S.B',
    'ALTER TABLE REVIEW_DB.LIVE_TESTS.A SWAP WITH OTHER_DB.S.B',
    'DELETE FROM OTHER_DB.S.T',
    'UPDATE OTHER_DB.S.T SET x = 1',
    'MERGE INTO OTHER_DB.S.T USING REVIEW_DB.LIVE_TESTS.B ON 1 = 1 '
    'WHEN MATCHED THEN DELETE',
    'COPY INTO @OTHER_DB.S.STAGE FROM REVIEW_DB.LIVE_TESTS.A',
    "COPY INTO 's3://bucket/x' FROM REVIEW_DB.LIVE_TESTS.A",
    'PUT file:///tmp/a.csv @~',
    "PUT file:///tmp/a.csv '@REVIEW_DB.LIVE_TESTS.UPLOADS'",
    'GRANT OWNERSHIP ON TABLE OTHER_DB.S.T TO ROLE R',
    'GRANT ROLE R TO USER U',
    'GRANT SELECT ON ALL TABLES IN SCHEMA REVIEW_DB.LIVE_TESTS TO ROLE R',
    'UNDROP TABLE OTHER_DB.S.T',
    "CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML('OTHER_DB.S', $$name: v$$)",
    # Case: Snowflake's quoted names are exact.
    'CREATE TABLE "review_db"."live_tests".t (i INT)',
    # Unqualified or partly qualified.
    'CREATE TABLE T (i INT)',
    'CREATE TABLE LIVE_TESTS.T (i INT)',
    'DROP TABLE IF EXISTS "users"',
    'INSERT INTO T VALUES (1)',
    'TRUNCATE TABLE T',
    # Schemas, databases and the account as a whole.
    'DROP SCHEMA REVIEW_DB.LIVE_TESTS',
    'DROP SCHEMA IF EXISTS REVIEW_DB.LIVE_TESTS',
    'CREATE SCHEMA REVIEW_DB.LIVE_TESTS',
    'CREATE SCHEMA IF NOT EXISTS REVIEW_DB.OTHER',
    'CREATE OR REPLACE SCHEMA REVIEW_DB.LIVE_TESTS',
    'ALTER SCHEMA REVIEW_DB.LIVE_TESTS RENAME TO REVIEW_DB.X',
    'CREATE DATABASE X',
    'DROP DATABASE REVIEW_DB',
    'ALTER WAREHOUSE W SUSPEND',
    "ALTER USER U SET PASSWORD = 'x'",
    # Not recognised, so refused.
    'CALL OTHER_DB.S.PROC()',
    "EXECUTE IMMEDIATE 'DROP TABLE OTHER_DB.S.T'",
    'BEGIN DROP TABLE OTHER_DB.S.T; END',
    "CREATE TABLE IDENTIFIER('REVIEW_DB.LIVE_TESTS.T') (i INT)",
    'WITH x AS (SELECT 1) INSERT INTO OTHER_DB.S.T SELECT * FROM x',
    'DECLARE x INT',
    '(SELECT 1)',
    # Multi-statement text: each statement checked.
    'SELECT 1; DROP TABLE OTHER_DB.S.T',
    'CREATE TABLE REVIEW_DB.LIVE_TESTS.T (i INT); DROP TABLE OTHER_DB.S.T;',
    "SELECT 'a'; DROP TABLE OTHER_DB.S.T -- ;",
    # Unterminated text, so the rest cannot be read.
    "SELECT 'x; DROP TABLE OTHER_DB.S.T",
    'SELECT 1 /* DROP TABLE OTHER_DB.S.T',
    'SELECT $$ x',
]

DATABRICKS_ALLOWED = [
    'SELECT * FROM `other`.`s`.`t`',
    'SELECT current_timezone() AS zone',
    'SELECT "DROP TABLE x" AS s',
    'SHOW TABLES IN `review-catalog`.`live_tests`',
    'USE CATALOG `other`',
    "SET TIME ZONE 'UTC'",
    'CREATE TABLE `review-catalog`.`live_tests`.`users` (`id` BIGINT)',
    'CREATE TABLE IF NOT EXISTS `Review-Catalog`.LIVE_TESTS.t (i INT)',
    'INSERT INTO `review-catalog`.`live_tests`.`users` (`id`) VALUES (1)',
    'INSERT OVERWRITE TABLE `review-catalog`.`live_tests`.t SELECT 1',
    'DROP TABLE IF EXISTS `review-catalog`.`live_tests`.`users`',
    'DROP VIEW IF EXISTS `review-catalog`.`live_tests`.`m`',
    'CREATE OR REPLACE VIEW `review-catalog`.`live_tests`.`m` WITH METRICS '
    'LANGUAGE YAML AS $$\nversion: 0.1\nsource: other.s.t\n$$',
    'CREATE VOLUME IF NOT EXISTS `review-catalog`.live_tests.uploads',
    'OPTIMIZE `review-catalog`.live_tests.t',
    'MSCK REPAIR TABLE `review-catalog`.live_tests.t',
    'CREATE SCHEMA IF NOT EXISTS `review-catalog`.live_tests',
]

DATABRICKS_REFUSED = [
    'CREATE TABLE `other`.`live_tests`.t (i INT)',
    'CREATE TABLE `review-catalog`.`other`.t (i INT)',
    'DROP TABLE `other`.`s`.`t`',
    'INSERT INTO `other`.s.t VALUES (1)',
    "INSERT OVERWRITE DIRECTORY '/tmp/x' SELECT 1",
    # A double-quoted text is a string on Databricks, never a name.
    'CREATE TABLE "review-catalog"."live_tests".t (i INT)',
    # Unquoted, the hyphen splits the catalog's name.
    'CREATE TABLE review-catalog.live_tests.t (i INT)',
    'CREATE TABLE t (i INT)',
    'DROP TABLE live_tests.t',
    "CREATE TABLE `review-catalog`.live_tests.t (i INT) LOCATION 's3://x'",
    'DROP SCHEMA `review-catalog`.live_tests CASCADE',
    'DROP CATALOG `review-catalog`',
    'CALL `review-catalog`.live_tests.p()',
    "CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML('`review-catalog`.live_tests', $$x$$)",
    'SELECT 1; DROP TABLE `other`.s.t',
]


@pytest.mark.parametrize('sql', SNOWFLAKE_ALLOWED)
def test_snowflake_reads_and_test_schema_writes_pass(sql: str) -> None:
    SNOWFLAKE.check(sql)


@pytest.mark.parametrize('sql', SNOWFLAKE_REFUSED)
def test_snowflake_writes_elsewhere_are_refused(sql: str) -> None:
    with pytest.raises(wg.WriteRefused):
        SNOWFLAKE.check(sql)


@pytest.mark.parametrize('sql', DATABRICKS_ALLOWED)
def test_databricks_reads_and_test_schema_writes_pass(sql: str) -> None:
    DATABRICKS.check(sql)


@pytest.mark.parametrize('sql', DATABRICKS_REFUSED)
def test_databricks_writes_elsewhere_are_refused(sql: str) -> None:
    with pytest.raises(wg.WriteRefused):
        DATABRICKS.check(sql)


def test_the_test_schema_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        'SDM_LIVE_TEST_DATABASE',
        'SDM_LIVE_TEST_SCHEMA',
        'SDM_LIVE_SNOWFLAKE_TEST_DATABASE',
        'SDM_LIVE_SNOWFLAKE_TEST_SCHEMA',
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(wg.WriteRefused, match='no test schema'):
        wg.WriteGuard('snowflake')
    with pytest.raises(ValueError):
        wg.WriteGuard('postgres', 'A', 'B')
    monkeypatch.setenv('SDM_LIVE_TEST_DATABASE', 'REVIEW_DB')
    monkeypatch.setenv('SDM_LIVE_TEST_SCHEMA', 'LIVE_TESTS')
    guard = wg.WriteGuard('snowflake')
    guard.check('DROP TABLE REVIEW_DB.LIVE_TESTS.T')
    # A warehouse's own variables come first.
    monkeypatch.setenv('SDM_LIVE_SNOWFLAKE_TEST_SCHEMA', 'OTHER')
    guard = wg.WriteGuard('snowflake')
    guard.check('DROP TABLE REVIEW_DB.OTHER.T')
    with pytest.raises(wg.WriteRefused):
        guard.check('DROP TABLE REVIEW_DB.LIVE_TESTS.T')


class _Cursor:
    def __init__(self, seen: list[str]) -> None:
        self.seen = seen

    def execute(self, sql: str, *arguments: Any) -> _Cursor:
        self.seen.append(sql)
        return self

    def executemany(self, sql: str, rows: Any) -> None:
        self.seen.append(sql)

    def execute_async(self, sql: str) -> dict[str, str]:
        self.seen.append(sql)
        return {'queryId': 'q'}

    def fetchall(self) -> list[tuple[int]]:
        return [(1,)]

    def close(self) -> None:
        pass


class _Driver:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def cursor(self) -> _Cursor:
        return _Cursor(self.seen)

    def execute_string(self, sql: str) -> list[Any]:
        self.seen.append(sql)
        return []

    def execute_stream(self, stream: Any) -> list[Any]:
        self.seen.append('stream')
        return []

    def close(self) -> None:
        pass


def test_a_guarded_connection_refuses_before_the_driver_sees_the_text() -> None:
    driver = _Driver()
    connection = wg.guard_connection(driver, SNOWFLAKE)
    write = 'DROP TABLE OTHER_DB.S.T'
    with connection.cursor() as cursor:
        for call in (
            lambda: cursor.execute(write),
            lambda: cursor.executemany(write, [()]),
            lambda: cursor.execute_async(write),
            lambda: connection.execute_string('SELECT 1; ' + write),
            lambda: connection.execute_stream(None),
        ):
            with pytest.raises(wg.WriteRefused):
                call()
        assert driver.seen == []
        # Reads and test-schema writes go through, the cursor still ours.
        assert cursor.execute('SELECT 1').fetchall() == [(1,)]
        cursor.executemany(
            'INSERT INTO REVIEW_DB.LIVE_TESTS.T VALUES (%s)', [(1,)]
        )
        assert cursor.execute_async('SELECT 2') == {'queryId': 'q'}
    connection.execute_string('SELECT 1; DROP TABLE REVIEW_DB.LIVE_TESTS.T')
    assert driver.seen == [
        'SELECT 1',
        'INSERT INTO REVIEW_DB.LIVE_TESTS.T VALUES (%s)',
        'SELECT 2',
        'SELECT 1; DROP TABLE REVIEW_DB.LIVE_TESTS.T',
    ]


def test_the_loaders_statements_pass_in_the_test_schema_only() -> None:
    # The contract loader's warehouse statements for a small case, in the
    # test schema and in another: every one passes in the first, every
    # write is refused in the second.
    import duckdb

    loader_spec = importlib.util.spec_from_file_location(
        't33_guard_databases', ROOT / 'contract/runners/databases.py'
    )
    assert loader_spec is not None and loader_spec.loader is not None
    loader = importlib.util.module_from_spec(loader_spec)
    sys.modules[loader_spec.name] = loader
    loader_spec.loader.exec_module(loader)
    duck = duckdb.connect()
    duck.execute(
        'CREATE TABLE users (id BIGINT PRIMARY KEY, name VARCHAR, '
        'tags VARCHAR[], seen_at TIMESTAMP)'
    )
    duck.execute("INSERT INTO users VALUES (1, 'a', ['x'], '2024-01-01')")
    for engine, guard, namespace in (
        ('snowflake', SNOWFLAKE, ('REVIEW_DB', 'LIVE_TESTS')),
        ('databricks', DATABRICKS, ('review-catalog', 'live_tests')),
    ):
        ours, _ = loader.warehouse_statements(duck, engine, namespace)
        assert ours
        for statement in ours:
            guard.check(statement)
        elsewhere, _ = loader.warehouse_statements(duck, engine, ('OTHER', 'S'))
        for statement in elsewhere:
            with pytest.raises(wg.WriteRefused):
                guard.check(statement)
