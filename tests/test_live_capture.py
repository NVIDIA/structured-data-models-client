# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live fixture loading stays within the configured warehouse test schema."""

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
LIVE = ROOT / 'scripts/live'


def _module(name: str):
    spec = importlib.util.spec_from_file_location(
        f'live_capture_{name}', LIVE / f'{name}.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


live_guard = _module('runner_guard')


def _runner_module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / 'contract/runners' / f'{name}.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_a_write_outside_the_test_schema_never_reaches_the_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review r3, r4: the guard is
    # installed on the driver's own cursor class; the case's placeholder
    # resolves to WRONG_SCHEMA, and neither the loader's writes nor a
    # statement on any cursor reach the driver outside ALLOWED_SCHEMA.
    import types

    import duckdb

    # The repository's guard, as the run loads it (scripts/live).
    guard = live_guard.load_guard()
    for dialect in ('SNOWFLAKE_', 'DATABRICKS_', 'POSTGRES_'):
        for name in ('DATABASE', 'SCHEMA'):
            monkeypatch.delenv(f'SDM_LIVE_{dialect}TEST_{name}', raising=False)
    monkeypatch.setenv('SDM_LIVE_TEST_DATABASE', 'REVIEW_DB')
    monkeypatch.setenv('SDM_LIVE_TEST_SCHEMA', 'ALLOWED_SCHEMA')
    monkeypatch.delenv('SDM_CAPTURE_CATALOG', raising=False)
    monkeypatch.delenv('SDM_CAPTURE_SCHEMA', raising=False)
    loader = _runner_module('databases')
    monkeypatch.setattr(loader, 'WRITE_GUARD', None)
    seen: list[str] = []

    class SnowflakeCursor:
        def execute(self, sql, *a, **k):
            seen.append(sql)

        def fetchall(self):
            return []

        def close(self):
            pass

    class Driver:
        def cursor(self):
            return SnowflakeCursor()

    fake = types.ModuleType('snowflake.connector.cursor')
    fake.SnowflakeCursor = SnowflakeCursor
    monkeypatch.setitem(sys.modules, 'snowflake.connector.cursor', fake)
    uninstall = live_guard.install_guard('snowflake', guard, loader)
    try:

        class Target:
            names = {'catalog': 'REVIEW_DB', 'schema': 'WRONG_SCHEMA'}
            view = None
            connection = Driver()

        duck = duckdb.connect()
        duck.execute('CREATE TABLE users (i INTEGER)')
        with pytest.raises(PermissionError, match='outside the test schema'):
            loader.load(duck, 'snowflake', Target())
        # A statement on the driver's cursor, not the loader's, is refused too.
        with pytest.raises(PermissionError):
            Driver().cursor().execute('DROP TABLE OTHER_DB.PUBLIC.T')
        assert seen == []
        # The test schema itself loads.
        Target.names = {'catalog': 'REVIEW_DB', 'schema': 'ALLOWED_SCHEMA'}
        loader.load(duck, 'snowflake', Target())
        assert any(
            s.startswith('CREATE TABLE "REVIEW_DB"."ALLOWED_SCHEMA"')
            for s in seen
        )
    finally:
        uninstall()
    assert loader.WRITE_GUARD is None
    Driver().cursor().execute('DROP TABLE OTHER_DB.PUBLIC.T')
    # No guard, no run; no test schema, or another than the cases', no run.
    with pytest.raises(SystemExit, match='no live write guard'):
        live_guard.load_guard(tmp_path / 'write_guard.py')
    monkeypatch.setenv('SDM_CAPTURE_SCHEMA', 'WRONG_SCHEMA')
    with pytest.raises(SystemExit, match='SDM_CAPTURE_SCHEMA is not'):
        live_guard.install_guard('snowflake', guard, loader)
    monkeypatch.delenv('SDM_LIVE_TEST_SCHEMA')
    with pytest.raises(SystemExit, match='no test schema'):
        live_guard.install_guard('snowflake', guard, loader)


def test_lakebase_runs_through_the_same_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Review r4: Lakebase has no check of its
    # own; its psycopg classes are guarded, and the loader's drops carry no
    # CASCADE.
    import psycopg

    guard = live_guard.load_guard()
    for name in ('DATABASE', 'SCHEMA'):
        monkeypatch.delenv(f'SDM_LIVE_POSTGRES_TEST_{name}', raising=False)
    monkeypatch.setenv('SDM_LIVE_TEST_SCHEMA', 'live_tests')
    monkeypatch.delenv('SDM_CAPTURE_SCHEMA', raising=False)
    loader = _runner_module('databases')
    monkeypatch.setattr(loader, 'WRITE_GUARD', None)
    uninstall = live_guard.install_guard('lakebase', guard, loader)
    try:
        for statement in (
            'DROP TABLE IF EXISTS "live_tests"."a" CASCADE',
            'DROP TABLE IF EXISTS "live_tests"."a", "other"."b"',
            'DROP TABLE IF EXISTS "live_tests"."a"; DROP TABLE "other"."b"',
        ):
            with pytest.raises(PermissionError):
                loader.WRITE_GUARD(statement)
            with pytest.raises(PermissionError):
                psycopg.Connection.execute(
                    object.__new__(psycopg.Connection), statement
                )
        loader.WRITE_GUARD('DROP TABLE IF EXISTS "live_tests"."a"')
    finally:
        uninstall()


class _CaseSensitiveSnowflake:
    r"""A Snowflake connection that keeps names as Snowflake does: a quoted
    name exactly, an unquoted one upper-cased; a table read by a name it
    does not hold, and a semantic view naming a table or column it does
    not hold, are errors, as live."""

    def __init__(self) -> None:
        self.tables: dict[str, list[str]] = {}
        self.ran: list[str] = []

    @staticmethod
    def _name(text: str) -> str:
        text = text.strip()
        return text[1:-1] if text.startswith('"') else text.upper()

    def cursor(self) -> Any:
        return _Cursor(self)


class _Cursor:
    def __init__(self, warehouse: _CaseSensitiveSnowflake) -> None:
        self.warehouse = warehouse
        self.result: Any = None
        self.description: Any = None

    def execute(self, sql: str, *arguments: Any, **keywords: Any) -> None:
        import pyarrow as pa
        import yaml

        held = self.warehouse
        held.ran.append(sql)
        last = r'\.("(?:[^"]|"")*"|\w+)'
        if sql.startswith('CREATE TABLE'):
            qualified = re.match(r'CREATE TABLE (\S+) \(', sql).group(1)
            table = held._name(re.findall(r'"(?:[^"]|"")*"|\w+', qualified)[-1])
            body = sql[sql.index('(') + 1 :]
            held.tables[table] = [
                held._name(c) for c in re.findall(r'("[^"]+"|\w+) \w', body)
            ]
        elif sql.startswith('DROP TABLE IF EXISTS'):
            held.tables.pop(
                held._name(re.search(last + r'$', sql).group(1)), None
            )
        elif sql.startswith('SHOW TABLES'):
            self.result = [('t', name) for name in held.tables]
        elif sql.startswith('CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML'):
            document = yaml.safe_load(sql.split('$$')[1])
            for entry in document['tables']:
                table = held._name(entry['base_table']['table'])
                if table not in held.tables:
                    raise RuntimeError(f'the view names no table {table}')
                for kind in ('dimensions', 'facts', 'time_dimensions'):
                    for item in entry.get(kind) or []:
                        if re.fullmatch(r'\w+', item['expr']) and (
                            held._name(item['expr']) not in held.tables[table]
                        ):
                            raise RuntimeError(f'no column {item["expr"]}')
        elif sql.startswith('SELECT * FROM'):
            table = held._name(re.search(last + r'$', sql).group(1))
            if table not in held.tables:
                raise RuntimeError(f'no table {table}')
            self.result = pa.table(
                {c: pa.array([], pa.int64()) for c in held.tables[table]}
            )
        elif sql.startswith('SHOW PARAMETERS'):
            self.description = [('key',), ('value',)]
            self.result = [('TIMEZONE', 'America/Los_Angeles')]

    def fetchall(self) -> Any:
        return self.result

    def fetch_arrow_all(self, force_return_table: bool = False) -> Any:
        return self.result

    def close(self) -> None:
        pass


def test_a_semantic_view_case_loads_creates_and_reads_back_through_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review r3 (P2): the recording mode's Live loads the case, creates its
    # semantic view and reads every table back on a case-sensitive
    # Snowflake: the YAML's unquoted base tables and columns resolve.
    config = tmp_path / 'live.json'
    config.write_text(
        json.dumps(
            {'format': 'sdm-live/1', 'legs': {'snowflake': {'driver': {}}}}
        )
    )
    monkeypatch.setenv('SDM_LIVE_CONFIG', str(config))
    monkeypatch.setenv('SDM_CAPTURE_CATALOG', 'REVIEW_DB')
    monkeypatch.setenv('SDM_CAPTURE_SCHEMA', 'ALLOWED_SCHEMA')
    runner = _runner_module('run_python')
    loader = _runner_module('databases')
    monkeypatch.setattr(loader, '_SCHEMAS', {})
    warehouse = _CaseSensitiveSnowflake()
    monkeypatch.setattr(
        runner, 'connect_leg', lambda *a, **k: (warehouse, lambda: None)
    )
    case = json.loads((ROOT / 'tests/semantic-view-case.json').read_text())
    live = runner.Live(case, runner.load_bases(), 'snowflake')
    try:
        assert sorted(name for name, _ in live.tables) == ['orders', 'users']
        assert any(
            sql.startswith('CALL SYSTEM$CREATE_SEMANTIC_VIEW')
            for sql in warehouse.ran
        )
    finally:
        live.close()


# Round 9: the run records what it can.

_GUARD_NAMES = {
    'databricks-sql': ('review-catalog', 'live_tests'),
    'databricks-spark': ('review-catalog', 'live_tests'),
    'snowflake': ('REVIEW_DB', 'LIVE_TESTS'),
    'lakebase': (None, 'live_tests'),
}


class _Recording:
    r"""A connection of any leg's driver recording each statement sent."""

    def __init__(self) -> None:
        self.ran: list[str] = []

    def cursor(self) -> _Recording:
        return self

    def execute(self, sql: str, *arguments: Any) -> _Recording:
        self.ran.append(sql)
        return self

    def sql(self, sql: str) -> _Recording:
        self.ran.append(sql)
        return self

    def collect(self) -> list[Any]:
        return []

    def fetchall(self) -> list[Any]:
        return []

    def close(self) -> None:
        pass


@pytest.mark.parametrize(
    'leg', ('databricks-sql', 'databricks-spark', 'snowflake', 'lakebase')
)
def test_the_recorders_own_statements_pass_each_legs_guard(leg: str) -> None:
    # the statements the recorder sends outside the
    # capture (the session's zone, the schema's tables, the read-back, the
    # Spark path's USE) pass the leg's guard.
    runner = _runner_module('run_python')
    loader = _runner_module('databases')
    guard_module = live_guard.load_guard()
    database, schema = _GUARD_NAMES[leg]
    guard = guard_module.WriteGuard(live_guard.DIALECT[leg], database, schema)
    names = {'catalog': database or '', 'schema': schema}
    namespace = (schema,) if leg == 'lakebase' else (database, schema)
    connection = _Recording()
    loader.held(leg, connection, namespace)
    statements = [
        runner._ZONE[leg],
        f'SELECT * FROM {runner._quoted(leg, names, "orders")}',
        *connection.ran,
    ]
    if leg == 'databricks-spark':
        statements += [f'USE CATALOG `{database}`', f'USE SCHEMA `{schema}`']
    for statement in statements:
        guard.check(statement)


@pytest.mark.parametrize('oid', [54844, 59540])
def test_lakebase_live_capture_names_composite_types_in_its_own_schema(
    oid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner_module('run_python')
    calls: list[tuple[str, Any]] = []
    closed: list[str] = []
    type_name = 'sdm_composite_users_f'

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc: Any) -> None:
            closed.append('cursor')

        def execute(self, sql: str, parameters: Any = None):
            calls.append((sql, parameters))
            return self

        def fetchall(self):
            return [(oid, type_name), (oid + 1, '_' + type_name)]

    connection = types.SimpleNamespace(cursor=Cursor)
    monkeypatch.setattr(
        runner,
        'live_configuration',
        lambda leg: ({}, {'schema': 'only_this_schema'}, []),
    )
    monkeypatch.setattr(runner, 'resolve_placeholders', lambda case: case)
    monkeypatch.setattr(
        runner,
        '_connection',
        lambda *args: types.SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(runner, '_duck_tables', lambda duck: [])
    monkeypatch.setattr(
        runner,
        'connect_leg',
        lambda *args, **kwargs: (connection, lambda: None),
    )
    monkeypatch.setattr(
        runner,
        '_sibling',
        lambda name: types.SimpleNamespace(load=lambda *args: None),
    )
    monkeypatch.setattr(runner, 'session_zone', lambda *args: 'UTC')
    live = runner.Live({'id': 'nested.struct', 'input': {}}, {}, 'lakebase')
    log = runner.DriverLog()
    live.captured(log)
    assert len(calls) == 1
    sql, parameters = calls[0]
    assert 'pg_type' in sql and 'pg_namespace' in sql
    assert parameters == ('only_this_schema',)
    assert closed == ['cursor']
    assert log.type_names == {oid: type_name, oid + 1: '_' + type_name}
    driver = types.SimpleNamespace(description=[('f', oid), ('unknown', 999)])
    assert runner._description(driver, log.type_names) == (
        ['f', 'unknown'],
        [type_name, '999'],
    )
    assert log.statements == []
    live.close()
