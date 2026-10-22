# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T3.2's recording mode (design choice 2, "What C-1 records") and the
capture checks it is judged by, one test per change (the package's test
author's). The recording mode is ``run_python.py --mode python-ref`` on a
warehouse leg: the driver capture (``capture``, ``DriverLog``) records
what the old client sent and got back, and the statement log
(``StatementLog``) is its query log, the twin ``capture_check.py``'s
``complete`` holds the capture's counts to (T2.1a fix 4).

- The capture on each warehouse leg (the brief's "Reference code to
  port"): ``capture(connection, DriverLog(), leg)`` records each statement
  with its parameters and each Arrow result read, through the Databricks
  SQL connector (``cursor.execute``, ``fetchall_arrow``), Snowflake's
  connector (``execute`` with ``qmark`` parameters, ``fetch_arrow_all``)
  and a Spark session (``spark.sql``, ``DataFrame.toArrow``, classic and
  Spark Connect, which serverless runs), and hands the source a connection
  the reference's dispatch reads as that warehouse
  (``sdm_connectors/source.py:51-85``).
- The statement log on each warehouse leg logs those drivers' statements,
  whatever object calls them; on Postgres and Lakebase it also logs
  psycopg's ``copy`` and ``stream`` (P2-T2.1a-2), and the runner enters it
  only as a context, so it is undone when a case fails (P2-T2.1a-2's
  "entered outside a ``try``/``finally``").
- The runner runs an S2 case on its ``warehouses`` only, keyed
  ``<case>@<leg>``.
- The placeholders a case writes for private names (``{catalog}``,
  ``{schema}`` and the three existing views' names, T3.2's brief, the S2
  test authors' list) are resolved from the environment for a live run,
  and a placeholder with nothing set refuses by its name.
- Each case's input hash covers what the old client's answer depends on
  (``input``, ``until`` and the base's statements) and nothing else (T3.4's
  brief: adding ``core`` must not make a case pending live).
- ``capture_check.py captures``: a capture of an unknown format version is
  refused; a case whose input hash differs from the one recorded, or with
  no capture on a leg it runs on, is listed pending live; ``--locked``
  fails on any pending one (locked check 1).
- ``capture_check.py commit``: a captures-only commit after the final run
  passes only when every capture it changes names the final candidate as
  its recording commit (the live-test plan; T3.10's landing check).
- ``contract/captures/**`` is a protected path.
- P2-MID2-7: the runner refuses an installed client whose contract version
  is not the checkout's.

Interfaces assumed (T3.2 defines them; this file is their first reader):

- ``run_python.WAREHOUSES == ('databricks-sql', 'databricks-spark',
  'snowflake', 'lakebase')``; ``run_python.capture`` and ``StatementLog``
  take each; ``run_python.case_databases(case)`` gives an S2 case's
  ``warehouses``;
- ``run_python.resolve_placeholders(case, environ)``: the case with each
  ``{name}`` in its ``input`` replaced by ``environ['SDM_CAPTURE_' +
  name.upper()]``, a ``RunnerFailure`` naming ``{name}`` (never a value)
  when one is unset;
- ``run_python.input_hash(case, bases=None)``: a hex SHA-256; ``bases`` as
  ``load_bases`` gives them, this checkout's by default;
- ``run_python.contract_refusal(installed, root=REPO)``: None when the
  installed contract version is the one ``rust/crates/sdm-contract``
  declares at ``root``, else why, naming both;
- a capture's manifest, a JSON file the capture writer (T3.3) writes beside
  the recorded tables: ``{"format": "sdm-capture/1", "recording_commit":
  SHA, "reference_commit": SHA, "legs": {LEG: {...versions}}, "cases":
  {"<case>@<leg>": {"input_hash": HEX, ...}}}``;
- ``capture_check.py captures --cases DIR [--locked] MANIFEST...``: exit 1
  naming a refused manifest; each pending item as a line ``pending live:
  <case>@<leg>``; exit 0 unless refused, or ``--locked`` with one pending;
- ``capture_check.py commit --candidate SHA REV``, run in the repository:
  exit 0 when REV changes only ``contract/captures/**`` and every manifest
  it adds or changes names SHA as its ``recording_commit``, else 1.

Each test listed in ``contract/tests/t32-pending.json`` failed when written,
before the implementation, and is held to failing (``xfail``, strict) until
the implementer takes it out of ``pending``; ``pending`` stays within
``authored``, the test author's list.
"""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
CAPTURE_CHECK = ROOT / 'contract/runners/capture_check.py'
RATCHET = ROOT / 'scripts/rust_core/ratchet.py'
VECTORS = ROOT / 'contract/tests/t32-case-format-vectors.json'
PENDING_FILE = ROOT / 'contract/tests/t32-pending.json'
WAREHOUSES = ('databricks-sql', 'databricks-spark', 'snowflake', 'lakebase')
QUIET = {'GH_PROMPT_DISABLED': '1', 'GIT_TERMINAL_PROMPT': '0'}


def _pending_names(key: str) -> set[str]:
    return set(json.loads(PENDING_FILE.read_text(encoding='utf-8'))[key])


def pending(test: Any) -> Any:
    """Holds ``test`` to failing while ``contract/tests/t32-pending.json``
    lists it."""
    assert test.__name__ in _pending_names('authored'), test.__name__
    if test.__name__ in _pending_names('pending'):
        return pytest.mark.xfail(
            strict=True,
            reason='T3.2: failing until implemented; once it passes, take '
            'it out of contract/tests/t32-pending.json pending',
        )(test)
    return test


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('t32_recording_run_python', RUNNER)


def _s2_case(leg: str = 'databricks-sql') -> dict[str, Any]:
    r"""The case-format vectors' S2 case, on ``leg``."""
    case = copy.deepcopy(
        json.loads(VECTORS.read_text(encoding='utf-8'))['case']
    )
    case['warehouses'] = [leg]
    return case


# --- fake drivers --------------------------------------------------------------
#
# The warehouse drivers are not in the test environment (the recording runs
# in the reference's environment, and Spark's in a serverless job). Each
# fake stands where the real one is imported from, with the real class
# names and call signatures, and answers a fixed table; the capture and the
# statement log find them by import as they find the real ones.


def _table() -> Any:
    import pyarrow as pa

    return pa.table({'user_id': pa.array([7, 8], pa.int64())})


def _install(
    monkeypatch: pytest.MonkeyPatch, dotted: str, **attributes: Any
) -> ModuleType:
    r"""A fake module at ``dotted`` (its parents made as packages), holding
    ``attributes``, in place of any real one for this test."""
    parts = dotted.split('.')
    parent: ModuleType | None = None
    for at in range(1, len(parts) + 1):
        name = '.'.join(parts[:at])
        module = sys.modules.get(name)
        if module is None or not getattr(module, '__t32_fake__', False):
            module = types.ModuleType(name)
            module.__path__ = []  # type: ignore[attr-defined]
            module.__t32_fake__ = True  # type: ignore[attr-defined]
            monkeypatch.setitem(sys.modules, name, module)
        if parent is not None:
            monkeypatch.setattr(parent, parts[at - 1], module, raising=False)
        parent = module
    assert parent is not None
    for key, value in attributes.items():
        # A class keeps the first fake module it is placed in, as a real
        # class keeps its defining module when a package re-exports it.
        if isinstance(value, type) and value.__module__ == __name__:
            value.__module__ = dotted
        setattr(parent, key, value)
    return parent


def _databricks_sql(monkeypatch: pytest.MonkeyPatch) -> Any:
    r"""A ``databricks.sql`` connection (``databricks/sql/client.py``)."""
    table = _table()

    class Cursor:
        description = [('user_id', 'bigint', None, None, None, None, None)]

        def execute(
            self, operation: str, parameters: Any = None, **kw: Any
        ) -> Any:
            return self

        def fetchall_arrow(self) -> Any:
            return table

        def close(self) -> None:
            pass

    class Connection:
        def cursor(self, **kw: Any) -> Any:
            return Cursor()

        def close(self) -> None:
            pass

    _install(
        monkeypatch,
        'databricks.sql.client',
        Cursor=Cursor,
        Connection=Connection,
    )
    return Connection()


def _snowflake(monkeypatch: pytest.MonkeyPatch) -> Any:
    r"""A ``snowflake.connector`` connection, ``qmark`` parameters."""
    table = _table()

    class SnowflakeCursor:
        description = [('USER_ID', 0, None, None, 38, 0, True)]

        def execute(self, command: str, params: Any = None, **kw: Any) -> Any:
            return self

        def fetch_arrow_all(self, *args: Any, **kw: Any) -> Any:
            return table

        def close(self) -> None:
            pass

    class SnowflakeConnection:
        def cursor(self, *args: Any, **kw: Any) -> Any:
            return SnowflakeCursor()

        def close(self) -> None:
            pass

    _install(
        monkeypatch,
        'snowflake.connector.cursor',
        SnowflakeCursor=SnowflakeCursor,
    )
    _install(
        monkeypatch,
        'snowflake.connector.connection',
        SnowflakeConnection=SnowflakeConnection,
    )
    _install(monkeypatch, 'snowflake.connector', paramstyle='qmark')
    return SnowflakeConnection()


def _spark(monkeypatch: pytest.MonkeyPatch, connect: bool) -> Any:
    r"""A Spark session, classic (``pyspark.sql.session``) or Spark Connect
    (``pyspark.sql.connect.session``), whose ``sql`` gives a DataFrame and
    whose ``DataFrame.toArrow`` runs it."""
    table = _table()
    where = 'pyspark.sql.connect' if connect else 'pyspark.sql'

    class DataFrame:
        def toArrow(self) -> Any:  # noqa: N802 - Spark's name
            return table

        def collect(self) -> list[Any]:
            return [(7,), (8,)]

    class SparkSession:
        def sql(self, sqlQuery: str, args: Any = None, **kw: Any) -> Any:  # noqa: N803
            return DataFrame()

    _install(monkeypatch, f'{where}.dataframe', DataFrame=DataFrame)
    _install(monkeypatch, f'{where}.session', SparkSession=SparkSession)
    _install(monkeypatch, where, SparkSession=SparkSession, DataFrame=DataFrame)
    return SparkSession()


#: Each warehouse's statement, its parameters as the reference passes them
#: (Databricks SQL and Spark a list of ``?`` markers' values, Snowflake a
#: ``qmark`` tuple), and how the leg runs it and reads Arrow back.
STATEMENT = 'SELECT user_id FROM orders WHERE user_id > ?'


def _run_databricks_sql(connection: Any) -> Any:
    cursor = connection.cursor()
    cursor.execute(STATEMENT, [6])
    return cursor.fetchall_arrow()


def _run_snowflake(connection: Any) -> Any:
    cursor = connection.cursor()
    cursor.execute(STATEMENT, params=(6,))
    return cursor.fetch_arrow_all()


def _run_spark(session: Any) -> Any:
    return session.sql(STATEMENT, args=[6]).toArrow()


def _leg_drivers(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[str, str, Any, Any]]:
    r"""Each warehouse leg's fake connection and its run: (leg, the driver
    module prefix the reference's dispatch reads, connection, run)."""
    yield (
        'databricks-sql',
        'databricks.sql',
        _databricks_sql(monkeypatch),
        _run_databricks_sql,
    )
    yield (
        'snowflake',
        'snowflake.connector',
        _snowflake(monkeypatch),
        _run_snowflake,
    )
    yield 'databricks-spark', 'pyspark.', _spark(monkeypatch, False), _run_spark
    yield 'databricks-spark', 'pyspark.', _spark(monkeypatch, True), _run_spark


def _dispatched_as(value: Any, prefix: str) -> bool:
    r"""Whether the reference's ``Source`` dispatch reads ``value`` as the
    driver at ``prefix`` (``sdm_connectors/source.py:68-106``: a Spark
    session by a ``SparkSession`` class under ``pyspark.``, any other by a
    class of its MRO under the driver's module)."""
    mro = type(value).__mro__
    if prefix == 'pyspark.':
        return any(
            cls.__name__ == 'SparkSession'
            and (cls.__module__ or '').startswith('pyspark.')
            for cls in mro
        )
    return any(
        (cls.__module__ or '') == prefix
        or (cls.__module__ or '').startswith(prefix + '.')
        for cls in mro
    )


def test_t32_the_pending_list_only_shrinks_here() -> None:
    assert _pending_names('pending') <= _pending_names('authored')


@pending
def test_t32_the_capture_records_each_warehouse_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""On each warehouse leg the capture hands the source a connection
    the reference reads as that warehouse, gives back what the driver gave,
    and records the statement as sent, its typed parameters and the Arrow
    result (its digest, as the capture records DuckDB's)."""
    run = runner()
    assert run.WAREHOUSES == WAREHOUSES
    expected = _table()
    seen = []
    for leg, prefix, connection, call in _leg_drivers(monkeypatch):
        log = run.DriverLog()
        captured = run.capture(connection, log, leg)
        assert _dispatched_as(captured, prefix), leg
        answer = call(captured)
        assert answer.equals(expected), leg
        [statement] = log.statements
        assert statement['sql'] == STATEMENT, leg
        assert statement['parameters'] == run._parameters([6]), leg
        [result] = statement['results']
        assert result['rows'] == 2, leg
        assert result['digest'] == run._arrow_digest(expected), leg
        seen.append(leg)
    assert seen == [
        'databricks-sql',
        'snowflake',
        'databricks-spark',
        'databricks-spark',
    ]


@pending
def test_t32_the_statement_log_logs_each_warehouse_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""The statement log on each warehouse leg logs the statement each
    driver runs, called on the driver's own objects (not through the
    capture), and leaves the driver as it was once the case is done."""
    run = runner()
    for leg, _, connection, call in _leg_drivers(monkeypatch):
        cls = (
            type(connection)
            if leg == 'databricks-spark'
            else type(connection.cursor())
        )
        method = 'sql' if leg == 'databricks-spark' else 'execute'
        before = cls.__dict__[method]
        with run.StatementLog(leg) as log:
            call(connection)
            assert cls.__dict__[method] is not before, leg
        assert log.texts == [STATEMENT], leg
        assert cls.__dict__[method] is before, leg


class _PsycopgSelf:
    r"""Stands for a psycopg cursor in a call to the class's method."""


@pending
@pytest.mark.parametrize('database', ['postgres', 'lakebase'])
def test_t32_the_statement_log_takes_psycopg_s_copy_and_stream(
    monkeypatch: pytest.MonkeyPatch, database: str
) -> None:
    r"""P2-T2.1a-2: on Postgres, and on Lakebase, which psycopg reaches,
    the statement log logs ``Cursor.copy`` and ``Cursor.stream`` as it
    logs ``execute``: each a statement the driver runs."""
    psycopg = pytest.importorskip('psycopg')
    called: list[str] = []

    def copy_(self: Any, statement: Any, params: Any = None, **kw: Any) -> Any:
        called.append('copy')
        return None

    def stream(self: Any, query: Any, params: Any = None, **kw: Any) -> Any:
        called.append('stream')
        yield (1,)

    monkeypatch.setattr(psycopg.Cursor, 'copy', copy_)
    monkeypatch.setattr(psycopg.Cursor, 'stream', stream)
    run = runner()
    with run.StatementLog(database) as log:
        psycopg.Cursor.copy(_PsycopgSelf(), 'COPY orders TO STDOUT')
        rows = list(psycopg.Cursor.stream(_PsycopgSelf(), 'SELECT 1'))
    assert rows == [(1,)]
    assert called == ['copy', 'stream']
    assert log.texts == ['COPY orders TO STDOUT', 'SELECT 1']
    assert psycopg.Cursor.copy is copy_ and psycopg.Cursor.stream is stream


@pending
def test_t32_the_statement_log_is_entered_only_as_a_context() -> None:
    r"""P2-T2.1a-2: the runner enters the statement log only in a ``with``,
    so a case that fails mid-call leaves the driver's classes as they were
    (today ``_call`` calls ``StatementLog(database).__enter__()`` outside a
    ``try``/``finally``)."""
    tree = ast.parse(RUNNER.read_text(encoding='utf-8'))
    in_with: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                in_with.add(id(item.context_expr))
    uses = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'StatementLog'
    ]
    assert uses, 'the runner never makes a statement log'
    outside = [node.lineno for node in uses if id(node) not in in_with]
    assert not outside, f'StatementLog made outside a with at lines {outside}'
    entered = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == '__enter__'
        and 'StatementLog' in ast.unparse(node.value)
    ]
    assert not entered, entered


@pending
def test_t32_an_s2_case_runs_on_its_warehouses_only() -> None:
    r"""Choice 1: an S2 case runs only on the legs its ``warehouses``
    names, keyed ``<case>@<leg>``; an S1 case without ``databases`` still
    runs on the three offline databases."""
    run = runner()
    assert run.case_databases(_s2_case('snowflake')) == ('snowflake',)
    both = _s2_case()
    both['warehouses'] = ['lakebase', 'databricks-sql']
    assert run.case_databases(both) == ('databricks-sql', 'lakebase')
    assert run.case_key('t32.format.s2', 'databricks-spark') == (
        't32.format.s2@databricks-spark'
    )
    s1 = json.loads(
        (ROOT / 'contract/cases/s1/base.binary.json').read_text(
            encoding='utf-8'
        )
    )
    assert run.case_databases(s1)[:3] == ('duckdb', 'sqlite', 'postgres')


@pending
def test_t32_placeholders_resolve_from_the_environment() -> None:
    r"""The S2 test authors' gap: ``{catalog}``, ``{schema}`` and the
    existing views' placeholders in a case's input are resolved from the
    environment for a live run, and a placeholder with nothing set refuses
    by its name, never by a value."""
    run = runner()
    case = _s2_case()
    case['input']['view'] = {
        'kind': 'metric_view',
        'name': '{shop_metric_view}',
        'yaml': 'source: "`{catalog}`.`{schema}`.orders"\n',
    }
    environ = {
        'SDM_CAPTURE_CATALOG': 'cat_t32',
        'SDM_CAPTURE_SCHEMA': 'sch_t32',
        'SDM_CAPTURE_SHOP_METRIC_VIEW': 'cat_t32.sch_t32.mv_t32',
    }
    resolved = run.resolve_placeholders(case, environ)
    assert resolved['input']['view'] == {
        'kind': 'metric_view',
        'name': 'cat_t32.sch_t32.mv_t32',
        'yaml': 'source: "`cat_t32`.`sch_t32`.orders"\n',
    }
    assert case['input']['view']['name'] == '{shop_metric_view}'
    assert resolved['why'] == case['why']
    unset = dict(environ)
    del unset['SDM_CAPTURE_SHOP_METRIC_VIEW']
    with pytest.raises(run.RunnerFailure) as refused:
        run.resolve_placeholders(case, unset)
    assert '{shop_metric_view}' in str(refused.value)
    assert 'cat_t32' not in str(refused.value)


@pending
def test_t32_the_input_hash_follows_the_input_and_base_only() -> None:
    r"""A case's input hash changes with its ``input``, its ``until`` and
    its base's statements, which the old client's answer depends on, and
    with nothing else: a case given ``core``, a new ``why`` or new paths
    keeps its capture."""
    run = runner()
    case = _s2_case()
    bases = run.load_bases()
    first = run.input_hash(case)
    assert re.fullmatch(r'[0-9a-f]{64}', first)
    assert run.input_hash(copy.deepcopy(case), bases) == first
    changed = []
    for key, value in (
        (
            ('input', 'pql'),
            'PREDICT COUNT(orders.*, 0, 7, days) > 0 FOR EACH users.user_id',
        ),
        (('input', 'setup'), ['CREATE TABLE t32 AS SELECT 1 AS x']),
        (('until',), 'predict'),
    ):
        other = copy.deepcopy(case)
        at = other
        for part in key[:-1]:
            at = at[part]
        at[key[-1]] = value
        changed.append(run.input_hash(other) != first)
    assert changed == [True, True, True]
    moved = copy.deepcopy(bases)
    moved[case['input']['base']] = [
        *moved[case['input']['base']],
        'CREATE TABLE t32_extra AS SELECT 1 AS x',
    ]
    assert run.input_hash(case, moved) != first
    kept = copy.deepcopy(case)
    kept['why'] = 'another why'
    kept['paths'] = ['schema.from-source/tables/databricks-sql']
    kept['routes'] = kept['paths']
    kept['cells'] = ['int64/foreign-key']
    kept['source'] = 'another source'
    kept['status'] = 'diagnostic'
    kept['expect'] = {
        'outcome': 'refusal',
        'stage': 'from_source',
        'error': 'StructuredDataError',
        'code': None,
    }
    kept['core'] = {
        'describe_source': {
            'provider': {'provider': 'sql', 'dialect': 'databricks'}
        }
    }
    assert run.input_hash(kept) == first


# --- capture_check.py --------------------------------------------------------------


def _manifest(
    path: Path, hashes: dict[str, str], commit: str = 'a' * 40, **over: Any
) -> Path:
    manifest = {
        'format': 'sdm-capture/1',
        'recording_commit': commit,
        'reference_commit': '43e55f4f58924384c3cb52f67c8ed2e73ba45fa1',
        'legs': {
            'databricks-sql': {
                'runtime': 'sql-warehouse',
                'python': '3.12.13',
                'pandas': '3.0.6',
                'driver': '4.0.0',
            }
        },
        'cases': {key: {'input_hash': value} for key, value in hashes.items()},
        **over,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    return path


def _cases_dir(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    r"""Two S2 cases on the SQL warehouse, written as case files."""
    cases = tmp_path / 'cases'
    cases.mkdir()
    made = []
    for name in ('t32.capture.one', 't32.capture.two'):
        case = _s2_case()
        case['id'] = name
        (cases / f'{name}.json').write_text(json.dumps(case), encoding='utf-8')
        made.append(case)
    return cases, made


def _capture_check(
    *arguments: str, cwd: Path = ROOT
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CAPTURE_CHECK), *arguments],
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        env={**os.environ, **QUIET},
        timeout=600,
    )


@pending
def test_t32_capture_check_refuses_an_unknown_capture_format(
    tmp_path: Path,
) -> None:
    r"""A capture in a format version the checker does not read is refused
    by name; ``sdm-capture/1`` is read."""
    cases, made = _cases_dir(tmp_path)
    run = runner()
    hashes = {
        f'{case["id"]}@databricks-sql': run.input_hash(case) for case in made
    }
    good = _manifest(tmp_path / 'good/capture.json', hashes)
    done = _capture_check('captures', '--cases', str(cases), str(good))
    assert done.returncode == 0, done.stdout + done.stderr
    bad = _manifest(
        tmp_path / 'bad/capture.json', hashes, format='sdm-capture/9'
    )
    done = _capture_check('captures', '--cases', str(cases), str(bad))
    assert done.returncode == 1, done.stdout + done.stderr
    assert 'sdm-capture/9' in done.stdout + done.stderr
    assert 'Traceback' not in done.stderr


@pending
def test_t32_capture_check_lists_a_case_changed_or_missing_as_pending_live(
    tmp_path: Path,
) -> None:
    r"""Choice 2, "When the capture runs": a case whose input hash is not
    the one recorded, or that runs on a leg the capture does not hold, is
    pending live: listed, not passed, and failing a locked run; a case
    whose hash matches is not listed."""
    cases, made = _cases_dir(tmp_path)
    run = runner()
    one, two = (f'{case["id"]}@databricks-sql' for case in made)
    current = run.input_hash(made[0])
    manifest = _manifest(tmp_path / 'c/capture.json', {one: current})
    done = _capture_check('captures', '--cases', str(cases), str(manifest))
    assert done.returncode == 0, done.stdout + done.stderr
    listed = re.findall(r'pending live: ([\w.@-]+)', done.stdout)
    assert listed == [two], done.stdout
    stale = _manifest(
        tmp_path / 'd/capture.json',
        {one: 'f' * 64, two: run.input_hash(made[1])},
    )
    done = _capture_check('captures', '--cases', str(cases), str(stale))
    assert re.findall(r'pending live: ([\w.@-]+)', done.stdout) == [one], (
        done.stdout
    )
    locked = _capture_check(
        'captures', '--locked', '--cases', str(cases), str(stale)
    )
    assert locked.returncode == 1, locked.stdout + locked.stderr
    whole = _manifest(
        tmp_path / 'e/capture.json',
        {one: current, two: run.input_hash(made[1])},
    )
    done = _capture_check(
        'captures', '--locked', '--cases', str(cases), str(whole)
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert 'pending live' not in done.stdout


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), *arguments],
        capture_output=True,
        text=True,
        check=True,
        stdin=subprocess.DEVNULL,
        env={
            **os.environ,
            **QUIET,
            'GIT_AUTHOR_NAME': 't32',
            'GIT_AUTHOR_EMAIL': 't32@example.invalid',
            'GIT_COMMITTER_NAME': 't32',
            'GIT_COMMITTER_EMAIL': 't32@example.invalid',
        },
        timeout=120,
    ).stdout.strip()


@pending
def test_t32_capture_check_holds_a_captures_only_commit_to_the_final_candidate(
    tmp_path: Path,
) -> None:
    r"""The live-test plan's one exception: a commit on the final candidate
    changing only ``contract/captures/**``, whose captures all name the
    final candidate as their recording commit, passes; one naming another
    commit, or changing a path outside the captures, is refused."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', 'main')
    (repo / 'contract/cases').mkdir(parents=True)
    (repo / 'contract/cases/a.json').write_text('{}\n')
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-q', '-m', 'the final candidate')
    candidate = _git(repo, 'rev-parse', 'HEAD')
    results = {}
    for name, commit, outside in (
        ('names the candidate', candidate, False),
        ('names another commit', 'b' * 40, False),
        ('changes a case too', candidate, True),
    ):
        _git(repo, 'checkout', '-q', '-B', name.replace(' ', '-'), candidate)
        _manifest(
            repo / 'contract/captures/e1/capture.json',
            {'x@databricks-sql': 'c' * 64},
            commit=commit,
        )
        if outside:
            (repo / 'contract/cases/a.json').write_text('{"changed": 1}\n')
        _git(repo, 'add', '-A')
        _git(repo, 'commit', '-q', '-m', name)
        done = _capture_check(
            'commit', '--candidate', candidate, 'HEAD', cwd=repo
        )
        assert 'Traceback' not in done.stderr, done.stderr
        results[name] = done.returncode
    assert results == {
        'names the candidate': 0,
        'names another commit': 1,
        'changes a case too': 1,
    }


@pending
def test_t32_captures_are_a_protected_path() -> None:
    r"""Choice 2: ``contract/captures/**`` is protected, so a capture
    changes only in a landing whose trailer names it."""
    ratchet = _module('t32_recording_ratchet', RATCHET)
    rules = ratchet.patterns(
        (ROOT / 'contract/protected-paths.txt').read_text(encoding='utf-8')
    )
    for path in (
        'contract/captures/c1/capture.json',
        'contract/captures/c1/databricks-sql/tables/orders.arrow',
    ):
        assert ratchet.protects(rules, path), path


@pending
def test_t32_the_runner_refuses_a_client_of_another_contract_version() -> None:
    r"""P2-MID2-7: a stale installed wheel is refused by the runner, which
    names the installed contract version and the checkout's."""
    run = runner()
    text = (ROOT / 'rust/crates/sdm-contract/src/lib.rs').read_text(
        encoding='utf-8'
    )
    [checkout] = re.findall(r'pub const CONTRACT_VERSION: u32 = (\d+);', text)
    version = int(checkout)
    assert run.contract_refusal(version) is None
    refused = run.contract_refusal(version - 1)
    assert refused is not None
    assert str(version - 1) in refused and str(version) in refused
