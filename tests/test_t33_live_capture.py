# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""C-1's and E-0's templates in dry mode (T3.3): the plan of each leg
reaches nothing and names what it records, what the loader refuses and
the environment it needs, by name only; the serverless job template fills
and passes ``dbx_runs.py``'s check with seeded values, and names no
workspace value of its own."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / 'scripts/live'


def _module(name: str):
    spec = importlib.util.spec_from_file_location(
        f't33_live_{name}', LIVE / f'{name}.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


capture_run = _module('capture_run')
dbx_runs = _module('dbx_runs')


@pytest.mark.parametrize('leg', ['snowflake', 'lakebase'])
def test_a_leg_s_plan_reaches_nothing_and_names_its_needs(
    leg: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in capture_run.ENVIRONMENT[leg]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('SDM_CAPTURE_SCHEMA', 'zq_private_schema')
    planned = capture_run.plan(leg)
    assert planned['campaign'] == 'C-1'
    assert len(planned['record']) > 200
    # The batching cases stay off the warehouse legs (KO3-45).
    assert not any(
        '100000' in path or '100001' in path for path in planned['record']
    )
    assert planned['environment']['SDM_CAPTURE_SCHEMA'] is True
    assert planned['environment']['SDM_LIVE_CONFIG'] is False
    assert 'zq_private_schema' not in json.dumps(planned)
    # The Databricks legs one after the other.
    order = planned['order']
    assert (
        abs(order.index('databricks-sql') - order.index('databricks-spark')) > 1
    )
    monkeypatch.delenv(capture_run.GUARD_VARIABLE, raising=False)
    with pytest.raises(SystemExit, match='no live write guard'):
        capture_run.run(leg, ROOT, ROOT / 'never-written')
    assert not (ROOT / 'never-written').exists()


def test_the_job_template_fills_and_passes_the_runs_check() -> None:
    text = (LIVE / 'capture.job.template.json').read_text()
    template = json.loads(text)
    assert not re.search(
        r'cloud\.databricks\.com|snowflakecomputing|/Users/', text
    )
    values = {
        'FOLDER': '/Workspace/seeded/folder',
        'NOTEBOOK': 'capture_job',
        'LEG': 'lakebase',
        'CAMPAIGN': 'c1',
        'REPO_DIR': '/Volumes/seeded/repo',
        'REFERENCE_DIR': '/Volumes/seeded/reference',
        'VOLUME_DIR': '/Volumes/seeded/volume',
        'SHA': 'a' * 40,
        'SHA7': 'a' * 7,
        'UUID': '00000000-0000-0000-0000-000000000000',
        'ENVIRONMENT_VERSION': '4',
    }
    payload = dbx_runs.fill(
        {k: v for k, v in template.items() if not k.startswith('_')}, values
    )
    dbx_runs.check(payload, values['FOLDER'], 'unused-policy')
    parameters = payload['tasks'][0]['notebook_task']['base_parameters']
    assert parameters['leg'] == 'lakebase' and parameters['campaign'] == 'c1'


def test_snowflake_s_plan_creates_a_semantic_view_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review r2 (Claude): a semantic view created from a case's YAML comes
    # first, its statements shown, its tables named as the YAML reads them.
    cases = tmp_path / 'cases'
    (cases / 's2').mkdir(parents=True)
    (cases / 's2/view.semantic.binary.json').write_text(
        (ROOT / 'tests/t33-semantic-view-case.json').read_text()
    )
    (cases / 's1').mkdir()
    (cases / 's1/base.regression.json').write_text(
        (ROOT / 'contract/cases/s1/base.regression.json').read_text()
    )
    monkeypatch.setattr(capture_run, 'CASES', cases)
    planned = capture_run.plan('snowflake')
    assert planned['record'][0].endswith('view.semantic.binary.json')
    first = planned['first']
    assert first['case'] == 'view.semantic.binary'
    statements = '\n'.join(first['statements'])
    assert (
        'CREATE TABLE "{catalog}"."{schema}"."users" ("USER_ID"' in statements
    )
    assert 'SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML' in statements


def test_the_job_imports_the_reference_checkout_s_own_client(
    tmp_path: Path,
) -> None:
    # Review r3 (Codex P2): the serverless job installs the reference's
    # wheels, but the recording runs the checkout's own client, which the
    # runner requires; the bootstrap names the checkout's source.
    import subprocess

    reference = tmp_path / 'reference'
    client = reference / capture_run.CLIENT_SOURCE / 'sdm_client'
    client.mkdir(parents=True)
    (client / '__init__.py').write_text('')
    site = tmp_path / 'site'
    site.mkdir()
    capture_run.bootstrap_reference(reference, site)
    probe = (
        'import importlib.util, site, sys; from pathlib import Path; '
        f'site.addsitedir({str(site)!r}); '
        "spec = importlib.util.spec_from_file_location('r', "
        f'{str(ROOT / "contract/runners/run_python.py")!r}); '
        'module = importlib.util.module_from_spec(spec); '
        "sys.modules['r'] = module; spec.loader.exec_module(module); "
        f'module._require_reference_client(Path({str(reference)!r})); '
        "print('ok')"
    )
    done = subprocess.run(
        [sys.executable, '-I', '-S', '-c', probe],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.stdout.strip() == 'ok', done.stderr[-2000:]
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(SystemExit, match='no reference client'):
        capture_run.bootstrap_reference(empty, site)


#: A write guard with the owner's guard's interface (WriteGuard,
#: guard_connection; the guard itself lives outside the repository until
#: T3.7): a write naming a schema other than the test schema is refused.
FAKE_GUARD = """
class WriteRefused(PermissionError):
    pass


class WriteGuard:
    def __init__(self, dialect, database=None, schema=None):
        self.schema = schema

    def check(self, sql):
        words = sql.split()
        if words[0].upper() in ('SELECT', 'SHOW', 'DESCRIBE'):
            return
        if f'"{self.schema}".' not in sql and f'`{self.schema}`.' not in sql:
            raise WriteRefused('a write outside the test schema')


def guard_connection(conn, guard):
    class Cursor:
        def __init__(self, cursor):
            self._cursor = cursor

        def execute(self, sql, *a, **k):
            guard.check(sql)
            return self._cursor.execute(sql, *a, **k)

        def __getattr__(self, name):
            return getattr(self._cursor, name)

    class Connection:
        def cursor(self, *a, **k):
            return Cursor(conn.cursor(*a, **k))

        def __getattr__(self, name):
            return getattr(conn, name)

    return Connection()
"""


def test_a_write_outside_the_test_schema_never_reaches_the_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review r3 (Codex P2-4): the guard allows ALLOWED_SCHEMA, the case's
    # placeholder resolves to WRONG_SCHEMA; the loader's first write is
    # refused before the driver sees anything.
    import duckdb

    guard_path = tmp_path / 'write_guard.py'
    guard_path.write_text(FAKE_GUARD)
    guard = capture_run.load_guard(guard_path)
    monkeypatch.setenv('SDM_LIVE_TEST_DATABASE', 'REVIEW_DB')
    monkeypatch.setenv('SDM_LIVE_TEST_SCHEMA', 'ALLOWED_SCHEMA')
    loader = capture_run._module('databases')
    monkeypatch.setattr(loader, 'WRITE_GUARD', None)

    class Runner:
        RunnerFailure = RuntimeError

        @staticmethod
        def connect_leg(*arguments, **keywords):
            return Driver(), lambda: None

    seen: list[str] = []

    class Cursor:
        def execute(self, sql, *a, **k):
            seen.append(sql)

        def fetchall(self):
            return []

        def close(self):
            pass

    class Driver:
        def cursor(self):
            return Cursor()

    capture_run.install_guard('snowflake', guard, Runner, loader)
    connection, _ = Runner.connect_leg('snowflake', {}, {}, None)

    class Target:
        names = {'catalog': 'REVIEW_DB', 'schema': 'WRONG_SCHEMA'}
        view = None

    Target.connection = connection
    duck = duckdb.connect()
    duck.execute('CREATE TABLE users (i INTEGER)')
    with pytest.raises(PermissionError, match='outside the test schema'):
        loader.load(duck, 'snowflake', Target())
    assert seen == []
    # The test schema itself loads.
    Target.names = {'catalog': 'REVIEW_DB', 'schema': 'ALLOWED_SCHEMA'}
    loader.load(duck, 'snowflake', Target())
    assert any(
        s.startswith('CREATE TABLE "REVIEW_DB"."ALLOWED_SCHEMA"') for s in seen
    )
    # No guard, no run.
    with pytest.raises(SystemExit, match='no live write guard'):
        capture_run.load_guard(None)


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
        elif 'CURRENT_SETTING' in sql:
            self.result = pa.table({'ZONE': ['UTC']})

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
    # Snowflake: the YAML's quoted base tables and unquoted columns resolve.
    config = tmp_path / 'live.json'
    config.write_text(
        json.dumps(
            {'format': 'sdm-live/1', 'legs': {'snowflake': {'driver': {}}}}
        )
    )
    monkeypatch.setenv('SDM_LIVE_CONFIG', str(config))
    monkeypatch.setenv('SDM_CAPTURE_CATALOG', 'REVIEW_DB')
    monkeypatch.setenv('SDM_CAPTURE_SCHEMA', 'ALLOWED_SCHEMA')
    runner = capture_run._named('run_python')
    loader = capture_run._named('databases')
    monkeypatch.setattr(loader, '_SCHEMAS', {})
    warehouse = _CaseSensitiveSnowflake()
    monkeypatch.setattr(
        runner, 'connect_leg', lambda *a, **k: (warehouse, lambda: None)
    )
    case = json.loads((ROOT / 'tests/t33-semantic-view-case.json').read_text())
    live = runner.Live(case, runner.load_bases(), 'snowflake')
    try:
        assert sorted(name for name, _ in live.tables) == ['orders', 'users']
        assert any(
            sql.startswith('CALL SYSTEM$CREATE_SEMANTIC_VIEW')
            for sql in warehouse.ran
        )
    finally:
        live.close()


def test_the_job_runs_in_its_notebook_with_its_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review r3 (Claude P2): the job's run reads the live configuration
    # and the placeholders' names from its environment file, and runs in
    # the notebook's own process, where the Spark leg's session is.
    import ast

    source = (LIVE / 'capture_job.py').read_text()
    tree = ast.parse(source)
    calls = {
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert 'capture_run.run' in calls
    assert 'capture_run.load_environment' in calls
    # No child process runs the recording.
    runs = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == 'subprocess.run'
    ]
    assert not any("'run'" in call for call in runs), runs
    template = json.loads((LIVE / 'capture.job.template.json').read_text())
    parameters = template['tasks'][0]['notebook_task']['base_parameters']
    assert {'environment_file', 'guard_path'} <= set(parameters)
    # The environment file: its names set, a value never printed, a name
    # outside what a run reads refused.
    path = tmp_path / 'environment.json'
    path.write_text(
        json.dumps({'SDM_CAPTURE_SCHEMA': 'zq_hidden', 'SDM_LIVE_CONFIG': 'x'})
    )
    for name in ('SDM_CAPTURE_SCHEMA', 'SDM_LIVE_CONFIG'):
        monkeypatch.delenv(name, raising=False)
    assert capture_run.load_environment(path) == [
        'SDM_CAPTURE_SCHEMA',
        'SDM_LIVE_CONFIG',
    ]
    import os

    assert os.environ['SDM_CAPTURE_SCHEMA'] == 'zq_hidden'
    path.write_text(json.dumps({'HOME': '/x'}))
    with pytest.raises(SystemExit, match='HOME'):
        capture_run.load_environment(path)
