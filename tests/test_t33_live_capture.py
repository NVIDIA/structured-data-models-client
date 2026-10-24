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
import types
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
    with pytest.raises(SystemExit, match='no live write guard'):
        capture_run.run(
            leg, ROOT, ROOT / 'never-written', guard_path=ROOT / 'no-guard.py'
        )
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
        'LIVE_CONFIG': '/Volumes/seeded/volume/live.json',
        'CAPTURE_CATALOG': 'seeded_catalog',
        'CAPTURE_SCHEMA': 'seeded_schema',
        'TEST_DATABASE': 'seeded_catalog',
        'TEST_SCHEMA': 'seeded_schema',
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


def test_a_write_outside_the_test_schema_never_reaches_the_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review r3 (Codex P2-4), r4 (Codex P0-1, Claude P1-1): the guard is
    # installed on the driver's own cursor class; the case's placeholder
    # resolves to WRONG_SCHEMA, and neither the loader's writes nor a
    # statement on any cursor reach the driver outside ALLOWED_SCHEMA.
    import types

    import duckdb

    # The repository's guard, as the run loads it (scripts/live).
    guard = capture_run.load_guard()
    for dialect in ('SNOWFLAKE_', 'DATABRICKS_', 'POSTGRES_'):
        for name in ('DATABASE', 'SCHEMA'):
            monkeypatch.delenv(f'SDM_LIVE_{dialect}TEST_{name}', raising=False)
    monkeypatch.setenv('SDM_LIVE_TEST_DATABASE', 'REVIEW_DB')
    monkeypatch.setenv('SDM_LIVE_TEST_SCHEMA', 'ALLOWED_SCHEMA')
    monkeypatch.delenv('SDM_CAPTURE_CATALOG', raising=False)
    monkeypatch.delenv('SDM_CAPTURE_SCHEMA', raising=False)
    loader = capture_run._module('databases')
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
    uninstall = capture_run.install_guard('snowflake', guard, loader)
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
        capture_run.load_guard(tmp_path / 'write_guard.py')
    monkeypatch.setenv('SDM_CAPTURE_SCHEMA', 'WRONG_SCHEMA')
    with pytest.raises(SystemExit, match='SDM_CAPTURE_SCHEMA is not'):
        capture_run.install_guard('snowflake', guard, loader)
    monkeypatch.delenv('SDM_LIVE_TEST_SCHEMA')
    with pytest.raises(SystemExit, match='no test schema'):
        capture_run.install_guard('snowflake', guard, loader)


def test_lakebase_runs_through_the_same_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Review r4 (Codex P0-4, Claude P2-2): Lakebase has no check of its
    # own; its psycopg classes are guarded, and the loader's drops carry no
    # CASCADE.
    import psycopg

    guard = capture_run.load_guard()
    for name in ('DATABASE', 'SCHEMA'):
        monkeypatch.delenv(f'SDM_LIVE_POSTGRES_TEST_{name}', raising=False)
    monkeypatch.setenv('SDM_LIVE_TEST_SCHEMA', 'live_tests')
    monkeypatch.delenv('SDM_CAPTURE_SCHEMA', raising=False)
    assert not hasattr(capture_run, '_postgres_check')
    loader = capture_run._module('databases')
    monkeypatch.setattr(loader, 'WRITE_GUARD', None)
    uninstall = capture_run.install_guard('lakebase', guard, loader)
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
    assert 'capture_run.job_environment' in calls
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
    assert set(capture_run.JOB_ENVIRONMENT) <= set(parameters)
    # The guard is the repository's, never a file on the volume.
    assert not any('guard' in name for name in parameters)
    # The parameters set the run's variables; an empty one sets none.
    for variable in capture_run.JOB_ENVIRONMENT.values():
        monkeypatch.delenv(variable, raising=False)
    found = capture_run.job_environment(
        {
            'capture_schema': 'zq_hidden',
            'live_config': '/v/c.json',
            'test_schema': '',
        }
    )
    assert found == ['SDM_LIVE_CONFIG', 'SDM_CAPTURE_SCHEMA']
    import os

    assert os.environ['SDM_CAPTURE_SCHEMA'] == 'zq_hidden'
    assert 'SDM_LIVE_TEST_SCHEMA' not in os.environ


# Round 9 (Claude P1-1, P1-2): the run records what it can.

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


@pytest.mark.parametrize('leg', capture_run.LEGS)
def test_the_recorders_own_statements_pass_each_legs_guard(leg: str) -> None:
    # Claude r9 P1-1: the statements the recorder sends outside the
    # capture (the session's zone, the schema's tables, the read-back, the
    # Spark path's USE) pass the leg's guard.
    runner = capture_run._module('run_python')
    loader = capture_run._module('databases')
    guard_module = capture_run.load_guard()
    database, schema = _GUARD_NAMES[leg]
    guard = guard_module.WriteGuard(capture_run.DIALECT[leg], database, schema)
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


@pytest.mark.parametrize('leg', capture_run.LEGS)
def test_a_plan_leaves_the_not_live_cases_out_by_name(leg: str) -> None:
    # Claude r9 P1-2: a case whose loads the guard refuses by design (its
    # name holds a backtick or a line break) is out of the leg's plan,
    # named with its reason, never in what the leg records.
    planned = capture_run.plan(leg)
    recorded = {Path(path).stem for path in planned['record']}
    left_out = {
        case for case, (legs, _) in capture_run.NOT_LIVE.items() if leg in legs
    }
    assert left_out, leg
    assert not recorded & left_out
    named = {entry.split(':', 1)[0] for entry in planned['not_live']}
    assert named == left_out
    assert all('guard' in entry for entry in planned['not_live'])


def test_a_failed_case_is_recorded_and_the_rest_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Claude r9 P1-2, r10 P1 and P2s: one case's failure ends that case
    # alone, written with its real reason (the runner's failure, any other
    # error, or a guard refusal of a statement the old client sent, which
    # its call would otherwise record as the reference's answer); the
    # others are recorded and imported, and the run exits 1.
    names = ('a', 'b', 'c', 'd', 'e')
    paths = [str(tmp_path / f'{name}.json') for name in names]
    monkeypatch.setattr(
        capture_run,
        'plan',
        lambda leg, e0=False: {
            'record': paths,
            'environment': {'SDM_LIVE_CONFIG': True},
        },
    )
    monkeypatch.setattr(capture_run, 'install_guard', lambda *a: lambda: None)
    guard = capture_run.load_guard()
    imported: list[list[str]] = []

    class Failure(Exception):
        pass

    def load_cases(given: list[Path]) -> list[tuple[dict[str, Any], str]]:
        return [({'id': Path(given[0]).stem}, 'digest')]

    def run(mode: str, cases: Any, reference: Any, **options: Any) -> list[Any]:
        case = cases[0][0]['id']
        if case == 'b':
            raise Failure(
                'lakebase could not be opened or loaded (WriteRefused)'
            )
        if case == 'c':
            raise ValueError('a driver error outside the runner')
        if case == 'd':
            # The guard refuses one of the old client's own reads; its call
            # turns that into the case's refusal outcome, and the run goes on.
            guard.REFUSALS.append(
                'SELECT SETSEED: SETSEED(), a function on no list'
            )
        options['record'].write_text(json.dumps({'case': case}) + '\n')
        return [{'case': case}]

    runner = types.SimpleNamespace(
        run=run, load_cases=load_cases, RunnerFailure=Failure
    )
    monkeypatch.setattr(
        capture_run,
        '_named',
        lambda name: (
            runner if name == 'run_python' else types.SimpleNamespace()
        ),
    )
    monkeypatch.setattr(capture_run, 'load_guard', lambda *a: guard)
    captures = capture_run._module('captures')

    def import_recording(path: Path, out: Path) -> list[Path]:
        imported.append(
            [
                json.loads(line)['case']
                for line in Path(path).read_text().splitlines()
            ]
        )
        return []

    monkeypatch.setattr(captures, 'import_recording', import_recording)
    status = capture_run.run('lakebase', tmp_path, tmp_path / 'out')
    assert status == 1
    assert imported == [['a', 'e']]
    failed = json.loads((tmp_path / 'out' / 'lakebase.failed.json').read_text())
    assert failed == {
        'b': 'lakebase could not be opened or loaded (WriteRefused)',
        'c': 'ValueError: a driver error outside the runner',
        'd': 'the write guard refused: SELECT SETSEED: SETSEED(), a function on no list',
    }


def test_the_job_loads_the_reference_client_in_its_own_process(
    tmp_path: Path,
) -> None:
    # C-1's Spark leg: a notebook's environment reads no .pth file, so the
    # job puts the checkout's client source first on sys.path in the
    # recording's own process, over any sdm_client imported already, and
    # the runner's check that the client is the checkout's passes.
    import subprocess

    reference = tmp_path / 'reference'
    client = reference / capture_run.CLIENT_SOURCE / 'sdm_client'
    client.mkdir(parents=True)
    (client / '__init__.py').write_text('')
    stale = tmp_path / 'installed/sdm_client'
    stale.mkdir(parents=True)
    (stale / '__init__.py').write_text('')
    probe = (
        'import importlib.util, sys; from pathlib import Path; '
        f'sys.path.insert(0, {str(stale.parent)!r}); import sdm_client; '
        "spec = importlib.util.spec_from_file_location('c', "
        f'{str(LIVE / "capture_run.py")!r}); '
        'capture = importlib.util.module_from_spec(spec); '
        'spec.loader.exec_module(capture); '
        f'capture.use_reference_client(Path({str(reference)!r})); '
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
    # The job writes no .pth file (capture_run.py bootstrap) any more.
    assert "'bootstrap'" not in (LIVE / 'capture_job.py').read_text()
    with pytest.raises(SystemExit, match='no reference client'):
        capture_run.use_reference_client(tmp_path / 'installed')


def test_the_job_installs_only_its_hosts_wheels(tmp_path: Path) -> None:
    # C-1's Spark leg: serverless runs on x86_64 and aarch64 hosts, so the
    # wheel folder holds both platforms' builds, and the job hands pip only
    # the ones its own host installs (an unsupported wheel named to pip
    # fails the whole install).
    from packaging.tags import Tag

    names = [
        'structured_data_models_api-1.0.0-py3-none-any.whl',
        'structured_data_models_relational_engine-1.0.0-cp312-cp312-'
        'manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl',
        'structured_data_models_relational_engine-1.0.0-cp312-cp312-'
        'manylinux_2_28_aarch64.whl',
        'duckdb-1.5.6-cp312-cp312-manylinux_2_26_aarch64.'
        'manylinux_2_28_aarch64.whl',
    ]
    for name in names:
        (tmp_path / name).write_bytes(b'')
    arm = [
        Tag('cp312', 'cp312', 'manylinux_2_28_aarch64'),
        Tag('py3', 'none', 'any'),
    ]
    chosen = capture_run.installable_wheels(tmp_path, arm)
    assert [Path(path).name for path in chosen] == sorted(
        name for name in names if 'x86_64' not in name
    )
    x86 = [
        Tag('cp312', 'cp312', 'manylinux_2_28_x86_64'),
        Tag('py3', 'none', 'any'),
    ]
    assert [
        Path(path).name
        for path in capture_run.installable_wheels(tmp_path, x86)
    ] == sorted(names[:2])
    # The job hands pip the chosen wheels, never the folder's every wheel.
    assert (
        'capture_run.installable_wheels'
        in (LIVE / 'capture_job.py').read_text()
    )
