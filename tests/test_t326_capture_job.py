# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The capture job's four fixes from the Spark recording's root-cause
analysis (2026-10-24; T3.26): a case whose tables are the ones the case
before it loaded is not loaded again; each case's progress, and each
failure, is written as the run goes; a case past its time limit fails
alone; and each leg's job timeout is the template's, by leg."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
import types
from pathlib import Path
from typing import Any

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / 'scripts/live'


def _module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


capture_run = _module(LIVE / 'capture_run.py', 't326_capture_run')
dbx_runs = _module(LIVE / 'dbx_runs.py', 't326_dbx_runs')
databases = _module(ROOT / 'contract/runners/databases.py', 't326_databases')


def _duck(*statements: str) -> Any:
    connection = duckdb.connect()
    for statement in statements:
        connection.execute(statement)
    return connection


SHOP = (
    'CREATE TABLE users (user_id BIGINT PRIMARY KEY, age INTEGER)',
    'INSERT INTO users VALUES (1, 20), (2, 30)',
    'CREATE TABLE orders (order_id BIGINT, user_id BIGINT '
    'REFERENCES users (user_id), price DOUBLE)',
    'INSERT INTO orders VALUES (10, 1, 1.5), (11, 2, 2.5)',
)


def test_a_case_whose_tables_equal_the_previous_case_s_is_not_loaded_again() -> (
    None
):
    ran: list[str] = []
    schema = databases.CaseSchema('databricks', ran.append, ('cat', 'sch'))
    first = schema.load(_duck(*SHOP))
    loaded = list(ran)
    assert any(s.startswith('CREATE TABLE') for s in loaded)
    # The same tables, built again by the next case: nothing is sent.
    assert schema.load(_duck(*SHOP)) == first
    assert ran == loaded
    # A row changed, a constraint dropped, or a view created since: the
    # schema is loaded again.
    changed = (*SHOP[:-1], 'INSERT INTO orders VALUES (10, 1, 9.5)')
    for build in (
        lambda: schema.load(_duck(*changed)),
        lambda: schema.load(
            _duck(*(s.replace(' REFERENCES users (user_id)', '') for s in SHOP))
        ),
    ):
        before = len(ran)
        build()
        assert any(s.startswith('DROP TABLE') for s in ran[before:])
        assert any(s.startswith('CREATE TABLE') for s in ran[before:])
    schema.load(_duck(*SHOP))
    schema.views.append(('metric_view', 'mv'))
    before = len(ran)
    schema.load(_duck(*SHOP))
    assert any(s.startswith('DROP VIEW') for s in ran[before:])
    assert any(s.startswith('CREATE TABLE') for s in ran[before:])
    # Cleared, it is loaded again.
    schema.clear()
    before = len(ran)
    schema.load(_duck(*SHOP))
    assert any(s.startswith('CREATE TABLE') for s in ran[before:])


def _fake_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    behave: dict[str, Any],
) -> tuple[list[str], Any]:
    r"""``capture_run.run``'s collaborators faked: five planned cases, each
    run by ``behave[case]`` (a callable of the record path) or recorded."""
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
    monkeypatch.setattr(capture_run, 'load_guard', lambda *a: guard)

    class Failure(Exception):
        pass

    def load_cases(given: list[Path]) -> list[tuple[dict[str, Any], str]]:
        return [({'id': Path(given[0]).stem}, 'digest')]

    def run(mode: str, cases: Any, reference: Any, **options: Any) -> list[Any]:
        case = cases[0][0]['id']
        if case in behave:
            behave[case](options['record'])
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
    imported: list[str] = []
    captures = capture_run._module('captures')
    monkeypatch.setattr(
        captures,
        'import_recording',
        lambda path, out: (
            imported.extend(
                json.loads(line)['case']
                for line in Path(path).read_text().splitlines()
            )
            or []
        ),
    )
    return imported, Failure


def test_each_case_s_progress_and_each_failure_are_written_as_the_run_goes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / 'out'
    seen: dict[str, Any] = {}

    def fail_b(record: Path) -> None:
        raise ValueError('a driver error')

    def look_at_d(record: Path) -> None:
        # Written before case d runs: the progress of a to c, and b's
        # failure.
        seen['progress'] = [
            json.loads(line)
            for line in (out / 'lakebase.progress.jsonl')
            .read_text()
            .splitlines()
        ]
        seen['failed'] = json.loads((out / 'lakebase.failed.json').read_text())

    imported, _ = _fake_run(
        tmp_path, monkeypatch, {'b': fail_b, 'd': look_at_d}
    )
    assert capture_run.run('lakebase', tmp_path, out) == 1
    assert [(p['case'], p['status']) for p in seen['progress']] == [
        ('a', 'recorded'),
        ('b', 'failed'),
        ('c', 'recorded'),
    ]
    assert [p['number'] for p in seen['progress']] == [0, 1, 2]
    assert all(p['seconds'] >= 0 for p in seen['progress'])
    assert seen['failed'] == {'b': 'ValueError: a driver error'}
    assert imported == ['a', 'c', 'd', 'e']
    whole = (out / 'lakebase.progress.jsonl').read_text().splitlines()
    assert [json.loads(line)['case'] for line in whole] == list('abcde')


def test_a_case_past_its_time_limit_fails_alone_and_the_run_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def hang(record: Path) -> None:
        # A statement that never returns, interrupted by the limit; a
        # handler of Exception inside the runner does not swallow it.
        try:
            time.sleep(30)
        except Exception:  # noqa: BLE001 - what a runner's handler does
            pytest.fail('the time limit was taken for an ordinary error')

    imported, _ = _fake_run(tmp_path, monkeypatch, {'c': hang})
    started = time.monotonic()
    status = capture_run.run(
        'lakebase', tmp_path, tmp_path / 'out', case_timeout=0.3
    )
    assert time.monotonic() - started < 10
    assert status == 1
    failed = json.loads((tmp_path / 'out' / 'lakebase.failed.json').read_text())
    assert list(failed) == ['c']
    assert 'time limit' in failed['c']
    assert imported == ['a', 'b', 'd', 'e']
    # The limit is the job's parameter, and a default the heaviest case
    # recorded (549 s on Spark) is well within.
    template = json.loads((LIVE / 'capture.job.template.json').read_text())
    parameters = template['tasks'][0]['notebook_task']['base_parameters']
    assert int(parameters['case_timeout']) == capture_run.CASE_TIMEOUT
    assert capture_run.CASE_TIMEOUT >= 3 * 549


def test_each_leg_s_job_timeout_is_the_template_s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('SDM_DBX_WORKSPACE_FOLDER', '/Workspace/seeded')
    monkeypatch.setenv('SDM_DBX_POLICY_ID', 'seeded-policy')
    monkeypatch.setenv('SDM_DBX_USER', 'seeded-user')
    monkeypatch.setenv('SDM_DBX_VOLUME_DIR', '/Volumes/seeded/volume')
    monkeypatch.delenv('SDM_DBX_CLUSTER_EXTRA', raising=False)
    filled = [
        'CAMPAIGN=c1',
        'REPO_DIR=/Volumes/seeded/repo',
        'REFERENCE_DIR=/Volumes/seeded/reference',
        'ENVIRONMENT_VERSION=4',
        'LIVE_CONFIG=/Volumes/seeded/volume/live.json',
        'CAPTURE_CATALOG=seeded_catalog',
        'CAPTURE_SCHEMA=seeded_schema',
        'TEST_DATABASE=seeded_catalog',
        'TEST_SCHEMA=seeded_schema',
        'CAPTURE_NAMES={}',
    ]
    template = json.loads((LIVE / 'capture.job.template.json').read_text())
    timeouts = template['_timeout_seconds']
    # The Spark leg took about 7.8 hours (261 cases at about 107 seconds).
    assert timeouts['databricks-spark'] >= 9 * 3600
    for leg in ('databricks-spark', 'lakebase'):
        out = tmp_path / f'{leg}.json'
        arguments = argparse.Namespace(
            template=LIVE / 'capture.job.template.json',
            out=out,
            target='serverless-env4',
            notebook='capture_job',
            sha='a' * 40,
            set=[f'LEG={leg}', *filled],
            folder=None,
            policy_id=None,
            user=None,
            volume_dir=None,
            profile=None,
            cluster_extra=None,
        )
        assert dbx_runs.run_spec(arguments) == 0
        payload = json.loads(out.read_text())
        assert payload['timeout_seconds'] == timeouts[leg]
        assert [task['timeout_seconds'] for task in payload['tasks']] == [
            timeouts[leg]
        ]
        assert '_timeout_seconds' not in payload
    arguments.out = tmp_path / 'other.json'
    arguments.set = ['LEG=snowflake', *filled]
    with pytest.raises(dbx_runs.Refused, match='timeout'):
        dbx_runs.run_spec(arguments)


class _Warehouse:
    r"""A Databricks test schema standing in for the warehouse: the tables
    each ``CREATE TABLE`` makes and each ``DROP TABLE`` removes; told to,
    it fails the second drop it is sent, after the first went through."""

    NAME = re.compile(
        r'^(CREATE|DROP) TABLE (?:IF EXISTS )?`cat`\.`sch`\.`(\w+)`'
    )

    def __init__(self) -> None:
        self.tables: set[str] = set()
        self.drops = 0
        self.fail: Any = None

    def __call__(self, statement: str) -> None:
        found = self.NAME.match(statement)
        if found is None:
            return
        kind, table = found.groups()
        if kind == 'DROP':
            self.drops += 1
            if self.fail is not None and self.drops == 2:
                self.fail()
            self.tables.discard(table)
        else:
            self.tables.add(table)


@pytest.mark.parametrize('exit', ['timeout', 'driver error'])
def test_a_case_cut_short_mid_clear_leaves_the_next_case_to_load_again(
    exit: str,
) -> None:
    # Codex r1 P1: case A loads; case B, other tables, is cut short while
    # its load clears A's (past its time limit, or by a driver error);
    # case A again finds A's tables gone, and loads them.
    warehouse = _Warehouse()
    schema = databases.CaseSchema('databricks', warehouse, ('cat', 'sch'))
    schema.load(_duck(*SHOP))
    assert warehouse.tables == {'users', 'orders'}
    other = (*SHOP[:1], 'INSERT INTO users VALUES (1, 21), (2, 30)', *SHOP[2:])
    warehouse.drops = 0
    if exit == 'timeout':
        warehouse.fail = lambda: time.sleep(30)
        with pytest.raises(capture_run.CaseTimedOut):
            with capture_run._limited(0.3):
                schema.load(_duck(*other))
    else:

        def refused() -> None:
            raise RuntimeError('the driver lost its connection')

        warehouse.fail = refused
        with pytest.raises(RuntimeError):
            schema.load(_duck(*other))
    # One of A's tables is gone.
    assert warehouse.tables != {'users', 'orders'}
    warehouse.fail = None
    schema.load(_duck(*SHOP))
    assert warehouse.tables == {'users', 'orders'}


def test_a_case_limit_off_the_main_thread_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Claude r1 P2: no signal reaches a thread other than the main one, so
    # a run asked to limit each case there refuses to start, saying why,
    # rather than running with no limit; asked for none, it runs.
    import threading

    _fake_run(tmp_path, monkeypatch, {})
    outcome: dict[str, Any] = {}

    def run(limit: float | None) -> None:
        try:
            outcome[limit] = capture_run.run(
                'lakebase',
                tmp_path,
                tmp_path / f'out-{limit}',
                case_timeout=limit,
            )
        except SystemExit as refused:
            outcome[limit] = str(refused)

    for limit in (5.0, None):
        worker = threading.Thread(target=run, args=(limit,))
        worker.start()
        worker.join()
    assert 'main thread' in outcome[5.0]
    assert not (tmp_path / 'out-5.0').exists()
    assert outcome[None] == 0
