# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The recorded nightly failures: scoped baselines and omitted capture inputs."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.tooling
ROOT = Path(__file__).resolve().parents[1]


def _module(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        f't46_{name}', ROOT / f'scripts/rust_core/{name}.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def ratchet() -> Any:
    return _module('ratchet')


@pytest.fixture(scope='module')
def coverage() -> Any:
    return _module('coverage')


def _baselines() -> dict[str, Any]:
    path = ROOT / 'contract/baselines/ref-pandas3/python-core.jsonl'
    return {
        line['case']: line
        for line in map(json.loads, path.read_text().splitlines())
        if line['case'].split('@')[0] == 'target.aggregation.list-distinct'
    }


def _baseline_report(
    ratchet: Any,
    database: str | None,
    missing: bool = False,
    changed: bool = False,
    missing_baseline: bool = False,
) -> Any:
    held = _baselines()
    bare = 'target.aggregation.list-distinct'
    key = bare if database in (None, 'duckdb') else f'{bare}@{database}'
    # These are precisely the held T4.5 outcome projections from the CI failure.
    line: dict[str, Any] = {}
    for field, value in held[key]['projection'].items():
        target = line
        *parents, last = field.split('.')
        for parent in parents:
            target = target.setdefault(parent, {})
        target[last] = value
    report = ratchet.Report()
    before = copy.deepcopy(held)
    proposed = copy.deepcopy(held)
    if changed:
        proposed[key]['projection']['outcome.result'] = 'success'
    if missing_baseline:
        proposed.pop(key)
    ratchet.check_baselines(
        report,
        {'python-core': held},
        {'python-core': proposed},
        list(held[key]['projection']),
        list(held[key]['projection']),
        {'python-core': {} if missing else {key: line}},
        {bare: ({'status': 'acceptance'}, '')},
        [],
        set(),
        {},
        None,
        database=database,
    )
    assert held == before
    return report


@pytest.mark.parametrize('database', ['duckdb', 'sqlite', 'postgres'])
def test_declared_database_checks_its_unchanged_baseline(
    ratchet: Any, database: str
) -> None:
    assert _baseline_report(ratchet, database).problems == []
    problems = _baseline_report(ratchet, database, missing=True).problems
    assert any('baselined, and not observed' in problem for problem in problems)
    assert any(
        'proposed baseline is not the observation' in problem
        for problem in problems
    )


def test_omitted_scope_keeps_all_database_requirements(ratchet: Any) -> None:
    problems = _baseline_report(ratchet, None).problems
    for database in ('sqlite', 'postgres'):
        assert any(
            f'@{database}:' in problem and 'not observed' in problem
            for problem in problems
        )


@pytest.mark.parametrize('database', ['duckdb', 'sqlite', 'postgres'])
def test_scope_preserves_baseline_equality_and_acceptance_requirements(
    ratchet: Any, database: str
) -> None:
    changed = _baseline_report(ratchet, database, changed=True).problems
    assert any('proposed baseline is not the observation' in p for p in changed)
    absent = _baseline_report(ratchet, database, missing_baseline=True).problems
    assert any('acceptance case with no proposed baseline' in p for p in absent)


def _observation() -> dict[str, Any]:
    path = next((ROOT / 'contract/reference').glob('*.jsonl'))
    line = json.loads(path.read_text().splitlines()[0])
    line['identity']['runner'] = 'python-core'
    return line


@pytest.mark.parametrize(
    'fault',
    ['foreign', 'cell', 'runner', 'empty', 'mixed-cells', 'malformed-key'],
)
def test_scope_refuses_foreign_or_invalid_evidence(
    ratchet: Any, fault: str
) -> None:
    line = _observation()
    observed = {'python-core': {line['case']: line}}
    control = ratchet.Report()
    ratchet.check_database_scope(control, observed, 'duckdb')
    assert control.problems == []
    if fault == 'foreign':
        line['case'] += '@sqlite'
        observed = {'python-core': {line['case']: line}}
    elif fault == 'cell':
        line['identity']['cell'] = 'duckdb'
    elif fault == 'runner':
        line['identity']['runner'] = 'rust-host'
    elif fault == 'mixed-cells':
        extra = copy.deepcopy(line)
        extra['case'] = 'another.case'
        extra['identity']['cell'] = 'floor-pandas2.2'
        observed['python-core'][extra['case']] = extra
    elif fault == 'malformed-key':
        line['case'] += '@duckdb'
        observed = {'python-core': {line['case']: line}}
    else:
        observed = {}
    report = ratchet.Report()
    ratchet.check_database_scope(report, observed, 'duckdb')
    assert report.problems


def test_scope_without_observation_files_is_refused(
    ratchet: Any, tmp_path: Path
) -> None:
    report = ratchet.check(tmp_path, 'unused', [], database='duckdb')
    assert report.problems == ['scope: duckdb: no observations']


@pytest.mark.parametrize('mode', ['full', 'locked'])
def test_partial_scope_cannot_claim_aggregate_mode(
    ratchet: Any, tmp_path: Path, mode: str
) -> None:
    report = ratchet.check(
        tmp_path, 'unused', [], database='duckdb', **{mode: True}
    )
    assert any('scope' in problem for problem in report.problems)
    done = subprocess.run(
        [
            'bash',
            str(ROOT / 'scripts/rust_core/gate.sh'),
            '--base',
            'HEAD',
            '--database',
            'duckdb',
            *(['--full'] if mode == 'full' else ['--locked', '4a']),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 2
    assert 'scope' in done.stderr


@pytest.mark.parametrize('database', ['duckdb', 'sqlite', 'postgres'])
def test_scoped_rust_runner_uses_its_actual_cli(
    tmp_path: Path, database: str
) -> None:
    # Exercise the gate's actual runner function with Cargo recorded rather
    # than compiled. DuckDB is the Rust CLI default; its explicit database
    # parser accepts SQLite and memory, while Postgres requires a recording.
    source = (ROOT / 'scripts/rust_core/gate.sh').read_text()
    observe = 'observe() {' + source.split('observe() {', 1)[1].split(
        '\nrefuse() {', 1
    )[0]
    (tmp_path / 'rust').mkdir()
    arguments = tmp_path / 'cargo-arguments.jsonl'
    done = subprocess.run(
        ['bash', '-uc',
         'candidate=$PWD; differs=0; runs=rust-host; made=$PWD; '
         'database_args=(--database "$database"); '
         'cargo() { printf "%s\\n" "$@" > "$recorded"; }; '
         + observe + '\nobserve'],
        cwd=tmp_path,
        env={**os.environ, 'database': database, 'recorded': str(arguments)},
        capture_output=True, text=True, check=False,
    )
    if database == 'postgres':
        assert done.returncode != 0
        assert 'Postgres' in done.stdout + done.stderr
        assert not arguments.exists()
    else:
        assert done.returncode == 0, done.stdout + done.stderr
        argv = arguments.read_text().splitlines()
        assert argv[:7] == ['run', '-q', '-p', 'sdm-testkit', '--bin', 'sdm-contract-run', '--locked']
        if database == 'duckdb':
            assert '--database' not in argv
        else:
            position = argv.index('--database')
            assert argv[position + 1] == 'sqlite'


def _captured() -> tuple[dict[str, Any], dict[str, Any]]:
    case = json.loads(
        (ROOT / 'contract/cases/s2/java.databricks-sql.types.json').read_text()
    )
    capture = json.loads(
        (
            ROOT
            / 'contract/captures/databricks-sql/java.databricks-sql.types/capture.json'
        ).read_text()
    )
    return case, capture['reference']


def test_actual_databricks_capture_credits_omitted_inputs(
    coverage: Any,
) -> None:
    case, line = _captured()
    cells = coverage.observed(case, line)
    assert {'binary/feature', 'list/feature', 'int64/primary-key'} <= cells
    assert 'binary/primary-key' not in cells
    assert 'list/target' not in cells
    assert coverage._undeclared(line) == {'users', 'orders'}


@pytest.mark.parametrize(
    'fault',
    [
        'absent',
        'failed',
        'nonempty',
        'boolean-count',
        'schema',
        'types',
        'results',
        'method',
        'parameter-type',
        'parameter-expression',
        'parameter-count',
        'parameter-empty',
        'duplicate',
        'unrelated-sql',
        'missing-error',
        'digest',
        'unencodable',
        'multiple-results',
    ],
)
def test_constraint_credit_requires_successful_unambiguous_empty_lookup(
    coverage: Any, fault: str
) -> None:
    case, line = _captured()
    statements = line['driver']['capture']['statements']
    statement = next(
        s
        for s in statements
        if 'table_constraints tc' in s['sql']
        and s['parameters'][1]['value'] == "'users'"
    )
    if fault == 'absent':
        statements.remove(statement)
    elif fault == 'failed':
        statement['error'] = 'DatabaseError'
    elif fault == 'nonempty':
        statement['results'][0]['rows'] = 1
    elif fault == 'boolean-count':
        statement['results'][0]['rows'] = False
    elif fault == 'schema':
        statement['results'][0]['values']['columns'] = ['column_name']
    elif fault == 'types':
        statement['results'][0]['values']['types'] = ['integer'] * 3
    elif fault == 'results':
        statement['results'] = []
    elif fault == 'method':
        statement['results'][0]['method'] = 'fetchone'
    elif fault == 'parameter-type':
        statement['parameters'][1]['type'] = 'builtins.bytes'
    elif fault == 'parameter-expression':
        statement['parameters'][1]['value'] = "'us' + 'ers'"
    elif fault == 'parameter-count':
        statement['parameters'].append(
            copy.deepcopy(statement['parameters'][1])
        )
    elif fault == 'parameter-empty':
        statement['parameters'][1]['value'] = "''"
    elif fault == 'duplicate':
        statements.append(copy.deepcopy(statement))
    elif fault == 'missing-error':
        statement.pop('error')
    elif fault == 'digest':
        statement['results'][0]['digest'] = '0' * 64
    elif fault == 'unencodable':
        statement['results'][0]['values']['unencodable'] = ['constraint_name']
    elif fault == 'multiple-results':
        statement['results'].append(copy.deepcopy(statement['results'][0]))
    else:
        statement['sql'] += ' OR 1 = 1'
    assert not {'binary/feature', 'list/feature'} & coverage.observed(
        case, line
    )
