# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""base.regression's schema on the core route against the reference (T1.11b).

Run where the contract tests run (CI's ``engine`` job until T1.18):

    python -m pytest contract/tests -q

On ``python-core``, ``Schema.from_source`` runs on the core: its operation
is ``describe_source``, and the public schema is the reference's, from the
committed reference observation. The driver capture shows the core's SQL
for the schema is the reference's: the same statements, in order, text for
text, each returning the same number of rows (decision H9 gates on the
count and the results; the text is the renderings' own claim). The case is
diagnostic until T1.29, so the comparator reports its differences and
gates none (``test_runners.py`` runs it over every case); this test holds
the schema and the statements.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
CASES = ROOT / 'contract/cases/s1'
REFERENCE = next((ROOT / 'contract/reference').glob('*.jsonl'))


def _reference() -> dict[str, Any]:
    for line in REFERENCE.read_text().splitlines():
        observation = json.loads(line)
        if observation['case'] == 'base.regression':
            return observation
    raise AssertionError('no reference observation of base.regression')


def _run(mode: str, out: Path) -> dict[str, Any]:
    run = subprocess.run(
        [
            sys.executable,
            '-I',
            str(RUNNER),
            '--mode',
            mode,
            '--out',
            str(out),
            # base.regression alone: base.adventureworks (T1.21b) has no
            # core route yet (contract/refusal-maps).
            str(CASES / 'base.regression.json'),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    [observation] = [json.loads(line) for line in out.read_text().splitlines()]
    return observation


def test_base_regression_s_schema_on_the_core_is_the_reference_s(
    tmp_path: Path,
) -> None:
    reference = _reference()
    core = _run('python-core', tmp_path / 'python-core.jsonl')
    old = _run('python-old', tmp_path / 'python-old.jsonl')
    # From T1.11c the call's materialize runs on the core too
    # (test_materialize_route.py).
    assert core['identity']['operations'] == [
        'describe_source',
        'materialize',
        'predict',
    ]
    assert core['core']['error'] is None
    assert core['core']['described'] is not None
    assert old['identity']['native_loaded'] is False
    # The inputs, step trace and description sdm-testkit's
    # tests/duckdb_provider.rs holds rust-host to: CI gates the two hosts'
    # equality here until T1.18's comparator does.
    pinned = json.loads(
        (ROOT / 'contract/pinned/base.regression.core.json').read_text()
    )
    for field in ('inputs', 'fetches', 'described'):
        assert core['core'][field] == pinned[field], field
    for observed in (core, old):
        assert observed['public']['schema'] == reference['public']['schema']
        assert observed['outcome'] == reference['outcome']


def test_the_core_s_sql_for_the_schema_is_the_reference_s(
    tmp_path: Path,
) -> None:
    reference = _reference()['driver']['capture']['statements']
    core = _run('python-core', tmp_path / 'python-core.jsonl')
    asked = core['core']['fetches']
    # Schema.from_source asks one statement per fetch, the first ten; what
    # follows is materialize's (test_materialize_route.py).
    asked = asked[:10]
    ran = core['driver']['capture']['statements'][: len(asked)]
    assert len(asked) == 10
    assert [s['sql'] for s in ran] == [s['sql'] for s in reference[:10]]
    assert [
        statement['text']
        for fetch in asked
        for statement in fetch['plan']['statements']
    ] == [s['sql'] for s in ran]
    assert [[r['rows'] for r in s['results']] for s in ran] == [
        [r['rows'] for r in s['results']] for s in reference[:10]
    ]
    # Each fixed-shape answer is checked against its shape (decision E36).
    shaped = [
        fetch
        for fetch in asked
        if fetch['plan']['expect']['expect'] == 'schema'
    ]
    assert [f['request']['request'] for f in shaped] == [
        'describe',
        'describe',
        'describe',
        'describe',
        'describe',
        'describe',
        'describe',
        'key_counts',
    ]


def _runner() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location('run_python', RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['run_python'] = module
    spec.loader.exec_module(module)
    return module


def test_the_schema_built_from_the_description_is_the_description() -> None:
    # Building the Python Schema re-runs its own key marking and link
    # checks (the live object's edit semantics, E55(b)); over the core's
    # description they change nothing, which every python-core run checks
    # (run_python.presented, held for base.regression, S0 and the
    # AdventureWorks probes). Seeded: a description the object's marking
    # would override, and links it would rewrite, are both named.
    import os
    import warnings

    import duckdb
    from sdm_client import Schema, _routing
    from sdm_client._rust import schema as bridge
    from sdm_connectors import DuckDBSource

    runner = _runner()
    connection = duckdb.connect()
    for statement in runner.load_bases()['shop']:
        connection.execute(statement)
    previous = os.environ.get(_routing.ENVIRONMENT)
    os.environ[_routing.ENVIRONMENT] = ','.join(_routing.FROM_SOURCE_DUCKDB)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(DuckDBSource(connection))
    finally:
        if previous is None:
            os.environ.pop(_routing.ENVIRONMENT, None)
        else:
            os.environ[_routing.ENVIRONMENT] = previous
    described = schema._core_description
    assert described is not None
    assert runner.presented(schema) == []

    def built(change: Any) -> list[str]:
        copied = json.loads(json.dumps(described))
        change(copied)
        made = bridge.build(Schema, schema.source, copied)
        made._core_description = copied
        found: list[str] = runner.presented(made)
        return found

    def unjoined_identifier(d: dict[str, Any]) -> None:
        [users] = [t for t in d['tables'] if t['name'] == 'users']
        [age] = [c for c in users['columns'] if c['name'] == 'age']
        age.update(stype='ID', stype_before_key='numerical')

    assert built(unjoined_identifier) == ['users.age: its meaning']

    def a_link_twice(d: dict[str, Any]) -> None:
        d['links'] = d['links'] * 2

    assert built(a_link_twice) == ['the links']
    assert built(lambda d: None) == []


def test_the_runners_hold_each_core_schema_to_its_description() -> None:
    # The check above is called where the runners build a schema (T1.21d
    # fix round 2): a run_python case and a refusal-map probe each fail
    # when it finds a departure, seeded by making it find one, and pass
    # without the seed.
    import copy
    import os

    import pytest

    runner = _runner()
    spec = importlib_module('refusal_map')
    bases = runner.load_bases()
    [(case, _)] = runner.load_cases([CASES / 'base.regression.json'])
    case = copy.deepcopy(case)
    case['until'] = 'from_source'
    probe = next(
        p['python-core']
        for p in spec.load(
            ROOT / 'contract/refusal-maps/base.adventureworks.json'
        )['probes']
        if p['id'] == 'no-text-dates'
    )
    previous = os.environ.get('SDM_CORE_ROUTES')
    os.environ['SDM_CORE_ROUTES'] = ','.join(case['routes'])
    honest = runner.presented
    try:
        outcome = runner._call(case, bases)[0]
        assert outcome['result'] == 'success', outcome
        seen, _ = spec.observe_python_core(runner, probe, bases)
        assert seen == spec.success('from_source'), seen
        runner.presented = lambda schema: ['seeded']
        with pytest.raises(runner.RunnerFailure, match='departs from it'):
            runner._call(case, bases)
        with pytest.raises(spec.MapFailure, match='departs from it'):
            spec.observe_python_core(runner, probe, bases)
    finally:
        runner.presented = honest
        if previous is None:
            os.environ.pop('SDM_CORE_ROUTES', None)
        else:
            os.environ['SDM_CORE_ROUTES'] = previous


def importlib_module(name: str) -> Any:
    import importlib.util

    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / f'contract/runners/{name}.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
