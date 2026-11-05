# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public schemas preserve Rust descriptions, including edit semantics."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
CASES = ROOT / 'contract/cases/s1'


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
    import warnings

    import duckdb
    from sdm_client import Schema
    from sdm_client._rust import schema as bridge
    from sdm_connectors import DuckDBSource

    runner = _runner()
    connection = duckdb.connect()
    for statement in runner.load_bases()['shop']:
        connection.execute(statement)
    # The description the core answered, as the schema was built from it:
    # the schema keeps none (T3.16), and the runner is handed the one the
    # run answered.
    answered: list[Any] = []
    build = bridge.build

    def kept(cls: Any, source: Any, described: Any) -> Any:
        answered.append(json.loads(json.dumps(described)))
        return build(cls, source, described)

    bridge.build = kept  # type: ignore[assignment]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(DuckDBSource(connection))
    finally:
        bridge.build = build  # type: ignore[assignment]
    [described] = answered
    assert runner.presented(schema, described) == []

    def built(change: Any) -> list[str]:
        copied = json.loads(json.dumps(described))
        change(copied)
        made = bridge.build(Schema, schema.source, copied)
        found: list[str] = runner.presented(made, copied)
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


def test_runner_rejects_a_changed_schema_description() -> None:
    import pytest

    runner = _runner()
    bases = runner.load_bases()
    [(case, _)] = runner.load_cases([CASES / 'base.regression.json'])
    case['until'] = 'from_source'
    assert runner._call(case, bases)[0]['result'] == 'success'
    honest = runner.presented
    try:
        runner.presented = lambda schema, described=None: ['seeded']
        with pytest.raises(runner.RunnerFailure, match='departs from it'):
            runner._call(case, bases)
    finally:
        runner.presented = honest


def test_recording_restores_native_boundary() -> None:
    from sdm_client._rust import run

    runner = _runner()
    path = ROOT / 'contract/cases/s0/s0.accept.sum-window.json'
    data = path.read_bytes()
    original = run.native
    observations = runner.run(
        'python-core', [(json.loads(data), runner.digest(data))]
    )
    assert observations[0]['outcome']['result'] == 'success'
    assert run.native is original


def test_recording_restores_native_boundary_after_failure(
    monkeypatch: Any,
) -> None:
    import pytest
    from sdm_client._rust import run

    runner = _runner()
    path = ROOT / 'contract/cases/s0/s0.accept.sum-window.json'
    data = path.read_bytes()
    original = run.native

    def fail() -> None:
        raise RuntimeError('seeded environment failure')

    monkeypatch.setattr(runner, '_versions', fail)
    with pytest.raises(RuntimeError, match='seeded environment failure'):
        runner.run('python-core', [(json.loads(data), runner.digest(data))])
    assert run.native is original
