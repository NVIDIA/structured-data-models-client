# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``handoff.py``'s reference step reads each case's legs from the case
(T3.3): a warehouse-only S2 case never runs on DuckDB, SQLite or Postgres,
its warehouse legs run on python-core offline, and a leg with no capture
yet is named pending live, never run or passed."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/rust_core'))
_spec = importlib.util.spec_from_file_location(
    't33_handoff_legs', ROOT / 'scripts/rust_core/handoff.py'
)
assert _spec is not None and _spec.loader is not None
handoff = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(handoff)


def test_a_warehouse_only_case_runs_on_its_legs_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = json.loads(
        (ROOT / 'contract/cases/s1/base.regression.json').read_text()
    )
    case['id'] = 't33.handoff.s2-only'
    case['slice'] = 'S2'
    case['warehouses'] = ['databricks-sql', 'snowflake']
    case.pop('databases', None)
    path = tmp_path / 'case.json'
    path.write_text(json.dumps(case))
    ran: list[list[str]] = []

    def run(argv, cwd, env=None):  # noqa: ANN001, ARG001
        ran.append([str(a) for a in argv])
        return subprocess.CompletedProcess(argv, 0, '', '')

    monkeypatch.setattr(handoff, 'run', run)
    monkeypatch.setattr(handoff, 'case_index', lambda root: {case['id']: path})
    step = handoff.step_reference(
        ROOT, [], [case['id']], sys.executable, 'HEAD', tmp_path
    )
    runs = [
        argv for argv in ran if any(a.endswith('run_python.py') for a in argv)
    ]
    assert runs == [], runs
    assert step.problems == [], step.problems
    assert (
        't33.handoff.s2-only@databricks-sql: pending live (no capture yet)'
        in (step.notes)
    )
    assert 't33.handoff.s2-only@snowflake: pending live (no capture yet)' in (
        step.notes
    )
