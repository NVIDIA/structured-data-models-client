# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI provider configuration, dependency pins and timezone invariants."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / '.github/workflows/ci.yml'
# --- the workflow --------------------------------------------------------------


def _workflow() -> dict[str, Any]:
    return yaml.safe_load(CI.read_text(encoding='utf-8'))


# --- exit 7: the carried CI rows ---------------------------------------------


def _runs(job: dict[str, Any]) -> list[str]:
    return [str(step.get('run', '')) for step in job.get('steps', [])]


def test_the_numeric_spelling_cases_run_on_ci_s_postgres_service() -> None:
    r"""P2--1: ``test_postgres_source_contracts.py`` runs in a job with
    the Postgres service, and every job with the service requires it
    (``SDM_REQUIRE_POSTGRES``), so a missing server fails, never skips."""
    workflow = _workflow()
    top = workflow.get('env') or {}
    with_service = {
        name: job
        for name, job in workflow['jobs'].items()
        if 'postgres' in json.dumps(job.get('services') or {})
    }
    assert with_service
    for name, job in with_service.items():
        env = {**top, **(job.get('env') or {})}
        if env.get('SDM_REQUIRE_POSTGRES'):
            continue
        # The JDBC matrix needs Postgres only in its service-backed
        # parity step; a step's environment overrides the job's.
        tests = [
            step
            for step in job.get('steps', [])
            if '-m pytest' in str(step.get('run', ''))
        ]
        assert tests, name
        for step in tests:
            assert {
                **env,
                **(step.get('env') or {}),
            }.get('SDM_REQUIRE_POSTGRES'), name
    assert any(
        'test_postgres_source_contracts.py' in run
        for job in with_service.values()
        for run in _runs(job)
    )


def test_a_step_scoped_postgres_requirement_cannot_be_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = {
        'env': {'SDM_REQUIRE_POSTGRES': '1'},
        'run': 'python -m pytest contract/tests/test_postgres_source_contracts.py',
    }
    workflow = {
        'jobs': {
            'jdbc': {
                'services': {'postgres': {'image': 'postgres'}},
                'steps': [step],
            }
        }
    }
    monkeypatch.setitem(globals(), '_workflow', lambda: workflow)
    test_the_numeric_spelling_cases_run_on_ci_s_postgres_service()
    step['env'].clear()
    with pytest.raises(AssertionError, match='jdbc'):
        test_the_numeric_spelling_cases_run_on_ci_s_postgres_service()


def test_pytest_timeout_is_pinned_wherever_it_is_installed() -> None:
    r"""P2--3: ``pytest-timeout`` is pinned in every install."""
    found = re.findall(r'pytest-timeout\S*', CI.read_text(encoding='utf-8'))
    assert found
    unpinned = [
        each for each in found if not re.match(r'pytest-timeout==', each)
    ]
    assert not unpinned, unpinned


def test_no_step_marks_the_test_kit_s_sources_newer() -> None:
    r"""P2--1: no step ``touch``es ``sdm-testkit``'s sources, which
    forced its rebuild on every run."""
    for name, job in _workflow()['jobs'].items():
        for run in _runs(job):
            for line in run.splitlines():
                assert not (
                    re.search(r'\btouch\b', line) and 'sdm-testkit' in line
                ), (name, line)


def test_ci_sets_a_process_time_zone() -> None:
    r"""M5: CI sets the process ``TZ`` for every job that runs the tests or
    the runners, so a test that sets only DuckDB's session zone cannot pass
    on the runner's default alone."""
    workflow = _workflow()
    top = workflow.get('env') or {}
    for name, job in workflow['jobs'].items():
        if not any(
            'pytest' in run or 'run_python.py' in run for run in _runs(job)
        ):
            continue
        env = {**top, **(job.get('env') or {})}
        assert env.get('TZ'), name


_PER_CONNECTION_ZONE = re.compile(
    r'^\s*SET\s+(?:SESSION\s+|LOCAL\s+)?TimeZone\b', re.IGNORECASE
)
_GLOBAL_ZONE = re.compile(r'^\s*SET\s+GLOBAL\s+TimeZone\b', re.IGNORECASE)


def _setups(node: Any, where: str) -> list[tuple[str, list[str]]]:
    r"""Each list of statements under a ``setup`` or ``witness`` key, at any
    depth, with the id of the nearest object that has one."""
    found: list[tuple[str, list[str]]] = []
    if isinstance(node, dict):
        where = str(node.get('id', where))
        for key, value in node.items():
            if key in ('setup', 'witness'):
                statements = [value] if isinstance(value, str) else value
                if isinstance(statements, list):
                    found.append(
                        (where, [s for s in statements if isinstance(s, str)])
                    )
            found += _setups(value, where)
    elif isinstance(node, list):
        for each in node:
            found += _setups(each, where)
    return found


def test_no_vector_sets_its_zone_per_connection() -> None:
    r"""C16 (rest): ``type-set-vectors.json``'s witnesses
    ``timestamp.us.utc`` and ``timestamp.us.named-zone`` are re-recorded
    with ``SET GLOBAL TimeZone``, so no vector file's setup sets the zone
    per connection alone, which a cursor does not take (a session zone
    beside a global one, in the same setup, is the zone checks' own
    seed, as ``tests/test_contract_cases.py`` allows)."""
    per_connection = []
    for path in sorted((ROOT / 'contract').glob('*-vectors.json')):
        document = json.loads(path.read_text(encoding='utf-8'))
        for where, setup in _setups(document, '?'):
            if any(_GLOBAL_ZONE.search(each) for each in setup):
                continue
            per_connection += [
                (path.name, where, each)
                for each in setup
                if _PER_CONNECTION_ZONE.search(each)
            ]
    assert not per_connection, per_connection
