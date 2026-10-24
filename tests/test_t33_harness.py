# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T3.3's acceptance tests for the carried CI rows (T3.3's brief, exit 7:
P2-T2.14c-1, P2-T2.1a-3, P2-T2.16a-1, P2-T2.11-4, C28, OF-7's phase
predicate, M5, C16), written by the package's test author before the
implementation. The stand-in's tests (exits 3, 5 and 6) went with the
stand-in (decision B-lite).

Each test listed in ``contract/tests/t33-pending.json``'s ``pending`` (as
``py:<function name>``) failed when written and is held to failing
(``xfail``, strict) until the implementer takes it out.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / '.github/workflows/ci.yml'
PENDING_FILE = ROOT / 'contract/tests/t33-pending.json'


def _pending_names(key: str) -> set[str]:
    return set(json.loads(PENDING_FILE.read_text(encoding='utf-8'))[key])


def pending(test: Any) -> Any:
    r"""Holds ``test`` to failing while ``t33-pending.json`` lists it."""
    if f'py:{test.__name__}' in _pending_names('pending'):
        return pytest.mark.xfail(
            strict=True,
            reason='T3.3: failing until implemented; once it passes, take '
            f'py:{test.__name__} out of contract/tests/t33-pending.json',
        )(test)
    return test


# --- the workflow --------------------------------------------------------------


def _workflow() -> dict[str, Any]:
    return yaml.safe_load(CI.read_text(encoding='utf-8'))


# --- exit 7: the carried CI rows ---------------------------------------------


def _runs(job: dict[str, Any]) -> list[str]:
    return [str(step.get('run', '')) for step in job.get('steps', [])]


@pending
def test_t33_the_numeric_spelling_cases_run_on_ci_s_postgres_service() -> None:
    r"""P2-T2.14c-1: ``test_t214c_numeric_spelling.py`` runs in a job with
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
        'test_t214c_numeric_spelling.py' in run
        for job in with_service.values()
        for run in _runs(job)
    )


def test_a_step_scoped_postgres_requirement_cannot_be_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = {
        'env': {'SDM_REQUIRE_POSTGRES': '1'},
        'run': 'python -m pytest contract/tests/test_t214c_numeric_spelling.py',
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
    test_t33_the_numeric_spelling_cases_run_on_ci_s_postgres_service()
    step['env'].clear()
    with pytest.raises(AssertionError, match='jdbc'):
        test_t33_the_numeric_spelling_cases_run_on_ci_s_postgres_service()


@pending
def test_t33_pytest_timeout_is_pinned_wherever_it_is_installed() -> None:
    r"""P2-T2.1a-3: ``pytest-timeout`` is pinned in every install."""
    found = re.findall(r'pytest-timeout\S*', CI.read_text(encoding='utf-8'))
    assert found
    unpinned = [
        each for each in found if not re.match(r'pytest-timeout==', each)
    ]
    assert not unpinned, unpinned


@pending
def test_t33_no_step_marks_the_test_kit_s_sources_newer() -> None:
    r"""P2-T2.16a-1: no step ``touch``es ``sdm-testkit``'s sources, which
    forced its rebuild on every run."""
    for name, job in _workflow()['jobs'].items():
        for run in _runs(job):
            for line in run.splitlines():
                assert not (
                    re.search(r'\btouch\b', line) and 'sdm-testkit' in line
                ), (name, line)


def _job_block(job: str) -> str:
    r"""The workflow's lines of ``job``, its comments included."""
    lines = CI.read_text(encoding='utf-8').splitlines()
    start = lines.index(f'  {job}:')
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if re.match(r'  [A-Za-z0-9_-]+:\s*$', lines[i])
        ),
        len(lines),
    )
    return '\n'.join(lines[start:end])


@pending
def test_t33_the_engine_job_s_comments_say_what_happened() -> None:
    r"""P2-T2.11-4: the engine job's comment says two runs reached its limit,
    not three, and the comment on ``TIMEOUT_CEILINGS`` names the engine
    job it holds."""
    assert 'three runs' not in _job_block('engine')
    text = (ROOT / 'tests/test_ci_workflow.py').read_text(encoding='utf-8')
    before = text[: text.index('TIMEOUT_CEILINGS = {')].splitlines()
    comment = []
    for line in reversed(before):
        if not line.startswith('#'):
            if comment:
                break
            continue
        comment.append(line)
    assert 'engine' in ' '.join(comment), comment


@pending
def test_t33_the_skip_step_reads_only_push_runs() -> None:
    r"""C28: the landing job's skip step selects runs whose ``.event`` is
    ``push`` in its ``jq``, as ``land.py`` does."""
    steps = _workflow()['jobs']['landing']['steps']
    landing = next(s for s in steps if s.get('id') == 'landing')
    selects = re.findall(r"--jq '([^']*)", landing['run'])
    assert selects
    assert all(re.search(r'\.event\s*==\s*"push"', each) for each in selects)


def _evaluate(expression: str, event: str, ref: str) -> bool:
    r"""A GitHub Actions ``if:`` expression over ``github.event_name`` and
    ``github.ref_name``, with ``startsWith``, ``endsWith`` and ``contains``
    (each ignoring case, as Actions compares)."""

    def starts(text: str, prefix: str) -> bool:
        return text.lower().startswith(prefix.lower())

    def ends(text: str, suffix: str) -> bool:
        return text.lower().endswith(suffix.lower())

    def holds(where: Any, what: str) -> bool:
        if isinstance(where, list):
            return any(str(each).lower() == what.lower() for each in where)
        return what.lower() in str(where).lower()

    text = expression.strip()
    if text.startswith('${{') and text.endswith('}}'):
        text = text[3:-2]
    text = text.replace('github.event_name', '_event').replace(
        'github.ref_name', '_ref'
    )
    text = text.replace('&&', ' and ').replace('||', ' or ')
    text = re.sub(r'!(?!=)', ' not ', text)
    names = {
        '_event': event,
        '_ref': ref,
        'startsWith': starts,
        'endsWith': ends,
        'contains': holds,
        'fromJSON': json.loads,
    }
    return bool(eval(text, {'__builtins__': {}}, names))  # noqa: S307


#: Branch names that are no phase branch, though they look like one, and
#: those that are.
NOT_PHASE = (
    'rust-core-phase-3a1',
    'rust-core-phase-03',
    'rust-core-phase-3.1',
    'rust-core-phase-3-lane-h',
    'rust-core-phase-3-topic-T3.3-tests',
)
PHASE = ('rust-core-phase-0', 'rust-core-phase-3', 'rust-core-phase-12')


@pending
def test_t33_the_phase_predicate_lets_only_phase_branches_through() -> None:
    r"""OF-7: the step a push to the phase branch runs is predicated on
    ``rust-core-phase-N`` alone, N a number with no leading zero: a name
    with a letter before its last digit, a leading zero or a dot is no
    phase branch, nor is a lane or topic under one."""
    steps = _workflow()['jobs']['landing']['steps']
    step = next(
        s
        for s in steps
        if s.get('name') == 'A push to the phase branch lands one package'
    )
    for name in PHASE:
        assert _evaluate(step['if'], 'push', name), name
    for name in NOT_PHASE:
        assert not _evaluate(step['if'], 'push', name), name
    assert not _evaluate(step['if'], 'pull_request', PHASE[1])


@pending
def test_t33_ci_sets_a_process_time_zone() -> None:
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


@pending
def test_t33_no_vector_sets_its_zone_per_connection() -> None:
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
