# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T3.3's acceptance tests for the warehouse harness around the stand-in
(T3.3's brief, exits 3, 5, 6 and 7), written by the package's test author
before the implementation:

- exit 3: ``scripts/rust_core/l0_deps.py`` passes with the stand-in, its
  module and its ``sdm-standin`` binary, in the workspace;
- exit 5: a difference on one warehouse leg only stops ``handoff.py``'s
  stamp (the reference step runs each touched case's warehouse legs on the
  stand-in);
- exit 6: CI's leg jobs, ``contract-python (ref-pandas3, <leg>)`` and
  ``cross-host (<leg>)`` for each warehouse leg, run the cases on that leg
  and judge them;
- exit 7: the carried CI rows the brief names (P2-T2.14c-1, P2-T2.1a-3,
  P2-T2.16a-1, P2-T2.11-4, C28, OF-7's phase predicate, M5, C16).

Each test listed in ``contract/tests/t33-pending.json``'s ``pending`` (as
``py:<function name>``) failed when written and is held to failing
(``xfail``, strict) until the implementer takes it out.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / '.github/workflows/ci.yml'
PENDING_FILE = ROOT / 'contract/tests/t33-pending.json'
#: The four warehouse legs (Phase 3 design, choice 1).
LEGS = ('databricks-sql', 'databricks-spark', 'snowflake', 'lakebase')


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


# --- exit 3: l0_deps.py with the stand-in in the workspace -----------------


@pending
def test_t33_l0_deps_passes_with_the_stand_in_in_the_workspace() -> None:
    r"""Exit 3 (choice 2, "Where it lives"): the stand-in is one module of
    ``sdm-testkit`` and its ``sdm-standin`` binary, and ``l0_deps.py``
    holds unchanged over them: nothing shipped reaches ``sdm-testkit``."""
    assert (ROOT / 'rust/crates/sdm-testkit/src/standin.rs').is_file()
    metadata = subprocess.run(
        ['cargo', 'metadata', '--no-deps', '--format-version', '1'],
        cwd=ROOT / 'rust',
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert metadata.returncode == 0, metadata.stderr[-2000:]
    testkit = next(
        package
        for package in json.loads(metadata.stdout)['packages']
        if package['name'] == 'sdm-testkit'
    )
    assert any(
        target['name'] == 'sdm-standin' and 'bin' in target['kind']
        for target in testkit['targets']
    ), [target['name'] for target in testkit['targets']]
    done = subprocess.run(
        [sys.executable, 'scripts/rust_core/l0_deps.py'],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, (done.stdout + done.stderr)[-3000:]


# --- the workflow --------------------------------------------------------------


def _workflow() -> dict[str, Any]:
    return yaml.safe_load(CI.read_text(encoding='utf-8'))


def _matrix(job: dict[str, Any]) -> list[dict[str, Any]]:
    r"""Each combination of a job's static matrix; one empty one when it
    has none."""
    matrix = (job.get('strategy') or {}).get('matrix')
    if not isinstance(matrix, dict):
        return [{}]
    axes = {
        key: value
        for key, value in matrix.items()
        if key not in ('include', 'exclude') and isinstance(value, list)
    }
    combos = [
        dict(zip(axes, values, strict=True))
        for values in itertools.product(*axes.values())
    ] or ([{}] if not matrix.get('include') else [])
    return combos + [dict(each) for each in matrix.get('include', [])]


def _filled(value: Any, matrix: dict[str, Any]) -> Any:
    r"""``value`` with each ``${{ matrix.<key> }}`` filled in."""
    if isinstance(value, str):
        return re.sub(
            r'\$\{\{\s*matrix\.([A-Za-z0-9_-]+)\s*\}\}',
            lambda found: str(matrix.get(found.group(1), found.group(0))),
            value,
        )
    if isinstance(value, dict):
        return {key: _filled(item, matrix) for key, item in value.items()}
    if isinstance(value, list):
        return [_filled(item, matrix) for item in value]
    return value


def _jobs_named() -> dict[str, dict[str, Any]]:
    r"""Every job of the workflow by the name CI shows, each matrix
    combination its own job, its matrix values filled in."""
    named: dict[str, dict[str, Any]] = {}
    for key, job in _workflow()['jobs'].items():
        for combo in _matrix(job):
            filled = _filled(job, combo)
            named[str(filled.get('name', key))] = filled
    return named


# --- exit 6: the leg jobs ----------------------------------------------------


@pending
@pytest.mark.parametrize('leg', LEGS)
def test_t33_ci_s_leg_jobs_run_and_judge_each_warehouse_leg(leg: str) -> None:
    r"""Exit 6: ``contract-python (ref-pandas3, <leg>)`` runs python-core on
    the leg in the reference cell and judges it with the comparator, and
    ``cross-host (<leg>)`` judges rust-host on the same leg; Lakebase's
    stand-in is the local Postgres CI runs, required where it is named."""
    jobs = _jobs_named()
    python_job = jobs.get(f'contract-python (ref-pandas3, {leg})')
    host_job = jobs.get(f'cross-host ({leg})')
    assert python_job is not None, sorted(jobs)
    assert host_job is not None, sorted(jobs)
    for job in (python_job, host_job):
        assert isinstance(job.get('timeout-minutes'), int)
        text = json.dumps(job)
        assert leg in text
        assert 'compare.py' in text
    python_text = json.dumps(python_job)
    assert 'run_python.py' in python_text and 'python-core' in python_text
    assert (python_job.get('env') or {}).get('CELL') == 'ref-pandas3'
    assert 'rust-host' in json.dumps(host_job)
    if leg == 'lakebase':
        services = python_job.get('services') or {}
        assert any(
            str(service.get('image', '')).startswith(('postgres', '${{'))
            or 'postgres' in json.dumps(service)
            for service in services.values()
        ), services
        assert (python_job.get('env') or {}).get('SDM_REQUIRE_POSTGRES')


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
        assert env.get('SDM_REQUIRE_POSTGRES'), name
    assert any(
        'test_t214c_numeric_spelling.py' in run
        for job in with_service.values()
        for run in _runs(job)
    )


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


# --- exit 5: a warehouse-only difference stops the stamp ---------------------

PHASE_BRANCH = 'rust-core-phase-3'
PACKAGE = 'T3.99'
TOPIC = f'{PHASE_BRANCH}-t3.99'
HANDOFF = 'scripts/rust_core/handoff.py'
#: An S1 case, which runs on every warehouse leg by default, and the leg
#: its difference is seeded on: Lakebase, judged on the local Postgres.
SEEDED_CASE = 'base.regression'
SEEDED_LEG = 'lakebase'
GIT_ENV = {
    'GIT_CONFIG_GLOBAL': os.devnull,
    'GIT_CONFIG_NOSYSTEM': '1',
    'GIT_AUTHOR_NAME': 'Test Person',
    'GIT_AUTHOR_EMAIL': 'test@example.com',
    'GIT_COMMITTER_NAME': 'Test Person',
    'GIT_COMMITTER_EMAIL': 'test@example.com',
    'GIT_TERMINAL_PROMPT': '0',
}


def _git(where: Path, *args: str) -> str:
    done = subprocess.run(
        ['git', '-C', str(where), *args],
        env={**os.environ, **GIT_ENV},
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=300,
    )
    if done.returncode != 0:
        raise AssertionError(f'git {args}: {done.stderr.strip()}')
    return done.stdout.strip()


def _clone(tmp_path: Path) -> Path:
    r"""A clone of this repository, its HEAD and this worktree's changes
    under ``scripts/``, ``tests/``, ``contract/`` and ``.claude/`` as the
    phase head, and a topic branch on it holding one package commit."""
    origin, work = tmp_path / 'origin.git', tmp_path / 'work'
    _git(tmp_path, 'clone', '-q', '--bare', '--shared', str(ROOT), str(origin))
    _git(tmp_path, 'clone', '-q', '--shared', str(origin), str(work))
    _git(work, 'checkout', '-q', '--detach', _git(ROOT, 'rev-parse', 'HEAD'))
    changed = _git(ROOT, 'diff', '--name-only', 'HEAD').splitlines()
    added = _git(
        ROOT, 'ls-files', '--others', '--exclude-standard'
    ).splitlines()
    for name in {*changed, *added}:
        if not name.startswith(('scripts/', 'tests/', 'contract/', '.claude/')):
            continue
        source, target = ROOT / name, work / name
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        elif target.exists():
            target.unlink()
    if _git(work, 'status', '--porcelain'):
        _git(work, 'add', '-A')
        _git(work, 'commit', '-q', '-s', '-m', 'T3.3: the tools under test')
    head = _git(work, 'rev-parse', 'HEAD')
    _git(
        work, 'push', '-q', 'origin',
        f'{head}:refs/heads/{PHASE_BRANCH}', f'{head}:refs/heads/rust-core',
    )  # fmt: skip
    _git(work, 'fetch', '-q', 'origin')
    _git(work, 'checkout', '-q', '-b', TOPIC, head)
    probe = work / 'docs/t399-probe.md'
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text('A probe for the handoff test.\n', encoding='utf-8')
    _git(work, 'add', '-A')
    _git(work, 'commit', '-q', '-s', '-m', f'{PACKAGE}: a probe')
    return work


#: ``--python`` for the handoff: the core interpreter, which, when told
#: to, makes python-core's answer to the seeded case on the seeded leg
#: alone differ from the old client's (a float64 column read as float32).
SEEDING_PYTHON = r"""#!{python}
import json, os, subprocess, sys

args = sys.argv[1:]
done = subprocess.run([os.environ['T33_CORE_PYTHON'], *args])
key = os.environ['T33_SEED_CASE'] + '@' + os.environ['T33_SEED_LEG']
if (
    any(a.endswith('run_python.py') for a in args)
    and 'python-core' in args
    and '--out' in args
):
    path = args[args.index('--out') + 1]

    def differ(value):
        if isinstance(value, dict):
            return {{k: differ(v) for k, v in value.items()}}
        if isinstance(value, list):
            return [differ(v) for v in value]
        return 'float32' if value == 'float64' else value

    if os.path.exists(path):
        with open(path) as handle:
            lines = handle.read().splitlines()
        out = []
        for line in lines:
            seen = json.loads(line)
            if seen.get('case') == key:
                seen['public'] = differ(seen.get('public'))
                line = json.dumps(seen, sort_keys=True)
            out.append(line)
        with open(path, 'w') as handle:
            handle.write(''.join(line + '\n' for line in out))
sys.exit(done.returncode)
"""


@pending
@pytest.mark.tooling
def test_t33_a_difference_on_one_warehouse_leg_stops_the_stamp(
    tmp_path: Path,
) -> None:
    r"""Exit 5: the handoff's reference step runs each touched case's
    warehouse legs on the stand-in, from the committed capture, so a
    difference seeded on one warehouse leg alone (the DuckDB, SQLite and
    Postgres legs as the old client answered) stops the stamp, naming the
    case on that leg."""
    core = os.environ.get('SDM_TEST_CORE_PYTHON', '')
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN', '')
    if not core or not dsn:
        pytest.skip(
            'the reference step needs SDM_TEST_CORE_PYTHON (the candidate '
            "client's interpreter) and SDM_POSTGRES_TEST_DSN"
        )
    work = _clone(tmp_path)
    sha = _git(work, 'rev-parse', 'HEAD')
    seeding = tmp_path / 'seeding-python'
    seeding.write_text(SEEDING_PYTHON.format(python=sys.executable))
    seeding.chmod(0o755)
    env = {
        **{
            k: v
            for k, v in os.environ.items()
            if not k.startswith(('SDM_', 'GIT_'))
        },
        **GIT_ENV,
        'SDM_ORCH_STATE': str(tmp_path / 'state'),
        'SDM_PHASE_BRANCH': PHASE_BRANCH,
        'SDM_POSTGRES_TEST_DSN': dsn,
        'T33_CORE_PYTHON': core,
        'T33_SEED_CASE': SEEDED_CASE,
        'T33_SEED_LEG': SEEDED_LEG,
    }
    done = subprocess.run(
        [
            sys.executable, str(work / HANDOFF), 'run', '--package', PACKAGE,
            '--case', SEEDED_CASE, '--python', str(seeding),
        ],
        cwd=work, env=env, capture_output=True, text=True,
        stdin=subprocess.DEVNULL, timeout=3600,
    )  # fmt: skip
    said = done.stdout + done.stderr
    assert done.returncode == 1, said[-3000:]
    assert f'{SEEDED_CASE}@{SEEDED_LEG}' in said, said[-3000:]
    common = Path(_git(work, 'rev-parse', '--git-common-dir'))
    if not common.is_absolute():
        common = work / common
    assert not (common / 'sdm-handoff' / f'{sha}.json').exists()
