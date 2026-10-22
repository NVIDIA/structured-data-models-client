# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T3.0 fix round 1: one test per Codex round-1 finding the implementation
fixed (the stack's push base is in tests/test_contract_ratchet.py, the guard's
quoting and claims in tests/test_t30_process.py's command lists).

- handoff.py's previous-round run reads the previous tree's sources, not
  the installed fixed build (finding 2);
- a changed ``#[cfg(test)]`` block counts as a test, spliced into the
  previous source (finding 3);
- affected.py selects the package and binding suites (finding 4);
- a case on Postgres alone gets no stamp without a server (finding 5);
- the timing guard's history skips cancelled jobs (finding 8);
- the dry run reports a tool that failed naming no kind as incomplete, and
  judges with the checkpoint's tools (findings 9 and 10).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts/rust_core'
GIT_ENV = {
    'GIT_CONFIG_GLOBAL': os.devnull,
    'GIT_CONFIG_NOSYSTEM': '1',
    'GIT_AUTHOR_NAME': 'Test Person',
    'GIT_AUTHOR_EMAIL': 'test@example.com',
    'GIT_COMMITTER_NAME': 'Test Person',
    'GIT_COMMITTER_EMAIL': 'test@example.com',
}


def _module(name: str, path: Path) -> ModuleType:
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


handoff = _module('t30_fix1_handoff', SCRIPTS / 'handoff.py')
affected = _module('t30_fix1_affected', SCRIPTS / 'affected.py')
ci_timing = _module(
    't30_fix1_ci_timing', SCRIPTS / 'orchestration/ci_timing.py'
)


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ['git', '-C', str(repo), *args],
        env={**os.environ, **GIT_ENV},
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def _commit(repo: Path, files: dict[str, str], subject: str) -> str:
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text, encoding='utf-8')
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-q', '-m', subject)
    return _git(repo, 'rev-parse', 'HEAD')


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    where = tmp_path / 'repo'
    where.mkdir()
    _git(where, 'init', '-q', '-b', 'rust-core-phase-3-t3.99')
    return where


# --- finding 2: the previous round's sources ------------------------------------


def test_the_previous_round_runs_on_its_own_sources(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = 'packages/demo/src/demo_answer/__init__.py'
    test = (
        'from demo_answer import ANSWER\n\n\n'
        'def test_the_fixed_answer() -> None:\n'
        '    assert ANSWER == 2\n'
    )
    base = _commit(repo, {'README.md': 'x\n'}, 'base')
    previous = _commit(repo, {source: 'ANSWER = 1\n'}, 'T3.99: the answer')
    _commit(
        repo,
        {source: 'ANSWER = 2\n', 'tests/test_demo_answer.py': test},
        'T3.99 fix 1: the answer is 2',
    )
    # The fixed build is what the interpreter has installed.
    monkeypatch.setenv('PYTHONPATH', str(repo / 'packages/demo/src'))
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    step = handoff.step_fails_before(
        repo, base, previous, sys.executable, scratch
    )
    assert step.ok, step.problems
    assert any('fails on' in note for note in step.notes), step.notes


# --- finding 3: inline Rust tests ------------------------------------------------

RUST_OLD = 'pub fn lag() -> u32 {\n    1\n}\n'
RUST_NEW = (
    'pub fn lag() -> u32 {\n    2\n}\n\n'
    '#[cfg(test)]\nmod tests {\n    use super::*;\n\n'
    '    #[test]\n    fn two() {\n        assert_eq!(lag(), 2, "{}", "}");\n'
    '    }\n}\n'
)


def test_a_changed_cfg_test_block_is_spliced_into_the_previous_source(
    repo: Path,
) -> None:
    path = 'rust/crates/demo/src/lag.rs'
    previous = _commit(repo, {path: RUST_OLD}, 'T3.99: lag')
    _commit(repo, {path: RUST_NEW}, 'T3.99 fix 1: lag is 2')
    spliced = handoff.inline_tests(repo, previous, path)
    assert spliced is not None
    # The previous production code under the new test block.
    assert spliced.startswith(RUST_OLD)
    assert '#[cfg(test)]' in spliced and 'assert_eq!(lag(), 2' in spliced
    block = handoff.test_block(RUST_NEW)
    assert block is not None and RUST_NEW[block[0] : block[1]].endswith('}\n}')
    assert handoff.is_test(path) is False
    # An unchanged test block is no test change.
    assert (
        handoff.inline_tests(repo, _git(repo, 'rev-parse', 'HEAD'), path)
        is None
    )


# --- finding 4: the product suites -----------------------------------------------


@pytest.mark.parametrize(
    ('path', 'suite'),
    [
        (
            'packages/structured-data-models-client/src/sdm_client/query.py',
            'packages/structured-data-models-client/tests',
        ),
        (
            'packages/structured-data-models-pql/src/predictive_query/__init__.py',
            'packages/structured-data-models-pql/tests',
        ),
        ('rust/bindings/sdm-py/src/lib.rs', 'rust/bindings/sdm-py/tests'),
        ('rust/crates/sdm-core/src/lib.rs', '-p sdm-core'),
    ],
)
def test_a_product_change_selects_its_package_suites(
    path: str, suite: str
) -> None:
    printed = [' '.join(s) for s in affected.suites([path])]
    assert any(suite in line for line in printed), printed


# --- finding 5: a Postgres-only case ---------------------------------------------


def test_a_postgres_only_case_gets_no_stamp_without_a_server(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('SDM_POSTGRES_TEST_DSN', raising=False)
    case = {
        'id': 's9.only.postgres',
        'databases': ['postgres'],
        'exclusion': 'KO9-1',
    }
    base = _commit(
        repo,
        {'contract/reference/r.jsonl': '{"case": "other"}\n'},
        'base',
    )
    _commit(
        repo,
        {'contract/cases/s9/s9.only.postgres.json': json.dumps(case)},
        'T3.99: a Postgres case',
    )
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    step = handoff.step_reference(
        repo,
        ['contract/cases/s9/s9.only.postgres.json'],
        [],
        sys.executable,
        base,
        scratch,
    )
    assert not step.ok
    assert any('Postgres alone' in p for p in step.problems), step.problems


# --- finding 8: cancelled jobs in the timing history ------------------------------


def test_the_timing_history_skips_cancelled_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = ci_timing.Limit('build', ci_timing.re.compile('build'), 120)

    def job(minutes: float, conclusion: str) -> dict[str, Any]:
        return {
            'name': 'build',
            'conclusion': conclusion,
            'started_at': '2026-10-22T00:00:00Z',
            'completed_at': (
                f'2026-10-22T{int(minutes) // 60:02d}:{int(minutes) % 60:02d}:00Z'
            ),
            'html_url': '',
        }

    history = [
        (1, 100, 'success'),
        (2, 5, 'cancelled'),
        (3, 5, 'cancelled'),
        (4, 100, 'success'),
        (5, 50, 'success'),
    ]
    jobs = {n: [job(m, c)] for n, m, c in history}
    runs = [
        {'id': n, 'head_branch': 'rust-core-phase-3-lane-a'}
        for n, _, _ in history
    ]

    def api(path: str, tries: int = 4) -> Any:
        if path.endswith('/runs/1'):
            return runs[0]
        return {'workflow_runs': runs[1:] if 'page=1' in path else []}

    monkeypatch.setattr(ci_timing, 'api', api)
    monkeypatch.setattr(ci_timing, 'jobs_of', lambda n: jobs[n])
    found = ci_timing.from_api(1, 40, [limit])
    # The two completed runs past the cancellations are read.
    assert [r['run'] for r in found] == [1, 2, 3, 4, 5]


# --- findings 9 and 10: the dry run ---------------------------------------------


def test_the_dry_run_counts_no_failed_tool_as_a_judgement() -> None:
    dryrun = _module('t30_fix1_dryrun', SCRIPTS / 'gate_dryrun.py')
    text = (SCRIPTS / 'gate_dryrun.py').read_text(encoding='utf-8')
    # The tools are the checkpoint's worktree's, not HEAD's.
    assert "judge / 'scripts/rust_core/coverage.py'" in text
    assert "judge / 'contract/runners/run_python.py'" in text
    kinds = [dryrun.Kind('k', {'id': 's9.k'})]
    loose = dryrun.attribute(
        'coverage', 'coverage: FAIL\nerror: unreadable manifest\n', kinds, True
    )
    # A failure naming no kind is unattributed, so the step is incomplete.
    assert loose and not kinds[0].gaps
