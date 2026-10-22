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

# Tests of the checking and orchestration tools, some copying the checkout:
# CI runs them in its tooling jobs (tooling_tests.py).
pytestmark = pytest.mark.tooling

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
    assert any('SDM_POSTGRES_TEST_DSN' in p for p in step.problems), (
        step.problems
    )


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


# --- Claude r1 P2-1: the judging list leaves product work alone -------------------

judging = _module('t30_fix1_judging', SCRIPTS / 'judging.py')
RULES = judging.patterns(
    (SCRIPTS / 'judging-paths.txt').read_text(encoding='utf-8')
)


def test_what_product_packages_change_is_no_judging_path() -> None:
    product_work = [
        'scripts/rust_core/bridge_freeze.json',
        'scripts/rust_core/bridge_changes.json',
        'scripts/rust_core/bridge_freeze.py',
        'scripts/rust_core/selftest.py',
        'scripts/rust_core/wheel_smoke.py',
        'contract/tools/forecast_record.py',
        'contract/runners/databases.py',
        'contract/t216-encoding-vectors.json',
        '.github/workflows/ci.yml',
    ]
    assert judging.judging(RULES, product_work) == []
    deciding = [
        'scripts/rust_core/gate.sh',
        'scripts/rust_core/ratchet.py',
        'scripts/rust_core/compare.py',
        'scripts/rust_core/coverage.py',
        'scripts/rust_core/land.py',
        'scripts/rust_core/lane.py',
        'contract/tools/refusal_census.py',
        'contract/case.schema.json',
        'contract/runners/run_python.py',
    ]
    assert judging.judging(RULES, deciding) == sorted(deciding)


def test_most_of_phase_2_s_landings_change_no_judging_path() -> None:
    def git(*args: str) -> str:
        return subprocess.run(
            ['git', '-C', str(ROOT), *args], capture_output=True, text=True
        ).stdout

    phase = 'refs/remotes/origin/rust-core-phase-2'
    if not git('rev-parse', '--verify', '--quiet', phase).strip():
        pytest.skip('no origin/rust-core-phase-2 in this clone')
    landings = []
    for line in git(
        'log', '--first-parent', '--format=%H%x00%s', phase
    ).splitlines():
        sha, _, subject = line.partition('\0')
        if subject.startswith(('Land T2', 'Land CI')):
            base = git(
                'log', '-1', '--format=%(trailers:key=Base-SHA,valueonly)', sha
            ).strip()
            if base:
                landings.append((base, sha))
    assert len(landings) >= 40, len(landings)
    touching = [
        sha
        for base, sha in landings
        if judging.judging(
            RULES,
            [p for p in git('diff', '--name-only', base, sha).split('\n') if p],
        )
    ]
    # Phase 2's judging packages (the T2.1 judging series, T2.0's tooling)
    # touch the list; the product packages, mostly, do not.
    assert len(touching) * 2 < len(landings), (len(touching), len(landings))


# --- Claude r1 P2-2 to P2-4: the guard -----------------------------------------------

guard = _module('t30_fix1_guard', SCRIPTS / 'orchestration/guard_cmd.py')


@pytest.mark.parametrize(
    'command',
    [
        '# clean the old build\nrm -rf /home/user/code/repo/rust/target',
        'cd /home/user/code/repo  # the clone\ngit stash',
        'echo start # note\ngit push origin +topic',
        'for d in /home/user/old*; do rm -rf "$d"; done',
        'if true; then sudo true; fi',
        'git fetch -pq',
        'bash scripts/rust_core/prepush.sh --full',
    ],
)
def test_the_guard_reads_every_line_and_every_segment(command: str) -> None:
    assert guard.check_command(command, '/home/user', None) is not None


def test_a_redirection_is_no_rm_target(tmp_path: Path) -> None:
    os.environ['SDM_SCRATCH'] = str(tmp_path)
    try:
        for command in (
            f'rm -rf {tmp_path}/x 2>/dev/null',
            f'rm -rf {tmp_path}/x > /dev/null 2>&1',
        ):
            assert guard.check_command(command, str(tmp_path), None) is None
    finally:
        del os.environ['SDM_SCRATCH']


def test_a_round_brief_s_candidate_needs_a_stamp(
    repo: Path, tmp_path: Path
) -> None:
    previous = _commit(repo, {'a.txt': 'a\n'}, 'T3.99: round 1')
    sha = _commit(repo, {'a.txt': 'b\n'}, 'T3.99 fix 1: round 2')
    brief = tmp_path / 'round-briefs' / 'T3.99-r2.md'
    brief.parent.mkdir()
    # A real brief's shape (T3.0-r3): the candidate, the previous round,
    # the reference; only the candidate needs the stamp (Claude r3 P2-1).
    brief.write_text(
        f'Candidate: {sha}\n'
        f'Round 2 of T3.99 at SHA {sha}.\n'
        f'Review `{previous[:8]}..{sha[:8]}` against round 1.\n'
        f'Scope: match the reference (python-ref at {"6298913c"}).\n',
        encoding='utf-8',
    )
    prompt = f'Read and follow the brief at {brief}.'
    found = guard.check_agent(prompt, str(repo))
    assert found is not None and sha[:12] in found.rule
    assert previous[:12] not in found.rule
    stamp = handoff.stamp_path(
        Path(_git(repo, 'rev-parse', '--absolute-git-dir')), sha
    )
    stamp.parent.mkdir(parents=True)
    stamp.write_text(json.dumps({'sha': sha, 'ok': True}), encoding='utf-8')
    assert guard.check_agent(prompt, str(repo)) is None
    # Docs work on the reviewer brief names no SHA, and passes.
    edit = 'Tighten the wording of docs/design/rust-core/orchestration/briefs/reviewer.md'
    assert guard.check_agent(edit, str(repo)) is None


# --- Claude r1 P2-5: a stacked package's own commits -------------------------------


def test_a_stacked_package_is_its_commits_above_the_stack(repo: Path) -> None:
    base = _commit(repo, {'a.txt': 'a\n'}, 'Land T3.97: the head')
    _commit(repo, {'b.txt': 'b\n'}, 'T3.98 fix 1: below')
    _git(repo, 'commit', '-q', '--allow-empty', '-m', 'Land T3.98: below')
    landing = _git(repo, 'rev-parse', 'HEAD')
    _commit(repo, {'c.txt': 'c\n'}, 'T3.99: the package')
    assert handoff.stack_floor(repo, base) == landing


# --- fix round 2 ---------------------------------------------------------------------
# The fails-before redesign, Postgres, mypy, the binding suite, the guard's
# comments and briefs, protected pushes, a cancelled stack top, the train's
# scope and the dry run's gate (Codex r2 and Claude r2, the owner's Q14).

lane = _module('t30_fix2_lane', SCRIPTS / 'lane.py')
land = _module('t30_fix2_land', SCRIPTS / 'land.py')


def test_a_rust_fix_with_python_tests_only_is_refused(
    repo: Path, tmp_path: Path
) -> None:
    path = 'rust/crates/demo/src/lag.rs'
    base = _commit(repo, {'README.md': 'x\n'}, 'base')
    previous = _commit(repo, {path: RUST_OLD}, 'T3.99: lag')
    _commit(
        repo,
        {
            path: RUST_OLD.replace('1', '2'),
            'tests/test_lag.py': 'def test_lag() -> None:\n    assert True\n',
        },
        'T3.99 fix 1: lag is 2',
    )
    assert handoff.rust_production(repo, previous, [path]) == [path]
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    step = handoff.step_fails_before(
        repo, base, previous, sys.executable, scratch, 'T3.99'
    )
    assert not step.ok
    assert any('adds no Rust test' in p for p in step.problems), step.problems
    # A test block alone is no production change.
    block = RUST_NEW[RUST_NEW.index('#[cfg(test)]') :]
    _commit(repo, {path: RUST_OLD + '\n' + block}, 'T3.99 tests: lag')
    assert handoff.rust_production(repo, previous, [path]) == []


def test_a_case_naming_postgres_with_others_needs_the_server(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('SDM_POSTGRES_TEST_DSN', raising=False)
    case = {
        'id': 's9.mixed',
        'databases': ['sqlite', 'postgres'],
        'exclusion': 'KO9-1',
    }
    base = _commit(
        repo, {'contract/reference/r.jsonl': '{"case": "other"}\n'}, 'base'
    )
    _commit(
        repo,
        {'contract/cases/s9/s9.mixed.json': json.dumps(case)},
        'T3.99: a mixed case',
    )
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    # A python that fails at once: no SQLite leg can pass here either way.
    step = handoff.step_reference(
        repo,
        ['contract/cases/s9/s9.mixed.json'],
        [],
        '/bin/false',
        base,
        scratch,
    )
    assert any(
        'Postgres' in p and 'SDM_POSTGRES_TEST_DSN' in p for p in step.problems
    )


def test_mypy_judges_the_lane_s_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane_python = str(tmp_path / 'lane' / 'bin' / 'python')
    monkeypatch.setattr(
        handoff,
        'run',
        lambda argv, cwd, env=None: subprocess.CompletedProcess(
            argv, 0 if argv[0] == sys.executable else 1, '', ''
        ),
    )
    command = handoff.mypy(lane_python, tmp_path)
    assert command[:3] == [sys.executable, '-m', 'mypy']
    assert command[-2:] == ['--python-executable', lane_python]


def test_the_binding_suite_runs_on_the_test_ops_build(tmp_path: Path) -> None:
    kit = os.environ.get('SDM_KIT_PYTHON', '')
    if not kit:
        pytest.skip(
            'the binding suite needs SDM_KIT_PYTHON, the test-ops build'
        )
    step = handoff.step_affected(ROOT, ['rust/bindings/sdm-py/src/lib.rs'], kit)
    assert step.ok, step.problems
    assert any('--native-build test-ops' in n for n in step.notes), step.notes


@pytest.mark.parametrize(
    'command',
    [
        '# Document `git config` reads\ngit status --short',
        '# Do not use $(sudo true)\ngit status --short',
    ],
)
def test_a_commented_example_is_no_command(command: str) -> None:
    assert guard.check_command(command, '/home/user', None) is None


@pytest.mark.parametrize(
    'command',
    [
        'git push origin HEAD:rust-core',
        'git push origin src:refs/heads/rust-core',
        'git push origin +HEAD:refs/heads/main',
        'git push origin --delete rust-core',
        'git push origin :rust-core-reference',
        'git push origin rust-core',
        'git push origin main',
        'git push --all origin',
    ],
)
def test_a_push_to_a_protected_branch_is_refused(command: str) -> None:
    found = guard.check_command(command, '/home/user', None)
    assert found is not None and 'owner' in found.rule, found


def test_the_owner_s_token_lets_a_protected_push_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = tmp_path / 'owner-answers.md'
    answers.write_text('## 2026-10-22: merge Phase 3\n', encoding='utf-8')
    monkeypatch.setenv('SDM_OWNER_ANSWERS', str(answers))
    monkeypatch.setattr(guard, 'STATE', tmp_path / 'state')
    approved = 'SDM_APPROVED=2026-10-22-merge-phase-3 git push origin rust-core'
    assert guard.check_command(approved, '/home/user', None) is None


def test_the_pre_push_hook_refuses_a_protected_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refs = tmp_path / 'refs'
    for remote in ('refs/heads/rust-core', 'refs/heads/main'):
        refs.write_text(f'refs/heads/x {"a" * 40} {remote} {"0" * 40}\n')
        args = lane.argparse.Namespace(pushed_refs=str(refs), remote='origin')
        with pytest.raises(lane.Refusal, match='only the owner'):
            lane.check_pushed(args, None, print, 'x')
    answers = tmp_path / 'owner-answers.md'
    answers.write_text('## 2026-10-22: merge Phase 3\n', encoding='utf-8')
    monkeypatch.setenv('SDM_OWNER_ANSWERS', str(answers))
    monkeypatch.setenv('SDM_APPROVED', '2026-10-22-merge-phase-3')
    monkeypatch.setenv('SDM_ORCH_STATE', str(tmp_path / 'state'))
    assert lane.check_pushed(args, None, print, 'x') == 0


def test_a_round_brief_without_its_candidate_line_is_refused(
    repo: Path, tmp_path: Path
) -> None:
    sha = _commit(repo, {'a.txt': 'a\n'}, 'T3.99: the candidate')
    missing = f'Read and follow the brief at {tmp_path}/round-briefs/none.md.'
    assert guard.check_agent(missing, str(repo)) is not None
    brief = tmp_path / 'round-briefs' / 'T3.99-r1.md'
    brief.parent.mkdir()
    brief.write_text(f'Review T3.99 at {sha}.\n', encoding='utf-8')
    found = guard.check_agent(
        f'Read and follow the brief at {brief}.', str(repo)
    )
    assert found is not None and 'candidate line' in found.rule


def test_a_cancelled_stack_top_asks_for_a_rerun(tmp_path: Path) -> None:
    selftest = _module('t30_fix2_land_selftest', SCRIPTS / 'land_selftest.py')
    stack = selftest.t30_stacked(tmp_path, n=2)
    stack.green(2, conclusion='cancelled')
    code = stack.stack_land(2)
    assert code == 2, stack.text
    assert 'gh run rerun' in stack.text and 'IMPLICATED' not in stack.text
    assert stack.remote_phase() == stack.h


def test_a_cancelled_job_is_implicated_in_nothing() -> None:
    def run(argv: list[str]) -> Any:
        if argv[:3] == ['gh', 'run', 'view'] and '--json' in argv:
            jobs = [
                {
                    'name': 'rust-test',
                    'conclusion': 'cancelled',
                    'databaseId': 1,
                }
            ]
            return land.Completed(0, json.dumps({'jobs': jobs}))
        if argv[:2] == ['git', 'log']:
            return land.Completed(0, 'a' * 40 + '\0T3.99: rust\n')
        if argv[:2] == ['git', 'diff-tree']:
            return land.Completed(0, 'rust/crates/x/src/lib.rs\0')
        return land.Completed(0, '')

    report = land.implicated(land.Tools(run=run), {'databaseId': 5}, 'b', 't')
    assert report == []


def test_auto_land_reruns_a_cancelled_stack_top_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('SDM_PHASE_BRANCH', 'rust-core-phase-3')
    auto = _module('t30_fix2_auto_land', SCRIPTS / 'orchestration/auto_land.py')
    reruns: list[tuple[str, ...]] = []
    refused = (
        'REFUSED: the run on the stack top was cancelled (run 77 on x: '
        'cancelled); nothing failed: rerun it (`gh run rerun 77`) and land again'
    )
    monkeypatch.setattr(
        auto,
        'land_py',
        lambda q, *a: subprocess.CompletedProcess(a, 2, refused, ''),
    )
    monkeypatch.setattr(
        auto,
        'sh',
        lambda *a, cwd=None: (
            reruns.append(a) or subprocess.CompletedProcess(a, 0, '', '')
        ),
    )
    entry = tmp_path / 'stack.json'
    queued = {'stack_tip': 'c' * 40, 'worktree': str(tmp_path), 'title': 't'}
    entry.write_text(json.dumps(queued))
    for _ in range(3):
        auto.process_stack_land(entry, json.loads(entry.read_text()), 'b')
    assert entry.exists(), 'the entry stays queued'
    assert [a[:3] for a in reruns] == [('gh', 'run', 'rerun')] * 2


def test_a_checker_fix_rides_the_train_and_product_data_stays_out() -> None:
    checker = [
        'scripts/rust_core/bridge_freeze.py',
        'scripts/rust_core/compare.py',
    ]
    assert judging.not_train(RULES, checker) == []
    assert (
        judging.judging(RULES, ['scripts/rust_core/bridge_freeze.json']) == []
    )


def test_the_dry_run_without_its_gate_is_incomplete(tmp_path: Path) -> None:
    import yaml

    process = _module('t30_fix2_process', ROOT / 'tests/test_t30_process.py')
    _, work, _ = process._clone(tmp_path)
    template = json.loads(
        next(
            (work / 'contract/cases').rglob(f'{process.SEEDED_CASE}.json')
        ).read_text(encoding='utf-8')
    )
    plan = tmp_path / 'kinds.yaml'
    plan.write_text(
        yaml.safe_dump({'kinds': [{'name': 'binary', 'case': template}]})
    )
    done = subprocess.run(
        ['bash', str(work / 'scripts/rust_core/gate_dryrun.sh'), str(plan)],
        cwd=work,
        env=process._env(tmp_path, SDM_PHASE_BRANCH=process.PHASE),
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=3600,
    )
    said = done.stdout + done.stderr
    assert done.returncode == 1, said[-2000:]
    assert 'INCOMPLETE gate not run' in said, said[-2000:]


@pytest.mark.parametrize('command', ['git clean -fdx', 'cargo clean'])
def test_deleting_build_output_is_refused(command: str) -> None:
    assert guard.check_command(command, '/home/user', None) is not None


# --- fix round 3 ---------------------------------------------------------------------


def test_a_push_to_rust_core_is_refused_from_an_unregistered_worktree(
    tmp_path: Path,
) -> None:
    process = _module('t30_fix3_process', ROOT / 'tests/test_t30_process.py')
    origin, work, head = process._clone(tmp_path)
    env = {**os.environ, **GIT_ENV, 'SDM_ORCH_STATE': str(tmp_path / 'state')}
    env.pop('SDM_APPROVED', None)
    subprocess.run(
        ['bash', 'scripts/rust_core/prepush.sh', '--install-hook'],
        cwd=work, env=env, check=True, capture_output=True, text=True,
    )  # fmt: skip
    other = tmp_path / 'other'
    _git(work, 'worktree', 'add', '-q', '-B', 'rust-core', str(other), head)
    # A commit the remote lacks, so each push updates its branch.
    _commit(other, {'t399.txt': 'x\n'}, 'T3.99: an unlanded commit')
    for spec in ('rust-core', 'HEAD:refs/heads/main', ':rust-core-reference'):
        done = subprocess.run(
            ['git', '-C', str(other), 'push', '-q', 'origin', spec],
            env=env, capture_output=True, text=True,
        )  # fmt: skip
        assert done.returncode != 0, (spec, done.stderr)
        assert 'only the owner' in done.stderr, done.stderr
    # A topic branch goes through the protected check (the cheap tier
    # runs only in registered worktrees).
    done = subprocess.run(
        ['git', '-C', str(other), 'push', '-q', 'origin',
         'HEAD:refs/heads/rust-core-phase-3-t3.99'],
        env=env, capture_output=True, text=True,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr


def test_a_binding_source_and_test_change_runs_only_the_binding_command() -> (
    None
):
    printed = [
        ' '.join(s)
        for s in affected.suites(
            [
                'rust/bindings/sdm-py/src/lib.rs',
                'rust/bindings/sdm-py/tests/test_binding.py',
            ]
        )
    ]
    assert printed == [' '.join(affected.BINDING)], printed


def test_a_python_fix_with_a_binding_test_is_no_rust_fix(repo: Path) -> None:
    previous = _commit(repo, {'packages/p/src/p/a.py': 'A = 1\n'}, 'T3.99: a')
    paths = ['packages/p/src/p/a.py', 'rust/bindings/sdm-py/tests/test_a.py']
    assert handoff.rust_production(repo, previous, paths) == []


def test_a_space_inside_a_rust_literal_is_a_production_change(
    repo: Path,
) -> None:
    path = 'rust/crates/demo/src/text.rs'
    previous = _commit(repo, {path: 'pub fn s() -> &str {\n    "ab"\n}\n'}, 'a')
    _commit(repo, {path: 'pub fn s() -> &str {\n    "a b"\n}\n'}, 'b')
    assert handoff.rust_production(repo, previous, [path]) == [path]


def test_a_train_of_a_checker_fix_alone_is_allowed() -> None:
    problem = land.judging_problem(
        ['scripts/rust_core/bridge_freeze.py'], RULES, 'T3.98', True, True, None
    )
    assert problem is None, problem
    # Its data stays product work: no judging path, no train needed.
    assert (
        judging.judging(RULES, ['scripts/rust_core/bridge_freeze.json']) == []
    )


@pytest.mark.parametrize(
    'command',
    [
        'git push --repo=origin HEAD:rust-core',
        'git push --repo origin HEAD:rust-core',
        'git push origin heads/main',
        'cargo +stable clean',
    ],
)
def test_the_guard_s_round_3_spellings(command: str) -> None:
    assert guard.check_command(command, '/home/user', None) is not None


# --- fix round 4 ---------------------------------------------------------------------


def _fake_python(where: Path, installed: Path) -> str:
    """An interpreter whose installed package lives at ``installed``: it
    answers source_path's find_spec probe, and runs pytest otherwise."""
    script = where / 'python'
    script.write_text(
        '#!/usr/bin/env bash\n'
        f'if [ "$1" = -c ]; then echo {installed}; exit 0; fi\n'
        f'exec {sys.executable} "$@"\n'
    )
    script.chmod(0o755)
    return str(script)


def test_each_interpreter_gets_its_own_build_in_the_previous_tree(
    tmp_path: Path,
) -> None:
    tree = tmp_path / 'tree'
    (tree / 'packages/demo/src/demo_answer').mkdir(parents=True)
    pythons = []
    for name in ('production', 'test-ops'):
        installed = tmp_path / name / 'demo_answer'
        installed.mkdir(parents=True)
        (installed / '_native.abi3.so').write_text(name)
        (tmp_path / name / 'bin').mkdir()
        pythons.append(_fake_python(tmp_path / name / 'bin', installed))
    link = tree / 'packages/demo/src/demo_answer/_native.abi3.so'
    handoff.source_path(tree, pythons[0])
    assert link.read_text() == 'production'
    handoff.source_path(tree, pythons[1])
    assert link.read_text() == 'test-ops'


def test_two_batches_that_pass_on_the_previous_tree_show_no_fix(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The control (Codex r4): a plain batch and a binding batch whose tests
    # all pass on the previous implementation satisfy nothing.
    conftest = (
        'def pytest_addoption(parser):\n'
        "    parser.addoption('--native-build', default='')\n"
    )
    passing = 'def test_it() -> None:\n    assert True\n'
    base = _commit(repo, {'README.md': 'x\n'}, 'base')
    previous = _commit(
        repo,
        {
            'packages/demo/src/demo_answer/__init__.py': 'ANSWER = 1\n',
            'rust/bindings/sdm-py/tests/conftest.py': conftest,
        },
        'T3.99: the answer',
    )
    _commit(
        repo,
        {
            'packages/demo/src/demo_answer/__init__.py': 'ANSWER = 2\n',
            'tests/test_demo_plain.py': passing,
            'rust/bindings/sdm-py/tests/test_demo_binding.py': passing,
        },
        'T3.99 fix 1: the answer is 2',
    )
    monkeypatch.setenv('SDM_KIT_PYTHON', sys.executable)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    step = handoff.step_fails_before(
        repo, base, previous, sys.executable, scratch, 'T3.99'
    )
    assert not step.ok
    assert any('none shows the fix' in p for p in step.problems), step.problems
