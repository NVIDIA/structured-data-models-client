# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""watchdog.py (T3.2 fix 4, the owner's speed change 4): an idle lane, a red
CI job and a slip each print one line, and a healthy state prints nothing.
The lanes are real git worktrees; gh, the clock and the process scan are
injected."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'scripts/rust_core/orchestration/watchdog.py'
PHASE = 'rust-core-phase-3'


def load():
    sys.path.insert(0, str(PATH.parent))
    spec = importlib.util.spec_from_file_location('t32_watchdog', PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


watchdog = load()

#: The lane's commit time; ``now`` is two hours later.
COMMITTED = '2026-10-22T10:00:00+00:00'
NOW = '2026-10-22T12:00:00Z'


def git(cwd: Path, *args: str) -> str:
    env = {
        **os.environ,
        'GIT_CONFIG_GLOBAL': os.devnull,
        'GIT_CONFIG_NOSYSTEM': '1',
        'GIT_AUTHOR_NAME': 't',
        'GIT_AUTHOR_EMAIL': 't@example.com',
        'GIT_COMMITTER_NAME': 't',
        'GIT_COMMITTER_EMAIL': 't@example.com',
        'GIT_AUTHOR_DATE': COMMITTED,
        'GIT_COMMITTER_DATE': COMMITTED,
    }
    return subprocess.run(
        ['git', *args], cwd=cwd, env=env, check=True, capture_output=True,
        text=True,
    ).stdout  # fmt: skip


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A clone with the phase branch and a lane worktree one commit ahead."""
    main = tmp_path / 'repo'
    main.mkdir()
    git(main, 'init', '-q', '-b', PHASE)
    git(main, 'commit', '-q', '--allow-empty', '-m', 'Land T3.2: the train')
    lane = tmp_path / 'lane-a'
    git(main, 'worktree', 'add', '-q', '-b', f'{PHASE}-lane-a', str(lane))
    git(lane, 'commit', '-q', '--allow-empty', '-m', 'T3.3: work')
    return main


def forecast(tmp_path: Path) -> Path:
    path = tmp_path / 'forecast.yaml'
    path.write_text(
        "start: '2026-10-22T00:00:00Z'\n"
        'packages:\n  - {id: T3.2, hours: 4}\n  - {id: T3.3, hours: 2}\n',
        encoding='utf-8',
    )
    return path


def green(args: Sequence[str]) -> str:
    if list(args[:2]) == ['run', 'list']:
        return json.dumps(
            [
                {'databaseId': 7, 'headBranch': f'{PHASE}-lane-a',
                 'status': 'completed', 'conclusion': 'success',
                 'createdAt': '2026-10-22T11:00:00Z'},
                {'databaseId': 8, 'headBranch': PHASE, 'event': 'schedule',
                 'status': 'completed', 'conclusion': 'success',
                 'createdAt': '2026-10-22T02:17:00Z'},
            ]
        )  # fmt: skip
    raise AssertionError(f'unexpected gh {args}')


def run(
    capsys: pytest.CaptureFixture[str],
    repo: Path,
    *extra: str,
    gh=green,
    cwds: Sequence[str] = (),
) -> list[str]:
    code = watchdog.main(
        ['--phase', PHASE, '--repo', str(repo), '--now', NOW, *extra],
        gh=gh,
        cwds=lambda: list(cwds),
    )
    assert code == 0
    return capsys.readouterr().out.splitlines()


def test_a_healthy_state_prints_nothing(repo, tmp_path, capsys):
    lane = str(tmp_path / 'lane-a')
    # A process works in the lane, CI is green, T3.2 landed and T3.3 is
    # within the slip threshold (due 06:00, 6 h behind, threshold 8).
    out = run(
        capsys, repo, '--forecast', str(forecast(tmp_path)),
        '--slip-hours', '8', cwds=[lane + '/rust'],
    )  # fmt: skip
    assert out == []


def test_a_lane_with_no_commit_and_no_process_is_idle(repo, tmp_path, capsys):
    out = run(capsys, repo, cwds=['/elsewhere'])
    lane = tmp_path / 'lane-a'
    assert out == [
        f'IDLE {lane} ({PHASE}-lane-a): no commit for 120 min and no process'
    ]


def test_a_red_run_names_its_failed_jobs(repo, tmp_path, capsys):
    def red(args: Sequence[str]) -> str:
        if list(args[:2]) == ['run', 'list']:
            return json.dumps(
                [
                    {'databaseId': 9, 'headBranch': f'{PHASE}-lane-a',
                     'status': 'completed', 'conclusion': 'failure',
                     'createdAt': '2026-10-22T11:30:00Z'},
                    {'databaseId': 8, 'headBranch': f'{PHASE}-lane-a',
                     'status': 'completed', 'conclusion': 'success',
                     'createdAt': '2026-10-22T11:00:00Z'},
                    {'databaseId': 5, 'headBranch': 'rust-core-phase-2-lane-e',
                     'status': 'completed', 'conclusion': 'failure',
                     'createdAt': '2026-10-22T11:40:00Z'},
                    {'databaseId': 4, 'headBranch': PHASE, 'event': 'schedule',
                     'status': 'completed', 'conclusion': 'success',
                     'createdAt': '2026-10-22T02:17:00Z'},
                ]
            )  # fmt: skip
        assert list(args) == ['run', 'view', '9', '--json', 'jobs']
        return json.dumps(
            {
                'jobs': [
                    {'name': 'lint', 'conclusion': 'success'},
                    {'name': 'contract (ref-pandas3)', 'conclusion': 'failure'},
                ]
            }
        )

    out = run(capsys, repo, gh=red, cwds=[str(tmp_path / 'lane-a')])
    assert out == [f'RED CI {PHASE}-lane-a run 9: contract (ref-pandas3)']


def test_a_gh_failure_is_one_line_not_a_crash(repo, tmp_path, capsys):
    def down(args: Sequence[str]) -> str:
        raise RuntimeError('gh run list: network down')

    out = run(capsys, repo, gh=down, cwds=[str(tmp_path / 'lane-a')])
    assert out == ['WATCHDOG gh failed: gh run list: network down']


def test_a_package_past_its_forecast_and_not_landed_slips(
    repo, tmp_path, capsys
):
    out = run(
        capsys, repo, '--forecast', str(forecast(tmp_path)),
        cwds=[str(tmp_path / 'lane-a')],
    )  # fmt: skip
    # T3.2 landed; T3.3 was due at 06:00 and is 6 h behind at 12:00.
    assert out == [
        'SLIP T3.3: expected by 2026-10-22 06:00Z, not landed; 6.0 h behind'
    ]


def test_a_lane_at_the_phase_head_with_no_package_is_not_idle(tmp_path, capsys):
    main = tmp_path / 'repo'
    main.mkdir()
    git(main, 'init', '-q', '-b', PHASE)
    git(main, 'commit', '-q', '--allow-empty', '-m', 'Land T3.2: the train')
    git(main, 'worktree', 'add', '-q', '-b', f'{PHASE}-lane-b',
        str(tmp_path / 'lane-b'))  # fmt: skip
    assert run(capsys, main) == []


def test_a_red_nightly_is_one_fix_item_and_a_missing_one_is_dispatched(
    repo, tmp_path, capsys
):
    calls: list[list[str]] = []

    def nightly(args: Sequence[str]) -> str:
        calls.append(list(args))
        if list(args[:2]) == ['run', 'list'] and '--branch' in args:
            return json.dumps(
                [
                    {'databaseId': 31, 'headBranch': PHASE, 'event': 'schedule',
                     'headSha': 'a' * 40, 'status': 'completed',
                     'conclusion': 'failure', 'createdAt': '2026-10-21T02:17:00Z'},
                    {'databaseId': 30, 'headBranch': PHASE, 'event': 'push',
                     'status': 'completed', 'conclusion': 'success',
                     'createdAt': '2026-10-21T09:00:00Z'},
                ]
            )  # fmt: skip
        if list(args[:2]) == ['run', 'list']:
            return '[]'
        if list(args[:2]) == ['workflow', 'run']:
            return ''
        if list(args[:3]) == ['run', 'view', '--job']:
            assert args[3] == '9' and args[4] == '--log-failed', args
            return (
                'engine\tpytest\tFAILED packages/c/tests/test_x.py::test_y - x\n'
                'engine\tpytest\tFAILED packages/c/tests/test_x.py::test_y - x\n'
            )
        assert list(args) == ['run', 'view', '31', '--json', 'jobs']
        return json.dumps(
            {'jobs': [{'name': 'engine', 'conclusion': 'failure', 'databaseId': 9},
                      {'name': 'lint', 'conclusion': 'success', 'databaseId': 10}]}
        )  # fmt: skip

    items = tmp_path / 'state/phase-3/fix-items.md'
    lane = [str(tmp_path / 'lane-a')]
    for _ in range(2):
        out = run(
            capsys, repo, '--fix-items', str(items), '--dispatch',
            gh=nightly, cwds=lane,
        )  # fmt: skip
        # Red names its failing tests; none finished in 26 h is missing
        # (the owner, "2026-10-23 early: nightly forced-route run").
        tests = 'engine; failing tests: packages/c/tests/test_x.py::test_y'
        assert out == [
            f'NIGHTLY {PHASE}: dispatched (none in 24 h)',
            f'NIGHTLY {PHASE}: none finished in 26 h',
            f'NIGHTLY {PHASE} run 31: {tests}',
        ], out
    # One item each, however often the watchdog reads them.
    assert items.read_text().splitlines() == [
        f'- [ ] nightly on {PHASE} missing 2026-10-22: none finished in 26 h',
        f'- [ ] nightly run 31 on {PHASE} {"a" * 12}: {tests}',
    ]
    assert [
        'workflow',
        'run',
        'ci.yml',
        '--ref',
        PHASE,
        '-f',
        'fresh=true',
    ] in calls
