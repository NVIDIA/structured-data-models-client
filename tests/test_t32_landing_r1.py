# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Claude's round-1 findings on T3.2's landing tools (T3.2 fix 5):

- the train may carry the CI workflow the owner moved off the landing path
  (P2-6);
- a stamp is written only for the SHA its steps ran on, on a clean tree
  (P2-8);
- a lane is pushed only from its own worktree, where the lane check runs
  (P2-10).

A stamp made for another package, and each stacked package's own stamp
(P2-7, P2-9), are land_selftest.py's cases.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/rust_core'))
GIT_ENV = {
    'GIT_AUTHOR_NAME': 't',
    'GIT_AUTHOR_EMAIL': 't@example.invalid',
    'GIT_COMMITTER_NAME': 't',
    'GIT_COMMITTER_EMAIL': 't@example.invalid',
    'GIT_CONFIG_GLOBAL': os.devnull,
    'GIT_CONFIG_NOSYSTEM': '1',
}


def _module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HANDOFF = _module(ROOT / 'scripts/rust_core/handoff.py', 't32_r1_handoff')
JUDGING = _module(ROOT / 'scripts/rust_core/judging.py', 't32_r1_judging')


def _git(where: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['git', '-C', str(where), *args],
        env={**os.environ, **GIT_ENV},
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )


def test_the_train_carries_the_ci_workflow() -> None:
    rules = JUDGING.patterns((ROOT / JUDGING.FILE).read_text(encoding='utf-8'))
    assert JUDGING.not_train(rules, ['.github/workflows/ci.yml']) == []
    assert JUDGING.not_train(rules, ['.github/workflows/other.yml']) == [
        '.github/workflows/other.yml'
    ]


def test_no_stamp_for_a_tree_that_changed_while_the_steps_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / 'repo'
    repo.mkdir()
    for args in (
        ('init', '-q', '-b', 'topic'),
        ('commit', '-q', '--allow-empty', '-m', 'base'),
    ):
        assert _git(repo, *args).returncode == 0
    base = _git(repo, 'rev-parse', 'HEAD').stdout.strip()
    (repo / 'a.txt').write_text('a\n')
    _git(repo, 'add', 'a.txt')
    _git(repo, 'commit', '-q', '-m', 'T9.9: a change')
    sha = _git(repo, 'rev-parse', 'HEAD').stdout.strip()
    monkeypatch.chdir(repo)
    for name in ('step_lint', 'step_own_tests'):
        monkeypatch.setattr(
            HANDOFF, name, lambda *a, _n=name, **k: HANDOFF.Step(_n)
        )

    def edits(*args: object, **kw: object) -> object:
        (repo / 'a.txt').write_text('edited during the run\n')
        return HANDOFF.Step('own tests')

    monkeypatch.setattr(HANDOFF, 'step_own_tests', edits)
    argv = ['run', '--package', 'T9.9', '--base', base, '--python', 'py']
    assert HANDOFF.main(argv) == 1
    assert not (repo / '.git/sdm-handoff' / f'{sha}.json').exists()
    events = (repo / '.git/sdm-handoff/events.jsonl').read_text()
    assert json.loads(events.splitlines()[-1])['event'] == 'failed'


def test_a_lane_is_pushed_only_from_its_own_worktree(tmp_path: Path) -> None:
    origin, work = tmp_path / 'origin.git', tmp_path / 'work'
    assert _git(tmp_path, 'init', '-q', '--bare', str(origin)).returncode == 0
    assert _git(tmp_path, 'init', '-q', '-b', 'main', str(work)).returncode == 0
    _git(work, 'commit', '-q', '--allow-empty', '-m', 'base')
    _git(work, 'remote', 'add', 'origin', str(origin))
    installed = subprocess.run(
        ['bash', str(ROOT / 'scripts/rust_core/prepush.sh'), '--install-hook'],
        cwd=work,
        env={**os.environ, **GIT_ENV},
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr
    detached = tmp_path / 'detached'
    _git(work, 'worktree', 'add', '-q', '--detach', str(detached))
    _git(detached, 'commit', '-q', '--allow-empty', '-m', 'T2.1e: foreign')
    lane = 'rust-core-phase-3-lane-a'
    pushed = _git(detached, 'push', 'origin', f'HEAD:refs/heads/{lane}')
    assert pushed.returncode != 0, pushed.stderr
    assert f'REFUSED: {lane} is a lane' in pushed.stderr, pushed.stderr
    assert _git(origin, 'rev-parse', '--verify', '-q', lane).returncode != 0
    # Another branch from there is the hook's business no more than before.
    topic = _git(detached, 'push', 'origin', 'HEAD:refs/heads/scratch-topic')
    assert topic.returncode == 0, topic.stderr
