#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The self-tests of replay_jobs.py, on real temporary git repositories.

Each case makes a repository with a small workflow of its own, commits,
changes something, and runs ``replay_jobs.py --root`` on it as a process,
as prepush.sh does: no mock stands between the script and git. The cases
hold that a worktree with uncommitted or untracked changes is refused
before anything is replayed (a replay of HEAD would not test them), that
a changed workflow variable selects the jobs reading it, that a changed
path selects the jobs naming it, and, end to end, that a clean replay
runs a job's steps in a fresh environment and fails when a step fails.
The end-to-end cases make a virtual environment with ``uv``. Run it as

    uv run --no-project --python 3.12 --with pyyaml \\
        python3 scripts/rust_core/replay_selftest.py

Decision E43.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / 'replay_jobs.py'

WORKFLOW = """\
name: CI
on: push
env:
  FLOOR: one
jobs:
  reads-floor:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: '3.12'
      - run: echo "floor $FLOOR" && python -c "print('reads-floor ran')"
  names-pkg:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: '3.12'
      - run: python pkg/check.py
  rust:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: Swatinem/rust-cache@0000000000000000000000000000000000000000
      - run: cargo test
"""


@dataclass
class Result:
    name: str
    ok: bool
    detail: str


class Repo:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.git('init', '--quiet', '--initial-branch', 'main')
        self.write('.github/workflows/ci.yml', WORKFLOW)
        self.write('pkg/check.py', 'import sys\nsys.exit(0)\n')
        self.write('README', 'base\n')
        self.base = self.commit('base')

    def git(self, *args: str) -> str:
        return subprocess.run(
            ['git', '-c', 'user.name=t', '-c', 'user.email=t@example.invalid',
             *args],
            cwd=self.root, check=True, capture_output=True, text=True,
        ).stdout  # fmt: skip

    def write(self, path: str, text: str) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def commit(self, message: str) -> str:
        self.git('add', '-A')
        self.git('commit', '--quiet', '--allow-empty', '-m', message)
        return self.git('rev-parse', 'HEAD').strip()

    def replay(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), '--root', str(self.root),
             '--base', self.base, *args],
            capture_output=True, text=True, timeout=600,
        )  # fmt: skip


Check = Callable[[Repo], tuple[bool, str]]
CASES: list[tuple[str, Check]] = []


def case(name: str) -> Callable[[Check], Check]:
    def register(check: Check) -> Check:
        CASES.append((name, check))
        return check

    return register


def shown(done: subprocess.CompletedProcess[str]) -> str:
    return f'exit {done.returncode}: {(done.stdout + done.stderr)[-500:]}'


@case('refused: a tracked file changed and not committed')
def _(repo: Repo) -> tuple[bool, str]:
    # Codex's reproduction: a broken file the replay of HEAD would miss.
    repo.write('pkg/check.py', 'this is not python\n')
    done = repo.replay('--job', 'names-pkg')
    ok = (
        done.returncode == 2
        and 'refused' in done.stdout
        and 'pkg/check.py' in done.stdout
        and '==' not in done.stdout
    )
    return ok, 'refused before any replay' if ok else shown(done)


@case('refused: an untracked file')
def _(repo: Repo) -> tuple[bool, str]:
    repo.write('pkg/new_test.py', 'import missing\n')
    done = repo.replay('--list')
    ok = done.returncode == 2 and 'pkg/new_test.py' in done.stdout
    return ok, 'refused' if ok else shown(done)


@case('refused: an uncommitted ci.yml edit')
def _(repo: Repo) -> tuple[bool, str]:
    repo.write('.github/workflows/ci.yml', WORKFLOW.replace('one', 'two'))
    done = repo.replay('--list')
    ok = done.returncode == 2 and 'ci.yml' in done.stdout
    return ok, 'refused' if ok else shown(done)


@case('selects: a changed workflow variable selects the job reading it')
def _(repo: Repo) -> tuple[bool, str]:
    repo.write('.github/workflows/ci.yml', WORKFLOW.replace('one', 'two'))
    repo.commit('floor two')
    done = repo.replay('--list')
    ok = (
        done.returncode == 0
        and 'replay reads-floor: it reads FLOOR, which changed' in done.stdout
        and 'names-pkg' not in done.stdout
    )
    return ok, 'reads-floor only' if ok else shown(done)


@case('selects: a changed path selects the job naming it')
def _(repo: Repo) -> tuple[bool, str]:
    repo.write('pkg/check.py', 'import sys\nsys.exit(0)  # changed\n')
    repo.commit('pkg changed')
    done = repo.replay('--list')
    ok = (
        done.returncode == 0
        and 'replay names-pkg: it names pkg' in done.stdout
        and 'reads-floor' not in done.stdout
    )
    return ok, 'names-pkg only' if ok else shown(done)


@case('selects: nothing for a change no job reads')
def _(repo: Repo) -> tuple[bool, str]:
    repo.write('README', 'changed\n')
    repo.commit('readme')
    done = repo.replay('--list')
    ok = done.returncode == 0 and 'no job to replay' in done.stdout
    return ok, 'none' if ok else shown(done)


@case('refused: a job with an action it cannot replay')
def _(repo: Repo) -> tuple[bool, str]:
    done = repo.replay('--job', 'rust', '--list')
    ok = done.returncode == 2 and 'cannot replay rust' in done.stdout
    return ok, 'refused' if ok else shown(done)


@case('end to end: a clean replay runs the committed steps')
def _(repo: Repo) -> tuple[bool, str]:
    repo.write('.github/workflows/ci.yml', WORKFLOW.replace('one', 'two'))
    repo.commit('floor two')
    done = repo.replay()
    ok = done.returncode == 0 and 'ok   reads-floor' in done.stdout
    return ok, 'ran green' if ok else shown(done)


@case('end to end: a committed failing step fails the replay')
def _(repo: Repo) -> tuple[bool, str]:
    repo.write('pkg/check.py', 'import not_installed_anywhere\n')
    repo.commit('pkg breaks')
    done = repo.replay()
    ok = (
        done.returncode == 1
        and 'FAIL names-pkg' in done.stdout
        and 'not_installed_anywhere' in done.stdout
    )
    return ok, 'failed as CI would' if ok else shown(done)


def main() -> int:
    results: list[Result] = []
    for name, check in CASES:
        with tempfile.TemporaryDirectory(prefix='sdm-replay-test-') as tmp:
            try:
                ok, detail = check(Repo(Path(tmp).resolve()))
            except Exception as error:  # a crash is a failed self-test
                ok, detail = False, f'raised {type(error).__name__}: {error}'
        results.append(Result(name, ok, detail.replace('\n', ' | ')[:500]))
    for result in results:
        print(
            f'{"ok  " if result.ok else "FAIL"} {result.name}: {result.detail}'
        )
    failed = [r for r in results if not r.ok]
    print(
        f'{len(results) - len(failed)} of {len(results)} self-tests as expected'
    )
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
