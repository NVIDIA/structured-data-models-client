# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T3.2's process changes (the package's test author's), one test per
change, each on real git as ``scripts/rust_core/lane_selftest.py`` runs
``lane.py`` (its ``World``: a bare remote, a seed clone, the clone lane.py
runs in, and a fake ``gh``):

- ``lane.py start --brief BRIEF`` runs ``brief_check.py`` and refuses a
  brief missing a required heading, naming it (exit 4 of T3.2's brief);
- from Phase 3 on, ``lane.py`` refuses a package ID with a letter
  (``T3.4a``; the owner's coarse packages, and T2.14a1's broken landings),
  while ``T3.4`` passes;
- ``lane.py`` never starts a lane on a finished phase, one whose branch
  ``rust-core`` already holds (T3.8's run made ``rust-core-phase-2-lane-e``
  from the ``rust-core`` checkout, 2026-10-22);
- ``handoff.py`` removes its ``previous`` worktree when it is stopped
  (SIGTERM), so no registered, dirty ``sdm-handoff-*`` worktree is left;
- ``clean_tmp.sh`` never lists a directory holding a listening socket (the
  live Codex daemon's) as removable.

Each test listed in ``contract/tests/t32-pending.json`` failed when written,
before the implementation, and is held to failing (the ``pending``
decorator, as T2.10's ``held``) until the implementer takes it out of
``pending``; ``pending`` stays within
``authored``, the test author's list.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUST_CORE = ROOT / 'scripts/rust_core'
PENDING_FILE = ROOT / 'contract/tests/t32-pending.json'
TEMPLATE = ROOT / 'docs/design/rust-core/orchestration/briefs/package.md'
PHASE_3 = 'rust-core-phase-3'


def _pending_names(key: str) -> set[str]:
    return set(json.loads(PENDING_FILE.read_text(encoding='utf-8'))[key])


def pending(test: Any) -> Any:
    """Holds ``test`` to failing while ``contract/tests/t32-pending.json``
    lists it, as T2.10's ``held`` does: a pending test passes only by
    failing, and once it passes it fails, naming itself, until the
    implementer takes it out of ``pending``."""
    assert test.__name__ in _pending_names('authored'), test.__name__

    @functools.wraps(test)
    def held(*arguments: Any, **keywords: Any) -> None:
        if test.__name__ not in _pending_names('pending'):
            test(*arguments, **keywords)
            return
        try:
            test(*arguments, **keywords)
        except Exception:  # noqa: BLE001 - pending: failing, as when written
            return
        raise AssertionError(
            f'{test.__name__} passes now: take it out of '
            'contract/tests/t32-pending.json pending'
        )

    return held


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        if str(path.parent) not in sys.path:
            sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _world(tmp_path: Path) -> Any:
    selftest = _module('t32_lane_selftest', RUST_CORE / 'lane_selftest.py')
    tmp_path.mkdir(parents=True)
    return selftest.World(tmp_path)


def _phase_3(world: Any) -> None:
    r"""The remote holds ``rust-core-phase-3`` at a landing, and the tools
    are pointed at it."""
    world.git('checkout', '--quiet', '-B', 'p3', world.base, cwd=world.seed)
    world.commit(world.seed, 'Land T3.1: the design')
    world.git(
        'push',
        '--quiet',
        'origin',
        f'HEAD:refs/heads/{PHASE_3}',
        cwd=world.seed,
    )
    world.git('fetch', '--quiet', 'origin')
    world.env['SDM_PHASE_BRANCH'] = PHASE_3


def _start(world: Any, package: str, *extra: str) -> Any:
    return world.lane(
        'start',
        '--lane',
        '3a',
        '--package',
        package,
        '--worktree',
        str(world.tmp / 'lane-3a'),
        '--dry-run',
        *extra,
    )


def _brief(path: Path, drop: str | None = None) -> Path:
    r"""A brief holding each required heading of the template, but
    ``drop``, each followed by a line of text."""
    check = _module('t32_brief_check', RUST_CORE / 'brief_check.py')
    headings = [h for h in check.required(TEMPLATE) if h != drop]
    path.write_text(
        '# T3.4: a brief\n\n'
        + ''.join(f'{heading}\n\nWritten.\n\n' for heading in headings),
        encoding='utf-8',
    )
    return path


@pending
def test_t32_lane_start_refuses_a_brief_missing_a_heading(
    tmp_path: Path,
) -> None:
    r"""``lane.py start --brief`` runs ``brief_check.py``: a brief missing
    a required heading is refused, naming it, before anything changes; a
    brief holding them all passes the dry run."""
    world = _world(tmp_path / 'w')
    _phase_3(world)
    check = _module('t32_brief_check', RUST_CORE / 'brief_check.py')
    required = check.required(TEMPLATE)
    assert len(required) >= 3
    dropped = required[1]
    before = world.local_state()
    refused = _start(
        world, 'T3.4', '--brief', str(_brief(tmp_path / 'short.md', dropped))
    )
    assert refused.code != 0, refused.text
    assert 'unrecognized arguments' not in refused.text, refused.text
    assert dropped.removeprefix('## ') in refused.text, refused.text
    assert world.local_state() == before
    whole = _start(world, 'T3.4', '--brief', str(_brief(tmp_path / 'whole.md')))
    assert whole.code == 0, whole.text
    assert 'nothing pushed' in whole.text


@pending
def test_t32_lane_refuses_a_package_id_with_a_letter_from_phase_3(
    tmp_path: Path,
) -> None:
    r"""From Phase 3 on a package ID is an integer pair (``T3.4``), never
    lettered (``T3.4a``): the dry run refuses the letter, naming the ID,
    and takes the integer."""
    world = _world(tmp_path / 'w')
    _phase_3(world)
    taken = _start(world, 'T3.4')
    assert taken.code == 0, taken.text
    refused = _start(world, 'T3.4a')
    assert refused.code != 0, refused.text
    assert 'T3.4a' in refused.text


@pending
def test_t32_lane_never_starts_on_a_finished_phase(tmp_path: Path) -> None:
    r"""A phase ``rust-core`` already holds is finished: pointed at it,
    ``lane.py start`` refuses, naming it, and makes no lane of it."""
    world = _world(tmp_path / 'w')
    world.git('checkout', '--quiet', '-B', 'p2', world.base, cwd=world.seed)
    world.commit(world.seed, 'Land T2.16: the phase end')
    world.git(
        'push',
        '--quiet',
        'origin',
        'HEAD:refs/heads/rust-core-phase-2',
        'HEAD:refs/heads/rust-core',
        cwd=world.seed,
    )
    world.git('fetch', '--quiet', 'origin')
    world.env['SDM_PHASE_BRANCH'] = 'rust-core-phase-2'
    before = world.refs()
    ran = world.lane(
        'start',
        '--lane',
        'e',
        '--package',
        'T3.8',
        '--worktree',
        str(world.tmp / 'lane-e'),
    )
    assert ran.code != 0, ran.text
    assert 'rust-core-phase-2' in ran.text
    assert world.refs() == before
    assert not any('lane-e' in ref for ref in world.refs())


@pending
def test_t32_handoff_removes_its_previous_worktree_when_stopped(
    tmp_path: Path,
) -> None:
    r"""A handoff stopped (SIGTERM) while it runs a fix's tests on the
    previous round's tree leaves no ``sdm-handoff-*`` worktree registered
    and no scratch behind."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith('GIT_')},
        'GIT_CONFIG_GLOBAL': str(tmp_path / 'gitconfig'),
        'GIT_CONFIG_NOSYSTEM': '1',
        'GIT_TERMINAL_PROMPT': '0',
        'TMPDIR': str(tmp_path),
    }
    (tmp_path / 'gitconfig').write_text(
        '[user]\n\tname = T\n\temail = t@example.invalid\n', encoding='utf-8'
    )

    def git(*arguments: str) -> str:
        return subprocess.run(
            ['git', '-C', str(repo), *arguments],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            stdin=subprocess.DEVNULL,
            timeout=120,
        ).stdout.strip()

    git('init', '--quiet', '-b', 'work')
    (repo / 'README.md').write_text('base\n', encoding='utf-8')
    git('add', '-A')
    git('commit', '--quiet', '-m', 'Land T3.1: base')
    base = git('rev-parse', 'HEAD')
    (repo / 'notes.md').write_text('round 1\n', encoding='utf-8')
    git('add', '-A')
    git('commit', '--quiet', '-m', 'T3.2: round 1')
    previous = git('rev-parse', 'HEAD')
    (repo / 'tests').mkdir()
    (repo / 'tests/test_t32_slow.py').write_text(
        'import time\n\n\ndef test_slow():\n    time.sleep(60)\n',
        encoding='utf-8',
    )
    git('add', '-A')
    git('commit', '--quiet', '-m', 'T3.2 fix 1: a slow test')
    process = subprocess.Popen(
        [
            sys.executable,
            str(RUST_CORE / 'handoff.py'),
            'run',
            '--package',
            'T3.2',
            '--base',
            base,
            '--previous',
            previous,
            '--python',
            sys.executable,
        ],
        cwd=repo,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 90
        while 'sdm-handoff-' not in git('worktree', 'list', '--porcelain'):
            assert process.poll() is None, (
                'the handoff ended before its worktree'
            )
            assert time.monotonic() < deadline, 'no previous worktree was made'
            time.sleep(0.5)
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)
    left = [
        line
        for line in git('worktree', 'list', '--porcelain').splitlines()
        if 'sdm-handoff-' in line
    ]
    assert left == [], left
    assert not list(tmp_path.glob('sdm-handoff-*'))


@pending
def test_t32_clean_tmp_never_lists_a_directory_holding_a_listening_socket(
    tmp_path: Path,
) -> None:
    r"""A stale directory holding a listening socket (the live Codex
    daemon's) is not removable; a stale one beside it without one is, so
    the dry run reads the directory."""
    roots = {
        name: tmp_path / name for name in ('tmp', 'target', 'cells', 'state')
    }
    for path in roots.values():
        path.mkdir()
    plain = roots['tmp'] / 't32-stale'
    plain.mkdir()
    (plain / 'old.txt').write_text('old\n', encoding='utf-8')
    daemon = roots['tmp'] / 't32-daemon'
    daemon.mkdir()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(daemon / 'daemon.sock'))
        listener.listen(1)
        done = subprocess.run(
            [
                'bash',
                str(RUST_CORE / 'orchestration/clean_tmp.sh'),
                '--dry-run',
            ],
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            env={
                **os.environ,
                'AGE': '0',
                'SDM_TMP_ROOT': str(roots['tmp']),
                'SDM_TARGET_ROOT': str(roots['target']),
                'SDM_CELLS_ROOT': str(roots['cells']),
                'SDM_ORCH_STATE': str(roots['state']),
                'SDM_SCRATCH': '',
            },
            timeout=300,
        )
    finally:
        listener.close()
    assert f'would remove {plain}' in done.stdout, done.stdout + done.stderr
    assert f'would remove {daemon}' not in done.stdout, done.stdout
