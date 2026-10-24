# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The implementer's unit tests of `scripts/rust_core/prepush.sh`'s cheap
tier skipping the tooling self-tests; the acceptance tests are the test
author's, in `tests/test_t132e_prepush_selftests.py`. Run in a temporary
repository, with stand-ins for `uv`, `uvx` and `cargo` that record what they
were asked to run: an explicit cheap check skips them without tooling changes
and runs all four when tooling changed. Automatic push hooks run ownership
checks only, following the owner's review-before-checks rule."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ('land', 'lane', 'replay_jobs', 'prune_caches')
SELFTESTS = ('land_selftest', 'lane_selftest', 'replay_selftest')


def _lane(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A repository whose `rust-core-phase-1` holds prepush.sh and the four
    tools, with a lane branch checked out, and the environment that runs
    prepush.sh in it against the stand-ins."""
    repo = tmp_path / 'repo'
    scripts = repo / 'scripts/rust_core'
    scripts.mkdir(parents=True)
    (repo / 'rust').mkdir()
    shutil.copy(ROOT / 'scripts/rust_core/prepush.sh', scripts / 'prepush.sh')
    for name in TOOLS + SELFTESTS:
        (scripts / f'{name}.py').write_text('# stand-in\n')
    (repo / 'rust/.keep').write_text('')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    log = tmp_path / 'ran.log'
    for tool in ('uv', 'uvx', 'cargo'):
        stub = bin_dir / tool
        stub.write_text(f'#!/usr/bin/env bash\necho "{tool} $*" >>"{log}"\n')
        stub.chmod(0o755)
    # The caller's SDM_PHASE_BRANCH names another phase than this
    # repository's; the branch checked out says which.
    env = {
        **{k: v for k, v in os.environ.items() if k != 'SDM_PHASE_BRANCH'},
        'PATH': f'{bin_dir}{os.pathsep}{os.environ["PATH"]}',
        'TMPDIR': str(tmp_path),
        'GIT_CONFIG_GLOBAL': os.devnull,
        'GIT_CONFIG_NOSYSTEM': '1',
        'GIT_AUTHOR_NAME': 'test',
        'GIT_AUTHOR_EMAIL': 'test@example.com',
        'GIT_COMMITTER_NAME': 'test',
        'GIT_COMMITTER_EMAIL': 'test@example.com',
    }
    for args in (
        ('init', '-q', '-b', 'rust-core-phase-1'),
        ('add', '-A'),
        ('commit', '-q', '-m', 'phase head'),
        ('checkout', '-q', '-b', 'rust-core-phase-1-lane-1a'),
    ):
        subprocess.run(
            ['git', *args], cwd=repo, env=env, check=True, capture_output=True
        )
    return repo, env


def _prepush(
    repo: Path, env: dict[str, str], pushed: str | None = None
) -> tuple[str, str]:
    """prepush.sh's output and what the stand-ins were asked to run; given
    `pushed`, as the hook runs it, with `pushed` as git's list of refs."""
    log = repo.parent / 'ran.log'
    log.unlink(missing_ok=True)
    args = [] if pushed is None else ['--pushed-refs']
    done = subprocess.run(
        ['bash', 'scripts/rust_core/prepush.sh', *args],
        cwd=repo,
        env=env,
        input=pushed if pushed is not None else '',
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout, log.read_text() if log.exists() else ''


def test_an_unchanged_lane_py_skips_its_self_test_and_says_why(
    tmp_path: Path,
) -> None:
    repo, env = _lane(tmp_path)
    out, ran = _prepush(repo, env)
    assert 'lane_selftest.py' not in ran
    assert 'lane.py check' in ran
    skipped = [line for line in out.splitlines() if line.startswith('skipped')]
    assert len(skipped) == 1, out
    assert 'lane.py self-tests' in skipped[0]
    assert 'rust-core-phase-1' in skipped[0]


def test_changing_lane_py_runs_all_four(tmp_path: Path) -> None:
    repo, env = _lane(tmp_path)
    (repo / 'scripts/rust_core/lane.py').write_text('# changed\n')
    subprocess.run(
        ['git', 'commit', '-qam', 'lane.py changed'],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
    )
    out, ran = _prepush(repo, env)
    for selftest in SELFTESTS:
        assert f'scripts/rust_core/{selftest}.py' in ran
    assert 'prune_caches.py --selftest' in ran
    assert not any(line.startswith('skipped') for line in out.splitlines())


def test_a_pushed_ref_does_not_start_package_checks_even_when_tools_changed(
    tmp_path: Path,
) -> None:
    repo, env = _lane(tmp_path)
    git = {'cwd': repo, 'env': env, 'check': True, 'capture_output': True}
    subprocess.run(['git', 'checkout', '-q', '-b', 'other'], **git)
    (repo / 'scripts/rust_core/lane.py').write_text('# changed\n')
    subprocess.run(['git', 'commit', '-qam', 'lane.py changed'], **git)
    sha = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], text=True, **git
    ).stdout.strip()
    subprocess.run(
        ['git', 'checkout', '-q', 'rust-core-phase-1-lane-1a'], **git
    )
    pushed = f'refs/heads/other {sha} refs/heads/other {"0" * 40}\n'
    out, ran = _prepush(repo, env, pushed)
    assert ran == ''
    assert 'push safety passed' in out
    # An empty ref list also must not start checks before review.
    out, ran = _prepush(repo, env, '')
    assert ran == ''
    assert 'push safety passed' in out
