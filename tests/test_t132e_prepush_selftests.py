# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""When ``scripts/rust_core/prepush.sh``'s cheap tier may skip the tooling
self-tests: the independent test author's acceptance tests for the change
of 5667163c (T1.32e), from the review's rule (Codex, "dependency-trigger
coverage is incomplete": ``replay_jobs.py`` runs ``native_wheel.py``, whose
change left ``replay_selftest.py`` skipped).

The rule held here, the conservative fallback the review asked for: the
four self-tests (``land.py``'s, ``lane.py``'s, ``replay_jobs.py``'s and
``prune_caches.py``'s) may be skipped only when no file under
``scripts/rust_core/`` differs from the merge base with the phase branch,
committed, uncommitted or untracked; any change there runs all four.
``--full`` runs them always, and so does a repository with no phase branch
to measure against. A skip is said, in one line naming the merge base.

Each test runs the real ``prepush.sh`` in a throwaway repository whose
``rust-core-phase-1`` holds it and stand-in tools, a lane branch checked
out, with stand-ins for ``uv``, ``uvx`` and ``cargo`` that record what they
were asked to run (``tests/test_prepush_selftests.py``'s method; nothing of
this repository is run or changed). The seeds: an untracked
``native_wheel.py``; a dirty ``land.py``; a committed change to another
helper under ``scripts/rust_core/``; a change only outside it (a docs file);
``--full``; no phase branch.

Gating (``contract/tests/t132e_gate.py``): a test named in
``t132e-pending.json``'s ``pending["p2-exits"]`` (as ``py:<test id>``)
failed when written, against 5667163c's narrower per-script rule. It still
runs and must still fail; once it passes it fails, naming itself, until it
is taken out of the list.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
#: The four self-tests: prepush.sh's name for each, and what the stand-ins
#: record when it runs.
SELFTESTS = {
    'land.py self-tests': 'scripts/rust_core/land_selftest.py',
    'lane.py self-tests': 'scripts/rust_core/lane_selftest.py',
    'replay_jobs.py self-tests': 'scripts/rust_core/replay_selftest.py',
    'prune_caches.py self-tests': 'scripts/rust_core/prune_caches.py --selftest',
}
#: The tools the phase head holds, beside prepush.sh.
TOOLS = (
    'land',
    'land_selftest',
    'lane',
    'lane_selftest',
    'replay_jobs',
    'replay_selftest',
    'prune_caches',
    'coverage',
    'compare',
)


def _gate() -> Any:
    name = 't132e_gate'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'contract/tests/t132e_gate.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


G = _gate()


def _git(repo: Path, env: dict[str, str], *args: str) -> None:
    subprocess.run(
        ['git', *args], cwd=repo, env=env, check=True, capture_output=True
    )


def _repo(tmp_path: Path, phase: bool = True) -> tuple[Path, dict[str, str]]:
    r"""A repository whose first branch (``rust-core-phase-1``, or ``main``
    without a phase branch) holds prepush.sh, stand-in tools and a docs
    file, a lane branch checked out; and the environment running prepush.sh
    in it against the stand-ins."""
    repo = tmp_path / 'repo'
    scripts = repo / 'scripts/rust_core'
    scripts.mkdir(parents=True)
    (repo / 'rust').mkdir()
    (repo / 'docs').mkdir()
    shutil.copy(ROOT / 'scripts/rust_core/prepush.sh', scripts / 'prepush.sh')
    for name in TOOLS:
        (scripts / f'{name}.py').write_text('# stand-in\n')
    (repo / 'rust/.keep').write_text('')
    (repo / 'docs/notes.md').write_text('notes\n')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    log = tmp_path / 'ran.log'
    for tool in ('uv', 'uvx', 'cargo'):
        stub = bin_dir / tool
        stub.write_text(f'#!/usr/bin/env bash\necho "{tool} $*" >>"{log}"\n')
        stub.chmod(0o755)
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
    for key in [k for k in env if k.startswith('GIT_') and 'DIR' in k]:
        del env[key]
    first = 'rust-core-phase-1' if phase else 'main'
    _git(repo, env, 'init', '-q', '-b', first)
    _git(repo, env, 'add', '-A')
    _git(repo, env, 'commit', '-q', '-m', 'phase head')
    _git(repo, env, 'checkout', '-q', '-b', 'rust-core-phase-1-lane-1a')
    return repo, env


def _prepush(repo: Path, env: dict[str, str], *args: str) -> tuple[str, str]:
    r"""prepush.sh's output and what the stand-ins were asked to run."""
    log = repo.parent / 'ran.log'
    log.unlink(missing_ok=True)
    done = subprocess.run(
        ['bash', 'scripts/rust_core/prepush.sh', *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout, log.read_text() if log.exists() else ''


def _ran(ran: str) -> set[str]:
    r"""The self-tests the stand-ins were asked to run, by prepush.sh's
    name."""
    return {
        name
        for name, command in SELFTESTS.items()
        if any(line.endswith(command) for line in ran.splitlines())
    }


def _skip_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith('skipped')]


def _all_run(out: str, ran: str) -> None:
    assert _ran(ran) == set(SELFTESTS), ran
    assert _skip_lines(out) == [], out


def _commit(repo: Path, env: dict[str, str], message: str) -> None:
    _git(repo, env, 'add', '-A')
    _git(repo, env, 'commit', '-q', '-m', message)


def _untracked_native_wheel(repo: Path, env: dict[str, str]) -> None:
    # replay_jobs.py runs native_wheel.py: a new one, not yet added.
    (repo / 'scripts/rust_core/native_wheel.py').write_text('# new\n')


def _dirty_land(repo: Path, env: dict[str, str]) -> None:
    (repo / 'scripts/rust_core/land.py').write_text('# changed, uncommitted\n')


def _committed_helper(repo: Path, env: dict[str, str]) -> None:
    (repo / 'scripts/rust_core/coverage.py').write_text('# changed\n')
    _commit(repo, env, 'a helper changed')


def _untracked_new_helper(repo: Path, env: dict[str, str]) -> None:
    (repo / 'scripts/rust_core/l0_helpers.py').write_text('# new\n')


def _dirty_prepush(repo: Path, env: dict[str, str]) -> None:
    path = repo / 'scripts/rust_core/prepush.sh'
    path.write_text(path.read_text() + '\n# changed, uncommitted\n')


#: Each change under scripts/rust_core/, which must run all four.
CHANGES = {
    'untracked-native-wheel': _untracked_native_wheel,
    'dirty-land': _dirty_land,
    'committed-unrelated-helper': _committed_helper,
    'untracked-new-helper': _untracked_new_helper,
    'dirty-prepush': _dirty_prepush,
}


@pytest.mark.parametrize('change', list(CHANGES))
def test_any_change_under_scripts_rust_core_runs_all_four(
    change: str, tmp_path: Path
) -> None:
    def check() -> None:
        repo, env = _repo(tmp_path)
        CHANGES[change](repo, env)
        out, ran = _prepush(repo, env)
        _all_run(out, ran)

    G.held(
        'p2-exits',
        f'py:test_any_change_under_scripts_rust_core_runs_all_four[{change}]',
        check,
    )


def test_a_change_only_outside_scripts_rust_core_skips_them_and_says_so(
    tmp_path: Path,
) -> None:
    repo, env = _repo(tmp_path)
    (repo / 'docs/notes.md').write_text('notes, changed\n')
    _commit(repo, env, 'docs changed')
    (repo / 'docs/more.md').write_text('untracked\n')
    out, ran = _prepush(repo, env)
    assert _ran(ran) == set(), ran
    # The rest of the cheap tier still ran.
    assert 'lane.py check' in ran
    [line] = _skip_lines(out)
    for name in SELFTESTS:
        assert name in line, line
    assert 'rust-core-phase-1' in line, line


def test_an_unchanged_lane_skips_them(tmp_path: Path) -> None:
    repo, env = _repo(tmp_path)
    out, ran = _prepush(repo, env)
    assert _ran(ran) == set(), ran
    assert len(_skip_lines(out)) == 1, out


@pytest.mark.parametrize('change', ['none', 'docs', 'dirty-land'])
def test_full_always_runs_them(change: str, tmp_path: Path) -> None:
    repo, env = _repo(tmp_path)
    if change == 'docs':
        (repo / 'docs/notes.md').write_text('notes, changed\n')
        _commit(repo, env, 'docs changed')
    elif change == 'dirty-land':
        _dirty_land(repo, env)
    out, ran = _prepush(repo, env, '--full')
    _all_run(out, ran)


@pytest.mark.parametrize('change', ['none', 'docs'])
def test_with_no_phase_branch_they_always_run(
    change: str, tmp_path: Path
) -> None:
    repo, env = _repo(tmp_path, phase=False)
    if change == 'docs':
        (repo / 'docs/notes.md').write_text('notes, changed\n')
        _commit(repo, env, 'docs changed')
    out, ran = _prepush(repo, env)
    _all_run(out, ran)


def test_the_reading_of_what_ran_fails_closed() -> None:
    # The stand-ins' log read by the command's end: a self-test named only
    # in passing is not one run, and an empty log runs none.
    assert _ran('') == set()
    assert _ran('uv run python3 scripts/rust_core/prune_caches.py') == set()
    assert _ran(
        'uv run --quiet python3 scripts/rust_core/prune_caches.py --selftest\n'
        'uv run python3 scripts/rust_core/land_selftest.py'
    ) == {'prune_caches.py self-tests', 'land.py self-tests'}
    with pytest.raises(AssertionError):
        _all_run('', 'uv run python3 scripts/rust_core/land_selftest.py')
    with pytest.raises(AssertionError):
        _all_run(
            'skipped, unchanged: lane.py self-tests',
            '\n'.join(f'uv {c}' for c in SELFTESTS.values()),
        )
