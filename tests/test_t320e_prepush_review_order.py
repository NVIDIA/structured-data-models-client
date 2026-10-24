# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real pushes enforce ownership and protected refs without running checks."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.tooling


def _run(
    repo: Path, env: dict[str, str], *command: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=repo,
        env=env,
        input='',
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.fixture
def push_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / 'repo'
    repo.mkdir()
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(('GIT_', 'SDM_'))
    }
    env.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM='1',
        GIT_AUTHOR_NAME='test',
        GIT_AUTHOR_EMAIL='test@example.com',
        GIT_COMMITTER_NAME='test',
        GIT_COMMITTER_EMAIL='test@example.com',
        GIT_TERMINAL_PROMPT='0',
        SDM_PHASE_BRANCH='rust-core-phase-3',
        SDM_ORCH_STATE=str(tmp_path / 'state'),
    )
    scripts = repo / 'scripts/rust_core'
    scripts.mkdir(parents=True)
    shutil.copy(ROOT / 'scripts/rust_core/prepush.sh', scripts / 'prepush.sh')
    (scripts / 'lane.py').write_text(
        'import os, pathlib, sys\n'
        'assert sys.argv[1:3] == ["check", "--pushed-refs"]\n'
        'refs = pathlib.Path(sys.argv[3]).read_text()\n'
        'assert "refs/heads/" in refs\n'
        'pathlib.Path("ownership.log").write_text(refs)\n'
        'sys.exit(int(os.environ.get("OWNERSHIP_EXIT", "0")))\n'
    )
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name in ('uv', 'uvx', 'cargo'):
        tool = bin_dir / name
        tool.write_text(
            '#!/usr/bin/env bash\n'
            'echo "unexpected package check" >> checks.log\n'
            'exit 86\n'
        )
        tool.chmod(0o755)
    env['PATH'] = f'{bin_dir}{os.pathsep}{env["PATH"]}'
    for command in (
        ('git', 'init', '-q', '-b', 'rust-core-phase-3-topic-T3.20e'),
        ('git', 'add', 'scripts'),
        ('git', 'commit', '-q', '-m', 'T3.20e: fixture'),
        ('git', 'init', '--bare', '-q', str(tmp_path / 'remote.git')),
        ('git', 'remote', 'add', 'origin', str(tmp_path / 'remote.git')),
        ('bash', 'scripts/rust_core/prepush.sh', '--install-hook'),
    ):
        done = _run(repo, env, *command)
        assert done.returncode == 0, done.stdout + done.stderr
    return repo, env


def test_review_candidate_push_runs_ownership_without_package_checks(
    push_repo: tuple[Path, dict[str, str]],
) -> None:
    repo, env = push_repo
    done = _run(repo, env, 'git', 'push', 'origin', 'HEAD')
    assert done.returncode == 0, done.stdout + done.stderr
    assert 'topic-T3.20e' in (repo / 'ownership.log').read_text()
    assert not (repo / 'checks.log').exists()


def test_review_candidate_push_still_refuses_bad_ownership(
    push_repo: tuple[Path, dict[str, str]],
) -> None:
    repo, env = push_repo
    done = _run(
        repo, {**env, 'OWNERSHIP_EXIT': '1'}, 'git', 'push', 'origin', 'HEAD'
    )
    assert done.returncode != 0
    assert (repo / 'ownership.log').exists()
    assert not (repo / 'checks.log').exists()
    remote = _run(repo, env, 'git', 'ls-remote', '--heads', 'origin')
    assert remote.returncode == 0 and remote.stdout == ''


@pytest.mark.parametrize('branch', ['main', 'rust-core', 'rust-core-reference'])
def test_review_push_does_not_weaken_protected_branch_guard(
    push_repo: tuple[Path, dict[str, str]], branch: str
) -> None:
    repo, env = push_repo
    done = _run(repo, env, 'git', 'push', 'origin', f'HEAD:refs/heads/{branch}')
    assert done.returncode != 0
    assert 'REFUSED' in done.stderr
    assert not (repo / 'ownership.log').exists()
    assert not (repo / 'checks.log').exists()


def test_review_push_is_refused_while_handoff_runs(
    push_repo: tuple[Path, dict[str, str]],
) -> None:
    repo, env = push_repo
    # Worktree config is fixture state, never the shared repository's config.
    done = _run(repo, env, 'git', 'config', 'sdm.package', 'T3.20e')
    assert done.returncode == 0, done.stderr
    (repo / 'scripts/rust_core/handoff.py').write_text(
        'import sys\nprint("handoff still running")\nsys.exit(1)\n'
    )
    done = _run(repo, env, 'git', 'push', 'origin', 'HEAD')
    assert done.returncode != 0
    assert 'handoff still running' in done.stderr
    assert not (repo / 'ownership.log').exists()
    assert not (repo / 'checks.log').exists()


def test_existing_registered_worktree_uses_installed_safety_checker(
    push_repo: tuple[Path, dict[str, str]],
) -> None:
    repo, env = push_repo
    legacy = repo.parent / 'legacy'
    done = _run(
        repo, env, 'git', 'worktree', 'add', '-b', 'other-review-topic', str(legacy)
    )
    assert done.returncode == 0, done.stdout + done.stderr
    (legacy / 'scripts/rust_core/prepush.sh').write_text(
        '#!/usr/bin/env bash\necho "obsolete package checks" >&2\nexit 86\n'
    )
    registered = repo / '.git/hooks/prepush-worktrees'
    with registered.open('a') as stream:
        stream.write(f'{legacy}\n')
    done = _run(legacy, env, 'git', 'push', 'origin', 'HEAD')
    assert done.returncode == 0, done.stdout + done.stderr
    assert (legacy / 'ownership.log').exists()
    assert not (legacy / 'checks.log').exists()
