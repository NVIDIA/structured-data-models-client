# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI's workflow under decision E131 (``.github/workflows/ci.yml``).

The landing job's step, its text as CI runs it (``bash -e``), against real
temporary git repositories: it says skip for an empty landing commit pushed
to the phase branch or a branch under it, and for nothing else. Every other
job is left out when it says skip, directly or through the jobs it needs.
Every job has a timeout. Each image is named once and pinned by digest,
and ``scripts/rust_core/pull_image.sh`` retries a pull, refuses an image
not pinned by digest, and logs in to ghcr.io alone.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / '.github/workflows/ci.yml'
PULL = ROOT / 'scripts/rust_core/pull_image.sh'
PHASE = 'rust-core-phase-1'
LANE = 'rust-core-phase-1-lane-1a'
SKIP = "needs.landing.outputs.skip != 'true'"
DIGEST = re.compile(r'@sha256:[0-9a-f]{64}$')
GIT_ENV = {
    'GIT_CONFIG_GLOBAL': os.devnull,
    'GIT_CONFIG_NOSYSTEM': '1',
    'GIT_AUTHOR_NAME': 'Test Person',
    'GIT_AUTHOR_EMAIL': 'test@example.com',
    'GIT_COMMITTER_NAME': 'Test Person',
    'GIT_COMMITTER_EMAIL': 'test@example.com',
}


def _workflow() -> dict[str, Any]:
    return yaml.safe_load(CI.read_text(encoding='utf-8'))


def _needs(job: dict[str, Any]) -> list[str]:
    needs = job.get('needs', [])
    return [needs] if isinstance(needs, str) else list(needs)


def _landing_step() -> dict[str, Any]:
    steps = _workflow()['jobs']['landing']['steps']
    return next(s for s in steps if s.get('id') == 'landing')


# --- the landing job's step ---------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), *args],
        env={**os.environ, **GIT_ENV},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', 'work')
    (repo / 'a.txt').write_text('a\n')
    _git(repo, 'add', 'a.txt')
    _git(repo, 'commit', '-q', '-m', 'Land T1.1: the phase so far')
    (repo / 'b.txt').write_text('b\n')
    _git(repo, 'add', 'b.txt')
    _git(repo, 'commit', '-q', '-m', 'T1.2: the candidate')
    return repo


def _skip(repo: Path, tmp_path: Path, event: str, ref: str) -> str:
    output = tmp_path / 'output'
    output.write_text('')
    done = subprocess.run(
        ['bash', '--noprofile', '--norc', '-e', '-c', _landing_step()['run']],
        cwd=repo,
        env={
            **os.environ,
            **GIT_ENV,
            'GITHUB_EVENT_NAME': event,
            'GITHUB_REF_NAME': ref,
            'GITHUB_OUTPUT': str(output),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    lines = output.read_text().splitlines()
    assert len(lines) == 1 and lines[0].startswith('skip='), lines
    return lines[0].removeprefix('skip=')


def _landing(repo: Path, subject: str = 'Land T1.2: the candidate') -> None:
    _git(
        repo, 'commit', '-q', '--allow-empty', '-m',
        f'{subject}\n\nBase-SHA: x\nCheckpoint-SHA: x',
    )  # fmt: skip


@pytest.mark.parametrize('ref', [PHASE, LANE, f'{PHASE}-t1.99'])
def test_an_empty_landing_commit_pushed_under_the_phase_branch_skips(
    tmp_path: Path, ref: str
) -> None:
    repo = _repo(tmp_path)
    _landing(repo)
    assert _skip(repo, tmp_path, 'push', ref) == 'true'


@pytest.mark.parametrize(
    ('event', 'ref'),
    [('pull_request', PHASE), ('push', 'main'), ('push', 'rust-core')],
)
def test_elsewhere_a_landing_commit_runs_everything(
    tmp_path: Path, event: str, ref: str
) -> None:
    repo = _repo(tmp_path)
    _landing(repo)
    assert _skip(repo, tmp_path, event, ref) == 'false'


def test_a_candidate_runs_everything(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    assert _skip(repo, tmp_path, 'push', LANE) == 'false'


def test_a_landing_commit_that_changes_a_file_runs_everything(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    (repo / 'b.txt').write_text('changed at landing\n')
    _git(repo, 'commit', '-q', '-am', 'Land T1.2: the candidate')
    assert _skip(repo, tmp_path, 'push', PHASE) == 'false'


@pytest.mark.parametrize(
    'subject', ['T1.2: an empty commit', 'Landed T1.2: x', 'land T1.2: x']
)
def test_an_empty_commit_not_a_landing_runs_everything(
    tmp_path: Path, subject: str
) -> None:
    repo = _repo(tmp_path)
    _landing(repo, subject)
    assert _skip(repo, tmp_path, 'push', PHASE) == 'false'


def test_a_merge_commit_runs_everything(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _git(repo, 'checkout', '-q', '-b', 'side', 'HEAD^')
    _git(repo, 'commit', '-q', '--allow-empty', '-m', 'T1.3: side')
    _git(repo, 'checkout', '-q', 'work')
    # A merge whose tree is its first parent's, with a landing's subject.
    _git(repo, 'merge', '-q', '-s', 'ours', '-m', 'Land T1.2: merged', 'side')
    assert _git(repo, 'rev-parse', 'HEAD^{tree}') == _git(
        repo, 'rev-parse', 'HEAD^^{tree}'
    )
    assert _skip(repo, tmp_path, 'push', PHASE) == 'false'


# --- every other job is left out on a landing commit -----------------------------


def test_every_job_is_left_out_when_the_landing_job_says_skip() -> None:
    r"""A job is left out by its own condition on the landing job's
    output, or by needing a job that is left out, with no status function
    in its condition to run it anyway. The landing job itself runs."""
    jobs = _workflow()['jobs']
    assert 'landing' in jobs and not _needs(jobs['landing'])
    left_out: set[str] = set()
    pending = [name for name in jobs if name != 'landing']
    while pending:
        progressed = False
        for name in list(pending):
            job = jobs[name]
            needs = set(_needs(job)) - {'landing'}
            if not needs <= left_out:
                continue
            condition = str(job.get('if', ''))
            if SKIP in condition:
                assert 'landing' in _needs(job), name
            else:
                assert needs, f'{name} is not left out on a landing commit'
                assert not re.search(
                    r'always\(\)|cancelled\(\)|failure\(\)', condition
                ), f'{name} runs on a landing commit: {condition}'
            left_out.add(name)
            pending.remove(name)
            progressed = True
        assert progressed, f'a cycle or an unknown job in {pending}'


def test_every_job_has_a_timeout() -> None:
    for name, job in _workflow()['jobs'].items():
        minutes = job.get('timeout-minutes')
        assert isinstance(minutes, int) and 0 < minutes <= 90, (name, minutes)


# --- the images -----------------------------------------------------------------


def test_each_image_is_named_once_and_pinned_by_digest() -> None:
    workflow = _workflow()
    env = workflow['env']
    for key in ('POSTGRES_IMAGE', 'MANYLINUX'):
        assert DIGEST.search(env[key]), env[key]
    text = CI.read_text(encoding='utf-8')
    for key in ('POSTGRES_IMAGE', 'MANYLINUX'):
        assert text.count(env[key]) == 1, key
    jobs = workflow['jobs']
    landing = jobs['landing']['outputs']['postgres-image']
    assert landing == '${{ env.POSTGRES_IMAGE }}'
    for name in ('test', 'engine'):
        service = jobs[name]['services']['postgres']
        assert service['image'] == '${{ needs.landing.outputs.postgres-image }}'
        assert 'landing' in _needs(jobs[name])
        for value in service['credentials'].values():
            assert "startsWith(env.POSTGRES_IMAGE, 'ghcr.io/')" in value


def test_every_image_ci_runs_is_pulled_with_retries_first() -> None:
    jobs = _workflow()['jobs']
    for name in ('native-wheel', 'native-test-ops', 'wheel-smoke'):
        runs = [str(s.get('run', '')) for s in jobs[name]['steps']]
        pulls = [i for i, r in enumerate(runs) if 'pull_image.sh' in r]
        runs_image = [
            i
            for i, r in enumerate(runs)
            if 'docker run' in r or 'wheel_smoke.py --lane' in r
        ]
        assert pulls and runs_image and pulls[0] < runs_image[0], name


def _fake_docker(tmp_path: Path, fail_pulls: int) -> tuple[Path, Path]:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    calls = tmp_path / 'calls'
    count = tmp_path / 'count'
    count.write_text('0')
    docker = bin_dir / 'docker'
    docker.write_text(
        '#!/usr/bin/env bash\n'
        f'echo "$*" >> {str(calls)!r}\n'
        'if [ "$1" = login ]; then cat > /dev/null; exit 0; fi\n'
        f'n=$(cat {str(count)!r}); echo $((n + 1)) > {str(count)!r}\n'
        f'[ "$n" -ge {fail_pulls} ]\n'
    )
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR)
    return bin_dir, calls


def _pull(
    tmp_path: Path, image: str, fail_pulls: int, **env: str
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    bin_dir, calls = _fake_docker(tmp_path, fail_pulls)
    done = subprocess.run(
        ['bash', str(PULL), image],
        env={
            'PATH': f'{bin_dir}{os.pathsep}{os.environ["PATH"]}',
            'PULL_DELAYS': '0 0 0',
            **env,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    lines = calls.read_text().splitlines() if calls.exists() else []
    return done, lines


IMAGE = 'public.ecr.aws/docker/library/postgres@sha256:' + 'a' * 64
GHCR = 'ghcr.io/example/postgres@sha256:' + 'b' * 64


def test_a_pull_refused_twice_succeeds_on_the_third_attempt(
    tmp_path: Path,
) -> None:
    done, calls = _pull(tmp_path, IMAGE, fail_pulls=2)
    assert done.returncode == 0, done.stderr
    assert calls == [f'pull --quiet {IMAGE}'] * 3


def test_a_pull_refused_every_time_fails_after_four_attempts(
    tmp_path: Path,
) -> None:
    done, calls = _pull(tmp_path, IMAGE, fail_pulls=99)
    assert done.returncode == 1
    assert len(calls) == 4 and 'no pull in 4 attempts' in done.stderr


@pytest.mark.parametrize(
    'image', ['postgres:17', 'ghcr.io/example/postgres@sha256:abc', '']
)
def test_an_image_not_pinned_by_digest_is_refused(
    tmp_path: Path, image: str
) -> None:
    done, calls = _pull(tmp_path, image, fail_pulls=0)
    assert done.returncode != 0 and calls == []


def test_only_ghcr_io_is_logged_in_to_and_the_token_is_not_printed(
    tmp_path: Path,
) -> None:
    secret = 'not-a-real-token'
    login = {'GITHUB_TOKEN': secret, 'GITHUB_ACTOR': 'me'}
    seen = {}
    for name, image, env in (
        ('ecr', IMAGE, login),
        ('ghcr', GHCR, login),
        ('anonymous', GHCR, {}),
    ):
        (tmp_path / name).mkdir()
        done, calls = _pull(tmp_path / name, image, 0, **env)
        assert done.returncode == 0, done.stderr
        assert secret not in done.stdout + done.stderr + '\n'.join(calls)
        seen[name] = calls
    assert seen['ecr'] == [f'pull --quiet {IMAGE}']
    assert seen['ghcr'] == [
        'login ghcr.io --username me --password-stdin',
        f'pull --quiet {GHCR}',
    ]
    assert seen['anonymous'] == [f'pull --quiet {GHCR}']
