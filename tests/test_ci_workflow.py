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


# --- CI4: the independent test author's audit ----------------------------------
#
# Each ``test_ci4_*`` test is held by ``contract/tests/ci4_gate.py``: one
# listed in ``ci4-pending.json`` must still fail, as when it was written.


def _ci4_gate() -> Any:
    import importlib.util
    import sys

    name = 'ci4_gate'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'contract/tests/ci4_gate.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


CI4 = _ci4_gate()


def _held(request: pytest.FixtureRequest, check: Any) -> None:
    CI4.held('workflow', f'py:{request.node.name}', check)


def _expression(condition: object) -> str:
    text = str(condition).strip()
    if text.startswith('${{') and text.endswith('}}'):
        text = text[3:-2]
    return ' '.join(text.split())


def test_ci4_the_pending_list_only_shrinks() -> None:
    assert CI4.only_shrinks() == []
    assert set(CI4.gating()['pending']) == {'ratchet', 'workflow'}


def test_ci4_the_landing_job_always_runs_and_hands_on_its_step_s_answer(
    request: pytest.FixtureRequest,
) -> None:
    def check() -> None:
        landing = _workflow()['jobs']['landing']
        # No condition and no needs: were it left out, every job would be.
        assert 'if' not in landing and not _needs(landing)
        assert landing['outputs']['skip'] == '${{ steps.landing.outputs.skip }}'
        checkout = landing['steps'][0]
        assert str(checkout['uses']).startswith('actions/checkout@')
        # The commit pushed and its parent, and no other ref.
        assert checkout['with'] == {'fetch-depth': 2}
        ids = [s.get('id') for s in landing['steps']]
        assert ids.count('landing') == 1

    _held(request, check)


def _clone(repo: Path, tmp_path: Path) -> Path:
    r"""The repository as actions/checkout leaves it: HEAD and its parent."""
    clone = tmp_path / 'clone'
    subprocess.run(
        ['git', 'clone', '-q', '--depth', '2', f'file://{repo}', str(clone)],
        env={**os.environ, **GIT_ENV},
        capture_output=True,
        check=True,
    )
    return clone


@pytest.mark.parametrize('empty', [True, False])
def test_ci4_the_step_decides_alike_in_a_clone_of_depth_two(
    tmp_path: Path, request: pytest.FixtureRequest, empty: bool
) -> None:
    def check() -> None:
        repo = _repo(tmp_path)
        if empty:
            _landing(repo)
        else:
            (repo / 'b.txt').write_text('changed at landing\n')
            _git(repo, 'commit', '-q', '-am', 'Land T1.2: the candidate')
        clone = _clone(repo, tmp_path)
        assert _git(clone, 'rev-parse', '--is-shallow-repository') == 'true'
        wanted = 'true' if empty else 'false'
        assert _skip(clone, tmp_path, 'push', PHASE) == wanted

    _held(request, check)


@pytest.mark.parametrize(
    'event',
    [
        'pull_request',
        'pull_request_target',
        'workflow_dispatch',
        'merge_group',
        'schedule',
        '',
    ],
)
def test_ci4_an_empty_landing_commit_runs_everything_but_on_a_push(
    tmp_path: Path, request: pytest.FixtureRequest, event: str
) -> None:
    def check() -> None:
        repo = _repo(tmp_path)
        _landing(repo)
        for ref in (PHASE, LANE):
            assert _skip(repo, tmp_path, event, ref) == 'false', ref

    _held(request, check)


@pytest.mark.parametrize(
    'ref',
    [
        'rust-core-phase-10',
        'rust-core-phase-1x',
        'rust-core-phase-2',
        'rust-core-phase-2-lane-a',
        'xrust-core-phase-1',
        'rust-core-phase-',
        'refs/heads/rust-core-phase-1',
    ],
)
def test_ci4_an_empty_landing_commit_runs_everything_off_the_phase_branch(
    tmp_path: Path, request: pytest.FixtureRequest, ref: str
) -> None:
    def check() -> None:
        repo = _repo(tmp_path)
        _landing(repo)
        assert _skip(repo, tmp_path, 'push', ref) == 'false'

    _held(request, check)


def test_ci4_a_root_commit_titled_land_runs_everything(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        repo = tmp_path / 'repo'
        repo.mkdir()
        _git(repo, 'init', '-q', '-b', 'work')
        _git(repo, 'commit', '-q', '--allow-empty', '-m', 'Land T1.1: root')
        assert _skip(repo, tmp_path, 'push', PHASE) == 'false'

    _held(request, check)


def test_ci4_a_skip_condition_cannot_be_overridden(
    request: pytest.FixtureRequest,
) -> None:
    r"""A job's own condition on the landing job's answer is one conjunct
    of its condition: nothing joined with ``||`` can run it anyway."""

    def check() -> None:
        for name, job in _workflow()['jobs'].items():
            condition = _expression(job.get('if', ''))
            if SKIP not in condition:
                continue
            assert '||' not in condition, (name, condition)
            parts = [p.strip() for p in condition.split('&&')]
            assert SKIP in parts, (name, condition)
            assert all(p in (SKIP, '!cancelled()') for p in parts), (
                name,
                condition,
            )

    _held(request, check)


def _images() -> list[str]:
    r"""Every registry image reference written in the workflow."""
    text = CI.read_text(encoding='utf-8')
    return re.findall(
        r'(?<![\w./-])(?:[a-z0-9-]+\.)+[a-z]{2,}(?::\d+)?/[a-z0-9._/-]+'
        r'(?::[\w.-]+)?(?:@sha256:[0-9a-f]+)?',
        text,
    )


def test_ci4_every_image_ci_names_is_pinned_by_digest(
    request: pytest.FixtureRequest,
) -> None:
    def check() -> None:
        workflow = _workflow()
        images = [
            i
            for i in _images()
            if not i.startswith(('github.com/', 'docs.github.com/'))
        ]
        assert images, 'no image found'
        for image in images:
            assert DIGEST.search(image), image
        env = workflow['env']
        named = {env['POSTGRES_IMAGE'], env['MANYLINUX']}
        assert set(images) == named, set(images) ^ named
        for name, job in workflow['jobs'].items():
            assert 'container' not in job, name
            for service in (job.get('services') or {}).values():
                assert service['image'] == (
                    '${{ needs.landing.outputs.postgres-image }}'
                ), name
        # wheel-smoke's images, as wheel_smoke.py names them, by lane.
        lanes = workflow['jobs']['wheel-smoke']['strategy']['matrix']['lane']
        assert lanes
        for lane in lanes:
            done = subprocess.run(
                [
                    'python3',
                    str(ROOT / 'scripts/rust_core/wheel_smoke.py'),
                    'image',
                    lane,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            assert DIGEST.search(done.stdout.strip()), (lane, done.stdout)

    _held(request, check)


def test_ci4_every_container_ci_runs_is_pulled_first_by_pull_image_sh(
    request: pytest.FixtureRequest,
) -> None:
    r"""In every job, each step that runs a container comes after a
    ``pull_image.sh`` step pulling that same image, and nothing pulls by
    other means."""

    def check() -> None:
        text = CI.read_text(encoding='utf-8')
        assert 'docker pull' not in text
        assert '--pull' not in text
        seen = 0
        for name, job in _workflow()['jobs'].items():
            pulls: list[str] = []
            for step in job.get('steps', []):
                run = str(step.get('run', ''))
                if 'pull_image.sh' in run:
                    assert run.strip().startswith(
                        'scripts/rust_core/pull_image.sh '
                    ), (name, run)
                    assert '\n' not in run.strip(), (name, run)
                    pulls.append(run.strip().split(' ', 1)[1])
                if 'docker run' in run:
                    seen += 1
                    assert '"$MANYLINUX"' in run, (name, run)
                    assert '"$MANYLINUX"' in pulls, (name, pulls)
                if '--container' in run and 'wheel_smoke.py' in run:
                    seen += 1
                    lane = '--lane ${{ matrix.lane }}'
                    assert lane in run, (name, run)
                    assert (
                        '"$(python3 scripts/rust_core/wheel_smoke.py image '
                        '${{ matrix.lane }})"' in pulls
                    ), (name, pulls)
        assert seen >= 3, seen

    _held(request, check)


def test_ci4_the_token_goes_only_to_ghcr_io(
    request: pytest.FixtureRequest,
) -> None:
    r"""GITHUB_TOKEN reaches only pull_image.sh, which logs in to ghcr.io
    alone, and the service credentials, which are empty unless the image
    is on ghcr.io; no other secret is used, and nothing wider than
    reading packages is granted."""

    def check() -> None:
        workflow = _workflow()
        assert workflow['permissions'] == {
            'contents': 'read',
            'packages': 'read',
        }
        assert 'secrets.' not in str(workflow.get('env'))
        text = CI.read_text(encoding='utf-8')
        found = 0
        for name, job in workflow['jobs'].items():
            assert 'secrets.' not in str(job.get('env')), name
            assert 'permissions' not in job, name
            for step in job.get('steps', []):
                env = step.get('env') or {}
                with_ = step.get('with') or {}
                if 'secrets.' in str(with_):
                    raise AssertionError((name, step))
                for key, value in env.items():
                    if 'secrets.' not in str(value):
                        continue
                    found += 1
                    assert (key, value) == (
                        'GITHUB_TOKEN',
                        '${{ secrets.GITHUB_TOKEN }}',
                    ), (name, key)
                    assert (
                        str(step['run'])
                        .strip()
                        .startswith('scripts/rust_core/pull_image.sh ')
                    ), (name, step)
            for service in (job.get('services') or {}).values():
                credentials = service.get('credentials')
                if credentials is None:
                    continue
                found += 2
                ghcr = "startsWith(env.POSTGRES_IMAGE, 'ghcr.io/')"
                assert credentials == {
                    'username': f"${{{{ {ghcr} && github.actor || '' }}}}",
                    'password': (
                        f"${{{{ {ghcr} && secrets.GITHUB_TOKEN || '' }}}}"
                    ),
                }, (name, credentials)
        assert found >= 4, found
        # Every secret written in the file is one of those (YAML anchors
        # write a step or a service once for several jobs).
        allowed = {
            'GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}',
            "password: ${{ startsWith(env.POSTGRES_IMAGE, 'ghcr.io/') && "
            "secrets.GITHUB_TOKEN || '' }}",
        }
        for line in text.splitlines():
            if 'secrets' in line and not line.strip().startswith('#'):
                assert line.strip() in allowed, line

    _held(request, check)


@pytest.mark.parametrize(
    'image',
    [
        'ghcr.io.example.com/x/postgres@sha256:' + 'c' * 64,
        'example.com/ghcr.io/postgres@sha256:' + 'c' * 64,
        'public.ecr.aws/ghcr.io/postgres@sha256:' + 'c' * 64,
    ],
)
def test_ci4_a_registry_that_only_looks_like_ghcr_io_gets_no_token(
    tmp_path: Path, request: pytest.FixtureRequest, image: str
) -> None:
    def check() -> None:
        secret = 'not-a-real-token'
        done, calls = _pull(
            tmp_path, image, 0, GITHUB_TOKEN=secret, GITHUB_ACTOR='me'
        )
        assert done.returncode == 0, done.stderr
        assert calls == [f'pull --quiet {image}']

    _held(request, check)
