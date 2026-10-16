# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI's workflow under decision E131 (``.github/workflows/ci.yml``).

The landing job's step, its text as CI runs it (``bash -e``), against real
temporary git repositories: it says skip for an empty landing commit pushed
to a phase branch (``rust-core-phase-N``, any N; T2.0) or a branch under
it, and for nothing else. Every other
job is left out when it says skip, directly or through the jobs it needs.
Every job has a timeout. Each image is named once and pinned by digest,
and ``scripts/rust_core/pull_image.sh`` retries a pull, refuses an image
not pinned by digest, and logs in to ghcr.io alone.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
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


#: The GitHub API as the skip step reaches it: ``gh`` (``gh run list`` or
#: ``gh api``) and ``curl``, faked on PATH, answering from a file of runs.
#: A run there: ``sha``, ``id``, ``branch``, ``event``, ``status``,
#: ``conclusion``, ``created``, ``workflow`` (the file, ``ci.yml``).
FAKE_API = r"""
import json, os, re, subprocess, sys
from urllib.parse import parse_qsl, urlparse

WORKFLOWS = {'ci.yml': (1, 'CI'), 'publish.yml': (2, 'Publish')}


def log(argv):
    with open(os.environ['FAKE_CALLS'], 'a') as handle:
        handle.write(json.dumps(argv) + '\n')


def runs(sha=None, event=None, branch=None, status=None, workflow=None):
    with open(os.environ['FAKE_RUNS']) as handle:
        every = json.load(handle)
    found = []
    for run in every:
        number, name = WORKFLOWS[run['workflow']]
        if sha and run['sha'] != sha:
            continue
        if event and run['event'] != event:
            continue
        if branch and run['branch'] != branch:
            continue
        if workflow and workflow not in (
            run['workflow'], '.github/workflows/' + run['workflow'],
            name, str(number),
        ):
            continue
        if status and status not in (run['status'], run['conclusion']):
            continue
        found.append(run)
    return sorted(found, key=lambda run: run['created'], reverse=True)


def rest(run):
    number, name = WORKFLOWS[run['workflow']]
    done = run['status'] == 'completed'
    return {
        'id': run['id'], 'run_number': run['id'], 'run_attempt': 1,
        'head_sha': run['sha'], 'head_branch': run['branch'],
        'event': run['event'], 'status': run['status'],
        'conclusion': run['conclusion'] if done else None,
        'created_at': run['created'], 'updated_at': run['created'],
        'run_started_at': run['created'],
        'path': '.github/workflows/' + run['workflow'], 'name': name,
        'workflow_id': number, 'display_title': 'a commit',
        'html_url': 'https://github.com/o/r/actions/runs/%d' % run['id'],
        'url': 'https://api.github.com/repos/o/r/actions/runs/%d' % run['id'],
    }


def cli(run):
    number, name = WORKFLOWS[run['workflow']]
    done = run['status'] == 'completed'
    return {
        'databaseId': run['id'], 'number': run['id'], 'attempt': 1,
        'headSha': run['sha'], 'headBranch': run['branch'],
        'event': run['event'], 'status': run['status'],
        'conclusion': run['conclusion'] if done else '',
        'createdAt': run['created'], 'updatedAt': run['created'],
        'startedAt': run['created'], 'name': name, 'workflowName': name,
        'workflowDatabaseId': number, 'displayTitle': 'a commit',
        'url': 'https://github.com/o/r/actions/runs/%d' % run['id'],
    }


def answer(path, query):
    match = re.search(r'/actions/(?:workflows/([^/]+)/)?runs/?$', path)
    if not match:
        return None
    found = runs(
        sha=query.get('head_sha'), event=query.get('event'),
        branch=query.get('branch'), status=query.get('status'),
        workflow=match.group(1),
    )
    return {'total_count': len(found), 'workflow_runs': [rest(r) for r in found]}


def emit(value, jq=None):
    text = json.dumps(value)
    if jq is None:
        sys.stdout.write(text + '\n')
        return 0
    done = subprocess.run(
        ['jq', '-r', jq], input=text, capture_output=True, text=True
    )
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    return done.returncode


def options(argv, valued):
    flags, rest_ = {}, []
    i = 0
    while i < len(argv):
        arg = argv[i]
        key, eq, value = arg.partition('=')
        if arg.startswith('-') and eq and key in valued:
            flags.setdefault(key, []).append(value)
        elif arg in valued and i + 1 < len(argv):
            flags.setdefault(arg, []).append(argv[i + 1])
            i += 1
        elif arg.startswith('-'):
            flags.setdefault(arg, []).append('')
        else:
            rest_.append(arg)
        i += 1
    return flags, rest_


def first(flags, *names):
    for name in names:
        if flags.get(name):
            return flags[name][-1]
    return None


def gh(argv):
    if os.environ.get('FAKE_API_FAILS'):
        sys.stderr.write('HTTP 502: Bad Gateway\n')
        return 1
    if argv[:2] == ['run', 'list']:
        flags, _ = options(argv[2:], {
            '--commit', '-c', '--branch', '-b', '--event', '-e', '--status',
            '-s', '--workflow', '-w', '--limit', '-L', '--json', '--jq',
            '-q', '--repo', '-R', '--user', '-u', '--created', '--template',
            '-t',
        })
        found = runs(
            sha=first(flags, '--commit', '-c'),
            event=first(flags, '--event', '-e'),
            branch=first(flags, '--branch', '-b'),
            status=first(flags, '--status', '-s'),
            workflow=first(flags, '--workflow', '-w'),
        )
        limit = first(flags, '--limit', '-L')
        found = [cli(r) for r in found][: int(limit) if limit else 20]
        fields = first(flags, '--json')
        if fields:
            keep = fields.split(',')
            found = [{k: r[k] for k in keep if k in r} for r in found]
        return emit(found, first(flags, '--jq', '-q'))
    if argv[:1] == ['api']:
        flags, rest_ = options(argv[1:], {
            '-X', '--method', '-f', '-F', '--raw-field', '--field', '-H',
            '--header', '--jq', '-q', '--hostname', '--template', '-t',
            '--cache', '--input',
        })
        url = urlparse(rest_[0])
        query = dict(parse_qsl(url.query))
        for name in ('-f', '-F', '--raw-field', '--field'):
            for field in flags.get(name, []):
                key, _, value = field.partition('=')
                query[key] = value
        found = answer(url.path, query)
        if found is None:
            sys.stderr.write('gh: Not Found (HTTP 404)\n')
            return 1
        return emit(found, first(flags, '--jq', '-q'))
    sys.stderr.write('fake gh: unsupported %r\n' % argv)
    return 1


def curl(argv):
    flags, rest_ = options(argv, {
        '-H', '--header', '-o', '--output', '-w', '--write-out', '-X',
        '--request', '-d', '--data', '--data-urlencode', '-u', '--user',
        '--retry', '--retry-delay', '--retry-max-time', '-m', '--max-time',
        '--connect-timeout', '-A', '--user-agent', '--url',
    })
    urls = [a for a in rest_ if re.match(r'https?://', a)] + flags.get('--url', [])
    fail = any(f in flags for f in ('-f', '--fail', '--fail-with-body', '-sf', '-fs', '-fsS', '-sSf', '-fsSL', '-sSfL'))
    url = urlparse(urls[0])
    api = os.environ.get('GITHUB_API_URL', 'https://api.github.com')
    if url.netloc != urlparse(api).netloc:
        sys.stderr.write('curl: (6) Could not resolve host: %s\n' % url.netloc)
        return 6
    query = dict(parse_qsl(url.query))
    if '-G' in flags or '--get' in flags:
        for name in ('-d', '--data', '--data-urlencode'):
            for field in flags.get(name, []):
                key, _, value = field.partition('=')
                query[key] = value
    if os.environ.get('FAKE_API_FAILS'):
        found, code = {'message': 'Bad Gateway'}, 502
    else:
        found = answer(url.path, query)
        code = 200 if found is not None else 404
        if found is None:
            found = {'message': 'Not Found'}
    if code >= 400 and fail:
        sys.stderr.write('curl: (22) The requested URL returned error: %d\n' % code)
        return 22
    body = json.dumps(found)
    target = first(flags, '-o', '--output')
    if target and target != '-':
        with open(target, 'w') as handle:
            handle.write(body)
        body = ''
    write = first(flags, '-w', '--write-out')
    if write:
        body += write.replace('%{http_code}', str(code)).replace('\\n', '\n')
    sys.stdout.write(body)
    return 0


argv = sys.argv[1:]
log([os.path.basename(sys.argv[0])] + argv)
sys.exit(gh(argv) if os.path.basename(sys.argv[0]) == 'gh' else curl(argv))
"""


def _run(
    sha: str,
    run_id: int,
    branch: str = LANE,
    event: str = 'push',
    status: str = 'completed',
    conclusion: str = 'success',
    created: str = '2026-10-14T00:00:00Z',
    workflow: str = 'ci.yml',
) -> dict[str, Any]:
    return {
        'sha': sha,
        'id': run_id,
        'branch': branch,
        'event': event,
        'status': status,
        'conclusion': conclusion if status == 'completed' else None,
        'created': created,
        'workflow': workflow,
    }


def _decide(
    repo: Path,
    tmp_path: Path,
    event: str,
    ref: str,
    runs: list[dict[str, Any]],
    api_fails: bool = False,
) -> tuple[str, list[list[str]]]:
    r"""The skip step's answer, ``true`` or ``false``, or ``failed`` when
    the step itself fails (closed), with the GitHub API faked to know
    ``runs``; and the calls the fakes took."""
    here = tmp_path / f'api-{len(list(tmp_path.glob("api-*")))}'
    bin_dir = here / 'bin'
    bin_dir.mkdir(parents=True)
    for name in ('gh', 'curl'):
        tool = bin_dir / name
        tool.write_text(f'#!{sys.executable}\n{FAKE_API}')
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    (here / 'runs.json').write_text(json.dumps(runs))
    calls = here / 'calls'
    calls.write_text('')
    output = here / 'output'
    output.write_text('')
    head = _git(repo, 'rev-parse', 'HEAD')
    event_file = here / 'event.json'
    event_file.write_text(
        json.dumps({'after': head, 'ref': f'refs/heads/{ref}'})
    )
    nowhere = 'http://127.0.0.1:9'
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(('GITHUB_', 'GH_', 'RUNNER_'))
    }
    env.update(
        {
            **GIT_ENV,
            'PATH': f'{bin_dir}{os.pathsep}{os.environ["PATH"]}',
            'HOME': str(here),
            'GH_CONFIG_DIR': str(here / 'gh'),
            'GH_TOKEN': 'fake-token',
            'GITHUB_TOKEN': 'fake-token',
            'GITHUB_ACTIONS': 'true',
            'GITHUB_API_URL': 'https://api.github.com',
            'GITHUB_SERVER_URL': 'https://github.com',
            'GITHUB_REPOSITORY': 'o/r',
            'GITHUB_REPOSITORY_OWNER': 'o',
            'GITHUB_WORKFLOW_REF': f'o/r/.github/workflows/ci.yml@refs/heads/{ref}',
            'GITHUB_RUN_ID': '999',
            'GITHUB_SHA': head,
            'GITHUB_REF': f'refs/heads/{ref}',
            'GITHUB_EVENT_NAME': event,
            'GITHUB_REF_NAME': ref,
            'GITHUB_EVENT_PATH': str(event_file),
            'GITHUB_OUTPUT': str(output),
            'FAKE_RUNS': str(here / 'runs.json'),
            'FAKE_CALLS': str(calls),
            # Nothing reaches the network but through the fakes.
            'HTTPS_PROXY': nowhere,
            'HTTP_PROXY': nowhere,
            'https_proxy': nowhere,
            'http_proxy': nowhere,
            'ALL_PROXY': nowhere,
        }
    )
    if api_fails:
        env['FAKE_API_FAILS'] = '1'
    done = subprocess.run(
        ['bash', '--noprofile', '--norc', '-eo', 'pipefail', '-c',
         _landing_step()['run']],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )  # fmt: skip
    called = [json.loads(x) for x in calls.read_text().splitlines()]
    if done.returncode != 0:
        return 'failed', called
    lines = [
        x for x in output.read_text().splitlines() if x.startswith('skip=')
    ]
    assert len(lines) == 1, (output.read_text(), done.stdout, done.stderr)
    return lines[0].removeprefix('skip='), called


def _skip(repo: Path, tmp_path: Path, event: str, ref: str) -> str:
    r"""The skip step's answer when every commit has a green push run on
    the lane: the landing's shape alone decides (the evidence cases below
    seed their runs one by one)."""
    seeded = [
        _run(sha, 100 + n)
        for n, sha in enumerate(_git(repo, 'rev-list', 'HEAD').split())
    ]
    found, _ = _decide(repo, tmp_path, event, ref, seeded)
    assert found in ('true', 'false'), found
    return found


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


#: The longest a job may run, in minutes: rust-test took 80.7 at most
#: (fix round 1, P2-5); the landing job, which every other job waits on,
#: does no more than read a few commits.
TIMEOUT_CEILINGS = {'landing': 10, 'rust-test': 120}
TIMEOUT_CEILING = 60


def test_every_job_has_a_timeout() -> None:
    jobs = _workflow()['jobs']
    assert set(TIMEOUT_CEILINGS) <= set(jobs)
    for name, job in jobs.items():
        minutes = job.get('timeout-minutes')
        ceiling = TIMEOUT_CEILINGS.get(name, TIMEOUT_CEILING)
        assert isinstance(minutes, int) and 0 < minutes <= ceiling, (
            name,
            minutes,
            ceiling,
        )


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
        # GitHub refuses a credential that evaluates to an empty string:
        # credentials come, unconditional, with an image on ghcr.io.
        if env['POSTGRES_IMAGE'].startswith('ghcr.io/'):
            assert service['credentials'] == {
                'username': '${{ github.actor }}',
                'password': '${{ secrets.GITHUB_TOKEN }}',
            }, name
        else:
            assert 'credentials' not in service, name


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
        # Every phase's branches are phase branches (T2.0): what is not one.
        'rust-core-phase-1x',
        'rust-core-phase-01',
        'rust-core-phase-2x-lane-a',
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


@pytest.mark.parametrize('ref', [PHASE, LANE])
def test_ci4_an_empty_moved_landing_commit_runs_everything(
    tmp_path: Path, request: pytest.FixtureRequest, ref: str
) -> None:
    r"""A landing carrying ``Moved-from`` (CI3's move) is judged by its
    own runs against its own Base-SHA (fix round 1, P2-6): its candidate
    never ran, and the commit it was moved from was judged against an
    older checkpoint."""

    def check() -> None:
        repo = _repo(tmp_path)
        _git(
            repo, 'commit', '-q', '--allow-empty', '-m',
            'Land T1.2: the candidate\n\nBase-SHA: x\nCheckpoint-SHA: x\n'
            'Moved-from: ' + 'a' * 40 + '\nMoved-from-run: 4242',
        )  # fmt: skip
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


def _service_image(workflow: dict[str, Any], service: dict[str, Any]) -> str:
    r"""The image a service runs: POSTGRES_IMAGE when it takes the landing
    job's output, which must be that env value, or its literal image."""
    image = str(service['image'])
    if image == '${{ needs.landing.outputs.postgres-image }}':
        outputs = workflow['jobs']['landing']['outputs']
        assert outputs['postgres-image'] == '${{ env.POSTGRES_IMAGE }}'
        return str(workflow['env']['POSTGRES_IMAGE'])
    assert '${{' not in image, image
    return image


#: The only credentials a service takes: the workflow's own, for ghcr.io.
#: GitHub refuses a service credential that evaluates to an empty string
#: (run 38659551659), so they are unconditional, and only on ghcr.io.
GHCR_CREDENTIALS = {
    'username': '${{ github.actor }}',
    'password': '${{ secrets.GITHUB_TOKEN }}',
}


#: A secret or the workflow's token, however the workflow names it.
TOKEN = r'secrets\.|github\.token'
#: How the skip step may hold the token, for ``gh`` or ``curl``.
API_TOKENS = [
    (key, value)
    for key in ('GH_TOKEN', 'GITHUB_TOKEN')
    for value in ('${{ secrets.GITHUB_TOKEN }}', '${{ github.token }}')
]


def _api_only(key: str, value: object, run: str) -> None:
    r"""The skip step's token goes to the GitHub API alone: held as gh
    or curl read it, its text runs no container, logs in nowhere, and
    names no host but the API's."""
    assert (key, value) in API_TOKENS, (key, value)
    assert 'docker' not in run and 'login' not in run, run
    for host in re.findall(r'https?://([^/\s"\')]+)', run):
        assert host == 'api.github.com', host


def test_ci4_the_token_goes_only_to_ghcr_io(
    request: pytest.FixtureRequest,
) -> None:
    r"""GITHUB_TOKEN reaches only pull_image.sh, which logs in to ghcr.io
    alone, and the credentials of a service whose image is on ghcr.io,
    which are exactly ``github.actor`` and ``secrets.GITHUB_TOKEN``; a
    service whose image is elsewhere has no credentials; and the landing
    job's skip step, which reads the candidate's runs from the GitHub API
    (round 4) and runs no container. No other secret is used, and nothing
    wider than reading packages and Actions runs is granted."""

    def check() -> None:
        workflow = _workflow()
        assert workflow['permissions'] in (
            {'contents': 'read', 'packages': 'read'},
            {'contents': 'read', 'packages': 'read', 'actions': 'read'},
        )
        assert not re.search(TOKEN, str(workflow.get('env')))
        steps = services = 0
        for name, job in workflow['jobs'].items():
            assert not re.search(TOKEN, str(job.get('env'))), name
            assert 'permissions' not in job, name
            assert 'container' not in job, name
            for step in job.get('steps', []):
                assert not re.search(TOKEN, str(step.get('with'))), (name, step)
                assert not re.search(TOKEN, str(step.get('run'))), (name, step)
                for key, value in (step.get('env') or {}).items():
                    if not re.search(TOKEN, str(value)):
                        continue
                    if name == 'landing' and step.get('id') == 'landing':
                        _api_only(key, value, str(step['run']))
                        continue
                    steps += 1
                    assert (key, value) == (
                        'GITHUB_TOKEN',
                        '${{ secrets.GITHUB_TOKEN }}',
                    ), (name, key)
                    run = str(step['run']).strip()
                    assert run.startswith('scripts/rust_core/pull_image.sh ')
                    assert '\n' not in run, (name, run)
            for service in (job.get('services') or {}).values():
                services += 1
                image = _service_image(workflow, service)
                if image.startswith('ghcr.io/'):
                    assert service.get('credentials') == GHCR_CREDENTIALS, (
                        name,
                        image,
                        service.get('credentials'),
                    )
                else:
                    assert 'credentials' not in service, (name, image)
        assert steps >= 3 and services >= 2, (steps, services)
        # Every secret written in the file is one of those (YAML anchors
        # write a step or a service once for several jobs).
        allowed = {
            'GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}',
            'password: ${{ secrets.GITHUB_TOKEN }}',
        } | {f'{key}: {value}' for key, value in API_TOKENS}
        for line in CI.read_text(encoding='utf-8').splitlines():
            if re.search(TOKEN, line) and not line.strip().startswith('#'):
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


# --- CI4: the landing job's phase-push check ----------------------------------
#
# Fix round 1 (P2-3): on every push to the phase branch the landing job
# runs ``ratchet.py checkpoint --phase-push <previous tip>``, so a push that
# does not land exactly one package as a fast-forward, its landing commit's
# Base-SHA the previous tip, turns the run red even when every other job is
# left out. The step's text runs here as CI runs it (``bash -eo pipefail``),
# in a clone of depth 2 of a real temporary origin, with the event file a
# push writes; its condition is evaluated for each event and branch.


def _phase_check_step() -> tuple[int, dict[str, Any]]:
    steps = _workflow()['jobs']['landing']['steps']
    found = [
        (i, s)
        for i, s in enumerate(steps)
        if 'ratchet.py checkpoint' in str(s.get('run', ''))
    ]
    assert len(found) == 1, found
    return found[0]


_TOKEN = re.compile(
    r"\s*(?:(?P<str>'(?:[^']|'')*')|(?P<op>==|!=|&&|\|\||!|\(|\)|,)"
    r'|(?P<name>[A-Za-z_][\w.-]*))'
)


def _evaluate(condition: str, context: dict[str, str]) -> bool:
    r"""A GitHub Actions ``if:`` of the forms CI uses: contexts, strings,
    ``==``, ``!=``, ``!``, ``&&``, ``||``, parentheses, ``startsWith``,
    ``endsWith``, ``contains``, ``fromJSON``, ``format`` and the status
    functions,
    every earlier
    step having succeeded. With no status function, ``success()`` is
    implied, as GitHub implies it."""
    text = _expression(condition)
    tokens: list[tuple[str, str]] = []
    at = 0
    while at < len(text):
        match = _TOKEN.match(text, at)
        assert match and match.end() > at, f'cannot read {text[at:]!r}'
        kind = match.lastgroup
        assert kind is not None
        tokens.append((kind, match.group(kind)))
        at = match.end()
        while at < len(text) and text[at] == ' ':
            at += 1
    status = {
        'success': True,
        'always': True,
        'failure': False,
        'cancelled': False,
    }

    def contains(a: Any, b: Any) -> bool:
        # An array holds the item, a string the substring (GitHub's).
        if isinstance(a, list):
            return str(b).lower() in [str(x).lower() for x in a]
        return str(b).lower() in str(a).lower()

    functions = {
        'startsWith': lambda a, b: str(a).lower().startswith(str(b).lower()),
        'endsWith': lambda a, b: str(a).lower().endswith(str(b).lower()),
        'contains': contains,
        'fromJSON': lambda a: json.loads(str(a)),
        'format': lambda text, *args: re.sub(
            r'\{(\d+)\}', lambda m: str(args[int(m[1])]), str(text)
        ),
    }
    pos = 0

    def peek() -> tuple[str, str] | None:
        return tokens[pos] if pos < len(tokens) else None

    def take(value: str | None = None) -> tuple[str, str]:
        nonlocal pos
        token = tokens[pos]
        assert value is None or token[1] == value, (token, value)
        pos += 1
        return token

    def primary() -> Any:
        kind, value = take()
        if value == '(':
            found = disjunction()
            take(')')
            return found
        if value == '!':
            return not truthy(primary())
        if kind == 'str':
            return value[1:-1].replace("''", "'")
        assert kind == 'name', value
        if peek() == ('op', '('):
            take('(')
            args = []
            while peek() != ('op', ')'):
                args.append(disjunction())
                if peek() == ('op', ','):
                    take(',')
            take(')')
            if value in status:
                return status[value]
            return functions[value](*args)
        if value in ('true', 'false'):
            return value == 'true'
        assert value in context, f'unknown context {value}'
        return context[value]

    def comparison() -> Any:
        left = primary()
        while peek() in (('op', '=='), ('op', '!=')):
            op = take()[1]
            right = primary()
            same = str(left).lower() == str(right).lower()
            left = same if op == '==' else not same
        return left

    def truthy(value: Any) -> bool:
        return bool(value) and value != 'false'

    def conjunction() -> Any:
        left = comparison()
        while peek() == ('op', '&&'):
            take()
            right = comparison()
            left = truthy(left) and truthy(right)
        return left

    def disjunction() -> Any:
        left = conjunction()
        while peek() == ('op', '||'):
            take()
            right = conjunction()
            left = truthy(left) or truthy(right)
        return left

    found = truthy(disjunction())
    assert pos == len(tokens), tokens[pos:]
    return found


@pytest.mark.parametrize(
    ('event', 'ref', 'runs'),
    [
        ('push', PHASE, True),
        ('push', LANE, False),
        ('push', f'{PHASE}-t1.99', False),
        ('push', f'{PHASE}-archive-lane-a-t1.99', False),
        # Every phase alike (T2.0), multi-digit ones included.
        ('push', 'rust-core-phase-2', True),
        ('push', 'rust-core-phase-10', True),
        ('push', 'rust-core-phase-123', True),
        ('push', 'rust-core-phase-2-lane-a', False),
        ('push', 'rust-core-phase-10-lane-a', False),
        ('push', 'rust-core-phase-10-t10.1', False),
        ('push', 'rust-core-phase-10-archive-a-t10.1-abc1234', False),
        # Names under the prefix that are no phase branch (fix 2).
        ('push', 'rust-core-phase-', False),
        ('push', 'rust-core-phase-two', False),
        ('push', 'rust-core-phase-10x', False),
        ('push', 'rust-core-phase-experiment', False),
        ('push', 'main', False),
        ('push', 'rust-core', False),
        ('pull_request', PHASE, False),
        ('pull_request', '7/merge', False),
        ('pull_request_target', PHASE, False),
        ('workflow_dispatch', PHASE, False),
        ('merge_group', PHASE, False),
    ],
)
def test_ci4_the_phase_check_runs_on_every_phase_push_and_nowhere_else(
    request: pytest.FixtureRequest, event: str, ref: str, runs: bool
) -> None:
    def check() -> None:
        _, step = _phase_check_step()
        condition = str(step.get('if', ''))
        # Whatever the skip step said: a landing commit's push is checked.
        assert 'skip' not in condition and 'steps.' not in condition
        context = {
            'github.event_name': event,
            'github.ref_name': ref,
            'github.ref': f'refs/heads/{ref}',
            'github.base_ref': PHASE if event.startswith('pull') else '',
            'github.head_ref': LANE if event.startswith('pull') else '',
        }
        assert _evaluate(condition, context) is runs, condition

    _held(request, check)


def test_ci4_a_failing_phase_check_turns_the_run_red(
    request: pytest.FixtureRequest,
) -> None:
    r"""The step's failure is the landing job's, which nothing runs past:
    the job always runs (no condition), nothing in it may fail quietly,
    and the check comes after a checkout of the pushed commit."""

    def check() -> None:
        landing = _workflow()['jobs']['landing']
        assert 'if' not in landing and not _needs(landing)
        assert not landing.get('continue-on-error'), landing
        at, step = _phase_check_step()
        assert not step.get('continue-on-error'), step
        assert 'shell' not in step or step['shell'] == 'bash', step
        steps = landing['steps']
        assert str(steps[0].get('uses', '')).startswith('actions/checkout@')
        assert 'ref' not in (steps[0].get('with') or {})
        assert at > 0
        for earlier in steps[:at]:
            assert not earlier.get('continue-on-error'), earlier
        # The step runs the checkout's own ratchet.py and its phase-push
        # check, never with a status or exit code swallowed.
        run = str(step['run'])
        assert 'scripts/rust_core/ratchet.py checkpoint' in run
        assert 'scripts/rust_core/ratchet.py where' in run
        assert '|| true' not in run and 'set +e' not in run
        assert not re.search(r'\|\|\s*(true|:|exit 0)', run), run

    _held(request, check)


#: A landing commit's trailers, as land.py writes them.
def _landing_message(subject: str, base: str, moved: str | None = None) -> str:
    lines = [
        subject,
        '',
        'Reviewed-by: independent reviewer and Codex',
        'Protected-paths: none',
        f'Base-SHA: {base}',
        f'Checkpoint-SHA: {base}',
    ]
    if moved:
        lines.append(f'Moved-from: {moved}')
    return '\n'.join(lines)


class _Phase:
    r"""A bare origin whose phase branch starts at a landing carrying
    ratchet.py, and a clone to push from."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.origin = tmp_path / 'origin.git'
        self.work = tmp_path / 'work'
        _git(tmp_path, 'init', '-q', '--bare', str(self.origin))
        _git(tmp_path, 'clone', '-q', str(self.origin), str(self.work))
        _git(self.work, 'checkout', '-q', '-b', PHASE)
        # ratchet.py and the branch names it reads (phases.py, T2.0).
        for name in ('ratchet.py', 'phases.py'):
            script = self.work / 'scripts/rust_core' / name
            script.parent.mkdir(parents=True, exist_ok=True)
            script.write_bytes((ROOT / 'scripts/rust_core' / name).read_bytes())
        _git(self.work, 'add', '-A')
        _git(
            self.work, 'commit', '-q', '-m',
            _landing_message('Land T1.13: the start', 'f' * 40),
        )  # fmt: skip
        self.l0 = self.push()

    def package(self, name: str) -> str:
        (self.work / f'{name}.txt').write_text(f'{name}\n')
        _git(self.work, 'add', '-A')
        _git(self.work, 'commit', '-q', '-m', f'{name}: the package')
        return _git(self.work, 'rev-parse', 'HEAD')

    def landing(self, name: str, base: str, moved: str | None = None) -> str:
        message = _landing_message(f'Land {name}: the package', base, moved)
        _git(self.work, 'commit', '-q', '--allow-empty', '-m', message)
        return _git(self.work, 'rev-parse', 'HEAD')

    def push(self) -> str:
        # The test's own temporary origin; a "+" lets it seed a push that
        # is not a fast-forward.
        _git(self.work, 'push', '-q', 'origin', f'+HEAD:refs/heads/{PHASE}')
        return _git(self.work, 'rev-parse', 'HEAD')

    def run(
        self, before: str, step: str
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        r"""The step's text in a fresh clone of depth 2 of the pushed
        phase branch, as actions/checkout leaves it, with the push's event."""
        clone = self.tmp / f'clone-{len(list(self.tmp.glob("clone-*")))}'
        _git(
            self.tmp, 'clone', '-q', '--depth', '2', '--branch', PHASE,
            f'file://{self.origin}', str(clone),
        )  # fmt: skip
        event = self.tmp / 'event.json'
        event.write_text(f'{{"before": "{before}"}}')
        output = self.tmp / 'output'
        output.write_text('')
        done = subprocess.run(
            ['bash', '--noprofile', '--norc', '-eo', 'pipefail', '-c', step],
            cwd=clone,
            env={
                'PATH': os.environ['PATH'],
                'HOME': str(self.tmp),
                **GIT_ENV,
                'GITHUB_EVENT_NAME': 'push',
                'GITHUB_REF_NAME': PHASE,
                'GITHUB_EVENT_PATH': str(event),
                'GITHUB_OUTPUT': str(output),
            },
            capture_output=True,
            text=True,
            check=False,
        )
        return done, clone


def _phase_scenario(phase: _Phase, scenario: str) -> str:
    r"""Seeds the push ``scenario`` names; returns its previous tip."""
    before = phase.l0
    if scenario in ('sound', 'wrong-base', 'moved'):
        p = phase.package('T1.15')
        base = {'sound': before, 'moved': before, 'wrong-base': p}[scenario]
        phase.landing('T1.15', base, 'a' * 40 if scenario == 'moved' else None)
    elif scenario == 'two-landings':
        phase.package('T1.15')
        first = phase.landing('T1.15', before)
        phase.package('T1.16')
        phase.landing('T1.16', first)
    elif scenario == 'not-a-landing':
        phase.package('T1.15')
    elif scenario == 'not-a-fast-forward':
        # The phase branch moved to another landing; this push replaces it.
        phase.package('T1.14')
        before = phase.landing('T1.14', phase.l0)
        phase.push()
        _git(phase.work, 'reset', '-q', '--hard', phase.l0)
        phase.package('T1.15')
        phase.landing('T1.15', before)
    else:
        raise AssertionError(scenario)
    phase.push()
    return before


@pytest.mark.parametrize(
    ('scenario', 'wanted'),
    [
        ('sound', None),
        ('moved', None),
        (
            'wrong-base',
            "the landing's Base-SHA is not the phase branch's previous tip",
        ),
        ('two-landings', 'it holds 2 landing commits'),
        ('not-a-landing', 'its tip is none'),
        ('not-a-fast-forward', 'not a fast-forward of the previous tip'),
    ],
)
def test_ci4_the_phase_check_in_a_clone_of_depth_two(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    scenario: str,
    wanted: str | None,
) -> None:
    def check() -> None:
        phase = _Phase(tmp_path)
        before = _phase_scenario(phase, scenario)
        _, step = _phase_check_step()
        done, clone = phase.run(before, str(step['run']))
        message = done.stdout + done.stderr
        if wanted is None:
            assert done.returncode == 0, message
            # Every other job is left out of a sound landing's run, and
            # the check still ran: its output names the checkpoint.
            assert done.stdout.split()[:2] == [before, before], message
            skip = _skip(clone, tmp_path, 'push', PHASE)
            assert skip == ('true' if scenario == 'sound' else 'false')
        else:
            assert done.returncode != 0, message
            # A previous tip the push dropped is on no branch the clone
            # fetches, so git's refusal, naming it, may come first.
            named = scenario == 'not-a-fast-forward' and before in message
            assert wanted in message or named, message

    _held(request, check)


def test_ci4_the_phase_check_reaches_a_previous_tip_past_the_clone_s_depth(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    r"""A push of a package's two commits and its landing: the previous tip
    is three commits down, past the checkout's depth of 2, and the step
    deepens the clone to reach it."""

    def check() -> None:
        phase = _Phase(tmp_path)
        before = phase.l0
        phase.package('T1.15')
        phase.package('T1.15 fix 1')
        phase.landing('T1.15', before)
        phase.push()
        clone = tmp_path / 'shallow'
        _git(
            tmp_path, 'clone', '-q', '--depth', '2', '--branch', PHASE,
            f'file://{phase.origin}', str(clone),
        )  # fmt: skip
        missing = subprocess.run(
            ['git', '-C', str(clone), 'cat-file', '-e', f'{before}^{{commit}}'],
            capture_output=True,
            check=False,
        )
        assert missing.returncode != 0, 'the previous tip is in the clone'
        _, step = _phase_check_step()
        done, _ = phase.run(before, str(step['run']))
        assert done.returncode == 0, done.stdout + done.stderr

    _held(request, check)


# --- CI4 round 4: a landing is skipped only on its candidate's evidence ---------
#
# Codex's P1 on cb46bc5a: the skip step decided on the landing's shape
# alone, so an untested candidate (a failing test, never pushed alone) with
# an empty ``Land`` commit on top, pushed together, skipped every job, and
# the phase check passed. Owner decision 30 skips a tree that already
# passed exact-SHA CI; it never infers that from a subject.
#
# Interface the cases hold the workflow to:
# - The skip step (id ``landing``) says ``skip=true`` only when, besides the
#   landing's shape, the underlying candidate has a green run: following
#   empty landing commits down (one parent, its parent's tree, ``Land ``),
#   the first commit that is not one; its newest run of ci.yml on its exact
#   SHA that is a push to the phase branch or a branch under it (never an
#   archive branch, never a pull request's) completed with conclusion
#   success. A wrapper's own run is never evidence, whatever its
#   conclusion: its jobs were left out.
# - Missing, failed, cancelled or unfinished evidence, or an API that does
#   not answer, never says skip: the step says ``skip=false`` (the full
#   jobs run) or fails (closed).
# - It reads the runs through ``gh`` or ``curl`` on PATH (here, fakes that
#   answer from the seeded runs; the network is closed to anything else),
#   holding the token as ``GH_TOKEN`` or ``GITHUB_TOKEN``; the workflow
#   grants ``actions: read``. In a clone of depth 2 it deepens itself
#   (``git fetch --deepen`` or ``--unshallow``) to follow wrappers down.
# - A push that is no landing never needs the API: it says ``skip=false``
#   and passes when the API is down.
#
# Each is held by ``contract/tests/ci4_gate.py``.

#: Runs of the candidate, newest last, by case; ``C`` the candidate, ``L``
#: the landing on it, ``W`` a second landing wrapping ``L``, ``P`` the
#: candidate's parent.
EVIDENCE: dict[str, tuple[bool, int, Any]] = {
    # name: (skips, wrappers, runs)
    'untested': (False, 1, lambda c: []),
    'green': (True, 1, lambda c: [_run(c['C'], 1)]),
    'green-on-a-topic-branch': (
        True,
        1,
        lambda c: [_run(c['C'], 1, branch=f'{PHASE}-t1.99')],
    ),
    'red-after-green': (
        False,
        1,
        lambda c: [
            _run(c['C'], 1),
            _run(
                c['C'], 2, conclusion='failure', created='2026-10-14T01:00:00Z'
            ),
        ],
    ),
    'cancelled-after-green': (
        False,
        1,
        lambda c: [
            _run(c['C'], 1),
            _run(
                c['C'],
                2,
                conclusion='cancelled',
                created='2026-10-14T01:00:00Z',
            ),
        ],
    ),
    'in-progress-after-green': (
        False,
        1,
        lambda c: [
            _run(c['C'], 1),
            _run(
                c['C'], 2, status='in_progress', created='2026-10-14T01:00:00Z'
            ),
        ],
    ),
    'queued': (False, 1, lambda c: [_run(c['C'], 1, status='queued')]),
    'pull-request-only': (
        False,
        1,
        lambda c: [_run(c['C'], 1, event='pull_request', branch=LANE)],
    ),
    'archive-branch-only': (
        False,
        1,
        lambda c: [_run(c['C'], 1, branch=f'{PHASE}-archive-lane-1a-t1.99')],
    ),
    'main-only': (False, 1, lambda c: [_run(c['C'], 1, branch='main')]),
    'another-workflow-only': (
        False,
        1,
        lambda c: [_run(c['C'], 1, workflow='publish.yml')],
    ),
    'parent-only': (False, 1, lambda c: [_run(c['P'], 1)]),
    'landing-run-only': (False, 1, lambda c: [_run(c['L'], 1)]),
    'wrapped-green': (True, 2, lambda c: [_run(c['C'], 1)]),
    'wrapped-wrapper-success-only': (
        False,
        2,
        lambda c: [_run(c['L'], 1), _run(c['W'], 2)],
    ),
    'wrapped-wrapper-skipped-only': (
        False,
        2,
        lambda c: [_run(c['L'], 1, conclusion='skipped')],
    ),
}


def _untested_push(
    tmp_path: Path, ref: str, wrappers: int
) -> tuple[Path, dict[str, str]]:
    r"""Codex's reproduction: a candidate adding a failing test, never
    tested, an empty ``Land T1.99`` with correct trailers on it (and, for
    ``wrappers`` 2, a second empty landing on that), pushed together to
    ``ref``; and a clone of depth 2 of it, as actions/checkout leaves it."""
    phase = _Phase(tmp_path)
    test = phase.work / 'tests/test_ci4_fault.py'
    test.parent.mkdir()
    test.write_text('def test_new_code():\n    assert False\n')
    _git(phase.work, 'add', '-A')
    _git(phase.work, 'commit', '-q', '-m', 'T1.99: untested faulty candidate')
    shas = {'P': phase.l0, 'C': _git(phase.work, 'rev-parse', 'HEAD')}
    shas['L'] = phase.landing('T1.99', phase.l0)
    if wrappers == 2:
        shas['W'] = phase.landing('T1.99', phase.l0)
    _git(phase.work, 'push', '-q', 'origin', f'HEAD:refs/heads/{ref}')
    clone = tmp_path / 'clone'
    _git(
        tmp_path, 'clone', '-q', '--depth', '2', '--branch', ref,
        f'file://{phase.origin}', str(clone),
    )  # fmt: skip
    return clone, shas


@pytest.mark.parametrize('ref', [PHASE, LANE])
@pytest.mark.parametrize('case', sorted(EVIDENCE))
def test_ci4_a_landing_is_skipped_only_on_its_candidate_s_green_run(
    tmp_path: Path, request: pytest.FixtureRequest, case: str, ref: str
) -> None:
    def check() -> None:
        skips, wrappers, runs = EVIDENCE[case]
        clone, shas = _untested_push(tmp_path, ref, wrappers)
        found, calls = _decide(clone, tmp_path, 'push', ref, runs(shas))
        if skips:
            assert found == 'true', (found, calls)
        else:
            assert found in ('false', 'failed'), (found, calls)

    _held(request, check)


@pytest.mark.parametrize('ref', [PHASE, LANE])
def test_ci4_an_api_that_does_not_answer_never_skips(
    tmp_path: Path, request: pytest.FixtureRequest, ref: str
) -> None:
    def check() -> None:
        clone, shas = _untested_push(tmp_path, ref, 1)
        green = [_run(shas['C'], 1)]
        found, calls = _decide(clone, tmp_path, 'push', ref, green, True)
        assert found in ('false', 'failed'), (found, calls)

    _held(request, check)


def test_ci4_a_push_that_is_no_landing_needs_no_api(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        repo = _repo(tmp_path)
        found, _ = _decide(repo, tmp_path, 'push', LANE, [], api_fails=True)
        assert found == 'false'

    _held(request, check)


def test_ci4_the_skip_step_reads_runs_with_the_workflow_token(
    request: pytest.FixtureRequest,
) -> None:
    r"""The skip step holds the token for gh or curl, and the workflow
    may read Actions runs."""

    def check() -> None:
        workflow = _workflow()
        assert workflow['permissions'] == {
            'contents': 'read',
            'packages': 'read',
            'actions': 'read',
        }
        env = _landing_step().get('env') or {}
        held = [(k, v) for k, v in env.items() if (k, v) in API_TOKENS]
        assert held, env
        run = str(_landing_step()['run'])
        assert re.search(r'\b(gh|curl)\b', run), run

    _held(request, check)


# --- every phase alike (T2.0) ---------------------------------------------------
#
# The workflow names no phase: patterns match every rust-core-phase-N, and
# where an expression cannot match a pattern (the concurrency, the phase
# check's condition) a phase branch is a name under rust-core-phase-
# holding no digit followed by "-", for any N.

PHASE_2 = 'rust-core-phase-2'
LANE_2 = f'{PHASE_2}-lane-a'


def test_t20_the_workflow_names_no_one_phase() -> None:
    text = CI.read_text(encoding='utf-8')
    assert not re.search(r'rust-core-phase-[0-9]', text)
    branches = _workflow()[True]['push']['branches']
    assert 'rust-core-phase-*' in branches
    assert '!rust-core-phase-*-archive-*' in branches


@pytest.mark.parametrize(
    ('ref', 'cancels', 'saves'),
    [
        (PHASE, False, False),
        (PHASE_2, False, False),
        (LANE, False, True),
        (LANE_2, False, True),
        ('rust-core-phase-7-lane-b', False, True),
        (f'{PHASE_2}-t2.3', True, False),
        # Multi-digit phases (T2.0 fix 1): no cap.
        ('rust-core-phase-10', False, False),
        ('rust-core-phase-10-lane-b', False, True),
        ('rust-core-phase-10-t10.2', True, False),
        ('rust-core-phase-123', False, False),
        # No phase branch, so cancelled like any branch (fix 2).
        ('rust-core-phase-', True, False),
        ('rust-core-phase-two', True, False),
        ('rust-core-phase-10x', True, False),
        ('rust-core-phase-experiment', True, False),
        ('rust-core', True, True),
        ('main', True, True),
        ('feature-x', True, False),
    ],
)
def test_t20_concurrency_and_caches_alike_in_every_phase(
    ref: str, cancels: bool, saves: bool
) -> None:
    workflow = _workflow()
    context = {
        'github.event_name': 'push',
        'github.ref_name': ref,
        'github.ref': f'refs/heads/{ref}',
    }
    condition = workflow['concurrency']['cancel-in-progress']
    assert _evaluate(condition, context) is cancels, condition
    condition = workflow['env']['SAVE_CACHES']
    assert _evaluate(condition, context) is saves, condition


@pytest.mark.parametrize('ref', [PHASE_2, LANE_2, f'{PHASE_2}-t2.3'])
def test_t20_an_empty_landing_commit_under_phase_2_skips_on_its_green_run(
    tmp_path: Path, ref: str
) -> None:
    repo = _repo(tmp_path)
    _landing(repo)
    seeded = [
        _run(sha, 100 + n, branch=LANE_2)
        for n, sha in enumerate(_git(repo, 'rev-list', 'HEAD').split())
    ]
    found, _ = _decide(repo, tmp_path, 'push', ref, seeded)
    assert found == 'true'


@pytest.mark.parametrize('ref', [PHASE_2, LANE_2])
def test_t20_another_phase_s_green_run_is_no_evidence(
    tmp_path: Path, ref: str
) -> None:
    # The candidate's only green run is on a Phase 1 lane: judged against
    # Phase 1's checkpoint, it does not stand for a Phase 2 landing.
    repo = _repo(tmp_path)
    _landing(repo)
    seeded = [
        _run(sha, 100 + n, branch=LANE)
        for n, sha in enumerate(_git(repo, 'rev-list', 'HEAD').split())
    ]
    found, _ = _decide(repo, tmp_path, 'push', ref, seeded)
    assert found == 'false'


@pytest.mark.parametrize(
    'ref',
    ['rust-core-phase-10', 'rust-core-phase-10-lane-a', 'rust-core-phase-123'],
)
def test_t20_a_multi_digit_phase_s_landing_skips_on_its_green_run(
    tmp_path: Path, ref: str
) -> None:
    repo = _repo(tmp_path)
    _landing(repo)
    seeded = [
        _run(sha, 100 + n, branch=f'{ref.split("-lane-")[0]}-lane-a')
        for n, sha in enumerate(_git(repo, 'rev-list', 'HEAD').split())
    ]
    found, _ = _decide(repo, tmp_path, 'push', ref, seeded)
    assert found == 'true'
    # Control: with no green run of its own phase it runs everything.
    found, _ = _decide(repo, tmp_path, 'push', ref, [])
    assert found == 'false'


@pytest.mark.parametrize('ref', ['rust-core-phase-02', 'rust-core-phase-x1'])
def test_t20_a_name_the_expression_lets_through_fails_the_phase_check_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ref: str
) -> None:
    r"""Fix 2: a name under the prefix that ends in a digit but is no phase
    branch passes the step's condition; the step itself then fails, naming
    it, rather than on a missing argument."""
    _, step = _phase_check_step()
    context = {
        'github.event_name': 'push',
        'github.ref_name': ref,
        'github.ref': f'refs/heads/{ref}',
        'github.base_ref': '',
        'github.head_ref': '',
    }
    assert _evaluate(str(step.get('if', '')), context) is True
    monkeypatch.setattr(sys.modules[__name__], 'PHASE', ref)
    phase = _Phase(tmp_path)
    before = _phase_scenario(phase, 'sound')
    done, _ = phase.run(before, str(step['run']))
    assert done.returncode == 1, done.stdout + done.stderr
    assert (
        f'{ref} is under rust-core-phase- but is no phase branch' in done.stdout
    )
