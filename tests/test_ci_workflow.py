# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI workflow structure, pinned images and installed-artifact checks.

Shell-step tests verify image pull retries, registry authentication and runtime
installation without downloading images or contacting model endpoints.
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
DIGEST = re.compile(r'@sha256:[0-9a-f]{64}$')


def _workflow() -> dict[str, Any]:
    return yaml.safe_load(CI.read_text(encoding='utf-8'))


def _needs(job: dict[str, Any]) -> list[str]:
    needs = job.get('needs', [])
    return [needs] if isinstance(needs, str) else list(needs)


def test_job_dependencies_exist_and_have_no_cycles() -> None:
    jobs = _workflow()['jobs']
    visited: set[str] = set()
    pending = set(jobs)
    while pending:
        ready = {name for name in pending if set(_needs(jobs[name])) <= visited}
        assert ready, f'unknown dependencies or cycle: {sorted(pending)}'
        visited.update(ready)
        pending.difference_update(ready)


#: The longest a job may run, in minutes: rust-test took 80.7 at most
#: (fix round 1, P2-5), about 31 now that its dependencies build at
#: opt-level 1 (a cold build more), and the timing guard's own test holds
#: its 120; the landing job, which every other job waits on,
#: does no more than read a few commits; the client test matrix, the
#: ref-pandas3 DuckDB route tests and the engine job (whose pytest step
#: took 40 minutes and reached its 60-minute limit on two runs) take
#: 14 to 17 minutes since split, so 60 (P2--14, P2--1);
#: the checking tools' tests, split out of those, take about 43.
TIMEOUT_CEILINGS = {
    'configuration': 10,
    'rust-test': 120,
    'test': 60,
    'contract-python-pandas3': 60,
    'tooling-root': 90,
}
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


#: A pip install naming one of this repository's distributions by name (a
#: requirement in quotes), not by path (``./packages/...``) or file.
OWN_DISTRIBUTIONS = (
    'structured-data-models-client',
    'structured-data-models-api',
    'structured-data-models-pql',
    'structured-data-models-connectors',
    'structured-data-models-relational-engine',
)
NAMED_OWN = re.compile(r'["\'](?:' + '|'.join(OWN_DISTRIBUTIONS) + ')')
#: A pip install of the candidate's own wheels without their dependencies.
OWN_FIRST = re.compile(
    r'pip3? install\b[^\n]*--no-deps[^\n]*(dist/\*\.whl|\$candidate)'
)


def _unguarded_named_installs(jobs: dict[str, Any]) -> list[str]:
    r"""The jobs that install one of this repository's distributions by name
    before installing the candidate's own wheels without their
    dependencies: there an index's published distribution of the same
    version stands in for the candidate's."""
    found = []
    for name, job in jobs.items():
        own_first = False
        for step in job.get('steps', []):
            run = step.get('run') if isinstance(step, dict) else None
            if not isinstance(run, str):
                continue
            for line in re.sub(r'\\\n\s*', ' ', run).split('\n'):
                if OWN_FIRST.search(line):
                    own_first = True
                elif re.search(r'pip3? install', line) and NAMED_OWN.search(
                    line
                ):
                    if not own_first:
                        found.append(name)
    return found


def test_a_job_installs_the_candidate_s_wheels_before_naming_them() -> None:
    assert _unguarded_named_installs(_workflow()['jobs']) == []
    # The fault it catches: a by-version install with only --find-links.
    seeded = {
        'sbom': {
            'steps': [
                {
                    'run': 'VERSION=$(cat dist/VERSION)\n'
                    'pip install --find-links "$GITHUB_WORKSPACE/dist" \\\n'
                    '  "structured-data-models-client[sqlite]==$VERSION"\n'
                }
            ]
        }
    }
    assert _unguarded_named_installs(seeded) == ['sbom']


# --- the images -----------------------------------------------------------------


def test_each_image_is_named_once_and_pinned_by_digest() -> None:
    workflow = _workflow()
    env = workflow['env']
    for key in ('POSTGRES_IMAGE', 'MANYLINUX'):
        assert DIGEST.search(env[key]), env[key]
    text = CI.read_text(encoding='utf-8')
    for key in ('POSTGRES_IMAGE', 'MANYLINUX'):
        assert text.count(env[key]) == 1, key
    # The aarch64 build's image, named in that job alone.
    aarch64 = workflow['jobs']['native-wheel-aarch64']['env']['MANYLINUX']
    assert DIGEST.search(aarch64) and 'manylinux2014_aarch64:' in aarch64
    assert text.count(aarch64) == 1
    jobs = workflow['jobs']
    configured = jobs['configuration']['outputs']['postgres-image']
    assert configured == '${{ env.POSTGRES_IMAGE }}'
    for name in ('test',):
        service = jobs[name]['services']['postgres']
        assert (
            service['image']
            == '${{ needs.configuration.outputs.postgres-image }}'
        )
        assert 'configuration' in _needs(jobs[name])
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
    for name in (
        'native-wheel',
        'native-test-ops',
        'native-wheel-aarch64',
        'wheel-smoke',
        'wheel-smoke-aarch64',
    ):
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


def _expression(condition: object) -> str:
    text = str(condition).strip()
    if text.startswith('${{') and text.endswith('}}'):
        text = text[3:-2]
    text = ' '.join(text.split())
    guard = "!(github.event_name == 'workflow_dispatch' && inputs.pql_runtime_only) && ("
    if text.startswith(guard) and text.endswith(')'):
        text = text[len(guard) : -1]
    return text


def _images() -> list[str]:
    r"""Every registry image reference written in the workflow; a Java
    module path in a JVM flag (``--add-opens=java.base/java.nio=...``, the
    Java jobs') has an image's shape and is none."""
    text = CI.read_text(encoding='utf-8')
    return [
        image
        for image in re.findall(
            r'(?<![\w./-])(?:[a-z0-9-]+\.)+[a-z]{2,}(?::\d+)?/[a-z0-9._/-]+'
            r'(?::[\w.-]+)?(?:@sha256:[0-9a-f]+)?',
            text,
        )
        if not image.startswith('java.base/')
    ]


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
        aarch64 = workflow['jobs']['native-wheel-aarch64']['env']['MANYLINUX']
        # The JAR-load jobs' clean JDK containers, the only job
        # containers: the runner pulls them, anonymously, from MCR.
        load = workflow['jobs']['java-load']
        jdks = {
            entry['image']
            for entry in load['strategy']['matrix']['include']
            if entry['image']
        }
        assert load['container'] == '${{ matrix.image }}'
        for image in jdks:
            assert image.startswith('mcr.microsoft.com/openjdk/jdk:'), image
        named = {env['POSTGRES_IMAGE'], env['MANYLINUX'], aarch64, *jdks}
        assert set(images) == named, set(images) ^ named
        for name, job in workflow['jobs'].items():
            assert name == 'java-load' or 'container' not in job, name
            for service in (job.get('services') or {}).values():
                assert service['image'] == (
                    '${{ needs.configuration.outputs.postgres-image }}'
                ), name
        # wheel-smoke's images, as wheel_smoke.py names them, by lane.
        lanes = [
            lane
            for name in ('wheel-smoke', 'wheel-smoke-aarch64')
            for lane in workflow['jobs'][name]['strategy']['matrix']['lane']
        ]
        assert len(lanes) == 4, lanes
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

    check()


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

    check()


def _service_image(workflow: dict[str, Any], service: dict[str, Any]) -> str:
    r"""The image a service runs: POSTGRES_IMAGE when it takes the landing
    job's output, which must be that env value, or its literal image."""
    image = str(service['image'])
    if image == '${{ needs.configuration.outputs.postgres-image }}':
        outputs = workflow['jobs']['configuration']['outputs']
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


def test_ci4_the_token_goes_only_to_ghcr_io(
    request: pytest.FixtureRequest,
) -> None:
    r"""GITHUB_TOKEN reaches only pull_image.sh, which logs in to ghcr.io
    alone, and the credentials of a service whose image is on ghcr.io,
    which are exactly ``github.actor`` and ``secrets.GITHUB_TOKEN``; a
    service whose image is elsewhere has no credentials; and the landing
    job's skip step, which reads the candidate's runs from the GitHub API
    (round 4) and runs no container, and the pinned Java artifact action
    reading this repository's current run. No other secret is used, and nothing
    wider than reading packages and Actions runs is granted."""

    def check() -> None:
        workflow = _workflow()
        assert workflow['permissions'] in (
            {'contents': 'read', 'packages': 'read'},
            {'contents': 'read', 'packages': 'read', 'actions': 'read'},
        )
        assert not re.search(TOKEN, str(workflow.get('env')))
        steps = services = artifact_downloads = 0
        for name, job in workflow['jobs'].items():
            assert not re.search(TOKEN, str(job.get('env'))), name
            assert job.get('permissions') is None, name
            # The JAR-load jobs' JDK containers are pulled
            # anonymously: a container named with no credentials.
            assert name == 'java-load' or 'container' not in job, name
            if 'container' in job:
                assert isinstance(job['container'], str), name
            for step in job.get('steps', []):
                if re.search(TOKEN, str(step.get('with'))):
                    assert name == 'java', (name, step)
                    assert step == {
                        'uses': (
                            'actions/download-artifact@'
                            '9000827ccba6bdab643e8b6fd33ac0654aef8333'
                        ),
                        'with': {
                            'name': 'java-jar',
                            'path': 'jar',
                            'github-token': '${{ github.token }}',
                            'repository': '${{ github.repository }}',
                            'run-id': '${{ github.run_id }}',
                        },
                    }, (name, step)
                    assert workflow['permissions'].get('actions') == 'read'
                    artifact_downloads += 1
                assert not re.search(TOKEN, str(step.get('run'))), (name, step)
                for key, value in (step.get('env') or {}).items():
                    if not re.search(TOKEN, str(value)):
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
        }
        artifact_token_lines = 0
        for line in CI.read_text(encoding='utf-8').splitlines():
            if re.search(TOKEN, line) and not line.strip().startswith('#'):
                if line.strip() == 'github-token: ${{ github.token }}':
                    artifact_token_lines += 1
                else:
                    assert line.strip() in allowed, line
        # The one new textual token occurrence must belong to the exact
        # structurally checked action, never another key or duplicate step.
        assert artifact_token_lines == artifact_downloads == 1, (
            artifact_token_lines,
            artifact_downloads,
        )

    check()


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

    check()


# Evaluate the supported workflow expressions in seeded event contexts.

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
    step having succeeded. With no status function, ``success`` is
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
    'ref', ['main', 'rust-core', 'feature-x', 'release-candidate']
)
@pytest.mark.parametrize('event', ['push', 'pull_request', 'workflow_dispatch'])
def test_superseded_runs_cancel_and_trusted_runs_warm_caches(ref, event):
    workflow = _workflow()
    assert workflow['concurrency']['cancel-in-progress'] is True
    context = {
        'github.event_name': event,
        'github.ref_name': ref,
        'github.ref': f'refs/heads/{ref}',
    }
    saves = event == 'workflow_dispatch' or (
        event == 'push' and ref in ('main', 'rust-core')
    )
    assert _evaluate(workflow['env']['SAVE_CACHES'], context) is saves
    assert _evaluate(workflow['env']['SAVE_PIP_CACHES'], context) is saves


def test_dependency_caches_restore_before_python_and_java_installation():
    workflow = _workflow()
    for name, job in workflow['jobs'].items():
        steps = job.get('steps', [])
        for index, step in enumerate(steps):
            if 'setup-python@' in step.get('uses', '') and name != 'test':
                assert (
                    steps[index + 1].get('uses')
                    == './.github/actions/python-cache'
                )
            if name in ('java', 'java-db') and 'setup-java@' in step.get(
                'uses', ''
            ):
                assert step['with']['cache'] == 'maven'


# --- CI7: aarch64 wheels and the Databricks runtime cells ---

#: The jobs and promise, which the design's CI7 row names:
#: the aarch64 native wheels beside x86_64, the two aarch64 smoke lanes,
#: and the six runtime cells. The row names the cells only as "six
#: ``runtime-install (<runtime>)`` jobs"; these are the six the locked set
#: named for check 10 until judged that check's CI part by land.py's
#: exact-SHA rule instead.
CI7_JOBS = (
    'native-wheel (aarch64, shipped)',
    'native-wheel (aarch64, test-ops)',
    'wheel-smoke (aarch64-floor)',
    'wheel-smoke (aarch64-newest)',
)
RUNTIME_CELLS = (
    'runtime-install (dbr14.3)',
    'runtime-install (dbr15.4)',
    'runtime-install (dbr16.4)',
    'runtime-install (dbr17.3)',
    'runtime-install (serverless5)',
    'runtime-install (serverless6)',
)
INSTALL_GUIDE = ROOT / 'docs/get-started/installation.md'
NATIVE_WHEEL = ROOT / 'scripts/rust_core/native_wheel.py'
WHEEL_NAME = 'structured_data_models_client-1.0.0-cp310-abi3-{}.whl'


def _runtime_matrix_entries(
    job: dict[str, Any], targeted: bool = False
) -> list[dict[str, Any]]:
    entries = job['strategy']['matrix']['include']
    if isinstance(entries, list):
        return entries
    # The dynamic matrix has exactly two literal JSON alternatives; validate
    # both instead of treating a targeted lane as the normal shipped matrix.
    choices = re.findall(r"'(\[.*?\])'", entries)
    assert len(choices) == 2
    return json.loads(choices[0 if targeted else 1])


def _job_names(job_id: str, job: dict[str, Any]) -> list[str]:
    r"""The names a job's runs take: its ``name`` with each matrix entry's
    values put in (a matrix of lists or of ``include`` entries)."""
    name = str(job.get('name', job_id))
    matrix = (job.get('strategy') or {}).get('matrix') or {}
    entries: list[dict[str, Any]] = (
        _runtime_matrix_entries(job) if 'include' in matrix else []
    )
    for key, values in matrix.items():
        if key != 'include':
            entries = [{key: value} for value in values]
    names = []
    for entry in entries or [{}]:
        named = name
        for key, value in entry.items():
            named = named.replace(f'${{{{ matrix.{key} }}}}', str(value))
        names.append(named)
    return names


def test_ci7_the_promised_jobs_are_jobs_of_the_workflow() -> None:
    """Keep ARM wheel/smoke coverage and all six supported runtime cells."""
    names = [
        name
        for job_id, job in _workflow()['jobs'].items()
        for name in _job_names(job_id, job)
    ]
    assert set(CI7_JOBS) <= set(names), sorted(set(CI7_JOBS) - set(names))
    runtimes = sorted(n for n in names if n.startswith('runtime-install ('))
    assert runtimes == sorted(RUNTIME_CELLS)


def test_ci7_the_aarch64_jobs_run_natively_on_arm() -> None:
    jobs = _workflow()['jobs']
    for name in ('native-wheel-aarch64', 'wheel-smoke-aarch64'):
        assert jobs[name]['runs-on'] == 'ubuntu-24.04-arm', name
    build = jobs['native-wheel-aarch64']
    entries = _runtime_matrix_entries(build)
    assert {e['args'] for e in entries} == {
        '--out native',
        '--out native --test-ops',
    }
    assert build['env']['NATIVE_ARGS'] == '${{ matrix.args }}'
    # The same steps as native-wheel's (YAML anchors), so one build script.
    native = jobs['native-wheel']['steps']
    for step in native[3:5]:
        assert step in build['steps'], step['name']
    smoke = jobs['wheel-smoke-aarch64']
    assert set(smoke['needs']) == {'native-wheel', 'native-wheel-aarch64'}


def test_ci7_rustup_init_is_checked_for_each_architecture() -> None:
    r"""The build script fetches rustup-init for the machine's
    architecture and checks it against that architecture's published
    SHA-256, and refuses an architecture it holds no checksum for."""
    run = next(
        str(s['run'])
        for s in _workflow()['jobs']['native-wheel']['steps']
        if s.get('name') == 'Build the wheel in manylinux2014'
    )
    assert (
        'x86_64) sum=4acc9acc76d5079515b46346a485974457b5a79893cfb01112423c89aeb5aa10'
        in run
    )
    assert (
        'aarch64) sum=9732d6c5e2a098d3521fca8145d826ae0aaa067ef2385ead08e6feac88fa5792'
        in run
    )
    assert '$arch-unknown-linux-gnu/rustup-init' in run
    assert 'sha256sum -c -' in run and '*) echo' in run


def _check_set(
    tmp_path: Path, *wheels: tuple[str, str]
) -> subprocess.CompletedProcess[str]:
    dirs = []
    for directory, name in wheels:
        (tmp_path / directory).mkdir(exist_ok=True)
        (tmp_path / directory / name).write_bytes(b'')
        if directory not in dirs:
            dirs.append(directory)
    return subprocess.run(
        [sys.executable, str(NATIVE_WHEEL), '--check-set', *dirs],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )


X86 = WHEEL_NAME.format('manylinux_2_17_x86_64.manylinux2014_x86_64')
ARM = WHEEL_NAME.format('manylinux_2_17_aarch64.manylinux2014_aarch64')


def test_ci7_the_wheel_set_holds_one_wheel_per_architecture(
    tmp_path: Path,
) -> None:
    done = _check_set(tmp_path, ('native-x86_64', X86), ('native', ARM))
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == [ARM, X86]


@pytest.mark.parametrize(
    ('wheels', 'why'),
    [
        # Seeded: a wheel set missing one architecture.
        ((('native-x86_64', X86),), "no wheel for ['aarch64']"),
        ((('native', ARM),), "no wheel for ['x86_64']"),
        (
            (('native-x86_64', X86), ('native', X86)),
            "no wheel for ['aarch64']",
        ),
        (
            (
                ('native-x86_64', X86),
                ('native', ARM),
                ('native', ARM.replace('1.0.0', '1.0.1')),
            ),
            'more than one wheel',
        ),
        (
            (('native-x86_64', X86), ('native', ARM.replace('1.0.0', '1.0.1'))),
            'disagree',
        ),
        (
            (
                ('native-x86_64', X86),
                ('native', WHEEL_NAME.format('manylinux_2_28_aarch64')),
            ),
            'platform tag',
        ),
    ],
)
def test_ci7_a_wheel_set_short_of_an_architecture_fails(
    tmp_path: Path, wheels: tuple[tuple[str, str], ...], why: str
) -> None:
    done = _check_set(tmp_path, *wheels)
    assert done.returncode != 0
    assert why in done.stderr, done.stderr


def test_ci7_the_aarch64_smoke_checks_the_wheel_set_before_the_lane() -> None:
    steps = _workflow()['jobs']['wheel-smoke-aarch64']['steps']
    runs = [str(s.get('run', '')) for s in steps]
    check = next(i for i, r in enumerate(runs) if '--check-set' in r)
    lane = next(i for i, r in enumerate(runs) if 'wheel_smoke.py --lane' in r)
    assert check < lane
    downloads = {
        s['with']['name']: s['with']['path']
        for s in steps
        if str(s.get('uses', '')).startswith('actions/download-artifact')
    }
    assert downloads == {
        'native-wheel': 'native-x86_64',
        'native-wheel-aarch64': 'native',
        'native-wheel-test-ops-aarch64': 'native-test-ops',
    }
    assert runs[check].split('--check-set', 1)[1].split() == [
        'native-x86_64',
        'native',
    ]


def _install_step() -> dict[str, Any]:
    steps = _workflow()['jobs']['runtime-install']['steps']
    return next(s for s in steps if s.get('id') == 'install')


def test_ci7_the_runtime_cells_run_the_documented_install_line() -> None:
    r"""The install step runs the guide's ``%pip install`` line as the
    guide writes it, but for the notebook's ``%``, and makes
    ``keep_numpy`` from the runtime's NumPy as the guide does."""
    guide = INSTALL_GUIDE.read_text(encoding='utf-8')
    (line,) = [
        ln.strip()
        for ln in guide.splitlines()
        if ln.strip().startswith('%pip install')
    ]
    assert "keep_numpy = f'numpy=={numpy.__version__}'" in guide
    run = str(_install_step()['run'])
    commands = [ln.strip() for ln in run.splitlines()]
    assert line.removeprefix('%') in commands, line
    assert '$keep_numpy' in line
    job = _workflow()['jobs']['runtime-install']
    assert _needs(job) == ['build']
    for entry in job['strategy']['matrix']['include']:
        pins = dict(p.split('==') for p in entry['preinstalled'].split())
        # 16.4's own pandas 1.5.3 has no wheel and no build on Python 3.12:
        # left out rather than stood in for, so the line installs one; 14.3's
        # release notes list no PyYAML.
        wanted = {'numpy', 'pyarrow'}
        if entry['runtime'] != 'dbr16.4':
            wanted.add('pandas')
        if entry['runtime'] != 'dbr14.3':
            wanted.add('PyYAML')
        assert set(pins) == wanted, entry
    # The floor the step holds pandas to is the client's own.
    project = (
        ROOT / 'packages/structured-data-models-client/pyproject.toml'
    ).read_text()
    assert '"pandas>=2.2.2"' in project
    assert 'printf \'%s\\n\' 2.2.2 "$pandas_after"' in run


#: A pip and a python for the install step, run as CI runs it: python
#: answers the interpreter tag and the NumPy version held in a file, and
#: runs the check of the installed wheels' records as passing unless
#: FAULT is swapped (the line replaced a candidate wheel); pip records
#: each call and, by FAULT, moves NumPy on install (an install that moves
#: the runtime's NumPy), fails the install (a resolution failure), leaves
#: pandas below the client's floor or fails pip check; otherwise the
#: install brings pandas 2.3.3, as it did on 14.3 and 15.4.
FAKE_PYTHON = """#!/usr/bin/env bash
case "$*" in
  *version_info*) echo cp312 ;;
  *numpy*) cat "$STATE/numpy" ;;
  *'"pandas"'*) [ -f "$STATE/pandas" ] && cat "$STATE/pandas" ;;
  scripts/check_wheel_provenance.py\\ *)
    echo "python $*" >> "$STATE/calls"
    [ "$FAULT" = swapped ] && { echo 'no longer the candidate wheel' >&2; exit 1; }
    exit 0 ;;
  *) exit 2 ;;
esac
"""
FAKE_PIP = """#!/usr/bin/env bash
echo "pip $*" >> "$STATE/calls"
case "$1" in
  install)
    [ "$2" = --quiet ] && [ "$3" = --no-deps ] && exit 0
    [ "$FAULT" = resolution ] && { echo 'ResolutionImpossible' >&2; exit 1; }
    [ "$FAULT" = moves ] && echo 2.1.3 > "$STATE/numpy"
    [ "$FAULT" = pandas-low ] || echo 2.3.3 > "$STATE/pandas"
    exit 0 ;;
  check) [ "$FAULT" = check ] && { echo 'x has requirement y' >&2; exit 1; }; exit 0 ;;
  *) exit 0 ;;
esac
"""


def _run_install(
    tmp_path: Path, fault: str, pandas: str | None = '1.5.3'
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    state = tmp_path / 'state'
    state.mkdir()
    (state / 'numpy').write_text('1.23.5\n')
    if pandas:
        (state / 'pandas').write_text(f'{pandas}\n')
    runtime = tmp_path / 'runtime'
    (runtime / 'bin').mkdir(parents=True)
    for name, text in (('python', FAKE_PYTHON), ('pip', FAKE_PIP)):
        tool = runtime / 'bin' / name
        tool.write_text(text)
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    (runtime / 'bin' / 'activate').write_text(
        f'PATH="{runtime / "bin"}:$PATH"\n'
    )
    work = tmp_path / 'work'
    for name in (
        X86,
        'structured_data_models_api-1.0.0-py3-none-any.whl',
        'structured_data_models_connectors-1.0.0-py3-none-any.whl',
    ):
        (work / 'dist').mkdir(parents=True, exist_ok=True)
        (work / 'dist' / name).write_bytes(b'')
    run = str(_install_step()['run']).replace('/tmp/runtime', str(runtime))
    done = subprocess.run(
        ['bash', '-e', '-c', run],
        cwd=work,
        env={
            'PATH': os.environ['PATH'],
            'STATE': str(state),
            'FAULT': fault,
            'GITHUB_WORKSPACE': str(work),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    calls = (state / 'calls').read_text().splitlines()
    return done, calls


def test_ci7_the_install_keeps_numpy_and_takes_the_candidate_s_wheels(
    tmp_path: Path,
) -> None:
    done, calls = _run_install(tmp_path, 'none')
    assert done.returncode == 0, done.stderr
    candidate = ' '.join(
        str(path) for path in sorted((tmp_path / 'work' / 'dist').glob('*.whl'))
    )
    assert calls == [
        # Install the candidate wheels before resolving connector extras.
        f'pip install --quiet --no-deps {candidate}',
        'pip install --only-binary=:all: '
        'structured-data-models-client[relational,databricks,databricks-serving] '
        'numpy==1.23.5',
        'pip check',
        f'python scripts/check_wheel_provenance.py --installed {candidate}',
        'pip freeze --all',
    ]
    assert "pandas: the runtime's 1.5.3, after the install 2.3.3" in done.stdout


def test_ci7_a_runtime_without_pandas_gets_one_at_the_floor(
    tmp_path: Path,
) -> None:
    done, _ = _run_install(tmp_path, 'none', pandas=None)
    assert done.returncode == 0, done.stderr
    assert "pandas: the runtime's none, after the install 2.3.3" in done.stdout


@pytest.mark.parametrize(
    ('fault', 'says'),
    [
        # Seeded: an install that moves the runtime's NumPy.
        ('moves', "moved the runtime's NumPy: numpy==1.23.5, now numpy==2.1.3"),
        # Seeded: a resolution failure.
        ('resolution', 'ResolutionImpossible'),
        ('check', 'has requirement'),
        ('swapped', 'no longer the candidate wheel'),
        ('pandas-low', "left pandas 1.5.3, below the client's floor 2.2.2"),
    ],
)
def test_ci7_a_runtime_cell_fails_on_each_seeded_fault(
    tmp_path: Path, fault: str, says: str
) -> None:
    done, _ = _run_install(tmp_path, fault)
    assert done.returncode != 0
    assert says in done.stderr, done.stderr


# ---: main's engine rows on every platform ------------------------------


# ---: the judging-tool tests in a job of their own, the timing guard,
# and the comparison base taken at push time ---------------------------------
#
# Tooling tests run separately from the supported Python version matrix.

T30_MARKER = 'tooling'


def _pytest_steps() -> list[tuple[str, str]]:
    """Each pytest run over the root suite (``tests``) or contract/tests:
    a run in a package's own directory (``cd packages/...``, the engine's
    own ``tests``) or over a package's or binding's tests is neither."""
    found = []
    for name, job in _workflow()['jobs'].items():
        for step in job.get('steps', []):
            run = str(step.get('run', ''))
            if 'pytest' not in run or re.search(r'(?m)^\s*cd\s', run):
                continue
            if _suites(run):
                found.append((name, run))
    return found


def _selects(run: str) -> bool:
    """Whether a pytest run selects the marker: ``-m <marker>``, quoted or
    not, or an expression holding it but not as ``not <marker>``."""
    expressions = [
        quoted or bare
        for _, quoted, bare in re.findall(
            r"-m\s+(?:(['\"])(.*?)\1|([^\s'\"]+))", run
        )
    ]
    return any(
        re.search(rf'\b{T30_MARKER}\b', expr)
        and not re.search(rf'not {T30_MARKER}\b', expr)
        for expr in expressions
    )


def _suites(run: str) -> set[str]:
    """The suites a pytest run names: ``tests``, ``contract/tests``."""
    return set(re.findall(r'(?<![\w/$-])((?:contract/)?tests)(?![\w/-])', run))


def _tooling_jobs() -> dict[str, dict[str, Any]]:
    """The jobs whose pytest runs select the marker: the judging-tool
    jobs (one per suite, when one job would outrun its limit)."""
    jobs = _workflow()['jobs']
    names = sorted({name for name, run in _pytest_steps() if _selects(run)})
    assert names, f'no job selects -m {T30_MARKER}'
    return {name: jobs[name] for name in names}


def test_the_judging_tool_tests_run_once_in_their_own_job() -> None:
    markers = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    assert re.search(rf'^\s*"{T30_MARKER}:', markers, re.M)
    tooling = _tooling_jobs()
    for name, job in tooling.items():
        matrix = job.get('strategy', {}).get('matrix', {})
        assert not matrix, f'{name} runs in a matrix: {matrix}'
    runs = [r for n, r in _pytest_steps() if n in tooling and _selects(r)]
    covered = [suite for run in runs for suite in sorted(_suites(run))]
    # Maintained tooling tests live in the root suite; contract tooling was retired.
    assert covered == ['tests'], (covered, runs)
    for other, run in _pytest_steps():
        if other not in tooling:
            assert f'not {T30_MARKER}' in run, f'{other} runs them: {run}'


def test_targeted_pql_runtime_cannot_enable_unrelated_ci_jobs() -> None:
    workflow = _workflow()
    jobs = workflow['jobs']
    guard = (
        "!(github.event_name == 'workflow_dispatch' && inputs.pql_runtime_only)"
    )
    for name, job in jobs.items():
        if name not in (
            'configuration',
            'nightly',
            'native-wheel-aarch64',
            'pql-runtime-aarch64',
        ):
            assert guard in job['if'], name
    assert '-pql-runtime' in workflow['concurrency']['group']
    assert 'not full CI' in workflow['run-name']
    assert _runtime_matrix_entries(jobs['native-wheel-aarch64'], True) == [
        {
            'build': 'shipped',
            'args': '--out native',
            'artifact': 'native-wheel-aarch64',
            'java': '',
        }
    ]
    pql = jobs['pql-runtime-aarch64']
    assert pql['needs'] == 'native-wheel-aarch64'
    assert pql['runs-on'] == 'ubuntu-24.04-arm'
    assert {
        entry['python'] for entry in pql['strategy']['matrix']['include']
    } == {'3.10', '3.14'}
    assert 'inputs.pql_runtime_only' in pql['if']


def test_ci_has_no_phase_branch_scheduler():
    workflow = _workflow()
    triggers = workflow[True]
    assert set(triggers) == {'pull_request', 'push', 'workflow_dispatch'}
    assert triggers['push']['branches'] == ['main', 'rust-core']
    assert 'nightly' not in workflow['jobs']
    assert 'gh workflow run' not in CI.read_text()


def test_ci_configuration_never_skips_jobs_based_on_migration_commits():
    workflow = _workflow()
    job = workflow['jobs']['configuration']
    assert set(job['outputs']) == {'pythons', 'postgres-image', 'version'}
    assert 'needs.landing' not in CI.read_text()
    assert json.loads(job['outputs']['pythons']) == [
        '3.10',
        '3.11',
        '3.12',
        '3.13',
        '3.14',
    ]
    assert job['outputs']['postgres-image'] == '${{ env.POSTGRES_IMAGE }}'
    assert not job.get('needs')
    assert 'ratchet.py' not in str(job)
    assert 'GH_TOKEN' not in str(job)


def test_contract_observations_run_only_the_native_python_client():
    jobs = _workflow()['jobs']
    step = next(
        step
        for step in jobs['contract-python-pandas3']['steps']
        if step.get('name') == 'Native Python observations'
    )
    script = step['run']
    assert '--mode python-core' in script
    assert 'python-old' not in script
    assert 'gate.sh' not in script
    assert 'ratchet.py' not in script
    assert 'check_observations.py' in script
    assert '--expect contract/expected/duckdb-python.json' in script
    assert '--cell "$CELL"' in script
    assert '--reference' not in script
    subprocess.run(['bash', '-n'], input=script, text=True, check=True)


def test_database_observations_use_current_expectations_and_keep_sql():
    jobs = _workflow()['jobs']
    step = next(
        step
        for step in jobs['contract-python-pandas3-sqlite']['steps']
        if step.get('name')
        == 'Native Python observations on the selected database'
    )
    script = step['run']
    assert 'check_observations.py' in script
    assert 'contract/expected/$DATABASE-python.json' in script
    assert '--out "$out/python-core.jsonl"' in script
    assert '--commit "$GITHUB_SHA"' in script
    assert '--native-sha256' in script
    assert '--reference' not in script
    assert 'ratchet.py' not in script
    subprocess.run(['bash', '-n'], input=script, text=True, check=True)


@pytest.mark.parametrize(
    ('job_name', 'version', 'duckdb_pin'),
    [
        ('contract-python-floor-pandas2-2', '3.10', '1.5.6'),
        ('contract-python-floor-pandas2-2-sqlite', '3.10', '1.5.6'),
        ('contract-python-floor-pandas2-2-postgres', '3.10', '1.5.6'),
        ('contract-python-pandas3', '3.12', '2.0.0'),
        ('contract-python-dbr17-pandas2', '3.12', '2.0.0'),
    ],
)
def test_contract_installer_preserves_supported_duckdb_runtime(
    tmp_path, job_name, version, duckdb_pin
):
    from packaging.requirements import Requirement

    job = _workflow()['jobs'][job_name]
    setup = next(
        step for step in job['steps'] if 'setup-python@' in step.get('uses', '')
    )
    assert setup['with']['python-version'] == version
    install = next(
        step['run']
        for step in job['steps']
        if step.get('name') == 'Install the cell'
    )
    constraints_script = install.split('pip install --upgrade pip')[0]
    subprocess.run(
        ['bash', '-euc', constraints_script],
        check=True,
        env={
            **os.environ,
            'RUNNER_TEMP': str(tmp_path),
            'CELL_PINS': job['env']['CELL_PINS'],
        },
    )
    active = [
        requirement
        for line in (tmp_path / 'cell.txt').read_text().splitlines()
        if (requirement := Requirement(line)).name == 'duckdb'
        and requirement.marker.evaluate({'python_version': version})
    ]
    assert len(active) == 1
    assert str(active[0].specifier) == '==' + duckdb_pin
    assert '$CELL_PINS duckdb pyyaml' in install


def test_unreleased_binaries_are_kept_one_day_and_shipped_wheels_are_not() -> (
    None
):
    uploads = {
        step['with']['name']: step['with']
        for job in _workflow()['jobs'].values()
        for step in job.get('steps', [])
        if str(step.get('uses', '')).startswith('actions/upload-artifact@')
    }
    unreleased = sorted(
        name
        for name in uploads
        if name.startswith('java-') or 'test-ops' in name
    )
    assert {'java-jar', 'java-test-ops-native', 'native-wheel-test-ops'} <= set(
        unreleased
    )
    assert {
        name: uploads[name].get('retention-days') for name in unreleased
    } == {name: 1 for name in unreleased}
    assert uploads['${{ matrix.artifact }}']['retention-days'] == (
        "${{ matrix.build == 'test-ops' && 1 || '' }}"
    )
    for name in ('native-wheel', 'native-wheel-${{ matrix.platform }}'):
        assert 'retention-days' not in uploads[name], name
