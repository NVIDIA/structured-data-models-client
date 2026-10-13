# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Publish this release's distributions to PyPI through Kitmaker's Portal.

Kitmaker does not accept an upload. It fetches each artifact from a URL and
publishes it itself, so the release workflow attaches the distributions to the
GitHub release first and this hands Portal those asset URLs.

Two passes, and the split is the point. Every project is validated and dry-run
to a terminal state before any of them publishes, because the five
distributions pin each other by exact version: a client live on PyPI without
the engine it requires is worse than a release that never started. Portal
batches are sequential and not atomic, and a published file cannot be
recalled, so before the first upload is the only moment this stops cheaply.

Anything already on PyPI at this version is verified and skipped rather than
resubmitted, so a run that failed halfway can be restarted.

Run from the workflow with ``assets.tsv`` already written and ``dist/``
downloaded. Exits non-zero on any failure; the workflow has no other way to
know this went wrong.
"""

import hashlib
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

API = 'https://kitmaker-portal.nvidia.com/api/v0'
PYPI = 'https://pypi.org/pypi'

# Ordered so nothing publishes before what it pins. The client requires api and
# pql exactly and reaches connectors and the engine through extras, so it goes
# last: a failed dependency leaves it absent rather than uninstallable.
PROJECTS = {
    6899: 'structured-data-models-api',
    6900: 'structured-data-models-pql',
    6901: 'structured-data-models-connectors',
    6902: 'structured-data-models-relational-engine',
    6898: 'structured-data-models-client',
}

# An unrecognised state fails rather than waits: polling an unexpected
# `rejected` for half an hour reports a timeout, which is the wrong problem.
PENDING = {
    'pending',
    'queued',
    'running',
    'processing',
    'in_progress',
    'inprogress',
    'started',
    'new',
    'submitted',
    'created',
    'accepted',
}
TERMINAL_OK = {'completed', 'complete', 'success', 'succeeded', 'done'}
TERMINAL_BAD = {
    'failed',
    'failure',
    'error',
    'errored',
    'cancelled',
    'canceled',
    'rejected',
    'aborted',
    'timeout',
    'timed_out',
}

POLL_SECONDS = 30
POLL_ATTEMPTS = 60

# Reads retry, submissions never: losing pass two to one connection reset is
# the half-published state the two passes exist to prevent, but a repeated POST
# could publish twice.
READ_RETRIES = 5
READ_BACKOFF = 4

# PyPI's JSON endpoint is CDN-cached and lags a publish by some seconds.
VERIFY_ATTEMPTS = 20
VERIFY_SECONDS = 15

TOKEN = os.environ['KITMAKER_API_TOKEN']
PIC = os.environ['PIC']


def canonical(name: str) -> str:
    r"""Kitmaker's comparison: case insensitive, with ``.``, ``_`` and ``-``
    all equivalent."""
    return re.sub(r'[._-]+', '-', name).lower()


def distribution_of(filename: str) -> str:
    r"""The distribution a built file belongs to.

    A wheel's first ``-`` separated field is its distribution, which cannot
    itself contain ``-``. An sdist is ``{name}-{version}.tar.gz``, and a
    normalised version contains no ``-`` either, so one split from the right
    is enough.
    """
    if filename.endswith('.whl'):
        return filename.split('-')[0]
    return filename.rsplit('-', 1)[0]


def version_of(filename: str) -> str:
    r"""The version a built file carries. The build job has already refused to
    continue unless all of them agree, so any one file answers."""
    if filename.endswith('.whl'):
        return filename.split('-')[1]
    return filename.rsplit('-', 1)[1].removesuffix('.tar.gz')


def call(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    attempts = READ_RETRIES if method == 'GET' else 1
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            f'{API}{path}',
            data=data,
            method=method,
            headers={
                'Authorization': f'Bearer {TOKEN}',
                'Content-Type': 'application/json',
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read() or b'{}')
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors='replace')[:400]
            # 4xx is an answer, not a blip, so it is not worth repeating.
            if attempt < attempts and error.code >= 500:
                time.sleep(READ_BACKOFF * attempt)
                continue
            raise SystemExit(
                f'{method} {path} returned {error.code}: {detail}'
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            if attempt < attempts:
                time.sleep(READ_BACKOFF * attempt)
                continue
            raise SystemExit(f'{method} {path} failed: {error}') from error
    raise SystemExit(f'{method} {path}: unreachable')


def on_pypi(name: str, version: str) -> dict[str, str] | None:
    r"""What PyPI holds for this version as ``{filename: sha256}``, or ``None``
    if the version is absent. Public, so it answers for what a user gets."""
    url = f'{PYPI}/{name}/{version}/json'
    for attempt in range(1, READ_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                body = json.loads(response.read())
            return {
                entry['filename']: entry['digests']['sha256']
                for entry in body.get('urls', [])
            }
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if attempt < READ_RETRIES and error.code >= 500:
                time.sleep(READ_BACKOFF * attempt)
                continue
            raise SystemExit(
                f'PyPI {name} {version}: HTTP {error.code}'
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            if attempt < READ_RETRIES:
                time.sleep(READ_BACKOFF * attempt)
                continue
            raise SystemExit(f'PyPI {name} {version}: {error}') from error
    raise SystemExit(f'PyPI {name} {version}: unreachable')


def wait_for(uuid: str, what: str) -> None:
    r"""Poll to a terminal state. Running out of attempts is a failure, not a
    pass: an unfinished release has told us nothing."""
    for _ in range(POLL_ATTEMPTS):
        status = str(call('GET', f'/status/{uuid}').get('status', '')).lower()
        if status in TERMINAL_OK:
            print(f'  {what}: {status}')
            return
        if status in TERMINAL_BAD:
            raise SystemExit(f'{what} finished as {status} ({uuid})')
        if status not in PENDING:
            raise SystemExit(
                f'{what} reported an unrecognised status {status!r} ({uuid}). '
                f'Refusing to keep waiting on a state this script cannot read.'
            )
        time.sleep(POLL_SECONDS)
    raise SystemExit(
        f'{what} never reached a terminal state after '
        f'{POLL_ATTEMPTS * POLL_SECONDS}s ({uuid})'
    )


def submit(project_id: int, name: str, urls: list[str], upload: bool) -> str:
    response = call(
        'POST',
        f'/projects/{project_id}/releases',
        {
            'project_name': name,
            'payload': [
                {
                    'pic': PIC,
                    'job_type': 'wheel-release-job',
                    'url': url,
                    'upload': upload,
                }
                for url in urls
            ],
        },
    )
    uuid = response.get('uuid') or response.get('release_uuid')
    if not uuid:
        raise SystemExit(f'{name} returned no release uuid: {response}')
    return str(uuid)


def settled(name: str, version: str, expected: dict[str, str]) -> bool:
    r"""Whether PyPI already holds exactly this project's files at this version.

    Wholly absent is work to do, wholly present is work done and skipped. A
    partial or mismatched version raises instead: PyPI will not replace a file,
    so publishing over it cannot repair either case.
    """
    held = on_pypi(name, version)
    if held is None:
        return False
    missing = sorted(set(expected) - set(held))
    differing = sorted(
        filename
        for filename, digest in expected.items()
        if filename in held and held[filename] != digest
    )
    if differing:
        raise SystemExit(
            f'{name} {version} is on PyPI with different bytes than this run '
            f'built: {differing}. PyPI does not allow a file to be replaced, '
            f'so this release cannot be completed from these artifacts.'
        )
    if missing:
        raise SystemExit(
            f'{name} {version} is partly on PyPI, missing {missing}. '
            f'Publishing again cannot add them under the same version; see '
            f'the recovery section in RELEASING.md.'
        )
    return True


def verify(name: str, version: str, expected: dict[str, str]) -> None:
    r"""Confirm PyPI holds the bytes this run built. Kitmaker fetches whatever
    is at the asset URL, so without this the only claim is that a job reported
    success."""
    for _ in range(VERIFY_ATTEMPTS):
        held = on_pypi(name, version)
        if held is not None and not set(expected) - set(held):
            wrong = sorted(f for f, d in expected.items() if held[f] != d)
            if wrong:
                raise SystemExit(
                    f'{name}: PyPI holds different bytes for {wrong}'
                )
            print(
                f'  verified on PyPI: {name} {version} ({len(expected)} files)'
            )
            return
        time.sleep(VERIFY_SECONDS)
    raise SystemExit(
        f'{name} {version} did not appear in full on PyPI after '
        f'{VERIFY_ATTEMPTS * VERIFY_SECONDS}s'
    )


def main() -> None:
    dist = pathlib.Path('dist')
    built = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dist.iterdir()
        if path.name.endswith(('.whl', '.tar.gz'))
    }
    if not built:
        raise SystemExit('dist/ holds no distributions')
    version = version_of(next(iter(built)))

    # Only files this run built may be published. A rerun against an existing
    # release can see assets left by an earlier one, and `--clobber` replaces
    # only the names that collide, so the release listing alone is not a safe
    # description of what belongs to this release.
    assets: dict[str, str] = {}
    for line in pathlib.Path('assets.tsv').read_text().splitlines():
        if not line.strip():
            continue
        filename, url = line.split('\t')
        if filename in built:
            assets[filename] = url

    missing = set(built) - assets.keys()
    if missing:
        raise SystemExit(f'not attached to the release: {sorted(missing)}')

    by_project: dict[int, list[str]] = {}
    digests: dict[int, dict[str, str]] = {}
    for project_id, name in PROJECTS.items():
        mine = {
            filename: url
            for filename, url in assets.items()
            if canonical(distribution_of(filename)) == canonical(name)
        }
        if not mine:
            raise SystemExit(f'nothing in this release belongs to {name}')
        by_project[project_id] = sorted(mine.values())
        digests[project_id] = {f: built[f] for f in mine}

    claimed = sum(len(urls) for urls in by_project.values())
    if claimed != len(assets):
        raise SystemExit(
            f'{len(assets) - claimed} built file(s) match no Portal project'
        )

    print(
        f'{len(assets)} distributions at {version} across {len(by_project)} projects'
    )

    # Pass one: nothing publishes until every project has validated and every
    # dry run has finished. One already live at this version is left alone.
    todo: dict[int, str] = {}
    for project_id, name in PROJECTS.items():
        if settled(name, version, digests[project_id]):
            print(f'already on PyPI, skipping: {name} {version}')
            continue
        todo[project_id] = name
        actual = call('GET', f'/projects/{project_id}').get('name')
        if actual != name:
            raise SystemExit(
                f'Portal project {project_id} is {actual!r}, not {name!r}'
            )
        print(f'dry run: {name} ({len(by_project[project_id])} files)')
        wait_for(
            submit(project_id, name, by_project[project_id], False),
            f'dry run {name}',
        )

    if not todo:
        print('every distribution is already published at this version')
        return

    # Pass two: from here a failure can leave part of the release live, which
    # is why the whole of pass one had to pass first.
    published = []
    summary = pathlib.Path('kitmaker-summary.txt')
    try:
        for project_id, name in todo.items():
            print(f'publishing: {name}')
            wait_for(
                submit(project_id, name, by_project[project_id], True), name
            )
            verify(name, version, digests[project_id])
            published.append(name)
    finally:
        lines = [f'published: {n}' for n in published]
        lines += [
            f'NOT published: {n}' for n in todo.values() if n not in published
        ]
        lines += [
            f'already published: {n}'
            for n in PROJECTS.values()
            if n not in todo
        ]
        summary.write_text('\n'.join(lines) + '\n')
        print('\n'.join(lines))


if __name__ == '__main__':
    sys.exit(main())
