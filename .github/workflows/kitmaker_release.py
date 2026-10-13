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
import http.client
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
# The five states Portal documents, and no more. An unrecognised one is kept
# waiting rather than failed: the docs say to continue bounded polling, record
# it and escalate if progress stops, and specifically not to read an
# unfamiliar nonterminal value as success.
PENDING = {'pending', 'processing', 'in_progress'}
TERMINAL_OK = {'completed'}
TERMINAL_BAD = {'failed'}

POLL_SECONDS = 30
POLL_ATTEMPTS = 60

# Reads retry, submissions never: losing pass two to one connection reset is
# the half-published state the two passes exist to prevent, but a repeated POST
# could publish twice.
READ_RETRIES = 5
READ_BACKOFF = 4

# Worth another go on a read. 404 only on `/status`, where it is normal for a
# second or two after a submit, before Portal knows the uuid; on
# `/projects/{id}` the same code means the project is not there and repeating
# it just delays the message.
RETRY_CODES = {408, 429}
RETRY_404_ON = '/status/'

# A 200 carrying a CDN error page raises `ValueError` from `json.loads`, and a
# truncated body raises `IncompleteRead`, which is an `HTTPException` and not
# an `OSError`. Neither is an answer; both are worth retrying.
TRANSIENT = (
    urllib.error.URLError,
    TimeoutError,
    OSError,
    ValueError,
    http.client.HTTPException,
)

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
            # Most 4xx is an answer rather than a blip, so it is not repeated.
            retryable = (
                error.code >= 500
                or error.code in RETRY_CODES
                or (error.code == 404 and path.startswith(RETRY_404_ON))
            )
            if attempt < attempts and retryable:
                time.sleep(READ_BACKOFF * attempt)
                continue
            raise SystemExit(
                f'{method} {path} returned {error.code}: {detail}'
            ) from error
        except TRANSIENT as error:
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
            # 404 is PyPI's answer for a version it does not have, not a fault.
            if error.code == 404:
                return None
            if attempt < READ_RETRIES and (
                error.code >= 500 or error.code in RETRY_CODES
            ):
                time.sleep(READ_BACKOFF * attempt)
                continue
            raise SystemExit(
                f'PyPI {name} {version}: HTTP {error.code}'
            ) from error
        except TRANSIENT as error:
            if attempt < READ_RETRIES:
                time.sleep(READ_BACKOFF * attempt)
                continue
            raise SystemExit(f'PyPI {name} {version}: {error}') from error
    raise SystemExit(f'PyPI {name} {version}: unreachable')


def wait_for(uuid: str, what: str) -> None:
    r"""Poll to a terminal state. Running out of attempts is a failure, not a
    pass: an unfinished release has told us nothing."""
    unfamiliar = set()
    for _ in range(POLL_ATTEMPTS):
        status = str(call('GET', f'/status/{uuid}').get('status', '')).lower()
        if status in TERMINAL_OK:
            print(f'  {what}: {status}')
            return
        if status in TERMINAL_BAD:
            raise SystemExit(f'{what} finished as {status} ({uuid})')
        if status not in PENDING and status not in unfamiliar:
            unfamiliar.add(status)
            print(f'  {what}: unrecognised status {status!r}, still waiting')
        time.sleep(POLL_SECONDS)
    seen = f', saw {sorted(unfamiliar)}' if unfamiliar else ''
    raise SystemExit(
        f'{what} never reached a terminal state after '
        f'{POLL_ATTEMPTS * POLL_SECONDS}s ({uuid}{seen})'
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
                    # Omitting this picks a strategy by size: under 1 GiB gives
                    # `upload-both`, at or above it gives `wheel-stub`, which
                    # cannot process an sdist. Everything here is far below the
                    # threshold, so the default would be right, but a size
                    # cliff between a wheel and its sdist is not something to
                    # leave implicit in a release path.
                    'mirroring_strategy': 'upload-both',
                }
                for url in urls
            ],
        },
    )
    uuid = response.get('release_uuid')
    if not uuid:
        raise SystemExit(f'{name} returned no release_uuid: {response}')
    return str(uuid)


def outstanding(
    name: str, version: str, expected: dict[str, str]
) -> dict[str, str]:
    r"""Which of this project's files are not on PyPI yet, as ``{filename:
    sha256}``.

    Empty means the version is already published with the bytes this run built,
    and is skipped; a partial version returns the rest, because PyPI accepts
    files into a version one at a time. Two states raise instead, since neither
    can be repaired by publishing more: a file held under different bytes, and a
    file held that this run did not build.
    """
    held = on_pypi(name, version)
    if held is None:
        return dict(expected)
    differing = sorted(
        filename
        for filename, digest in expected.items()
        if filename in held and held[filename] != digest
    )
    if differing:
        raise SystemExit(
            f'{name} {version} is on PyPI with different bytes than this run '
            f'built: {differing}. PyPI does not allow a file to be replaced.'
        )
    unknown = sorted(set(held) - set(expected))
    if unknown:
        raise SystemExit(
            f'{name} {version} is on PyPI with files this run did not build: '
            f'{unknown}.'
        )
    return {
        filename: digest
        for filename, digest in expected.items()
        if filename not in held
    }


def verify(name: str, version: str, expected: dict[str, str]) -> None:
    r"""Confirm PyPI holds exactly the bytes this run built. Kitmaker fetches
    whatever is at the asset URL, so without this the only claim is that a job
    reported success."""
    for _ in range(VERIFY_ATTEMPTS):
        held = on_pypi(name, version)
        if held is not None:
            unknown = sorted(set(held) - set(expected))
            if unknown:
                raise SystemExit(
                    f'{name} {version} on PyPI holds files this run did not '
                    f'build: {unknown}'
                )
            wrong = sorted(
                filename
                for filename, digest in expected.items()
                if filename in held and held[filename] != digest
            )
            if wrong:
                raise SystemExit(
                    f'{name}: PyPI holds different bytes for {wrong}'
                )
            if not set(expected) - set(held):
                print(
                    f'  verified on PyPI: {name} {version} '
                    f'({len(expected)} files)'
                )
                return
        time.sleep(VERIFY_SECONDS)
    raise SystemExit(
        f'{name} {version} did not appear in full on PyPI after '
        f'{VERIFY_ATTEMPTS * VERIFY_SECONDS}s'
    )


def pypi_state(name: str, version: str, expected: dict[str, str]) -> str:
    r"""What PyPI says about this project, in a phrase.

    Portal's own answer is not used here, and deliberately. Its batches are not
    atomic, so a failure can follow files that already landed, and a rejection
    is not proof that nothing did. Only the index knows, so the summary asks it
    rather than repeating what Portal said.

    Every failure is caught, a malformed answer and a `SystemExit` from the
    retry loop included: `SystemExit` is a `BaseException`, so catching
    `Exception` alone would miss the one this module raises most. This runs from
    a ``finally`` while an exception is already on its way out, and raising here
    would replace the reason the release stopped with the reason the summary
    could not be written, and leave no summary either.
    """
    try:
        held = on_pypi(name, version)
        if held is None:
            return 'not on PyPI'
        wrong = sorted(
            filename
            for filename, digest in expected.items()
            if filename in held and held[filename] != digest
        )
        if wrong:
            return f'on PyPI with UNEXPECTED bytes for {wrong}'
        unknown = sorted(map(str, set(held) - set(expected)))
        if unknown:
            return f'on PyPI with files this run did not build: {unknown}'
        absent = sorted(set(expected) - set(held))
        if absent:
            return f'PARTLY on PyPI, missing {absent}'
        return 'published'
    except (Exception, SystemExit) as error:  # noqa: BLE001 - see docstring
        return f'UNKNOWN, could not ask PyPI ({error})'


def write_summary(
    version: str,
    digests: dict[str, dict[str, str]],
    requested: list[str],
    skipped: list[str],
    partial: list[str],
) -> None:
    r"""What is live and what is not, written whatever happens.

    A project whose upload was asked for is described by what PyPI holds now,
    however that request ended, and so is one an earlier run left part-way:
    `not reached` would otherwise hide a version that is already partly live. A
    project neither of those is reported from what this run did.
    """
    lines = []
    for name in PROJECTS.values():
        if name in skipped:
            state = 'already published before this run'
        elif name in requested or name in partial:
            state = pypi_state(name, version, digests.get(name, {}))
        else:
            state = 'not reached'
        lines.append(f'{name}: {state}')
    text = '\n'.join(lines) + '\n'
    try:
        pathlib.Path('kitmaker-summary.txt').write_text(text)
    except (Exception, SystemExit) as error:  # noqa: BLE001 - see docstring
        print(f'could not write kitmaker-summary.txt: {error}')
    print(text, end='')


def main() -> None:
    version = ''
    by_name: dict[str, dict[str, str]] = {}
    requested: list[str] = []
    skipped: list[str] = []
    partial: list[str] = []
    try:
        built = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in pathlib.Path('dist').iterdir()
            if path.name.endswith(('.whl', '.tar.gz'))
        }
        if not built:
            raise SystemExit('dist/ holds no distributions')
        # The workflow gate checks this too. This is the thing that submits, so
        # it refuses as well: one version here is the version every file is
        # published under.
        versions = {version_of(filename) for filename in built}
        if len(versions) != 1:
            raise SystemExit(
                f'distributions disagree on the version: {sorted(versions)}'
            )
        version = versions.pop()

        # Only files this run built may be published. A rerun against an
        # existing release can see assets left by an earlier one, so the
        # release listing alone is not a safe description of this release.
        assets: dict[str, str] = {}
        stale: list[str] = []
        for line in pathlib.Path('assets.tsv').read_text().splitlines():
            if not line.strip():
                continue
            filename, url, digest = (line.split('\t') + ['', ''])[:3]
            if filename not in built:
                continue
            # Kitmaker fetches the asset, not the local file, and a published
            # file cannot be recalled. An asset left by an earlier run that
            # built different bytes would be published as this release and only
            # then fail verification, which is far too late. The digest comes
            # from the release listing, so this costs nothing.
            if digest.removeprefix('sha256:') != built[filename]:
                stale.append(
                    f'{filename} (release has {digest or "no digest"})'
                )
                continue
            assets[filename] = url
        if stale:
            raise SystemExit(
                'release assets are not the bytes this run built: '
                f'{sorted(stale)}. Re-run failed jobs rather than all jobs, '
                'so the original artifact is reused.'
            )

        absent = set(built) - assets.keys()
        if absent:
            raise SystemExit(f'not attached to the release: {sorted(absent)}')

        urls: dict[int, dict[str, str]] = {}
        for project_id, name in PROJECTS.items():
            mine = {
                filename: url
                for filename, url in assets.items()
                if canonical(distribution_of(filename)) == canonical(name)
            }
            if not mine:
                raise SystemExit(f'nothing in this release belongs to {name}')
            urls[project_id] = mine
            by_name[name] = {filename: built[filename] for filename in mine}

        claimed = sum(len(mine) for mine in urls.values())
        if claimed != len(assets):
            raise SystemExit(
                f'{len(assets) - claimed} built file(s) match no Portal project'
            )

        print(
            f'{len(assets)} distributions at {version} '
            f'across {len(PROJECTS)} projects'
        )

        # Pass one: nothing publishes until every project has validated and
        # every dry run has finished. A project already on PyPI at this version
        # is left alone, and a part-published one is reduced to what is missing.
        todo: dict[int, tuple[str, list[str]]] = {}
        for project_id, name in PROJECTS.items():
            pending = outstanding(name, version, by_name[name])
            if not pending:
                print(f'already on PyPI, skipping: {name} {version}')
                skipped.append(name)
                continue
            if len(pending) != len(by_name[name]):
                # An earlier run left this version part-way. Recorded so a
                # release that stops before reaching it does not describe it
                # as untouched.
                partial.append(name)
            todo[project_id] = (
                name,
                sorted(
                    url
                    for filename, url in urls[project_id].items()
                    if filename in pending
                ),
            )
            actual = call('GET', f'/projects/{project_id}').get('name')
            if actual != name:
                raise SystemExit(
                    f'Portal project {project_id} is {actual!r}, not {name!r}'
                )
            print(f'dry run: {name} ({len(pending)} files)')
            wait_for(
                submit(project_id, name, todo[project_id][1], False),
                f'dry run {name}',
            )

        if not todo:
            print('every distribution is already published at this version')
            return

        # Pass two: from here a failure can leave part of the release live,
        # which is why the whole of pass one had to pass first.
        for project_id, (name, project_urls) in todo.items():
            print(f'publishing: {name}')
            # Recorded before it is asked for, because from this line on the
            # files may be live whatever Portal goes on to say.
            requested.append(name)
            wait_for(submit(project_id, name, project_urls, True), name)
            verify(name, version, by_name[name])
    finally:
        write_summary(version, by_name, requested, skipped, partial)


if __name__ == '__main__':
    sys.exit(main())
