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

Run from the workflow with ``assets.tsv`` already written and ``dist/``
downloaded. Exits non-zero on any failure; the workflow has no other way to
know this went wrong.
"""

import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

API = 'https://kitmaker-portal.nvidia.com/api/v0'

# Portal project ids, created once and owned by the PIC. Each is checked
# against the name it is expected to carry before anything is sent: an id that
# has been renamed or reassigned would otherwise be trusted silently.
PROJECTS = {
    6898: 'structured-data-models-client',
    6899: 'structured-data-models-api',
    6900: 'structured-data-models-pql',
    6901: 'structured-data-models-connectors',
    6902: 'structured-data-models-relational-engine',
}

TERMINAL_OK = {'completed', 'success', 'succeeded'}
TERMINAL_BAD = {'failed', 'error', 'cancelled', 'canceled'}
POLL_SECONDS = 30
POLL_ATTEMPTS = 60

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


def call(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
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
        raise SystemExit(
            f'{method} {path} returned {error.code}: {detail}'
        ) from error
    except urllib.error.URLError as error:
        raise SystemExit(f'{method} {path} failed: {error.reason}') from error


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


def main() -> None:
    built = {
        path.name
        for path in pathlib.Path('dist').iterdir()
        if path.name.endswith(('.whl', '.tar.gz'))
    }
    if not built:
        raise SystemExit('dist/ holds no distributions')

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

    missing = built - assets.keys()
    if missing:
        raise SystemExit(f'not attached to the release: {sorted(missing)}')

    by_project: dict[int, list[str]] = {}
    for project_id, name in PROJECTS.items():
        urls = sorted(
            url
            for filename, url in assets.items()
            if canonical(distribution_of(filename)) == canonical(name)
        )
        if not urls:
            raise SystemExit(f'nothing in this release belongs to {name}')
        by_project[project_id] = urls

    claimed = sum(len(urls) for urls in by_project.values())
    if claimed != len(assets):
        raise SystemExit(
            f'{len(assets) - claimed} built file(s) match no Portal project'
        )

    print(f'{len(assets)} distributions across {len(by_project)} projects')

    # Pass one: nothing publishes until every project has validated and every
    # dry run has finished.
    for project_id, name in PROJECTS.items():
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

    # Pass two: from here a failure can leave part of the release live, which
    # is why the whole of pass one had to pass first.
    published = []
    summary = pathlib.Path('kitmaker-summary.txt')
    try:
        for project_id, name in PROJECTS.items():
            print(f'publishing: {name}')
            wait_for(
                submit(project_id, name, by_project[project_id], True), name
            )
            published.append(name)
    finally:
        lines = [f'published: {n}' for n in published]
        lines += [
            f'NOT published: {n}'
            for n in PROJECTS.values()
            if n not in published
        ]
        summary.write_text('\n'.join(lines) + '\n')
        print('\n'.join(lines))


if __name__ == '__main__':
    sys.exit(main())
