# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Pin the RelBench datasets ``Graph.from_relbench`` loads.

RelBench 3 moved its data from a download server with a published hash file
to Hugging Face dataset repositories, which carry no hash file and are read at
whatever ``main`` holds today. This records one commit per repository, and for
every dataset at that commit the schema its ``manifest.yaml`` declares and the
SHA-256 of each table's parquet file. The loader trusts only what is recorded
here, so a file that changed upstream fails its check rather than loading.

A parquet file is stored in Git LFS, whose pointer names the content's SHA-256,
so that digest is the LFS object id at the pinned commit. A manifest is a plain
Git blob; it is downloaded and checked against its Git object id before it is
parsed.

Run with no arguments to pin each repository's current ``main``, or pass
``repo=revision`` to pin a given commit::

    python scripts/pin_relbench.py
    python scripts/pin_relbench.py stanford-star/tgb=<commit>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

import yaml

REPOSITORIES = (
    'stanford-star/relbench-v1',
    'stanford-star/relbench-v2-extra',
    'stanford-star/dbinfer',
    'stanford-star/tgb',
)

HUB = 'https://huggingface.co'

_NEXT_LINK = re.compile(r'<([^>]+)>\s*;\s*rel="next"')

OUTPUT = (
    Path(__file__).resolve().parents[1]
    / 'packages/structured-data-models-relational-engine/src/sdm_relational_engine/rfm/assets'
    / 'relbench_registry.json'
)


def _get(url: str) -> tuple[bytes, str]:
    with urlopen(url) as response:
        return response.read(), response.headers.get('Link') or ''


def _head_revision(repository: str) -> str:
    body, _ = _get(f'{HUB}/api/datasets/{repository}/revision/main')
    return str(json.loads(body)['sha'])


def _tree(repository: str, revision: str) -> list[dict[str, Any]]:
    url: str | None = (
        f'{HUB}/api/datasets/{repository}/tree/{revision}?recursive=true'
    )
    entries: list[dict[str, Any]] = []
    while url is not None:
        body, link = _get(url)
        entries.extend(json.loads(body))
        # The listing is paged; the next page is only named in this header,
        # which may list other relations (prev, first) alongside it.
        found = _NEXT_LINK.search(link)
        url = found.group(1) if found else None
    return [entry for entry in entries if entry['type'] == 'file']


def _manifest(repository: str, revision: str, entry: dict[str, Any]) -> Any:
    path = quote(entry['path'])
    body, _ = _get(f'{HUB}/datasets/{repository}/resolve/{revision}/{path}')
    # Git's object id for the blob, which the tree listing names.
    blob = hashlib.sha1(b'blob %d\0' % len(body) + body).hexdigest()
    if blob != entry['oid']:
        sys.exit(
            f'{repository}@{revision}:{entry["path"]} does not match its Git '
            f'object id ({blob} != {entry["oid"]})'
        )
    return yaml.safe_load(body)


def _pin(repository: str, revision: str) -> dict[str, Any]:
    files = {entry['path']: entry for entry in _tree(repository, revision)}
    names = sorted(
        path.split('/')[0]
        for path in files
        if path.count('/') == 1 and path.endswith('/manifest.yaml')
    )
    datasets: dict[str, Any] = {}
    for name in names:
        manifest = _manifest(
            repository, revision, files[f'{name}/manifest.yaml']
        )
        tables: dict[str, Any] = {}
        for table, spec in manifest['tables'].items():
            entry = files.get(f'{name}/db/{table}.parquet')
            if entry is None or entry.get('lfs') is None:
                sys.exit(
                    f'{repository}@{revision}: {name} declares table {table!r} '
                    f'but holds no LFS file db/{table}.parquet for it'
                )
            tables[table] = {
                'sha256': entry['lfs']['oid'],
                'primary_key': spec.get('pkey'),
                'time_column': spec.get('time_col'),
                'foreign_keys': dict(spec.get('fkeys') or {}),
            }
        datasets[name] = {'repository': repository, 'tables': tables}
        print(f'  {name}: {len(tables)} tables', file=sys.stderr)
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        'pins',
        nargs='*',
        metavar='REPOSITORY=REVISION',
        help='pin a repository at this commit rather than at its main',
    )
    args = parser.parse_args()
    given = dict(pin.split('=', 1) for pin in args.pins)
    unknown = set(given) - set(REPOSITORIES)
    if unknown:
        sys.exit(f'not a RelBench repository: {sorted(unknown)}')

    revisions: dict[str, str] = {}
    datasets: dict[str, Any] = {}
    for repository in REPOSITORIES:
        revision = given.get(repository) or _head_revision(repository)
        print(f'{repository}@{revision}', file=sys.stderr)
        revisions[repository] = revision
        for name, dataset in _pin(repository, revision).items():
            if name in datasets:
                sys.exit(
                    f'{name} is in both {datasets[name]["repository"]} and '
                    f'{repository}; a bare name would be ambiguous'
                )
            datasets[name] = dataset

    registry = {
        'generated_by': 'scripts/pin_relbench.py',
        'revisions': revisions,
        'datasets': dict(sorted(datasets.items())),
    }
    OUTPUT.write_text(json.dumps(registry, indent=1) + '\n')
    print(f'wrote {len(datasets)} datasets to {OUTPUT}', file=sys.stderr)


if __name__ == '__main__':
    main()
