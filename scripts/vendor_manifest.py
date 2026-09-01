#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record and verify the exact bytes of a vendored tree.

A repo-wide identifier rename matches any text, including text inside a
vendored file. That has already turned a vendored artifact into a modified one
while its license still described it as upstream's bytes. A manifest of
per-file digests turns that silent edit into a failing test.

Write the manifest with ``record``, check it with ``verify``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MANIFEST_NAME = 'VENDOR_MANIFEST.json'

_SKIP_DIRS = {'__pycache__', '.pytest_cache', '.ruff_cache'}
_SKIP_SUFFIXES = {'.pyc', '.pyo'}


def iter_files(root: Path):
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.suffix in _SKIP_SUFFIXES:
            continue
        yield path


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    sha.update(path.read_bytes())
    return sha.hexdigest()


def build(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): digest(path) for path in iter_files(root)
    }


def record(root: Path, manifest_path: Path, upstream: dict[str, str]) -> int:
    payload = {
        'upstream': upstream,
        'tree': str(root.name),
        'files': build(root),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    print(f'recorded {len(payload["files"])} files to {manifest_path}')
    return 0


def verify(root: Path, manifest_path: Path) -> int:
    if not manifest_path.exists():
        print(f'no manifest at {manifest_path}', file=sys.stderr)
        return 1
    recorded = json.loads(manifest_path.read_text(encoding='utf-8'))['files']
    current = build(root)

    added = sorted(set(current) - set(recorded))
    removed = sorted(set(recorded) - set(current))
    changed = sorted(
        name
        for name in set(current) & set(recorded)
        if current[name] != recorded[name]
    )

    for name in added:
        print(f'ADDED    {name}', file=sys.stderr)
    for name in removed:
        print(f'REMOVED  {name}', file=sys.stderr)
    for name in changed:
        print(f'CHANGED  {name}', file=sys.stderr)

    if added or removed or changed:
        print(
            '\nThe vendored tree no longer matches its manifest. If a rename '
            'did this, revert it and exclude the vendored path. If the bump is '
            'deliberate, re-record with: '
            'python scripts/vendor_manifest.py record',
            file=sys.stderr,
        )
        return 1
    print(f'verified {len(current)} files')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('record', 'verify'))
    parser.add_argument(
        '--tree',
        default='packages/sdm-pql/src/predictive_query',
        help='Vendored tree to record or verify.',
    )
    parser.add_argument(
        '--manifest',
        default=None,
        help=f'Manifest path. Defaults to <tree>/../{MANIFEST_NAME}.',
    )
    parser.add_argument('--upstream-repository', default='')
    parser.add_argument('--upstream-revision', default='')
    parser.add_argument('--upstream-version', default='')
    args = parser.parse_args()

    root = Path(args.tree).resolve()
    if not root.is_dir():
        print(f'not a directory: {root}', file=sys.stderr)
        return 1
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else root.parent.parent / MANIFEST_NAME
    )

    if args.action == 'record':
        return record(
            root,
            manifest_path,
            {
                'repository': args.upstream_repository,
                'revision': args.upstream_revision,
                'version': args.upstream_version,
            },
        )
    return verify(root, manifest_path)


if __name__ == '__main__':
    raise SystemExit(main())
