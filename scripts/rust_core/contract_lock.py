#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate current protocol schemas and require a version bump for shape changes.

Only the current contract and PQL schema are retained. Their generated locks
record shape hashes, excluding documentation annotations. With --base, compare
against the actual target revision's newest locked version: reject rollback or
a changed schema under the same version. Git preserves historical schemas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = Path('rust/crates/sdm-contract/schema')
LOCK = SCHEMA_DIR / 'contract.lock'
LOCKS = (LOCK, SCHEMA_DIR / 'validated-query.lock')

# Keywords whose value is documentation only.
ANNOTATIONS = frozenset({'description', 'title', 'examples'})
# Keywords whose value maps names to schemas: the names are not keywords.
NAMED = frozenset(
    {
        'properties',
        'patternProperties',
        '$defs',
        'definitions',
        'dependentSchemas',
    }
)
# Keywords whose value is data, not a schema.
DATA = frozenset({'enum', 'const', 'default'})

Entries = dict[int, tuple[str, str]]


def project(node: Any) -> Any:
    """The schema ``node`` without its annotation keywords."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key in ANNOTATIONS:
                continue
            if key in NAMED and isinstance(value, dict):
                out[key] = {name: project(sub) for name, sub in value.items()}
            elif key in DATA:
                out[key] = value
            else:
                out[key] = project(value)
        return out
    if isinstance(node, list):
        return [project(item) for item in node]
    return node


def digest(body: bytes) -> str:
    """The lock hash of a schema file: the SHA-256 of its projection, as
    compact JSON with sorted keys. ``schema_lock.rs`` computes the same."""
    projected = project(json.loads(body))
    text = json.dumps(
        projected, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    )
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def parse_lock(text: str) -> Entries:
    """The lock's entries, ``{version: (sha256, file)}``."""
    entries: Entries = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) != 3 or not parts[1].startswith('sha256:'):
            raise ValueError(f'not a lock entry: {line!r}')
        version = int(parts[0])
        if version in entries:
            raise ValueError(f'version {version} is locked twice')
        entries[version] = (parts[1].removeprefix('sha256:'), parts[2])
    return entries


def check(
    current: str,
    base: Iterable[tuple[str, str]],
    read_schema: Callable[[str], bytes | None],
) -> list[str]:
    """Check one current entry, its file, and the target revision's contract."""
    problems = []
    try:
        now = parse_lock(current)
    except ValueError as error:
        return [f'the lock: {error}']
    if len(now) != 1:
        return ['the lock must hold exactly one current version']
    version, (locked, name) = next(iter(now.items()))
    if not re.fullmatch(
        rf'(?:sdm-contract|pql-validated-query)-{version}\.json', name
    ):
        problems.append(
            f'version {version}: schema filename does not match its version'
        )
    body = read_schema(name)
    if body is None:
        problems.append(f'version {version}: {name} is missing')
    else:
        try:
            found = digest(body)
        except ValueError as error:
            problems.append(f'version {version}: {name} is not JSON: {error}')
        else:
            if found != locked:
                problems.append(
                    f'version {version}: {name} no longer has its locked hash; a schema change needs a new version'
                )
    for commit, text in base:
        try:
            then = parse_lock(text)
            if not then:
                raise ValueError('empty lock')
        except ValueError as error:
            problems.append(
                f'the lock at {commit[:12]} cannot be read: {error}'
            )
            continue
        previous = max(then)
        if version < previous:
            problems.append(
                f'version {version}: rollback from version {previous} at {commit[:12]}'
            )
        elif version == previous and now[version] != then[previous]:
            problems.append(
                f'version {version}: schema changed without a version bump from {commit[:12]}'
            )
    return problems


def _git(*args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _base_lock(base: str, lock: Path) -> list[tuple[str, str]]:
    """Read the actual target revision, not an older merge base."""
    commit = _git('rev-parse', '--verify', f'{base}^{{commit}}').strip()
    try:
        _git('cat-file', '-e', f'{commit}:{lock}')
    except subprocess.CalledProcessError:
        return []  # the target predates the lock
    return [(commit, _git('show', f'{commit}:{lock}'))]


def unlocked_schemas(root: Path = REPO_ROOT) -> list[Path]:
    """Obsolete versioned files only; embedded payloads are never candidates."""
    directory = root / SCHEMA_DIR
    locked = {
        name
        for lock in LOCKS
        for _, name in parse_lock(
            (root / lock).read_text(encoding='utf-8')
        ).values()
    }
    return sorted(
        path
        for path in directory.glob('*.json')
        if re.fullmatch(
            r'(?:sdm-contract|pql-validated-query)-[0-9]+\.json', path.name
        )
        and path.name not in locked
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--base',
        default='origin/rust-core',
        help='target revision whose latest contract must not be rewritten or rolled back',
    )
    parser.add_argument(
        '--no-base',
        action='store_true',
        help='check current schema hashes only, without a target revision',
    )
    parser.add_argument(
        '--prune-unlocked',
        action='store_true',
        help='remove obsolete versioned schema files absent from both validated locks',
    )
    args = parser.parse_args(argv)

    failed = False
    for lock in LOCKS:
        failed |= _check_lock(lock, args.base, args.no_base)
    if args.prune_unlocked and not failed:
        for path in unlocked_schemas():
            path.unlink()
            print(f'removed unlocked schema {path.name}')
    return 1 if failed else 0


def _check_lock(lock: Path, base: str, no_base: bool) -> bool:
    """Check one generated current lock and, unless disabled, its target."""
    path = REPO_ROOT / lock
    current = path.read_text(encoding='utf-8') if path.is_file() else ''
    try:
        base_locks = [] if no_base else _base_lock(base, lock)
    except subprocess.CalledProcessError as error:
        print(
            f'{lock.name}: cannot read base {base}: {error.stderr.strip() or error}',
            file=sys.stderr,
        )
        return True

    def read_schema(name: str) -> bytes | None:
        path = REPO_ROOT / SCHEMA_DIR / name
        return path.read_bytes() if path.is_file() else None

    problems = check(current, base_locks, read_schema)
    for problem in problems:
        print(f'{lock.name}: {problem}', file=sys.stderr)
    if problems:
        return True
    version = next(iter(parse_lock(current)))
    print(
        f'{lock.name}: current version {version} verified'
        + (' (base not checked)' if no_base else f' against {base}')
    )
    return False


if __name__ == '__main__':
    sys.exit(main())
