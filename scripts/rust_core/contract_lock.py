#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The contract lock check: no entry merged into rust-core is rewritten.

``rust/crates/sdm-contract/schema/contract.lock`` holds one line per contract
version, ``<version> sha256:<hex> <schema file>``, and ``validated-query.lock``
beside it the same for the PQL validated-query format, which is versioned on
its own. The hash covers a projection of the schema that strips only the
annotation keywords (``description``, ``title``, ``examples``), keyword by
keyword, so a doc comment needs no new version while a property named
``description`` is still part of the shape (Phase 1 decision H7).

Entries present at the merge base with ``origin/rust-core`` are merged and
never rewritten; an entry not yet merged is a working version, rewritten at
each bump within a phase, and a lock holds at most one (decision H4). For
each lock, this check fails when

- a locked schema file no longer has its locked hash,
- an entry the lock at the merge base held is missing or different now,
- the lock holds more than one entry the merge base does not, or
- the lock at the base cannot be read, or there is no merge base.

That each committed schema is the one the types generate, and that each
current version has an entry, is checked by ``cargo test -p sdm-contract
--test schema_lock``. Run from anywhere in the repository; CI checks out
every branch so the merge base is there; ``--base`` names another base, and
the root Python test runs with ``--no-base``, since its checkout is shallow.
With no base to read, which entries are merged is not known: each but the
newest is taken as merged, so only the hashes are checked, and a phase's
working entry beside the merged ones passes (T2.1b: version 26 beside 25).
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    """Problems with the lock ``current``, given the lock at the merge base
    with rust-core as ``(commit, text)`` (none when the base has no lock)
    and a reader of committed schema files."""
    problems = []
    try:
        now = parse_lock(current)
    except ValueError as error:
        return [f'the lock: {error}']
    for version, (locked, name) in sorted(now.items()):
        body = read_schema(name)
        if body is None:
            problems.append(f'version {version}: {name} is missing')
            continue
        try:
            found = digest(body)
        except ValueError as error:
            problems.append(f'version {version}: {name} is not JSON: {error}')
            continue
        if found != locked:
            problems.append(
                f'version {version}: {name} no longer has its locked hash; '
                f'a schema change needs a new version'
            )
    merged: set[int] = set()
    for commit, text in base:
        try:
            then = parse_lock(text)
        except ValueError as error:
            problems.append(
                f'the lock at {commit[:12]} cannot be read: {error}'
            )
            continue
        for version, entry in sorted(then.items()):
            merged.add(version)
            if version not in now:
                problems.append(
                    f'version {version}, merged at {commit[:12]}, was removed'
                )
            elif now[version] != entry:
                problems.append(
                    f'version {version}, merged at {commit[:12]}, was rewritten'
                )
    working = sorted(set(now) - merged)
    if len(working) > 1:
        problems.append(
            f'versions {working} are not merged into rust-core; a lock holds '
            f'one working entry, rewritten at each bump within a phase'
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
    """The lock at the merge base of HEAD and ``base``, if it has one."""
    merge_base = _git('merge-base', 'HEAD', base).strip()
    try:
        _git('cat-file', '-e', f'{merge_base}:{lock}')
    except subprocess.CalledProcessError:
        return []  # the base predates the lock
    return [(merge_base, _git('show', f'{merge_base}:{lock}'))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--base',
        default='origin/rust-core',
        help='the branch whose merge base with HEAD is also checked',
    )
    parser.add_argument(
        '--no-base',
        action='store_true',
        help='check the hashes only, each entry but the newest taken as merged (a shallow checkout)',
    )
    args = parser.parse_args(argv)

    failed = False
    for lock in LOCKS:
        failed |= _check_lock(lock, args.base, args.no_base)
    return 1 if failed else 0


def _check_lock(lock: Path, base: str, no_base: bool) -> bool:
    """Checks one lock; whether it failed."""
    base_locks: list[tuple[str, str]] = []
    path = REPO_ROOT / lock
    current = path.read_text(encoding='utf-8') if path.is_file() else ''
    if no_base:
        # Each entry but the newest taken as merged, as the base would hold
        # it, so the hashes alone are checked; a lock that does not parse is
        # reported by ``check``.
        try:
            entries = parse_lock(current)
        except ValueError:
            entries = {}
        older = ''.join(
            f'{version} sha256:{locked} {name}\n'
            for version, (locked, name) in sorted(entries.items())
            if version != max(entries)
        )
        base_locks = [('(no base)', older)] if older else []
    else:
        try:
            base_locks = _base_lock(base, lock)
        except subprocess.CalledProcessError as error:
            print(
                f'{lock.name}: no merge base with {base}: '
                f'{error.stderr.strip() or error}',
                file=sys.stderr,
            )
            return True

    def read_schema(name: str) -> bytes | None:
        path = REPO_ROOT / SCHEMA_DIR / name
        return path.read_bytes() if path.is_file() else None

    problems = check(current, base_locks, read_schema)
    if not current and not problems:
        problems = [f'{lock} is missing']
    for problem in problems:
        print(f'{lock.name}: {problem}', file=sys.stderr)
    if problems:
        return True
    merged = set(parse_lock(base_locks[0][1])) if base_locks else set()
    entries = parse_lock(current)
    print(
        f'{lock.name}: {len(entries)} version(s) locked, '
        f'{len(merged)} at the merge base and unchanged, '
        f'{len(set(entries) - merged)} working'
        + (' (base not checked)' if no_base else '')
    )
    return False


if __name__ == '__main__':
    sys.exit(main())
