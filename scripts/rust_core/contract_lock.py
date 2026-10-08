#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The contract lock check: existing version entries are never rewritten.

``rust/crates/sdm-contract/schema/contract.lock`` holds one line per contract
version, ``<version> sha256:<hex> <schema file>``. This check fails when

- a locked schema file no longer has its locked hash,
- an entry any earlier commit's lock held is missing or different now,
  including the lock at the merge base with ``origin/rust-core``, which
  holds even if the branch's own history is rewritten, or
- an earlier lock, or the base, cannot be read.

That the committed schema is the one the types generate, and that the current
version has an entry, is checked by ``cargo test -p sdm-contract --test
schema_lock``. Run from anywhere in the repository; it reads the full history
of the lock file and the base branch, so CI checks out every commit of
every branch; ``--base`` names another base, and the root Python test runs
without one, since its checkout is shallow.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = Path('rust/crates/sdm-contract/schema')
LOCK = SCHEMA_DIR / 'contract.lock'

Entries = dict[int, tuple[str, str]]


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
    earlier: Iterable[tuple[str, str]],
    read_schema: Callable[[str], bytes | None],
) -> list[str]:
    """Problems with the lock ``current``, given the lock texts of earlier
    commits as ``(commit, text)`` and a reader of committed schema files."""
    problems = []
    try:
        now = parse_lock(current)
    except ValueError as error:
        return [f'contract.lock: {error}']
    for version, (digest, name) in sorted(now.items()):
        body = read_schema(name)
        if body is None:
            problems.append(f'version {version}: {name} is missing')
        elif hashlib.sha256(body).hexdigest() != digest:
            problems.append(
                f'version {version}: {name} no longer has its locked hash; '
                f'a schema change needs a new contract version'
            )
    for commit, text in earlier:
        try:
            then = parse_lock(text)
        except ValueError as error:
            problems.append(
                f'the lock at {commit[:12]} cannot be read: {error}'
            )
            continue
        for version, entry in sorted(then.items()):
            if version not in now:
                problems.append(
                    f'version {version}, locked at {commit[:12]}, was removed'
                )
            elif now[version] != entry:
                problems.append(
                    f'version {version}, locked at {commit[:12]}, was rewritten'
                )
    return problems


def _git(*args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _base_lock(base: str) -> list[tuple[str, str]]:
    """The lock at the merge base of HEAD and ``base``, if it has one."""
    merge_base = _git('merge-base', 'HEAD', base).strip()
    try:
        _git('cat-file', '-e', f'{merge_base}:{LOCK}')
    except subprocess.CalledProcessError:
        return []  # the base predates the lock
    return [(merge_base, _git('show', f'{merge_base}:{LOCK}'))]


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
        help='check the branch history only (a shallow checkout)',
    )
    args = parser.parse_args(argv)

    commits = _git('log', '--format=%H', '--', str(LOCK)).split()
    earlier = []
    for commit in commits:
        try:
            earlier.append((commit, _git('show', f'{commit}:{LOCK}')))
        except subprocess.CalledProcessError:
            continue  # the commit deleted the lock
    if not args.no_base:
        try:
            earlier.extend(_base_lock(args.base))
        except subprocess.CalledProcessError as error:
            print(
                f'contract lock: no merge base with {args.base}: '
                f'{error.stderr.strip() or error}',
                file=sys.stderr,
            )
            return 1

    def read_schema(name: str) -> bytes | None:
        path = REPO_ROOT / SCHEMA_DIR / name
        return path.read_bytes() if path.is_file() else None

    current = (REPO_ROOT / LOCK).read_text(encoding='utf-8')
    problems = check(current, earlier, read_schema)
    for problem in problems:
        print(f'contract lock: {problem}', file=sys.stderr)
    if problems:
        return 1
    print(
        f'contract lock: {len(parse_lock(current))} version(s) locked, '
        f'none rewritten across {len(earlier)} earlier lock(s)'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
