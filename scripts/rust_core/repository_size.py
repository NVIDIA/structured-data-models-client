#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bound the maintained checkout and require owners for oversized fixtures."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from contract_lock import unlocked_schemas

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path('contract/fixtures/maintained/consumers.json')
MAX_BYTES = 100_000_000
MAX_ADDED_LINES = 1_000_000
MAX_FIXTURE_BYTES = 250_000


def fixture(path: Path) -> bool:
    if path.parts[:2] == ('tests', 'golden') or path == Path(
        'tests/compatibility/manifest.json'
    ):
        return (
            False  # Generated API snapshots and the ledger have separate gates.
        )
    return path.suffix in {'.json', '.jsonl', '.arrow'} and (
        path.parts[0] == 'contract'
        or 'tests' in path.parts
        or 'testdata' in path.parts
        or 'fixtures' in path.parts
    )


def inventory(root: Path) -> list[Path]:
    files = (
        subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z'])
        .decode()
        .split('\0')
    )
    return [Path(name) for name in files if name and (root / name).is_file()]


def check(
    root: Path, paths: list[Path], base: str
) -> tuple[list[str], dict[str, int]]:
    failures = []
    sizes = {path: (root / path).stat().st_size for path in paths}
    total = sum(sizes.values())
    diff = subprocess.check_output(
        ['git', '-C', str(root), 'diff', '--numstat', base, '--'], text=True
    )
    added = sum(
        int(line.split('\t', 1)[0])
        for line in diff.splitlines()
        if not line.startswith('-\t')
    )
    if total >= MAX_BYTES:
        failures.append(
            f'{total:,} tracked bytes exceeds budget <{MAX_BYTES:,}'
        )
    if added >= MAX_ADDED_LINES:
        failures.append(
            f'{added:,} added lines exceeds budget <{MAX_ADDED_LINES:,} against {base}'
        )
    for path in unlocked_schemas(root):
        failures.append(f'unlocked historical schema: {path.relative_to(root)}')
    registry = json.loads((root / REGISTRY).read_text())
    for path, size in sizes.items():
        if path.parts[:2] in {
            ('contract', 'captures'),
            ('contract', 'reference'),
        }:
            failures.append(
                f'retired migration archive returned to checkout: {path}'
            )
        if fixture(path) and size > MAX_FIXTURE_BYTES:
            record = registry.get(str(path))
            if (
                not record
                or not record.get('reason')
                or not record.get('consumers')
            ):
                failures.append(
                    f'oversized fixture needs named consumers and reason: {path} ({size:,} bytes)'
                )
            elif size > record['max_bytes']:
                failures.append(
                    f'fixture exceeds reviewed size {record["max_bytes"]}: {path}'
                )
    for name, record in registry.items():
        if Path(name) not in sizes:
            failures.append(f'registered fixture is missing: {name}')
        for consumer in record['consumers']:
            if not (root / consumer).is_file():
                failures.append(f'{name}: missing consumer {consumer}')
    return failures, {
        'tracked_bytes': total,
        'added_lines': added,
        'tracked_files': len(paths),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--base',
        required=True,
        help='verified Git reference for the added-line budget',
    )
    args = parser.parse_args()
    try:
        failures, totals = check(ROOT, inventory(ROOT), args.base)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'repository size check cannot run: {error}', file=sys.stderr)
        return 1
    print(json.dumps(totals, sort_keys=True))
    for failure in failures:
        print(failure, file=sys.stderr)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
