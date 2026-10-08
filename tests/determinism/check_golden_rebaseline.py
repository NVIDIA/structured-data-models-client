#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the one intended re-baseline of the recorded goldens against a base.

    python tests/determinism/check_golden_rebaseline.py <base commit>

Passes when, comparing the base commit with ``HEAD``:

* ``tests/golden/public_api.json`` is unchanged;
* ``tests/golden/wire_payloads.json`` matches what
  ``scripts/snapshot_wire_payloads.py`` produces now;
* the ``## Unreleased`` section of ``CHANGELOG.md`` gained text; and, if any
  recorded payload changed, appeared or disappeared, that section names
  ``wire_payloads.json`` and every such scenario.

Whether the changelog explains *why* is for a reviewer to read.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
WIRE = 'tests/golden/wire_payloads.json'
PUBLIC_API = 'tests/golden/public_api.json'


def _git(*args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(REPO), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _unreleased(changelog: str) -> str:
    match = re.search(
        r'^## Unreleased[^\n]*\n(.*?)(?=^## |\Z)',
        changelog,
        flags=re.MULTILINE | re.DOTALL,
    )
    return match.group(1) if match else ''


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    base = argv[0]
    problems: list[str] = []

    if _git('diff', '--name-only', base, 'HEAD', '--', PUBLIC_API).strip():
        problems.append(f'{PUBLIC_API} changed; it must not')

    snapshot = subprocess.run(
        [sys.executable, str(REPO / 'scripts' / 'snapshot_wire_payloads.py')],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    if snapshot.returncode != 0:
        problems.append(
            f'{WIRE} does not match what the snapshot script produces:\n'
            f'{snapshot.stdout}{snapshot.stderr}'
        )

    before = json.loads(_git('show', f'{base}:{WIRE}'))
    after = json.loads((REPO / WIRE).read_text())
    changed = sorted(
        name
        for name in set(before) | set(after)
        if before.get(name) != after.get(name)
    )

    old_notes = _unreleased(_git('show', f'{base}:CHANGELOG.md'))
    new_notes = _unreleased((REPO / 'CHANGELOG.md').read_text())
    if not new_notes:
        problems.append('CHANGELOG.md has no "## Unreleased" section')
    elif new_notes.strip() == old_notes.strip():
        problems.append(
            'the Unreleased section of CHANGELOG.md gained no entry for '
            'the change to sampled output'
        )
    if new_notes and changed:
        if 'wire_payloads.json' not in new_notes:
            problems.append(
                'the Unreleased section of CHANGELOG.md does not name '
                'wire_payloads.json'
            )
        unnamed = [name for name in changed if name not in new_notes]
        if unnamed:
            problems.append(
                'the Unreleased section of CHANGELOG.md does not name these '
                'changed scenarios: ' + ', '.join(unnamed)
            )

    print(f'scenarios whose recorded payload changed: {changed or "none"}')
    for problem in problems:
        print(f'FAIL: {problem}', file=sys.stderr)
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
