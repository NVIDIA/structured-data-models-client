# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Replay recordings through the binding where today's Python can't be
imported (P85.independent).

    python _redrive.py [--root DIR] [--only NAME ...]

Every recording under ``DIR`` (by default ``tests/parity/requests``) is
replayed through ``sdm_core._native`` in this process, where pandas,
dateutil, pytz and the packages cannot be imported. The last line printed is
a JSON summary: ``{"recordings": N, "calls": M, "problems": [...]}``; the
exit status is 1 when there is a problem.
"""

from __future__ import annotations

import argparse
import importlib.abc
import json
import sys
from pathlib import Path

#: What may not be imported here, by top-level name.
BLOCKED = frozenset(
    {
        'pandas',
        'dateutil',
        'pytz',
        'sdm_client',
        'sdm_relational_engine',
        'predictive_query',
        'sdm_connectors',
        'sdm_api',
    }
)


class _Blocked(importlib.abc.MetaPathFinder):
    def find_spec(self, name: str, path: object, target: object = None) -> None:
        if name.partition('.')[0] in BLOCKED:
            raise ImportError(f'{name} cannot be imported here')
        return None


def main() -> int:
    sys.meta_path.insert(0, _Blocked())
    present = sorted(n for n in BLOCKED if n in sys.modules)
    assert not present, f'already imported: {present}'
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _request

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=_request.RECORDINGS)
    parser.add_argument('--only', nargs='*', default=None)
    args = parser.parse_args()

    from sdm_core import _native

    root = args.root.resolve()
    recordings = 0
    calls = 0
    problems: list[str] = []
    for directory in _request.recording_dirs(root):
        name = _request.load_json(directory / _request.CALLS_FILE)['scenario']
        if args.only and name not in args.only:
            continue
        recordings += 1
        try:
            count, found = _request.redrive(directory, _native, root)
        except Exception as error:  # noqa: BLE001 - reported
            count, found = 0, [f'{name}: {type(error).__name__}: {error}']
        calls += count
        problems += found
    leaked = sorted(n for n in BLOCKED if n in sys.modules)
    if leaked:
        problems.append(f'imported though blocked: {leaked}')
    print(
        json.dumps(
            {'recordings': recordings, 'calls': calls, 'problems': problems}
        )
    )
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
