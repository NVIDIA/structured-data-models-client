# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Writes ``contract/view-captures.json``: what the old client read and
answered converting each semantic model of the recordings package (T3.20,
``contract/captures/s2`` and ``s2b``), for the core's conversion tests
(T3.5): each ``view.*`` case's input, every statement it sent while
building the schema (stage ``from_source``), in order, with its parameters
and its rows (an Arrow IPC stream, in standard base64, each distinct stream
once) or its error, and the outcome and schema it answered.

The captures are read from a git ref, by default the recordings package's
branch, so nothing of them is copied into this branch but what the tests
read. Run from the repository root with any Python 3.12::

    python3 contract/fixtures/views/captures.py [REF]
"""

import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REF = 'origin/rust-core-phase-3-topic-T3.20b'
OUT = Path('contract/view-captures.json')
FORMAT = 'sdm-view-captures/1'
LEGS = ('s2/databricks-sql', 's2b/snowflake')


def git(*args: str) -> bytes:
    return subprocess.run(
        ['git', *args], capture_output=True, check=True
    ).stdout


def show(ref: str, path: str) -> bytes:
    return git('show', f'{ref}:{path}')


def captures(ref: str) -> list[str]:
    names = git(
        'ls-tree', '-r', '--name-only', ref, 'contract/captures'
    ).decode()
    return sorted(
        name.rsplit('/', 1)[0]
        for name in names.split()
        if name.endswith('/capture.json')
        and '/view.' in name
        and any(f'/captures/{leg}/' in name for leg in LEGS)
    )


def case(ref: str, directory: str, tables: dict[str, str]) -> dict[str, Any]:
    capture = json.loads(show(ref, f'{directory}/capture.json'))
    reference = capture['reference']
    driver = reference['driver']['capture']['statements']
    statements = []
    for at, statement in enumerate(capture['statements']):
        if driver[at].get('stage') != 'from_source':
            break
        result = statement.get('result')
        if result is not None:
            digest = result.rsplit('/', 1)[1].removesuffix('.arrows')
            if digest not in tables:
                tables[digest] = base64.b64encode(
                    show(ref, f'{directory}/{result}')
                ).decode()
            result = digest
        statements.append(
            {
                'text': statement['text'],
                'parameters': statement.get('parameters') or [],
                'result': result,
                'error': statement.get('error'),
            }
        )
    # Every case file is under ``s2``; ``s2b`` names the second recording.
    name = capture['case']
    inputs = json.loads(show(ref, f'contract/cases/s2/{name}.json'))
    return {
        'case': name,
        'leg': capture['leg'],
        'input': inputs['input'],
        'session': capture['session'],
        'statements': statements,
        'outcome': reference['outcome'],
        'warnings': reference['host'].get('warnings', []),
        'schema': (reference.get('public') or {}).get('schema'),
    }


def main() -> None:
    ref = sys.argv[1] if len(sys.argv) > 1 else REF
    commit = git('rev-parse', ref).decode().strip()
    tables: dict[str, str] = {}
    cases = [case(ref, directory, tables) for directory in captures(ref)]
    OUT.write_text(
        json.dumps(
            {
                'format': FORMAT,
                'recordings': commit,
                'cases': cases,
                'tables': dict(sorted(tables.items())),
            },
            indent=1,
        )
        + '\n'
    )


if __name__ == '__main__':
    main()
