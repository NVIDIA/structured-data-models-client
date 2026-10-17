# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1a's acceptance tests for the reference recorded on SQLite and
Postgres (the independent test author's).

The design (choices 2 and 3): the recorder records the reference for every
case, S0 and S1 (the owner's call: every case on all three databases), on
each database, one reference line per case and database, keyed
``<case>@sqlite`` and ``<case>@postgres`` (DuckDB's lines keep the bare
ID); the reference capture extends to the reference's SQLite and Postgres
drivers, so its statements and results there are recorded like DuckDB's.
A case's ``databases`` (with its owner-approved exclusion) narrows the set.

Recording needs the runner on those databases (T2.1a's contract loader and
``run_python.py --database``) and, for Postgres, a server: the recorder
reads ``SDM_POSTGRES_TEST_DSN`` (the pinned ``POSTGRES_IMAGE`` of
``.github/workflows/ci.yml`` locally in a container). Once recorded, these
tests read only the committed lines and need nothing running.

The runner test (:func:`test_the_old_client_on_sqlite_meets_the_reference_line`)
runs ``run_python.py --mode python-old --database sqlite`` on one case and
holds its outcome to the committed ``@sqlite`` line: the reference side
that a SQLite job runs when the core reports no SQLite provider.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATABASES = ('duckdb', 'sqlite', 'postgres')
#: A plain S1 case every database loads (int keys, a timestamp, a float).
SMOKE = 'base.regression'


def _reference() -> dict[str, dict[str, Any]]:
    [path] = sorted((ROOT / 'contract/reference').glob('*.jsonl'))
    return {
        line['case']: line
        for line in map(json.loads, path.read_text().splitlines())
    }


def _cases() -> list[dict[str, Any]]:
    r"""Every case, S0 and S1."""
    return [
        json.loads(path.read_text())
        for path in sorted((ROOT / 'contract/cases').rglob('*.json'))
    ]


def _key(case: str, database: str) -> str:
    return case if database == 'duckdb' else f'{case}@{database}'


def _wanted() -> list[str]:
    return [
        _key(case['id'], database)
        for case in _cases()
        for database in case.get('databases', DATABASES)
    ]


def _statements(line: dict[str, Any]) -> list[Any]:
    capture = (line.get('driver') or {}).get('capture') or {}
    found: list[Any] = capture.get('statements') or []
    return found


def test_every_case_has_its_duckdb_line() -> None:
    r"""Control: the 86 cases (24 S0, 62 S1) each have their bare DuckDB
    line, the capture of most holding statements."""
    reference = _reference()
    cases = _cases()
    assert len(cases) >= 86
    assert [c['id'] for c in cases if c['id'] not in reference] == []
    assert sum(bool(_statements(reference[c['id']])) for c in cases) > 40


def test_every_case_has_a_line_on_each_database() -> None:
    r"""Exit: the reference recorded for every case on all three
    databases, one line per case and database."""
    reference = _reference()
    missing = [key for key in _wanted() if key not in reference]
    assert missing == []


@pytest.mark.parametrize('database', ['sqlite', 'postgres'])
def test_the_capture_records_the_reference_s_statements_there(
    database: str,
) -> None:
    r"""Exit (choice 3): where the DuckDB line holds captured statements,
    the database's line does too, made by the reference's own driver, with
    the reference's identity."""
    reference = _reference()
    short = []
    for case in _cases():
        if database not in case.get('databases', DATABASES):
            continue
        bare, there = (
            reference[case['id']],
            reference[_key(case['id'], database)],
        )
        if _statements(bare) and not _statements(there):
            short.append(case['id'])
        assert there['identity']['runner'] == 'python-ref'
        assert there['identity']['reference'] == bare['identity']['reference']
    assert short == []


def test_the_old_client_on_sqlite_meets_the_reference_line(
    tmp_path: Path,
) -> None:
    r"""Exit: the reference side a SQLite job runs: the old client over the
    loaded SQLite database, its observation keyed ``<case>@sqlite`` and its
    outcome the committed line's."""
    out = tmp_path / 'python-old.jsonl'
    done = subprocess.run(
        [
            sys.executable,
            '-I',
            str(ROOT / 'contract/runners/run_python.py'),
            '--mode',
            'python-old',
            '--database',
            'sqlite',
            '--out',
            str(out),
            str(ROOT / f'contract/cases/s1/{SMOKE}.json'),
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr
    [line] = map(json.loads, out.read_text().splitlines())
    key = _key(SMOKE, 'sqlite')
    assert line['case'] == key
    assert line['outcome'] == _reference()[key]['outcome']
    assert _statements(line), 'the old client read SQLite through the capture'
