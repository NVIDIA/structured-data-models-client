# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Record ``padded_years.tsv``: the date literals pandas reads differently
on Linux than on macOS, for ``padded_years.rs``.

pandas guesses a literal's format by writing its reading back with Python's
``strftime``. On macOS, and on Linux from Python 3.14, ``%Y`` is written with
four digits for a year below 1000; on Linux up to Python 3.13 it is not. So
where the two differ, one platform guesses a day-first format and warns, and
the other reads the same date without the warning.

This reads a set of literals with ``pd.to_datetime`` here, which must be a
platform that pads, and in ``python:3.12-slim`` under Docker with the same
pandas, numpy and dateutil, and writes every literal whose readings differ.
The literals are the date examples of ``DATES.md`` and of the quirk table,
the 100,000 literals ``P5.dates`` generates, and every year below 1000
written with one to five digits in each field order. ``--check`` writes
nothing and fails if the file would change.

Run from the repository root with pandas installed:
``python rust/crates/pql/tests/padded_years.py [--check]``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
TSV = HERE / 'padded_years.tsv'
IMAGE = 'python:3.12-slim'
PQL_TESTS = 'tests/rust_core/pql'
# Reads the JSON list of literals on stdin and writes their readings, as
# `_pql.python_date` gives them, on stdout.
READER = (
    'import json, sys\n'
    f'sys.path.insert(0, {PQL_TESTS!r})\n'
    'from _pql import python_date\n'
    'literals = json.load(sys.stdin)\n'
    'json.dump({x: python_date(x) for x in literals}, sys.stdout)\n'
)
HEADER = (
    '# Date literals pandas 3 reads differently on Linux with Python 3.12 or\n'
    '# 3.13 than on macOS. Both read `value`; only where `strftime` pads %Y\n'
    '# (macOS, and Linux from Python 3.14) does pandas give `warning`.\n'
    '# Written by padded_years.py; checked by padded_years.rs.\n'
    'literal\tvalue\twarning\n'
)


def literals() -> list[str]:
    sys.path.insert(0, str(REPO / PQL_TESTS))
    from _generate import date_literals

    found = set(date_literals(20261003, 100_000))
    dates_md = (REPO / 'rust/crates/pql/DATES.md').read_text(encoding='utf-8')
    found |= set(re.findall(r'^\| D\d+ \| `([^`]+)`', dates_md, re.M))
    quirks = json.loads((HERE / 'accept_quirks.json').read_text('utf-8'))
    found |= {row['literal'] for row in quirks['dates']}
    for year in (0, 1, 9, 10, 50, 99, 100, 101, 999, 1000, 1999):
        for width in range(len(str(year)), 6):
            y = str(year).zfill(width)
            for m, d in ((1, 2), (2, 1), (12, 31), (2, 29), (13, 1), (31, 12)):
                for ms, ds in ((str(m), str(d)), (f'{m:02d}', f'{d:02d}')):
                    for sep in '/-.':
                        for time in ('', ' 12:30:00'):
                            found.add(f'{y}{sep}{ms}{sep}{ds}{time}')
                            found.add(f'{ms}{sep}{ds}{sep}{y}{time}')
                            found.add(f'{ds}{sep}{ms}{sep}{y}{time}')
    return sorted(found)


def read_here(texts: list[str]) -> dict[str, dict]:
    assert datetime.datetime(1, 1, 1).strftime('%Y') == '0001', (
        'run this where strftime pads %Y, such as macOS'
    )
    run = subprocess.run(
        [sys.executable, '-c', READER],
        input=json.dumps(texts),
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
    )
    return json.loads(run.stdout)


def read_on_linux(texts: list[str]) -> dict[str, dict]:
    import dateutil
    import numpy
    import pandas

    pins = (
        f'pandas=={pandas.__version__} numpy=={numpy.__version__} '
        f'python-dateutil=={dateutil.__version__}'
    )
    script = (
        f'pip install -q --root-user-action=ignore {pins} >/dev/null && '
        f'python -c "$READER"'
    )
    run = subprocess.run(
        [
            'docker',
            'run',
            '--rm',
            '-i',
            '-e',
            'READER',
            '-v',
            f'{REPO}:/repo:ro',
            '-w',
            '/repo',
            IMAGE,
            'sh',
            '-c',
            script,
        ],
        input=json.dumps(texts),
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, 'READER': READER},
    )
    return json.loads(run.stdout)


def table() -> str:
    texts = literals()
    here, linux = read_here(texts), read_on_linux(texts)
    rows = []
    for text in texts:
        padded, unpadded = here[text], linux[text]
        if padded == unpadded:
            continue
        # Only the warning is known to differ; anything else is news.
        assert padded['value'] == unpadded['value'], (text, padded, unpadded)
        assert len(padded['warnings']) == 1, (text, padded)
        assert unpadded['warnings'] == [], (text, unpadded)
        warning = padded['warnings'][0]['message']
        rows.append(f'{text}\t{padded["value"]}\t{warning}\n')
    print(f'{len(texts)} literals, {len(rows)} read differently on Linux')
    return HEADER + ''.join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    text = table()
    if args.check:
        if TSV.read_text(encoding='utf-8') != text:
            print(f'{TSV} is out of date', file=sys.stderr)
            return 1
        return 0
    TSV.write_text(text, encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
