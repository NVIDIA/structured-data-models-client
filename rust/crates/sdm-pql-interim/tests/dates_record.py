# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Records ``dates.json`` beside this file: the reference's reading of
generated date literals, for ``dates.rs``.

Each literal is read as ``Constant.typed_value`` reads a PQL date literal,
``pd.to_datetime(literal)`` with pandas' defaults
(``docs/design/rust-core/phase-0/date-rules.md``, "The PQL literal rules"):
the date and time it gives, the UTC offset in seconds when it gives one,
and the format of D4's day-first warning when it gives one.

- ``packed``: rule D8's literals. A deterministic generator (``SEED``) writes
  ``CANDIDATES`` literals with a run of five digits or more: a packed first
  field of six, eight, twelve or fourteen digits followed by fields of one to
  eight digits, or a long field among short ones, with and without a time of
  day. Every one the reference reads is kept, and every ``REFUSED_EVERY``-th
  one it refuses.
- ``plain``: D2 to D7 and D4 over a grid of day, month and year fields of one
  to four digits, in each order, with each separator and a time of day of
  fields of one to four digits. Every day-first reading and every year-0
  literal is kept, and every ``PLAIN_EVERY``-th other literal.
- ``format_guess``: day-first literals whose middle field, written with four
  digits, is how ``strftime`` writes the year (``13/0001/1``): pandas then
  guesses the format ``%d/%Y/%m`` and warns with it, where D4 names no
  format (register PQL-4). Every one is kept.
- ``local_year``: two-digit years read at 2026-12-31T23:30 in New York
  (2027-01-01T04:30Z), where the local year, 2026, is not UTC's: the
  reference's ``dateutil`` places a two-digit year by the machine's local
  year (``parserinfo._year``), so it is set to the year ``zoneinfo`` gives
  for that instant in that zone.

The two-digit years of ``packed`` and ``plain`` follow the year of the
clock the script runs at (``year``). The platform pads ``%Y`` (``pads_year``,
macOS; ``date-rules.md``, "What depends on the host"); the Linux side of D4
is the first attempt's ``padded_years.tsv``, which ``dates.rs`` reads from
the archive's copy here (``padded-years.tsv``).

Run it with the reference's interpreter, isolated, from the repository root:

    ~/code/sdmc-reference/.venv/bin/python -I \\
        rust/crates/sdm-pql-interim/tests/dates_record.py ~/code/sdmc-reference
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import random
import re
import subprocess
import sys
import time
import warnings
import zoneinfo
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OUT = HERE / 'dates.json'
SEED = 20261012
CANDIDATES = 150_000
REFUSED_EVERY = 40
PLAIN_EVERY = 12
FORMAT = re.compile(r'^Parsing dates in (.*?) format when dayfirst=False')


def _guard(reference: Path) -> Any:
    path = reference / 'scripts/reference_guard.py'
    spec = importlib.util.spec_from_file_location('reference_guard', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(literal: str) -> list[Any]:
    """``[literal, value, offset, format]``, value None where refused."""
    import pandas as pd

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            stamp = pd.to_datetime(literal)
        except Exception:  # noqa: BLE001 - every failure is a refusal
            stamp = None
    formats = {
        m.group(1)
        for w in caught
        if issubclass(w.category, UserWarning)
        and (m := FORMAT.match(str(w.message)))
    }
    assert len(formats) <= 1, literal
    said = next(iter(formats), None)
    if stamp is None:
        return [literal, None, None, said]
    assert stamp.microsecond == 0 and stamp.nanosecond == 0, literal
    offset = stamp.utcoffset()
    value = (
        f'{stamp.year:04d}-{stamp.month:02d}-{stamp.day:02d}T'
        f'{stamp.hour:02d}:{stamp.minute:02d}:{stamp.second:02d}'
    )
    return [
        literal,
        value,
        None if offset is None else int(offset.total_seconds()),
        said,
    ]


def _kept(literal: str, every: int) -> bool:
    digest = hashlib.sha256(literal.encode()).digest()
    return int.from_bytes(digest[:4], 'big') % every == 0


def packed_candidates() -> list[str]:
    rng = random.Random(SEED)

    def small(n: int, hi: int) -> str:
        return str(rng.randint(0, hi)).zfill(n)[-n:]

    def digits(n: int) -> str:
        return ''.join(rng.choice('0123456789') for _ in range(n))

    def packed(n: int) -> str:
        y = rng.choice(
            [rng.randint(1, 9999), 0, 9999, 2024, 1, rng.randint(0, 99)]
        )
        m = rng.choice([rng.randint(1, 12), 0, 13, 2])
        d = rng.choice([rng.randint(1, 31), 0, 29, 30, 31, 32])
        h = rng.choice([rng.randint(0, 23), 24])
        mi = rng.choice([rng.randint(0, 59), 60])
        s = rng.choice([rng.randint(0, 59), 60])
        if n == 6:
            return rng.choice(
                [
                    f'{y % 100:02d}{m:02d}{d:02d}',
                    f'{d:02d}{m:02d}{y % 100:02d}',
                    f'{m:02d}{d:02d}{y % 100:02d}',
                    f'{h:02d}{mi:02d}{s:02d}',
                ]
            )
        text = f'{y:04d}{m:02d}{d:02d}'
        if n >= 12:
            text += f'{h:02d}{mi:02d}'
        if n == 14:
            text += f'{s:02d}'
        return text

    def after(n: int) -> str:
        r = rng.random()
        if n == 6 and r < 0.6:
            return packed(6)
        if r < 0.7:
            return small(n, rng.choice([9, 23, 30, 59, 99, 9999]))
        return digits(n)

    def long_value(n: int) -> str:
        return rng.choice(
            [
                small(n, 31),
                small(n, 12),
                small(n, 9999),
                small(n, 99999),
                digits(n),
            ]
        )

    found = set()
    for _ in range(CANDIDATES):
        sep = rng.choice('/-.')
        if rng.random() < 0.6:
            first = packed(rng.choice([6, 8, 12, 14]))
            lengths = [1, 2, 2, 3, 4, 4, 5, 6, 6, 8]
            rest = [after(rng.choice(lengths)) for _ in range(2)]
            parts = [first, *rest]
        else:
            lengths = [rng.choice([1, 2, 3, 4]) for _ in range(3)]
            lengths[rng.randrange(3)] = rng.choice(
                [5, 6, 7, 8, 9, 10, 12, 14, 16, 18]
            )
            parts = [
                long_value(n)
                if n >= 5
                else small(n, rng.choice([12, 31, 99, 9999]))
                for n in lengths
            ]
        literal = sep.join(parts)
        if rng.random() < 0.35:
            h = small(
                rng.choice([1, 2, 2, 3, 4, 5, 6, 7, 8]),
                rng.choice([23, 24, 99]),
            )
            m = small(rng.choice([1, 2, 2, 3, 5, 6]), rng.choice([59, 60, 99]))
            s = small(rng.choice([1, 2, 2, 3, 5, 6]), rng.choice([59, 60]))
            literal += f' {h}:{m}:{s}'
        found.add(literal)
    return sorted(found)


def plain_candidates() -> list[str]:
    days = ['13', '31', '29', '013', '0013', '1', '01', '30', '32']
    months = ['1', '01', '12', '2', '02', '001', '13']
    years = [
        '2024',
        '0999',
        '999',
        '24',
        '0024',
        '1',
        '0001',
        '9999',
        '0000',
        '00',
        '0100',
        '100',
        '2000',
        '75',
        '76',
    ]
    times = [
        '',
        ' 1:2:3',
        ' 01:02:03',
        ' 001:02:03',
        ' 23:59:59',
        ' 24:00:00',
        ' 0001:00:00',
        ' 12:60:00',
        ' 1:002:03',
        ' 01:02:0003',
    ]
    found = set()
    for sep in '/-.':
        for d in days:
            for m in months:
                for y in years:
                    for t in times:
                        found.add(f'{d}{sep}{m}{sep}{y}{t}')
                        found.add(f'{m}{sep}{d}{sep}{y}{t}')
                        found.add(f'{y}{sep}{m}{sep}{d}{t}')
    return sorted(found)


def main(reference: Path) -> int:
    guard = _guard(reference)
    guard.require_isolation()
    from dateutil.parser import DEFAULTPARSER

    year = time.localtime().tm_year
    assert DEFAULTPARSER.info._year == year
    packed = []
    for literal in packed_candidates():
        row = read(literal)
        if row[1] is not None or _kept(literal, REFUSED_EVERY):
            packed.append(row)
    plain = []
    for literal in plain_candidates():
        row = read(literal)
        if (
            row[3] is not None
            or literal.startswith('0000')
            or _kept(literal, PLAIN_EVERY)
        ):
            plain.append(row)

    guessed = []
    for sep in '/-.':
        for day in ('13', '31', '20'):
            for middle in ('0001', '0002', '0009', '0010', '0012', '0013'):
                for last in ('1', '2', '9', '10', '12', '01', '0001'):
                    for t in ('', ' 1:2:3', ' 10:00:00'):
                        guessed.append(
                            read(f'{day}{sep}{middle}{sep}{last}{t}')
                        )

    instant = datetime.datetime(2027, 1, 1, 4, 30, tzinfo=datetime.UTC)
    zone = 'America/New_York'
    local = instant.astimezone(zoneinfo.ZoneInfo(zone)).year
    two_digit = [
        '1/2/76',
        '1/2/77',
        '1/2/75',
        '13/1/26',
        '1/2/26',
        '1/2/27',
        '760101-01-01',
        '1.2.3',
    ]
    saved = (DEFAULTPARSER.info._year, DEFAULTPARSER.info._century)
    DEFAULTPARSER.info._year, DEFAULTPARSER.info._century = (
        local,
        local // 100 * 100,
    )
    try:
        at_local = [read(literal) for literal in two_digit]
    finally:
        DEFAULTPARSER.info._year, DEFAULTPARSER.info._century = saved
    guard.require_isolation()

    commit = subprocess.run(
        ['git', '-C', str(reference), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    import pandas as pd

    document = {
        'format': 'sdm-pql-dates/1',
        'source': (
            f'rust-core-reference at {commit}, pandas {pd.__version__}: '
            'pd.to_datetime of each literal, recorded by dates_record.py '
            'beside this file'
        ),
        'year': year,
        'pads_year': datetime.date(1, 1, 1).strftime('%Y') == '0001',
        'packed': {'seed': SEED, 'candidates': CANDIDATES, 'rows': packed},
        'plain': plain,
        'format_guess': guessed,
        'local_year': {
            'now': instant.isoformat().replace('+00:00', 'Z'),
            'zone': zone,
            'year': local,
            'rows': at_local,
        },
    }
    text = json.dumps(document, ensure_ascii=False, separators=(',', ':'))
    # One row a line, for review.
    text = text.replace('],["', '],\n["')
    OUT.write_text(text + '\n', encoding='utf-8')
    print(
        f'packed {len(packed)} ({sum(r[1] is not None for r in packed)} read), '
        f'plain {len(plain)} ({sum(r[3] is not None for r in plain)} warned), '
        f'local year {local}'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main(Path(sys.argv[1])))
