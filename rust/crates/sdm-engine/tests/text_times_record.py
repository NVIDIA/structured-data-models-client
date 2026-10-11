# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Records ``text-times.json``, pandas' readings of columns of text as times
(T1.25), which ``text_times.rs`` holds ``sdm_engine::time::text`` to.

Run once, by hand, under the reference's pandas (3.0.6); CI never runs it,
and the vectors are not recorded again to pass a dependency bump (F4):

    git show origin/archive/rust-core-attempt-1:tests/parity/fixtures/dates.json \
        > CORPUS.json
    python rust/crates/sdm-engine/tests/text_times_record.py OUT.json CORPUS.json

Each column is read three ways, as the reference's engine reads text:
``coerce``, ``pd.to_datetime(values, errors='coerce')`` (its inference over
a preview); ``utc``, the same with ``utc=True`` (the time column's choice);
``each``, with ``unit='ns'`` (a SQL source's values, one by one, register
DT-7). A reading is each value's instant in microseconds since the epoch,
in UTC, or null; ``"MIXED"`` where pandas refuses offsets mixed; ``"ERR"``
for any other exception; ``"CLOCK"`` for a column holding ``now`` or
``today``, which pandas reads as the clock's time. ``day_first`` is the
format pandas warns it guessed with the day before the month (``Parsing
dates in ... format when dayfirst=False``) as it reads ``coerce``, or null.
``year`` is the clock's year, which places a two-digit year.

The columns: every value of a grid over ``date-rules.md``'s shapes (each
order, separator, year length, time and zone), alone and in random pairs
and fours; random columns of one shape with junk, codes and other shapes
mixed in; the T0.6 corpus's 121 column cases; and the codes and short
categories AdventureWorks holds. The grid and the random columns are drawn from a seeded generator,
so a second run writes the same file.
"""

from __future__ import annotations

import datetime
import json
import random
import re
import sys
import warnings

import pandas as pd

YEAR_FIRST = [
    '2024{a}01{b}05',
    '2023{a}12{b}31',
    '1999{a}02{b}29',
    '2024{a}02{b}29',
]
SLASH = [
    '01{a}05{b}2024',
    '15{a}03{b}2024',
    '1{a}5{b}2024',
    '3{a}15{b}2024',
    '01{a}05{b}24',
    '15{a}3{b}24',
    '1{a}5{b}124',
    '13{a}13{b}2020',
    '32{a}12{b}24',
]
SEPARATORS = [('-', '-'), ('/', '/'), ('-', '/')]
TIMES = [
    '',
    ' 10:30',
    'T10:30',
    ' 9:30',
    ' 10:30:15',
    'T10:30:15',
    ' 10:30:15.5',
    ' 10:30:15.123456',
    'T10:30:15.25',
    ' 23:59:60',
    ' 24:00',
]
ZONES = [
    '',
    'Z',
    ' Z',
    '+01:00',
    '+0100',
    ' +01:00',
    '-08:00',
    '+01',
    '+05:30',
    ' -0800',
    ' UTC',
    # Offsets at and past a day, and minutes past 59 below it (T1.25 fix 1).
    '+00',
    '-00',
    '+14',
    '+23:59',
    '-23:59',
    '+2359',
    '+24',
    '+24:00',
    '+2360',
    '-25:00',
    '+99:00',
    '-99',
    '+01:99',
    '+0160',
    ' +24:00',
    'Z+01:00',
    '+01:00:30',
]
JUNK = [
    'not a date',
    'AW00000001',
    '10-4020-000676',
    'SO43659',
    'PO522145787',
    '105041Vi84182',
    '4911-403C-98',
    'BK-R93R-62',
    '99/99/2020',
    '2021-13-45',
    'nan',
    'NaT',
    '',
    'March 3 2020',
    '20240101',
    '2024',
    '12',
    'abc',
    '2024-01-05 ',
    ' 2024-01-05',
    '2024-1-5',
    'x1',
    'Jan 5',
    '1-2-3',
    '12-345-678',
    '10.4020.000676',
    '123-4567-89',
    '1234-567-890',
    '1.2.3456',
    '2024.01.05',
    '5th',
    'T',
    'Z',
    'monday',
    '0000-01-05',
    '9999-12-31 23:59:60',
    '0001-01-01 00:00:00+01:00',
    'now',
    'today',
    '2024-01-05 10:00:00.1234567',
    'Café 2024-01-05',
    None,
]


def grid() -> list[str]:
    values = []
    for date in YEAR_FIRST + SLASH:
        for a, b in SEPARATORS:
            for time in TIMES:
                for zone in ZONES:
                    if zone and not time:
                        continue
                    values.append(date.format(a=a, b=b) + time + zone)
    return values


def random_columns(rng: random.Random, n: int) -> list[list[str | None]]:
    def date() -> str:
        y = rng.choice(
            [1999, 2020, 2024, 1850, 2100, 1, 9999, 0, rng.randint(1900, 2100)]
        )
        m = rng.choice([1, 2, 3, 11, 12, 13, rng.randint(1, 12)])
        d = rng.choice(
            [1, 2, 12, 13, 15, 28, 29, 30, 31, 0, 32, rng.randint(1, 28)]
        )
        return y, m, d

    def time() -> str:
        if rng.random() < 0.4:
            return ''
        t = f'{rng.choice([" ", "T"])}{rng.choice([0, 9, 10, 23, 24]):02d}:{rng.randint(0, 59):02d}'
        if rng.random() < 0.6:
            t += f':{rng.choice([0, 59, 60, rng.randint(0, 59)]):02d}'
            if rng.random() < 0.4:
                t += '.' + ''.join(
                    rng.choice('0123456789')
                    for _ in range(rng.choice([1, 3, 6, 9]))
                )
        if rng.random() < 0.35:
            t += rng.choice(['', ' ']) + rng.choice(
                ['Z', '+01:00', '+0100', '-08:00', '+01', 'UTC', 'CST']
            )
        return t

    columns = []
    for _ in range(n):
        lead = rng.choice(['year', 'year', 'slash'])
        sep = rng.choice('-/')
        ylen = rng.choice([4, 4, 2, 3])
        tail = time()
        column = []
        for _ in range(rng.randint(1, 10)):
            if rng.random() < 0.25:
                column.append(rng.choice(JUNK))
                continue
            y, m, d = date()
            if lead == 'year':
                v = f'{y:04d}{sep}{m:02d}{sep}{d:02d}'
            else:
                a, b = (d, m) if rng.random() < 0.5 else (m, d)
                v = f'{a:02d}{sep}{b:02d}{sep}{str(y % 10**ylen).zfill(ylen)}'
            column.append(v + (tail if rng.random() < 0.8 else time()))
        columns.append(column)
    return columns


def corpus() -> list[list[str | None]]:
    with open(sys.argv[2]) as corpus_file:
        cases = json.load(corpus_file)['cases']
    return [
        case['input']['values']
        for case in cases
        if case['operation'] in ('time_column.read', 'dates.read')
    ]


def read(values: list[str | None], **options: object) -> object:
    if any(value in ('now', 'today') for value in values):
        # The clock's time, which a second run would not repeat.
        return 'CLOCK'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            parsed = pd.to_datetime(
                pd.Series(values, dtype=object), errors='coerce', **options
            )
        except ValueError as error:
            return 'MIXED' if 'Mixed timezones' in str(error) else 'ERR'
        except Exception:  # noqa: BLE001
            return 'ERR'
    out = []
    for stamp in parsed:
        if pd.isna(stamp):
            out.append(None)
            continue
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert('UTC').tz_localize(None)
        out.append(int(stamp.as_unit('us')._value))
    return out


DAY_FIRST = re.compile(r'Parsing dates in (.*) format when dayfirst=False')


def day_first(values: list[str | None]) -> str | None:
    """The format pandas warns it guessed with the day first, reading
    ``values`` as ``coerce`` does, or None."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            pd.to_datetime(pd.Series(values, dtype=object), errors='coerce')
        except Exception:  # noqa: BLE001
            pass
    said = [DAY_FIRST.match(str(w.message)) for w in caught]
    formats = [m.group(1) for m in said if m]
    return formats[0] if formats else None


def main() -> None:
    rng = random.Random(25)
    values = grid()
    columns = [[v] for v in values]
    columns += [rng.sample(values, 2) for _ in range(1000)]
    columns += [rng.sample(values, 4) for _ in range(300)]
    columns += random_columns(rng, 1000)
    columns += corpus()
    columns += [[code] for code in JUNK]
    # Short words and letters, as AdventureWorks' categories hold them.
    columns += [
        ['M', 'S', 'L', 'XL'],
        ['S ', 'M ', 'R ', 'T '],
        ['H ', 'L ', 'M '],
        ['M', 'S', 'M', 'S'],
        ['Red', 'Black', 'Silver'],
        ['CM ', 'LB ', 'G  '],
        ['am', 'pm', 'XL'],
        ['T', 'Z', 'Q'],
        ['Mon', 'Jan', 'XX'],
        # Offsets of a day or more first, later, and throughout.
        ['2024-01-01T01:02:03+24:00', '2024-01-02T01:02:03+01:00'],
        ['2024-01-01T01:02:03+01:00', '2024-01-02T01:02:03+24:00'],
        ['2024-01-01T01:02:03+24:00', '2024-01-02T01:02:03+24:00'],
        ['2024-01-01T01:02:03+2360', '2024-01-02T01:02:03'],
        ['2024-01-01 01:02:03 -25:00', '2024-01-02 01:02:03 -23:59'],
        ['01/15/2024 01:02:03+24:00', '01/16/2024 01:02:03+01:00'],
        ['15/01/2024 01:02:03 +99:00', '16/01/2024 01:02:03 +01:99'],
        ['2024-01-01T01:02:03+01:99', '2024-01-02T01:02:03+0160'],
    ]
    # Two-digit years at the pivot, each side of both ends of the hundred
    # years around the clock's year (49 after it, 50 before it).
    year = datetime.date.today().year
    for yy in (year + 48, year + 49, year + 50, year - 50, year - 51):
        yy %= 100
        columns += [
            [f'01/05/{yy:02d}'],
            [f'15/3/{yy:02d} 10:30'],
            [f'1-5-{yy:02d}', f'2-6-{yy:02d}'],
        ]
    document = {
        'format': 'sdm-text-times/1',
        'pandas': pd.__version__,
        'year': datetime.date.today().year,
        'columns': [
            {
                'values': column,
                'coerce': read(column),
                'utc': read(column, utc=True),
                'each': read(column, unit='ns'),
                'day_first': day_first(column),
            }
            for column in columns
        ],
    }
    with open(sys.argv[1], 'w') as out:
        json.dump(document, out, separators=(',', ':'))
        out.write('\n')


if __name__ == '__main__':
    main()
