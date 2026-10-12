# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Records ``text-times.json``, the core's own vectors for text times
(T1.25b), which ``text_times.rs`` holds ``sdm_engine::time::text`` to.

The expected values are this file's reading of ``date-rules.md``'s "The
core's text formats", written from that spec with Python's ``re`` and
``datetime`` only, apart from the Rust reader, and not with pandas or
dateutil: every year-first value with dashes is cross-checked against
``datetime.fromisoformat`` to the microsecond. Run once, by hand; CI never
runs it, and the vectors are not recorded again to pass a dependency bump
(F4):

    python rust/crates/sdm-engine/tests/text_times_record.py OUT.json

Each column is read as the core reads text: ``column``, the values to UTC
instants in nanoseconds (as decimal text, past an int64), null for a
missing value, or ``"outside"`` (a value outside the
list, ``text_time``), ``"order"`` (an order of day and month the values
leave open, ``text_time_order``) or ``"mixed"`` (offsets mixed,
``text_time_zones``); ``utc``, the same read to UTC, where no offsets are
mixed; ``share``, the values read as times at fewest (the fewer of the two
orders' counts where the values decide none), those read in either order,
those and the values in the client's shapes read in neither, and the values
outside the shapes, or ``"mixed"``; ``day_first``, the format inference warns of, or null.
``year`` is the clock's year, which places a two-digit year.

The columns: a grid over the formats (each date, separator, year length,
time of day and zone) alone and decided either way round, the edges of
every field, values outside the list alone and among values in it, the
orders of day and month, two-digit years at the pivot, offsets mixed, and
random columns from a seeded generator, so a second run writes the same
file.
"""

from __future__ import annotations

import json
import random
import re
import sys
from datetime import date, datetime, timezone

YEAR = 2026

#: Unicode's White_Space, which a value's ends lose.
WHITE_SPACE = (
    '\t\n\x0b\x0c\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005'
    '\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000'
)

#: The client's two shapes, as date-rules.md's "Accepted shapes" writes
#: them, over ASCII digits.
ZONE = r'(Z|[+-]\d{2}(?::?\d{2})?|[A-Za-z][A-Za-z0-9_+/-]*)'
TIME = (
    r'(?:([ T])(\d{H}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?(?:( ?)' + ZONE + r')?)?'
)
YEAR_FIRST = re.compile(
    r'(\d{4})([-/])(\d{2})([-/])(\d{2})' + TIME.replace('{H}', '{2}'), re.ASCII
)
DAY_OR_MONTH = re.compile(
    r'(\d{1,2})([-/])(\d{1,2})([-/])(\d{2,4})' + TIME.replace('{H}', '{1,2}'),
    re.ASCII,
)

MISSING = ('missing',)
OUTSIDE = ('outside', False)
SHAPED = ('outside', True)


def shape(text: str) -> tuple[str, re.Match[str]] | None:
    for lead, pattern in (('year', YEAR_FIRST), ('dm', DAY_OR_MONTH)):
        match = pattern.fullmatch(text)
        if match:
            return lead, match
    return None


def is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def month_days(year: int, month: int) -> int:
    if month == 2:
        return 29 if is_leap(year) else 28
    return 30 if month in (4, 6, 9, 11) else 31


def epoch_days(year: int, month: int, day: int) -> int:
    """Days from 1970-01-01, by datetime's ordinals (the years 1 to 9999)."""
    return date(year, month, day).toordinal() - date(1970, 1, 1).toordinal()


LOW = epoch_days(1, 1, 1) * 86_400 * 10**9
HIGH = (
    (date(9999, 12, 31).toordinal() - date(1970, 1, 1).toordinal() + 1)
    * 86_400
    * 10**9
)


def pivot(yy: int) -> int:
    """A two-digit year in the hundred years around YEAR: 50 before it to
    49 after."""
    for year in range(YEAR - 50, YEAR + 50):
        if year % 100 == yy:
            return year
    raise AssertionError(yy)


def one(value: str | None, order: str) -> tuple:
    """``value`` read by the list, a day or month first value in
    ``order`` (``'mf'`` or ``'df'``)."""
    if value is None:
        return MISSING
    text = value.strip(WHITE_SPACE)
    if not text:
        return MISSING
    found = shape(text)
    if found is None:
        return OUTSIDE
    lead, m = found
    a, s1, b, s2, c, tsep, hh, mi, ss, frac, spaced, zone = m.groups()
    if s1 != s2:
        return SHAPED
    if ss is not None and int(ss) > 59:
        return SHAPED
    if frac is not None and len(frac) > 9:
        return SHAPED
    offset = None
    if zone is not None:
        if zone in ('Z', 'UTC', 'GMT'):
            offset = 0
        elif zone[0] in '+-':
            digits = zone[1:].replace(':', '')
            hours, minutes = int(digits[:2]), int(digits[2:] or 0)
            if hours > 23 or minutes > 59:
                return SHAPED
            offset = (1 if zone[0] == '+' else -1) * (
                hours * 3600 + minutes * 60
            )
        else:
            return SHAPED
    if lead == 'year':
        year, month, day = int(a), int(b), int(c)
    else:
        month, day = (int(a), int(b)) if order == 'mf' else (int(b), int(a))
        year = pivot(int(c)) if len(c) == 2 else int(c)
    hour, minute = int(hh or 0), int(mi or 0)
    # Fields naming no date or time of day: outside the list.
    if not (
        1 <= month <= 12 and 1 <= day <= 31 and hour <= 23 and minute <= 59
    ):
        return SHAPED
    if day > month_days(year, month):
        return SHAPED
    # The year written is 1 to 9999, before any offset moves the instant.
    if not 1 <= year <= 9999:
        return SHAPED
    second = int(ss or 0)
    nanos = int((frac or '').ljust(9, '0') or 0)
    seconds = (
        epoch_days(year, month, day) * 86_400
        + hour * 3600
        + minute * 60
        + second
    )
    at = (seconds - (offset or 0)) * 10**9 + nanos
    if not LOW <= at < HIGH:
        return SHAPED
    if (
        lead == 'year'
        and s1 == '-'
        and spaced in (None, '')
        and 1 <= year <= 9999
    ):
        # The cross-check: datetime's own ISO reader, to the microsecond.
        try:
            iso = datetime.fromisoformat(text)
        except ValueError:
            iso = None
        if iso is not None and (zone is None or zone not in ('UTC', 'GMT')):
            if iso.tzinfo is None:
                iso = iso.replace(tzinfo=timezone.utc)
            try:
                micros = (iso - datetime(1970, 1, 1, tzinfo=timezone.utc)) // (
                    datetime.resolution
                )
            except OverflowError:
                micros = None
            assert micros is None or micros == at // 1000, (text, micros, at)
    return ('time', at, offset)


def decided(values: list[str | None]) -> str | None:
    """The client's day_comes_first over the day or month first values."""
    first = second = False
    for value in values:
        if value is None:
            continue
        found = shape(value.strip(WHITE_SPACE))
        if found and found[0] == 'dm':
            m = found[1]
            first |= int(m.group(1)) > 12
            second |= int(m.group(3)) > 12
    if first == second:
        return None
    return 'df' if first else 'mf'


def readings(values: list[str | None]) -> tuple[list[tuple], list[tuple]]:
    order = decided(values)
    if order:
        read = [one(v, order) for v in values]
        return read, read
    return [one(v, 'mf') for v in values], [one(v, 'df') for v in values]


def mixed(read: list[tuple]) -> bool:
    return len({r[2] for r in read if r[0] == 'time'}) > 1


def column(values: list[str | None], utc: bool) -> object:
    mf, df = readings(values)
    if any(
        a[0] == 'outside' and b[0] == 'outside'
        for a, b in zip(mf, df, strict=True)
    ):
        return 'outside'
    if mf != df:
        return 'order'
    if not utc and mixed(mf):
        return 'mixed'
    return [str(r[1]) if r[0] == 'time' else None for r in mf]


def share(values: list[str | None]) -> object:
    mf, df = readings(values)
    if mixed(mf) or mixed(df):
        return 'mixed'
    fewest = min(sum(r[0] == 'time' for r in read) for read in (mf, df))
    listed = sum(
        a[0] == 'time' or b[0] == 'time' for a, b in zip(mf, df, strict=True)
    )
    shaped = sum(
        a == SHAPED and b == SHAPED for a, b in zip(mf, df, strict=True)
    )
    return [fewest, listed, listed + shaped, sum(r == OUTSIDE for r in mf)]


def day_first(values: list[str | None]) -> str | None:
    if decided(values) != 'df':
        return None
    for value in values:
        if value is None:
            continue
        found = shape(value.strip(WHITE_SPACE))
        if not found or found[0] != 'dm':
            continue
        a, s1, b, s2, c, tsep, hh, mi, ss, frac, spaced, zone = found[
            1
        ].groups()
        text = f'%d{s1}%m{s2}' + ('%y' if len(c) == 2 else '%Y')
        if tsep is not None:
            text += f'{tsep}%H:%M'
            if ss is not None:
                text += ':%S'
            if frac is not None:
                text += '.%f'
            if zone is not None:
                text += spaced + (
                    '%z' if zone == 'Z' or zone[0] in '+-' else '%Z'
                )
        return text
    return None


YEAR_FIRST_DATES = [
    '2024{s}01{s}05',
    '2023{s}12{s}31',
    '2024{s}02{s}29',
    '1999{s}02{s}29',
]
DAY_OR_MONTH_DATES = [
    '01{s}05{s}2024',
    '1{s}5{s}2024',
    '15{s}03{s}2024',
    '3{s}15{s}2024',
    '01{s}05{s}24',
    '15{s}3{s}24',
    '1{s}5{s}124',
    '05{s}05{s}2024',
    '13{s}13{s}2020',
    '31{s}04{s}2024',
    '29{s}02{s}23',
]
TIMES = [
    '',
    ' 10:30',
    'T10:30',
    ' 9:30',
    ' 10:30:15',
    'T10:30:15.5',
    ' 10:30:15.123456789',
    ' 10:30:15.1234567891',
    ' 23:59:59',
    ' 23:59:60',
    ' 24:00',
    ' 10:60',
]
ZONES = [
    'Z',
    ' Z',
    '+01:00',
    ' +01:00',
    '+0100',
    '+01',
    '-08:00',
    ' -0800',
    '+05:30',
    '+00',
    '-00',
    '+23:59',
    '-23:59',
    '+24:00',
    '+24',
    '-2400',
    '+01:60',
    '+0199',
    ' UTC',
    'UTC',
    ' GMT',
    ' utc',
    ' EST',
    ' Europe/Berlin',
    '+01:00:30',
    'Z+01:00',
]
#: Values outside the list, and a few at its edges' far side.
OTHERS = [
    'not a date',
    'AW00011205',
    '10-4020-000099',
    'SO43862',
    '55631Vi35295',
    'A00D9DD1-D961-405A-AEBB-AC9EA5BE7C15',
    'March 3 2020',
    'Jan 5',
    '20240101',
    '2024-1-5',
    '2024/1/05',
    '15.01.2024',
    '2024.01.05',
    '2024-01-05T',
    '2024-01-05 10:30 ',
    '2024-01-05  10:30',
    '2024-01-05t10:30',
    '2024-01-05T10:30z',
    '0000-01-05',
    '0000-12-31 23:00-02:00',
    '0000-12-31T23:00-02:00',
    '9999-12-31T23:00-02:00',
    '0001-01-01T00:30+01:00',
    '0001-01-01T00:30-01:00',
    '9999-12-31T22:00-01:00',
    '9999-12-31 23:00-02:00',
    '0001-01-01 00:00+01:00',
    '2024/01-05',
    '1/5-2024',
    '123/5/2024',
    '1/5/20245',
    'now',
    'today',
    'NaT',
    'nan',
    'monday',
    'T',
    'Z',
    '12',
    '2024',
    '\u0662\u0660\u0662\u0664-\u0660\u0661-\u0660\u0665',
    'Caf\xe9 2024-01-05',
]
EDGES = [
    '',
    ' ',
    '\u20022024-01-05\u3000',
    '\t2024-01-05\n',
    '\xa02024-01-05\u3000',
    '0001-01-01',
    '0001-01-01 00:00',
    '9999-12-31 23:59:59.999999999',
    '1/1/0001',
    '2024-00-10',
    '2024-01-00',
    '2024-01-32',
    '2023-02-29',
    '2100-02-29',
    '2000-02-29',
    '1/1/100',
    '2/29/00',
]


def grid() -> list[str]:
    values = []
    for dates, seps in (
        (YEAR_FIRST_DATES, ['-', '/']),
        (DAY_OR_MONTH_DATES, ['/', '-']),
    ):
        for template in dates:
            for sep in seps:
                d = template.format(s=sep)
                values += [d + t for t in TIMES]
    for d in ['2024-01-05', '15/03/2024', '1/5/24']:
        for t in [' 10:30', 'T10:30:15', ' 9:30:15.25']:
            if d.startswith('2024') and t.startswith(' 9'):
                continue
            values += [d + t + z for z in ZONES]
    return values


def random_columns(rng: random.Random, n: int) -> list[list[str | None]]:
    def one_date(lead: str, sep: str, ylen: int) -> str:
        y = rng.choice(
            [1999, 2020, 2024, 1850, 2100, 1, 9999, rng.randint(1900, 2100)]
        )
        m = rng.choice([1, 2, 3, 11, 12, 13, rng.randint(1, 12)])
        d = rng.choice(
            [1, 2, 12, 13, 15, 28, 29, 30, 31, 0, 32, rng.randint(1, 28)]
        )
        if lead == 'year':
            return f'{y:04d}{sep}{m:02d}{sep}{d:02d}'
        a, b = (d, m) if rng.random() < 0.5 else (m, d)
        return f'{a:02d}{sep}{b:02d}{sep}{str(y % 10**ylen).zfill(ylen)}'

    def one_time() -> str:
        if rng.random() < 0.4:
            return ''
        t = f'{rng.choice([" ", "T"])}{rng.choice([0, 9, 10, 23, 24]):02d}:{rng.randint(0, 59):02d}'
        if rng.random() < 0.6:
            t += f':{rng.choice([0, 59, 60, rng.randint(0, 59)]):02d}'
            if rng.random() < 0.4:
                t += '.' + ''.join(
                    rng.choice('0123456789')
                    for _ in range(rng.choice([1, 3, 6, 9, 10]))
                )
        if rng.random() < 0.35:
            t += rng.choice(['', ' ']) + rng.choice(
                ['Z', '+01:00', '+0100', '-08:00', '+01', 'UTC', 'CST']
            )
        return t

    columns = []
    for _ in range(n):
        lead = rng.choice(['year', 'year', 'dm'])
        sep = rng.choice('-/')
        ylen = rng.choice([4, 4, 2, 3])
        tail = one_time()
        column: list[str | None] = []
        for _ in range(rng.randint(1, 8)):
            roll = rng.random()
            if roll < 0.15:
                column.append(rng.choice(OTHERS + [None, '']))
                continue
            column.append(
                one_date(lead, sep, ylen)
                + (tail if roll < 0.85 else one_time())
            )
        columns.append(column)
    return columns


def columns() -> list[list[str | None]]:
    rng = random.Random(2512)
    values = grid()
    out: list[list[str | None]] = [[v] for v in values]
    # Day or month first values decided either way round by another value.
    for v in values:
        found = shape(v)
        if found and found[0] == 'dm':
            out += [[v, '13/01/2024'], [v, '01/13/2024']]
    out += [[v] for v in OTHERS + EDGES]
    # Outside values among values in the list, and among each other.
    for v in OTHERS:
        out += [['2024-01-05', v], [v, '2024-01-06', '2024-01-07']]
    out += [rng.sample(values, 2) for _ in range(300)]
    out += [rng.sample(values, 4) for _ in range(100)]
    out += random_columns(rng, 400)
    out += [
        # The orders: decided, contradicted, open, open but alike.
        ['01/02/2024', '13/02/2024'],
        ['01/02/2024', '02/13/2024'],
        ['01/02/2024'],
        ['13/02/2024', '02/13/2024'],
        ['05/05/2024', '2024-01-01'],
        ['13/13/2020', '01/02/2024'],
        ['13/13/2020', '15/03/2024'],
        ['15/03/2024 10:30', '01/02/2024 10:30+01:00'],
        ['01/02/2024', None, '', '13/02/2024'],
        # Offsets mixed, with and without one, and alike.
        ['2024-01-05 10:00+01:00', '2024-01-05 10:00+02:00'],
        ['2024-01-05 10:00+01:00', '2024-01-05 10:00'],
        ['2024-01-05 10:00Z', '2024-01-05 10:00+00:00', '2024-01-05 10:00 UTC'],
        ['2024-01-05 10:00-00', '2024-01-05 10:00 GMT'],
        ['2024-01-05 10:00+01:00', '2024-13-05 10:00'],
        ['2024-01-05 10:00+01:00', 'not a date'],
        # Missing values only.
        [None],
        [None, ''],
        ['', ' ', None],
    ]
    # Two-digit years at the pivot, each side of both ends.
    for yy in (YEAR + 48, YEAR + 49, YEAR + 50, YEAR - 50, YEAR - 51):
        yy %= 100
        out += [
            [f'01/15/{yy:02d}'],
            [f'15/3/{yy:02d} 10:30'],
            [f'1-13-{yy:02d}', f'2-6-{yy:02d}'],
        ]
    return out


def main() -> None:
    document = {
        'format': 'sdm-text-times/2',
        'year': YEAR,
        'columns': [
            {
                'values': c,
                'column': column(c, utc=False),
                'utc': column(c, utc=True),
                'share': share(c),
                'day_first': day_first(c),
            }
            for c in columns()
        ],
    }
    with open(sys.argv[1], 'w', encoding='utf-8') as out:
        json.dump(document, out, separators=(',', ':'), ensure_ascii=False)
        out.write('\n')


if __name__ == '__main__':
    main()
