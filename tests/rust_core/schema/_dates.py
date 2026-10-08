# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Generated text columns of dates, and the operations a column of dates is
put through.

A column is mostly one shape, with some values of another, malformed,
missing or not dates at all, because that is what decides whether a column
reads: year-first dates and times, with and without seconds, fractions and
zones; day- or month-first dates with one- or two-digit parts and two- or
four-digit years; compact ``YYYYMMDD``; month names; part codes shaped like
dates; impossible dates; implausible years; padding; and junk.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

SEED = 20261007
COUNT = 100_000

#: The operations a generated column is put through, besides ``dates.read``
#: with the conversion that works, where one does.
OPERATIONS = (
    'dates.day_comes_first',
    'dates.looks_unambiguous',
    'dates.shapes_written',
    'dates.reads_as_a_date',
    'dates.conversion_that_works',
    'time_column.read',
    'infer.timestamp',
)

_ZONES = (
    'Z',
    '+00:00',
    '+05:30',
    '-08:00',
    '+0530',
    '-0800',
    '+05',
    ' UTC',
    'UTC',
    ' GMT',
    ' EST',
    ' America/New_York',
    ' XYZ',
)
_MONTHS = (
    'Jan',
    'Feb',
    'Mar',
    'Apr',
    'May',
    'Jun',
    'Jul',
    'Aug',
    'Sep',
    'Oct',
    'Nov',
    'Dec',
)
_FULL_MONTHS = ('January', 'March', 'September', 'December')
_JUNK = (
    '',
    ' ',
    'n/a',
    'unknown',
    'TBD',
    '-',
    'abc',
    'AB-1234',
    '12345',
    '0',
    'NaT',
)
_COLUMNS = ('created_at', 'date', 'ts', 'value', 'x', 'updated', 'order_dt')


def _year(rng: random.Random) -> int:
    roll = rng.random()
    if roll < 0.85:
        return rng.randint(1990, 2030)
    if roll < 0.93:
        return rng.randint(1850, 1899)
    return rng.choice((2101, 2201, 4711, 9999, 1000, 1677, 2262))


def _day(rng: random.Random, month: int, year: int) -> int:
    if rng.random() < 0.03:
        return rng.choice((0, 29, 30, 31, 32))
    last = (
        31,
        29 if year % 4 == 0 else 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    )
    return rng.randint(1, last[(month - 1) % 12])


def _month(rng: random.Random) -> int:
    if rng.random() < 0.02:
        return rng.choice((0, 13))
    return rng.randint(1, 12)


def _time(rng: random.Random, sep: str) -> str:
    hour, minute = rng.randint(0, 23), rng.randint(0, 59)
    text = f'{sep}{hour:02d}:{minute:02d}'
    roll = rng.random()
    if roll < 0.6:
        text += f':{rng.randint(0, 59):02d}'
        if rng.random() < 0.3:
            text += '.' + str(rng.randint(0, 999_999)).zfill(rng.choice((3, 6)))
    if rng.random() < 0.02:
        text = f'{sep}{hour}:{minute:02d}'
    return text


class _Shape:
    r"""How one column writes its dates."""

    def __init__(self, rng: random.Random) -> None:
        self.family = rng.choices(
            (
                'iso',
                'iso-time',
                'slash',
                'slash-time',
                'compact',
                'month-name',
                'code',
                'junk',
            ),
            weights=(26, 20, 22, 10, 5, 6, 6, 5),
        )[0]
        self.sep = rng.choice(('-', '-', '-', '/'))
        self.time_sep = rng.choice((' ', 'T'))
        self.zone = rng.choice(_ZONES) if rng.random() < 0.3 else ''
        self.mixed_zones = rng.random() < 0.1
        self.dayfirst = rng.random() < 0.5
        self.short_year = rng.random() < 0.1
        self.pad = rng.random() < 0.15
        self.month_style = rng.randrange(4)

    def value(self, rng: random.Random) -> str:
        year, month = _year(rng), _month(rng)
        day = _day(rng, month, year)
        family = self.family
        if family in ('iso', 'iso-time'):
            text = f'{year:04d}{self.sep}{month:02d}{self.sep}{day:02d}'
            if rng.random() < 0.02:
                text = f'{year}{self.sep}{month}{self.sep}{day}'
            if family == 'iso-time':
                text += _time(rng, self.time_sep)
                zone = rng.choice(_ZONES) if self.mixed_zones else self.zone
                text += zone if rng.random() < 0.9 else ''
        elif family in ('slash', 'slash-time'):
            first, second = (day, month) if self.dayfirst else (month, day)
            width = rng.choice(('{:d}', '{:02d}'))
            shown_year = (
                f'{year % 100:02d}' if self.short_year else f'{year:04d}'
            )
            sep = self.sep if rng.random() < 0.95 else '-'
            text = f'{width.format(first)}{sep}{width.format(second)}{sep}{shown_year}'
            if family == 'slash-time':
                text += _time(rng, ' ')
                if self.zone and rng.random() < 0.8:
                    text += self.zone
        elif family == 'compact':
            text = f'{year:04d}{month:02d}{day:02d}'
        elif family == 'month-name':
            name = _MONTHS[(month - 1) % 12]
            text = (
                f'{name} {day:d}, {year}',
                f'{day:02d}-{name}-{year}',
                f'{_FULL_MONTHS[month % 4]} {day}, {year}',
                f'{day} {name} {year}',
            )[self.month_style]
        elif family == 'code':
            text = f'{rng.randint(1000, 9999)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}'
            if rng.random() < 0.3:
                text += rng.choice(('-A', 'X', '/7'))
        else:
            text = rng.choice(_JUNK)
        if self.pad and rng.random() < 0.5:
            text = rng.choice((' ', '  ', '\t')) + text + rng.choice(('', ' '))
        return text


def column(rng: random.Random) -> dict[str, Any]:
    r"""One generated column: its values, the options its operations take,
    and its name."""
    shape = _Shape(rng)
    other = _Shape(rng)
    size = rng.choice((1, 2, 3, 5, 8, 12, 12, 20, 20, 30))
    values: list[Any] = []
    noise = rng.choice((0.0, 0.0, 0.03, 0.1, 0.3))
    for _ in range(size):
        roll = rng.random()
        if roll < noise / 3:
            values.append(None)
        elif roll < noise * 2 / 3:
            values.append(rng.choice(_JUNK))
        elif roll < noise:
            values.append(other.value(rng))
        else:
            values.append(shape.value(rng))
    return {
        'values': values,
        'only_plausible_years': rng.random() < 0.7,
        'column': rng.choice(_COLUMNS),
    }


def columns(count: int = COUNT, seed: int = SEED) -> Iterator[dict[str, Any]]:
    rng = random.Random(seed)
    for _ in range(count):
        yield column(rng)


def inputs(generated: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    r"""Each operation and its input for one generated column."""
    values = generated['values']
    out: list[tuple[str, dict[str, Any]]] = []
    for operation in OPERATIONS:
        if operation == 'dates.conversion_that_works':
            given = {
                'values': values,
                'only_plausible_years': generated['only_plausible_years'],
            }
        elif operation == 'infer.timestamp':
            given = {'values': values, 'column': generated['column']}
        else:
            given = {'values': values}
        out.append((operation, given))
    return out
