# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Records ``time-ranges.json``, the zone answers at the ends of the time
range in every vendored zone (design, "Time": T1.25 adds vectors at both
ends of the int64 microsecond range and at 9999-12-31 in every vendored
zone), which ``time_ranges.rs`` holds ``sdm_engine::time`` to.

Run once, by hand, under the reference's Python (3.12, ``zoneinfo``); CI
never runs it (F4):

    python rust/crates/sdm-engine/tests/time_ranges_record.py OUT.json

Each zone is read from ``rust/testdata/tzdata-2026c`` by ``zoneinfo``, the
library pandas 3 resolves zones with. ``to_local`` is an instant's wall
clock and offset in the zone; ``localize`` is the instant a wall clock
names, or ``gap`` or ``fold`` where the zone skips or repeats it, as
pandas' ``tz_localize`` refuses by default. Python's ``datetime`` holds
the years 1 to 9999; an instant or wall clock outside them is moved by
whole 400-year Gregorian cycles (146,097 days, a whole number of weeks, so
every rule a zone database writes repeats with it) into the years 401 to
800, before every vendored zone's first transition, or 2401 to 2800,
after its last, read there, and moved back. ``pandas`` records what pandas
3 itself answers for the same instant (``Timestamp.tz_convert``), which
fails or overflows at the range's ends; the core answers by the rule
above (register DT-10).
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys
import zoneinfo

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3] / 'testdata/tzdata-2026c'
UTC = dt.timezone.utc
EPOCH = dt.datetime(1970, 1, 1, tzinfo=UTC)
CYCLE_DAYS = 146_097
CYCLE_NS = CYCLE_DAYS * 86_400 * 10**9
NS = 10**9

I64 = 2**63


def at(y: int, m: int, d: int, seconds: int = 0, nanos: int = 0) -> int:
    return (days_from_civil(y, m, d) * 86_400 + seconds) * NS + nanos


WALLS = [
    (1, 1, 1, 0, 0, 0, 0),
    (9999, 12, 31, 0, 0, 0, 0),
    (9999, 12, 31, 23, 59, 59, 999_999_000),
    (-290_308, 12, 21, 19, 59, 5, 224_192_000),
    (294_247, 1, 10, 4, 0, 54, 775_807_000),
]


def days_from_civil(y: int, m: int, d: int) -> int:
    y -= m <= 2
    era = y // 400
    yoe = y - era * 400
    doy = (153 * (m - 3 if m > 2 else m + 9) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * CYCLE_DAYS + doe - 719_468


def civil_from_days(z: int) -> tuple[int, int, int]:
    z += 719_468
    era = z // CYCLE_DAYS
    doe = z - era * CYCLE_DAYS
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + 3 if mp < 10 else mp - 9
    return yoe + era * 400 + (m <= 2), m, d


def civil(ns: int) -> str:
    seconds, nanos = divmod(ns, NS)
    days, of_day = divmod(seconds, 86_400)
    y, m, d = civil_from_days(days)
    year = f'{y:04d}' if 0 <= y <= 9999 else f'{y:+07d}'
    return (
        f'{year}-{m:02d}-{d:02d}T{of_day // 3600:02d}:{of_day // 60 % 60:02d}:'
        f'{of_day % 60:02d}.{nanos:09d}'
    )


def cycles_to(ns: int) -> int:
    """The cycles to move ``ns`` by into the years 401 to 800 or 2401 to
    2800, or 0 inside the years 2 to 9998."""
    low = days_from_civil(2, 1, 1) * 86_400 * NS
    high = days_from_civil(9998, 12, 31) * 86_400 * NS
    if low <= ns <= high:
        return 0
    start = days_from_civil(401 if ns < low else 2401, 1, 1) * 86_400 * NS
    return (ns - start) // CYCLE_NS


def offset_at(zone: zoneinfo.ZoneInfo, ns: int) -> int:
    moved = ns - cycles_to(ns) * CYCLE_NS
    when = EPOCH + dt.timedelta(microseconds=moved // 1_000)
    return int(when.astimezone(zone).utcoffset().total_seconds())


def to_local(zone: zoneinfo.ZoneInfo, ns: int) -> dict:
    offset = offset_at(zone, ns)
    return {'local': civil(ns + offset * NS), 'offset': offset}


def localize(zone: zoneinfo.ZoneInfo, wall: tuple) -> object:
    y, m, d, hh, mm, ss, nanos = wall
    wall_ns = (
        days_from_civil(y, m, d) * 86_400 + hh * 3600 + mm * 60 + ss
    ) * NS + nanos
    cycles = cycles_to(wall_ns)
    naive = dt.datetime(y - 400 * cycles, m, d, hh, mm, ss, nanos // 1_000)
    answers = set()
    for fold in (0, 1):
        offset = int(
            naive.replace(tzinfo=zone, fold=fold).utcoffset().total_seconds()
        )
        utc = wall_ns - offset * NS
        if to_local(zone, utc)['local'] == civil(wall_ns):
            answers.add(utc)
    if not answers:
        return 'gap'
    if len(answers) > 1:
        return 'fold'
    return str(answers.pop())


def pandas_answer(zone_name: str, ns: int) -> object:
    try:
        stamp = pd.Timestamp(ns // 1_000, unit='us', tz='UTC')
        with open(ROOT / zone_name, 'rb') as tzif:
            zone = zoneinfo.ZoneInfo.from_file(tzif, key=zone_name)
        local = stamp.tz_convert(zone)
        return {
            'local': local.tz_localize(None).isoformat(),
            'offset': int(local.utcoffset().total_seconds()),
        }
    except Exception as error:  # noqa: BLE001
        return {'error': type(error).__name__}


def instants() -> dict[str, int]:
    return {
        'int64 microseconds, least': (-I64) * 1_000,
        'int64 microseconds, greatest': (I64 - 1) * 1_000,
        'int64 nanoseconds, least': -I64,
        'int64 nanoseconds, greatest': I64 - 1,
        '0001-01-01T00:00Z': at(1, 1, 1),
        '9999-12-30T22:00Z': at(9999, 12, 30, 22 * 3600),
        '9999-12-31T00:00Z': at(9999, 12, 31),
        '9999-12-31T23:59:59.999999Z': at(9999, 12, 31, 86_399, 999_999_000),
        '1970-01-01T00:00Z': 0,
    }


def main() -> None:
    zones = sorted(
        str(path.relative_to(ROOT))
        for path in ROOT.rglob('*')
        if path.is_file() and path.name != '+VERSION'
    )
    vectors = []
    for name in zones:
        with open(ROOT / name, 'rb') as tzif:
            zone = zoneinfo.ZoneInfo.from_file(tzif, key=name)
        for label, ns in instants().items():
            vectors.append(
                {
                    'kind': 'to_local',
                    'zone': name,
                    'label': label,
                    'instant': str(ns),
                    'want': to_local(zone, ns),
                    'pandas': pandas_answer(name, ns),
                }
            )
        for wall in WALLS:
            vectors.append(
                {
                    'kind': 'localize',
                    'zone': name,
                    'wall': list(wall),
                    'want': localize(zone, wall),
                }
            )
    document = {
        'format': 'sdm-time-ranges/1',
        'tzdata': (ROOT / '+VERSION').read_text().strip(),
        'vectors': vectors,
    }
    with open(sys.argv[1], 'w') as out:
        json.dump(document, out, indent=1)
        out.write('\n')


if __name__ == '__main__':
    main()
