#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records ``contract/t136-host2-zone-vectors.json``: register P1B-81(b)'s
first value read with a zone (T1.36), measured through DuckDB 1.5.6 and the
public calls on both routes.

Written by T1.36's independent test author, a sibling of
``host2_values_record.py``, whose sources (``shop`` and a setup), calls and
answers it reuses. Run from this repository's root::

    python -I contract/tools/host2_zones_record.py \
        [--reference-python REF] [--candidate-python CAND]

The reference's ``Schema.from_source`` reads the first non-null value of
each time-column candidate as a Python ``datetime``
(``predictive_query/time_columns.py``, ``_refused_for_values``:
``present.iloc[0]``, pyarrow's ``TimestampScalar.as_py``). For a
``TIMESTAMPTZ`` that is the UTC instant moved to the zone DuckDB hands the
column over in: the UTC datetime built first, then moved, so it fails
unnamed (``OverflowError``) where either the UTC instant or the wall clock
there falls outside the years 1 to 9999, though the other does not.
Each vector here is ``users.f TIMESTAMPTZ`` (unless it says otherwise),
``2020-01-01 00:00+00`` in every row but user 0's, which holds the vector's
instant, read in the database's ``TimeZone`` its setup sets (``SET GLOBAL``).

``zones`` records the facts the rule rests on: the zone DuckDB hands a
``TIMESTAMPTZ`` over in, to the new connection both routes read on
(``cursor()``): the database's ``GLOBAL`` ``TimeZone``, not one ``SET`` on
the caller's connection, and not the zone an Arrow import came with (unset,
it is the machine's, so every vector sets it); and, per vector, the zone's offset at its instant and the wall clock
there, as Python's ``zoneinfo`` gives them (the reference's conversion).
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import re
import subprocess
import sys
import zoneinfo
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/t136-host2-zone-vectors.json'


def _values_recorder() -> ModuleType:
    name = 'host2_values_record'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / 'contract/tools/host2_values_record.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = _values_recorder()

ABOUT = (
    "Register P1B-81(b)'s first value read with a zone (T1.36 fix 11, "
    'written by the independent test author before the fix): what DuckDB '
    '1.5.6, the reference and the core do with a TIMESTAMPTZ whose instant '
    'lies within the years 1 to 9999 but whose wall clock, in the zone DuckDB '
    'hands it over in, does not. `vectors`: each a source (contract/'
    "bases.yaml's `shop`, then `setup`), the column it is about, its role, "
    'the zone the database is in (SET GLOBAL TimeZone) and the instant in '
    "user 0's row, and the public calls' answer on each route, recorded by "
    'contract/tools/host2_zones_record.py with the answers of '
    'contract/tools/host2_values_record.py: `reference` from the reference '
    "checkout's interpreter, `core_at_measurement` from the candidate on the "
    'core route, evidence of what the core did when measured, never an '
    'expectation. `zones`: the Arrow type a new connection to the database '
    '(cursor(), as both routes read) hands a TIMESTAMPTZ over as, for each '
    "zone and with the caller's connection SET otherwise; what an Arrow "
    "import's zone becomes; and per vector the zone's offset at its instant "
    "and the wall clock there, as Python's zoneinfo gives them (pyarrow's "
    'as_py: astimezone to zoneinfo.ZoneInfo, from the system tz database), '
    'the microseconds DuckDB holds, whether the UTC datetime and the wall '
    "clock each lie within the years, counted, and as_py's own answer. "
    "The reference's from_source fails unnamed (OverflowError, "
    'predictive_query/time_columns.py _refused_for_values, present.iloc[0]) '
    'exactly where the UTC datetime as_py builds first, or the wall clock it '
    'moves it to, leaves 0001-01-01T00:00:00 to '
    "9999-12-31T23:59:59.999999, both ends held. `zone.whole.*` (fix 11's "
    'review): the same instants in every row, a constant column the '
    'reference warns of, the warning showing the value at its wall clock '
    '("every row holds 0001-01-01 00:00:00-04:56:02"). Held by '
    "contract/tests/test_t136_host2_zones.py. `zone.preview.*` (fix 13's "
    'review): a value only past the 1,000 rows the preview reads, warned of '
    "with the whole table's MIN, which the reference reads through DuckDB's "
    "fetchall: outside Python's years DuckDB's own text, in UTC "
    '(`zones.fetchall_outside_the_years`). `p1b39`: register P1B-39 '
    '(approved), the reference without pytz (`reference`, its checkout), '
    'the reference with pytz (`reference_with_pytz`, the default route of '
    'an interpreter of the same environment with pytz installed, '
    '`p1b39.environment`) and the core, each with its from_source warnings '
    'whole. `p1b89`: register P1B-89 (owner-pending), recorded the same way: '
    "with pytz, DuckDB's fetchall fails where a zoned value's wall clock in "
    "the session's zone leaves the years, so the reference falls back to its "
    'preview there too.'
)

#: The value every other row holds.
ELSEWHERE = '2020-01-01 00:00:00'


def literal(instant: str) -> str:
    r"""``instant`` as the ``TIMESTAMPTZ`` literal written: as given where
    it carries its offset (``+09``, ``-05``), else in UTC (``+00``)."""
    return (
        instant
        if re.search(r'[+-]\d{2}(:\d{2})*$', instant)
        else instant + '+00'
    )


def first(
    zone: str, instant: str, name: str = 'f', before: tuple[str, ...] = ()
) -> list[str]:
    r"""``users.<name> TIMESTAMPTZ``, ``instant`` (UTC) in user 0's row and
    ``ELSEWHERE`` in the others, read under ``zone``: the database's
    (``GLOBAL``) time zone, the one a new connection to it starts with."""
    return [
        f"SET GLOBAL TimeZone = '{zone}'",
        *before,
        f'ALTER TABLE users ADD COLUMN {name} TIMESTAMPTZ',
        f"UPDATE users SET {name} = TIMESTAMPTZ '{ELSEWHERE}+00'",
        f"UPDATE users SET {name} = TIMESTAMPTZ '{literal(instant)}' "
        'WHERE user_id = 0',
    ]


def whole(zone: str, instant: str) -> list[str]:
    r"""``users.f TIMESTAMPTZ``, ``instant`` (UTC) in every row, read under
    ``zone``: a constant column, which both routes warn of, its value shown
    at its wall clock."""
    return [
        f"SET GLOBAL TimeZone = '{zone}'",
        'ALTER TABLE users ADD COLUMN f TIMESTAMPTZ',
        f"UPDATE users SET f = TIMESTAMPTZ '{literal(instant)}'",
    ]


def events(zone: str, instant: str, held_from: int = 1000) -> list[str]:
    r"""``events``, beside ``shop`` (so that the time columns' warning is
    given): 2,500 rows, ``f TIMESTAMPTZ`` null in rows before ``held_from``
    and ``instant`` (a literal) in the rest, read under ``zone``."""
    return [
        f"SET GLOBAL TimeZone = '{zone}'",
        'CREATE TABLE events AS SELECT i::BIGINT AS event_id, CASE WHEN '
        f"i < {held_from} THEN NULL ELSE TIMESTAMPTZ '{instant}' END AS f "
        'FROM range(2500) t(i)',
    ]


#: The role of a value only past the 1,000 rows the preview reads.
PREVIEW = (
    'events.f, 2,500 rows: null in the 1,000 the preview reads, the instant '
    'in the 1,500 after (a constant column, warned of with the whole '
    "table's MIN)"
)

NY = 'America/New_York'
TOKYO = 'Asia/Tokyo'
KIRITIMATI = 'Pacific/Kiritimati'
PAGO_PAGO = 'Pacific/Pago_Pago'
WHOLE = (
    "the entity table's only timestamp, the instant in every row (a "
    'constant column, warned of)'
)
ENTITY = "the entity table's only timestamp (its time column), user 0's row"


#: The review's reproduction (Codex, final review round 2), shop dropped.
def verbatim(instant: str) -> list[str]:
    r"""The review's reproduction (shop dropped) with ``instant`` (a
    ``TIMESTAMPTZ`` literal) in every row of ``users.ts``."""
    return [
        f"SET TimeZone='{NY}'",
        'DROP TABLE orders',
        'DROP TABLE users',
        'CREATE TABLE users AS SELECT i::BIGINT AS user_id, '
        f"TIMESTAMPTZ '{instant}' AS ts FROM range(8) t(i)",
    ]


VERBATIM = [
    f"SET TimeZone='{NY}'",
    'DROP TABLE orders',
    'DROP TABLE users',
    'CREATE TABLE users AS SELECT i::BIGINT AS user_id, '
    "TIMESTAMPTZ '0001-01-01 00:00:00+00' AS ts FROM range(8) t(i)",
]

#: name: (zone, instant in UTC, role, setup or None for ``first``).
_SPECS: dict[str, tuple[str, str, str, list[str] | None]] = {
    # The review's reproduction, verbatim, but for the database's zone: the
    # reference reads on a new connection (DuckDB's cursor()), which starts
    # in the database's zone, not in one SET on the caller's connection, and
    # that zone, unset, is the machine's (America/Los_Angeles where the
    # review measured). With it UTC the reference reads the same table.
    'zone.verbatim.los-angeles.year-1': (
        'America/Los_Angeles',
        '0001-01-01 00:00:00',
        "users.ts in every row, the table's only timestamp (the review's "
        'reproduction, shop dropped, the database in the zone the review '
        "machine's DuckDB starts in)",
        [
            "SET GLOBAL TimeZone = 'America/Los_Angeles'",
            *VERBATIM,
        ],
    ),
    'zone.verbatim.utc.year-1': (
        'UTC',
        '0001-01-01 00:00:00',
        "users.ts in every row (control: the review's reproduction, the "
        'database in UTC)',
        ["SET GLOBAL TimeZone = 'UTC'", *VERBATIM],
    ),
    # The caller's connection's own zone does not count.
    'zone.session.utc.global-new-york.year-1': (
        NY,
        '0001-01-01 00:00:00',
        ENTITY + " (the caller's connection SET to UTC, the database New York)",
        ["SET TimeZone = 'UTC'", *first(NY, '0001-01-01 00:00:00')],
    ),
    'zone.session.new-york.global-utc.year-1': (
        'UTC',
        '0001-01-01 00:00:00',
        ENTITY + " (control: the caller's connection SET to New York, the "
        'database UTC)',
        [f"SET TimeZone = '{NY}'", *first('UTC', '0001-01-01 00:00:00')],
    ),
    # The lower end, a negative offset: New York's LMT, -4:56:02.
    'zone.first.new-york.year-1': (NY, '0001-01-01 00:00:00', ENTITY, None),
    'zone.first.new-york.year-1.local-midnight-less-1us': (
        NY,
        '0001-01-01 04:56:01.999999',
        ENTITY,
        None,
    ),
    'zone.first.new-york.year-1.local-midnight': (
        NY,
        '0001-01-01 04:56:02',
        ENTITY + ' (control: the wall clock 0001-01-01 00:00:00 exactly)',
        None,
    ),
    'zone.first.new-york.year-1.0500': (
        NY,
        '0001-01-01 05:00:00',
        ENTITY + ' (control: 00:03:58 on the wall clock, not midnight)',
        None,
    ),
    'zone.first.new-york.year-1.hour-inside': (
        NY,
        '0001-01-01 05:56:02',
        ENTITY + ' (control: an hour inside the years)',
        None,
    ),
    'zone.first.utc.year-1': (
        'UTC',
        '0001-01-01 00:00:00',
        ENTITY + ' (control: read in UTC)',
        None,
    ),
    # The upper end, a positive offset: Tokyo's +9.
    'zone.first.tokyo.year-9999': (TOKYO, '9999-12-31 23:59:59', ENTITY, None),
    'zone.first.tokyo.year-9999.local-midnight': (
        TOKYO,
        '9999-12-31 15:00:00',
        ENTITY + ' (the wall clock 10000-01-01 00:00:00 exactly)',
        None,
    ),
    'zone.first.tokyo.year-9999.local-last-us': (
        TOKYO,
        '9999-12-31 14:59:59.999999',
        ENTITY + ' (control: the wall clock 9999-12-31 23:59:59.999999)',
        None,
    ),
    'zone.first.tokyo.year-9999.hour-inside': (
        TOKYO,
        '9999-12-31 13:59:59.999999',
        ENTITY + ' (control: an hour inside the years)',
        None,
    ),
    'zone.first.utc.year-9999': (
        'UTC',
        '9999-12-31 23:59:59.999999',
        ENTITY + ' (control: read in UTC)',
        None,
    ),
    'zone.first.new-york.year-9999': (
        NY,
        '9999-12-31 23:59:59.999999',
        ENTITY + ' (control: a negative offset at the upper end)',
        None,
    ),
    # The offset at that instant, not today's: Kiritimati is +14 now and
    # -10:29:20 (LMT) in the year 1; Pago Pago -11 now and +12:37:12 then.
    'zone.first.kiritimati.year-9999': (
        KIRITIMATI,
        '9999-12-31 10:00:00',
        ENTITY + ' (the wall clock 10000-01-01 00:00:00 exactly, at +14)',
        None,
    ),
    'zone.first.kiritimati.year-9999.local-last-us': (
        KIRITIMATI,
        '9999-12-31 09:59:59.999999',
        ENTITY + ' (control: the wall clock 9999-12-31 23:59:59.999999)',
        None,
    ),
    'zone.first.kiritimati.year-1': (
        KIRITIMATI,
        '0001-01-01 00:00:00',
        ENTITY + ' (+14 today, but -10:29:20 in the year 1)',
        None,
    ),
    'zone.first.kiritimati.year-1.local-midnight': (
        KIRITIMATI,
        '0001-01-01 10:29:20',
        ENTITY + ' (control: the wall clock 0001-01-01 00:00:00 at -10:29:20)',
        None,
    ),
    'zone.first.pago-pago.year-1': (
        PAGO_PAGO,
        '0001-01-01 00:00:00',
        ENTITY + ' (control: -11 today, but +12:37:12 in the year 1)',
        None,
    ),
    # A candidate that is not the time column, and one that is no candidate.
    'zone.candidate.new-york.year-1': (
        NY,
        '0001-01-01 00:00:00',
        'users.f, a candidate after users.created_at, the time column chosen '
        "(a creation stamp), user 0's row",
        first(
            NY,
            '0001-01-01 00:00:00',
            before=(
                'ALTER TABLE users ADD COLUMN created_at TIMESTAMPTZ',
                "UPDATE users SET created_at = TIMESTAMPTZ '2020-01-01 "
                "00:00:00+00' + to_days(user_id::INTEGER)",
            ),
        ),
    ),
    'zone.first.new-york.year-1.loaded_at': (
        NY,
        '0001-01-01 00:00:00',
        "users.loaded_at, user 0's row (control: a load stamp is no candidate)",
        first(NY, '0001-01-01 00:00:00', name='loaded_at'),
    ),
    # The same instants filling the column: the constant-column warning shows
    # the value at its wall clock, which pandas cannot always format (its
    # isoformat refuses a zoned value at local midnight of the year 1).
    'zone.whole.new-york.year-1.local-midnight': (
        NY,
        '0001-01-01 04:56:02',
        WHOLE + ' (control: the wall clock 0001-01-01 00:00:00 exactly)',
        whole(NY, '0001-01-01 04:56:02'),
    ),
    'zone.whole.new-york.year-1.hour-inside': (
        NY,
        '0001-01-01 05:56:02',
        WHOLE + ' (control: an hour inside the years)',
        whole(NY, '0001-01-01 05:56:02'),
    ),
    'zone.whole.pago-pago.year-1': (
        PAGO_PAGO,
        '0001-01-01 00:00:00',
        WHOLE + ' (control: the wall clock 0001-01-01 12:37:12)',
        whole(PAGO_PAGO, '0001-01-01 00:00:00'),
    ),
    'zone.whole.kiritimati.year-1.local-midnight': (
        KIRITIMATI,
        '0001-01-01 10:29:20',
        WHOLE + ' (control: the wall clock 0001-01-01 00:00:00 exactly)',
        whole(KIRITIMATI, '0001-01-01 10:29:20'),
    ),
    'zone.whole.utc.year-1': (
        'UTC',
        '0001-01-01 00:00:00',
        WHOLE + ' (control: read in UTC)',
        whole('UTC', '0001-01-01 00:00:00'),
    ),
    'zone.whole.tokyo.year-9999.local-last-us': (
        TOKYO,
        '9999-12-31 14:59:59.999999',
        WHOLE + ' (control: the wall clock 9999-12-31 23:59:59.999999)',
        whole(TOKYO, '9999-12-31 14:59:59.999999'),
    ),
    'zone.whole.kiritimati.year-9999.local-last-us': (
        KIRITIMATI,
        '9999-12-31 09:59:59.999999',
        WHOLE + ' (control: the wall clock 9999-12-31 23:59:59.999999)',
        whole(KIRITIMATI, '9999-12-31 09:59:59.999999'),
    ),
    'zone.whole.utc.year-9999': (
        'UTC',
        '9999-12-31 23:59:59.999999',
        WHOLE + ' (control: read in UTC)',
        whole('UTC', '9999-12-31 23:59:59.999999'),
    ),
    'zone.whole.new-york.year-1': (
        NY,
        '0001-01-01 00:00:00',
        WHOLE + ' (the wall clock before the year 1)',
        whole(NY, '0001-01-01 00:00:00'),
    ),
    # Fix 11's review (Codex): pyarrow builds the UTC datetime before it
    # moves it to the zone, so the reference fails where either is outside
    # the years, a wall clock within them notwithstanding.
    'zone.verbatim.new-york.year-10000': (
        NY,
        '10000-01-01 00:00:00',
        "users.ts in every row, the table's only timestamp (the review's "
        'reproduction, shop dropped): UTC past the years, the wall clock '
        '9999-12-31 19:00:00',
        [f"SET GLOBAL TimeZone = '{NY}'", *verbatim('10000-01-01 00:00:00+00')],
    ),
    'zone.first.new-york.year-10000': (
        NY,
        '10000-01-01 00:00:00',
        ENTITY + ' (UTC a microsecond past the years, the wall clock '
        '9999-12-31 19:00:00)',
        None,
    ),
    'zone.whole.new-york.year-10000': (
        NY,
        '10000-01-01 00:00:00',
        WHOLE + ' (UTC past the years, the wall clock within them)',
        whole(NY, '10000-01-01 00:00:00'),
    ),
    'zone.verbatim.tokyo.utc-year-0': (
        TOKYO,
        '0001-01-01 00:00:00+09',
        "users.ts in every row, the table's only timestamp (the review's "
        'reproduction, shop dropped): UTC 0000-12-31 15:00, before the '
        'years; the wall clock 0001-01-01 00:18:59 (local mean time)',
        [
            f"SET GLOBAL TimeZone = '{TOKYO}'",
            *verbatim('0001-01-01 00:00:00+09'),
        ],
    ),
    'zone.first.tokyo.utc-year-0': (
        TOKYO,
        '0001-01-01 00:00:00+09',
        ENTITY + ' (UTC before the years, the wall clock within them)',
        None,
    ),
    'zone.whole.tokyo.utc-year-0': (
        TOKYO,
        '0001-01-01 00:00:00+09',
        WHOLE + ' (UTC before the years, the wall clock within them)',
        whole(TOKYO, '0001-01-01 00:00:00+09'),
    ),
    # The same at each end's last microsecond, in a zone whose offset there
    # has seconds (Pago Pago: +12:37:12 local mean time in the year 1, -11
    # in the year 9999).
    'zone.first.pago-pago.utc-year-0.last-us': (
        PAGO_PAGO,
        '0001-12-31 (BC) 23:59:59.999999+00',
        ENTITY + ' (UTC a microsecond before the years, the wall clock '
        '0001-01-01 12:37:11.999999)',
        None,
    ),
    'zone.first.pago-pago.utc-year-10000.local-last-second': (
        PAGO_PAGO,
        '10000-01-01 10:59:59',
        ENTITY + ' (UTC past the years, the wall clock 9999-12-31 23:59:59)',
        None,
    ),
    # Their neighbours, the UTC instant and the wall clock both within the
    # years.
    'zone.first.new-york.year-9999.2300': (
        NY,
        '9999-12-31 23:00:00',
        ENTITY + ' (control: the wall clock 9999-12-31 18:00:00)',
        None,
    ),
    'zone.whole.new-york.year-9999.2300': (
        NY,
        '9999-12-31 23:00:00',
        WHOLE + ' (control: the wall clock 9999-12-31 18:00:00)',
        whole(NY, '9999-12-31 23:00:00'),
    ),
    'zone.first.tokyo.year-1': (
        TOKYO,
        '0001-01-01 00:00:00',
        ENTITY + ' (control: UTC the first instant of the years, the wall '
        'clock 0001-01-01 09:18:59)',
        None,
    ),
    'zone.whole.tokyo.year-1': (
        TOKYO,
        '0001-01-01 00:00:00',
        WHOLE + ' (control: the wall clock 0001-01-01 09:18:59)',
        whole(TOKYO, '0001-01-01 00:00:00'),
    ),
    'zone.first.pago-pago.year-9999': (
        PAGO_PAGO,
        '9999-12-31 23:59:59.999999',
        ENTITY + ' (control: UTC the last instant of the years, the wall '
        'clock 9999-12-31 12:59:59.999999)',
        None,
    ),
    # Fix 13's review: a value the preview never holds. events.f is null in
    # the 1,000 rows read and holds the instant in the 1,500 after, so both
    # routes take its start from the whole table's MIN and refuse it as
    # constant, the warning showing that MIN. The reference reads it
    # through DuckDB's fetchall, which hands a TIMESTAMPTZ outside Python's
    # years over as DuckDB's own text, in UTC.
    'zone.preview.new-york.year-10000': (
        NY,
        '10000-01-01 00:00:00',
        PREVIEW,
        events(NY, '10000-01-01 00:00:00+00'),
    ),
    'zone.preview.new-york.year-10000.fraction': (
        NY,
        '10000-06-01 12:34:56.5',
        PREVIEW + ' (a fraction of a second)',
        events(NY, '10000-06-01 12:34:56.5+00'),
    ),
    'zone.preview.tokyo.utc-year-0': (
        TOKYO,
        '0001-01-01 00:00:00+09',
        PREVIEW + ' (UTC before the years)',
        events(TOKYO, '0001-01-01 00:00:00+09'),
    ),
}

#: Register P1B-39 (approved, E83/E85(c)): without pytz the reference's
#: whole-table statement fails for any table holding a zoned candidate
#: (DuckDB's fetchall needs pytz for a TIMESTAMPTZ within Python's years),
#: and the reference chooses from its preview; the core reads the whole
#: table, as the reference does with pytz. name: (zone, instant, role,
#: setup).
_P1B39_SPECS: dict[str, tuple[str, str, str, list[str]]] = {
    'p1b39.held-past-the-preview': (
        'UTC',
        '2020-01-01 00:00:00+00',
        PREVIEW + ' (within the years)',
        events('UTC', '2020-01-01 00:00:00+00'),
    ),
    'p1b39.constant-in-every-row': (
        'UTC',
        '2020-01-01 00:00:00+00',
        'events.f, 2,500 rows, the instant in every row',
        events('UTC', '2020-01-01 00:00:00+00', held_from=0),
    ),
    'p1b39.control.varying': (
        'UTC',
        '2020-01-01 00:00:00+00',
        'events.f, 2,500 rows, the instant plus i hours in row i (control: '
        'the preview says what the whole table says)',
        [
            "SET GLOBAL TimeZone = 'UTC'",
            'CREATE TABLE events AS SELECT i::BIGINT AS event_id, '
            "TIMESTAMPTZ '2020-01-01 00:00:00+00' + to_hours(i) AS f "
            'FROM range(2500) t(i)',
        ],
    ),
}

#: Register P1B-89 (owner-pending; fix 13's re-check, P3-1): with pytz
#: installed, DuckDB's fetchall moves a TIMESTAMPTZ to the session's zone as
#: a Python datetime, which fails (OverflowError) where the wall clock there
#: leaves the years though the instant does not, so the reference's
#: whole-table statement fails all the same and it chooses from its
#: preview; the core reads the whole table. Without pytz this is P1B-39's
#: case.
_P1B89_SPECS: dict[str, tuple[str, str, str, list[str]]] = {
    'p1b89.held-past-the-preview.new-york.year-1': (
        NY,
        '0001-01-01 00:00:00+00',
        PREVIEW + ' (the wall clock before the year 1)',
        events(NY, '0001-01-01 00:00:00+00'),
    ),
    'p1b89.held-past-the-preview.tokyo.year-9999': (
        TOKYO,
        '9999-12-31 23:00:00+00',
        PREVIEW + ' (the wall clock past the year 9999)',
        events(TOKYO, '9999-12-31 23:00:00+00'),
    ),
    'p1b89.control.utc.year-1': (
        'UTC',
        '0001-01-01 00:00:00+00',
        PREVIEW + ' (control: read in UTC, the wall clock within the years)',
        events('UTC', '0001-01-01 00:00:00+00'),
    ),
    'p1b89.control.sentinel.new-york': (
        NY,
        '0001-01-01 00:00:00+00',
        'events.f, 2,500 rows, 2020-01-01+00 plus i hours in row i but row '
        "2,000's, which holds the instant (control: a realistic sentinel, "
        'past the preview)',
        [
            f"SET GLOBAL TimeZone = '{NY}'",
            'CREATE TABLE events AS SELECT i::BIGINT AS event_id, CASE WHEN '
            "i = 2000 THEN TIMESTAMPTZ '0001-01-01 00:00:00+00' ELSE "
            "TIMESTAMPTZ '2020-01-01 00:00:00+00' + to_hours(i) END AS f "
            'FROM range(2500) t(i)',
        ],
    ),
}


def _sources(
    specs: dict[str, tuple[str, str, str, list[str] | None]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, (zone, instant, role, setup) in specs.items():
        column = (
            'events.f'
            if name.startswith(('zone.preview.', 'p1b39.', 'p1b89.'))
            else 'users.ts'
            if name.startswith('zone.verbatim.')
            else 'users.loaded_at'
            if name.endswith('.loaded_at')
            else 'users.f'
        )
        out[name] = {
            'column': column,
            'role': role,
            'zone': zone,
            'instant': literal(instant),
            'setup': setup if setup is not None else first(zone, instant),
        }
    return out


SOURCES = _sources(_SPECS)
P1B39 = _sources(dict(_P1B39_SPECS))
P1B89 = _sources(dict(_P1B89_SPECS))
#: The sections each source of which is recorded with pytz too.
WITH_PYTZ = {'p1b39': ('P1B-39', P1B39), 'p1b89': ('P1B-89', P1B89)}


def wall_clock(zone: str, instant: str) -> dict[str, str]:
    r"""``instant`` (a ``TIMESTAMPTZ`` literal) as the reference reads it in
    ``zone`` (pyarrow's ``as_py``: ``datetime(1970, 1, 1)`` plus the
    microseconds, the UTC datetime, then ``astimezone`` to
    ``zoneinfo.ZoneInfo``): the microseconds DuckDB holds, the UTC datetime,
    the zone's offset there and the wall clock, each or the error it raises,
    and ``as_py``'s own answer."""
    import duckdb
    import pyarrow as pa

    [(micros,)] = (
        duckdb.connect()
        .execute(f"SELECT epoch_us(TIMESTAMPTZ '{instant}')")
        .fetchall()
    )
    tz = zoneinfo.ZoneInfo(zone)

    def said(run: Any) -> str:
        try:
            return str(run())
        except OverflowError as error:
            return f'OverflowError: {error}'

    try:
        utc = dt.datetime(1970, 1, 1) + dt.timedelta(microseconds=micros)
    except OverflowError as error:
        utc = None
        utc_said = f'OverflowError: {error}'
    else:
        utc_said = utc.isoformat()
    local: Any = None
    if utc is None:
        wall = utc_said
    else:
        try:
            local = utc.replace(tzinfo=dt.UTC).astimezone(tz)
        except OverflowError as error:
            wall = f'OverflowError: {error}'
        else:
            wall = local.isoformat()
    # The offset the wall clock was read at; where none was, the zone's at
    # the UTC datetime, or at the end of the years it passed.
    at = utc or (dt.datetime.max if micros > 0 else dt.datetime.min)
    offset = local.utcoffset() if local is not None else tz.utcoffset(at)
    assert offset is not None
    low = (dt.datetime.min - dt.datetime(1970, 1, 1)) // dt.timedelta(
        microseconds=1
    )
    high = (dt.datetime.max - dt.datetime(1970, 1, 1)) // dt.timedelta(
        microseconds=1
    )
    shifted = micros + offset // dt.timedelta(microseconds=1)
    return {
        'microseconds': str(micros),
        # Each reading against 0001-01-01T00:00 .. 9999-12-31T23:59:59.999999,
        # counted, so that a wall clock is known where its UTC datetime
        # cannot be built.
        'utc_within_years': low <= micros <= high,
        'wall_within_years': low <= shifted <= high,
        'utc': utc_said,
        'offset': str(offset),
        'wall_clock': wall,
        'as_py': said(
            lambda: (
                pa.scalar(micros, pa.timestamp('us', tz=zone))
                .as_py()
                .isoformat()
            )
        ),
    }


def zone_facts() -> dict[str, Any]:
    r"""What DuckDB hands a ``TIMESTAMPTZ`` over as, and each vector's zone
    offset and wall clock at its instant."""
    import duckdb
    import pyarrow as pa

    query = "SELECT TIMESTAMPTZ '2020-01-01 00:00:00+00'"

    def types(settings: list[str]) -> dict[str, str]:
        r"""The Arrow type the caller's connection and a new one to the
        same database (``cursor()``, as both routes read) hand over."""
        connection = duckdb.connect()
        for statement in settings:
            connection.execute(statement)
        with connection.cursor() as cursor:
            read = cursor.execute(query).to_arrow_table()
        return {
            'connection': str(
                connection.execute(query).to_arrow_table().schema.field(0).type
            ),
            'cursor': str(read.schema.field(0).type),
        }

    handed = {
        zone: types([f"SET GLOBAL TimeZone = '{zone}'"])['cursor']
        for zone in sorted({spec['zone'] for spec in SOURCES.values()})
    }
    sessions = {
        ' ; '.join(settings): types(settings)
        for settings in (
            [f"SET TimeZone = '{NY}'", "SET GLOBAL TimeZone = 'UTC'"],
            ["SET TimeZone = 'UTC'", f"SET GLOBAL TimeZone = '{NY}'"],
        )
    }
    # An Arrow import's own zone is not kept: DuckDB stores TIMESTAMPTZ (an
    # instant) and hands it back in the session's zone.
    imported = duckdb.connect()
    imported.execute("SET TimeZone = 'UTC'")
    imported.register(
        'arrow_in',
        pa.table({'f': pa.array([0], pa.timestamp('us', tz=KIRITIMATI))}),
    )
    imported.execute('CREATE TABLE t AS SELECT f FROM arrow_in')
    [(stored,)] = imported.execute('SELECT typeof(f) FROM t').fetchall()
    import_kept = {
        'imported': f'timestamp[us, tz={KIRITIMATI}]',
        'stored_as': stored,
        'handed_over_under_UTC': str(
            imported.execute('SELECT f FROM t')
            .to_arrow_table()
            .schema.field(0)
            .type
        ),
    }
    # What DuckDB's fetchall (the reference's read of the whole table's
    # MIN) hands over for a TIMESTAMPTZ outside Python's years: its own
    # text, in UTC, whatever the zone, and with or without pytz.
    fetched = duckdb.connect()
    fetched.execute(f"SET GLOBAL TimeZone = '{NY}'")
    outside = {
        spec['instant']: repr(
            fetched.execute(
                f"SELECT TIMESTAMPTZ '{spec['instant']}'"
            ).fetchall()[0][0]
        )
        for name, spec in SOURCES.items()
        if name.startswith('zone.preview.')
    }
    return {
        'version': duckdb.__version__,
        'handed_over': handed,
        'session_and_global': sessions,
        'arrow_import': import_kept,
        'fetchall_outside_the_years': outside,
        'wall_clocks': {
            name: {
                'zone': spec['zone'],
                'instant': spec['instant'],
                **wall_clock(spec['zone'], spec['instant']),
            }
            for name, spec in SOURCES.items()
        },
    }


def from_source_text(setup: list[Any], route: str) -> list[str]:
    r"""``Schema.from_source``'s warnings over ``shop`` and ``setup`` on
    ``route``, each whole, the core's ``[CODE]`` prefix set aside and
    DuckDB's missing-index advice left out (the answers keep only each
    warning's first line; the constant-column warning names its value in a
    later one)."""
    import re
    import warnings

    if route == 'core':
        os.environ['SDM_CORE_ROUTES'] = ','.join(R.CORE_ROUTES)
    else:
        os.environ.pop('SDM_CORE_ROUTES', None)
    from sdm_client import Schema
    from sdm_connectors import DuckDBSource

    connection = R.source(setup)
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        try:
            Schema.from_source(DuckDBSource(connection))
        except Exception:  # noqa: BLE001, S110 - the answers record it
            pass
    return [
        re.sub(r'^\[[A-Z_]+\] ', '', str(w.message))
        for w in shown
        if 'for optimal database querying' not in str(w.message)
    ]


def _run(python: str, argv: list[str]) -> tuple[list[str], int]:
    done = subprocess.run(
        [python, '-I', str(Path(__file__).resolve()), '--child', *argv],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    if done.returncode != 0:
        sys.stderr.write(done.stderr[-4000:])
    return done.stdout.strip().splitlines(), done.returncode


def main() -> int:
    if sys.argv[1:2] == ['--child']:
        if sys.argv[2] == 'environment':
            sys.argv[1:] = ['--child', 'environment']
            return int(R.main())
        route, names = sys.argv[3], sys.argv[4:]
        every = {**SOURCES, **P1B39, **P1B89}
        for name in names:
            got = R.answer(every[name]['setup'], route)
            got['from_source_text'] = from_source_text(
                every[name]['setup'], route
            )
            print(json.dumps({'name': name, **got}), flush=True)
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--reference-python', default=str(R.REFERENCE / '.venv/bin/python')
    )
    parser.add_argument('--candidate-python', default=sys.executable)
    parser.add_argument(
        '--pytz-python',
        help="an interpreter of the reference's environment with pytz "
        "installed, whose default route (the reference's code) records "
        "P1B-39's reference_with_pytz; without it they are kept as recorded",
    )
    args = parser.parse_args()
    previous = json.loads(OUT.read_text()) if OUT.exists() else {}

    def head(path: Path) -> str:
        return subprocess.run(
            ['git', '-C', str(path), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    pythons = {
        'reference': args.reference_python,
        'core': args.candidate_python,
    }
    answers: dict[str, dict[str, Any]] = {}
    environments: dict[str, Any] = {}
    for route, python in pythons.items():
        lines, code = _run(python, ['sources', route, *SOURCES, *P1B39, *P1B89])
        if code != 0:
            return code
        for line in lines:
            got = json.loads(line)
            entry = answers.setdefault(got.pop('name'), {})
            entry[f'{route}_text'] = got.pop('from_source_text')
            entry[route] = got
        lines, code = _run(python, ['environment'])
        if code != 0:
            return code
        environments[route] = json.loads(lines[-1])
    with_pytz: dict[str, Any] = {
        v['name']: {
            'answer': v['reference_with_pytz'],
            'text': v['from_source_text']['reference_with_pytz'],
        }
        for section in WITH_PYTZ
        for v in previous.get(section, {}).get('vectors', [])
    }
    pytz_environment = previous.get('p1b39', {}).get('environment')
    if args.pytz_python:
        lines, code = _run(
            args.pytz_python, ['sources', 'reference', *P1B39, *P1B89]
        )
        if code != 0:
            return code
        for line in lines:
            got = json.loads(line)
            text = got.pop('from_source_text')
            with_pytz[got.pop('name')] = {'answer': got, 'text': text}
        lines, code = _run(args.pytz_python, ['environment'])
        if code != 0:
            return code
        pytz_environment = {**json.loads(lines[-1]), 'pytz': True}
    document = {
        'format': 'sdm-host2-zone-vectors/1',
        'about': ABOUT,
        'recorded': {
            'by': 'contract/tools/host2_zones_record.py',
            'reference': head(R.REFERENCE),
            'candidate': head(ROOT),
            'environments': environments,
            'query': R.PQL,
            'indices': R.INDICES,
            'anchor_time': R.ANCHOR,
            'core_routes': R.CORE_ROUTES,
        },
        'zones': zone_facts(),
        'vectors': [
            {
                'name': name,
                **spec,
                'reference': answers[name]['reference'],
                'core_at_measurement': answers[name]['core'],
                'from_source_text': {
                    'reference': answers[name]['reference_text'],
                    'core_at_measurement': answers[name]['core_text'],
                },
            }
            for name, spec in SOURCES.items()
        ],
        **{
            section: {
                'entry': entry,
                'environment': pytz_environment,
                'vectors': [
                    {
                        'name': name,
                        **spec,
                        'reference': answers[name]['reference'],
                        'reference_with_pytz': with_pytz[name]['answer'],
                        'core_at_measurement': answers[name]['core'],
                        'from_source_text': {
                            'reference': answers[name]['reference_text'],
                            'reference_with_pytz': with_pytz[name]['text'],
                            'core_at_measurement': answers[name]['core_text'],
                        },
                    }
                    for name, spec in sources.items()
                ],
            }
            for section, (entry, sources) in WITH_PYTZ.items()
        },
    }
    OUT.write_text(json.dumps(document, indent=1) + '\n', encoding='utf-8')
    print(f'{len(SOURCES)} vectors recorded to {OUT}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
