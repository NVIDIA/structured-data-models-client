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
column over in, which fails unnamed (``OverflowError``) where the wall clock
there falls outside the years 1 to 9999, though the instant itself does not.
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
    'as_py: astimezone to zoneinfo.ZoneInfo, from the system tz database). '
    "The reference's from_source fails unnamed (OverflowError, "
    'predictive_query/time_columns.py _refused_for_values, present.iloc[0]) '
    'exactly where that wall clock leaves 0001-01-01T00:00:00 to '
    '9999-12-31T23:59:59.999999, both ends held. Held by '
    'contract/tests/test_t136_host2_zones.py.'
)

#: The value every other row holds.
ELSEWHERE = '2020-01-01 00:00:00'


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
        f"UPDATE users SET {name} = TIMESTAMPTZ '{instant}+00' WHERE user_id = 0",
    ]


NY = 'America/New_York'
TOKYO = 'Asia/Tokyo'
KIRITIMATI = 'Pacific/Kiritimati'
PAGO_PAGO = 'Pacific/Pago_Pago'
ENTITY = "the entity table's only timestamp (its time column), user 0's row"
#: The review's reproduction (Codex, final review round 2), shop dropped.
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
}


def _sources() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, (zone, instant, role, setup) in _SPECS.items():
        column = (
            'users.ts'
            if name.startswith('zone.verbatim.')
            else 'users.loaded_at'
            if name.endswith('.loaded_at')
            else 'users.f'
        )
        out[name] = {
            'column': column,
            'role': role,
            'zone': zone,
            'instant': instant + '+00',
            'setup': setup if setup is not None else first(zone, instant),
        }
    return out


SOURCES = _sources()


def wall_clock(zone: str, instant: str) -> dict[str, str]:
    r"""``instant`` (UTC, ``+00``) moved to ``zone`` as the reference moves
    it (pyarrow's ``as_py``: ``astimezone`` to ``zoneinfo.ZoneInfo``): the
    offset there and the wall clock, or the error it raises."""
    utc = dt.datetime.fromisoformat(instant.removesuffix('+00')).replace(
        tzinfo=dt.UTC
    )
    tz = zoneinfo.ZoneInfo(zone)
    try:
        local = utc.astimezone(tz)
    except OverflowError as error:
        offset = tz.utcoffset(utc.replace(tzinfo=None))
        return {
            'offset': str(offset),
            'wall_clock': f'OverflowError: {error}',
        }
    return {'offset': str(local.utcoffset()), 'wall_clock': local.isoformat()}


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
    return {
        'version': duckdb.__version__,
        'handed_over': handed,
        'session_and_global': sessions,
        'arrow_import': import_kept,
        'wall_clocks': {
            name: {
                'zone': spec['zone'],
                'instant': spec['instant'],
                **wall_clock(spec['zone'], spec['instant']),
            }
            for name, spec in SOURCES.items()
        },
    }


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
        for name in names:
            got = R.answer(SOURCES[name]['setup'], route)
            print(json.dumps({'name': name, **got}), flush=True)
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--reference-python', default=str(R.REFERENCE / '.venv/bin/python')
    )
    parser.add_argument('--candidate-python', default=sys.executable)
    args = parser.parse_args()

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
        lines, code = _run(python, ['sources', route, *SOURCES])
        if code != 0:
            return code
        for line in lines:
            got = json.loads(line)
            answers.setdefault(got.pop('name'), {})[route] = got
        lines, code = _run(python, ['environment'])
        if code != 0:
            return code
        environments[route] = json.loads(lines[-1])
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
            }
            for name, spec in SOURCES.items()
        ],
    }
    OUT.write_text(json.dumps(document, indent=1) + '\n', encoding='utf-8')
    print(f'{len(SOURCES)} vectors recorded to {OUT}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
