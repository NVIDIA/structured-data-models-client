# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.36 fix 11: P1B-81(b)'s first value read with a zone, held to the
reference.

Written by the package's independent test author, before the fix.

    python -m pytest contract/tests/test_t136_host2_zones.py -q

``contract/t136-host2-zone-vectors.json``
(``contract/tools/host2_zones_record.py``) records, through DuckDB 1.5.6 and
the public calls (``Schema.from_source``, ``materialize``,
``SDMClient.predict`` against the scripted endpoint), what the reference
does with a ``TIMESTAMPTZ`` first in a time-column candidate whose instant
lies within the years 1 to 9999.

The reference's ``from_source`` reads the first non-null value of each
time-column candidate as a Python ``datetime``
(``predictive_query/time_columns.py``, ``_refused_for_values``:
``present.iloc[0]``, pyarrow's ``TimestampScalar.as_py``): the UTC instant,
then ``astimezone`` to ``zoneinfo.ZoneInfo(<the column's Arrow zone>)``.
That zone is the one DuckDB hands the column over in to the new connection
the reference reads on (``cursor()``): the database's ``GLOBAL``
``TimeZone`` (unset, the machine's), not a ``TimeZone`` ``SET`` on the
caller's connection, nor the zone an Arrow import came with. The offset is
the system tz database's at that instant (its local mean time before a
zone's first transition: New York -4:56:02 and Kiritimati -10:29:20 in the
year 1, though Kiritimati is +14 now; Pago Pago +12:37:12, though -11 now).
Where that wall clock leaves 0001-01-01T00:00:00 to
9999-12-31T23:59:59.999999 (both ends held), the reference fails unnamed
(``OverflowError: date value out of range``). The core checked the UTC
instant alone (``rust/crates/sdm-schema/src/infer.rs``,
``past_a_datetime``), so it describes these sources.

The rule each vector is held to on the core route (E66(b): a reference
crash becomes a named refusal):

- the reference's ``from_source`` fails unnamed: the core refuses at
  ``from_source``, a ``StructuredDataError`` whose code is neither ``None``
  nor ``INTERNAL_ERROR`` (``VALUE_OUT_OF_RANGE`` expected), its details
  naming the table and the column;
- the reference describes the source: so does the core, as the reference
  does (columns, dtypes, stypes, keys, time columns and ``from_source``'s
  warnings, each whole, the core's ``[CODE]`` prefix set aside), and it
  ends at the
  stage the reference ends at, through ``predict`` or refused there by
  name.

Failing when written: the nine sources the reference fails unnamed (the
core describes each and refuses at ``materialize``). Passing when written:
the fourteen controls, and every check of the file and its facts.

Added after fix 11 (57af5ba6), from its review: ``zone.whole.*``, the same
instants filling the column. The reference refuses such a constant column
as the time column and warns, naming the value at its wall clock ("every
row holds 0001-01-01 00:00:00-04:56:02"); the core must say the same, word
for word. Before fix 11 (39ce95c5) the core crashed unnamed
(``NotImplementedError``, pandas' ``isoformat``) formatting that warning at
local midnight of the year 1 in New York and in Pago Pago. ``zone.whole.
new-york.year-1``, a wall clock before the year 1, is refused like the
first-row sources.

Added after fix 11's second commit (45c7d526), from its review (Codex):
pyarrow builds the UTC datetime first and moves it to the zone after, so
the reference also fails where the UTC instant leaves the years and the
wall clock does not (New York at 10000-01-01 00:00+00, local
9999-12-31 19:00; Tokyo at 0001-01-01 00:00+09, UTC 0000-12-31 15:00,
local 0001-01-01 00:18:59), first in the column, filling it and in the
review's own table, and at each end's last microsecond in Pago Pago,
whose offset has seconds. Their neighbours, both readings within the
years, are controls. The rule is now: either reading outside, the
reference fails unnamed and the core must refuse by name.

Each answer is the public calls on one route in this process: the default
route is the reference's code, and ``SDM_CORE_ROUTES`` forces the core.
Every vector sets the database's zone, so no answer depends on the
machine's.
"""

from __future__ import annotations

import datetime as dt
import functools
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / 'contract/t136-host2-zone-vectors.json'
REFERENCE_ENV = ROOT / 'contract/reference-env.txt'


def _module(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


Z = _module('host2_zones_record', ROOT / 'contract/tools/host2_zones_record.py')
R = Z.R
FILE: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))
ALL: list[dict[str, Any]] = FILE['vectors']
BY_NAME = {v['name']: v for v in ALL}
WALL = FILE['zones']['wall_clocks']

#: The sources whose reference from_source fails unnamed, as recorded.
REFUSED = [
    v['name']
    for v in ALL
    if v['reference']['outcome']['stage'] == 'from_source'
    and v['reference']['outcome']['result'] == 'refusal'
]
#: The sources the reference describes: the controls.
DESCRIBED = [v['name'] for v in ALL if v['reference']['described'] is not None]


def _routed(route: str, setup: list[Any]) -> dict[str, Any]:
    r"""The recorder's answer on ``route``, the environment as it was."""
    saved = os.environ.get('SDM_CORE_ROUTES')
    try:
        return dict(R.answer(setup, route))
    finally:
        if saved is None:
            os.environ.pop('SDM_CORE_ROUTES', None)
        else:
            os.environ['SDM_CORE_ROUTES'] = saved


@functools.cache
def _core(name: str) -> str:
    return json.dumps(_routed('core', BY_NAME[name]['setup']))


def core(name: str) -> dict[str, Any]:
    r"""The vector's answer on the core route, once per process."""
    loaded: dict[str, Any] = json.loads(_core(name))
    return loaded


def core_refusal(name: str) -> tuple[str, Any, Any]:
    r"""The core's ``from_source`` on the vector: the error's class, code and
    details, or ``('described', None, None)``."""
    saved = os.environ.get('SDM_CORE_ROUTES')
    os.environ['SDM_CORE_ROUTES'] = ','.join(R.CORE_ROUTES)
    try:
        from sdm_client import Schema
        from sdm_connectors import DuckDBSource

        connection = R.source(BY_NAME[name]['setup'])
        try:
            Schema.from_source(DuckDBSource(connection))
        except Exception as error:  # noqa: BLE001 - any is an answer
            code = getattr(error, 'code', None)
            return (
                type(error).__name__,
                None if code is None else str(code),
                getattr(error, 'details', None),
            )
        return ('described', None, None)
    finally:
        if saved is None:
            os.environ.pop('SDM_CORE_ROUTES', None)
        else:
            os.environ['SDM_CORE_ROUTES'] = saved


def core_text(name: str) -> list[str]:
    r"""The core's ``from_source`` warnings on the vector, each whole."""
    saved = os.environ.get('SDM_CORE_ROUTES')
    try:
        return list(Z.from_source_text(BY_NAME[name]['setup'], 'core'))
    finally:
        if saved is None:
            os.environ.pop('SDM_CORE_ROUTES', None)
        else:
            os.environ['SDM_CORE_ROUTES'] = saved


def _bodies(warnings: list[list[str]]) -> list[str]:
    r"""Each warning's first line, the core's ``[CODE]`` prefix set aside."""
    return [re.sub(r'^\[[A-Z_]+\] ', '', text) for _, text in warnings]


def _recorded_environment() -> bool:
    import duckdb
    import pandas as pd
    import pyarrow as pa

    wanted = FILE['recorded']['environments']['reference']
    return (pd.__version__, pa.__version__, duckdb.__version__) == (
        wanted['pandas'],
        wanted['pyarrow'],
        wanted['duckdb'],
    )


# --- the file -----------------------------------------------------------------


def test_the_vector_file_is_what_it_says() -> None:
    assert FILE['format'] == 'sdm-host2-zone-vectors/1'
    assert FILE['about'] == Z.ABOUT
    recorded = FILE['recorded']
    assert recorded['by'] == 'contract/tools/host2_zones_record.py'
    found = re.search(
        r'^reference: ([0-9a-f]{40})$', REFERENCE_ENV.read_text(), re.M
    )
    assert found is not None and recorded['reference'] == found[1]
    for environment in recorded['environments'].values():
        assert environment['duckdb'] == '1.5.6'
    assert recorded['core_routes'] == R.CORE_ROUTES
    # Every source the recorder runs, once, as it runs it.
    assert [v['name'] for v in ALL] == list(Z.SOURCES)
    for vector in ALL:
        spec = Z.SOURCES[vector['name']]
        for key in ('column', 'role', 'zone', 'instant', 'setup'):
            assert vector[key] == spec[key], (vector['name'], key)
        # The database's zone, set: no answer is the machine's.
        assert f"SET GLOBAL TimeZone = '{vector['zone']}'" in vector['setup']
        for route in ('reference', 'core_at_measurement'):
            outcome = vector[route]['outcome']
            assert outcome['result'] in ('success', 'refusal')
    # Not vacuous: both ends, refused and read, in zones of either sign.
    assert set(REFUSED) == {
        'zone.verbatim.los-angeles.year-1',
        'zone.session.utc.global-new-york.year-1',
        'zone.first.new-york.year-1',
        'zone.first.new-york.year-1.local-midnight-less-1us',
        'zone.first.tokyo.year-9999',
        'zone.first.tokyo.year-9999.local-midnight',
        'zone.first.kiritimati.year-9999',
        'zone.first.kiritimati.year-1',
        'zone.candidate.new-york.year-1',
        'zone.whole.new-york.year-1',
        # A UTC instant outside the years, its wall clock within them.
        'zone.verbatim.new-york.year-10000',
        'zone.first.new-york.year-10000',
        'zone.whole.new-york.year-10000',
        'zone.verbatim.tokyo.utc-year-0',
        'zone.first.tokyo.utc-year-0',
        'zone.whole.tokyo.utc-year-0',
        'zone.first.pago-pago.utc-year-0.last-us',
        'zone.first.pago-pago.utc-year-10000.local-last-second',
    }
    assert len(DESCRIBED) == 27
    # The constant columns: both ends, zones of either sign and UTC.
    whole = {n for n in DESCRIBED if n.startswith('zone.whole.')}
    assert len(whole) == 10
    for vector in ALL:
        assert set(vector['from_source_text']) == {
            'reference',
            'core_at_measurement',
        }
        if vector['name'] in whole:
            (text,) = vector['from_source_text']['reference']
            assert 'every row holds ' in text, vector['name']
    assert set(REFUSED) | set(DESCRIBED) == set(BY_NAME)


def test_duckdb_and_zoneinfo_hand_over_what_the_file_says() -> None:
    r"""The facts, re-measured: a new connection to the database reads a
    TIMESTAMPTZ in the database's zone, whatever the caller's connection
    or an Arrow import says; each zone's offset at each instant is its
    local mean time in the year 1 and its rule in the year 9999."""
    facts = Z.zone_facts()
    assert facts == FILE['zones']
    for zone, handed in facts['handed_over'].items():
        assert handed == f'timestamp[us, tz={zone}]'
    for settings, handed in facts['session_and_global'].items():
        session, database = re.findall(r"'([^']+)'", settings)
        assert handed == {
            'connection': f'timestamp[us, tz={session}]',
            'cursor': f'timestamp[us, tz={database}]',
        }
    assert facts['arrow_import']['handed_over_under_UTC'] == (
        'timestamp[us, tz=UTC]'
    )
    offsets = {
        'zone.first.new-york.year-1': '-1 day, 19:03:58',  # -4:56:02 LMT
        'zone.first.new-york.year-9999': '-1 day, 19:00:00',
        'zone.first.tokyo.year-9999': '9:00:00',
        'zone.first.kiritimati.year-1': '-1 day, 13:30:40',  # -10:29:20 LMT
        'zone.first.kiritimati.year-9999': '14:00:00',
        'zone.first.pago-pago.year-1': '12:37:12',  # LMT
        'zone.verbatim.los-angeles.year-1': '-1 day, 16:07:02',  # LMT
    }
    for name, offset in offsets.items():
        assert facts['wall_clocks'][name]['offset'] == offset, name


def test_the_reference_fails_unnamed_exactly_where_the_utc_instant_or_the_wall_clock_leaves_the_years() -> (
    None
):
    r"""The trigger, as recorded: the reference's from_source fails with a
    bare OverflowError exactly on the sources whose first value, read as
    pyarrow's as_py reads it, leaves the years 1 to 9999: its UTC datetime
    (built first) or its wall clock in the database's zone (moved to
    after), in a column not named as a load stamp (such a one is never
    read). Both ends are held (local 0001-01-01T00:00:00 and
    9999-12-31T23:59:59.999999 read, as are UTC's), and a microsecond past
    either fails, whichever of the two passes."""
    low, high = dt.datetime.min, dt.datetime.max
    for vector in ALL:
        name = vector['name']
        fact = WALL[name]
        within = fact['utc_within_years'] and fact['wall_within_years']
        # The counted readings are pyarrow's own answer.
        assert fact['as_py'].startswith('OverflowError') == (not within), name
        assert fact['utc'].startswith('OverflowError') == (
            not fact['utc_within_years']
        ), name
        if fact['utc_within_years']:
            assert fact['wall_clock'].startswith('OverflowError') == (
                not fact['wall_within_years']
            ), name
        # A load stamp is no time-column candidate, so is never read.
        candidate = not name.endswith('.loaded_at')
        assert (name in REFUSED) == (candidate and not within), (name, fact)
        if name in REFUSED:
            outcome = vector['reference']['outcome']
            assert (outcome['error'], outcome['code'], outcome['message']) == (
                'OverflowError',
                None,
                'date value out of range',
            ), name
            assert vector['reference']['described'] is None, name
        elif candidate:
            local = dt.datetime.fromisoformat(fact['wall_clock'])
            assert low <= local.replace(tzinfo=None) <= high, name
            assert fact['as_py'] == fact['wall_clock'], name
        else:
            assert not within, name
    # Not vacuous: each reading alone takes a source out, at each end.
    alone = {
        (fact['utc_within_years'], fact['wall_within_years'], end)
        for name, fact in WALL.items()
        if name in REFUSED
        for end in ('low' if int(fact['microseconds']) < 0 else 'high',)
    }
    assert alone >= {
        (True, False, 'low'),
        (True, False, 'high'),
        (False, True, 'low'),
        (False, True, 'high'),
    }
    edges = {
        'zone.first.new-york.year-1.local-midnight': '0001-01-01T00:00:00',
        'zone.first.kiritimati.year-1.local-midnight': '0001-01-01T00:00:00',
        'zone.first.tokyo.year-9999.local-last-us': (
            '9999-12-31T23:59:59.999999'
        ),
        'zone.first.kiritimati.year-9999.local-last-us': (
            '9999-12-31T23:59:59.999999'
        ),
    }
    for name, wall in edges.items():
        assert WALL[name]['wall_clock'].startswith(wall), name
        assert name in DESCRIBED, name


# --- the reference, re-measured where it was recorded -----------------------


@pytest.mark.parametrize('name', list(BY_NAME))
def test_the_reference_answers_as_recorded(name: str) -> None:
    r"""In ref-pandas3, the cell CI runs contract/tests in: the default
    route, the reference's code, answers as the reference checkout did."""
    assert _recorded_environment(), (
        'the reference answers were recorded in ref-pandas3 (pandas 3.0.6, '
        'pyarrow 25.0.1, DuckDB 1.5.6), where CI runs contract/tests'
    )
    got = _routed('reference', BY_NAME[name]['setup'])
    assert got == BY_NAME[name]['reference']
    saved = os.environ.get('SDM_CORE_ROUTES')
    try:
        text = Z.from_source_text(BY_NAME[name]['setup'], 'reference')
    finally:
        if saved is not None:
            os.environ['SDM_CORE_ROUTES'] = saved
    assert text == BY_NAME[name]['from_source_text']['reference']


# --- the core ------------------------------------------------------------------


@pytest.mark.parametrize('name', REFUSED)
def test_the_core_refuses_by_name_where_the_reference_fails_unnamed(
    name: str,
) -> None:
    r"""E66(b): the reference's from_source fails unnamed; the core's
    refuses there, by a code, naming the table and the column."""
    got = core(name)['outcome']
    assert (got['result'], got['stage'], got.get('error')) == (
        'refusal',
        'from_source',
        'StructuredDataError',
    ), (
        f'{name}: the reference fails from_source unnamed (OverflowError: the '
        f"first value's wall clock in {BY_NAME[name]['zone']} is outside the "
        f'years 1 to 9999); the core must refuse there by name, not: {got}'
    )
    assert got['code'] not in (None, 'INTERNAL_ERROR'), got
    error, code, details = core_refusal(name)
    assert (error, code) == ('StructuredDataError', got['code'])
    table, column = BY_NAME[name]['column'].split('.')
    assert isinstance(details, dict), details
    assert (details.get('table'), details.get('column')) == (table, column), (
        f'{name}: the refusal must name {table}.{column}: {details}'
    )


@pytest.mark.parametrize('name', DESCRIBED)
def test_the_core_describes_each_control_as_the_reference_does(
    name: str,
) -> None:
    r"""A wall clock within the years, at either end, in any zone: the
    reference describes the source, and so does the core, as the reference
    does, and it ends where the reference ends."""
    reference = BY_NAME[name]['reference']
    got = core(name)
    assert got['described'] is not None, (
        f'{name}: the reference describes this source and the core refuses '
        f'it at from_source: {got["outcome"]}'
    )
    assert got['described'] == reference['described']
    assert _bodies(got['warnings']['from_source']) == _bodies(
        reference['warnings']['from_source']
    )
    # Each warning whole: the constant-column one names the value at its
    # wall clock, in the database's zone, as the reference shows it.
    assert core_text(name) == BY_NAME[name]['from_source_text']['reference']
    wanted = reference['outcome']
    if wanted['result'] == 'success':
        assert got['outcome'] == wanted
        return
    assert (
        got['outcome']['result'],
        got['outcome']['stage'],
        got['outcome'].get('error'),
    ) == ('refusal', wanted['stage'], 'StructuredDataError'), got['outcome']
    assert got['outcome']['code'] not in (None, 'INTERNAL_ERROR')
