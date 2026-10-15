# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.36 fix group 3: register HOST-2's values, held to the reference.

Written by the package's independent test author, before the fix.

    python -m pytest contract/tests/test_t136_host2_values.py -q
    SDM_T136_BIG=1 python -m pytest contract/tests/test_t136_host2_values.py -q -k big

``contract/t136-host2-vectors.json`` (``contract/tools/host2_values_record.py``)
records, through DuckDB 1.5.6 and the public calls (``Schema.from_source``,
``materialize``, ``SDMClient.predict`` against the scripted endpoint), what
the reference does with the values HOST-2 says the core refuses:

- a ``TIMESTAMP_S`` or ``TIMESTAMP_MS`` holding ``'infinity'`` or
  ``'-infinity'`` (DuckDB hands either over as int64 max, or its negation,
  in the column's own unit, so past an int64 count of microseconds), as the
  time column, as a feature and as the entity table's only timestamp;
- a finite ``TIMESTAMP_S`` or ``TIMESTAMP_MS`` past int64 microseconds, which
  DuckDB's casts refuse but its Arrow import keeps (the same three roles);
- for comparison, ``TIMESTAMP`` (microseconds) and ``TIMESTAMP_NS``
  infinities and each type's largest finite values;
- (``big``, recorded with ``--big``) one ``VARCHAR`` or ``BLOB`` value of
  2**31 + 16 bytes, with and without ``arrow_large_buffer_size``.

The reference describes every value past int64 microseconds at
``from_source`` and fails only at ``materialize``, where each is a time past
what a nanosecond count holds. Register DT-14 (decision E88, the owner's,
2026-10-13: "times past the nanosecond range get VALUE_OUT_OF_RANGE") gives
that refusal on the core its code, and the core already gives it for a
microsecond infinity at the same stage. So the rule each vector is held to
on the core route:

- the reference succeeds through ``predict``: so does the core, describing
  the source as the reference does;
- the reference describes the source and refuses at ``materialize``: the
  core describes it as the reference does (columns, dtypes, stypes, keys,
  time columns, and ``from_source``'s warnings, the core's ``[CODE]`` prefix
  set aside) and refuses at ``materialize`` with ``VALUE_OUT_OF_RANGE``.

Failing when written: every ``TIMESTAMP_S`` and ``TIMESTAMP_MS`` vector past
int64 microseconds (the core refuses it ``VALUE_OUT_OF_RANGE`` at
``from_source``), and ``time.TIMESTAMP_NS.-infinity``, which the reference
takes through ``predict`` (``-infinity`` in nanoseconds is
1677-09-21 00:12:43.145224193, pandas' smallest timestamp) and the core
refuses ``VALUE_OUT_OF_RANGE`` at ``materialize``.

The ``big`` rows take about 7 GB and up to six minutes each, so the test
making their public calls is defined only with ``SDM_T136_BIG=1``; by
default only the recorded rows are held. Their rule: where the reference
fails at ``from_source`` with no code (DuckDB refuses to write the value without large buffers), the core
fails there too, by name (E66(b)); where the reference describes the source,
the core describes it as the reference does, and where the reference goes
on through ``predict`` (the ``BLOB``, which it leaves out), so does the
core. The ``VARCHAR`` with large buffers fails the reference's
``materialize`` with DuckDB's out-of-memory error on a 24 GB machine, an
answer of the machine, so only its ``from_source`` is held.

Added after fix 3 (97aefc9d), from its review (register P1B-81):

- (a) a finite value past microseconds in the time column makes DuckDB
  fail the time bounds' ``CAST(ts AS TIMESTAMP)``, the statement both
  routes send. The reference gives ``INTERNAL_ERROR`` (DuckDB's
  ``ConversionException``); the core cannot tell that failure from another
  without the driver's message, which the contract keeps out, so it gives
  ``PROVIDER_ERROR``, its request ``time_bounds`` (DT-12's shape, E66(b)):
  ``time.far.*`` are held to that, never ``INTERNAL_ERROR`` or a panic.
- (b) the reference's ``from_source`` reads the first non-null value of each
  time-column candidate (a timestamp column not named as a load stamp or a
  birth date) as a Python datetime (``predictive_query/time_columns.py``,
  ``_refused_for_values``: ``present.iloc[0]``, pyarrow's ``as_py``), which
  fails unnamed (``OverflowError``) past the years 1 to 9999. The
  ``first.*`` vectors put the values above in that first row (and a
  microsecond infinity and the year 10000, which fail the same way); the
  ``generated.*`` ones are the three sources of fix 3's differential where
  it happened. Where the reference fails so, the core refuses at
  ``from_source`` too, by name (E66(b)): a ``StructuredDataError`` with a
  code other than ``INTERNAL_ERROR``. Their controls (a nanosecond infinity,
  the year 9999, a column named ``loaded_at`` or ``birth_date``) and every
  vector above, whose value is not in a first row, the reference describes,
  and the core must too.

Each answer is the public calls on one route in this process: the default
route is the reference's code, and ``SDM_CORE_ROUTES`` forces the core
(the recorder's ``CORE_ROUTES``). The reference's answers are re-measured
here only in the environment they were recorded in.
"""

from __future__ import annotations

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
VECTORS = ROOT / 'contract/t136-host2-vectors.json'
REFERENCE_ENV = ROOT / 'contract/reference-env.txt'
#: The code DT-14 gives a time past the nanosecond range on the core.
PAST_NANOSECONDS = 'VALUE_OUT_OF_RANGE'
BIG = os.environ.get('SDM_T136_BIG') == '1'


def _module(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = _module(
    'host2_values_record', ROOT / 'contract/tools/host2_values_record.py'
)
FILE: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))
ALL: list[dict[str, Any]] = FILE['vectors']
BIG_ROWS: list[dict[str, Any]] = FILE['big']
BY_NAME = {v['name']: v for v in ALL}


def _kind(vector: dict[str, Any]) -> str:
    r"""The DuckDB type of the vector's column: its name's, or its
    import's."""
    part = vector['name'].split('.')[1]
    if part == 'far':
        return {'s': 'TIMESTAMP_S', 'ms': 'TIMESTAMP_MS'}[
            vector['name'].split('.')[2]
        ]
    return part


#: The vectors past int64 microseconds the core refused when written, none
#: in a row the reference reads first.
PAST_MICROSECONDS = [
    v['name']
    for v in ALL
    if v['name'].startswith(('time.', 'feature.', 'users.'))
    and _kind(v) in ('TIMESTAMP_S', 'TIMESTAMP_MS')
    and (v['name'].endswith('infinity') or '.far.' in v['name'])
]
#: P1B-81(a): finite values past microseconds in the time column, whose
#: time bounds DuckDB fails to cast.
TIME_BOUNDS = [v['name'] for v in ALL if v['name'].startswith('time.far.')]
#: P1B-81(b): the sources whose reference from_source fails unnamed.
READ_FIRST = [
    v['name']
    for v in ALL
    if v['reference']['outcome']['stage'] == 'from_source'
    and v['reference']['outcome']['result'] == 'refusal'
]
#: The sources the reference describes.
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


def _bodies(warnings: list[list[str]]) -> list[str]:
    r"""Each warning's first line, the core's ``[CODE]`` prefix set aside."""
    return [re.sub(r'^\[[A-Z_]+\] ', '', text) for _, text in warnings]


def _same_description(got: dict[str, Any], reference: dict[str, Any]) -> None:
    assert got['described'] == reference['described'], (
        'the core describes the source otherwise than the reference: '
        f'{got["described"]} against {reference["described"]}'
    )
    assert _bodies(got['warnings']['from_source']) == _bodies(
        reference['warnings']['from_source']
    )


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
    assert FILE['format'] == 'sdm-host2-vectors/1'
    recorded = FILE['recorded']
    assert recorded['by'] == 'contract/tools/host2_values_record.py'
    found = re.search(
        r'^reference: ([0-9a-f]{40})$', REFERENCE_ENV.read_text(), re.M
    )
    assert found is not None and recorded['reference'] == found[1]
    for environment in recorded['environments'].values():
        assert environment['duckdb'] == '1.5.6'
    assert recorded['core_routes'] == R.CORE_ROUTES
    # Every source the recorder runs, once, with the setup it runs.
    assert [v['name'] for v in ALL] == list(R.SOURCES)
    for vector in ALL:
        assert vector['setup'] == R.SOURCES[vector['name']]['setup']
        for route in ('reference', 'core_at_measurement'):
            outcome = vector[route]['outcome']
            assert outcome['result'] in ('success', 'refusal')
            assert (outcome['result'] == 'success') == (
                outcome['stage'] == 'predict' and 'code' not in outcome
            )
    # Not vacuous: both infinities and finite values past int64
    # microseconds, at s and ms, in each role.
    assert len(PAST_MICROSECONDS) == 22
    for role in ('time', 'feature', 'users'):
        for kind in ('TIMESTAMP_S', 'TIMESTAMP_MS'):
            for value in ('infinity', '-infinity'):
                assert f'{role}.{kind}.{value}' in PAST_MICROSECONDS
        for label in ('s.past', 's.before', 'ms.past'):
            assert f'{role}.far.{label}' in PAST_MICROSECONDS
    assert len(TIME_BOUNDS) == 4
    assert {n for n in READ_FIRST if n.startswith('generated.')} == {
        'generated.g24.users.TIMESTAMP_S',
        'generated.g30.feature.TIMESTAMP_MS',
        'generated.g32.feature.TIMESTAMP_S',
    }
    assert len(READ_FIRST) == 17


def test_the_reference_fails_unnamed_only_on_a_first_value_past_the_years() -> (
    None
):
    r"""P1B-81(b)'s trigger, as recorded: the reference's from_source fails
    with a bare OverflowError exactly on the sources holding, first in some
    time-column candidate, a value past the years 1 to 9999, and on no
    control; every other vector it describes."""
    for vector in ALL:
        name = vector['name']
        outcome = vector['reference']['outcome']
        if name in READ_FIRST:
            assert name.startswith(('first.', 'generated.')), name
            assert (outcome['error'], outcome['code']) == (
                'OverflowError',
                None,
            ), name
            assert vector['reference']['described'] is None, name
        else:
            assert name in DESCRIBED, name
    controls = [
        'first.users.TIMESTAMP_S.year-9999',
        'first.users.TIMESTAMP_NS.infinity',
        'first.users.TIMESTAMP_S.infinity.loaded_at',
        'first.users.TIMESTAMP_S.infinity.birth_date',
    ]
    assert set(controls) <= set(DESCRIBED)
    # Every first-row value past the years fails it, at s, ms and us, in
    # each role, after leading nulls too.
    assert {
        n for n in BY_NAME if n.startswith('first.') and n not in controls
    } <= set(READ_FIRST)
    # The 22 values past microseconds in a later row it describes.
    assert set(PAST_MICROSECONDS) <= set(DESCRIBED)


def test_each_value_past_int64_microseconds_is_one_the_reference_describes() -> (
    None
):
    r"""The reference's answer to each: described at from_source, refused
    at materialize as a time past the nanosecond range."""
    for name in PAST_MICROSECONDS:
        reference = BY_NAME[name]['reference']
        assert reference['described'] is not None, name
        outcome = reference['outcome']
        assert (outcome['result'], outcome['stage']) == (
            'refusal',
            'materialize',
        ), name
        assert outcome['code'] in ('INVALID_REQUEST', 'INTERNAL_ERROR'), name
        column = BY_NAME[name]['column']
        table, column_name = column.split('.')
        described = {
            c[0]: c[1:] for c in reference['described'][table]['columns']
        }
        assert described[column_name] == ['timestamp[us]', 'timestamp'], name


def test_duckdb_hands_over_what_the_file_says() -> None:
    r"""DuckDB's own facts, re-measured: an infinity at s or ms is int64
    max in its unit, so past int64 microseconds; a finite value past them is
    refused by DuckDB's casts but kept by its Arrow import; a duration or a
    date64 never comes out of it."""
    facts = R.duckdb_facts()
    assert facts == FILE['duckdb']
    biggest = 2**63 - 1
    for kind in ('TIMESTAMP_S', 'TIMESTAMP_MS'):
        assert facts['infinity'][f'{kind} infinity'][1] == biggest
        assert facts['infinity'][f'{kind} -infinity'][1] == -biggest
    for key, fact in facts['finite_past_int64_microseconds'].items():
        if key.startswith('SELECT '):
            assert fact.startswith('ConversionException'), key
        else:
            assert fact['stored_as'] in ('TIMESTAMP_S', 'TIMESTAMP_MS'), key
            assert fact['cast_to_text'].startswith('ConversionException')
    assert set(facts['duration_and_date64'].values()) == {
        'month_day_nano_interval',
        'date32[day]',
    }


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


# --- the core ------------------------------------------------------------------


@pytest.mark.parametrize('name', DESCRIBED)
def test_the_core_describes_each_source_as_the_reference_does(
    name: str,
) -> None:
    reference = BY_NAME[name]['reference']
    got = core(name)
    assert got['described'] is not None, (
        f'{name}: the reference describes this source and the core refuses '
        f'it at from_source: {got["outcome"]}'
    )
    _same_description(got, reference)


@pytest.mark.parametrize('name', list(BY_NAME))
def test_the_core_ends_where_the_reference_does(name: str) -> None:
    r"""Through predict where the reference gets there; refused by name at
    from_source where the reference's from_source fails unnamed (P1B-81(b),
    E66(b)); refused PROVIDER_ERROR of the time bounds where DuckDB fails
    their cast (P1B-81(a)); else refused at materialize, a time past the
    nanosecond range (DT-14)."""
    reference = BY_NAME[name]['reference']['outcome']
    got = core(name)['outcome']
    if reference['result'] == 'success':
        assert got == reference, (
            f'{name}: the reference takes this source through predict; the '
            f'core: {got}'
        )
        return
    if name in READ_FIRST:
        assert reference['code'] is None
        assert (got['result'], got['stage'], got.get('error')) == (
            'refusal',
            'from_source',
            'StructuredDataError',
        ), (
            f'{name}: the reference fails from_source unnamed '
            f'({reference["error"]}); the core must refuse there by name, '
            f'not: {got}'
        )
        assert got['code'] not in (None, 'INTERNAL_ERROR'), got
        return
    assert reference['stage'] == 'materialize'
    if name in TIME_BOUNDS:
        assert reference['code'] == 'INTERNAL_ERROR'
        assert 'ConversionException' in reference['message']
        assert (
            got['result'],
            got['stage'],
            got.get('error'),
            got.get('code'),
            got.get('request'),
        ) == (
            'refusal',
            'materialize',
            'StructuredDataError',
            'PROVIDER_ERROR',
            'time_bounds',
        ), (
            f"{name}: DuckDB fails the time bounds' cast, the reference "
            f'with INTERNAL_ERROR; the core must give PROVIDER_ERROR of the '
            f'time_bounds request, not: {got}'
        )
        return
    assert (got['result'], got['stage'], got.get('error'), got.get('code')) == (
        'refusal',
        'materialize',
        'StructuredDataError',
        PAST_NANOSECONDS,
    ), (
        f'{name}: the reference describes this source and refuses it at '
        f'materialize ({reference["code"]}); the core: {got}'
    )


def test_a_microsecond_infinity_is_what_the_rule_follows() -> None:
    r"""Control, passing when written: the core's answer for a TIMESTAMP
    (microseconds) infinity in each role is the rule's, so the rule asks of
    seconds and milliseconds what the core already does there."""
    for role in ('time', 'feature', 'users'):
        for value in ('infinity', '-infinity'):
            got = core(f'{role}.TIMESTAMP.{value}')
            _same_description(
                got, BY_NAME[f'{role}.TIMESTAMP.{value}']['reference']
            )
            assert (got['outcome']['stage'], got['outcome']['code']) == (
                'materialize',
                PAST_NANOSECONDS,
            )


# --- one value past int32 offsets ----------------------------------------------


def test_the_big_rows_are_recorded() -> None:
    r"""The measurement, held without running it: four rows, DuckDB itself
    refusing to write the value without large buffers, which the reference
    fails with no code; with them the reference describes the source, and
    the BLOB, left out, goes on through predict."""
    names = {row['name']: row for row in BIG_ROWS}
    assert set(names) == {
        f'big.{kind}.{mode}'
        for kind in ('varchar', 'blob')
        for mode in ('small', 'large')
    }
    for name, row in names.items():
        assert row['bytes'] == 2**31 + 16
        assert row['setup'] == R.big_setup(*name.split('.')[1:])
        reference = row['reference']
        if name.endswith('.small'):
            assert reference['outcome']['stage'] == 'from_source'
            assert reference['outcome']['code'] is None
            # DuckDB's own refusal to write the value with int32 offsets.
            assert (
                'maximum total string size for regular string buffers'
                in reference['outcome']['message']
            )
        else:
            assert reference['described'] is not None
    assert names['big.blob.large']['reference']['outcome'] == {
        'result': 'success',
        'stage': 'predict',
    }
    users = names['big.varchar.large']['reference']['described']['users']
    assert ['f', 'string', 'categorical'] in users['columns']


# About 7 GB and minutes each: defined only with SDM_T136_BIG=1, so the
# suite holds the recorded rows alone by default.
if BIG:

    @pytest.mark.parametrize('name', [row['name'] for row in BIG_ROWS])
    def test_big_a_value_past_int32_offsets_on_the_core(name: str) -> None:
        row = next(r for r in BIG_ROWS if r['name'] == name)
        reference = row['reference']
        got = _routed('core', row['setup'])
        if reference['described'] is None:
            # E66(b): the reference fails with no code; the core names it.
            assert got['outcome']['stage'] == 'from_source'
            assert got['outcome']['code'] is not None
            return
        assert got['described'] is not None, (
            f'{name}: the reference describes this source and the core refuses '
            f'it at from_source: {got["outcome"]}'
        )
        _same_description(got, reference)
        if reference['outcome']['result'] == 'success':
            assert got['outcome'] == reference['outcome']
