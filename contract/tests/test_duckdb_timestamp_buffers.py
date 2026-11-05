# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DuckDB timestamp and buffer boundary contracts through public calls.

Timestamp units, infinities and overflow keep their schema facts and produce
structured refusals at the appropriate stage.
Expected observations are retained fixtures, not regenerated from current code.
The optional SDM_GIANT_TESTS=1 cases allocate roughly 7 GB and may take minutes;
ordinary runs check their fixture inputs without allocating those large values.
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / 'contract/host2-vectors.json'
#: The code DT-14 gives a time past the nanosecond range on the core.
PAST_NANOSECONDS = 'VALUE_OUT_OF_RANGE'
BIG = os.environ.get('SDM_GIANT_TESTS') == '1'


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
    'duckdb_value_scenarios',
    ROOT / 'contract/support/duckdb_value_scenarios.py',
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


@functools.cache
def _core(name: str) -> str:
    return json.dumps(R.answer(BY_NAME[name]['setup']))


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


# --- the file -----------------------------------------------------------------


def test_the_vector_file_is_what_it_says() -> None:
    assert FILE['format'] == 'sdm-host2-vectors/1'
    assert [v['name'] for v in ALL] == list(R.SOURCES)
    for vector in ALL:
        assert vector['setup'] == R.SOURCES[vector['name']]['setup']
        outcome = vector['reference']['outcome']
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
    expected = copy.deepcopy(FILE['duckdb'])
    if int(duckdb.__version__.split('.')[0]) >= 2:
        expected['version'] = duckdb.__version__
        # DuckDB 2 reports the same rejected widening as an upstream internal
        # cast error. Preserve the exact numeric/type facts, independently
        # requiring each previously rejected conversion to remain rejected.
        for key, fact in expected['finite_past_int64_microseconds'].items():
            if key.startswith('Arrow '):
                error = facts['finite_past_int64_microseconds'][key][
                    'cast_to_text'
                ]
                assert error.startswith('InternalException: INTERNAL Error:')
                assert (
                    'Could not convert Timestamp to higher precision.' in error
                )
                fact['cast_to_text'] = error
    assert facts == expected
    biggest = 2**63 - 1
    for kind in ('TIMESTAMP_S', 'TIMESTAMP_MS'):
        assert facts['infinity'][f'{kind} infinity'][1] == biggest
        assert facts['infinity'][f'{kind} -infinity'][1] == -biggest
    for key, fact in facts['finite_past_int64_microseconds'].items():
        if key.startswith('SELECT '):
            assert fact.startswith('ConversionException'), key
        else:
            assert fact['stored_as'] in ('TIMESTAMP_S', 'TIMESTAMP_MS'), key
            expected_error = (
                'InternalException'
                if int(duckdb.__version__.split('.')[0]) >= 2
                else 'ConversionException'
            )
            assert fact['cast_to_text'].startswith(expected_error)
    assert set(facts['duration_and_date64'].values()) == {
        'month_day_nano_interval',
        'date32[day]',
    }


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
    if (
        name == 'time.TIMESTAMP_NS.max'
        and int(duckdb.__version__.split('.')[0]) >= 2
    ):
        # The warehouse contract uses a microsecond wall clock. DuckDB 2 rounds
        # this last finite ns value up, past the supported int64-ns range.
        # Current main also refuses it, with INTERNAL_ERROR; keep the core's
        # precise range error instead of reproducing that generic failure.
        with duckdb.connect() as connection:
            [(microseconds,)] = connection.execute(
                "SELECT epoch_us(CAST('2262-04-11 23:47:16.854775806'::TIMESTAMP_NS AS TIMESTAMP))"
            ).fetchall()
        assert microseconds * 1000 > 2**63 - 1
        assert (got['result'], got['stage'], got['code']) == (
            'refusal',
            'materialize',
            PAST_NANOSECONDS,
        )
        return
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


# About 7 GB and minutes each: defined only with SDM_GIANT_TESTS=1, so the
# suite holds the recorded rows alone by default.
if BIG:

    @pytest.mark.parametrize('name', [row['name'] for row in BIG_ROWS])
    def test_big_a_value_past_int32_offsets_on_the_core(name: str) -> None:
        r"""Large values can be described before materialization checks their representation.

        Binary is omitted and prediction proceeds; retained large text is
        described but materialization refuses its physical representation.
        The historical vectors remain evidence of the original gap.
        """
        row = next(r for r in BIG_ROWS if r['name'] == name)
        reference = row['reference']
        if reference['described'] is None:
            got = R.answer(row['setup'])
            assert got['outcome']['stage'] == 'from_source'
            assert got['outcome']['code'] is not None
            return
        assert name.endswith('.large'), name
        endpoint = os.environ.get('SDM_GIANT_ENDPOINT')
        if endpoint is not None and '.blob.' in name:
            assert endpoint.strip(), (
                'SDM_GIANT_ENDPOINT must name a serving endpoint'
            )
            import math

            import pandas as pd
            from sdm_client import (
                KumoRelational,
                PredictiveQuery,
                Schema,
                SDMClient,
            )
            from sdm_connectors import DuckDBSource

            # The recorded frozen fixture already succeeded through stub prediction.
            # This opt-in closes the actual-model gap without rerunning the reference.
            with R.source(row['setup']) as connection:
                schema = Schema.from_source(DuckDBSource(connection))
                assert 'f' not in schema['users'].columns
                request = PredictiveQuery(R.PQL, schema).materialize(
                    indices=R.INDICES, anchor_time=pd.Timestamp(R.ANCHOR)
                )
                with SDMClient.for_databricks_serving(endpoint) as client:
                    result = client.predict(
                        request, model=KumoRelational()
                    ).to_pandas()
                assert len(result) == len(R.INDICES)
                assert all(
                    math.isfinite(float(value))
                    for value in result['prediction']
                )
            return
        got = R.answer(row['setup'])
        assert got['described'] is not None, got['outcome']
        columns = got['described']['users']['columns']
        if '.blob.' in name:
            assert all(column[0] != 'f' for column in columns)
            assert got['outcome'] == {'result': 'success', 'stage': 'predict'}
        else:
            assert ['f', 'string', 'categorical'] in columns
            assert got['outcome']['stage'] == 'materialize'
            assert got['outcome']['code'] == 'UNSUPPORTED_TYPE'
            assert 'f' in got['outcome']['message']
