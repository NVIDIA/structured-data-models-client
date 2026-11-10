# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DuckDB timestamp and buffer boundary contracts through public calls.

Timestamp units, infinities and overflow keep their schema facts and produce
structured refusals at the appropriate stage.
Expected types and stages follow explicit source-value rules.
The optional SDM_GIANT_TESTS=1 cases allocate roughly 7 GB and may take minutes;
ordinary runs exercise scaled-capacity boundaries in the Rust widening suite.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[2]
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
BY_NAME = R.SOURCES
# These names explicitly exclude the column from time inference, or keep
# the first timestamp inside Python's calendar range.
FIRST_CONTROLS = {
    'first.users.TIMESTAMP_S.year-9999',
    'first.users.TIMESTAMP_NS.infinity',
    'first.users.TIMESTAMP_S.infinity.loaded_at',
    'first.users.TIMESTAMP_S.infinity.birth_date',
}
READ_FIRST = {
    name
    for name in BY_NAME
    if (name.startswith('first.') and name not in FIRST_CONTROLS)
    or name.startswith('generated.')
}
DESCRIBED = set(BY_NAME) - READ_FIRST
TIME_BOUNDS = {name for name in BY_NAME if name.startswith('time.far.')}


@functools.cache
def _core(name: str) -> str:
    return json.dumps(R.answer(BY_NAME[name]['setup']))


def core(name: str) -> dict[str, Any]:
    r"""The vector's answer on the core route, once per process."""
    loaded: dict[str, Any] = json.loads(_core(name))
    return loaded


def _assert_description(name: str, got: dict[str, Any]) -> None:
    described = got['described']
    assert described is not None, got['outcome']
    assert set(described) == {'orders', 'users'}
    assert described['orders']['primary_key'] == 'order_id'
    assert described['orders']['time_column'] == 'ts'
    assert described['users']['primary_key'] == 'user_id'
    table, column = BY_NAME[name]['column'].split('.')
    columns = {c[0]: c[1:] for c in described[table]['columns']}
    assert columns[column] == ['timestamp[us]', 'timestamp']
    assert described['users']['time_column'] == (
        'f' if table == 'users' and column == 'f' else None
    )
    inference_warnings = [
        text
        for _, text in got['warnings']['from_source']
        if text.startswith('[TIME_COLUMNS_GUESSED]')
    ]
    guessed = name.startswith('feature.') or column in (
        'loaded_at',
        'birth_date',
    )
    assert len(inference_warnings) == int(guessed)


def test_duckdb_timestamp_storage_and_conversion_boundaries() -> None:
    """Check physical units and overflow directly, without recorded errors."""
    facts = R.duckdb_facts()
    biggest = 2**63 - 1
    units = {
        'TIMESTAMP_S': 's',
        'TIMESTAMP_MS': 'ms',
        'TIMESTAMP': 'us',
        'TIMESTAMP_NS': 'ns',
    }
    for kind, unit in units.items():
        for literal, value in (('infinity', biggest), ('-infinity', -biggest)):
            assert facts['infinity'][f'{kind} {literal}'] == [
                f'timestamp[{unit}]',
                value,
            ]
    finite = facts['finite_past_int64_microseconds']
    for unit, value in (
        ('s', 10**16),
        ('s', -(10**16)),
        ('ms', 10**17),
        ('s', 9_223_372_036_855),
        ('s', 2**63 - 2),
    ):
        fact = finite[f'Arrow timestamp[{unit}] {value} imported']
        assert (
            fact['stored_as']
            == {'s': 'TIMESTAMP_S', 'ms': 'TIMESTAMP_MS'}[unit]
        )
        assert fact['handed_over'] == [f'timestamp[{unit}]', value]
        expected_error = (
            'InternalException'
            if int(duckdb.__version__.split('.')[0]) >= 2
            else 'ConversionException'
        )
        assert fact['cast_to_text'].startswith(expected_error)
    sql_casts = [
        value for key, value in finite.items() if key.startswith('SELECT ')
    ]
    assert len(sql_casts) == 3
    assert all(error.startswith('ConversionException') for error in sql_casts)
    assert facts['duration_and_date64'] == {
        'd (duration[s]) stored then read': 'month_day_nano_interval',
        'm (duration[ms]) stored then read': 'month_day_nano_interval',
        'e (date64) stored then read': 'date32[day]',
    }


# --- the core ------------------------------------------------------------------


@pytest.mark.parametrize('name', sorted(DESCRIBED))
def test_timestamp_schema_facts(name: str) -> None:
    _assert_description(name, core(name))


@pytest.mark.parametrize('name', list(BY_NAME))
def test_timestamp_failure_stage_and_code(name: str) -> None:
    got = core(name)['outcome']
    if name in READ_FIRST:
        assert (got['result'], got['stage'], got.get('error')) == (
            'refusal',
            'from_source',
            'StructuredDataError',
        )
        assert got['code'] not in (None, 'INTERNAL_ERROR')
        return
    # A nanosecond infinity is a finite instant when widened to microseconds.
    # The upper bound in a time column rounds outside the supported ns range.
    valid = (
        name.endswith('.finite')
        or ('TIMESTAMP_NS' in name and not name.startswith('time.'))
        or name == 'time.TIMESTAMP_NS.-infinity'
    )
    if name == 'time.TIMESTAMP_NS.max':
        valid = int(duckdb.__version__.split('.')[0]) < 2
        if not valid:
            with duckdb.connect() as connection:
                [(us,)] = connection.execute(
                    "SELECT epoch_us(CAST('2262-04-11 23:47:16.854775806'::TIMESTAMP_NS AS TIMESTAMP))"
                ).fetchall()
            assert us * 1000 > 2**63 - 1
    if valid:
        assert got == {'result': 'success', 'stage': 'predict'}
        return
    assert (got['result'], got['stage'], got.get('error')) == (
        'refusal',
        'materialize',
        'StructuredDataError',
    )
    if name in TIME_BOUNDS:
        assert got['code'] == 'PROVIDER_ERROR'
        assert got['request'] == 'time_bounds'
    else:
        assert got['code'] == PAST_NANOSECONDS


# --- one value past int32 offsets ----------------------------------------------


# Explicit opt-in: each real oversized-value scenario allocates roughly 7 GB.
if BIG:

    @pytest.mark.parametrize('kind', ['varchar', 'blob'])
    @pytest.mark.parametrize('mode', ['small', 'large'])
    def test_big_a_value_past_int32_offsets_on_the_core(
        kind: str, mode: str
    ) -> None:
        r"""Large values can be described before materialization checks their representation.

        Binary is omitted and prediction proceeds; retained large text is
        described but materialization refuses its physical representation.
        """
        setup = R.big_setup(kind, mode)
        if mode == 'small':
            got = R.answer(setup)
            assert got['outcome']['stage'] == 'from_source'
            assert got['outcome']['code'] is not None
            return
        endpoint = os.environ.get('SDM_GIANT_ENDPOINT')
        if endpoint is not None and kind == 'blob':
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

            with R.source(setup) as connection:
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
        got = R.answer(setup)
        assert got['described'] is not None, got['outcome']
        columns = got['described']['users']['columns']
        if kind == 'blob':
            assert all(column[0] != 'f' for column in columns)
            assert got['outcome'] == {'result': 'success', 'stage': 'predict'}
        else:
            assert ['f', 'string', 'categorical'] in columns
            assert got['outcome']['stage'] == 'materialize'
            assert got['outcome']['code'] == 'UNSUPPORTED_TYPE'
            assert 'f' in got['outcome']['message']
