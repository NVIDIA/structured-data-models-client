# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interval observations preserve exact fields from shared vectors.

The Rust testkit consumes the same vectors; no historical comparator is needed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import duckdb
import pyarrow as pa
import pytest

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / 'contract/arrow-interval-vectors.json'
OLD_VECTORS = ROOT / 'contract/result-vectors.json'


def _module(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_python = _module('run_python', ROOT / 'contract/runners/run_python.py')


def _vectors() -> list[dict[str, Any]]:
    loaded = json.loads(VECTORS.read_text(encoding='utf-8'))
    assert loaded['format'] == 'sdm-result-vectors/1'
    vectors: list[dict[str, Any]] = loaded['vectors']
    return vectors


def _text(row: Any) -> str:
    return json.dumps(row, ensure_ascii=True, separators=(',', ':'))


def _digest(texts: list[str]) -> str:
    return hashlib.sha256(('[' + ','.join(texts) + ']').encode()).hexdigest()


# --- the vectors built as Arrow ------------------------------------------------------


def _limits() -> Any:
    return pa.table(
        {
            'v': pa.array(
                [
                    pa.MonthDayNano(
                        [2147483647, -2147483648, 9223372036854775807]
                    ),
                    pa.MonthDayNano([-1, 0, -9223372036854775808]),
                    None,
                ],
                pa.month_day_nano_interval(),
            )
        }
    )


def _date64() -> Any:
    return pa.table({'d': pa.array([0, None], pa.date64())})


BUILT: dict[str, Callable[[], Any]] = {
    "intervals at their fields' limits": _limits,
    'a date64 stays unencodable': _date64,
}


def _read(vector: dict[str, Any]) -> tuple[Any, list[str] | None]:
    if 'build' in vector:
        return BUILT[vector['build']](), None
    cursor = duckdb.connect().cursor()
    cursor.execute(vector['sql'])
    types = [str(column[1]) for column in cursor.description]
    return cursor.to_arrow_table(), types


# --- the vector file -------------------------------------------------------------------


def test_the_vector_file_is_what_it_says() -> None:
    vectors = _vectors()
    assert len({v['name'] for v in vectors}) == len(vectors)
    others = {
        v['name']
        for v in json.loads(
            (ROOT / 'contract/arrow-map-union-vectors.json').read_text()
        )['vectors']
    }
    # Names unique across 2c's and 2e's files: the rust-host pending list
    # names vectors of both.
    assert not others & {v['name'] for v in vectors}
    intervals = []

    def walk(cell: Any) -> None:
        if not isinstance(cell, list):
            return
        if cell[0] == 'interval':
            intervals.append(cell)
        elif cell[0] == 'list':
            for item in cell[1]:
                walk(item)
        elif cell[0] == 'struct':
            for _, item in cell[1]:
                walk(item)
        elif cell[0] == 'map':
            for key, item in cell[1]:
                walk(key)
                walk(item)
        elif cell[0] == 'union':
            walk(cell[3])

    for vector in vectors:
        assert ('sql' in vector) != ('build' in vector), vector['name']
        if 'build' in vector:
            assert vector['build'] in BUILT, vector['name']
        for row in vector['rows']:
            for cell in row:
                walk(cell)
        texts = [_text(row) for row in vector['rows']]
        assert vector['values']['ordered'] == _digest(texts)
        assert vector['values']['unordered'] == _digest(sorted(texts))
    for cell in intervals:
        assert len(cell) == 4
        assert isinstance(cell[1], int) and isinstance(cell[2], int)
        assert isinstance(cell[3], str) and int(cell[3]) == int(cell[3])
    # Not vacuous: negative fields, a count past 2**53, the int64 limits.
    counts = {int(cell[3]) for cell in intervals}
    assert any(n > 2**53 for n in counts) and min(counts) == -(2**63)
    assert any(cell[1] < 0 and cell[2] < 0 for cell in intervals)


def test_the_built_vectors_are_the_arrow_they_say() -> None:
    for vector in _vectors():
        if 'build' in vector:
            table, _ = _read(vector)
            assert [
                {'name': f.name, 'type': str(f.type)} for f in table.schema
            ] == vector['arrow']


@pytest.mark.parametrize('vector', _vectors(), ids=lambda v: v['name'])
def test_arrow_reads_intervals_as_the_vector_says(
    vector: dict[str, Any],
) -> None:
    table, types = _read(vector)
    assert [
        {'name': f.name, 'type': str(f.type)} for f in table.schema
    ] == vector['arrow']

    def reads() -> None:
        assert run_python.arrow_values(table, types) == vector['values']

    reads()


@pytest.mark.parametrize(
    'vector',
    [v for v in _vectors() if 'sql' in v],
    ids=lambda v: v['name'],
)
def test_an_interval_read_as_rows_stays_unencodable(
    vector: dict[str, Any],
) -> None:
    r"""``fetchall`` reads an interval as a ``timedelta``, its months
    folded into days: still unencodable, never compared as if exact."""
    assert vector['fetchall'] == 'unencodable'
    cursor = duckdb.connect().cursor()
    cursor.execute(vector['sql'])
    names = [column[0] for column in cursor.description]
    types = [str(column[1]) for column in cursor.description]
    read = run_python.result_values(names, types, cursor.fetchall())
    assert read['unencodable']


def test_the_older_interval_vectors_take_the_form() -> None:
    r"""``contract/result-vectors.json``'s interval vectors, which both
    encoders are held to as well, spell the same form."""
    older = {
        v['name']: v
        for v in json.loads(OLD_VECTORS.read_text(encoding='utf-8'))['vectors']
    }

    def taken() -> None:
        assert older['an interval']['rows'] == [
            [['interval', 1, 0, '0'], ['interval', 0, 30, '0']]
        ]
        assert older['an interval']['values']['unencodable'] == []
        nested = older['nested nanoseconds and intervals']
        assert nested['rows'][0][2] == [
            'struct',
            [['i', ['interval', 0, 1, '0']]],
        ]
        assert nested['values']['unencodable'] == []

    taken()
