# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The canonical form of a statement's result (decision H9; T1.11e).

    python -m pytest contract/tests/test_result_vectors.py -q

The driver capture records each result's rows in a canonical form
(``run_python.result_values``) that the comparator compares across runners:
each vector of ``contract/result-vectors.json`` run on DuckDB gives the
vector's cells and digests whether its rows are read with ``fetchall`` (as
the reference reads its catalog) or as Arrow (as the core's providers read
every result), and the pure-Rust host gives the same from Arrow
(``sdm-testkit``'s ``tests/result_vectors.rs``). A value the form does not
hold is named, never taken as equal.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / 'contract/result-vectors.json'


def _module(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / 'contract/runners' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_python = _module('run_python')


def _vectors() -> list[dict[str, Any]]:
    loaded = json.loads(VECTORS.read_text(encoding='utf-8'))
    assert loaded['format'] == 'sdm-result-vectors/1'
    vectors: list[dict[str, Any]] = loaded['vectors']
    return vectors


def test_the_vector_file() -> None:
    vectors = _vectors()
    # Not vacuous: every kind of cell the form names but the zoned and
    # duration ones (which need pytz to read through fetchall) appears,
    # and one vector has no rows.
    kinds = set()

    def walk(cell: Any) -> None:
        if isinstance(cell, list):
            kinds.add(cell[0])
            if cell[0] == 'list':
                for item in cell[1]:
                    walk(item)
            if cell[0] == 'struct':
                for _, item in cell[1]:
                    walk(item)
        else:
            kinds.add(type(cell).__name__)

    for vector in vectors:
        for row in vector['rows']:
            for cell in row:
                walk(cell)
    assert kinds >= {
        'i', 'f', 's', 'b', 'd', 'date', 'ts', 'time', 'list', 'struct',
        'bool', 'NoneType',
    }  # fmt: skip
    assert any(vector['rows'] == [] for vector in vectors)
    assert len({vector['name'] for vector in vectors}) == len(vectors)


@pytest.mark.parametrize('vector', _vectors(), ids=lambda v: v['name'])
def test_rows_and_arrow_read_alike(vector: dict[str, Any]) -> None:
    cursor = duckdb.connect().cursor()
    cursor.execute(vector['sql'])
    names = [column[0] for column in cursor.description]
    types = [str(column[1]) for column in cursor.description]
    rows = cursor.fetchall()
    odd: list[str] = []
    assert [
        [run_python.canonical_cell(value, odd) for value in row] for row in rows
    ] == vector['rows']
    assert odd == []
    assert run_python.result_values(names, types, rows) == vector['values']
    cursor.execute(vector['sql'])
    table = cursor.to_arrow_table()
    assert [
        {'name': field.name, 'type': str(field.type)} for field in table.schema
    ] == vector['arrow']
    assert run_python.arrow_values(table, types) == vector['values']


def test_order_counts_only_in_ordered() -> None:
    rows = [(1, 'a'), (2, 'b'), (2, 'b')]
    one = run_python.result_values(['x', 'y'], None, rows)
    other = run_python.result_values(['x', 'y'], None, rows[::-1])
    assert one['unordered'] == other['unordered']
    assert one['ordered'] != other['ordered']
    # A duplicate counts.
    fewer = run_python.result_values(['x', 'y'], None, rows[:2])
    assert fewer['unordered'] != one['unordered']


def test_values_keep_their_kind_and_exactness() -> None:
    def cells(*values: Any) -> list[Any]:
        return [run_python.canonical_cell(value, []) for value in values]

    # An integer and a float of the same value differ; so do a decimal's
    # scales, -0.0 and 0.0, and null and NaN.
    assert cells(1)[0] != cells(1.0)[0]
    assert cells(Decimal('1.5'))[0] != cells(Decimal('1.50'))[0]
    assert cells(-0.0)[0] != cells(0.0)[0]
    assert cells(None)[0] != cells(float('nan'))[0]
    # Floats colliding as text keep their bits; decimals colliding as
    # float64 stay apart.
    assert cells(0.1)[0] == ['f', '3fb999999999999a']
    assert (
        cells(Decimal('0.100000000000000001'))[0]
        != cells(Decimal('0.100000000000000002'))[0]
    )


def test_a_value_the_form_does_not_hold_is_named() -> None:
    odd: list[str] = []
    cell = run_python.canonical_cell(object(), odd)
    assert cell == ['?', 'builtins.object']
    assert odd == ['builtins.object']
    values = run_python.result_values(['x'], None, [(object(),)])
    assert values['unencodable'] == ['builtins.object']
