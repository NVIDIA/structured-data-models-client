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
    # Not vacuous: every kind of cell the form names but a duration (which
    # DuckDB's SQL cannot make) appears, an interval in its form (T1.32g
    # part 2e; what stays unencodable is held by t132g-p2e-vectors.json),
    # nanosecond times differ from each other, and one vector has no rows.
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
        assert vector['fetchall'] in ('same', 'unencodable', 'not read')
        for row in vector['rows']:
            for cell in row:
                walk(cell)
    assert kinds >= {
        'i', 'f', 's', 'b', 'd', 'date', 'ts', 'tstz', 'time', 'list',
        'struct', 'interval', 'bool', 'NoneType',
    }  # fmt: skip
    assert any(vector['rows'] == [] for vector in vectors)
    assert len({vector['name'] for vector in vectors}) == len(vectors)
    [nanos] = [v for v in vectors if v['name'].startswith('nanosecond')]
    assert nanos['rows'][0][:2] == [['time', '1'], ['time', '2']]


def _cursor(vector: dict[str, Any]) -> Any:
    connection = duckdb.connect()
    if 'setup' in vector:
        connection.execute(vector['setup'])
    return connection.cursor()


@pytest.mark.parametrize('vector', _vectors(), ids=lambda v: v['name'])
def test_arrow_reads_as_the_vector_says(vector: dict[str, Any]) -> None:
    cursor = _cursor(vector)
    cursor.execute(vector['sql'])
    types = [str(column[1]) for column in cursor.description]
    table = cursor.to_arrow_table()
    assert [
        {'name': field.name, 'type': str(field.type)} for field in table.schema
    ] == vector['arrow']
    assert run_python.arrow_values(table, types) == vector['values']


@pytest.mark.parametrize(
    'vector',
    [v for v in _vectors() if v['fetchall'] != 'not read'],
    ids=lambda v: v['name'],
)
def test_rows_read_as_arrow_or_are_unencodable(vector: dict[str, Any]) -> None:
    r"""A row read gives the Arrow read's cells and digests, or, for a
    type the driver reads with less than its precision (``TIME_NS``,
    ``TIMESTAMP_NS``, ``INTERVAL``), unencodable cells, so a value the
    driver truncated is never compared as equal."""
    cursor = _cursor(vector)
    cursor.execute(vector['sql'])
    names = [column[0] for column in cursor.description]
    types = [str(column[1]) for column in cursor.description]
    rows = cursor.fetchall()
    read = run_python.result_values(names, types, rows)
    if vector['fetchall'] == 'same':
        odd: list[str] = []
        assert [
            [run_python.canonical_cell(value, odd) for value in row]
            for row in rows
        ] == vector['rows']
        assert odd == []
        assert read == vector['values']
    else:
        assert read['unencodable']
        assert all('read as rows' in name or name.startswith('arrow:')
                   for name in read['unencodable'])  # fmt: skip
        assert read['unordered'] != vector['values']['unordered']


def test_nanoseconds_a_microsecond_read_would_lose_are_kept() -> None:
    r"""Two ``TIME_NS`` values a nanosecond apart differ as Arrow, nested
    or not (Codex's repro); read as rows they are unencodable."""

    def read(clock: str) -> Any:
        cursor = duckdb.connect().cursor()
        query = f"SELECT TIME_NS '{clock}' AS c, [TIME_NS '{clock}'] AS l"
        cursor.execute(query)
        arrow = run_python.arrow_values(cursor.to_arrow_table())
        cursor.execute(query)
        rows = run_python.result_values(
            ['c', 'l'], ['TIME_NS', 'TIME_NS[]'], cursor.fetchall()
        )
        return arrow, rows

    one, one_rows = read('00:00:00.000000001')
    two, _ = read('00:00:00.000000002')
    assert one['unordered'] != two['unordered']
    assert one['unencodable'] == []
    assert one_rows['unencodable'] == [
        'duckdb:TIME_NS read as rows',
        'duckdb:TIME_NS[] read as rows',
    ]


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


@pytest.mark.parametrize(
    ('duckdb_type', 'lossy'),
    [
        # Field names that look like types are names, not types.
        ('STRUCT(INTERVAL_COUNT INTEGER)', False),
        ('STRUCT(TIMESTAMP_NS_VALID BOOLEAN, "UNION" VARCHAR)', False),
        ("ENUM('INTERVAL', 'TIME_NS')", False),
        ('STRUCT(k INTEGER, v VARCHAR)', False),
        ('INTEGER[2]', False),
        ('DECIMAL(10,2)', False),
        ('TIMESTAMP WITH TIME ZONE', False),
        ('MAP(VARCHAR, INTEGER)', True),
        # The types themselves, at any depth.
        ('TIME_NS', True),
        ('STRUCT("a b" TIME_NS)', True),
        ('STRUCT("x""y" STRUCT(t TIMESTAMP_NS))', True),
        ('INTERVAL[][]', True),
        ('UNION(i INTEGER, j VARCHAR)', True),
        ('STRUCT(u UNION(i INTEGER))[]', True),
        # A spelling it cannot read fails closed.
        ('STRUCT(a', True),
    ],
)
def test_lossiness_is_read_from_the_types(
    duckdb_type: str, lossy: bool
) -> None:
    assert run_python._lossy_as_rows(duckdb_type) is lossy


def test_a_union_read_as_rows_never_compares_equal() -> None:
    r"""``fetchall`` gives a union's value without its tag, so ``i := 1``
    and ``j := 1`` would read alike: read as rows, both are unencodable."""
    reads = []
    for member in ('i', 'j'):
        cursor = duckdb.connect().cursor()
        cursor.execute(
            f'SELECT union_value({member} := 1)::UNION(i INTEGER, j INTEGER)'
            ' AS u'
        )
        types = [str(column[1]) for column in cursor.description]
        reads.append(run_python.result_values(['u'], types, cursor.fetchall()))
    assert all(read['unencodable'] for read in reads)


def test_struct_fields_named_like_types_read_as_rows_and_as_arrow() -> None:
    cursor = duckdb.connect().cursor()
    query = (
        'SELECT struct_pack(INTERVAL_COUNT := 1) AS a, '
        'struct_pack(TIMESTAMP_NS_VALID := true) AS b'
    )
    cursor.execute(query)
    types = [str(column[1]) for column in cursor.description]
    rows = run_python.result_values(['a', 'b'], types, cursor.fetchall())
    cursor.execute(query)
    arrow = run_python.arrow_values(cursor.to_arrow_table(), types)
    assert rows['unencodable'] == []
    assert rows == arrow
