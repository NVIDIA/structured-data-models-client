# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32g part 2e: an interval's values in canonical form.

Written by the package's independent test author, before the change.

    python -m pytest contract/tests/test_t132g_interval_values.py -q

``compare.py``'s ``sql.result`` rule fails every result holding an
unencodable value, even against itself, and both encoders write Arrow's
``month_day_nano_interval`` as unencodable (E111(b) left it so).
time.feature-interval holds one in ``from_source`` statement 6, the
reference's sample of ``users``, so it fails on every pairing of runners.
The orchestrator's decision: a canonical form, as part 2c gave maps and
unions one.

``contract/t132g-p2e-vectors.json`` spells it out (its ``about``): an
interval is ``["interval", months, days, nanoseconds]``, months and days
JSON integers (int32), nanoseconds a decimal string (int64, as each
64-bit count of the form is: ``i``, ``ts``, ``dur``), a null interval
``null``; alike inside a list, a struct, a map or a chosen union member.
Each vector, run on DuckDB (``sql``) or built as Arrow (``build``: the
fields' limits; a ``date64``, which the form still does not hold), must
read as the vector says through ``run_python.arrow_values`` here and
through ``sdm_testkit::values`` in ``rust/crates/sdm-testkit/tests/
t132g_interval_values.rs``. Arrow's year-month and day-time intervals,
which neither DuckDB nor pyarrow writes, stay unencodable (the Rust test
builds them).

- the comparator: an interval result equals itself, and differs from one
  with other months, days or nanoseconds, another sign, or a zero for a
  null;
- part 2c's vector that kept an interval unencodable in a map and a
  chosen union member takes the form (``contract/t132g-p2c-vectors.json``,
  updated here; its test is 2c's); ``contract/result-vectors.json``'s
  interval vectors take it too;
- an interval read as rows (``fetchall``) stays unencodable;
- the committed reference's time.feature-interval sample compares equal to
  itself, and is the new encoder's reading of that sample run again on
  DuckDB: its line recorded again (an entry naming it on python-ref, the
  owner's, under the ratchet's reference rule).

Gating (``contract/tests/t132g_gate.py``): a test named in
``t132g-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``) failed
when written. It still runs and must still fail; once it passes it fails,
naming itself, until it is taken out of the list.
"""

from __future__ import annotations

import copy
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
import yaml

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / 'contract/t132g-p2e-vectors.json'
OLD_VECTORS = ROOT / 'contract/result-vectors.json'
CASE = 'time.feature-interval'


def _module(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


G = _module('t132g_gate', ROOT / 'contract/tests/t132g_gate.py')
run_python = _module('run_python', ROOT / 'contract/runners/run_python.py')
compare = _module('compare', ROOT / 'scripts/rust_core/compare.py')
maps = _module(
    't132g_map_union_tests',
    ROOT / 'contract/tests/test_t132g_map_union_values.py',
)


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
            (ROOT / 'contract/t132g-p2c-vectors.json').read_text()
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
    vector: dict[str, Any], request: pytest.FixtureRequest
) -> None:
    table, types = _read(vector)
    assert [
        {'name': f.name, 'type': str(f.type)} for f in table.schema
    ] == vector['arrow']

    def reads() -> None:
        assert run_python.arrow_values(table, types) == vector['values']

    G.held('exits', f'py:{request.node.name}', reads)


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


def test_the_older_interval_vectors_take_the_form(
    request: pytest.FixtureRequest,
) -> None:
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

    G.held('exits', f'py:{request.node.name}', taken)


# --- the comparator ------------------------------------------------------------------


INTERVALS = (
    "SELECT INTERVAL '1 month 2 days 3 microseconds' AS i, "
    '[INTERVAL 1 DAY, NULL] AS l'
)


def test_the_comparator_takes_an_interval_result_as_equal_to_itself(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    def equal() -> None:
        found = maps._sql_result_texts(
            maps._compared(tmp_path, INTERVALS, INTERVALS)
        )
        assert found == [], found

    G.held('exits', f'py:{request.node.name}', equal)


DIFFERENT = {
    'other months': (
        "SELECT INTERVAL '2 month 2 days 3 microseconds' AS i, "
        '[INTERVAL 1 DAY, NULL] AS l'
    ),
    'other days': (
        "SELECT INTERVAL '1 month 3 days 3 microseconds' AS i, "
        '[INTERVAL 1 DAY, NULL] AS l'
    ),
    'other nanoseconds': (
        "SELECT INTERVAL '1 month 2 days 4 microseconds' AS i, "
        '[INTERVAL 1 DAY, NULL] AS l'
    ),
    'another sign': (
        "SELECT -INTERVAL '1 month 2 days 3 microseconds' AS i, "
        '[INTERVAL 1 DAY, NULL] AS l'
    ),
    'a zero for a null': (
        "SELECT INTERVAL '1 month 2 days 3 microseconds' AS i, "
        '[INTERVAL 1 DAY, INTERVAL 0 DAY] AS l'
    ),
    'thirty days for a month': (
        "SELECT INTERVAL '30 days 2 days 3 microseconds' AS i, "
        '[INTERVAL 1 DAY, NULL] AS l'
    ),
}


@pytest.mark.parametrize('seed', sorted(DIFFERENT))
def test_the_comparator_takes_a_different_interval_as_different(
    tmp_path: Path, seed: str, request: pytest.FixtureRequest
) -> None:
    def different() -> None:
        found = maps._sql_result_texts(
            maps._compared(tmp_path, INTERVALS, DIFFERENT[seed])
        )
        assert found, found
        assert all('canonical form does not hold' not in t for t in found)
        assert any('rows differ' in t for t in found), found

    G.held('exits', f'py:{request.node.name}', different)


# --- the reference's sample ----------------------------------------------------------


def test_the_interval_reference_sample_compares_equal_to_itself(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    [path] = (ROOT / 'contract/cases').glob(f'*/{CASE}.json')
    reference = maps._reference_of(CASE)
    observation = copy.deepcopy(reference)
    observation['identity']['runner'] = 'python-old'
    observation['identity']['reference'] = None
    paths = []
    for name, line in (('python-ref', reference), ('python-old', observation)):
        written = tmp_path / f'{name}.jsonl'
        written.write_text(json.dumps(line) + '\n', encoding='utf-8')
        paths.append(written)

    def equal() -> None:
        report = compare.compare([path], paths[0], paths[1:])
        # A diagnostic case reports rather than fails: both count.
        said = [t for _, t in report.failures] + list(report.notes)
        assert not [t for t in said if 'sql.result' in t or 'canonical' in t]

    G.held('exits', f'py:{request.node.name}', equal)


def test_the_interval_reference_sample_is_the_new_encoder_s_reading(
    request: pytest.FixtureRequest,
) -> None:
    r"""The sample (``from_source`` statement 6, ``users`` read as Arrow)
    run again on DuckDB, from the case's base and setup, and read by the
    new encoder, is the reference line's: the line recorded again."""
    case = json.loads(
        next((ROOT / 'contract/cases').glob(f'*/{CASE}.json')).read_text()
    )
    bases = yaml.safe_load((ROOT / 'contract/bases.yaml').read_text())
    connection = duckdb.connect()
    for statement in [
        *bases['bases'][case['input']['base']],
        *case['input']['setup'],
    ]:
        connection.execute(statement)
    recorded = [
        s
        for s in maps._reference_of(CASE)['driver']['capture']['statements']
        if s['stage'] == 'from_source'
    ][6]
    [read] = recorded['results']
    assert read['method'] == 'to_arrow_table'
    cursor = connection.cursor()
    cursor.execute(recorded['sql'])
    types = [str(column[1]) for column in cursor.description]
    table = cursor.to_arrow_table()
    assert 'month_day_nano_interval' in [str(f.type) for f in table.schema]
    assert [
        {'name': f.name, 'type': str(f.type)} for f in table.schema
    ] == read['schema']

    def recorded_again() -> None:
        assert run_python.arrow_values(table, types) == read['values']
        assert read['values']['unencodable'] == []

    G.held('exits', f'py:{request.node.name}', recorded_again)
