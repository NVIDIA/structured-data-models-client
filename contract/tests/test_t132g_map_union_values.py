# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32g part 2c: a map's and a union's values in canonical form.

Written by the package's independent test author, before the change.

    python -m pytest contract/tests/test_t132g_map_union_values.py -q

``compare.py``'s ``sql.result`` rule fails every statement result that
holds an unencodable value, even against itself, and a map or a union read
as Arrow is unencodable today (``run_python._arrow_cells``,
``sdm-testkit``'s ``values``). nested.map and nested.union hold one in
``from_source`` statement 6, the reference's sample of ``users``, so they
fail on every pairing of runners whatever the core does. The owner's
direction: a real canonical form, not a skip.

``contract/t132g-p2c-vectors.json`` spells the form out (its ``about``): a
map is ``["map", [[key, value], ...]]`` in Arrow's entry order, duplicates
kept, a null map ``null`` and an empty one ``["map", []]``; a union is
``["union", type code, member name, cell]``, a null member keeping its
tag. Each vector, run on DuckDB (``sql``) or built as Arrow (``build``:
DuckDB writes no duplicate map key and no dense union), must read as the
vector says through ``run_python.arrow_values`` here and through
``sdm_testkit::values`` in ``rust/crates/sdm-testkit/tests/
t132g_map_union_values.rs``, so the two encoders agree through the file.

- the comparator, through ``compare.compare``: a statement result holding
  maps or unions equals itself, and differs from one with another value,
  the same entries in another order, or another union member;
- a value the form does not hold stays unencodable, inside a map or a
  chosen member too, and a result holding one still fails ``sql.result``
  (fails closed): an interval was the seed until part 2e gave intervals a
  form (``contract/t132g-p2e-vectors.json``); DuckDB's ``VARINT``, which
  it writes as an opaque Arrow extension, is now; a map or a union read
  as rows (``fetchall``) stays unencodable;
- ``contract/result-vectors.json``'s map and union vectors take the form;
- the committed reference's nested.map and nested.union samples hold no
  unencodable value, so neither fails ``sql.result`` against itself (their
  reference lines recorded again with the form: an entry naming them on
  python-ref, the owner's, under the ratchet's reference rule).

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

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / 'contract/t132g-p2c-vectors.json'
OLD_VECTORS = ROOT / 'contract/result-vectors.json'
REFERENCE = next((ROOT / 'contract/reference').glob('*.jsonl'))
REGRESSION = ROOT / 'contract/cases/s1/base.regression.json'
INTERVAL = 'arrow:month_day_nano_interval'


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


def _vectors() -> list[dict[str, Any]]:
    loaded = json.loads(VECTORS.read_text(encoding='utf-8'))
    assert loaded['format'] == 'sdm-result-vectors/1'
    vectors: list[dict[str, Any]] = loaded['vectors']
    return vectors


def _text(row: Any) -> str:
    return json.dumps(row, ensure_ascii=True, separators=(',', ':'))


def _digest(texts: list[str]) -> str:
    return hashlib.sha256(('[' + ','.join(texts) + ']').encode()).hexdigest()


# --- the vectors built as Arrow: DuckDB writes no such value ------------------


def _duplicate_key() -> Any:
    return pa.table(
        {
            'm': pa.array(
                [[('a', 1), ('a', 2)]],
                type=pa.map_(pa.string(), pa.int32()),
            )
        }
    )


def _dense() -> Any:
    return pa.table(
        {
            'u': pa.UnionArray.from_dense(
                pa.array([3, 7, 7], pa.int8()),
                pa.array([0, 0, 1], pa.int32()),
                [pa.array([5], pa.int32()), pa.array([None, 'x'], pa.string())],
                ['i', 's'],
                [3, 7],
            )
        }
    )


def _sparse() -> Any:
    return pa.table(
        {
            'u': pa.UnionArray.from_sparse(
                pa.array([7, 3], pa.int8()),
                [
                    pa.array([0, None], pa.int32()),
                    pa.array(['y', None], pa.string()),
                ],
                ['i', 's'],
                [3, 7],
            )
        }
    )


BUILT: dict[str, Callable[[], Any]] = {
    'a map with a duplicate key': _duplicate_key,
    'a dense union with codes 3 and 7 and a null member': _dense,
    'a sparse union with codes 3 and 7 and a null member': _sparse,
}


def _read(vector: dict[str, Any]) -> tuple[Any, list[str] | None]:
    r"""The vector's table, read as Arrow, and DuckDB's type names (None
    for a built one)."""
    if 'build' in vector:
        return BUILT[vector['build']](), None
    cursor = duckdb.connect().cursor()
    cursor.execute(vector['sql'])
    types = [str(column[1]) for column in cursor.description]
    return cursor.to_arrow_table(), types


# --- the vector file -----------------------------------------------------------


def test_the_vector_file_is_what_it_says() -> None:
    vectors = _vectors()
    assert len({v['name'] for v in vectors}) == len(vectors)
    kinds: set[str] = set()

    def walk(cell: Any) -> None:
        if not isinstance(cell, list):
            return
        kinds.add(cell[0])
        if cell[0] == 'list':
            for item in cell[1]:
                walk(item)
        elif cell[0] == 'struct':
            for _, item in cell[1]:
                walk(item)
        elif cell[0] == 'map':
            for key, value in cell[1]:
                walk(key)
                walk(value)
        elif cell[0] == 'union':
            assert isinstance(cell[1], int) and isinstance(cell[2], str)
            assert len(cell) == 4
            walk(cell[3])

    for vector in vectors:
        assert ('sql' in vector) != ('build' in vector), vector['name']
        if 'build' in vector:
            assert vector['build'] in BUILT, vector['name']
        for row in vector['rows']:
            for cell in row:
                walk(cell)
        # Each vector's digests are its rows'.
        texts = [_text(row) for row in vector['rows']]
        assert vector['values']['ordered'] == _digest(texts)
        assert vector['values']['unordered'] == _digest(sorted(texts))
    assert {'map', 'union', 'list', 'struct', 'interval'} <= kinds
    arrow = {f['type'] for v in vectors for f in v['arrow']}
    # Not vacuous: sparse and dense unions, a map of maps, a duplicate key.
    assert any(t.startswith('dense_union<') for t in arrow)
    assert any(t.startswith('sparse_union<') for t in arrow)
    assert any('map<string, map<' in t for t in arrow)
    [duplicate] = [
        v for v in vectors if v['name'] == 'a map with a duplicate key'
    ]
    [[[_, entries]]] = duplicate['rows']
    assert entries[0][0] == entries[1][0]


def test_the_built_vectors_are_the_arrow_they_say() -> None:
    for vector in _vectors():
        if 'build' in vector:
            table, _ = _read(vector)
            assert [
                {'name': f.name, 'type': str(f.type)} for f in table.schema
            ] == vector['arrow']


@pytest.mark.parametrize('vector', _vectors(), ids=lambda v: v['name'])
def test_arrow_reads_maps_and_unions_as_the_vector_says(
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
def test_a_map_or_union_read_as_rows_stays_unencodable(
    vector: dict[str, Any],
) -> None:
    r"""``fetchall`` reads a map as a dict and a union without its tag:
    still unencodable, never compared as if exact."""
    assert vector['fetchall'] == 'unencodable'
    cursor = duckdb.connect().cursor()
    cursor.execute(vector['sql'])
    names = [column[0] for column in cursor.description]
    types = [str(column[1]) for column in cursor.description]
    read = run_python.result_values(names, types, cursor.fetchall())
    assert read['unencodable']
    assert read['unordered'] != vector['values']['unordered']


def test_the_older_vectors_take_the_form(
    request: pytest.FixtureRequest,
) -> None:
    r"""``contract/result-vectors.json``'s map and union vectors, which
    both encoders are held to as well, spell the same form."""
    older = {
        v['name']: v
        for v in json.loads(OLD_VECTORS.read_text(encoding='utf-8'))['vectors']
    }

    def taken() -> None:
        assert older['a map']['rows'] == [[['map', [[['s', 'k'], ['i', '1']]]]]]
        assert older['a map']['values']['unencodable'] == []
        assert older['unions of one value under two tags']['rows'] == [
            [
                ['union', 0, 'i', ['i', '1']],
                ['union', 1, 'j', ['i', '1']],
            ]
        ]
        assert (
            older['unions of one value under two tags']['values']['unencodable']
            == []
        )

    G.held('exits', f'py:{request.node.name}', taken)


# --- the comparator ------------------------------------------------------------


def _reference_of(case: str) -> dict[str, Any]:
    for line in REFERENCE.read_text().splitlines():
        observation: dict[str, Any] = json.loads(line)
        if observation['case'] == case:
            return observation
    raise AssertionError(f'no reference observation of {case}')


#: base.regression's from_source statement read as Arrow (to_arrow_table).
ARROW_READ = 3


def _sql_result(table: Any, types: list[str]) -> dict[str, Any]:
    return {
        'values': run_python.arrow_values(table, types),
        'schema': [{'name': f.name, 'type': str(f.type)} for f in table.schema],
        'rows': table.num_rows,
    }


def _with(observation: dict[str, Any], result: dict[str, Any]) -> None:
    statement = observation['driver']['capture']['statements'][ARROW_READ]
    statement['results'][0].update(copy.deepcopy(result))


def _compared(tmp: Path, theirs: str, mine: str) -> Any:
    r"""base.regression's reference against itself as python-old, but for
    its Arrow read of ``from_source``, which holds ``theirs``' result on
    the reference and ``mine``'s on python-old (each a query DuckDB runs
    and reads as Arrow)."""

    def result(sql: str) -> dict[str, Any]:
        cursor = duckdb.connect().cursor()
        cursor.execute(sql)
        types = [str(column[1]) for column in cursor.description]
        return _sql_result(cursor.to_arrow_table(), types)

    reference = _reference_of('base.regression')
    observation = copy.deepcopy(reference)
    observation['identity']['runner'] = 'python-old'
    observation['identity']['reference'] = None
    _with(reference, result(theirs))
    _with(observation, result(mine))
    paths = []
    for name, line in (('python-ref', reference), ('python-old', observation)):
        path = tmp / f'{name}.jsonl'
        path.write_text(json.dumps(line) + '\n', encoding='utf-8')
        paths.append(path)
    return compare.compare([REGRESSION], paths[0], paths[1:])


def _sql_result_texts(report: Any) -> list[str]:
    return [text for rule, text in report.failures if rule == 'sql.result']


MAPS = "SELECT MAP {'a': 1, 'b': 2} AS m, [MAP {'k': [1]}, NULL] AS l"
UNIONS = (
    'SELECT union_value(i := 1)::UNION(i INTEGER, j INTEGER) AS u, '
    '[union_value(s := NULL::VARCHAR)::UNION(i INTEGER, s VARCHAR)] AS l'
)


@pytest.mark.parametrize('sql', [MAPS, UNIONS], ids=['maps', 'unions'])
def test_the_comparator_takes_a_map_or_union_result_as_equal_to_itself(
    tmp_path: Path, sql: str, request: pytest.FixtureRequest
) -> None:
    def equal() -> None:
        found = _sql_result_texts(_compared(tmp_path, sql, sql))
        assert found == [], found

    G.held('exits', f'py:{request.node.name}', equal)


DIFFERENT = {
    'another map value': (
        MAPS,
        "SELECT MAP {'a': 1, 'b': 3} AS m, [MAP {'k': [1]}, NULL] AS l",
    ),
    'another entry order': (
        MAPS,
        "SELECT MAP {'b': 2, 'a': 1} AS m, [MAP {'k': [1]}, NULL] AS l",
    ),
    'an empty map for a null': (
        MAPS,
        "SELECT MAP {'a': 1, 'b': 2} AS m, [MAP {'k': [1]}, MAP {}] AS l",
    ),
    'another union member': (
        UNIONS,
        'SELECT union_value(j := 1)::UNION(i INTEGER, j INTEGER) AS u, '
        '[union_value(s := NULL::VARCHAR)::UNION(i INTEGER, s VARCHAR)] AS l',
    ),
    'another member for a null': (
        UNIONS,
        'SELECT union_value(i := 1)::UNION(i INTEGER, j INTEGER) AS u, '
        '[union_value(i := NULL::INTEGER)::UNION(i INTEGER, s VARCHAR)] AS l',
    ),
    'another union value': (
        UNIONS,
        'SELECT union_value(i := 2)::UNION(i INTEGER, j INTEGER) AS u, '
        '[union_value(s := NULL::VARCHAR)::UNION(i INTEGER, s VARCHAR)] AS l',
    ),
}


@pytest.mark.parametrize('seed', sorted(DIFFERENT))
def test_the_comparator_takes_a_different_map_or_union_as_different(
    tmp_path: Path, seed: str, request: pytest.FixtureRequest
) -> None:
    theirs, mine = DIFFERENT[seed]

    def different() -> None:
        found = _sql_result_texts(_compared(tmp_path, theirs, mine))
        # The rows differ, not merely a value the form does not hold.
        assert found, found
        assert all('canonical form does not hold' not in t for t in found)
        assert any('rows differ' in t for t in found), found

    G.held('exits', f'py:{request.node.name}', different)


#: A value the form does not hold: DuckDB's VARINT, which it writes as an
#: opaque Arrow extension (an interval was this seed until part 2e).
UNHELD = {
    'in a map': "SELECT MAP {'k': 1::VARINT} AS m",
    'a chosen union member': (
        'SELECT union_value(v := 1::VARINT)::UNION(n INTEGER, v VARINT) AS u'
    ),
    'alone': 'SELECT 1::VARINT AS i',
}


@pytest.mark.parametrize('seed', sorted(UNHELD))
def test_a_result_holding_an_unencodable_value_still_fails_closed(
    tmp_path: Path, seed: str
) -> None:
    sql = UNHELD[seed]
    found = _sql_result_texts(_compared(tmp_path, sql, sql))
    assert any('canonical form does not hold' in t for t in found), found


# --- the reference's samples -------------------------------------------------------


@pytest.mark.parametrize('case', ['nested.map', 'nested.union'])
def test_the_reference_s_sample_compares_equal_to_itself(
    tmp_path: Path, case: str, request: pytest.FixtureRequest
) -> None:
    [path] = (ROOT / 'contract/cases').glob(f'*/{case}.json')
    reference = _reference_of(case)
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
        # Diagnostic cases report rather than fail: both count.
        said = [t for _, t in report.failures] + list(report.notes)
        assert not [t for t in said if 'sql.result' in t or 'canonical' in t]
        unencodable = [
            name
            for statement in reference['driver']['capture']['statements']
            for read in statement['results']
            for name in read['values']['unencodable']
        ]
        assert unencodable == [], unencodable

    G.held('exits', f'py:{request.node.name}', equal)
