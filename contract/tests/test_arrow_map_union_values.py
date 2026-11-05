# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map and union observations preserve canonical values from shared vectors.

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
VECTORS = ROOT / 'contract/t132g-p2c-vectors.json'
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


def test_the_older_vectors_take_the_form() -> None:
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

    taken()
