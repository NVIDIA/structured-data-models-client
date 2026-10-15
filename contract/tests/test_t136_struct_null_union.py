# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.36 P3 C3: a struct with a null row holding a union reads, whole.

Written by the package's independent test author, before the fix.

    python -m pytest contract/tests/test_t136_struct_null_union.py -q

``run_python._arrow_cells`` reads a struct through ``StructArray.flatten()``,
which folds the struct's validity into each child. A union has no validity
of its own, so when a struct with a null row holds one, Arrow C++ aborts
the process (``array_nested.cc``: ``Check failed: (data_->buffers[0]) ==
(nullptr)``, SIGABRT, exit 134 from a shell), and the runner with it.
DuckDB writes such a struct (T1.32g p2c found it).

``contract/t136-struct-union-vectors.json`` holds the cells such a read must
give, in the canonical form of ``contract/t132g-p2c-vectors.json``'s
``about``: a null struct row is ``null``; a non-null one ``["struct",
[[name, cell], ...]]``; a union member ``["union", type code, member name,
cell]``. Its vectors: the struct DuckDB writes, one with other members and a
null union beside the union, a list of such structs (each list's structs a
slice of the child at its offset), a struct in a struct with the outer one
null, a map whose values are such structs, and built ones DuckDB cannot
write: a dense union in a struct with a null row, and lists of structs
holding a dense or a sparse union, so a read that drops a struct's offset
gives other cells.

A value under a null struct row is hidden: no cell shows it, so it names
nothing unencodable either, as the reading before the fix (``flatten()``
nulled it) and ``sdm-testkit``'s ``values`` (row by row) agree. Reading a
struct's children as stored (``StructArray.field``) reaches those values;
the second set of vectors, added by the test author after review of the
first fix, holds that: built (DuckDB nulls a null struct's children
itself), each a null struct row hiding a value the form does not hold
(a ``date64``) as a child, as the member a union chooses at that row
(dense and sparse), in a list's struct at an offset, in a map's value and
in a struct in a struct; each expects the cells it gave before and
``unencodable`` empty. A control holds a visible ``date64`` beside a
hidden one, which still names ``arrow:date64[ms]``. Every vector's
``unencodable`` is exactly the names its visible cells show as ``["?",
name]``.

Each vector is read through ``run_python.arrow_values`` in a child process
(this file run as a script, given the vector's name), so the abort is this
test failing, named, and not the pytest session dying. Nothing else of the
form changes: ``test_t132g_map_union_values.py`` and
``test_t132g_interval_values.py`` still hold every other value.

Not read by ``sdm-testkit``'s Rust tests: the abort is pyarrow's, and this
file's vectors are held to the Python runner only.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import duckdb
import pyarrow as pa
import pytest

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / 'contract/t136-struct-union-vectors.json'
#: Seconds a child read may take; a read is milliseconds.
TIMEOUT = 120


def _module(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _vectors() -> list[dict[str, Any]]:
    loaded = json.loads(VECTORS.read_text(encoding='utf-8'))
    assert loaded['format'] == 'sdm-result-vectors/1'
    vectors: list[dict[str, Any]] = loaded['vectors']
    return vectors


def _text(row: Any) -> str:
    return json.dumps(row, ensure_ascii=True, separators=(',', ':'))


def _digest(texts: list[str]) -> str:
    return hashlib.sha256(('[' + ','.join(texts) + ']').encode()).hexdigest()


# --- the vectors built as Arrow: DuckDB writes no dense union -----------------


def _dense(
    codes: list[int], at: list[int], ints: list[int], texts: list[Any]
) -> Any:
    return pa.UnionArray.from_dense(
        pa.array(codes, pa.int8()),
        pa.array(at, pa.int32()),
        [pa.array(ints, pa.int32()), pa.array(texts, pa.string())],
        ['i', 's'],
        [3, 7],
    )


def _sparse(codes: list[int], ints: list[Any], texts: list[Any]) -> Any:
    return pa.UnionArray.from_sparse(
        pa.array(codes, pa.int8()),
        [pa.array(ints, pa.int32()), pa.array(texts, pa.string())],
        ['i', 's'],
        [3, 7],
    )


def _structs(union: Any, nulls: list[bool]) -> Any:
    r"""A struct of ``u`` (the union) and ``n`` (its row's index), null
    where ``nulls`` says."""
    return pa.StructArray.from_arrays(
        [union, pa.array(range(len(nulls)), pa.int32())],
        names=['u', 'n'],
        mask=pa.array(nulls),
    )


def _dense_struct() -> Any:
    union = _dense([7, 3, 3], [0, 0, 1], [5, 6], ['x'])
    return pa.table({'a': _structs(union, [False, True, False])})


def _lists(structs: Any, offsets: list[int]) -> Any:
    return pa.table(
        {'l': pa.ListArray.from_arrays(pa.array(offsets, pa.int32()), structs)}
    )


def _dense_lists() -> Any:
    r"""Two lists of five structs, the second its structs from offset 2:
    each struct's cells differ, so a read at offset 0 gives others."""
    union = _dense([7, 3, 3, 7, 7], [0, 0, 1, 1, 2], [10, 11], ['a', 'b', 'c'])
    return _lists(_structs(union, [False, True, False, True, False]), [0, 2, 5])


def _sparse_lists() -> Any:
    union = _sparse(
        [7, 3, 7, 7, 3],
        [None, 21, None, None, 24],
        ['p', None, 'r', 'q', None],
    )
    return _lists(_structs(union, [True, False, False, True, False]), [0, 2, 5])


# --- a value the form does not hold, hidden under a null struct row ---------

#: The day a hidden (or visible) date64 holds.
DAY = datetime.date(2020, 1, 1)


def _dates(days: list[Any]) -> Any:
    return pa.array(days, pa.date64())


def _dated(days: list[Any], nulls: list[bool]) -> Any:
    r"""A struct of ``d`` (a date64) and ``n`` (its row's index)."""
    return pa.StructArray.from_arrays(
        [_dates(days), pa.array(range(len(nulls)), pa.int32())],
        names=['d', 'n'],
        mask=pa.array(nulls),
    )


def _hidden_date() -> Any:
    r"""Row 0 visible, its date null; row 1 null, hiding a date."""
    return pa.table({'a': _dated([None, DAY], [False, True])})


def _hidden_dense_member() -> Any:
    r"""Row 1, null, hides the union's date64 member, chosen there."""
    union = pa.UnionArray.from_dense(
        pa.array([3, 7], pa.int8()),
        pa.array([0, 0], pa.int32()),
        [pa.array([5], pa.int32()), _dates([DAY])],
        ['i', 'd'],
        [3, 7],
    )
    return pa.table({'a': _structs(union, [False, True])})


def _hidden_sparse_member() -> Any:
    union = pa.UnionArray.from_sparse(
        pa.array([3, 7], pa.int8()),
        [pa.array([5, None], pa.int32()), _dates([None, DAY])],
        ['i', 'd'],
        [3, 7],
    )
    return pa.table({'a': _structs(union, [False, True])})


def _hidden_date_lists() -> Any:
    r"""The second list's structs from offset 2; its null struct (3) hides
    a date. Read from offset 0 its structs would be 0 to 2, all valid, and
    their indices other."""
    return _lists(
        _dated(
            [None, None, None, DAY, None], [False, False, False, True, False]
        ),
        [0, 2, 5],
    )


def _hidden_date_map() -> Any:
    r"""A map whose ``j`` value is a null struct hiding a date."""
    return pa.table(
        {
            'm': pa.MapArray.from_arrays(
                pa.array([0, 2], pa.int32()),
                pa.array(['k', 'j']),
                _dated([None, DAY], [False, True]),
            )
        }
    )


def _hidden_date_inner() -> Any:
    r"""A struct in a struct; the outer null in row 1, the inner valid
    there, holding a date."""
    inner = pa.StructArray.from_arrays([_dates([None, DAY])], names=['d'])
    outer = pa.StructArray.from_arrays(
        [inner, pa.array([0, 1], pa.int32())],
        names=['o', 'n'],
        mask=pa.array([False, True]),
    )
    return pa.table({'a': outer})


def _visible_date() -> Any:
    r"""The control: row 0 visible with a date, row 1 null hiding one."""
    return pa.table({'a': _dated([DAY, DAY], [False, True])})


BUILT: dict[str, Callable[[], Any]] = {
    'a built struct with a null row holding a dense union': _dense_struct,
    'built lists of structs holding a dense union, null structs among them': (
        _dense_lists
    ),
    'built lists of structs holding a sparse union, null structs among them': (
        _sparse_lists
    ),
    'a null struct row hiding a date64': _hidden_date,
    "a null struct row hiding a dense union's chosen date64 member": (
        _hidden_dense_member
    ),
    "a null struct row hiding a sparse union's chosen date64 member": (
        _hidden_sparse_member
    ),
    'lists of structs at an offset, a null struct hiding a date64': (
        _hidden_date_lists
    ),
    "a map's value a null struct hiding a date64": _hidden_date_map,
    'a struct in a null struct hiding a date64': _hidden_date_inner,
    'a visible date64 beside a null struct row hiding one': _visible_date,
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


def _child(name: str) -> None:
    r"""Run as a script: read the named vector through ``arrow_values`` and
    print what it gives, as JSON, on one line."""
    run_python = _module('run_python', ROOT / 'contract/runners/run_python.py')
    [vector] = [v for v in _vectors() if v['name'] == name]
    table, types = _read(vector)
    print(json.dumps(run_python.arrow_values(table, types)), flush=True)


# --- the vector file -----------------------------------------------------------


def test_the_vector_file_is_what_it_says() -> None:
    vectors = _vectors()
    assert len({v['name'] for v in vectors}) == len(vectors)
    nulls = 0
    for vector in vectors:
        assert ('sql' in vector) != ('build' in vector), vector['name']
        if 'build' in vector:
            assert vector['build'] == vector['name']
            assert vector['build'] in BUILT, vector['name']
            assert vector['values']['types'] is None, vector['name']
        # Every vector holds a struct, and lists unencodable exactly the
        # names its visible cells show.
        assert any('struct<' in f['type'] for f in vector['arrow'])
        assert vector['values']['unencodable'] == sorted(
            {name for row in vector['rows'] for name in _shown(row)}
        ), vector['name']
        assert vector['values']['columns'] == [
            f['name'] for f in vector['arrow']
        ]
        # Each vector's digests are its rows'.
        texts = [_text(row) for row in vector['rows']]
        assert vector['values']['ordered'] == _digest(texts), vector['name']
        assert vector['values']['unordered'] == _digest(sorted(texts))
        nulls += sum(cell is None for row in vector['rows'] for cell in row)
    # Not vacuous: null struct rows at the top, sparse and dense unions,
    # DuckDB's and built.
    assert nulls >= 3
    arrow = {f['type'] for v in vectors for f in v['arrow']}
    assert any('dense_union<' in t for t in arrow)
    assert any('sparse_union<' in t for t in arrow)
    assert {'sql', 'build'} <= {k for v in vectors for k in v}
    # Hidden date64s, as a child and as dense and sparse chosen members,
    # and the control that names a visible one.
    assert sum('date64' in t for t in arrow) >= 5
    assert any('dense_union<i: int32=3, d: date64[ms]=7>' in t for t in arrow)
    assert any('sparse_union<i: int32=3, d: date64[ms]=7>' in t for t in arrow)
    assert [v['values']['unencodable'] for v in vectors].count(
        ['arrow:date64[ms]']
    ) == 1


def _shown(cell: Any) -> list[str]:
    r"""The names of the unencodable values a cell shows, ``["?", name]``."""
    if not isinstance(cell, list):
        return []
    if len(cell) == 2 and cell[0] == '?' and isinstance(cell[1], str):
        return [cell[1]]
    return [name for item in cell for name in _shown(item)]


def test_the_built_vectors_are_the_arrow_they_say() -> None:
    r"""Building is safe in process: only reading a struct aborts."""
    for vector in _vectors():
        if 'build' in vector:
            table, _ = _read(vector)
            assert [
                {'name': f.name, 'type': str(f.type)} for f in table.schema
            ] == vector['arrow']


def test_the_built_lists_hold_structs_at_an_offset() -> None:
    r"""A built list's second row is a slice of its structs at offset 2, a
    null struct among them, so a read that drops the offset is caught;
    the date64 one hides its date there."""
    for name in BUILT:
        table, _ = _read({'build': name})
        if not pa.types.is_list(table.schema.field(0).type):
            continue
        structs = table.column(0).chunk(0)[1].values
        assert structs.offset == 2, name
        assert structs.null_count >= 1, name


def _died(returncode: int) -> str:
    if returncode < 0:
        return f'killed by {signal.Signals(-returncode).name} ({returncode})'
    return f'exit {returncode}'


@pytest.mark.parametrize('vector', _vectors(), ids=lambda v: v['name'])
def test_a_struct_with_a_null_row_holding_a_union_reads_as_the_vector_says(
    vector: dict[str, Any],
) -> None:
    done = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), vector['name']],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        cwd=ROOT,
        check=False,
    )
    assert done.returncode == 0, (
        f'reading {vector["name"]!r} through run_python.arrow_values died: '
        f'{_died(done.returncode)} (SIGABRT is exit 134 from a shell); '
        f'stderr: {done.stderr.strip()[-2000:]}'
    )
    lines = done.stdout.strip().splitlines()
    assert lines, f'no values printed; stderr: {done.stderr.strip()[-2000:]}'
    got = json.loads(lines[-1])
    table, _ = _read(vector)
    assert [
        {'name': f.name, 'type': str(f.type)} for f in table.schema
    ] == vector['arrow']
    assert got['unencodable'] == vector['values']['unencodable'], (
        f'{vector["name"]!r} lists unencodable {got["unencodable"]}, the '
        f'vector {vector["values"]["unencodable"]}: a value hidden under a '
        'null struct row is in no cell, so it names nothing unencodable; '
        'only a value a visible row shows does'
    )
    assert got == vector['values'], (
        f'{vector["name"]!r} read otherwise than the vector says; its cells: '
        f'{_text(vector["rows"])}'
    )


if __name__ == '__main__':
    _child(sys.argv[1])
