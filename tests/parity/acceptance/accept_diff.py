# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P2.diff: the first-divergence diff tool finds each seeded change at the
right place, and reports nothing for identical inputs.

These seeds are independent of the tool's own self-test, which must pass too.
Payload seeds start from ``tests/golden/wire_payloads.json``; transcript seeds
from a stored SQLite transcript.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from _support import (
    DIFF,
    GOLDEN,
    REPO,
    deep,
    execute_steps,
    load,
    output,
    read_arrow,
    recorded,
    run,
    write_arrow,
)

_TOKEN = re.compile(
    r"""\.?([A-Za-z_][A-Za-z0-9_#\-]*)"""
    r"""|\[(\d+)\]"""
    r"""|\["((?:[^"\\]|\\.)*)"\]"""
    r"""|\['((?:[^'\\]|\\.)*)'\]"""
)


def tokens(path: str) -> list[Any]:
    path = path.strip().rstrip(':')
    if path.startswith('$'):
        path = path[1:]
    if path in ('', '.', '<root>'):
        return []
    out: list[Any] = []
    at = 0
    while at < len(path):
        match = _TOKEN.match(path, at)
        if not match or match.end() == at:
            raise AssertionError(f'cannot read the path {path!r}')
        key, index, dq, sq = match.groups()
        if index is not None:
            out.append(int(index))
        else:
            out.append(key if key is not None else dq if dq is not None else sq)
        at = match.end()
    return out


@dataclass
class Want:
    r"""Where a seeded change must be reported.

    The reported path must start with ``at``. If ``exact``, it must also be
    no deeper than ``at``; otherwise any deeper path under ``at`` is fine.
    ``also`` lists other acceptable paths, each reported exactly.
    """

    at: list[Any]
    exact: bool = True
    also: tuple[tuple[Any, ...], ...] = ()


def _cell_names(document: Any, path: list[Any]) -> list[Any]:
    r"""``path`` with a row cell's column index replaced by its name."""
    for i in range(len(path) - 2):
        if path[i] == 'rows' and isinstance(path[i + 1], int):
            if len(path) > i + 2 and isinstance(path[i + 2], int):
                table = document
                for part in path[:i]:
                    table = table[part]
                if isinstance(table, dict) and 'columns' in table:
                    named = list(path)
                    named[i + 2] = table['columns'][path[i + 2]]
                    return named
    return path


def _accepts(want: Want, reported: list[Any], document: Any) -> bool:
    for candidate in (reported, _cell_names(document, reported)):
        if tuple(candidate) in want.also:
            return True
        if candidate[: len(want.at)] == want.at:
            if not want.exact or len(candidate) == len(want.at):
                return True
    return False


def _diff(a: Path, b: Path) -> tuple[int, str]:
    proc = run(DIFF, a, b, timeout=120)
    assert proc.returncode in (0, 1), output(proc)
    return proc.returncode, proc.stdout


def _first_path(stdout: str) -> list[Any]:
    first = next((ln for ln in stdout.splitlines() if ln.strip()), '')
    return tokens(first.split()[0] if first.split() else '')


# Payload sets.


def _payload_set() -> list[dict[str, Any]]:
    golden = json.loads(GOLDEN.read_text())
    return deep(
        [
            golden['tabular/direct-classification'][0],
            golden['relational/pql-count-aggregation'][0],
        ]
    )


_ORDERS = [1, 'context', 'related_tables', 'orders']
_INSTANCE = [1, 'context', 'instance_table']
_CONFIG = [1, 'inference', 'inference_config']


def _at(document: Any, path: list[Any]) -> Any:
    for part in path:
        document = document[part]
    return document


def _set(document: Any, path: list[Any], value: Any) -> None:
    _at(document, path[:-1])[path[-1]] = value


def _col(document: Any, table: list[Any], name: str) -> int:
    return _at(document, table)['columns'].index(name)


def _seed_value(table: list[Any], row: int, column: str, new: Callable) -> Any:
    def seed(doc: Any) -> Want:
        c = _col(doc, table, column)
        cell = [*table, 'rows', row, c]
        _set(doc, cell, new(_at(doc, cell)))
        return Want([*table, 'rows', row, column])

    return seed


def _seed_leaf(path: list[Any], new: Callable, **want: Any) -> Any:
    def seed(doc: Any) -> Want:
        _set(doc, path, new(_at(doc, path)))
        return Want(path, **want)

    return seed


def _swap_rows(doc: Any) -> Want:
    rows = _at(doc, [*_ORDERS, 'rows'])
    rows[5], rows[6] = rows[6], rows[5]
    return Want([*_ORDERS, 'rows', 5], exact=False)


def _drop_middle_row(doc: Any) -> Want:
    table = [1, 'predict', 'related_tables', 'orders']
    del _at(doc, [*table, 'rows'])[10]
    return Want([*table, 'rows', 10], exact=False)


def _drop_last_row(doc: Any) -> Want:
    rows = _at(doc, [*_INSTANCE, 'rows'])
    last = len(rows) - 1
    rows.pop()
    return Want(
        [*_INSTANCE, 'rows', last],
        exact=False,
        also=((*_INSTANCE, 'rows'),),
    )


def _remove_key(doc: Any) -> Want:
    columns = [1, 'schema', 'instance_table', 'columns', 'TARGET']
    del _at(doc, columns)['stype']
    return Want([*columns, 'stype'], also=(tuple(columns),))


def _add_key(doc: Any) -> Want:
    _at(doc, _CONFIG)['extra'] = 1
    return Want([*_CONFIG, 'extra'], also=(tuple(_CONFIG),))


def _swap_columns(doc: Any) -> Want:
    columns = _at(doc, [*_ORDERS, 'columns'])
    columns[0], columns[1] = columns[1], columns[0]
    # The rows don't move, so the column list is the first difference.
    return Want([*_ORDERS, 'columns', 0], also=((*_ORDERS, 'columns'),))


def _swap_relationships(doc: Any) -> Want:
    rel = _at(doc, [1, 'schema', 'relationships'])
    rel[0], rel[1] = rel[1], rel[0]
    return Want([1, 'schema', 'relationships', 0], exact=False)


def _extra_payload(doc: Any) -> Want:
    doc.append(deep(doc[0]))
    return Want([2], also=((),))


def _first_tabular_row_value_to_string(doc: Any) -> Want:
    cell = [0, 'context', 'instance_table', 'rows', 0, 0]
    _set(doc, cell, str(_at(doc, cell)))
    return Want([0, 'context', 'instance_table', 'rows', 0, 'user_id'])


PAYLOAD_SEEDS = {
    'float value': _seed_value(_ORDERS, 7, 'price', lambda v: v + 0.5),
    'integer to float': _seed_value(_INSTANCE, 1, '__kumo_entity_ref_0', float),
    'timestamp': _seed_value(
        _ORDERS, 3, 'ordered_at', lambda v: v.replace('T00', 'T01')
    ),
    'value to null': _seed_value(_ORDERS, 2, 'price', lambda v: None),
    'integer to string': _first_tabular_row_value_to_string,
    'schema dtype': _seed_leaf(
        [1, 'schema', 'related_tables', 'orders', 'columns', 'price', 'dtype'],
        lambda v: 'float32',
    ),
    'one character': _seed_leaf(
        [
            1,
            'schema',
            'related_tables',
            'people',
            'columns',
            'segment',
            'stype',
        ],
        lambda v: v[:-1] + v[-1].upper(),
    ),
    'source query': _seed_leaf(
        [1, 'metadata', 'source_query'], lambda v: v.replace('30', '31')
    ),
    'task kind': _seed_leaf([1, 'task', 'kind'], lambda v: 'classification'),
    'bool flipped': _seed_leaf([*_CONFIG, 'column_shuffle'], lambda v: not v),
    'bool to integer': _seed_leaf([*_CONFIG, 'column_shuffle'], int),
    'output field in the first payload': _seed_leaf(
        [0, 'output', 'fields', 0], lambda v: 'probabilities'
    ),
    'reordered row': _swap_rows,
    'dropped row': _drop_middle_row,
    'dropped last row': _drop_last_row,
    'removed key': _remove_key,
    'added key': _add_key,
    'reordered columns': _swap_columns,
    'reordered relationships': _swap_relationships,
    'extra payload': _extra_payload,
}


@pytest.mark.parametrize('seed', sorted(PAYLOAD_SEEDS))
def test_payload_seed_is_found_in_place(seed: str, tmp_path: Path) -> None:
    before = _payload_set()
    after = deep(before)
    want = PAYLOAD_SEEDS[seed](after)
    assert json.dumps(before) != json.dumps(after), 'the seed changed nothing'
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    a.write_text(json.dumps(before))
    b.write_text(json.dumps(after))
    code, stdout = _diff(a, b)
    assert code == 1, f'{seed}: no difference reported'
    reported = _first_path(stdout)
    assert _accepts(want, reported, before), (
        f'{seed}: reported {reported}, expected {want}'
    )


def test_identical_payload_sets_report_nothing(tmp_path: Path) -> None:
    payloads = _payload_set()
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    a.write_text(json.dumps(payloads))
    b.write_text(json.dumps(deep(payloads)))
    assert _diff(a, b) == (0, '')


def test_key_order_and_layout_are_not_differences(tmp_path: Path) -> None:
    payloads = _payload_set()
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    a.write_text(json.dumps(payloads, separators=(',', ':')))
    b.write_text(json.dumps(payloads, indent=3, sort_keys=True))
    assert _diff(a, b) == (0, '')


# Transcripts.


def _sqlite_transcript() -> Path:
    for entry in recorded():
        if entry['backend'] != 'sqlite':
            continue
        path = REPO / entry['transcript']
        transcript = load(path)
        steps = execute_steps(transcript)
        with_result = [
            i
            for i in steps
            if isinstance(transcript['steps'][i].get('result'), str)
        ]
        if len(steps) >= 3 and with_result:
            return path
    raise AssertionError('no SQLite transcript with three SQL steps')


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    source = _sqlite_transcript()
    a_dir, b_dir = tmp_path / 'a', tmp_path / 'b'
    shutil.copytree(source.parent, a_dir)
    shutil.copytree(source.parent, b_dir)
    return a_dir / source.name, b_dir / source.name


def _middle(transcript: dict[str, Any]) -> int:
    steps = execute_steps(transcript)
    return steps[len(steps) // 2]


def _t_sql_character(t: dict[str, Any], _: Path) -> Want:
    k = _middle(t)
    sql = t['steps'][k]['sql']
    at = next(i for i, c in enumerate(sql) if c.isalpha())
    t['steps'][k]['sql'] = sql[:at] + sql[at].swapcase() + sql[at + 1 :]
    return Want(['steps', k], also=(('steps', k, 'sql'),))


def _t_sql_whitespace(t: dict[str, Any], _: Path) -> Want:
    k = _middle(t)
    t['steps'][k]['sql'] += ' '
    return Want(['steps', k], also=(('steps', k, 'sql'),))


def _t_param(t: dict[str, Any], _: Path) -> Want:
    k = _middle(t)
    step = t['steps'][k]
    params = step.get('params')
    if (
        isinstance(params, list)
        and params
        and not isinstance(params[0], (list, dict))
    ):
        value = params[0]
        params[0] = value + 1 if type(value) is int else f'{value}x'
        return Want(
            ['steps', k],
            also=(('steps', k, 'params'), ('steps', k, 'params', 0)),
        )
    step['params'] = [*params, 7] if isinstance(params, list) else [7]
    return Want(['steps', k], exact=False)


def _t_dropped_step(t: dict[str, Any], _: Path) -> Want:
    k = _middle(t)
    del t['steps'][k]
    return Want(['steps', k], exact=False)


def _t_swapped_steps(t: dict[str, Any], _: Path) -> Want:
    steps = execute_steps(t)
    for a, b in pairwise(steps):
        if b == a + 1 and t['steps'][a] != t['steps'][b]:
            t['steps'][a], t['steps'][b] = t['steps'][b], t['steps'][a]
            return Want(['steps', a], exact=False)
    raise AssertionError('no two adjacent, different SQL steps')


def _t_step_result(t: dict[str, Any], path: Path) -> Want:
    import pyarrow as pa

    for k in execute_steps(t):
        ref = t['steps'][k].get('result')
        if not isinstance(ref, str):
            continue
        table, kind = read_arrow(path.parent / ref)
        if table.num_rows == 0:
            continue
        rows = table.to_pylist()
        for key, value in rows[0].items():
            if type(value) in (int, float, str):
                rows[0][key] = (
                    value + 1 if not isinstance(value, str) else value + 'x'
                )
                write_arrow(
                    path.parent / ref,
                    pa.Table.from_pylist(rows, schema=table.schema),
                    kind,
                )
                return Want(['steps', k], exact=False)
    raise AssertionError('no recorded SQL result to change')


def _rows_path(value: Any, path: list[Any]) -> list[Any] | None:
    r"""The path of the first table with at least two rows."""
    if isinstance(value, dict):
        rows = value.get('rows')
        if isinstance(rows, list) and len(rows) >= 2 and rows[1]:
            return path
        for key, child in value.items():
            found = _rows_path(child, [*path, key])
            if found is not None:
                return found
    return None


def _t_payload_value(t: dict[str, Any], _: Path) -> Want:
    table = _rows_path(t['payloads'][0], ['payloads', 0])
    assert table is not None, 'no payload table with two rows'
    cell = [*table, 'rows', 1, 0]
    value = _at(t, cell)
    _set(t, cell, value + 1 if type(value) is int else f'{value}x')
    return Want([*table, 'rows', 1], exact=False)


TRANSCRIPT_SEEDS = {
    'sql character': _t_sql_character,
    'sql trailing space': _t_sql_whitespace,
    'sql parameter': _t_param,
    'dropped step': _t_dropped_step,
    'swapped steps': _t_swapped_steps,
    'recorded result value': _t_step_result,
    'payload value': _t_payload_value,
}


@pytest.mark.parametrize('seed', sorted(TRANSCRIPT_SEEDS))
def test_transcript_seed_is_found_in_place(seed: str, tmp_path: Path) -> None:
    a, b = _pair(tmp_path)
    original = load(a)
    transcript = deep(original)
    want = TRANSCRIPT_SEEDS[seed](transcript, b)
    b.write_text(json.dumps(transcript, indent=1))
    code, stdout = _diff(a, b)
    assert code == 1, f'{seed}: no difference reported'
    reported = _first_path(stdout)
    assert _accepts(want, reported, original), (
        f'{seed}: reported {reported}, expected {want}'
    )


def test_identical_transcripts_in_different_places_report_nothing(
    tmp_path: Path,
) -> None:
    a, b = _pair(tmp_path)
    assert _diff(a, b) == (0, '')


def test_every_stored_transcript_equals_itself(tmp_path: Path) -> None:
    for entry in recorded():
        path = REPO / entry['transcript']
        assert _diff(path, path) == (0, ''), entry['name']


def test_the_tools_own_self_test_passes() -> None:
    proc = run(DIFF, '--self-test', timeout=300)
    assert proc.returncode == 0, output(proc)


def test_unreadable_input_is_an_error_not_a_verdict(tmp_path: Path) -> None:
    good = tmp_path / 'good.json'
    good.write_text(json.dumps(_payload_set()))
    bad = tmp_path / 'bad.json'
    bad.write_text('{"not json')
    proc = run(DIFF, good, bad, timeout=120)
    assert proc.returncode not in (0, 1), output(proc)
    proc = run(DIFF, good, tmp_path / 'missing.json', timeout=120)
    assert proc.returncode not in (0, 1), output(proc)


def test_there_are_at_least_twenty_seeds() -> None:
    assert len(PAYLOAD_SEEDS) + len(TRANSCRIPT_SEEDS) >= 20
