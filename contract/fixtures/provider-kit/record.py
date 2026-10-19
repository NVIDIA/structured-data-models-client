#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Recomputes the provider kit's expected answers from its dataset (T2.5).

    python -I contract/fixtures/provider-kit/record.py

Builds ``dataset.sql`` in DuckDB, reads each table's rows whole, and
computes every answer ``expected.json`` holds in plain Python, from the
request's meaning as the contract states it (``sdm_contract::data``),
never through ``sdm-data``'s SQL: so the answers are not the rendering's
own, and a rendering that drifts is caught by the kit, not copied into
it. Exits 0 when every committed answer is the recomputed one, and 1,
printing each recomputed answer that differs, otherwise. It writes
nothing: the committed file is the test author's, the hand-checked
answers in it worked out by hand first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATASET = HERE / 'dataset.sql'
EXPECTED = HERE / 'expected.json'


def tables() -> dict[str, tuple[list[str], list[tuple[Any, ...]]]]:
    r"""Each table of the dataset, built in DuckDB: its column names and
    rows, in DuckDB's scan order."""
    import duckdb

    connection = duckdb.connect()
    try:
        connection.execute(DATASET.read_text(encoding='utf-8'))
        names = [
            row[0]
            for row in connection.execute(
                'SELECT table_name FROM duckdb_tables() ORDER BY table_name'
            ).fetchall()
        ]
        out = {}
        for name in names:
            cursor = connection.execute(f'SELECT * FROM "{name}"')
            columns = [d[0] for d in cursor.description]
            out[name] = (columns, cursor.fetchall())
        return out
    finally:
        connection.close()


def _table(
    data: dict[str, tuple[list[str], list[tuple[Any, ...]]]],
    relation: dict[str, Any],
) -> list[dict[str, Any]]:
    (name,) = relation['source']
    columns, rows = data[name]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def _kept(row: dict[str, Any], filters: list[dict[str, Any]]) -> bool:
    for condition in filters:
        assert condition['filter'] == 'not_null', condition
        if row[condition['column']] is None:
            return False
    return True


def answer(
    data: dict[str, tuple[list[str], list[tuple[Any, ...]]]],
    request: dict[str, Any],
    seeds: list[list[int]] | None,
) -> tuple[list[str], list[list[Any]]]:
    r"""The columns and rows ``request`` asks for, by the contract's meaning
    of each request, over every row of ``data``."""
    kind = request['request']
    if kind in ('scan', 'preview'):
        rows = [
            [row[c] for c in request['columns']]
            for row in _table(data, request['table'])
            if _kept(row, request.get('filters', []))
        ]
        return list(request['columns']), rows
    if kind == 'count_rows':
        rows = _table(data, request['table'])
        kept = [row for row in rows if _kept(row, request['filters'])]
        return ['rows'], [[len(kept)]]
    if kind == 'key_counts':
        out = []
        for position, key in enumerate(request['keys']):
            rows = _table(data, key['table'])
            whole = [
                tuple(row[c] for c in key['columns'])
                for row in rows
                if all(row[c] is not None for c in key['columns'])
            ]
            missing = sum(
                1 for row in rows if any(row[c] is None for c in key['columns'])
            )
            out.append([position, len(rows), len(set(whole)), missing])
        return ['k', 'n', 'd', 'z'], out
    if kind == 'repeated_key':
        key = request['key']
        held: dict[tuple[Any, ...], int] = {}
        for row in _table(data, request['table']):
            value = tuple(row[c] for c in key)
            if all(v is not None for v in value):
                held[value] = held.get(value, 0) + 1
        repeated = [list(v) for v, n in held.items() if n > 1]
        # At most one row comes back; the kit's answer is only defined
        # where at most one value repeats.
        assert len(repeated) <= 1, repeated
        return list(key), repeated
    assert seeds is not None, request
    (key,) = request['key']
    rows = _table(data, request['table'])
    if kind == 'lookup':
        out = []
        for seed_key, batch in seeds:
            matching = [row for row in rows if row[key] == seed_key]
            if matching:
                # Which row of a key several hold is not defined: the kit
                # asks only for keys one row holds.
                assert len(matching) == 1, (seed_key, matching)
                out.append(
                    [batch, len(matching)]
                    + [matching[0][c] for c in request['columns']]
                )
        columns = ['__kumo_batch__', '__kumo_matches__', *request['columns']]
        return columns, out
    assert kind == 'neighbors', request
    out = []
    for seed_key, batch in seeds:
        matching = [row for row in rows if row[key] == seed_key]
        if request.get('per_seed') is not None:
            for term in reversed(request['order']):
                assert not term['as_timestamp'], term
                assert all(row[term['column']] is not None for row in matching)
                matching.sort(
                    key=lambda row, c=term['column']: row[c],
                    reverse=term['descending'],
                )
            matching = matching[: request['per_seed']]
        out.extend(
            [batch] + [row[c] for c in request['columns']] for row in matching
        )
    return ['__kumo_batch__', *request['columns']], out


def check() -> list[str]:
    r"""Each case whose committed answer is not the recomputed one, with
    the recomputed answer; empty when every one agrees."""
    expected = json.loads(EXPECTED.read_text(encoding='utf-8'))
    data = tables()
    problems = []
    for case in expected['cases']:
        if 'answer' not in case:
            continue
        wanted = case['answer']
        columns, rows = answer(data, case['request'], case.get('seeds'))
        if 'any_of' in wanted:
            got = {
                'columns': columns,
                'count': min(wanted['count'], len(rows)),
                'any_of': rows,
            }
        else:
            got = {
                'columns': columns,
                'ordered': wanted['ordered'],
                'rows': rows,
            }
            if not wanted['ordered']:
                # A multiset: the committed order is the test author's.
                got['rows'] = sorted(rows, key=json.dumps)
                wanted = {
                    **wanted,
                    'rows': sorted(wanted['rows'], key=json.dumps),
                }
        if got != wanted:
            problems.append(f'{case["id"]}: recomputed {json.dumps(got)}')
    return problems


def main() -> int:
    problems = check()
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
