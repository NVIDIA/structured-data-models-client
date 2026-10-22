# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T3.3's test fixtures: a capture of one case on one warehouse leg, built
offline from the ``shop`` base of ``contract/bases.yaml`` as the old client
would have recorded it (Phase 3 design, choice 2, "What C-1 records"), and
the fetch steps the core sends the stand-in.

A capture, as the writer takes it and the converter gives it back
(``contract/runners/captures.py``'s ``write(path, capture)`` and
``read(path)``), is plain data:

- ``case``, ``leg``: the case's id and its warehouse leg;
- ``recording_commit``, ``reference_commit``: the commit C-1 recorded from,
  and the reference's;
- ``versions``: the leg's ``runtime``, ``python``, ``pandas`` and ``driver``;
- ``input_hash``: the hash of the case's input;
- ``session``: the warehouse session's ``time_zone`` and ``null_order``
  (``nulls_first`` or ``nulls_last`` for an ascending order);
- ``reference``: the old client's observation line;
- ``tables``: each table as the warehouse returned it, a ``pyarrow.Table`` by
  name;
- ``derived``: each relation the old client derived from a view, evaluated
  by the warehouse (``view``, ``table``, ``columns``, ``filter``, ``data``);
- ``draws``: each random draw, ``table``, ``seed``, ``fraction`` (the
  percentage as the old client wrote it into ``TABLESAMPLE``) and ``rows``;
- ``statements``: each statement the old client sent, ``text`` and
  ``parameters`` (as the old client passed them, JSON values), and its
  ``result`` (a ``pyarrow.Table``) or ``error``.

``read`` gives it back with ``format`` added, the capture's format version.
Catalog and schema appear as the placeholders ``{catalog}`` and ``{schema}``,
which the writer puts in place of their names before anything is written.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
#: The reference's commit, whose observation lines the fixture borrows.
REFERENCE_COMMIT = '43e55f4f58924384c3cb52f67c8ed2e73ba45fa1'
#: The old client's statement reading the users table on Databricks, as
#: the catalog-qualified name reads (``RE/backend/databricks/table.py``).
USERS = '`{catalog}`.`{schema}`.`users`'
#: The draw the fixture recorded: the users table, a seed past 2**31, and
#: the percentage as Python's ``str(float)`` writes it
#: (``RE/backend/databricks/sampler.py:255-262``).
DRAW_SEED = 2**31 + 7
DRAW_FRACTION = '12.5'
#: A recorded statement, with a parameter, and the result the warehouse
#: gave: rows no table holds, so a served answer shows it was replayed.
REPLAYED_TEXT = f'SELECT `user_id`\nFROM {USERS}\nLIMIT ?'
REPLAYED_PARAMETERS = [3]
REPLAYED_ROWS = [1001, 1002, 1003]


def _bases() -> dict[str, list[str]]:
    import yaml

    document = yaml.safe_load((ROOT / 'contract/bases.yaml').read_text())
    return document['bases']


def shop_tables() -> dict[str, Any]:
    r"""The ``shop`` base's tables as Arrow, as a warehouse would return
    them through its driver."""
    import duckdb

    duck = duckdb.connect()
    for statement in _bases()['shop']:
        duck.execute(statement)
    tables = {}
    for name in ('users', 'orders'):
        reader = duck.execute(f'SELECT * FROM {name} ORDER BY 1')
        fetch = getattr(reader, 'to_arrow_table', None) or (
            reader.fetch_arrow_table
        )
        tables[name] = fetch()
    duck.close()
    return tables


def reference_line(case: str = 'base.regression') -> dict[str, Any]:
    r"""The old client's committed observation line of ``case``."""
    path = ROOT / f'contract/reference/{REFERENCE_COMMIT}.jsonl'
    with path.open(encoding='utf-8') as lines:
        for line in lines:
            found = json.loads(line)
            if found.get('case') == case:
                return found
    raise KeyError(case)


def capture(leg: str = 'databricks-sql') -> dict[str, Any]:
    r"""A capture of one case on ``leg``: the shop tables, one draw and one
    statement with its result."""
    import pyarrow as pa

    tables = shop_tables()
    users = tables['users']
    picked = users.filter(
        pa.array([i in (2, 5) for i in range(users.num_rows)])
    )
    return {
        'case': 't33.standin.shop',
        'leg': leg,
        'recording_commit': '1' * 40,
        'reference_commit': REFERENCE_COMMIT,
        'versions': {
            'runtime': 'fixture',
            'python': '3.12',
            'pandas': '3.0',
            'driver': 'fixture',
        },
        'input_hash': hashlib.sha256(b't33.standin.shop').hexdigest(),
        'session': {'time_zone': 'UTC', 'null_order': 'nulls_first'},
        'reference': reference_line(),
        'tables': tables,
        'derived': [],
        'draws': [
            {
                'table': 'users',
                'seed': DRAW_SEED,
                'fraction': DRAW_FRACTION,
                'rows': picked,
            }
        ],
        'statements': [
            {
                'text': REPLAYED_TEXT,
                'parameters': list(REPLAYED_PARAMETERS),
                'result': pa.table(
                    {'user_id': pa.array(REPLAYED_ROWS, pa.int64())}
                ),
            }
        ],
    }


def without(found: dict[str, Any], **drop: Any) -> dict[str, Any]:
    r"""``found`` with a table (``table=<name>``) or every draw
    (``draws=True``) left out."""
    changed = dict(found)
    if 'table' in drop:
        changed['tables'] = {
            name: data
            for name, data in found['tables'].items()
            if name != drop['table']
        }
    if drop.get('draws'):
        changed['draws'] = []
    return changed


# --- the steps the core sends ------------------------------------------------


def _i64(value: int) -> dict[str, str]:
    return {'t': 'i64', 'v': str(value)}


def _sql(text: str, params: list[Any] | None = None) -> dict[str, Any]:
    return {
        'plan': 'sql',
        'statements': [
            {
                'dialect': 'databricks',
                'text': text,
                'params': [
                    {'param': 'value', 'value': value} for value in params or []
                ],
            }
        ],
        'expect': {'expect': 'discover'},
    }


def scan_users(step_id: int = 1) -> dict[str, Any]:
    r"""A scan of the users table's keys, by a statement the old client
    never sent: answered on the local engine."""
    return {
        'id': step_id,
        'request': {
            'request': 'scan',
            'table': {'source': ['users']},
            'columns': ['user_id'],
            'filters': [],
            'limit': None,
        },
        'plan': _sql(f'SELECT `user_id`\nFROM {USERS}'),
    }


def preview_users(text: str, limit: int, step_id: int = 1) -> dict[str, Any]:
    r"""A preview of the users table's keys, its plan's statement ``text``
    binding the limit as a parameter."""
    return {
        'id': step_id,
        'request': {
            'request': 'preview',
            'table': {'source': ['users']},
            'columns': ['user_id'],
            'limit': limit,
        },
        'plan': _sql(text, [_i64(limit)]),
    }


def sample_users(
    seed: int = DRAW_SEED, fraction: str = DRAW_FRACTION, step_id: int = 1
) -> dict[str, Any]:
    r"""An entity-sampling request on the users table, its plan the old
    client's Databricks statement with ``fraction`` and ``seed`` as
    written."""
    return {
        'id': step_id,
        'request': {
            'request': 'sample_entities',
            'table': {'source': ['users']},
            'key': ['user_id'],
            'n': 2,
            'seed': {'t': 'u64', 'v': str(seed)},
            'columns': ['user_id'],
            'filters': [{'column': 'user_id', 'filter': 'not_null'}],
        },
        'plan': _sql(
            f'SELECT `user_id`\nFROM {USERS}\n'
            f'TABLESAMPLE ({fraction} PERCENT) REPEATABLE ({seed})\n'
            'WHERE `user_id` IS NOT NULL\nLIMIT 2'
        ),
    }
