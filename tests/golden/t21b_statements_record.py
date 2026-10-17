# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Records ``t21b-statements.json``: every SQL statement the core hands the
Python host's DuckDB provider on the public route, in order, for
``Schema.from_source`` and ``PredictiveQuery.materialize`` over the ``shop``
base, once as it is and once with ``orders`` held in a table named ``a.b``
beside a schema ``a`` holding a table ``b`` (T2.1b).

Recorded at contract version 25, before the table source became a
``Relation``: v26 renders a one-part relation as v25 rendered the name, so
``tests/test_t21b_contract_v26.py`` holds every later core to these texts
(the design's "SQL goldens unchanged"; "a dotted source name still reads as
one table, a golden shows it"). Regenerated, never edited, and only when a
rendering changes on purpose:

    python tests/golden/t21b_statements_record.py
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).with_name('t21b-statements.json')

QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
ANCHOR = '2024-03-01'
INDICES = [0, 1]

#: ``orders`` moved into a table named ``a.b``, and a decoy ``a.b`` (schema
#: ``a``, table ``b``) whose prices differ, so a name split on its dot reads
#: other rows.
DOTTED = (
    'CREATE TABLE "a.b" AS SELECT * FROM orders',
    'DROP TABLE orders',
    'CREATE SCHEMA a',
    'CREATE TABLE a.b AS SELECT order_id, user_id, price * 100 AS price, ts'
    ' FROM "a.b"',
)

RUNS: dict[str, tuple[str, tuple[str, ...]]] = {
    'plain': ('orders', ()),
    'dotted': ('a.b', DOTTED),
}


def shop() -> list[str]:
    r"""The ``shop`` base of ``contract/bases.yaml``: the plain list items
    under its name."""
    lines = (
        (ROOT / 'contract/bases.yaml').read_text(encoding='utf-8').splitlines()
    )
    start = lines.index('  shop:') + 1
    statements = []
    for line in lines[start:]:
        if not line.startswith('  - '):
            break
        statements.append(line[len('  - ') :])
    return statements


def run(orders: str, setup: tuple[str, ...]) -> tuple[Any, list[list[str]]]:
    r"""The payload of ``QUERY`` materialized on the core over the shop base
    with ``setup`` run and ``orders`` named as the source of ``orders``, and
    the statement texts of each fetch the provider was handed, in order."""
    import duckdb
    import pandas as pd
    from sdm_client import PredictiveQuery, Schema, _routing
    from sdm_client._rust.providers.duckdb import DuckDBProvider
    from sdm_connectors import DuckDBSource

    seen: list[list[str]] = []
    answer = DuckDBProvider.__call__

    def recording(self: Any, step: Any, keys: Any = None) -> Any:
        seen.append([s['text'] for s in step['plan']['statements']])
        return answer(self, step, keys)

    forced = ','.join(
        _routing.FROM_SOURCE_DUCKDB_TABLES + _routing.MATERIALIZE_DUCKDB
    )
    before = os.environ.get(_routing.ENVIRONMENT)
    os.environ[_routing.ENVIRONMENT] = forced
    DuckDBProvider.__call__ = recording  # type: ignore[method-assign]
    try:
        connection = duckdb.connect()
        for statement in [*shop(), *setup]:
            connection.execute(statement)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(
                DuckDBSource(connection), {'users': 'users', 'orders': orders}
            )
            request = PredictiveQuery(QUERY, schema).materialize(
                indices=INDICES, anchor_time=pd.Timestamp(ANCHOR)
            )
    finally:
        DuckDBProvider.__call__ = answer  # type: ignore[method-assign]
        if before is None:
            os.environ.pop(_routing.ENVIRONMENT, None)
        else:
            os.environ[_routing.ENVIRONMENT] = before
    return request._payloads[0], seen


def main() -> None:
    from sdm_client import _native

    document = {
        'about': (
            "Every SQL statement the core hands the Python host's DuckDB "
            'provider, per fetch in order, for Schema.from_source and '
            'PredictiveQuery.materialize over the shop base, as it is '
            '("plain") and with orders held in a table named "a.b" beside a '
            'schema a holding a table b ("dotted"). Recorded by '
            't21b_statements_record.py at the contract version named; '
            'regenerated, never edited (T2.1b).'
        ),
        'contract': _native.CONTRACT_VERSION,
        'query': QUERY,
        'anchor_time': ANCHOR,
        'indices': INDICES,
        'dotted_setup': list(DOTTED),
        'statements': {
            name: run(orders, setup)[1]
            for name, (orders, setup) in RUNS.items()
        },
    }
    GOLDEN.write_text(
        json.dumps(document, indent=1, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
