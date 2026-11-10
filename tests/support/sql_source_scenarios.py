# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute source-name scenarios and expose current payloads and SQL to tests."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

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
    from sdm_client import PredictiveQuery, Schema
    from sdm_client._rust.providers.duckdb import DuckDBProvider
    from sdm_connectors import DuckDBSource

    seen: list[list[str]] = []
    answer = DuckDBProvider.__call__

    def recording(self: Any, step: Any, keys: Any = None) -> Any:
        seen.append([s['text'] for s in step['plan']['statements']])
        return answer(self, step, keys)

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
    return request._payloads[0], seen
