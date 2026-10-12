# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Records the reference's payload bytes for generated DuckDB sources
(T1.28w exit 3, the package's independent test author), which
``test_t128w_payload_bytes.py`` holds python-core's to.

Run with the reference checkout's interpreter (the reference client and
its relational engine importable), from anywhere:

    <reference>/.venv/bin/python -I \\
        packages/structured-data-models-client/tests/t128w_payload_bytes_record.py

Each sample is a DuckDB source made by ``sources()`` (seeded, so the same
statements every run): a ``users`` table keyed by an integer or by text,
an ``orders`` table referring to it with a time column, and columns of
integer, float, decimal, text, boolean and time types of several widths,
with nulls and NaN where the sample says. The reference materializes the
regression over it; the payload is recorded as the bytes of its compact
JSON (``separators=(',', ':')``, ``allow_nan=False``, ``ensure_ascii=False``),
the reference engine's own spelling of a payload's size
(``payload_size_bytes``), or the exception's class where it refuses.
Nothing is written by hand.
"""

from __future__ import annotations

import json
import random
import sys
import warnings
from pathlib import Path
from typing import Any

OUT = Path(__file__).with_name('t128w-payload-bytes.json')
QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
ANCHOR = '2024-03-01'

PRICE_TYPES = [
    'INTEGER',
    'BIGINT',
    'DOUBLE',
    'FLOAT',
    'SMALLINT',
    'DECIMAL(10,2)',
]
EXTRA_TYPES = [
    ('VARCHAR', lambda r: f"'w{r.randint(0, 5)} x{r.randint(0, 3)}'"),
    ('BOOLEAN', lambda r: r.choice(['true', 'false'])),
    ('DOUBLE', lambda r: repr(r.uniform(-1e6, 1e6))),
    ('INTEGER', lambda r: str(r.randint(-1000, 1000))),
    ('BIGINT', lambda r: str(r.randint(-(2**53), 2**53))),
    (
        'TIMESTAMP',
        lambda r: (
            f"TIMESTAMP '2023-{r.randint(1, 12):02d}-{r.randint(1, 28):02d} 0{r.randint(0, 9)}:00:00'"
        ),
    ),
    (
        'DATE',
        lambda r: f"DATE '2023-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}'",
    ),
    ('FLOAT', lambda r: repr(round(r.uniform(-100, 100), 3))),
]


def sources(count: int = 30) -> list[dict[str, Any]]:
    r"""The samples: each its statements, indices and seed."""
    made = []
    for n in range(count):
        r = random.Random(1000 + n)
        text_key = r.random() < 0.3
        users = r.randint(3, 12)
        orders = r.randint(0, 40)
        price = r.choice(PRICE_TYPES)
        extras = r.sample(EXTRA_TYPES, r.randint(0, 4))
        nulls = r.random() < 0.4
        key_type = 'VARCHAR' if text_key else 'BIGINT'

        def key(i: int, text_key: bool = text_key) -> str:
            return f"'u{i}'" if text_key else str(i)

        user_cols = ['user_id ' + key_type + ' PRIMARY KEY', 'age INTEGER']
        user_cols += [f'u_x{j} {t}' for j, (t, _) in enumerate(extras)]
        statements = [f'CREATE TABLE users ({", ".join(user_cols)})']
        for i in range(users):
            cells = [
                key(i),
                'NULL'
                if nulls and r.random() < 0.3
                else str(r.randint(18, 90)),
            ]
            cells += [
                'NULL' if nulls and r.random() < 0.3 else make(r)
                for _, make in extras
            ]
            statements.append(f'INSERT INTO users VALUES ({", ".join(cells)})')
        statements.append(
            'CREATE TABLE orders (order_id BIGINT PRIMARY KEY, user_id '
            f'{key_type} REFERENCES users(user_id), price {price}, '
            'created_at TIMESTAMP)'
        )
        for i in range(orders):
            if price.startswith(('DOUBLE', 'FLOAT', 'DECIMAL')):
                value = repr(round(r.uniform(-50, 500), 2))
            else:
                value = str(r.randint(-50, 500))
            if nulls and r.random() < 0.15:
                value = 'NULL'
            when = f"TIMESTAMP '2024-0{r.randint(1, 2)}-{r.randint(1, 28):02d} {r.randint(0, 23):02d}:{r.randint(0, 59):02d}:00'"
            statements.append(
                f'INSERT INTO orders VALUES ({i}, {key(r.randrange(users))}, {value}, {when})'
            )
        indices = sorted(r.sample(range(users), min(users, r.randint(1, 3))))
        made.append(
            {
                'id': f'gen-{n:02d}',
                'statements': statements,
                'indices': [f'u{i}' if text_key else i for i in indices],
                'seed': r.randint(0, 2**31),
            }
        )
    return made


def materialize(sample: dict[str, Any]) -> dict[str, Any]:
    import duckdb
    import pandas as pd
    from sdm_connectors import DuckDBSource

    from sdm_client import PredictiveQuery, Schema

    connection = duckdb.connect()
    for statement in sample['statements']:
        connection.execute(statement)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(DuckDBSource(connection))
            request = PredictiveQuery(QUERY, schema).materialize(
                indices=sample['indices'],
                anchor_time=pd.Timestamp(ANCHOR),
                random_seed=sample['seed'],
            )
        payload = request._payloads[0]
        text = json.dumps(
            payload, separators=(',', ':'), allow_nan=False, ensure_ascii=False
        )
    except Exception as refused:  # noqa: BLE001 - recorded by class
        return {'refused': type(refused).__name__}
    return {'payload': text}


def main() -> int:
    import sdm_relational_engine  # noqa: F401 - the reference's engine

    recorded = []
    for sample in sources():
        recorded.append({**sample, **materialize(sample)})
    reference = Path(sys.executable).resolve()
    OUT.write_text(
        json.dumps(
            {
                'recorder': Path(__file__).name,
                'query': QUERY,
                'anchor_time': ANCHOR,
                'samples': recorded,
            },
            indent=1,
            ensure_ascii=False,
        )
        + '\n',
        encoding='utf-8',
    )
    made = sum('payload' in s for s in recorded)
    print(
        f'{OUT.name}: {made} payloads, {len(recorded) - made} refusals ({reference})'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
