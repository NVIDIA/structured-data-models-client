# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record the reference's labels for ``test_labels_match_the_reference.py``.

Run with the reference checkout's interpreter (the reference environment of
``contract/reference-env.txt``), from this repository's root::

    ~/code/sdmc-reference/.venv/bin/python \\
        packages/structured-data-models-client/tests/labels_reference_record.py

Each case is a DuckDB source built from ``BASE`` and the case's own rows, a
query and an anchor time. The reference materializes it over the source and
this script keeps, of the payload, the task and the context examples' rows
(instance, entity, anchor time and label), or, when the reference refuses
the call, the refusal's class and code. Nothing here is written by hand but
the inputs; every expected value is the reference's.
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from sdm_connectors import DuckDBSource

from sdm_client import PredictiveQuery, Schema

OUT = Path(__file__).with_name('labels_reference.json')

USER_TYPES = {
    'segment': 'VARCHAR',
    'active': 'BOOLEAN',
    'tier': 'DECIMAL(38,18)',
    'score': 'DOUBLE',
}


def _users(n: int, **columns: list[Any]) -> list[str]:
    r"""``users`` with a key and the columns named, nothing else, so the
    payload carries no column the case does not need."""
    names = [name for name in USER_TYPES if name in columns]
    create = ', '.join(
        ['user_id BIGINT PRIMARY KEY']
        + [f'{name} {USER_TYPES[name]}' for name in names]
    )
    rows = []
    for i in range(n):
        values = [str(i + 1)]
        for name in names:
            value = columns[name][i]
            values.append('NULL' if value is None else str(value))
        rows.append(f'({", ".join(values)})')
    return [
        f'CREATE TABLE users ({create})',
        f'INSERT INTO users VALUES {", ".join(rows)}',
    ]


def _orders(rows: list[tuple[Any, ...]], decimal: bool) -> list[str]:
    r"""``rows`` as (user, price, qty, amount, ts); ids in order. The
    DECIMAL(38,18) ``amount`` only where the case asks for it."""
    create = (
        'CREATE TABLE orders (order_id BIGINT PRIMARY KEY, '
        'user_id BIGINT REFERENCES users(user_id), price DOUBLE, qty BIGINT, '
        + ('amount DECIMAL(38,18), ' if decimal else '')
        + 'ts TIMESTAMP)'
    )
    out = []
    for i, (user, price, qty, amount, ts) in enumerate(rows):
        values = [
            str(i + 1),
            str(user),
            'NULL' if price is None else f'CAST({price!r} AS DOUBLE)',
            'NULL' if qty is None else str(qty),
        ]
        if decimal:
            values.append(
                'NULL'
                if amount is None
                else f"CAST('{amount}' AS DECIMAL(38,18))"
            )
        values.append(f"TIMESTAMP '{ts}'")
        out.append(f'({", ".join(values)})')
    return [create, f'INSERT INTO orders VALUES {", ".join(out)}']


# Orders about a leap-day anchor: the context example at 2024-02-29 (one
# month before 2024-03-31) has the window (2024-02-29, 2024-03-29], so an
# order at the anchor itself and one a microsecond after the window's end
# fall outside it.
MONTH_END = [
    (1, 1.0, 1, '1', '2024-02-29 00:00:00'),
    (1, 2.0, 2, '2', '2024-02-29 00:00:00.000001'),
    (1, 4.0, 4, '4', '2024-03-29 00:00:00'),
    (1, 8.0, 8, '8', '2024-03-29 00:00:00.000001'),
    (1, 16.0, 16, '16', '2024-03-31 00:00:00'),
    (2, 32.0, 32, '32', '2024-03-15 00:00:00'),
    (2, None, None, None, '2024-03-16 00:00:00'),
    (3, 64.0, 64, '64', '2024-02-28 23:59:59'),
    (3, 128.0, 128, '128', '2024-03-30 00:00:00'),
    (4, None, None, None, '2024-03-01 00:00:00'),
    (5, 0.5, 5, '0.5', '2024-01-31 00:00:00'),
    (5, 0.25, 6, '0.25', '2024-01-10 00:00:00'),
]

# The same in a common year: 2023-02-28 + 1 month is 2023-03-28.
COMMON_YEAR = [
    (
        u,
        p,
        q,
        a,
        ts.replace('2024-02-29', '2023-02-28')
        .replace('2024-03-29', '2023-03-28')
        .replace('2024-', '2023-'),
    )
    for (u, p, q, a, ts) in MONTH_END
]

# float32 sums where Kahan's compensation decides the label: 2**24 plus
# ones, which a plain float32 sum loses whatever the order.
KAHAN = (
    [(1, 16777216.0, 1, '1', f'2024-03-{d:02d} 00:00:00') for d in (2, 3)]
    + [(1, 1.0, 1, '1', f'2024-03-{d:02d} 00:00:00') for d in range(4, 12)]
    + [(2, 0.1, 1, '0.1', f'2024-03-{d:02d} 00:00:00') for d in range(2, 22)]
    + [(3, 1e30, 1, '1', '2024-03-02 00:00:00')]
    + [(3, 3.0, 1, '1', f'2024-03-{d:02d} 00:00:00') for d in range(3, 9)]
    + [
        (4, 0.1, 1, '0.1', '2024-03-02 00:00:00'),
        (4, 0.2, 1, '0.2', '2024-03-03 00:00:00'),
    ]
    + [(5, 1.0000001, 1, '1', f'2024-03-{d:02d} 00:00:00') for d in range(2, 9)]
)


def _a_day_later(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    r"""``rows`` a day later, inside the window after the 2024-03-02
    context anchor rather than at it; one order before the
    window keeps the data reaching back to the anchor."""
    return [
        (*row[:4], str(pd.Timestamp(row[4]) + pd.Timedelta(days=1)))
        for row in rows
    ] + [(4, None, None, None, '2024-03-01 00:00:00')]


KAHAN = _a_day_later(KAHAN)

# DECIMAL(38,18) values whose exact sum a float sum does not keep.
DECIMALS = [
    (1, 0.1, 1, '0.100000000000000001', '2024-03-02 00:00:00'),
    (1, 0.2, 1, '0.200000000000000002', '2024-03-03 00:00:00'),
    (2, 1.0, 1, '1000000000000000.1', '2024-03-02 00:00:00'),
    (2, 1.0, 1, '-1000000000000000', '2024-03-03 00:00:00'),
    (3, 1.0, 1, '123456789.123456789123456789', '2024-03-02 00:00:00'),
    (4, 1.0, 1, '0.000000000000000001', '2024-03-02 00:00:00'),
    (5, 1.0, 1, '-0.333333333333333333', '2024-03-02 00:00:00'),
    (5, 1.0, 1, '-0.333333333333333333', '2024-03-03 00:00:00'),
    (5, 1.0, 1, '0.000000000000000002', '2024-03-04 00:00:00'),
]
DECIMALS = _a_day_later(DECIMALS)

# DECIMAL(38,18) sums at float32's midpoint 1 + 2**-24
# (1.000000059604644775390625): just above it (rounded once, the next
# float32 up, 1.0000001192092896; through float64 first, the midpoint
# itself and then 1.0 by ties to even), just below it (1.0 either way), and
# the negative of the first (T2.0c's exact decimals, single rounding).
MIDPOINT = _a_day_later(
    [
        (1, 1.0, 1, '1.000000059604644776', '2024-03-02 00:00:00'),
        (2, 1.0, 1, '1.000000059604644775', '2024-03-02 00:00:00'),
        (3, 1.0, 1, '-1.000000059604644776', '2024-03-02 00:00:00'),
    ]
)

LEAP_ANCHOR = [
    (1, 1.0, 1, '1', '2023-02-28 00:00:00'),
    (1, 2.0, 2, '2', '2023-03-01 00:00:00'),
    (1, 4.0, 4, '4', '2024-02-28 00:00:00'),
    (1, 8.0, 8, '8', '2024-02-28 00:00:00.000001'),
    (2, 16.0, 16, '16', '2024-02-29 00:00:00'),
    (3, 32.0, 32, '32', '2023-06-30 00:00:00'),
]

ANCHOR = '2024-03-31'
MARCH = '2024-04-01'

CASES: list[dict[str, Any]] = []


def case(
    name: str,
    covers: str,
    query: str,
    orders: list[tuple[Any, ...]],
    anchor: str,
    users: int = 5,
    indices: list[int] | None = None,
    **user_columns: list[Any],
) -> None:
    CASES.append(
        {
            'name': name,
            'covers': covers,
            'setup': [
                *_users(users, **user_columns),
                *_orders(orders, decimal='orders.amount' in query),
            ],
            'query': query,
            'anchor': anchor,
            'indices': list(range(1, users + 1))
            if indices is None
            else indices,
        }
    )


MONTH_END_EXIT = 'exit: DateOffset window ends at month ends and leap days'
REGRESSION_EXIT = "exit: base.regression's labels, compared exactly"
CLASSIFICATION_EXIT = 'exit: binary and multiclass labels match the reference'

for agg in ('SUM', 'COUNT', 'AVG', 'MIN', 'MAX'):
    case(
        f'{agg.lower()}-month-leap',
        f'{MONTH_END_EXIT}; probe: {agg} windows, nulls, empty windows',
        f'PREDICT {agg}(orders.price, 0, 1, months) FOR EACH users.user_id',
        MONTH_END,
        ANCHOR,
    )
    case(
        f'{agg.lower()}-month-common',
        f'{MONTH_END_EXIT}; probe: {agg} windows in a common year',
        f'PREDICT {agg}(orders.price, 0, 1, months) FOR EACH users.user_id',
        COMMON_YEAR,
        '2023-03-31',
    )
    case(
        f'{agg.lower()}-int-month',
        f'{REGRESSION_EXIT}; probe: {agg} over an integer column',
        f'PREDICT {agg}(orders.qty, 0, 1, months) FOR EACH users.user_id',
        MONTH_END,
        ANCHOR,
    )
    case(
        f'{agg.lower()}-decimal',
        f'{REGRESSION_EXIT}; probe: {agg} over DECIMAL(38,18)',
        f'PREDICT {agg}(orders.amount, 0, 30, days) FOR EACH users.user_id',
        DECIMALS,
        MARCH,
    )
    case(
        f'{agg.lower()}-kahan',
        f'{REGRESSION_EXIT}; probe: float32 Kahan sums',
        f'PREDICT {agg}(orders.price, 0, 30, days) FOR EACH users.user_id',
        KAHAN,
        MARCH,
    )

case(
    'sum-decimal-midpoint',
    f'{REGRESSION_EXIT}; probe: SUM over DECIMAL(38,18) at a float32 midpoint',
    'PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.user_id',
    MIDPOINT,
    MARCH,
)
case(
    'sum-offset-window-months',
    f'{MONTH_END_EXIT}; a window that starts after the anchor',
    'PREDICT SUM(orders.price, 1, 2, months) FOR EACH users.user_id',
    MONTH_END
    + [
        (1, 256.0, 1, '1', '2024-04-29 00:00:00'),
        (2, 512.0, 1, '1', '2024-04-30 00:00:00'),
    ],
    '2024-05-31',
)
case(
    'sum-twelve-months-leap-anchor',
    f'{MONTH_END_EXIT}; twelve months back from a leap day',
    'PREDICT SUM(orders.price, 0, 12, months) FOR EACH users.user_id',
    LEAP_ANCHOR,
    '2025-02-28',
)
case(
    'sum-month-from-jan-31',
    f'{MONTH_END_EXIT}; Jan 31 + 1 month',
    'PREDICT SUM(orders.price, 0, 1, months) FOR EACH users.user_id',
    [
        (1, 1.0, 1, '1', '2024-02-29 00:00:00'),
        (1, 2.0, 1, '1', '2024-02-29 00:00:00.000001'),
        (2, 4.0, 1, '1', '2024-03-01 00:00:00'),
        (3, 8.0, 1, '1', '2024-01-31 00:00:00'),
        (4, 16.0, 1, '1', '2024-01-01 00:00:00'),
    ],
    '2024-02-29',
)
case(
    'count-star-weeks',
    f'{REGRESSION_EXIT}; COUNT(*) over weeks',
    'PREDICT COUNT(orders.*, 0, 2, weeks) FOR EACH users.user_id',
    MONTH_END,
    ANCHOR,
)
case(
    'sum-all-null',
    f'{REGRESSION_EXIT}; probe: an all-null target column',
    'PREDICT SUM(orders.price, 0, 1, months) FOR EACH users.user_id',
    [(u, None, None, None, ts) for (u, _, _, _, ts) in MONTH_END],
    ANCHOR,
)
case(
    'avg-all-null',
    f'{REGRESSION_EXIT}; probe: an all-null target column, no label',
    'PREDICT AVG(orders.price, 0, 1, months) FOR EACH users.user_id',
    [(u, None, None, None, ts) for (u, _, _, _, ts) in MONTH_END],
    ANCHOR,
)

# Conditions: the binary task, the constant read at the target's type.
for name, query, rows in [
    ('count-gt-0', 'COUNT(orders.*, 0, 1, months) > 0', MONTH_END),
    ('count-ge-2', 'COUNT(orders.*, 0, 1, months) >= 2', MONTH_END),
    ('count-eq-1', 'COUNT(orders.price, 0, 1, months) = 1', MONTH_END),
    ('count-ne-1', 'COUNT(orders.*, 0, 1, months) != 1', MONTH_END),
    ('count-lt-1', 'COUNT(orders.*, 0, 1, months) < 1', MONTH_END),
    ('count-le-1', 'COUNT(orders.*, 0, 1, months) <= 1', MONTH_END),
    ('sum-eq-float', 'SUM(orders.price, 0, 30, days) = 0.3', KAHAN),
    ('max-eq-float', 'MAX(orders.price, 0, 30, days) = 0.1', KAHAN),
    ('sum-gt-float', 'SUM(orders.price, 0, 30, days) > 16777224', KAHAN),
    ('max-int-gt-float', 'MAX(orders.qty, 0, 1, months) > 7.5', MONTH_END),
    ('avg-eq-float', 'AVG(orders.price, 0, 30, days) = 1.0000001', KAHAN),
    ('sum-decimal-gt', 'SUM(orders.amount, 0, 30, days) > 0.3', DECIMALS),
]:
    case(
        f'binary-{name}',
        f'{CLASSIFICATION_EXIT}; probe: conditions ({query})',
        f'PREDICT {query} FOR EACH users.user_id',
        rows,
        MARCH if rows is not MONTH_END else ANCHOR,
    )

SEGMENTS = [
    "'b'",
    "'a'",
    "'c'",
    "'a'",
    "'B'",
    "'10'",
    "'2'",
    "'b'",
    "'z'",
    "'a'",
    "'é'",
    "''",
]
case(
    'multiclass-static-varchar',
    f'{CLASSIFICATION_EXIT}; probe: a static multiclass target',
    'PREDICT users.segment FOR EACH users.user_id',
    MONTH_END,
    ANCHOR,
    users=12,
    indices=[1, 2],
    segment=SEGMENTS,
)
case(
    'binary-static-boolean',
    f'{CLASSIFICATION_EXIT}; probe: a static binary target',
    'PREDICT users.active FOR EACH users.user_id',
    MONTH_END,
    ANCHOR,
    users=6,
    indices=[1, 2],
    active=['true', 'false', 'NULL', 'true', 'false', 'true'],
)
case(
    'multiclass-static-decimal',
    "the map's decimal.class-target-38-18 static target",
    'PREDICT users.tier FOR EACH users.user_id',
    MONTH_END,
    ANCHOR,
    users=6,
    indices=[1, 2],
    tier=[
        "CAST('1.5' AS DECIMAL(38,18))",
        "CAST('2' AS DECIMAL(38,18))",
        'NULL',
        "CAST('1.500000000000000001' AS DECIMAL(38,18))",
        "CAST('2' AS DECIMAL(38,18))",
        "CAST('-3' AS DECIMAL(38,18))",
    ],
)
case(
    'regression-static-double',
    "the map's materialize static_target",
    'PREDICT users.score FOR EACH users.user_id',
    MONTH_END,
    ANCHOR,
    users=6,
    indices=[1, 2],
    score=['0.1', '16777217.0', 'NULL', '-0.0', '1e-40', '3.4e38'],
)
case(
    'binary-static-condition',
    f'{CLASSIFICATION_EXIT}; a condition on a static column',
    'PREDICT users.score > 0.1 FOR EACH users.user_id',
    MONTH_END,
    ANCHOR,
    users=6,
    indices=[1, 2],
    score=['0.1', '0.2', 'NULL', '0.1000001', '0.0999999', '1'],
)


def _record(spec: dict[str, Any]) -> dict[str, Any]:
    connection = duckdb.connect()
    for statement in spec['setup']:
        connection.execute(statement)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(DuckDBSource(connection))
            request = PredictiveQuery(spec['query'], schema).materialize(
                indices=spec['indices'],
                anchor_time=pd.Timestamp(spec['anchor']),
            )
    except Exception as error:  # noqa: BLE001 - the outcome is recorded
        return {
            'outcome': 'refused',
            'class': type(error).__name__,
            'code': getattr(error, 'code', None),
        }
    payload = request._payloads[0]
    table = payload['context']['instance_table']
    return {
        'outcome': 'labels',
        'task': payload['task'],
        'columns': table['columns'],
        'rows': json.loads(json.dumps(table['rows'], default=str)),
    }


def main() -> None:
    reference = Path.home() / 'code/sdmc-reference'
    commit = subprocess.run(
        ['git', '-C', str(reference), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    out = {
        'recorded_by': 'labels_reference_record.py',
        'reference': commit,
        'cases': [{**spec, 'expected': _record(spec)} for spec in CASES],
    }
    OUT.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    print(f'{len(CASES)} cases written to {OUT}', file=sys.stderr)


if __name__ == '__main__':
    main()
