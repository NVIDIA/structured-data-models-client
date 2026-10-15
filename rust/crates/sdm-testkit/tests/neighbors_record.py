# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Records ``neighbors.json``, the reference's SQL sampler's two neighbor
reads on DuckDB (``_by_fkey`` and ``_by_time``, T1.26b), which
``neighbors.rs`` holds the core's ``neighbors`` requests to (register
ORD-2, decision F2).

Run once, by hand, in the reference's environment from the reference
checkout; CI never runs it, and the vectors are not recorded again to pass a
dependency bump (F4):

    python rust/crates/sdm-testkit/tests/neighbors_record.py OUT.json

The reads are the reference sampler's own, on a sampler its client builds
(``RelationalMaterializer._get_model()._sampler``) over a DuckDB source of
the tables ``setup`` makes, with its global zone ``UTC`` (the machine's zone
the reference's cursors read in) and the source connection's session zone
``America/New_York``, which no read sees (register DT-13). Each case gives
the seeds, and the rows the read returned: ``_by_fkey``'s in its order
(``ordered``), ``_by_time``'s as a multiset. A row is the seed's position
and each column asked for, a time as nanoseconds since the epoch in UTC
(the instant a zoned time is), ``null`` for a missing value.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import duckdb
import pandas as pd

SETUP = [
    'CREATE TABLE users (uid BIGINT PRIMARY KEY)',
    'INSERT INTO users SELECT i FROM range(1, 6) t(i)',
    'CREATE TABLE ev (eid BIGINT, uid BIGINT, ts TIMESTAMP, val DOUBLE)',
    # uid 1: a row at the anchor, a microsecond either side of it, two rows
    # tying on time (the key breaks the tie), a missing time, a time before
    # 1970.
    'INSERT INTO ev VALUES '
    "(1, 1, '2024-01-01 00:00:00', 1.0), "
    "(2, 1, '2023-12-31 23:59:59.999999', 2.0), "
    "(3, 1, '2024-01-01 00:00:00.000001', 3.0), "
    "(7, 1, '2023-12-30 00:00:00', 4.0), "
    "(5, 1, '2023-12-30 00:00:00', 5.0), "
    '(6, 1, NULL, 6.0), '
    "(8, 1, '1960-06-15 12:00:00', 7.0), "
    "(9, 1, '2024-01-02 00:00:00', 8.0), "
    "(10, 1, '2024-01-02 00:00:00.000001', 9.0)",
    # uid 2: more rows than a seed keeps, ties at the cut.
    "INSERT INTO ev SELECT 100 + i, 2, TIMESTAMP '2023-06-01' + to_hours((i // 2)::BIGINT), i::DOUBLE FROM range(9) t(i)",
    # uid 4: times before 1970 only.
    "INSERT INTO ev VALUES (200, 4, '1969-12-31 23:59:59.5', 10.0), (201, 4, '1900-01-01 00:00:00', 11.0)",
    'CREATE TABLE evtz (eid BIGINT, uid BIGINT, ts TIMESTAMPTZ, val DOUBLE)',
    # 03:00Z is 22:00 the day before in New York: after a 02:00 anchor in
    # UTC's wall clock, before it in New York's.
    'INSERT INTO evtz VALUES '
    "(1, 1, '2024-01-01 03:00:00+00', 1.0), "
    "(2, 1, '2024-01-01 01:00:00+00', 2.0), "
    "(3, 1, '2024-01-01 02:00:00+00', 3.0), "
    "(4, 2, '2024-07-01 12:00:00+02', 4.0)",
]
ZONES = ["SET GLOBAL TimeZone='UTC'", "SET TimeZone='America/New_York'"]
COLUMNS = ['eid', 'uid', 'ts', 'val']


def ns(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert('UTC').tz_localize(None)
    return int(stamp.as_unit('ns').value)


def cell(column: str, value: object) -> object:
    if column == 'ts':
        return ns(value)
    if value is None or pd.isna(value):
        return None
    return float(value) if column == 'val' else int(value)


def rows(df: pd.DataFrame, batch: object) -> list[list[object]]:
    return [
        [int(b)] + [cell(c, df[c].iloc[i]) for c in COLUMNS]
        for i, b in enumerate(batch)
    ]


def sampler(connection: duckdb.DuckDBPyConnection) -> object:
    from sdm_client import Relationship, Schema
    from sdm_client.materializers.relational import RelationalMaterializer
    from sdm_connectors import DuckDBSource

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(
            DuckDBSource(connection),
            primary_keys={'users': 'uid', 'ev': 'eid', 'evtz': 'eid'},
            time_columns={'ev': 'ts', 'evtz': 'ts'},
            relationships=[
                Relationship('ev', 'uid', 'users', 'uid'),
                Relationship('evtz', 'uid', 'users', 'uid'),
            ],
            check_keys=False,
        )
        return RelationalMaterializer(schema)._get_model()._sampler


def main() -> None:
    connection = duckdb.connect()
    for statement in SETUP + ZONES:
        connection.execute(statement)
    read = sampler(connection)
    cases = []

    def by_fkey(
        name: str, table: str, seeds: list[tuple[int, str]], keep: int
    ) -> None:
        keys = pd.Series([k for k, _ in seeds], dtype='int64')
        anchors = pd.Series(
            [pd.Timestamp(a) for _, a in seeds], dtype='datetime64[ns]'
        )
        df, batch = read._by_fkey(
            table, 'uid', keys, keep, anchors, set(COLUMNS)
        )
        cases.append(
            {
                'id': name,
                'form': 'fkey',
                'table': table,
                'per_seed': keep,
                'seeds': [
                    [k, ns(a)] for k, a in zip(keys, anchors, strict=True)
                ],
                'ordered': True,
                'rows': rows(df, batch),
            }
        )

    def by_time(
        name: str,
        table: str,
        seeds: list[tuple[int, str]],
        start: int | None,
        end: int,
    ) -> None:
        keys = pd.Series([k for k, _ in seeds], dtype='int64')
        anchors = pd.Series(
            [pd.Timestamp(a) for _, a in seeds], dtype='datetime64[ns]'
        )
        low = None if start is None else pd.DateOffset(days=start)
        high = pd.DateOffset(days=end)
        df, batch = read._by_time(
            table, 'uid', keys, anchors, low, high, set(COLUMNS)
        )
        cases.append(
            {
                'id': name,
                'form': 'time',
                'table': table,
                'seeds': [
                    [
                        k,
                        None if low is None else ns(a + low),
                        ns(a + high),
                    ]
                    for k, a in zip(keys, anchors, strict=True)
                ],
                'ordered': False,
                'rows': rows(df, batch),
            }
        )

    by_fkey('fkey/anchor-equality', 'ev', [(1, '2024-01-01 00:00:00')], 32)
    by_fkey('fkey/sub-second', 'ev', [(1, '2023-12-31 23:59:59.999999')], 32)
    by_fkey('fkey/ties-at-the-cut', 'ev', [(1, '2023-12-31 00:00:00')], 1)
    by_fkey('fkey/ties-kept-by-key', 'ev', [(2, '2024-01-01')], 3)
    by_fkey('fkey/negative', 'ev', [(1, '1961-01-01'), (4, '1970-01-01')], 32)
    by_fkey(
        'fkey/several-seeds-one-neighbor',
        'ev',
        [
            (1, '2024-01-03'),
            (1, '2024-01-01 00:00:00.000001'),
            (2, '2023-06-01 02:00'),
            (1, '2024-01-03'),
        ],
        2,
    )
    by_fkey('fkey/no-row', 'ev', [(3, '2024-01-01'), (5, '2024-01-01')], 32)
    by_fkey(
        'fkey/timestamptz',
        'evtz',
        [(1, '2024-01-01 02:00:00'), (2, '2024-07-01 10:00:00')],
        32,
    )
    by_time(
        'time/window-ends',
        'ev',
        [(1, '2024-01-01'), (1, '2023-12-31 00:00:00')],
        0,
        1,
    )
    by_time('time/sub-second', 'ev', [(1, '2023-12-31 23:59:59.999999')], 0, 1)
    by_time(
        'time/no-start', 'ev', [(1, '2024-01-01'), (4, '2000-01-01')], None, 0
    )
    by_time(
        'time/negative', 'ev', [(4, '1969-12-31'), (1, '1960-06-15')], -1, 1
    )
    by_time(
        'time/several-seeds-one-neighbor',
        'ev',
        [
            (1, '2023-12-31'),
            (1, '2023-12-31 12:00'),
            (2, '2023-06-01'),
            (1, '2023-12-31'),
        ],
        -2,
        2,
    )
    by_time(
        'time/timestamptz',
        'evtz',
        [(1, '2023-12-31 02:00:00'), (2, '2024-07-01')],
        0,
        1,
    )
    document = {
        'description': (
            "The reference's SQL sampler's neighbor reads on DuckDB "
            '(_by_fkey, _by_time), recorded by neighbors_record.py'
        ),
        'environment': {
            'duckdb': duckdb.__version__,
            'pandas': pd.__version__,
        },
        'setup': SETUP,
        'zones': ZONES,
        'columns': COLUMNS,
        'cases': cases,
    }
    Path(sys.argv[1]).write_text(
        json.dumps(document, indent=1) + '\n', encoding='utf-8'
    )


if __name__ == '__main__':
    main()
