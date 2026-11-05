#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type map scenarios: independent inputs and current-client test helpers."""

from __future__ import annotations

import platform
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/type-map-vectors.json'

#: (DuckDB type, literal): the canonical-text inputs.
CANONICAL = [
    ('DECIMAL(10,2)', '100.00'),
    ('DECIMAL(10,0)', '100'),
    ('DECIMAL(12,4)', '100.5000'),
    ('DECIMAL(10,2)', '100.50'),
    ('DECIMAL(10,2)', '-0.00'),
    ('DECIMAL(10,0)', '0'),
    ('DECIMAL(12,4)', '0.0000'),
    ('DECIMAL(10,2)', '-1.50'),
    ('DECIMAL(10,2)', '0.10'),
    ('DECIMAL(18,10)', '-0.0000000001'),
    ('DECIMAL(38,38)', '0.00000000000000000000000000000000000001'),
    ('DECIMAL(38,0)', '99999999999999999999999999999999999999'),
    ('DECIMAL(38,0)', '-99999999999999999999999999999999999999'),
    ('DECIMAL(38,18)', '12345678901234567890.123456789012345678'),
    ('DECIMAL(38,10)', '1000000.0000000000'),
    ('DECIMAL(18,8)', '1234567890.12345601'),
    ('DECIMAL(18,8)', '1234567890.12345602'),
    ('DECIMAL(4,0)', '-9999'),
]
#: The pair SQLite's numeric affinity would make one value.
DISTINCT = ('DECIMAL(18,8)', ['1234567890.12345601', '1234567890.12345602'])
#: Equal keys at two scales, each pair one value.
LEFT = ('DECIMAL(10,2)', ['100.50', '-0.00', '7.00', '12345678.90', '-3.10'])
RIGHT = (
    'DECIMAL(12,4)',
    ['100.5000', '0.0000', '7.0000', '12345678.9000', '-3.1000'],
)


def canonical(value: Decimal) -> str:
    r"""The design's canonical text of a decimal."""
    if value == 0:
        return '0'
    text = format(value, 'f')
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return text


def _read(duck: Any, duck_type: str, literal: str) -> tuple[Decimal, str]:
    table = duck.sql(
        f"SELECT CAST('{literal}' AS {duck_type}) AS v, "
        f"CAST(CAST('{literal}' AS {duck_type}) AS VARCHAR) AS t"
    ).to_arrow_table()
    [value] = table.column('v').to_pylist()
    [text] = table.column('t').to_pylist()
    return value, text


def _common_prefix(a: str, b: str) -> int:
    count = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        count += 1
    return count


def _sqlite_distinct(declared: str, keys: list[str]) -> dict[str, Any]:
    connection = sqlite3.connect(':memory:')
    connection.execute(f'CREATE TABLE t (k {declared})')
    connection.executemany('INSERT INTO t VALUES (?)', [(k,) for k in keys])
    stored = connection.execute('SELECT typeof(k) FROM t').fetchall()
    [(distinct,)] = connection.execute('SELECT count(DISTINCT k) FROM t')
    connection.close()
    return {
        'declared': declared,
        'storage_classes': sorted({s for (s,) in stored}),
        'distinct': distinct,
    }


def record() -> dict[str, Any]:
    import duckdb
    import pyarrow

    duck = duckdb.connect()
    vectors = []
    for duck_type, literal in CANONICAL:
        value, text = _read(duck, duck_type, literal)
        vectors.append(
            {
                'duckdb_type': duck_type,
                'literal': literal,
                'canonical': canonical(value),
                'duckdb_text': text,
                'python_str': str(value),
            }
        )
    by_text: dict[str, set[str]] = {}
    for vector in vectors:
        by_text.setdefault(vector['canonical'], set()).add(vector['literal'])

    duck_type, keys = DISTINCT
    texts = [canonical(_read(duck, duck_type, k)[0]) for k in keys]
    assert len(set(texts)) == 2, texts
    distinct = {
        'duckdb_type': duck_type,
        'keys': keys,
        'canonical': texts,
        'common_significant_digits': _common_prefix(
            keys[0].replace('.', ''), keys[1].replace('.', '')
        ),
        'sqlite_probe': [
            _sqlite_distinct('DECIMAL(18,8)', keys),
            _sqlite_distinct('TEXT', texts),
        ],
    }

    duck.execute(f'CREATE TABLE a (k {LEFT[0]})')
    duck.execute(f'CREATE TABLE b (k {RIGHT[0]})')
    duck.executemany('INSERT INTO a VALUES (?)', [[k] for k in LEFT[1]])
    duck.executemany('INSERT INTO b VALUES (?)', [[k] for k in RIGHT[1]])
    [(joined,)] = duck.sql(
        'SELECT count(*) FROM a JOIN b ON a.k = b.k'
    ).fetchall()
    left_texts = [canonical(_read(duck, LEFT[0], k)[0]) for k in LEFT[1]]
    right_texts = [canonical(_read(duck, RIGHT[0], k)[0]) for k in RIGHT[1]]
    join = {
        'left': {
            'duckdb_type': LEFT[0],
            'keys': LEFT[1],
            'canonical': left_texts,
        },
        'right': {
            'duckdb_type': RIGHT[0],
            'keys': RIGHT[1],
            'canonical': right_texts,
        },
        'duckdb_joined': joined,
    }
    return {
        'format': 'sdm-type-map-vectors/1',
        'about': 'Exact decimal text and cross-scale key equality, including SQLite TEXT storage to avoid numeric-affinity rounding.',
        'map': {
            'decimal': {
                'duckdb': 'DECIMAL(p,s)',
                'sqlite': 'TEXT',
                'sqlite_value': 'canonical',
                'postgres': 'numeric(p,s)',
            }
        },
        'canonical_text': vectors,
        'one_text': sorted(
            (
                {'canonical': t, 'literals': sorted(ls)}
                for t, ls in by_text.items()
                if len(ls) > 1
            ),
            key=lambda item: item['canonical'],
        ),
        'sqlite_distinct_keys': distinct,
        'join_across_scales': join,
        'recorded': {
            'by': 'contract/support/type_map_scenarios.py',
            'environment': {
                'python': platform.python_version(),
                'duckdb': duckdb.__version__,
                'pyarrow': pyarrow.__version__,
                'sqlite': sqlite3.sqlite_version,
            },
        },
    }
