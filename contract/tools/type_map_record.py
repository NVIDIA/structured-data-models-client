#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the decimal part of the per-engine type map into
``contract/type-map-vectors.json`` (T2.1a's vectors, design choice 2).

Run from this repository's root with any interpreter that has DuckDB and
pyarrow (the stdlib ``sqlite3`` is the SQLite probed)::

    python -I contract/tools/type_map_record.py

Written by T2.1a's independent test author before the implementation. The
inputs are written here; every value the file states is computed:

- ``canonical_text``: each DuckDB decimal literal of its type, read from
  DuckDB through Arrow as the loader reads it (``to_pylist``, a
  ``decimal.Decimal`` at the column's scale), and its canonical text by the
  design's one definition: plain positional notation, no exponent, no
  trailing fractional zeros, no trailing point, ``-0`` written ``0``.
  DuckDB's own text (``CAST(... AS VARCHAR)``) and Python's ``str`` are
  recorded beside it, so a vector whose canonical text equals both proves
  nothing more than they do (``100.00`` and ``1E-38`` show where they
  differ). Equal values of different scales have one text: ``100.00`` and
  ``100``, ``-0.00`` and ``0``.
- ``sqlite_distinct_keys``: two keys that agree in their first 15
  significant digits (and more) and differ after, as a DuckDB decimal.
  SQLite gives a column declared ``DECIMAL`` numeric affinity, which stores
  the text as a REAL; the probe stores the pair in such a column and in a
  ``TEXT`` column and records how many distinct values each kept, so the
  vector is shown to be one the declared type would lose (KO2-35).
- ``join_across_scales``: equal keys held as ``DECIMAL(10,2)`` on one table
  and ``DECIMAL(12,4)`` on the other (lesson 19); DuckDB's own join of the
  two is recorded, the count every loaded engine must match.
- ``map``: the decimal's stored type per engine, as the design states it:
  SQLite ``TEXT`` holding the canonical text, Postgres ``numeric`` with the
  column's precision and scale.

The rest of the type map (every other type of the type set, per engine) is
the implementer's, added to the same file under review.
"""

from __future__ import annotations

import json
import platform
import sqlite3
import sys
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
        'about': (
            "T2.1a's type map (design, choice 2), written by the package's "
            'independent test author before the implementation, recorded by '
            'contract/tools/type_map_record.py: the one definition of a '
            "decimal's canonical text (plain positional notation, no "
            'exponent, no trailing fractional zeros, no trailing point, -0 '
            'written 0), which the Python loader and the Rust key binders '
            'each implement and both run; a SQLite decimal stored in a column '
            'declared TEXT as that text, since a column declared DECIMAL '
            'takes numeric affinity and rounds what it stores '
            '(sqlite_distinct_keys, KO2-35); and equal keys of different '
            'scales that must still join on every engine (join_across_scales, '
            'lesson 19). `map` is the stored type per engine; the rest of the '
            "type set's entries are the implementer's, added here under review."
        ),
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
            'by': 'contract/tools/type_map_record.py',
            'environment': {
                'python': platform.python_version(),
                'duckdb': duckdb.__version__,
                'pyarrow': pyarrow.__version__,
                'sqlite': sqlite3.sqlite_version,
            },
        },
    }


def main() -> int:
    document = record()
    OUT.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    print(f'{OUT.relative_to(ROOT)}: {len(document["canonical_text"])} vectors')
    return 0


if __name__ == '__main__':
    sys.exit(main())
