# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P55.caps, Snowflake: a seeded block sample ahead of the full scans.

``DESIGN-scan-caps.md`` A6-A8 for Snowflake, through the public API over the
stub driver (``_warehouse_stub.py``), whose ``SHOW TABLES`` lists each table
with the ``rows`` and ``bytes`` a check gives it.

The fraction is ``f = max(T / rows, 128 / max(1, bytes / 16 MiB))``, ``T``
100,000 for the inference sample (DV4) and ``max(100,000, 10 n)`` for a
seeded entity draw of ``n`` rows (DV5), rounded up to a power of two; the
prefilter applies to a listed standard table when that is at most 1/8, and
its literal is ``repr(100 f)``:

* A6: DV4 reads ``FROM t SAMPLE SYSTEM (p) SEED (0)``, the rest of today's
  statement unchanged; small, unlisted (a view), hybrid, external, Iceberg
  and dynamic tables keep today's statement; a prefiltered read with fewer
  than 1,000 rows is read again with today's statement;
* A7: DV5 reads ``FROM t SAMPLE SYSTEM (p) SEED (s)`` before ``WHERE``, ``s``
  the draw's seed modulo 2^31; it falls back below ``n`` rows; every other
  statement is today's;
* A8: the boundary at 1/8, the partition floor, one literal per power-of-two
  bucket, and monotonic in the table's size.

Today's statements come from running ``BASE``'s code in a child process.
"""

from __future__ import annotations

import json
import math
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from _fixes import assert_imports_base, base_env, run_child

TIB = 1 << 40
GIB = 1 << 30
MIB = 1 << 20
ORDERS = '"DB"."SHOP"."ORDERS"'
USERS = '"DB"."SHOP"."USERS"'


@pytest.fixture(scope='module')
def base_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp('base')
    assert_imports_base(base_env(root), root)
    return root


_cache: dict[tuple[str, str], Any] = {}


def _run(config: dict[str, Any], base_root: Path | None = None) -> Any:
    text = json.dumps(config, sort_keys=True)
    key = ('base' if base_root else 'head', text)
    if key not in _cache:
        env = base_env(base_root) if base_root else None
        _cache[key] = run_child('warehouse', 'snowflake', '-', text, env=env)
    return _cache[key]


def _listing(rows: int, bytes_: int, **flags: str) -> dict[str, Any]:
    import _warehouse_stub as stub

    return stub.listing(rows, bytes_, **flags)


def _samples(result: Any, table: str) -> list[str]:
    return [
        s['sql']
        for s in result['statements']
        if '__kumo_order_' in s['sql'] and f'FROM {table}' in s['sql']
    ]


def _today(base_root: Path, table: str) -> str:
    found = _samples(_run({'questions': []}, base_root), table)
    assert len(found) == 1, found
    return found[0]


def _power_of_two_at_or_above(value: Fraction) -> Fraction:
    exponent = math.ceil(math.log2(value)) if value > 0 else 0
    while Fraction(2) ** (exponent - 1) >= value:
        exponent -= 1
    while Fraction(2) ** exponent < value:
        exponent += 1
    return Fraction(2) ** exponent


def _fraction(rows: int, bytes_: int, target: int = 100_000) -> Fraction:
    partitions = max(Fraction(1), Fraction(bytes_, 16 * MIB))
    return _power_of_two_at_or_above(
        max(Fraction(target, rows), Fraction(128) / partitions)
    )


def _literal(fraction: Fraction) -> str:
    return repr(100 * float(fraction))


def _with_sample(sql: str, table: str, clause: str) -> str:
    r"""Today's statement with ``clause`` right after its table, on one
    line: where the clause goes is checked, not its whitespace."""
    head, sep, tail = sql.partition(f'FROM {table}')
    assert sep, sql
    return _flat(f'{head}FROM {table} {clause}{tail}')


def _flat(sql: str) -> str:
    return ' '.join(sql.split())


def _matches(sent: list[str], want: list[str]) -> bool:
    r"""Equal, comparing a prefiltered statement on one line and any other
    byte for byte."""
    if len(sent) != len(want):
        return False
    for got, expected in zip(sent, want, strict=True):
        if 'SAMPLE SYSTEM' in expected:
            if _flat(got) != expected:
                return False
        elif got != expected:
            return False
    return True


# A6: the inference sample ####################################################


def test_a6_a_large_table_reads_a_seeded_block_sample(base_root: Path) -> None:
    config = {
        'questions': [],
        'listings': {'orders': _listing(10_000_000, TIB)},
        'prefiltered_rows': 1000,
    }
    sent = _samples(_run(config), ORDERS)
    want = _with_sample(
        _today(base_root, ORDERS), ORDERS, 'SAMPLE SYSTEM (1.5625) SEED (0)'
    )
    assert _matches(sent, [want]), sent
    # The other table, listed small, is read as today.
    assert _samples(_run(config), USERS) == [_today(base_root, USERS)]


def test_a6_too_few_rows_fall_back_to_todays_statement(
    base_root: Path,
) -> None:
    config = {
        'questions': [],
        'listings': {'orders': _listing(10_000_000, TIB)},
        'prefiltered_rows': 999,
    }
    today = _today(base_root, ORDERS)
    sent = _samples(_run(config), ORDERS)
    assert _matches(
        sent,
        [_with_sample(today, ORDERS, 'SAMPLE SYSTEM (1.5625) SEED (0)'), today],
    ), sent


@pytest.mark.parametrize(
    'listed',
    [
        pytest.param({'rows': 60, 'bytes_': MIB}, id='small'),
        pytest.param(None, id='view'),
        pytest.param(
            {'rows': 10**9, 'bytes_': TIB, 'is_hybrid': 'Y'}, id='hybrid'
        ),
        pytest.param(
            {'rows': 10**9, 'bytes_': TIB, 'is_external': 'Y'}, id='external'
        ),
        pytest.param(
            {'rows': 10**9, 'bytes_': TIB, 'is_iceberg': 'Y'}, id='iceberg'
        ),
        pytest.param(
            {'rows': 10**9, 'bytes_': TIB, 'is_dynamic': 'Y'}, id='dynamic'
        ),
    ],
)
def test_a6_ineligible_tables_keep_todays_statement(
    base_root: Path, listed: dict[str, Any] | None
) -> None:
    config = {
        'questions': [],
        'listings': {'orders': None if listed is None else _listing(**listed)},
        'prefiltered_rows': 1000,
    }
    assert _samples(_run(config), ORDERS) == [_today(base_root, ORDERS)]


# A7: the seeded entity draw ##################################################


def _draws(result: Any) -> list[str]:
    return [
        s['sql']
        for s in result['statements']
        if 'MD5_NUMBER_LOWER64' in s['sql'] and f'FROM {USERS}' in s['sql']
    ]


def _without_listings(result: Any) -> list[Any]:
    return [s for s in result['statements'] if not s['sql'].startswith('SHOW')]


def test_a7_a_large_entity_table_draws_from_a_block_sample(
    base_root: Path,
) -> None:
    base_config = {'questions': ['count'], 'nullable_keys': True}
    today = _run(base_config, base_root)
    config = dict(base_config, listings={'users': _listing(10_000_000, TIB)})
    head = _run(config)

    draws = _draws(today)
    assert len(draws) == 1, draws
    draw = draws[0]
    assert '\nWHERE' in draw, draw
    seed = int(re.search(r"MD5_NUMBER_LOWER64\('(\d+):", draw).group(1))
    n = int(re.search(r'LIMIT (\d+)$', draw).group(1))
    literal = _literal(_fraction(10_000_000, TIB, target=max(100_000, 10 * n)))
    clause = f'SAMPLE SYSTEM ({literal}) SEED ({seed % 2147483648})'
    prefiltered = _with_sample(draw, USERS, clause)
    assert prefiltered.index('SAMPLE SYSTEM') < prefiltered.index(' WHERE ')

    # Twelve users answer the prefiltered draw, fewer than it needs: it is
    # drawn again with today's statement. The inference sample of USERS,
    # also large, falls back the same way; every other statement is today's.
    expected = []
    for statement in _without_listings(today):
        sql = statement['sql']
        if sql == draw:
            expected.append({'sql': prefiltered, 'params': statement['params']})
        elif '__kumo_order_' in sql and f'FROM {USERS}' in sql:
            expected.append(
                {
                    'sql': _with_sample(
                        sql, USERS, 'SAMPLE SYSTEM (1.5625) SEED (0)'
                    ),
                    'params': statement['params'],
                }
            )
        expected.append(statement)
    got = _without_listings(head)
    assert [s['params'] for s in got] == [s['params'] for s in expected]
    assert _matches([s['sql'] for s in got], [s['sql'] for s in expected]), [
        s['sql'] for s in got
    ]
    assert head['payloads'] == today['payloads']


def test_a7_small_entity_tables_draw_as_today(base_root: Path) -> None:
    config = {'questions': ['count', 'sum'], 'nullable_keys': True}
    assert _without_listings(_run(config)) == _without_listings(
        _run(config, base_root)
    )


def test_a7_a_draw_without_a_seed_or_of_given_ids_is_unchanged(
    base_root: Path,
) -> None:
    r"""Only the seeded draw is prefiltered: ``SAMPLE ROW`` and the reads of
    given ids stay as they are, however large the table."""
    config = {
        'questions': ['count', 'sum'],
        'listings': {'users': _listing(10_000_000, TIB)},
    }
    head = _run(config)
    for statement in head['statements']:
        sql = statement['sql']
        if 'SAMPLE ROW' in sql or ' IN (' in sql or 'FLATTEN' in sql:
            assert 'SAMPLE SYSTEM' not in sql, sql


# A8: the fraction ############################################################


@pytest.mark.parametrize(
    ('rows', 'bytes_', 'literal'),
    [
        (800_000, TIB, '12.5'),
        (799_999, TIB, None),
        (1_600_000, TIB, '6.25'),
        (3_199_999, TIB, '6.25'),
        (3_200_000, TIB, '3.125'),
        (10_000_000, TIB, '1.5625'),
        (100_000_000, TIB, '0.1953125'),
        (10**9, 4 * GIB, None),
        (10**9, 64 * GIB, '3.125'),
        (10**9, 16 * GIB, '12.5'),
        (60, MIB, None),
    ],
)
def test_a8_the_fraction_and_its_literal(
    base_root: Path, rows: int, bytes_: int, literal: str | None
) -> None:
    fraction = _fraction(rows, bytes_)
    assert (fraction <= Fraction(1, 8)) == (literal is not None)
    if literal is not None:
        assert _literal(fraction) == literal
    config = {
        'questions': [],
        'listings': {'orders': _listing(rows, bytes_)},
        'prefiltered_rows': 1000,
    }
    today = _today(base_root, ORDERS)
    want = (
        today
        if literal is None
        else _with_sample(today, ORDERS, f'SAMPLE SYSTEM ({literal}) SEED (0)')
    )
    sent = _samples(_run(config), ORDERS)
    assert _matches(sent, [want]), sent


def test_a8_the_fraction_falls_as_the_table_grows() -> None:
    sizes = [10**k for k in range(5, 13)]
    fractions = [_fraction(rows, TIB) for rows in sizes]
    assert fractions == sorted(fractions, reverse=True)
