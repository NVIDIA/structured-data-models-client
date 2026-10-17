#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference's handling of DuckDB decimals into
``contract/decimal-vectors.json`` (T1.32b's vectors).

Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root::

    ~/code/sdmc-reference/.venv/bin/python -I contract/tools/decimal_record.py

Written by T1.32b's independent test author before the implementation.
This is ingestion, not model quality, so the data is generated: each vector
is ``contract/bases.yaml``'s ``shop`` changed by its ``setup`` (DuckDB
statements) and asked ``decimal.feature-38-0-wide``'s question unless it
names its own ``pql``, for its own ``indices`` (default ``[0, 1]``) at its
own ``anchor_time`` (default ``2024-03-01``; null leaves it out, as a static
question's case does):

- ``feature.*``: a column ``f`` of DECIMAL(p, s) for p in 4, 9, 18, 19 and
  38 and s in 0, 2, 10 and p (where s <= p), on the entity table and on the
  fact table, holding small values of both signs, zero and a null
  (``spread``), or values at the declared precision (``max``: the largest
  and smallest the type holds and their neighbours); and, for five types,
  every value zero, every value negative, or every value null (the
  DECIMAL(18, 2) all-null column is T1.32d's ``null.decimal-18-2``, which
  T1.32b clears: it is not recorded again here);
- ``wide.*``: what DuckDB holds past 38 digits, HUGEINT and UHUGEINT at
  their ends, which DuckDB hands over as ``decimal128(38, 0)`` (DuckDB
  produces no ``decimal256``: ``duckdb.arrow`` records the Arrow type it
  gives for each vector's column, so a DuckDB that does is seen);
- ``target.*``: a regression target over a decimal column, ``SUM`` of the
  fact table's ``price`` held as each type (``spread`` and ``max`` values),
  and ``AVG``, ``MIN``, ``MAX`` and ``COUNT`` over four types;
- ``overflow.*``: a sum at and past the declared precision (DECIMAL(4, 0):
  9999 + 1, and its neighbours, signs and wider types, 38 digits
  included), two orders of one user in one label window;
- ``class.*``: classification targets: a static decimal target of three
  classes (as ``decimal.class-target-38-18``, at other types), a static
  decimal target of many values, and binary questions comparing a decimal
  sum or a decimal column with a number;
- ``key.*``: decimal keys beyond T1.32a's (``key.decimal-18-0`` and
  ``key.decimal-38-10``): other precisions, a scale above zero, the
  foreign key alone or the primary key alone decimal, the fact table's own
  key, negative keys, keys at the precision's end, and a null foreign key,
  the entities given as integers (the reference fails INTERNAL_ERROR on
  every decimal key it samples, so these are marked; the entities' cast is
  T1.32a's);
- ``time.*``: decimal columns beside the time column: an epoch in seconds
  or with a fraction, and a date as yyyymmdd digits, as features and
  declared as the time column, and a decimal in place of the time column.

The core is handed the entities and anchor time as the client's bridge
builds them (``core_indices``, T1.32a's ``host_arrow``; ``core_anchor``),
recorded beside each vector for ``rust-host``; and the statements the
reference ran per public call (``statements``, ``compare.py``'s counts, its
capture complete against DuckDB's query log). The document's ``cases``
holds the family's three cases' committed reference observations projected
the same way.

Each vector is a case in ``contract/cases``' format, written to a temporary
directory and run by ``contract/runners/run_python.py --mode python-ref``
through the public calls (``Schema.from_source``, ``materialize``,
``SDMClient.predict`` against the scripted endpoint). The reference's answer
is kept as ``contract/tools/empty_null_record.py``'s ``project`` reads it
(the outcome: result, stage, class and code, no message text, H17; the
public schema with each column's dtype and stype; the warning classes; each
request as the ``payload`` rule compares it, its digest and the rows and
null cells of each table it carries; the result), and beside it what this
family is about, for each of the vector's ``columns``:

- ``described``: the dtype and stype ``Schema.from_source`` gave it;
- ``request``: how it reached the model, in each part of each request
  (``context`` and ``predict``): the wire dtype and stype the request's
  schema declares for it, and its cells, each as its JSON text (a number
  ``{"number": "<digits>"}``, a text, or null) with how many times it
  appears.

A column is ``[table, column]``; the table ``instance_table`` is the
request's own (``TARGET``, the labels). And what DuckDB holds after the
setup: each named column's catalog type and the Arrow type DuckDB hands
over for it.

``register`` marks the vectors where the core is expected to differ. It is
derived from the reference's answer, never chosen (T1.32d's rule): the
reference fails with an unnamed exception (no ``StructuredDataError``
code) or with ``INTERNAL_ERROR`` (E66(b)'s class). The mark decides no
policy: any vector, marked or not, may differ from the reference only under
an approved register entry whose ``refs`` name it with exactly the
transition it makes, ``REGISTER_REF`` + ``<id or glob> <from> -> <to>``,
each side ``success`` or ``refusal <CODE>`` (the reference's unnamed
exception is ``refusal UNNAMED``).

Nothing here is written by hand but the inputs. ``VECTORS``, ``case_for``,
``run`` and ``project`` are imported by
``contract/tests/test_t132b_decimal_vectors.py``, so ``python-core`` is
observed and projected by the same code.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from decimal import Context, Decimal, localcontext
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/decimal-vectors.json'
TEMPLATE = ROOT / 'contract/cases/s1/decimal.feature-38-0-wide.json'
RUNNER = ROOT / 'contract/runners/run_python.py'
FORMAT = 'sdm-decimal-vectors/1'
#: How a register entry names the vectors it covers, in its ``refs``: this
#: prefix, a vector's id or a glob of ids (``fnmatch``), and the transition
#: it declares, for example
#: ``contract/decimal-vectors.json#class.static.* refusal INTERNAL_ERROR ->
#: refusal UNSUPPORTED_REQUEST``.
REGISTER_REF = 'contract/decimal-vectors.json#'
#: Each vector's case id is this prefix and the vector's id.
CASE_PREFIX = 't132b.'
#: The question every vector asks unless it names its own.
QUESTION = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
#: The entities every vector asks for unless it names its own.
INDICES = [0, 1]
#: The anchor time every vector gives unless it names its own (null: none).
ANCHOR = '2024-03-01'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


#: T1.32d's recorder: the public answer's projection (``project``), the
#: payload rule's digest, the outcome classes and the register marks are
#: its, so both families' vectors are read by one code.
E = _module('empty_null_record', ROOT / 'contract/tools/empty_null_record.py')

# --- the decimal types and their values --------------------------------------

#: The precisions and scales the brief names: p in 4, 9, 18, 19, 38 and s in
#: 0, 2, 10, p, wherever DuckDB takes s <= p.
TYPES: list[tuple[int, int]] = [
    (p, s) for p in (4, 9, 18, 19, 38) for s in sorted({0, 2, 10, p}) if s <= p
]


def sql_type(p: int, s: int) -> str:
    return f'DECIMAL({p}, {s})'


def tag(p: int, s: int) -> str:
    r"""The type in a vector id: ``d4-2``."""
    return f'd{p}-{s}'


def largest(p: int, s: int) -> Decimal:
    r"""The largest value DECIMAL(p, s) holds: p nines, s of them after the
    point."""
    return Decimal(f'{10**p - 1}E-{s}')


def ulp(s: int) -> Decimal:
    return Decimal(f'1E-{s}')


def text(value: Decimal | None, s: int) -> str | None:
    r"""A value written with exactly s digits after the point, from its
    digits (no context, no float)."""
    if value is None:
        return None
    sign, digits, exponent = value.as_tuple()
    assert isinstance(exponent, int)
    unscaled = int(''.join(map(str, digits)) or '0') * 10 ** (exponent + s)
    assert exponent + s >= 0, (value, s)
    whole, fraction = divmod(abs(unscaled), 10**s)
    body = f'{whole}.{fraction:0{s}d}' if s else f'{whole}'
    return f'-{body}' if sign and unscaled else body


def small(p: int, s: int) -> list[Decimal]:
    r"""Eight small values of both signs, zero among them, that the type
    holds: whole numbers where s is 0, fractions of one where s is p."""
    if s == 0:
        values = ['-3', '-1', '0', '1', '2', '5', '7', '12']
    elif s == p:
        values = ['-0.75', '-0.1', '0', '0.05', '0.25', '0.5', '0.01', '0.99']
    else:
        values = ['-2.5', '-1', '0', '0.25', '1', '2', '3.75', '1.01']
    return [Decimal(v) for v in values]


#: The value shapes: each user's (or order's, by ``order_id % 8``) value,
#: None a null.
def shape(name: str, p: int, s: int) -> list[Decimal | None]:
    with localcontext(Context(prec=200)):
        return _shape(name, p, s)


def _shape(name: str, p: int, s: int) -> list[Decimal | None]:
    top, one = largest(p, s), ulp(s)
    values: dict[str, list[Decimal | None]] = {
        'spread': [*small(p, s)[:7], None],
        'max': [top, -top, top - one, -(top - one), one, -one, Decimal(0), top],
        'zero': [Decimal(0)] * 8,
        'negative': [-abs(v) if v else -one for v in small(p, s)],
        'null': [None] * 8,
    }
    return values[name]


def literal(value: Decimal | None, p: int, s: int) -> str:
    r"""A value as DuckDB text cast to the type (never a float literal)."""
    if value is None:
        return f'NULL::{sql_type(p, s)}'
    return f"'{text(value, s)}'::{sql_type(p, s)}"


def by_row(key: str, values: list[Decimal | None], p: int, s: int) -> str:
    r"""A CASE over ``key % 8`` giving each row its value of ``values``."""
    whens = ' '.join(
        f'WHEN {i} THEN {literal(v, p, s)}' for i, v in enumerate(values)
    )
    return f'CASE {key} % 8 {whens} END'


def add_f(table: str, values: list[Decimal | None], p: int, s: int) -> str:
    key = 'user_id' if table == 'users' else 'order_id'
    return (
        f'CREATE OR REPLACE TABLE {table} AS SELECT *, '
        f'{by_row(key, values, p, s)} AS f FROM {table}'
    )


def price_as(values: list[Decimal | None], p: int, s: int) -> str:
    return (
        f'ALTER TABLE orders ALTER price TYPE {sql_type(p, s)} '
        f'USING {by_row("order_id", values, p, s)}'
    )


# --- the vectors ----------------------------------------------------------------


def _features() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for table in ('users', 'orders'):
        for name in ('spread', 'max'):
            for p, s in TYPES:
                out.append(
                    {
                        'id': f'feature.{table}.{tag(p, s)}.{name}',
                        'columns': [[table, 'f']],
                        'setup': [add_f(table, shape(name, p, s), p, s)],
                    }
                )
    for p, s in [(4, 0), (9, 2), (19, 10), (38, 0), (38, 38)]:
        for name in ('zero', 'negative', 'null'):
            out.append(
                {
                    'id': f'feature.users.{tag(p, s)}.{name}',
                    'columns': [['users', 'f']],
                    'setup': [add_f('users', shape(name, p, s), p, s)],
                }
            )
    return out


#: Past 38 digits: DuckDB's 128-bit integers at their ends.
HUGEINT_MAX = 2**127 - 1
UHUGEINT_MAX = 2**128 - 1


def _wide() -> list[dict[str, Any]]:
    def column(values: list[str], sql: str) -> list[str]:
        whens = ' '.join(
            f"WHEN {i} THEN '{v}'::{sql}" for i, v in enumerate(values)
        )
        return [
            'CREATE OR REPLACE TABLE users AS SELECT *, '
            f'CASE user_id % 8 {whens} END AS f FROM users'
        ]

    def eight(values: list[int]) -> list[str]:
        return [str(values[i % len(values)]) for i in range(8)]

    return [
        {
            'id': 'wide.hugeint-max',
            'columns': [['users', 'f']],
            'setup': column(
                eight([HUGEINT_MAX, HUGEINT_MAX - 1, 0, 1]), 'HUGEINT'
            ),
        },
        {
            'id': 'wide.hugeint-min',
            'columns': [['users', 'f']],
            'setup': column(
                eight([-HUGEINT_MAX - 1, -HUGEINT_MAX, 0, -1]), 'HUGEINT'
            ),
        },
        {
            'id': 'wide.hugeint-39-digits',
            'columns': [['users', 'f']],
            'setup': column(
                eight([10**38, -(10**38), 10**38 + 7, 2]), 'HUGEINT'
            ),
        },
        {
            'id': 'wide.uhugeint-max',
            'columns': [['users', 'f']],
            'setup': column(
                eight([UHUGEINT_MAX, UHUGEINT_MAX - 1, 0, 1]), 'UHUGEINT'
            ),
        },
        {
            'id': 'wide.uhugeint-past-hugeint',
            'columns': [['users', 'f']],
            'setup': column(eight([HUGEINT_MAX + 1, 10**38, 3, 4]), 'UHUGEINT'),
        },
        {
            'id': 'wide.hugeint-small',
            'columns': [['users', 'f']],
            'setup': column(eight([0, 1, -1, 12345]), 'HUGEINT'),
        },
    ]


TARGET = [['instance_table', 'TARGET'], ['orders', 'price']]


def _targets() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in ('spread', 'max'):
        for p, s in TYPES:
            out.append(
                {
                    'id': f'target.sum.{tag(p, s)}.{name}',
                    'columns': TARGET,
                    'setup': [price_as(shape(name, p, s), p, s)],
                }
            )
    for aggregation in ('AVG', 'MIN', 'MAX', 'COUNT'):
        for p, s in [(4, 2), (18, 0), (38, 10), (38, 38)]:
            out.append(
                {
                    'id': f'target.{aggregation.lower()}.{tag(p, s)}.spread',
                    'columns': TARGET,
                    'setup': [price_as(shape('spread', p, s), p, s)],
                    'pql': (
                        f'PREDICT {aggregation}(orders.price, 0, 30, days) '
                        'FOR EACH users.user_id'
                    ),
                }
            )
    return out


def _overflow() -> list[dict[str, Any]]:
    r"""Two orders of user 0 in the label window after the context's
    anchor 2024-01-31 (its own orders are 56 days apart, so the window
    holds these two alone), every other price zero: the window's sum is
    ``a + b`` (and ``+ c``, a third order, where given)."""

    def pair(
        p: int, s: int, terms: list[str], pql: str = QUESTION
    ) -> dict[str, Any]:
        setup = [
            f'ALTER TABLE orders ALTER price TYPE {sql_type(p, s)} USING 0'
        ]
        for i, term in enumerate(terms):
            setup.append(
                f'INSERT INTO orders VALUES ({1000 + i}, 0, '
                f"'{term}'::{sql_type(p, s)}, "
                f"TIMESTAMP '2024-02-10' + INTERVAL {i} DAY)"
            )
        return {'columns': TARGET, 'setup': setup, 'pql': pql}

    big = {
        (p, s): text(largest(p, s), s)
        for p, s in [
            (4, 0),
            (4, 2),
            (9, 0),
            (18, 0),
            (18, 2),
            (19, 0),
            (38, 0),
            (38, 38),
        ]
    }
    out = [
        {'id': 'overflow.d4-0.sum-past', **pair(4, 0, ['9999', '1'])},
        {'id': 'overflow.d4-0.sum-at', **pair(4, 0, ['9998', '1'])},
        {
            'id': 'overflow.d4-0.sum-past-negative',
            **pair(4, 0, ['-9999', '-1']),
        },
        {'id': 'overflow.d4-0.sum-cancels', **pair(4, 0, ['9999', '1', '-1'])},
        {
            'id': 'overflow.d4-0.sum-far-past',
            **pair(4, 0, ['9999', '9999', '9999']),
        },
        {'id': 'overflow.d4-2.sum-past', **pair(4, 2, ['99.99', '0.01'])},
        {'id': 'overflow.d4-2.sum-at', **pair(4, 2, ['99.98', '0.01'])},
        {'id': 'overflow.d9-0.sum-past', **pair(9, 0, [big[(9, 0)], '1'])},
        {'id': 'overflow.d18-0.sum-past', **pair(18, 0, [big[(18, 0)], '1'])},
        {
            'id': 'overflow.d18-2.sum-past',
            **pair(18, 2, [big[(18, 2)], '0.01']),
        },
        {'id': 'overflow.d19-0.sum-past', **pair(19, 0, [big[(19, 0)], '1'])},
        {'id': 'overflow.d38-0.sum-past', **pair(38, 0, [big[(38, 0)], '1'])},
        {'id': 'overflow.d38-0.sum-at', **pair(38, 0, [big[(38, 0)], '0'])},
        {
            'id': 'overflow.d38-38.sum-past',
            **pair(38, 38, [big[(38, 38)], text(ulp(38), 38) or '']),
        },
        {
            'id': 'overflow.d4-0.avg-past',
            **pair(
                4,
                0,
                ['9999', '1'],
                'PREDICT AVG(orders.price, 0, 30, days) FOR EACH users.user_id',
            ),
        },
        {
            'id': 'overflow.d4-0.max-past',
            **pair(
                4,
                0,
                ['9999', '1'],
                'PREDICT MAX(orders.price, 0, 30, days) FOR EACH users.user_id',
            ),
        },
        {
            'id': 'overflow.d4-0.binary-past',
            **pair(
                4,
                0,
                ['9999', '1'],
                'PREDICT SUM(orders.price, 0, 30, days) > 5000 '
                'FOR EACH users.user_id',
            ),
        },
    ]
    return out


def _tier(p: int, s: int, values: list[str]) -> str:
    whens = ' '.join(
        f"WHEN {i} THEN '{v}'::{sql_type(p, s)}" for i, v in enumerate(values)
    )
    return (
        'CREATE OR REPLACE TABLE users AS SELECT *, '
        f'CASE user_id % 3 {whens} END AS tier FROM users'
    )


def _classes() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    static = 'PREDICT users.tier FOR EACH users.user_id'
    three = {
        (4, 0): ['1', '2', '3'],
        (4, 2): ['0.10', '0.20', '0.30'],
        (9, 9): ['0.100000000', '0.200000000', '0.300000000'],
        (18, 2): ['1.50', '2.50', '-3.25'],
        (38, 0): ['1', '2', '3'],
        (38, 10): ['0.1000000000', '0.2000000000', '0.3000000000'],
    }
    for (p, s), values in three.items():
        out.append(
            {
                'id': f'class.static-three.{tag(p, s)}',
                'columns': [['instance_table', 'TARGET'], ['users', 'tier']],
                'setup': [_tier(p, s, values)],
                'pql': static,
                'anchor_time': None,
            }
        )
    for p, s in [(4, 2), (18, 0), (38, 10)]:
        out.append(
            {
                'id': f'class.static-many.{tag(p, s)}',
                'columns': [['instance_table', 'TARGET'], ['users', 'f']],
                'setup': [
                    add_f(
                        'users', shape('spread', p, s)[:7] + [Decimal(1)], p, s
                    )
                ],
                'pql': 'PREDICT users.f FOR EACH users.user_id',
                'anchor_time': None,
            }
        )
    for p, s in [(4, 2), (18, 0), (38, 10)]:
        out.append(
            {
                'id': f'class.binary-sum.{tag(p, s)}',
                'columns': TARGET,
                'setup': [price_as(shape('spread', p, s), p, s)],
                'pql': (
                    'PREDICT SUM(orders.price, 0, 30, days) > 0 '
                    'FOR EACH users.user_id'
                ),
            }
        )
    out.append(
        {
            'id': 'class.binary-sum-fraction.d4-2',
            'columns': TARGET,
            'setup': [price_as(shape('spread', 4, 2), 4, 2)],
            'pql': (
                'PREDICT SUM(orders.price, 0, 30, days) > 0.25 '
                'FOR EACH users.user_id'
            ),
        }
    )
    for p, s in [(4, 2), (38, 0)]:
        out.append(
            {
                'id': f'class.binary-static.{tag(p, s)}',
                'columns': [['instance_table', 'TARGET'], ['users', 'f']],
                'setup': [add_f('users', shape('spread', p, s), p, s)],
                'pql': 'PREDICT users.f > 0 FOR EACH users.user_id',
                'anchor_time': None,
            }
        )
    return out


def _alter_key(table: str, column: str, sql: str, using: str = '') -> str:
    return f'ALTER TABLE {table} ALTER {column} TYPE {sql}' + (
        f' USING {using}' if using else ''
    )


KEYS = [['users', 'user_id'], ['orders', 'user_id']]


def _keys() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p, s in [(4, 0), (9, 0), (19, 0), (38, 0), (9, 2), (18, 2)]:
        out.append(
            {
                'id': f'key.both.{tag(p, s)}',
                'columns': KEYS,
                'setup': [
                    _alter_key('users', 'user_id', sql_type(p, s)),
                    _alter_key('orders', 'user_id', sql_type(p, s)),
                ],
            }
        )
    for p, s in [(4, 0), (18, 0)]:
        out.append(
            {
                'id': f'key.foreign-only.{tag(p, s)}',
                'columns': KEYS,
                'setup': [_alter_key('orders', 'user_id', sql_type(p, s))],
            }
        )
        out.append(
            {
                'id': f'key.primary-only.{tag(p, s)}',
                'columns': KEYS,
                'setup': [_alter_key('users', 'user_id', sql_type(p, s))],
            }
        )
    for p, s in [(18, 0), (38, 0)]:
        out.append(
            {
                'id': f'key.fact-key.{tag(p, s)}',
                'columns': [['orders', 'order_id']],
                'setup': [_alter_key('orders', 'order_id', sql_type(p, s))],
            }
        )
    out.append(
        {
            'id': 'key.negative.d9-0',
            'columns': KEYS,
            'setup': [
                _alter_key('users', 'user_id', 'DECIMAL(9, 0)', 'user_id - 4'),
                _alter_key('orders', 'user_id', 'DECIMAL(9, 0)', 'user_id - 4'),
            ],
            'indices': [-4, 0],
        }
    )
    for p, base in [(18, 10**17 * 9), (19, 10**19 - 10), (38, 10**37)]:
        using = f"user_id + '{base}'::DECIMAL({p}, 0)"
        out.append(
            {
                'id': f'key.at-precision.d{p}-0',
                'columns': KEYS,
                'setup': [
                    _alter_key('users', 'user_id', f'DECIMAL({p}, 0)', using),
                    _alter_key('orders', 'user_id', f'DECIMAL({p}, 0)', using),
                ],
                'indices': [base, base + 1],
            }
        )
    # The key-type pair matrix (design, T2.17's exit, moved into T2.0c):
    # primary and foreign keys of different decimal precisions and scales
    # holding equal values (1.00 against 1.0000) join, in both directions;
    # and values float64 would merge (10**17 + 0 to 7, 16 apart at most in
    # a double) stay distinct.
    for (pp, ps), (fp, fs) in [
        ((10, 2), (12, 4)),
        ((12, 4), (10, 2)),
        ((18, 0), (20, 2)),
        ((20, 2), (18, 0)),
    ]:
        out.append(
            {
                'id': f'key.pair.{tag(pp, ps)}.{tag(fp, fs)}',
                'columns': KEYS,
                'setup': [
                    _alter_key('users', 'user_id', sql_type(pp, ps)),
                    _alter_key('orders', 'user_id', sql_type(fp, fs)),
                ],
            }
        )
    wide = 10**17
    for (pp, ps), (fp, fs) in [((38, 0), (38, 2)), ((38, 2), (38, 0))]:
        out.append(
            {
                'id': f'key.pair-merging.{tag(pp, ps)}.{tag(fp, fs)}',
                'columns': KEYS,
                'setup': [
                    _alter_key(
                        'users',
                        'user_id',
                        sql_type(pp, ps),
                        f"user_id + '{wide}'::{sql_type(pp, ps)}",
                    ),
                    _alter_key(
                        'orders',
                        'user_id',
                        sql_type(fp, fs),
                        f"user_id + '{wide}'::{sql_type(fp, fs)}",
                    ),
                ],
                'indices': [wide, wide + 1],
            }
        )
    out.append(
        {
            'id': 'key.null-foreign.d18-0',
            'columns': KEYS,
            'setup': [
                _alter_key('users', 'user_id', 'DECIMAL(18, 0)'),
                _alter_key('orders', 'user_id', 'DECIMAL(18, 0)'),
                'UPDATE orders SET user_id = NULL WHERE order_id % 10 = 1',
            ],
        }
    )
    return out


def _times() -> list[dict[str, Any]]:
    epoch = 'epoch(ts)::DECIMAL(18, 0)'
    epoch_ms = '(epoch_ms(ts) / 1000)::DECIMAL(18, 3)'
    yyyymmdd = "strftime(ts, '%Y%m%d')::DECIMAL(8, 0)"

    def on_orders(expression: str) -> str:
        return (
            'CREATE OR REPLACE TABLE orders AS SELECT *, '
            f'{expression} AS f FROM orders'
        )

    def declared(column: str) -> dict[str, Any]:
        return {'time_columns': {'orders': column}}

    return [
        {
            'id': 'time.epoch-feature',
            'columns': [['orders', 'f']],
            'setup': [on_orders(epoch)],
        },
        {
            'id': 'time.epoch-ms-feature',
            'columns': [['orders', 'f']],
            'setup': [on_orders(epoch_ms)],
        },
        {
            'id': 'time.yyyymmdd-feature',
            'columns': [['orders', 'f']],
            'setup': [on_orders(yyyymmdd)],
        },
        {
            'id': 'time.epoch-declared',
            'columns': [['orders', 'f']],
            'setup': [on_orders(epoch)],
            'from_source': declared('f'),
        },
        {
            'id': 'time.yyyymmdd-declared',
            'columns': [['orders', 'f']],
            'setup': [on_orders(yyyymmdd)],
            'from_source': declared('f'),
        },
        {
            'id': 'time.epoch-in-place',
            'columns': [['orders', 'ts']],
            'setup': [
                _alter_key('orders', 'ts', 'DECIMAL(18, 0)', 'epoch(ts)'),
            ],
        },
        {
            'id': 'time.epoch-in-place-declared',
            'columns': [['orders', 'ts']],
            'setup': [
                _alter_key('orders', 'ts', 'DECIMAL(18, 0)', 'epoch(ts)'),
            ],
            'from_source': declared('ts'),
        },
    ]


#: The vectors: each its id, the decimal ``columns`` it is about, and its
#: setup, after ``shop``'s statements; and, where it varies them, its
#: question (``pql``), entities (``indices``), anchor time
#: (``anchor_time``, null for none) and the tables it describes
#: (``from_source``, as ``Schema.from_source``'s keywords).
VECTORS: list[dict[str, Any]] = (
    _features()
    + _wide()
    + _targets()
    + _overflow()
    + _classes()
    + _keys()
    + _times()
)


# --- the cases -----------------------------------------------------------------


def core_anchor(anchor: str | None) -> Any:
    r"""The anchor time as the core's contract scalar (the case's
    ``core.materialize.anchor_time``): a naive timestamp in nanoseconds, as
    the client's bridge hands it over."""
    if anchor is None:
        return None
    from datetime import datetime, timezone

    moment = datetime.fromisoformat(anchor).replace(tzinfo=timezone.utc)
    nanoseconds = int(moment.timestamp()) * 1_000_000_000
    return {'t': 'ts', 'v': str(nanoseconds), 'unit': 'ns', 'zone': None}


#: T1.32a's recorder: the entities as the host's natural Arrow column.
K = _module('key_cast_record', ROOT / 'contract/tools/key_cast_record.py')


def core_indices(indices: list[Any]) -> list[dict[str, Any]]:
    r"""The entities as the core's contract scalars (the case's
    ``core.materialize.indices``): the host's natural Arrow column of them,
    as the client's bridge hands it over (T1.32a's ``host_arrow``): int64
    for the small ones, uint64 past int64, text past uint64."""
    return list(K.host_arrow(list(indices))['values'])


def case_for(vector: dict[str, Any], core: bool = False) -> dict[str, Any]:
    r"""The vector as a case: ``decimal.feature-38-0-wide``'s, its setup,
    question, entities, anchor time and tables the vector's; with ``core``,
    forced onto the core's routes (T1.32d's ``core_routes``), and otherwise
    on none (the runner's own choice, as the reference has)."""
    template = json.loads(TEMPLATE.read_text(encoding='utf-8'))
    case = copy.deepcopy(template)
    case['id'] = CASE_PREFIX + vector['id']
    case['status'] = 'diagnostic'
    case['routes'] = E.core_routes(vector) if core else []
    case['cells'] = []
    case['input']['setup'] = list(vector['setup'])
    case['input']['pql'] = vector.get('pql', QUESTION)
    indices = list(vector.get('indices', INDICES))
    case['input']['materialize']['indices'] = indices
    anchor = vector.get('anchor_time', ANCHOR)
    if anchor is None:
        case['input']['materialize'].pop('anchor_time', None)
    else:
        case['input']['materialize']['anchor_time'] = anchor
    case['core']['describe_source']['source'] = E.core_source(
        vector.get('from_source')
    )
    if 'from_source' in vector:
        case['input']['from_source'] = copy.deepcopy(vector['from_source'])
    case['core']['materialize']['indices'] = core_indices(indices)
    case['core']['materialize']['anchor_time'] = core_anchor(anchor)
    case['expect'] = {'outcome': 'success', 'stage': 'predict'}
    case['why'] = f'T1.32b vector {vector["id"]}'
    case['source'] = 'contract/tools/decimal_record.py'
    case.pop('register', None)
    return case


def run(
    mode: str,
    vectors: list[dict[str, Any]],
    directory: Path,
    *,
    core: bool = False,
    reference: Path | None = None,
    capture: bool = True,
) -> dict[str, dict[str, Any]]:
    r"""``vectors`` as cases through ``run_python.py --mode mode``, in this
    interpreter, from ``directory``, forced onto the core with ``core``:
    each observation by vector id."""
    cases = directory / 'cases'
    cases.mkdir(parents=True, exist_ok=True)
    for vector in vectors:
        case = case_for(vector, core)
        (cases / f'{case["id"]}.json').write_text(
            json.dumps(case, indent=1), encoding='utf-8'
        )
    out = directory / f'{mode}{"" if capture else "-off"}.jsonl'
    command = [sys.executable, '-I', str(RUNNER), '--mode', mode]
    if reference is not None:
        command += ['--reference', str(reference)]
    if not capture:
        command.append('--no-capture')
    command += ['--out', str(out), str(cases)]
    done = subprocess.run(
        command, capture_output=True, text=True, cwd=ROOT, check=False
    )
    if done.returncode != 0:
        raise RuntimeError(f'{mode} run failed:\n{done.stdout}\n{done.stderr}')
    observed = {}
    for line in out.read_text(encoding='utf-8').splitlines():
        observation = json.loads(line)
        observed[observation['case'][len(CASE_PREFIX) :]] = observation
    return observed


# --- the public answer ---------------------------------------------------------


def cell_text(cell: Any) -> str:
    r"""A cell of a request as its JSON text, keys sorted: a number
    ``{"number": "<digits>"}`` keeps its digits as sent."""
    return json.dumps(cell, sort_keys=True, separators=(',', ':'))


def column_in(parsed: Any, part: str, table: str, column: str) -> Any:
    r"""How one column reached the model in one part (``context`` or
    ``predict``) of a request: the dtype and stype its schema declares and
    its cells counted, or null when the part holds no such column."""
    if not isinstance(parsed, dict):
        return None
    section = parsed.get(part) or {}
    schema = parsed.get('schema') or {}
    if table == 'instance_table':
        holder = section.get('instance_table')
        declared = (schema.get('instance_table') or {}).get('columns') or {}
    else:
        holder = (section.get('related_tables') or {}).get(table)
        declared = ((schema.get('related_tables') or {}).get(table) or {}).get(
            'columns'
        ) or {}
    if not isinstance(holder, dict):
        return None
    names = holder.get('columns') or []
    if column not in names:
        return None
    at = names.index(column)
    cells = Counter(
        cell_text(row[at])
        for row in holder.get('rows') or []
        if isinstance(row, list) and len(row) > at
    )
    wire = declared.get(column) or {}
    return {
        'dtype': wire.get('dtype'),
        'stype': wire.get('stype'),
        'cells': dict(sorted(cells.items())),
    }


def columns_projection(
    observation: dict[str, Any], vector: dict[str, Any]
) -> dict[str, Any]:
    r"""For each of the vector's columns: its public dtype and stype
    (``described``), and how it reached the model in each request
    (``request``, by part), keyed ``table.column``."""
    public = observation.get('public') or {}
    schema = public.get('schema') or {}
    out: dict[str, Any] = {}
    for table, column in vector['columns']:
        described = None
        for entry in schema.get('tables') or []:
            if entry.get('name') != table:
                continue
            for each in entry.get('columns') or []:
                if each.get('name') == column:
                    described = {
                        'dtype': each.get('dtype'),
                        'stype': each.get('stype'),
                    }
        requests = []
        for payload in public.get('payloads') or []:
            parsed = payload.get('parsed')
            requests.append(
                {
                    part: column_in(parsed, part, table, column)
                    for part in ('context', 'predict')
                }
            )
        out[f'{table}.{column}'] = {
            'described': described,
            'request': requests or None,
        }
    return out


def project(
    observation: dict[str, Any], vector: dict[str, Any]
) -> dict[str, Any]:
    r"""The public answer of one run of ``vector``, as both Python runners
    record it: T1.32d's projection and the vector's columns."""
    answer = E.project(observation, vector)
    answer['columns'] = columns_projection(observation, vector)
    return answer


def statements(observation: dict[str, Any]) -> dict[str, int]:
    r"""The statements a run ran per public call, as ``compare.py`` counts
    them, its capture complete against DuckDB's query log
    (``capture_check.py``'s ``complete``)."""
    compare = _module('compare', ROOT / 'scripts/rust_core/compare.py')
    capture = _module(
        'capture_check', ROOT / 'contract/runners/capture_check.py'
    )
    problems = capture.complete(observation)
    if problems:
        raise ValueError(f'{observation["case"]}: {problems}')
    counted = compare.counts(observation)
    assert counted is not None
    return dict(counted)


#: The family's cases and their decimal columns, projected as the vectors'
#: are: the document's ``cases`` holds the reference's committed
#: observation of each (``contract/reference/<commit>.jsonl``) so read, so
#: ``rust-host``'s reading of a request is held to this one's on the
#: reference's own bytes.
CASE_COLUMNS: dict[str, list[list[str]]] = {
    'decimal.class-target-38-18': [
        ['instance_table', 'TARGET'],
        ['users', 'tier'],
    ],
    'decimal.feature-38-0-wide': [['users', 'f']],
    'decimal.target-38-18': [
        ['instance_table', 'TARGET'],
        ['orders', 'price'],
    ],
    # T2.0c's exact-decimal cases.
    **{
        f'decimal.{name}': [['instance_table', 'TARGET'], ['users', 'd']]
        for name in (
            'compare-finer-eq',
            'compare-finer-gt',
            'compare-finer-negative',
            'compare-past-precision',
            'compare-past-precision-gt',
            'compare-same-scale',
        )
    },
    'decimal.feature-inexact-38-18': [['users', 'f'], ['users', 'g']],
    'decimal.key-wide': [['users', 'user_id'], ['orders', 'user_id']],
    'decimal.sum-10-2': [['instance_table', 'TARGET'], ['orders', 'amount']],
    'decimal.sum-past-38': [
        ['instance_table', 'TARGET'],
        ['orders', 'amount'],
    ],
    'decimal.target-regression-collide': [
        ['instance_table', 'TARGET'],
        ['users', 't'],
    ],
}


def reference_observations() -> dict[str, dict[str, Any]]:
    r"""The committed ``python-ref`` observations, by case."""
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    commit = next(
        line.split(': ', 1)[1].strip()
        for line in env.splitlines()
        if line.startswith('reference: ')
    )
    path = ROOT / f'contract/reference/{commit}.jsonl'
    out = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        observation = json.loads(line)
        out[observation['case']] = observation
    return out


def cases_projection() -> dict[str, Any]:
    observed = reference_observations()
    return {
        case: project(observed[case], {'columns': columns})
        for case, columns in CASE_COLUMNS.items()
    }


outcome_class = E.outcome_class
register = E.register
UNNAMED = E.UNNAMED


# --- what DuckDB holds ---------------------------------------------------------


def duckdb_holds(vector: dict[str, Any]) -> dict[str, Any]:
    r"""Each named column's catalog type and the Arrow type DuckDB hands
    over for it, after the setup (the request's own columns have none)."""
    import duckdb

    connection = duckdb.connect()
    try:
        for statement in E.shop() + vector['setup']:
            connection.execute(statement)
        out: dict[str, Any] = {}
        for table, column in vector['columns']:
            if table == 'instance_table':
                continue
            found = connection.execute(
                'SELECT data_type FROM information_schema.columns '
                'WHERE table_name = ? AND column_name = ?',
                [table, column],
            ).fetchone()
            arrow = connection.execute(
                f'SELECT "{column}" FROM "{table}"'
            ).to_arrow_table()
            out[f'{table}.{column}'] = {
                'type': found[0] if found else None,
                'arrow': str(arrow.schema.field(0).type),
            }
        return out
    finally:
        connection.close()


def main() -> int:
    import duckdb
    import numpy as np
    import pandas as pd
    import pyarrow as pa

    reference = Path.home() / 'code/sdmc-reference'
    commit = subprocess.run(
        ['git', '-C', str(reference), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    import sdm_client

    if not Path(sdm_client.__file__).resolve().is_relative_to(reference):
        print("sdm_client is not the reference checkout's", file=sys.stderr)
        return 1
    only = sys.argv[1:]
    chosen = [
        v
        for v in VECTORS
        if not only or any(v['id'].startswith(o) for o in only)
    ]
    with tempfile.TemporaryDirectory(prefix='t132b-record-') as tmp:
        observed = run('python-ref', chosen, Path(tmp), reference=reference)
    vectors = []
    for vector in chosen:
        answer = project(observed[vector['id']], vector)
        why = register(answer)
        entry = {key: copy.deepcopy(value) for key, value in vector.items()}
        # What the core is handed, as the bridge builds it: rust-host
        # reads these, python-core's bridge derives them itself.
        entry['core_indices'] = core_indices(vector.get('indices', INDICES))
        entry['core_anchor'] = core_anchor(vector.get('anchor_time', ANCHOR))
        entry['duckdb'] = duckdb_holds(vector)
        entry['reference'] = answer
        # The statements the reference ran per public call (compare.py's
        # counts), its capture held complete against DuckDB's query log.
        entry['statements'] = statements(observed[vector['id']])
        entry['register'] = {'required': bool(why), 'why': why}
        vectors.append(entry)
        outcome = answer['outcome']
        print(
            vector['id'],
            outcome['result'],
            outcome['stage'],
            outcome['error'] or '',
            outcome['code'] or '',
            answer['warnings'],
            '(register)' if why else '',
            file=sys.stderr,
        )
    if only:
        print(json.dumps(vectors, indent=1))
        return 0
    document = {
        'format': FORMAT,
        'recorded_by': 'contract/tools/decimal_record.py',
        'reference': commit,
        'cell': 'ref-pandas3',
        'environment': {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'pyarrow': pa.__version__,
            'numpy': np.__version__,
            'duckdb': duckdb.__version__,
        },
        'template': 'contract/cases/s1/decimal.feature-38-0-wide.json',
        'question': QUESTION,
        'indices': INDICES,
        'anchor_time': ANCHOR,
        'register_ref': REGISTER_REF,
        'payload_exclusions': E.exclusions(),
        'cases': cases_projection(),
        'vectors': vectors,
    }
    OUT.write_text(json.dumps(document, indent=1) + '\n', encoding='utf-8')
    required = sum(v['register']['required'] for v in vectors)
    print(
        f'{len(vectors)} vectors written to {OUT.relative_to(ROOT)}, '
        f'{required} marked register entry required',
        file=sys.stderr,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
