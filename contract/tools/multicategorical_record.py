#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference's handling of text that may hold several
categories per value into ``contract/multicategorical-vectors.json``
(T1.32e part 2's vectors: the multicategorical port).

Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root::

    ~/code/sdmc-reference/.venv/bin/python -I contract/tools/multicategorical_record.py

Written by T1.32e's independent test author before the implementation.
The reference (``sdm_relational_engine.rfm.metadata_inference``'s
``contains_multicategorical``, run by ``infer_stype`` after the key and
time checks) reads the first 500 rows of a text column's preview (the
first 1,000 rows the source hands over), drops the nulls, counts the
characters ``;``, ``:``, ``|`` and a tab, splits on the most frequent one,
and calls the column ``multicategorical`` when it then holds more than 1.5
times as many distinct values as distinct parts and at most 100 parts.
The core refuses any text column holding one of those characters in its
first 500 values today (rule ``multicategorical``), so ordinary text, a
time of day or a URL, is refused.

Each vector is ``contract/bases.yaml``'s ``shop`` changed by its ``setup``
(DuckDB statements): a text column ``f`` added to ``users`` (the entity
table) or ``orders`` (the fact table), on ``shop``'s own 8 users or on
generated tables of :data:`ROWS` (200) or :data:`LARGE` (1,200) users,
asked ``feature.bool``'s question, ``PREDICT SUM(orders.price, 0, 30,
days) FOR EACH users.user_id``, unless it names its own, for its own
``indices`` (default ``[0, 1]``; the large tables' ``[0, 1, 1100, 1199]``,
so values past the preview reach the request):

- ``sep.*``: each separator the reference recognizes, and two it does not
  (a comma, a space); the boundary of the 1.5 ratio and of 100 parts;
  empty parts and spaces around them;
- ``text.*``: text that must not be refused for holding a separator: times
  of day (``09:30``), URLs, ratios (``3:1``), free text with an occasional
  separator, addresses with colons;
- ``mixed.*``: two separators in one column, the most frequent winning, a
  tie, and a separator the reference does not count beside one it does;
- ``null.*`` and ``empty.*``: nulls and empty strings among the values,
  and a column of them only;
- ``single.*``: one value throughout, with or without a separator;
- ``large.*``: 1,200 rows whose values past row 500 or past the 1,000-row
  preview differ from those before;
- ``fact.*``: the column on the fact table;
- ``question.*``: a question naming the column.

Each vector is a case in ``contract/cases``' format, written to a
temporary directory and run by ``contract/runners/run_python.py --mode
python-ref`` through the public calls (``Schema.from_source``,
``materialize``, ``SDMClient.predict`` against the scripted endpoint), and
this script keeps the reference's answer as :func:`project` reads it from
the observation: ``empty_null_record.project``'s (the outcome, the public
schema with each column's dtype and stype, the warning classes, each
request's digest as the ``payload`` rule compares it with its tables' rows
and null cells, the result), and ``wire``: how the column travels in each
request, per table carrying it (``context.related_tables.users`` and the
like), the declaration the request's ``schema`` gives it (its dtype and
stype) and its cells, in the order sent. Beside it, what DuckDB holds:
the tables' row counts, ``f``'s type, and how many times each separator
occurs in ``f``'s first 500 rows, the 1,000-row preview and the whole
table.

A vector marked ``run_day`` (``fact.time-of-day``) holds text times of
day, which the reference reads as times on the day it runs: in its
requests, the day of each timestamp cell of ``f`` is written as
:data:`RUN_DAY` (``at_run_day``) before the digest and the wire are taken,
by the recorder and by both hosts' tests alike, so the vector holds on any
day.

``register`` marks the vectors where the core is expected to differ. It is
derived from the reference's answer, never chosen: the reference fails
with an unnamed exception or with ``INTERNAL_ERROR`` (E66(b)). The mark
decides no policy: any vector may differ from the reference only under an
approved register entry whose ``refs`` name it with exactly the transition
it makes, ``REGISTER_REF`` + ``<id or glob> <from> -> <to>`` (T1.32a's
binding-transition form, as ``empty_null_record.outcome_class`` reads each
side).

Nothing here is written by hand but the inputs: every expected value is
the reference's. ``VECTORS``, ``case_for``, ``run`` and ``project`` are
imported by ``contract/tests/test_t132e_multicategorical.py``, so
``python-core`` is observed and projected by the same code.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/multicategorical-vectors.json'
TEMPLATE = ROOT / 'contract/cases/s1/feature.bool.json'
RUNNER = ROOT / 'contract/runners/run_python.py'
FORMAT = 'sdm-multicategorical-vectors/1'
REGISTER_REF = 'contract/multicategorical-vectors.json#'
CASE_PREFIX = 't132e2.'
QUESTION = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
INDICES = [0, 1]
#: The generated tables' users, and the large ones'.
ROWS = 200
LARGE = 1200
#: The large tables' entities: two in the first 500 rows, two past the
#: preview.
LARGE_INDICES = [0, 1, 1100, 1199]
#: The characters the reference counts as separators.
SEPARATORS = {'semicolon': ';', 'colon': ':', 'pipe': '|', 'tab': '\t'}


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


#: T1.32d's recorder: the payload rule's digest, the projection of the
#: public answer, the outcome classes and the register's derivation, shared
#: so every family's vectors are read alike.
EN = _module('empty_null_record', ROOT / 'contract/tools/empty_null_record.py')


# --- the inputs ----------------------------------------------------------------


def _literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def cycle(values: list[str | None], over: str = 'i') -> str:
    r"""A DuckDB expression taking ``values`` in turn by row number."""
    items = ', '.join('NULL' if v is None else _literal(v) for v in values)
    return f'([{items}]::VARCHAR[])[1 + {over} % {len(values)}]'


def combos(sep: str) -> list[str]:
    r"""Three categories, one to three per value, joined by ``sep``: eight
    distinct values of three parts (multi-category by the reference's
    ratio)."""
    return [
        f'a{sep}b',
        f'a{sep}c',
        f'b{sep}c',
        f'a{sep}b{sep}c',
        'a',
        'b',
        'c',
        f'c{sep}a',
    ]


def generated(
    f_users: str | None, rows: int, f_orders: str | None = None
) -> list[str]:
    r"""``users`` and ``orders`` replaced by generated tables of ``rows``
    users and ten orders each over 420 days before the anchor, ``f`` on
    either."""
    users_f = f', {f_users} AS f' if f_users is not None else ''
    orders_f = f', {f_orders} AS f' if f_orders is not None else ''
    return [
        'CREATE OR REPLACE TABLE users AS SELECT i::BIGINT AS user_id, '
        f"(20 + i % 40)::INTEGER AS age, ['US', 'DE', 'IN'][1 + i % 3] AS "
        f'country{users_f} FROM range({rows}) t(i)',
        'CREATE OR REPLACE TABLE orders AS SELECT i::BIGINT AS order_id, '
        f'(i % {rows})::BIGINT AS user_id, (10 + (i % 97) * 1.5)::DOUBLE AS '
        "price, TIMESTAMP '2023-01-01' + to_days(((i * 7) % 420)::INTEGER) "
        f'AS ts{orders_f} FROM range({rows * 10}) t(i)',
    ]


def on_shop(expression: str, table: str = 'users') -> list[str]:
    r"""``f`` added to one of ``shop``'s own tables."""
    return [
        f'CREATE OR REPLACE TABLE {table} AS SELECT *, {expression} AS f '
        f'FROM {table}'
    ]


def _vector(
    vid: str, setup: list[str], table: str = 'users', **more: Any
) -> dict[str, Any]:
    return {'id': vid, 'table': table, 'setup': setup, **more}


def _late(early: str, late: str, at: int) -> str:
    return f'CASE WHEN i < {at} THEN {early} ELSE {late} END'


def _separators() -> list[dict[str, Any]]:
    out = [
        _vector(f'sep.{name}', generated(cycle(combos(sep)), ROWS))
        for name, sep in SEPARATORS.items()
    ]
    out += [
        # On shop's own eight users.
        _vector('sep.semicolon-8-rows', on_shop(cycle(combos(';'), 'user_id'))),
        _vector('sep.colon-8-rows', on_shop(cycle(combos(':'), 'user_id'))),
        # Separators the reference does not count.
        _vector('sep.comma', generated(cycle(combos(',')), ROWS)),
        _vector('sep.space', generated(cycle(combos(' ')), ROWS)),
        _vector('sep.slash', generated(cycle(combos('/')), ROWS)),
        # Spaces around the parts, empty parts, a separator at either end.
        _vector('sep.semicolon-spaced', generated(cycle(combos('; ')), ROWS)),
        _vector(
            'sep.semicolon-empty-parts',
            generated(
                cycle(['a;;b', 'a;', ';b', 'a;b', 'b;c;', ';', 'a', 'c;;']),
                ROWS,
            ),
        ),
        # The ratio's boundary: six pairs of four parts (6 = 1.5 x 4, not
        # more) and seven values of the same four parts.
        _vector(
            'sep.ratio-at-1.5',
            generated(cycle(['a;b', 'a;c', 'a;d', 'b;c', 'b;d', 'c;d']), ROWS),
        ),
        _vector(
            'sep.ratio-over-1.5',
            generated(
                cycle(['a;b', 'a;c', 'a;d', 'b;c', 'b;d', 'c;d', 'a;b;c']),
                ROWS,
            ),
        ),
        # Every value's parts its own: twice as many parts as values.
        _vector(
            'sep.every-part-unique',
            generated("'u' || i::VARCHAR || ';v' || i::VARCHAR", ROWS),
        ),
        # The part count's boundary: 100 parts, and 101, each in some 500
        # distinct pairs over the first 500 rows (far over 1.5 times the
        # parts).
        _vector(
            'sep.parts-100',
            generated(
                "'t' || (i % 100)::VARCHAR || ';t' || "
                '((i // 100 + i) % 100)::VARCHAR',
                LARGE,
            ),
            indices=LARGE_INDICES,
        ),
        _vector(
            'sep.parts-101',
            generated(
                "'t' || (i % 101)::VARCHAR || ';t' || "
                '((i // 101 + i) % 101)::VARCHAR',
                LARGE,
            ),
            indices=LARGE_INDICES,
        ),
    ]
    return out


def _texts() -> list[dict[str, Any]]:
    hours = "lpad((i % 24)::VARCHAR, 2, '0')"
    quarter = "['00', '15', '30', '45'][1 + (i // 24) % 4]"
    return [
        # Times of day.
        _vector(
            'text.time-of-day', generated(f"{hours} || ':' || {quarter}", ROWS)
        ),
        _vector(
            'text.time-of-day-seconds',
            generated(f"{hours} || ':' || {quarter} || ':05'", ROWS),
        ),
        _vector(
            'text.time-of-day-8-rows',
            on_shop(cycle(['09:30', '10:15', '17:45', '08:00'], 'user_id')),
        ),
        _vector(
            'text.time-of-day-am-pm',
            generated(
                "((i % 12) + 1)::VARCHAR || ':' || " + quarter + ' || '
                "[' AM', ' PM'][1 + (i // 12) % 2]",
                ROWS,
            ),
        ),
        # URLs: one per row, and a few shared.
        _vector(
            'text.url-unique',
            generated("'https://example.com/item/' || i::VARCHAR", ROWS),
        ),
        _vector(
            'text.url-few',
            generated(
                cycle(
                    [
                        'https://a.example/x',
                        'https://b.example/y',
                        'http://c.example/z',
                    ]
                ),
                ROWS,
            ),
        ),
        _vector(
            'text.url-with-port-and-query',
            generated(
                "'http://host.example:8080/p?a=' || (i % 7)::VARCHAR || "
                "';b=' || (i % 3)::VARCHAR",
                ROWS,
            ),
        ),
        # Ratios.
        _vector(
            'text.ratio',
            generated(
                "((i % 5) + 1)::VARCHAR || ':' || ((i // 5) % 4 + 1)::VARCHAR",
                ROWS,
            ),
        ),
        _vector(
            'text.aspect-ratio',
            generated(cycle(['16:9', '4:3', '21:9', '1:1', '3:2']), ROWS),
        ),
        # Free text, a separator now and then.
        _vector(
            'text.free-occasional-colon',
            generated(
                "CASE WHEN i % 50 = 0 THEN 'note: see order ' || i::VARCHAR "
                "ELSE 'a plain remark about order ' || i::VARCHAR END",
                ROWS,
            ),
        ),
        _vector(
            'text.free-occasional-semicolon',
            generated(
                "CASE WHEN i % 40 = 0 THEN 'late; refunded ' || i::VARCHAR "
                "ELSE 'delivered on time ' || i::VARCHAR END",
                ROWS,
            ),
        ),
        _vector(
            'text.free-few-occasional-colon',
            generated(
                cycle(['fine', 'damaged', 'returned', 'see note: box', 'late']),
                ROWS,
            ),
        ),
        # Addresses holding colons.
        _vector(
            'text.mac-address',
            generated(
                "'aa:bb:cc:' || lpad(to_hex(i % 256), 2, '0') || ':' || "
                "lpad(to_hex((i * 3) % 256), 2, '0') || ':ff'",
                ROWS,
            ),
        ),
        _vector(
            'text.ipv6',
            generated("'2001:db8::' || to_hex(i)", ROWS),
        ),
        _vector(
            'text.windows-path',
            generated("'C:\\data\\file' || i::VARCHAR || '.csv'", ROWS),
        ),
        # No separator at all: the controls.
        _vector(
            'text.plain-categories',
            generated(cycle(['red', 'green', 'blue']), ROWS),
        ),
        _vector(
            'text.plain-unique', generated("'customer ' || i::VARCHAR", ROWS)
        ),
    ]


def _mixed() -> list[dict[str, Any]]:
    return [
        # Both separators, the pipe the more frequent.
        _vector(
            'mixed.semicolon-and-pipe',
            generated(
                cycle(['a|b|c', 'a|b;c', 'b|c', 'a;b', 'c|a|b', 'a', 'b|a']),
                ROWS,
            ),
        ),
        # As often one as the other: the first seen wins.
        _vector(
            'mixed.semicolon-pipe-tie',
            generated(
                cycle(['a;b', 'c|d', 'a;c', 'b|d', 'a', 'd', 'b;d', 'a|c']),
                ROWS,
            ),
        ),
        _vector(
            'mixed.pipe-semicolon-tie',
            generated(
                cycle(['c|d', 'a;b', 'b|d', 'a;c', 'a', 'd', 'a|c', 'b;d']),
                ROWS,
            ),
        ),
        # The pipe the more frequent, and a multi-category column split on
        # it (split on the semicolon, it would not be one).
        _vector(
            'mixed.pipe-majority-multi',
            generated(
                cycle(
                    [*combos('|'), 'a;1'],
                ),
                ROWS,
            ),
        ),
        # The semicolon the more frequent, and no multi-category column
        # split on it (split on the pipe, it would be one).
        _vector(
            'mixed.semicolon-majority-not-multi',
            generated(
                cycle(
                    [
                        *combos('|'),
                        'u1;u2;u3;u4;u5',
                        'v1;v2;v3;v4;v5',
                        'w1;w2;w3;w4;w5;w6',
                    ]
                ),
                ROWS,
            ),
        ),
        # As many pipes as semicolons, the first seen deciding: split on
        # the pipe, seven parts of eleven values (one); on the semicolon,
        # fifteen (none).
        _vector(
            'mixed.tie-pipe-first',
            generated(
                cycle(
                    [
                        'a|b',
                        'p1;p2',
                        'a|c',
                        'p3;p4',
                        'b|c',
                        'p5;p6',
                        'c|a',
                        'p7;p8',
                        'a',
                        'b',
                        'c',
                    ]
                ),
                ROWS,
            ),
        ),
        _vector(
            'mixed.tie-semicolon-first',
            generated(
                cycle(
                    [
                        'p1;p2',
                        'a|b',
                        'p3;p4',
                        'a|c',
                        'p5;p6',
                        'b|c',
                        'p7;p8',
                        'c|a',
                        'a',
                        'b',
                        'c',
                    ]
                ),
                ROWS,
            ),
        ),
        # Key-value pairs: a colon in each pair, a semicolon between them.
        _vector(
            'mixed.key-value-pairs',
            generated(
                cycle(['k:1;m:2', 'k:1', 'm:2;k:1', 'k:2;m:1', 'm:1', 'k:2']),
                ROWS,
            ),
        ),
        # A comma, not counted, beside a semicolon.
        _vector(
            'mixed.comma-and-semicolon',
            generated(
                cycle(['a,b;c', 'a;b', 'b,c', 'a;c,b', 'c', 'a,b', 'b;a']),
                ROWS,
            ),
        ),
        # A time of day beside multi-category values.
        _vector(
            'mixed.times-and-categories',
            generated(
                cycle(['09:30', 'a;b', '10:15', 'a;c', 'b;c', 'a', 'b', 'c']),
                ROWS,
            ),
        ),
    ]


def _nulls_and_empties() -> list[dict[str, Any]]:
    base = combos(';')
    return [
        _vector('null.some', generated(cycle([*base, None, None]), ROWS)),
        _vector(
            'null.most',
            generated(
                cycle(['a;b', 'b;c', 'a;b;c', *[None] * 7, 'a', 'c']), ROWS
            ),
        ),
        _vector(
            'null.all-but-one',
            generated("CASE WHEN i = 7 THEN 'a;b' END", ROWS),
        ),
        _vector('null.all', generated('NULL::VARCHAR', ROWS)),
        _vector('empty.some', generated(cycle([*base, '', '']), ROWS)),
        _vector('empty.all', generated("''", ROWS)),
        _vector('empty.and-null', generated(cycle([*base, '', None]), ROWS)),
        _vector(
            'empty.only-separators',
            generated(cycle([';', ';;', '', 'a;b', 'a', 'b;a', 'b']), ROWS),
        ),
    ]


def _singles() -> list[dict[str, Any]]:
    return [
        _vector('single.one-value', generated("'solo'", ROWS)),
        _vector('single.one-value-with-separator', generated("'a;b'", ROWS)),
        _vector('single.one-part-repeated', generated("'a;a'", ROWS)),
        _vector(
            'single.one-category-or-two', generated(cycle(['a', 'a;b']), ROWS)
        ),
        _vector('single.one-value-8-rows', on_shop("'a;b'")),
    ]


def _large() -> list[dict[str, Any]]:
    multi = cycle(combos(';'))
    plain = cycle(['a', 'b', 'c'])
    new = cycle(['x;y', 'y;z', 'x;z', 'x;y;z', 'x'])
    many = "'w' || (i % 300)::VARCHAR || ';w' || ((i * 7) % 300)::VARCHAR"
    times = "lpad((i % 24)::VARCHAR, 2, '0') || ':30'"

    def large(vid: str, expression: str) -> dict[str, Any]:
        return _vector(vid, generated(expression, LARGE), indices=LARGE_INDICES)

    return [
        large('large.multi-throughout', multi),
        large('large.plain-then-multi-after-500', _late(plain, multi, 500)),
        large('large.plain-then-multi-after-1000', _late(plain, multi, 1000)),
        large('large.multi-then-plain-after-500', _late(multi, plain, 500)),
        large('large.multi-then-plain-after-1000', _late(multi, plain, 1000)),
        large('large.multi-new-parts-after-1000', _late(multi, new, 1000)),
        large('large.multi-many-parts-after-1000', _late(multi, many, 1000)),
        large(
            'large.pipe-then-semicolon-after-1000',
            _late(cycle(combos('|')), multi, 1000),
        ),
        large('large.times-then-multi-after-1000', _late(times, multi, 1000)),
        large('large.nulls-first-500', _late('NULL::VARCHAR', multi, 500)),
        large('large.nulls-first-1000', _late('NULL::VARCHAR', multi, 1000)),
    ]


def _fact() -> list[dict[str, Any]]:
    return [
        _vector(
            'fact.semicolon',
            generated(None, ROWS, cycle(combos(';'))),
            table='orders',
        ),
        _vector(
            'fact.pipe',
            generated(None, ROWS, cycle(combos('|'))),
            table='orders',
        ),
        # Times of day as text, which the reference reads as times on the
        # day it runs: compared under the run-day placeholder.
        _vector(
            'fact.time-of-day',
            generated(None, ROWS, "lpad((i % 24)::VARCHAR, 2, '0') || ':45'"),
            table='orders',
            run_day=True,
        ),
        _vector(
            'fact.semicolon-8-rows',
            on_shop(cycle(combos(';'), 'order_id'), 'orders'),
            table='orders',
        ),
        _vector(
            'fact.and-entity',
            generated(cycle(combos(';')), ROWS, cycle(combos('|'))),
            table='both',
        ),
    ]


def _questions() -> list[dict[str, Any]]:
    multi = generated(cycle(combos(';')), ROWS, cycle(combos(';')))
    return [
        _vector(
            'question.static-target',
            multi,
            table='both',
            pql='PREDICT users.f FOR EACH users.user_id',
        ),
        _vector(
            'question.first-of-fact',
            multi,
            table='both',
            pql='PREDICT FIRST(orders.f, 0, 30, days) FOR EACH users.user_id',
        ),
        _vector(
            'question.list-distinct',
            multi,
            table='both',
            pql=(
                'PREDICT LIST_DISTINCT(orders.f, 0, 30, days) '
                'FOR EACH users.user_id'
            ),
        ),
        _vector(
            'question.count-distinct',
            multi,
            table='both',
            pql=(
                'PREDICT COUNT_DISTINCT(orders.f, 0, 30, days) '
                'FOR EACH users.user_id'
            ),
        ),
        _vector(
            'question.filter-on-it',
            multi,
            table='both',
            pql=(
                "PREDICT SUM(orders.price WHERE orders.f = 'a;b', 0, 30, days) "
                'FOR EACH users.user_id'
            ),
        ),
    ]


#: The vectors: each its id, the table its column ``f`` is on (``users``,
#: ``orders`` or ``both``), its setup after ``shop``'s statements, and,
#: where it varies them, its question (``pql``) and entities (``indices``).
VECTORS: list[dict[str, Any]] = [
    *_separators(),
    *_texts(),
    *_mixed(),
    *_nulls_and_empties(),
    *_singles(),
    *_large(),
    *_fact(),
    *_questions(),
]


# --- the cases -----------------------------------------------------------------


def case_for(vector: dict[str, Any], core: bool = False) -> dict[str, Any]:
    r"""The vector as a case: ``feature.bool``'s, its setup, question and
    entities the vector's; with ``core``, forced onto the core's routes
    (``empty_null_record.core_routes``), and otherwise on none (the
    runner's own choice, as the reference has)."""
    template = json.loads(TEMPLATE.read_text(encoding='utf-8'))
    case = copy.deepcopy(template)
    case['id'] = CASE_PREFIX + vector['id']
    case['status'] = 'diagnostic'
    case['routes'] = EN.core_routes(vector) if core else []
    case['cells'] = []
    case['input']['setup'] = list(vector['setup'])
    case['input']['pql'] = vector.get('pql', QUESTION)
    indices = list(vector.get('indices', INDICES))
    case['input']['materialize']['indices'] = indices
    case['core']['describe_source']['source'] = EN.core_source(None)
    case['core']['materialize']['indices'] = EN.core_indices(indices)
    case['expect'] = {'outcome': 'success', 'stage': 'predict'}
    case['why'] = f'T1.32e part 2 vector {vector["id"]}'
    case['source'] = 'contract/tools/multicategorical_record.py'
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


def _sections(parsed: Any) -> list[tuple[str, dict[str, Any]]]:
    r"""Each table a request body carries rows of, by its path in the body
    (``context.related_tables.users``, ``predict.instance_table`` and the
    like), in the body's order."""
    found: list[tuple[str, dict[str, Any]]] = []
    if not isinstance(parsed, dict):
        return found
    for section in ('context', 'predict'):
        body = parsed.get(section)
        if not isinstance(body, dict):
            continue
        instance = body.get('instance_table')
        if isinstance(instance, dict):
            found.append((f'{section}.instance_table', instance))
        related = body.get('related_tables')
        if isinstance(related, dict):
            for name, table in related.items():
                if isinstance(table, dict):
                    found.append((f'{section}.related_tables.{name}', table))
    return found


def _declared(parsed: dict[str, Any], path: str, column: str) -> Any:
    r"""The request schema's declaration of ``column`` in the table at
    ``path``: its dtype and stype (``None`` where it gives none)."""
    schema = parsed.get('schema')
    if not isinstance(schema, dict):
        return None
    if path.endswith('.instance_table'):
        table = schema.get('instance_table')
    else:
        table = (schema.get('related_tables') or {}).get(path.rsplit('.', 1)[1])
    if not isinstance(table, dict):
        return None
    declared = (table.get('columns') or {}).get(column)
    if not isinstance(declared, dict):
        return None
    return {'dtype': declared.get('dtype'), 'stype': declared.get('stype')}


#: A table's cells of the column are kept whole up to this many rows (the
#: predicted entities' tables); past it, summarized.
WHOLE = 20
#: How many of the distinct cells a summary lists, sorted.
LISTED = 20


def _kind(cell: Any) -> str:
    if cell is None:
        return 'null'
    if isinstance(cell, str):
        return 'string'
    if isinstance(cell, list):
        return 'list'
    if isinstance(cell, dict) and set(cell) == {'number'}:
        return 'number'
    return type(cell).__name__


def cells_digest(cells: list[Any]) -> str:
    r"""The SHA-256 of a column's cells in the order sent, as the
    ``payload`` rule writes a body (numbers by kind and exact value, sorted
    keys, no spaces, UTF-8 unescaped): ``sdm-testkit``'s
    ``tests/t132e_multicategorical.rs`` writes the same bytes."""
    canonical = json.dumps(
        EN._canonical(cells),
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def column_on_the_wire(declared: Any, cells: list[Any]) -> dict[str, Any]:
    r"""One table's column as sent: its declaration, its row and null
    counts, the kinds of its cells (a string, a list of parts, a null), the
    count of its distinct cells and the first :data:`LISTED` of them in
    sorted order, the digest of every cell in order, and, up to
    :data:`WHOLE` rows, the cells themselves."""
    distinct = sorted(
        {json.dumps(cell, sort_keys=True, ensure_ascii=False) for cell in cells}
    )
    out: dict[str, Any] = {
        'declared': declared,
        'rows': len(cells),
        'nulls': sum(cell is None for cell in cells),
        'kinds': sorted({_kind(cell) for cell in cells}),
        'distinct': len(distinct),
        'listed': [json.loads(cell) for cell in distinct[:LISTED]],
        'digest': cells_digest(cells),
    }
    if len(cells) <= WHOLE:
        out['cells'] = cells
    return out


def wire(payloads: Any, column: str = 'f') -> Any:
    r"""How ``column`` travels in each request: per table carrying it
    (``column_on_the_wire``), in the body's order; null when no request
    was made."""
    if not payloads:
        return None
    out = []
    for each in payloads:
        parsed = each.get('parsed')
        tables: dict[str, Any] = {}
        for path, table in _sections(parsed):
            columns = table.get('columns')
            rows = table.get('rows')
            if not isinstance(columns, list) or column not in columns:
                continue
            at = columns.index(column)
            tables[path] = column_on_the_wire(
                _declared(parsed, path, column),
                [
                    row[at] if isinstance(row, list) and at < len(row) else None
                    for row in rows or []
                ],
            )
        out.append(tables)
    return out


#: What stands for the run's day in a vector marked ``run_day``.
RUN_DAY = '<run-day>'
_TIMESTAMP = re.compile(r'(?P<day>[0-9]{4}-[0-9]{2}-[0-9]{2})(?P<rest>T.*)')


def at_run_day(parsed: Any, column: str = 'f') -> Any:
    r"""A request body with the day of each timestamp cell of ``column``
    written as :data:`RUN_DAY`, where every such cell of the body names the
    same day: the day the run happened, for text holding times of day,
    which the reference (pandas) reads as times on the day it runs. Cells of
    two days, or of another column, are left as they are, so a body that
    is not the run-day case still differs."""
    days = {
        found['day']
        for _, table in _sections(parsed)
        if isinstance(table.get('columns'), list) and column in table['columns']
        for row in table.get('rows') or []
        if isinstance(row, list)
        and table['columns'].index(column) < len(row)
        and isinstance(cell := row[table['columns'].index(column)], str)
        and (found := _TIMESTAMP.fullmatch(cell)) is not None
    }
    if len(days) != 1:
        return parsed
    out = copy.deepcopy(parsed)
    for _, table in _sections(out):
        columns = table.get('columns')
        if not isinstance(columns, list) or column not in columns:
            continue
        at = columns.index(column)
        for row in table.get('rows') or []:
            if not isinstance(row, list) or at >= len(row):
                continue
            found = (
                _TIMESTAMP.fullmatch(row[at])
                if isinstance(row[at], str)
                else None
            )
            if found is not None:
                row[at] = RUN_DAY + found['rest']
    return out


def project(
    observation: dict[str, Any], vector: dict[str, Any]
) -> dict[str, Any]:
    r"""The public answer of one run of ``vector``, as both Python runners
    record it: ``empty_null_record.project``'s, and the column's ``wire``;
    for a vector marked ``run_day``, each request read through
    :func:`at_run_day` first (its digest and wire included)."""
    if vector.get('run_day'):
        observation = copy.deepcopy(observation)
        for each in (observation.get('public') or {}).get('payloads') or []:
            each['parsed'] = at_run_day(each.get('parsed'))
    answer: dict[str, Any] = EN.project(observation, vector)
    answer['wire'] = wire((observation.get('public') or {}).get('payloads'))
    return answer


outcome_class = EN.outcome_class
register = EN.register


# --- what DuckDB holds ---------------------------------------------------------


def duckdb_holds(vector: dict[str, Any]) -> dict[str, Any]:
    r"""The tables' row counts after the setup, ``f``'s type on each table
    holding it, and per such table how many times each separator the
    reference counts occurs in ``f``'s first 500 rows, its 1,000-row
    preview and the whole table (DuckDB's own order, the preview's), as
    the reference counts them, character by character."""
    import duckdb

    connection = duckdb.connect()
    try:
        for statement in EN.shop() + vector['setup']:
            connection.execute(statement)
        rows = {
            name: connection.execute(
                f'SELECT count(*) FROM "{name}"'
            ).fetchone()[0]
            for name in ('users', 'orders')
        }
        tables = (
            ['users', 'orders']
            if vector['table'] == 'both'
            else [vector['table']]
        )
        out: dict[str, Any] = {'rows': rows, 'f': {}}
        for table in tables:
            found = connection.execute(
                'SELECT data_type FROM information_schema.columns '
                "WHERE table_name = ? AND column_name = 'f'",
                [table],
            ).fetchone()
            values = [
                v
                for (v,) in connection.execute(
                    f'SELECT f FROM "{table}"'
                ).fetchall()
            ]

            def held(part: list[Any]) -> dict[str, int]:
                return {
                    name: sum(v.count(sep) for v in part if isinstance(v, str))
                    for name, sep in SEPARATORS.items()
                }

            out['f'][table] = {
                'type': found[0] if found else None,
                'nulls': sum(v is None for v in values),
                'first_500': held(values[:500]),
                'preview': held(values[:1000]),
                'whole': held(values),
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
    ids = [v['id'] for v in VECTORS]
    if len(set(ids)) != len(ids):
        print('a vector id twice', file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix='t132e2-record-') as tmp:
        observed = run('python-ref', VECTORS, Path(tmp), reference=reference)
    vectors = []
    for vector in VECTORS:
        answer = project(observed[vector['id']], vector)
        why = register(answer)
        entry = {key: copy.deepcopy(value) for key, value in vector.items()}
        entry['duckdb'] = duckdb_holds(vector)
        entry['reference'] = answer
        entry['register'] = {'required': bool(why), 'why': why}
        vectors.append(entry)
        outcome = answer['outcome']
        stypes = {
            table['name']: column['stype']
            for table in (answer['schema'] or {}).get('tables', [])
            for column in table['columns']
            if column['name'] == 'f'
        }
        print(
            vector['id'],
            outcome['result'],
            outcome['stage'],
            outcome['error'] or '',
            outcome['code'] or '',
            stypes,
            answer['warnings'],
            '(register)' if why else '',
            file=sys.stderr,
        )
    document = {
        'format': FORMAT,
        'recorded_by': 'contract/tools/multicategorical_record.py',
        'reference': commit,
        'cell': 'ref-pandas3',
        'environment': {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'pyarrow': pa.__version__,
            'numpy': np.__version__,
            'duckdb': duckdb.__version__,
        },
        'template': 'contract/cases/s1/feature.bool.json',
        'question': QUESTION,
        'indices': INDICES,
        'register_ref': REGISTER_REF,
        'payload_exclusions': EN.exclusions(),
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
