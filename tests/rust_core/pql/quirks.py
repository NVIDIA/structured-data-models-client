#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Record today's Python answer beside every grammar and date quirk the
Phase 5 brief lists, for the table-driven Rust test.

    quirks.py [--check]

Writes ``rust/crates/pql/tests/accept_quirks.json``: one row per input, each
naming the quirk it pins (``QUIRKS``), with what today's
``predictive_query.validate`` gives for a query, or what ``pd.to_datetime``
gives for a date literal, as ``_pql`` records them. ``--check`` writes
nothing and fails if the stored table differs from what today's Python gives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pql import PQL_CRATE, python_date, python_result, schemas  # noqa: E402

TABLE = PQL_CRATE / 'tests' / 'accept_quirks.json'

#: Every quirk the brief lists ("Grammar quirks", "Syntax errors", "Dates"),
#: by the id the table's rows use.
QUIRKS = {
    'case-keywords': 'keywords match only exact lower or UPPER case',
    'case-aggregations': 'aggregations match only exact lower or UPPER case',
    'case-operators': 'relational operators match only exact lower or UPPER '
    'case',
    'case-null': 'NULL matches only NULL or null',
    'case-time-units': 'time units match only exact lower or UPPER case',
    'case-bool': 'BOOL also accepts True and False',
    'multiword-spacing': 'multi-word tokens need exactly one space',
    'multiword-case': 'multi-word tokens need one case throughout',
    'longest-token': 'the longest token wins',
    'first-rule-on-tie': 'on equal length, the rule listed first wins',
    'date-vs-decimal': '2024.01.02 is a DATE, 2024.01 a DECIMAL',
    'fqn-then-id': 'a.b.c is FQN_COLUMN(a.b) then ID(c)',
    'precedence': 'NOT binds tighter than AND, which binds tighter than OR',
    'decimal-sign': "DECIMAL reuses INT's sign, so --3.5 lexes",
    'date-digit-counts': 'DATE accepts any digit counts',
    'whitespace': 'whitespace includes the Unicode spaces at '
    'PQLGrammar.g4:214, and nothing else',
    'comment': '# starts a comment, to the end of the line',
    'backticks': 'backticks quote identifiers',
    'one-element-array': 'the grammar actions reject a one-element array',
    'syntax-no-span': 'syntax errors are PQL_PARSE_SYNTAX with no span',
    'code-point-columns': "columns count code points, as ANTLR's Python "
    'runtime does, from 0',
    'lines': 'rows count lines from 1, ended by \\n',
    'end-col': 'end_col is the start column of the last token',
    'int-bounds': 'integers are bounded by int64, window offsets by int32 '
    '(np.iinfo)',
    'date-month-first': 'an ambiguous date is read month first',
    'date-day-first-fallback': 'a date impossible month first is read day '
    'first, with a warning',
    'date-two-digit-year': 'two-digit years',
    'date-time-part': 'the time part',
    'date-refused': 'impossible dates are refused',
}

SHOP, QUOTED, PEOPLE = 'shop', 'quoted', 'people'
UNICODE_WS = '\t\n\r\x0b\x0c\x1c\x1d\x1e\x1f\x85\xa0                　'
COUNT = 'COUNT(orders.*, 0, 30, days)'
USERS = 'FOR EACH users.user_id'


def _q(target: str = COUNT, entity: str = USERS, rest: str = '') -> str:
    text = f'PREDICT {target} {entity}'
    return f'{text} {rest}' if rest else text


def _where(condition: str) -> str:
    return _q(rest=f'WHERE {condition}')


def query_rows() -> list[tuple[str, str, str, str, str]]:
    r"""``(quirk, id, schema, profile, query)`` for every query row."""
    rows: list[tuple[str, str, str, str, str]] = []

    def add(quirk, name, query, schema=SHOP, profile='RELATIONAL'):
        assert quirk in QUIRKS, quirk
        rows.append((quirk, f'{quirk}/{name}', schema, profile, query))

    # Keywords, in each case the grammar takes and some it does not.
    add(
        'case-keywords',
        'lower',
        'predict count(orders.*, 0, 30, days) '
        'for each users.user_id where users.age > 3',
    )
    add(
        'case-keywords',
        'upper',
        'PREDICT COUNT(orders.*, 0, 30, DAYS) '
        'FOR EACH users.user_id WHERE users.age > 3',
    )
    add('case-keywords', 'title-predict', f'Predict {COUNT} {USERS}')
    add('case-keywords', 'title-where', _q(rest='Where users.age > 3'))
    add('case-keywords', 'title-assuming', _q(rest='Assuming users.age > 3'))
    add(
        'case-keywords',
        'upper-assuming',
        _q(
            target='COUNT(orders.*, 0, 30, days) > 0',
            rest='ASSUMING COUNT(reviews.*, 0, 30, days) > 1',
        ),
    )
    add(
        'case-keywords',
        'lower-rank-top',
        _q(target='LIST_DISTINCT(orders.product_id, 0, 30, days) rank top 5'),
    )
    add(
        'case-keywords',
        'upper-rank-top',
        _q(target='LIST_DISTINCT(orders.product_id, 0, 30, days) RANK TOP 5'),
    )
    add(
        'case-keywords',
        'title-rank',
        _q(target='LIST_DISTINCT(orders.product_id, 0, 30, days) Rank TOP 5'),
    )
    add(
        'case-keywords',
        'lower-classify',
        _q(target='COUNT(orders.*, 0, 30, days) > 0 classify'),
    )
    add(
        'case-keywords',
        'lower-forecast',
        _q(target='SUM(orders.amount, 0, 7, days) forecast 3 timeframes'),
    )
    add(
        'case-keywords',
        'title-timeframes',
        _q(target='SUM(orders.amount, 0, 7, days) FORECAST 3 Timeframes'),
    )
    add('case-keywords', 'mixed-not', _where('nOT users.age > 3'))
    add(
        'case-keywords',
        'lower-and-or',
        _where('users.age > 3 and users.age < 9 or users.vip = true'),
    )
    add('case-keywords', 'title-and', _where('users.age > 3 And users.age < 9'))
    # Aggregations.
    for name, text in (
        ('lower', 'count'),
        ('upper', 'COUNT'),
        ('title', 'Count'),
        ('lower-distinct', 'count_distinct'),
        ('mixed-distinct', 'COUNT_distinct'),
        ('lower-list', 'list_distinct'),
    ):
        column = (
            'orders.*'
            if 'distinct' not in text.lower()
            else ('orders.product_id')
        )
        add(
            'case-aggregations',
            name,
            _q(target=f'{text}({column}, 0, 30, days)'),
        )
    add(
        'case-aggregations',
        'lower-sum',
        _q(target='sum(orders.amount, 0, 30, days)'),
    )
    add(
        'case-aggregations',
        'title-avg',
        _q(target='Avg(orders.amount, 0, 30, days)'),
    )
    # Relational operators.
    for name, op in (
        ('lower-like', "like '%US%'"),
        ('upper-like', "LIKE '%US%'"),
        ('title-like', "Like '%US%'"),
        ('lower-not-like', "not like '%US%'"),
        ('upper-not-like', "NOT LIKE '%US%'"),
        ('lower-starts-with', "starts with 'U'"),
        ('upper-ends-with', "ENDS WITH 'S'"),
        ('title-starts-with', "Starts With 'U'"),
        ('lower-in', "in ('US', 'FR')"),
        ('upper-is-in', "IS IN ('US', 'FR')"),
        ('upper-contains', "CONTAINS 'U'"),
        ('lower-not-contains', "not contains 'U'"),
    ):
        add('case-operators', name, _where(f'users.country {op}'))
    add('case-operators', 'lower-is-not', _where('users.age is not NULL'))
    add('case-operators', 'title-is', _where('users.age Is NULL'))
    # NULL.
    add('case-null', 'upper', _where('users.age IS NULL'))
    add('case-null', 'lower', _where('users.age IS null'))
    add('case-null', 'title', _where('users.age IS Null'))
    # Time units.
    add('case-time-units', 'lower', _q(target='COUNT(orders.*, 0, 3, weeks)'))
    add('case-time-units', 'upper', _q(target='COUNT(orders.*, 0, 3, HOURS)'))
    add('case-time-units', 'title', _q(target='COUNT(orders.*, 0, 3, Days)'))
    add('case-time-units', 'unknown', _q(target='COUNT(orders.*, 0, 3, years)'))
    # BOOL.
    for value in ('true', 'false', 'True', 'False', 'TRUE', 'FALSE', 'tRUE'):
        add('case-bool', value, _where(f'users.vip = {value}'))
    # Multi-word tokens: exactly one space, one case.
    add('multiword-spacing', 'for-each-one-space', _q())
    add(
        'multiword-spacing',
        'for-each-two-spaces',
        _q(entity='FOR  EACH users.user_id'),
    )
    add(
        'multiword-spacing',
        'for-each-tab',
        _q(entity='FOR\tEACH users.user_id'),
    )
    add(
        'multiword-spacing',
        'for-each-newline',
        _q(entity='FOR\nEACH users.user_id'),
    )
    add(
        'multiword-spacing',
        'for-each-ideographic',
        _q(entity='FOR　EACH users.user_id'),
    )
    for op in (
        'IS NOT',
        'IS IN',
        'NOT LIKE',
        'NOT CONTAINS',
        'STARTS WITH',
        'ENDS WITH',
    ):
        value = {
            'IS NOT': 'NULL',
            'IS IN': "('US', 'FR')",
            'NOT LIKE': "'%US%'",
        }.get(op, "'U'")
        slug = op.lower().replace(' ', '-')
        add(
            'multiword-spacing',
            f'{slug}-one-space',
            _where(f'users.country {op} {value}'),
        )
        add(
            'multiword-spacing',
            f'{slug}-two-spaces',
            _where(f'users.country {op.replace(" ", "  ")} {value}'),
        )
        add(
            'multiword-case',
            f'{slug}-mixed',
            _where(
                f'users.country {op.split()[0]} {op.split()[1].lower()} {value}'
            ),
        )
    add('multiword-case', 'for-each-mixed', _q(entity='FOR each users.user_id'))
    add('multiword-case', 'for-each-lower', _q(entity='for each users.user_id'))
    # Longest token wins; on a tie, the rule listed first.
    add('longest-token', 'countx', _q(target='countx(orders.*, 0, 30, days)'))
    add('longest-token', 'predictx', f'PREDICTX {COUNT} {USERS}')
    add('longest-token', 'trueish', _where('users.vip = trueish'))
    add('longest-token', 'nullx', _where('users.age IS nullx'))
    add(
        'longest-token',
        'for-each-over-for',
        _q(entity='FOR EACH users.user_id'),
    )
    add('longest-token', 'for-alone', _q(entity='FOR users.user_id = 1'))
    add(
        'longest-token',
        'not-like-over-not',
        _where("users.country NOT LIKE '%x%'"),
    )
    add(
        'longest-token',
        'neg-infinity',
        _q(target='COUNT(orders.*, -infinity, 0, days)'),
    )
    add(
        'longest-token',
        'neg-inf-then-id',
        _q(target='COUNT(orders.*, -infx, 0, days)'),
    )
    add('longest-token', 'exponent-decimal', _where('users.age > 1e1'))
    add('longest-token', 'int', _where('users.age > 10'))
    add('first-rule-on-tie', 'bool-over-id', _where('users.vip = true'))
    add(
        'first-rule-on-tie',
        'time-unit-over-id',
        _q(target='COUNT(orders.*, 0, 3, days)'),
    )
    add(
        'first-rule-on-tie',
        'keyword-as-table',
        _q(target='COUNT(days.*, 0, 3, days)'),
    )
    add(
        'first-rule-on-tie',
        'aggregation-over-id',
        _q(target='count(orders.*, 0, 3, days)'),
    )
    add('date-vs-decimal', 'date', _where('users.signup > 2024.01.02'))
    add('date-vs-decimal', 'decimal', _where('users.age > 2024.01'))
    add('date-vs-decimal', 'decimal-on-time', _where('users.signup > 2024.01'))
    add('fqn-then-id', 'target', _q(target='orders.amount.c'))
    add('fqn-then-id', 'where', _where('users.age.x > 3'))
    add('fqn-then-id', 'entity', _q(entity='FOR EACH a.b.c'))
    # Precedence.
    a = 'COUNT(orders.*, 0, 30, days) > 0'
    b = 'COUNT(reviews.*, 0, 30, days) > 0'
    c = 'COUNT(orders.*, 0, 7, days) > 1'
    add('precedence', 'or-and-not', _q(target=f'{a} OR {b} AND NOT {c}'))
    add('precedence', 'not-and-or', _q(target=f'NOT {a} AND {b} OR {c}'))
    add('precedence', 'parenthesised', _q(target=f'({a} OR {b}) AND {c}'))
    add('precedence', 'not-parenthesised', _q(target=f'NOT ({a} OR {b})'))
    add('precedence', 'and-chain', _q(target=f'{a} AND {b} AND {c}'))
    add('precedence', 'or-chain', _q(target=f'{a} OR {b} OR {c}'))
    add(
        'precedence',
        'filter',
        _where('users.age > 3 OR users.age < 1 AND NOT users.vip = true'),
    )
    # DECIMAL's sign.
    for name, value in (
        ('double-minus-decimal', '--3.5'),
        ('minus-decimal', '-3.5'),
        ('double-minus-int', '--3'),
        ('double-minus-exponent', '--3e2'),
        ('minus-minus-exponent', '-3e-2'),
    ):
        add('decimal-sign', name, _where(f'orders.amount > {value}'))
    # DATE's digits.
    for name, value in (
        ('one-digit-each', '1-1-1'),
        ('zero-padded', '0001-01-01'),
        ('long-year', '12345-1-1'),
        ('long-day', '2024-1-001'),
        ('dots', '1.1.1'),
        ('slashes', '1/2/3'),
        ('with-time', '2024-1-1 1:2:3'),
        ('long-time-parts', '2024-1-1 001:002:003'),
    ):
        add('date-digit-counts', name, _where(f'users.signup > {value}'))
    # Whitespace: each character the grammar skips, and two it does not.
    for char in UNICODE_WS:
        add(
            'whitespace',
            f'u{ord(char):04x}',
            f'PREDICT{char}{COUNT}{char}FOR EACH{char}users.user_id',
        )
    for char in ('​', '᠎', '﻿'):
        add(
            'whitespace',
            f'not-u{ord(char):04x}',
            f'PREDICT{char}{COUNT} {USERS}',
        )
    # Comments.
    add('comment', 'line', f'PREDICT {COUNT} # the window\n{USERS}')
    add('comment', 'at-end', f'PREDICT {COUNT} {USERS} # no newline after')
    add('comment', 'carriage-return', f'PREDICT {COUNT} # cr\r{USERS}')
    add('comment', 'hides-the-rest', f'PREDICT {COUNT} # {USERS}')
    add('comment', 'inside-string', _where("users.country = 'a#b'"))
    add('comment', 'only', '# PREDICT')
    # Backticks.
    add(
        'backticks',
        'both',
        _q(
            target='COUNT(`orders`.*, 0, 30, days)',
            entity='FOR EACH `users`.`user_id`',
        ),
    )
    add(
        'backticks',
        'spaces',
        'PREDICT SUM(`Order Lines`.`Unit Price`, 0, '
        '30, days) FOR EACH Customers.`Customer ID`',
        schema=QUOTED,
    )
    add(
        'backticks',
        'unknown-with-dot',
        _q(target='COUNT(`orders.x`.*, 0, 30, days)'),
    )
    add('backticks', 'keyword-name', _where('users.`where` > 3'))
    add('backticks', 'empty', _where('users.`` > 3'))
    add('backticks', 'newline-inside', _where('users.`a\nb` > 3'))
    add('backticks', 'unclosed', _where('users.`age > 3'))
    add('backticks', 'non-ascii', _where('users.`âge` > 3'))
    # One-element arrays.
    for name, value in (
        ('one', '(1)'),
        ('one-trailing-comma', '(1,)'),
        ('two', '(1, 2)'),
        ('nested', '((1, 2), 3)'),
        ('nested-one', '((1), 2)'),
    ):
        add('one-element-array', name, _where(f'users.age IN {value}'))
    # Syntax errors carry no span.
    for name, query in (
        ('empty', ''),
        ('blank', '   '),
        ('no-predict', f'{COUNT} {USERS}'),
        ('no-for-each', f'PREDICT {COUNT}'),
        ('trailing', _q(rest='extra')),
        ('unclosed-string', _where("users.country = 'US")),
        ('bad-escape', _where("users.country = 'a\\qb'")),
        ('bad-char', _q(rest=';')),
        ('window-backwards', _q(target='COUNT(orders.*, 30, 0, days)')),
        ('not-like-without-percent', _where("users.country NOT LIKE 'x'")),
    ):
        add('syntax-no-span', name, query)
    # Positions: code points, rows from 1, end_col.
    for name, text in (
        ('ascii', 'ab'),
        ('latin', 'é'),
        ('astral', '😀😀'),
        ('combining', 'é'),
    ):
        add(
            'code-point-columns',
            f'{name}-error',
            _where(f"users.country = '{text}' AND users.nope = 1"),
        )
        add(
            'code-point-columns',
            f'{name}-tree',
            _q(
                target=f"COUNT(orders.* WHERE orders.status = '{text}', 0, 30, "
                'days)'
            ),
        )
    add('code-point-columns', 'quoted-name', _where('users.`😀` = 1'))
    add('lines', 'second-line', f'PREDICT {COUNT}\nFOR EACH users.nope')
    add('lines', 'crlf', f'PREDICT {COUNT}\r\nFOR EACH users.nope')
    add('lines', 'cr-only', f'PREDICT {COUNT}\rFOR EACH users.nope')
    add('lines', 'tree-on-three-lines', f'PREDICT\n{COUNT}\n{USERS}')
    add('end-col', 'aggregation', _q(target='SUM(orders.nope, 0, 30, days)'))
    add(
        'end-col',
        'condition',
        _q(target='SUM(orders.amount, 0, 30, days) > 100'),
    )
    add(
        'end-col',
        'filter',
        _q(
            target='COUNT(orders.* WHERE orders.status = '
            "'x' AND orders.nope > 1, 0, 30, days)"
        ),
    )
    # Integer bounds.
    for name, value in (
        ('int64-max', '9223372036854775807'),
        ('int64-over', '9223372036854775808'),
        ('int64-min', '-9223372036854775808'),
        ('int64-under', '-9223372036854775809'),
    ):
        add('int-bounds', name, _where(f'users.age > {value}'))
    for name, value in (
        ('int32-max', '2147483647'),
        ('int32-over', '2147483648'),
    ):
        add(
            'int-bounds',
            f'window-{name}',
            _q(target=f'COUNT(orders.*, 0, {value}, days)'),
        )
    add(
        'int-bounds',
        'window-int32-min',
        _q(target='COUNT(orders.*, -2147483648, 0, days)'),
    )
    add(
        'int-bounds',
        'window-int32-under',
        _q(target='COUNT(orders.*, -2147483649, 0, days)'),
    )
    add(
        'int-bounds',
        'top-k-over',
        _q(
            target='LIST_DISTINCT(orders.product_id, 0, 30, days) RANK TOP '
            '2147483648'
        ),
    )
    # Dates inside a query: what validation makes of them.
    for quirk, name, value in (
        ('date-month-first', 'dots', '01.02.2024'),
        ('date-month-first', 'slashes', '01/02/2024'),
        ('date-day-first-fallback', 'leap-day', '29.02.2024'),
        ('date-day-first-fallback', 'thirteenth', '13/01/2024'),
        ('date-two-digit-year', 'last', '24-01-05'),
        ('date-two-digit-year', 'all-one-digit', '1-1-1'),
        ('date-time-part', 'noon', '2024-01-02 12:30:00'),
        ('date-time-part', 'hour-24', '2024-01-02 24:00:00'),
        ('date-refused', 'not-a-leap-year', '2023-02-29'),
        ('date-refused', 'month-13', '2024-13-01'),
    ):
        add(quirk, f'query-{name}', _where(f'users.signup > {value}'))
    return rows


def date_rows() -> list[tuple[str, str, str]]:
    r"""``(quirk, id, literal)`` for every date-literal row."""
    rows: list[tuple[str, str, str]] = []

    def add(quirk: str, literal: str) -> None:
        assert quirk in QUIRKS, quirk
        rows.append((quirk, f'{quirk}/{literal}', literal))

    for literal in (
        '01.02.2024',
        '01/02/2024',
        '01-02-2024',
        '1/2/2024',
        '12/11/2024',
        '2024-01-02',
        '2024/01/02',
        '2024.01.02',
    ):
        add('date-month-first', literal)
    for literal in (
        '29.02.2024',
        '13/01/2024',
        '31-12-2024',
        '31-1-2024',
        '30/04/2024 10:00:00',
        '29.02.2023',
        '31/04/2024',
    ):
        add('date-day-first-fallback', literal)
    for literal in (
        '24-01-05',
        '1-1-1',
        '01-02-03',
        '05/06/07',
        '32-01-01',
        '00-01-01',
        '1/2/3',
        '12.12.12',
        '2024-1-1',
        '100-1-1',
        '0001-01-01',
        '9999-12-31',
        '10000-01-01',
    ):
        add('date-two-digit-year', literal)
    for literal in (
        '2024-01-02 23:59:59',
        '2024-01-02 00:00:00',
        '2024-01-02 1:2:3',
        '2024-01-02 001:002:003',
        '2024-01-02 24:00:00',
        '2024-01-02 23:60:00',
        '2024-01-02 23:59:60',
        '1/2/2024 12:00:00',
        '230203-06-09',
        '20240101-01-01',
    ):
        add('date-time-part', literal)
    for literal in (
        '2023-02-29',
        '2024-02-29',
        '1900-02-29',
        '2000-02-29',
        '2100-02-29',
        '2024-13-01',
        '2024-00-10',
        '2024-04-31',
        '2024-01-00',
        '13/13/2024',
        '0-0-0',
        '2024-001-001',
        '1-1-20240101',
    ):
        add('date-refused', literal)
    for literal in ('1-1-1', '123-4-5', '1.2.3456', '0001/1/1'):
        add('date-digit-counts', literal)
    return rows


def record() -> dict[str, Any]:
    named = dict(zip((SHOP, QUOTED, PEOPLE), schemas(), strict=True))
    assert list(named[QUOTED]['tables']) == ['Order Lines', 'Customers']
    cases = []
    for quirk, id_, schema, profile, query in query_rows():
        cases.append(
            {
                'id': id_,
                'quirk': quirk,
                'query': query,
                'schema': schema,
                'profile': profile,
                'python': python_result(query, named[schema], profile),
            }
        )
    dates = [
        {
            'id': id_,
            'quirk': quirk,
            'literal': literal,
            'python': python_date(literal),
        }
        for quirk, id_, literal in date_rows()
    ]
    ids = [row['id'] for row in cases + dates]
    assert len(ids) == len(set(ids)), 'quirk row ids are not unique'
    for row in cases:
        assert 'exception' not in row['python'], row
    return {
        'quirks': QUIRKS,
        'schemas': named,
        'queries': cases,
        'dates': dates,
    }


def render(table: dict[str, Any]) -> str:
    return json.dumps(table, indent=1) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    text = render(record())
    if args.check:
        if not TABLE.exists() or TABLE.read_text(encoding='utf-8') != text:
            print(f"{TABLE} is not what today's Python gives", file=sys.stderr)
            return 1
        print(f"{TABLE.name} matches today's Python")
        return 0
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(text, encoding='utf-8')
    print(f'wrote {TABLE}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
