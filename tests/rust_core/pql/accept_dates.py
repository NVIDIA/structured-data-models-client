# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P5.dates: the core reads date literals as today's PQL does, with
``pd.to_datetime``, without calling it.

- 100,000 literals generated over every shape the grammar's ``DATE`` token
  accepts (``_generate.date_literals``, seeded with ``SEED``) give, from
  ``pql_parse_date``, the same date, the same refusals and the same warnings
  as pandas.
- ``rust/crates/pql/DATES.md`` states the rules, and every rule has at least
  one example in its examples table, each checked against pandas and the
  core (``CONTRACT.md`` gives the layout).
"""

from __future__ import annotations

import collections
import re

from _generate import date_literals
from _pql import PQL_CRATE, python_date, rust_date

SEED = 20261003
COUNT = 100_000
DATES_MD = PQL_CRATE / 'DATES.md'
# The grammar's datetime rule, DATE TIME?, as the tree joins it.
SHAPE = re.compile(r'(\d+)([/.-])(\d+)\2(\d+)( \d+:\d+:\d+)?')
RULE = re.compile(r'^\s*(?:[-*+]\s+|#{1,6}\s+|\d+\.\s+)?\*\*(D\d+)\b', re.M)
BACKTICKED = re.compile(r'`([^`]+)`')


def test_generated_literals_match_pandas():
    seen: collections.Counter[str] = collections.Counter()
    failures, total = [], 0
    for literal in date_literals(SEED, COUNT):
        total += 1
        match = SHAPE.fullmatch(literal)
        assert match, f'the generator wrote {literal!r}, not a DATE'
        want = python_date(literal)
        try:
            got = rust_date(literal)
        except Exception as error:  # noqa: BLE001 - reported per literal
            failures.append(f'{literal!r}: raised {error!r}')
            continue
        if got != want:
            failures.append(f'{literal!r}: {got} != pandas {want}')
        fields = [match.group(i) for i in (1, 3, 4)]
        seen[f'sep {match.group(2)}'] += 1
        seen['time' if match.group(5) else 'no time'] += 1
        seen['refused' if want['value'] is None else 'read'] += 1
        if want['warnings']:
            seen['warned'] += 1
        if want['value'] and want['value'][5:10] == '02-29':
            seen['leap day'] += 1
        if want['value'] and want['value'][8:10] in ('30', '31'):
            seen['month end'] += 1
        if want['value'] and max(len(f) for f in fields) <= 2:
            seen['two-digit year'] += 1
        if want['value'] and len(want['value']) > 19:
            seen['zone or fraction'] += 1
    summary = f'seed {SEED}: {total} literals, {dict(seen)}'
    print(summary)
    assert not failures, (
        f'{summary}\n{len(failures)} differ from pandas:\n'
        + '\n'.join(failures[:60])
    )
    assert total == COUNT
    for kind, floor in (
        ('sep /', 20_000),
        ('sep -', 20_000),
        ('sep .', 20_000),
        ('time', 20_000),
        ('read', 30_000),
        ('refused', 30_000),
        ('warned', 3_000),
        ('leap day', 300),
        ('month end', 3_000),
        ('two-digit year', 3_000),
    ):
        assert seen[kind] >= floor, f'{kind}: {seen[kind]} < {floor}; {summary}'


def _examples(text: str) -> list[tuple[str, str, str]]:
    r"""``(rule, literal, result)`` for each row of each examples table."""
    rows, in_table = [], False
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if not line.strip().startswith('|'):
            in_table = False
            continue
        if [c.lower() for c in cells[:3]] == ['rule', 'literal', 'result']:
            in_table = True
            continue
        if not in_table or set(''.join(cells)) <= set('-: '):
            continue
        rows.append((cells[0], cells[1], cells[2] if len(cells) > 2 else ''))
    return rows


def test_dates_md_states_rules_each_with_examples_that_hold():
    assert DATES_MD.is_file(), f'{DATES_MD} does not exist'
    text = DATES_MD.read_text(encoding='utf-8')
    rules = RULE.findall(text)
    assert len(rules) == len(set(rules)), f'a rule id repeats: {rules}'
    assert len(rules) >= 6, f'DATES.md states {len(rules)} rules: {rules}'
    rows = _examples(text)
    assert rows, 'DATES.md has no | Rule | Literal | Result | table'
    covered: set[str] = set()
    failures = []
    kinds: collections.Counter[str] = collections.Counter()
    for rule, literal_cell, result in rows:
        literal = BACKTICKED.fullmatch(literal_cell)
        if rule not in rules:
            failures.append(f'{rule!r}: not a rule DATES.md states')
            continue
        if not literal or not SHAPE.fullmatch(literal.group(1)):
            failures.append(f'{rule}: {literal_cell!r} is not a DATE literal')
            continue
        literal = literal.group(1)
        covered.add(rule)
        if 'refused' in result.lower():
            stated = None
        else:
            value = BACKTICKED.search(result)
            if not value:
                failures.append(f'{rule} {literal}: no result in {result!r}')
                continue
            stated = value.group(1)
        warns = 'warning' in result.lower()
        want = python_date(literal)
        if (stated, warns) != (want['value'], bool(want['warnings'])):
            failures.append(
                f'{rule} {literal}: DATES.md says {stated} '
                f'{"with" if warns else "without"} a warning; pandas gives '
                f'{want}'
            )
        got = rust_date(literal)
        if got != want:
            failures.append(
                f'{rule} {literal}: the core gives {got}, pandas {want}'
            )
        match = SHAPE.fullmatch(literal)
        assert match
        fields = [match.group(i) for i in (1, 3, 4)]
        if want['value'] is None:
            kinds['refused'] += 1
        if want['warnings']:
            kinds['day first, with a warning'] += 1
        if match.group(5) and want['value']:
            kinds['time part'] += 1
        if want['value'] and max(len(f) for f in fields) <= 2:
            kinds['two-digit year'] += 1
        first, second = int(fields[0]), int(fields[1])
        if (
            want['value']
            and not want['warnings']
            and len(fields[2]) == 4
            and first != second
            and first <= 12
            and second <= 12
            and int(want['value'][5:7]) == first
        ):
            kinds['month first'] += 1
    assert not failures, '\n'.join(failures)
    missing = sorted(set(rules) - covered)
    assert not missing, f'rules without an example: {missing}'
    for kind in (
        'month first',
        'day first, with a warning',
        'two-digit year',
        'time part',
        'refused',
    ):
        assert kinds[kind], f'no example in DATES.md shows: {kind}'
