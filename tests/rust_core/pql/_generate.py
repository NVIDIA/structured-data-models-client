# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Seeded generators for the Phase 5 acceptance checks: predictive queries
derived from ``PQLGrammar.g4``, and the literals its ``DATE`` token accepts.

Queries are written against the corpus schemas. Most follow the grammar's
productions with names and constants drawn from the schema, which is how
valid ones come out; some take a schema-blind path, and some are then
mutated a token at a time, which is how the invalid ones come out. Every
choice comes from one ``random.Random``, so a seed fixes the sequence.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

PROFILES = ('RELATIONAL', 'TABULAR', 'TIMESERIES', 'ANOMALY', 'FINE_TUNING')
MODES = ('TRAINING', 'INFERENCE')

AGGRS = (
    'SUM',
    'AVG',
    'MIN',
    'MAX',
    'COUNT',
    'COUNT_DISTINCT',
    'FIRST',
    'LAST',
    'LIST_DISTINCT',
)
SIMPLE_OPS = ('=', '!=', '<', '<=', '>', '>=')
WORD_OPS = (
    'IS',
    'IS NOT',
    'IN',
    'IS IN',
    'LIKE',
    'NOT LIKE',
    'CONTAINS',
    'NOT CONTAINS',
    'STARTS WITH',
    'ENDS WITH',
)
TIME_UNITS = ('days', 'weeks', 'months', 'hours', 'minutes', 'seconds')
NEG_INFTY = ('-INF', '-INFINITY', '-inf', '-infinity')
BOOLS = ('true', 'false', 'True', 'False', 'TRUE', 'FALSE')
NULLS = ('NULL', 'null')
# The whitespace the grammar skips (PQLGrammar.g4, WS), beyond the space.
UNICODE_SPACES = (
    '\t',
    '\n',
    '\r',
    '\x0b',
    '\x0c',
    '\x1c',
    '\x1d',
    '\x1e',
    '\x1f',
    '\x85',
    '\xa0',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    ' ',
    '　',
)
# Characters the grammar does not skip, and no token starts with.
JUNK = (';', '*', '@', '$', '%', '^', '&', '{', '}', '[', ']', '​', '|')
INT_EDGES = (
    '0',
    '1',
    '-1',
    '2147483647',
    '2147483648',
    '-2147483648',
    '-2147483649',
    '9223372036854775807',
    '9223372036854775808',
    '-9223372036854775808',
    '-9223372036854775809',
    '99999999999999999999999',
    '007',
    '-0',
)


@dataclass
class Table:
    name: str
    cols: list[dict[str, Any]]
    pkey: str | None
    time_col: str | None


class QueryGen:
    r"""Queries over one schema, as the corpus stores it."""

    def __init__(self, rng: random.Random, schema: dict[str, Any]):
        self.rng = rng
        # 0 for a query written to follow the schema, 1 for one that also
        # takes the turns that usually make it invalid.
        self.noise = 0
        self.tables = [
            Table(name, spec['cols'], spec.get('pkey'), spec.get('time_col'))
            for name, spec in schema['tables'].items()
        ]
        self.links: dict[str, set[str]] = {t.name: set() for t in self.tables}
        for group in schema.get('col_groups', []):
            names = {c['table_name'] for c in group['columns']}
            for name in names:
                self.links[name] |= names - {name}

    def odd(self, p: float) -> bool:
        r"""Whether to take a turn that usually makes the query invalid."""
        return self.rng.random() < p * self.noise

    def kw(self, word: str) -> str:
        r"""A keyword as the grammar takes it, lower or UPPER case, and now
        and then in a case it does not take."""
        rng = self.rng
        if self.odd(0.04):
            if rng.random() < 0.5:
                return word.title()
            return ''.join(
                c.upper() if rng.random() < 0.5 else c.lower() for c in word
            )
        return word.upper() if rng.random() < 0.5 else word.lower()

    # --- names -------------------------------------------------------------

    def ident(self, name: str) -> str:
        r"""A name as a query spells it: bare when the grammar's ``ID``
        allows it, else in backticks; now and then quoted anyway."""
        bare = (
            name
            and (name[0].isascii() and (name[0].isalpha() or name[0] == '_'))
            and all(c.isascii() and (c.isalnum() or c == '_') for c in name)
        )
        if not bare or self.rng.random() < 0.05:
            return f'`{name}`'
        if self.odd(0.01):
            return name.upper()
        return name

    def fqn(self, table: Table, col: str | None) -> str:
        right = '*' if col is None else self.ident(col)
        return f'{self.ident(table.name)}.{right}'

    def table(self) -> Table:
        return self.rng.choice(self.tables)

    def column(self, table: Table | None = None, wildcard: float = 0.1):
        r"""A column token and the column it names (``None`` for ``*`` or
        for a name the schema does not have)."""
        rng = self.rng
        table = table or self.table()
        if rng.random() < wildcard:
            return self.fqn(table, None), table, None
        if self.odd(0.04):
            name = rng.choice(['nope', 'Amount', 'x y'])
            return self.fqn(table, name), table, None
        if self.odd(0.02):
            return (
                f'{rng.choice(["nope", "Orders", "missing"])}.'
                f'{self.ident(rng.choice(table.cols)["name"])}',
                table,
                None,
            )
        col = rng.choice(table.cols)
        return self.fqn(table, col['name']), table, col

    # --- constants ---------------------------------------------------------

    def string(self) -> str:
        rng = self.rng
        body = rng.choice(
            ['x', 'US', 'a%', '%b', '%c%', 'é', '😀', 'two words', 'NULL', '1']
        )
        if self.odd(0.3):
            body = rng.choice(
                ['', "it\\'s", 'tab\\there', 'bad\\q', 'x\\u0041', 'a\\']
            )
        if rng.random() < 0.15:
            return '"' + body.replace("\\'", "'") + '"'
        return "'" + body + "'"

    def number(self, decimal: bool = False) -> str:
        rng = self.rng
        if self.odd(0.1):
            return rng.choice(INT_EDGES)
        if decimal or rng.random() < 0.2:
            whole = str(rng.randint(-100, 1000))
            if rng.random() < 0.1:
                return whole + rng.choice(['e3', 'E-2', 'e+1'])
            text = f'{whole}.{rng.randint(0, 99)}'
            if rng.random() < 0.05:
                text += rng.choice(['e10', 'E-9'])
            if self.odd(0.05):
                text += 'e400'
            if self.odd(0.02):
                text = '-' + text
            return text
        return str(rng.randint(-50, 500))

    def constant(self, col: dict[str, Any] | None, depth: int = 0) -> str:
        rng = self.rng
        dtype = (col or {}).get('dtype')
        if self.odd(0.15):
            dtype = rng.choice(
                ['int', 'float', 'string', 'bool', 'time', 'null', 'array']
            )
        if dtype == 'null' or rng.random() < 0.04:
            return rng.choice(NULLS)
        if (
            dtype in ('array', 'stringlist', 'intlist', 'floatlist')
            or self.odd(0.08)
        ) and depth < 2:
            size = 1 if self.odd(0.15) else rng.choice([2, 2, 3, 4])
            inner = dict(col or {})
            if dtype in ('stringlist', 'array'):
                inner['dtype'] = rng.choice(['string', 'int'])
            elif dtype in ('intlist', 'floatlist'):
                inner['dtype'] = dtype[:-4]
            items = [self.constant(inner, depth + 1) for _ in range(size)]
            trailing = ',' if size == 1 and rng.random() < 0.5 else ''
            return '(' + ', '.join(items) + trailing + ')'
        if dtype == 'int':
            return self.number()
        if dtype == 'float':
            return self.number(decimal=rng.random() < 0.7)
        if dtype == 'bool':
            return rng.choice(BOOLS)
        if dtype == 'time':
            return date_literal(rng) if self.noise else _plain_date(rng)
        return self.string()

    # --- clauses -------------------------------------------------------------

    def op(self) -> str:
        rng = self.rng
        if rng.random() < 0.55:
            return rng.choice(SIMPLE_OPS)
        op = rng.choice(WORD_OPS)
        if self.odd(0.03):
            return op.replace(' ', rng.choice(['  ', '\t', '\n']))
        if self.odd(0.03):
            return op.title()
        return op.upper() if rng.random() < 0.5 else op.lower()

    def word_op(self, op: str) -> str:
        return op.upper() if self.rng.random() < 0.5 else op.lower()

    def sound_comparison(self, table: Table) -> str:
        r"""A comparison the type checks accept: an operator and a constant
        that suit the column's type."""
        rng = self.rng
        plain = [
            c
            for c in table.cols
            if c['name'] not in (table.pkey, table.time_col)
            and c['stype'] != 'ID'
        ] or table.cols
        col = rng.choice(plain)
        token = self.fqn(table, col['name'])
        dtype = col['dtype']
        if rng.random() < 0.08:
            return f'{token} {self.word_op(rng.choice(["IS", "IS NOT"]))} ' + (
                rng.choice(NULLS)
            )
        if dtype in ('int', 'float'):
            if rng.random() < 0.15:
                items = ', '.join(
                    self.number() for _ in range(rng.randint(2, 4))
                )
                return f'{token} {self.word_op("IN")} ({items})'
            return f'{token} {rng.choice(SIMPLE_OPS)} ' + self.number(
                decimal=dtype == 'float' and rng.random() < 0.5
            )
        if dtype == 'string':
            roll = rng.random()
            if roll < 0.4:
                return f'{token} {rng.choice(["=", "!="])} {self.string()}'
            if roll < 0.6:
                op = self.word_op(rng.choice(['LIKE', 'NOT LIKE']))
                return f"{token} {op} '%{rng.choice(['a', 'x', 'US'])}%'"
            if roll < 0.8:
                op = self.word_op(rng.choice(['STARTS WITH', 'ENDS WITH']))
                return f'{token} {op} {self.string()}'
            items = ', '.join(self.string() for _ in range(rng.randint(2, 3)))
            return f'{token} {self.word_op("IN")} ({items})'
        if dtype == 'bool':
            return f'{token} = {rng.choice(BOOLS)}'
        if dtype == 'time':
            op = rng.choice(['<', '>', '<=', '>=', '=', '!='])
            return f'{token} {op} {_plain_date(rng)}'
        if dtype.endswith('list'):
            op = self.word_op(rng.choice(['CONTAINS', 'NOT CONTAINS']))
            return f'{token} {op} {self.string()}'
        return f'{token} {self.op()} {self.constant(col)}'

    def comparison(self, table: Table | None = None) -> str:
        rng = self.rng
        if not self.noise and table is not None and rng.random() < 0.85:
            return self.sound_comparison(table)
        if rng.random() < 0.15:
            return f'{self.aggregation(table, nested=True)} {self.op()} ' + (
                self.number()
            )
        token, _, col = self.column(table, wildcard=0.03)
        return f'{token} {self.op()} {self.constant(col)}'

    def condition(self, table: Table | None = None, depth: int = 0) -> str:
        rng = self.rng
        roll = rng.random()
        if depth >= 3 or roll < 0.55 or (not self.noise and depth >= 1):
            return self.comparison(table)
        if roll < 0.65:
            return f'{self.kw("not")} {self.condition(table, depth + 1)}'
        if roll < 0.9:
            joint = self.kw(rng.choice(['and', 'or']))
            return (
                f'{self.condition(table, depth + 1)} {joint} '
                f'{self.condition(table, depth + 1)}'
            )
        return f'({self.condition(table, depth + 1)})'

    def window(self) -> str:
        rng = self.rng
        roll = rng.random()
        if roll < 0.1:
            return ''
        if rng.random() < 0.12:
            start = rng.choice(NEG_INFTY)
        elif self.odd(0.08):
            start = rng.choice(INT_EDGES)
        else:
            start = str(rng.choice([0, 0, 0, -30, -7, 1, 7, 30, 60, -1]))
        if self.odd(0.08):
            end = rng.choice(INT_EDGES)
        elif self.odd(0.2):
            end = str(rng.choice([0, -10, 1, 7]))
        else:
            low = int(start) if start.lstrip('-').isdigit() else -1
            end = str(rng.choice([1, 7, 10, 30, 30, 60, 90, 365]) + max(low, 0))
        text = f', {start}, {end}'
        if rng.random() < 0.85:
            unit = rng.choice(TIME_UNITS)
            text += f', {self.kw(unit)}'
        return text

    def aggregation(
        self, table: Table | None = None, nested: bool = False
    ) -> str:
        rng = self.rng
        if not self.noise and table is not None and rng.random() < 0.8:
            return self.sound_aggregation(table)
        name = rng.choice(AGGRS)
        aggr = name if rng.random() < 0.6 else name.lower()
        if self.odd(0.02):
            aggr = name.title()
        # Aggregate over a table linked to the entity's, as a target does.
        if table is not None and self.links[table.name] and rng.random() < 0.8:
            linked = rng.choice(sorted(self.links[table.name]))
            target = next(t for t in self.tables if t.name == linked)
        else:
            target = self.table()
        wildcard = 0.5 if name == 'COUNT' else 0.05
        token, _, _ = self.column(target, wildcard=wildcard)
        if rng.random() < 0.15:
            token = (
                f'{token} {self.kw("where")} '
                f'{self.condition(target, depth=2 if nested else 1)}'
            )
        return f'{aggr}({token}{self.window()})'

    def sound_aggregation(self, entity: Table) -> str:
        r"""An aggregation over a table linked to the entity's, with a time
        column and a window, of a column the aggregation takes."""
        rng = self.rng
        facts = [
            t
            for t in self.tables
            if t.time_col and t.name in self.links[entity.name]
        ]
        if not facts:
            return self.aggregation(None)
        fact = rng.choice(facts)
        numeric = [
            c
            for c in fact.cols
            if c['stype'] == 'numerical' and c['dtype'] in ('int', 'float')
        ]
        roll = rng.random()
        if roll < 0.4 or not numeric:
            inner = self.fqn(fact, None)
            name = 'COUNT'
        elif roll < 0.85:
            inner = self.fqn(fact, rng.choice(numeric)['name'])
            name = rng.choice(['SUM', 'AVG', 'MIN', 'MAX'])
        else:
            keys = [c for c in fact.cols if c['stype'] == 'ID'] or fact.cols
            inner = self.fqn(fact, rng.choice(keys)['name'])
            name = rng.choice(['COUNT_DISTINCT', 'LIST_DISTINCT', 'FIRST'])
        if rng.random() < 0.15:
            inner = f'{inner} {self.kw("where")} {self.sound_comparison(fact)}'
        aggr = name if rng.random() < 0.6 else name.lower()
        start = rng.choice([0, 0, 0, 7, 30])
        end = start + rng.choice([1, 7, 10, 30, 60, 90])
        unit = self.kw(rng.choice(TIME_UNITS))
        if rng.random() < 0.1:
            return f'{aggr}({inner}, {start}, {end})'
        return f'{aggr}({inner}, {start}, {end}, {unit})'

    def sound_target(self, entity: Table) -> str:
        rng = self.rng
        roll = rng.random()
        if roll < 0.5:
            text = self.sound_aggregation(entity)
            if rng.random() < 0.3:
                text += (
                    f' {rng.choice([">", "=", "<=", ">="])} {rng.randint(0, 5)}'
                )
            return text
        if roll < 0.85:
            plain = [
                c
                for c in entity.cols
                if c['name'] not in (entity.pkey, entity.time_col)
                and c['stype']
                in ('numerical', 'categorical', 'multicategorical')
            ]
            if plain:
                return self.fqn(entity, rng.choice(plain)['name'])
        return self.sound_comparison(entity)

    def target(self, entity: Table) -> str:
        rng = self.rng
        if not self.noise and rng.random() < 0.85:
            return self.sound_target(entity)
        roll = rng.random()
        if roll < 0.45:
            text = self.aggregation(entity)
            if rng.random() < 0.3:
                text += f' {self.op()} {self.number()}'
            return text
        if roll < 0.75:
            token, _, _ = self.column(
                entity if rng.random() < 0.8 else None, wildcard=0.02
            )
            return token
        return self.condition(entity)

    def problem_spec(self) -> str:
        rng = self.rng
        roll = rng.random()
        if roll < 0.7 or (not self.noise and roll < 0.9):
            return ''
        if roll < 0.8:
            count = rng.choice(['1', '3', '7', '12'])
            if self.odd(0.3):
                count = rng.choice(['0', '-2', '1000', '100000'])
            return f'{self.kw("forecast")} {count} {self.kw("timeframes")}'
        kind = self.kw(rng.choice(['rank', 'classify']))
        k = rng.choice(['1', '5', '10', '20'])
        if self.odd(0.3):
            k = rng.choice(['0', '-1', '100000', '2147483648'])
        top = f'{self.kw("top")} {k}'
        roll = rng.random()
        if roll < 0.5:
            return f'{kind} {top}'
        if roll < 0.8:
            return kind
        return top

    def entity(self, table: Table) -> str:
        rng = self.rng
        if table.pkey and (not self.noise or rng.random() < 0.85):
            token = self.fqn(table, table.pkey)
            col = next(c for c in table.cols if c['name'] == table.pkey)
        else:
            token, _, col = self.column(table, wildcard=0.05)
        roll = rng.random()
        if not self.noise:
            roll = min(roll * 2, 0.99) if roll >= 0.15 else roll
        if roll < 0.12:
            op = rng.choice(['=', 'IN', 'in', '=', 'IS IN', '>'])
            token = f'{token} {op} {self.constant(col)}'
        if roll < 0.15 or 0.12 <= roll < 0.3:
            token = f'{token} {self.kw("where")} {self.condition(table, 1)}'
        return token

    def query(self) -> str:
        rng = self.rng
        self.noise = 0 if rng.random() < 0.5 else 1
        keyed = [t for t in self.tables if t.pkey] or self.tables
        entity = rng.choice(keyed)
        parts = [self.kw('predict'), self.target(entity)]
        spec = self.problem_spec()
        if spec:
            parts.append(spec)
        parts.append(self.kw('for each') if not self.odd(0.15) else 'FOR')
        parts.append(self.entity(entity))
        if self.odd(0.15):
            parts.append(f'{self.kw("assuming")} {self.condition()}')
        return self.join(parts)

    def join(self, parts: list[str]) -> str:
        rng = self.rng
        out = []
        for part in parts:
            out.append(part)
            roll = rng.random()
            if roll < 0.9:
                out.append(' ')
            elif roll < 0.95:
                out.append(rng.choice(UNICODE_SPACES))
            elif roll < 0.98:
                out.append(' # a comment\n')
            else:
                out.append('  ')
        text = ''.join(out[:-1])
        if rng.random() < 0.03:
            text = ' ' + text + ' '
        return text


# --- invalid queries: one change to a query from the grammar ----------------

VOCABULARY = (
    'PREDICT',
    'FOR EACH',
    'FOR',
    'WHERE',
    'ASSUMING',
    'AND',
    'OR',
    'NOT',
    'COUNT',
    'SUM',
    '(',
    ')',
    ',',
    '.',
    '=',
    '>',
    'IS',
    'IN',
    'NULL',
    'days',
    'RANK',
    'TOP',
    'CLASSIFY',
    'FORECAST',
    'TIMEFRAMES',
    '10',
    '-1',
    '1.5',
    "'x'",
    '2024-01-01',
    '12:00:00',
    'orders.amount',
    'users.*',
    'foo',
    '`a b`',
    '-INF',
)


def _tokens(text: str) -> list[str]:
    r"""``text`` split into words and single punctuation characters."""
    out, word = [], ''
    for c in text:
        if c.isspace():
            if word:
                out.append(word)
                word = ''
            out.append(c)
        elif c in '(),':
            if word:
                out.append(word)
                word = ''
            out.append(c)
        else:
            word += c
    if word:
        out.append(word)
    return out


def mutate(rng: random.Random, text: str) -> str:
    tokens = _tokens(text)
    solid = [i for i, t in enumerate(tokens) if not t.isspace()]
    if not solid:
        return text + rng.choice(VOCABULARY)
    i = rng.choice(solid)
    roll = rng.random()
    if roll < 0.2:
        del tokens[i]
    elif roll < 0.35:
        tokens.insert(i, tokens[i] + ' ')
    elif roll < 0.5 and len(solid) > 1:
        j = rng.choice(solid)
        tokens[i], tokens[j] = tokens[j], tokens[i]
    elif roll < 0.7:
        tokens[i] = rng.choice(VOCABULARY)
    elif roll < 0.8:
        tokens.insert(i, rng.choice(VOCABULARY) + ' ')
    elif roll < 0.9:
        word = tokens[i]
        tokens[i] = ''.join(
            c.swapcase() if rng.random() < 0.5 else c for c in word
        )
    else:
        pos = rng.randrange(len(text) + 1)
        return text[:pos] + rng.choice(JUNK + ("'", '"', '`', '#')) + text[pos:]
    return ''.join(tokens)


def queries(
    seed: int, n: int, schemas: list[dict[str, Any]]
) -> Iterator[dict[str, Any]]:
    r"""``n`` generated cases: ``query``, ``schema``, ``profile`` and
    ``mode`` (``None`` for the profile's own)."""
    rng = random.Random(seed)
    gens = [QueryGen(rng, schema) for schema in schemas]
    for _ in range(n):
        k = rng.randrange(len(schemas))
        text = gens[k].query()
        if gens[k].noise and rng.random() < 0.5:
            text = mutate(rng, text)
            if rng.random() < 0.2:
                text = mutate(rng, text)
        profile = 'RELATIONAL' if rng.random() < 0.55 else rng.choice(PROFILES)
        mode = rng.choice(MODES) if rng.random() < 0.15 else None
        yield {
            'query': text,
            'schema': schemas[k],
            'profile': profile,
            'mode': mode,
        }


# --- date literals -------------------------------------------------------------

LEAP_YEARS = (1600, 1896, 1904, 2000, 2004, 2020, 2024, 2400, 4, 96, 400)
COMMON_YEARS = (1700, 1800, 1900, 2001, 2023, 2100, 2200, 2300, 1, 99, 100)


def _digits(rng: random.Random, value: int, width: int | None = None) -> str:
    text = str(value)
    if width is not None and len(text) < width:
        text = text.zfill(width)
    return text


def _year(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.35:
        return _digits(rng, rng.randint(1900, 2100), 4)
    if roll < 0.55:
        return _digits(rng, rng.randint(0, 99), rng.choice([1, 2, 2]))
    if roll < 0.7:
        return _digits(rng, rng.choice(LEAP_YEARS + COMMON_YEARS), 4)
    if roll < 0.8:
        return _digits(rng, rng.randint(1, 9999), rng.choice([None, 4]))
    if roll < 0.9:
        return _digits(rng, rng.randint(100, 999), rng.choice([3, 4]))
    return _digits(rng, rng.randint(0, 10 ** rng.randint(1, 9)))


def _month(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.85:
        return _digits(rng, rng.randint(1, 12), rng.choice([None, 2]))
    if roll < 0.95:
        return _digits(rng, rng.choice([0, 13, 31, 99]), rng.choice([None, 2]))
    return _digits(rng, rng.randint(0, 999), rng.choice([None, 3]))


def _day(rng: random.Random, month: str, year: str) -> str:
    roll = rng.random()
    if roll < 0.55:
        return _digits(rng, rng.randint(1, 28), rng.choice([None, 2]))
    if roll < 0.85:
        # Month ends and leap days, where most of the rules are.
        return _digits(rng, rng.choice([28, 29, 30, 31]), rng.choice([None, 2]))
    if roll < 0.95:
        return _digits(rng, rng.choice([0, 32, 99]), rng.choice([None, 2]))
    return _digits(rng, rng.randint(0, 9999), rng.choice([None, 3, 4]))


def _time(rng: random.Random) -> str:
    def part(top: int, edge: tuple[int, ...]) -> str:
        if rng.random() < 0.9:
            value = rng.randint(0, top)
        else:
            value = rng.choice(edge)
        return _digits(rng, value, rng.choice([None, 2, 2, 2, 3]))

    return (
        f'{part(23, (24, 25, 99))}:{part(59, (60, 61, 99))}:'
        f'{part(59, (60, 61, 99))}'
    )


def _plain_date(rng: random.Random) -> str:
    r"""A date written the usual way, which every reading takes."""
    year, month, day = (
        rng.randint(1990, 2030),
        rng.randint(1, 12),
        rng.randint(1, 28),
    )
    text = rng.choice(
        [f'{year}-{month:02d}-{day:02d}', f'{month}/{day}/{year}']
    )
    if rng.random() < 0.2:
        text += f' {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00'
    return text


def date_literal(rng: random.Random) -> str:
    r"""A literal the grammar's ``datetime`` rule accepts: ``DATE`` with
    ``/``, ``-`` or ``.`` between three runs of digits, and now and then
    ``TIME``, as the tree joins them (one space)."""
    sep = rng.choice('/-.')
    roll = rng.random()
    year = _year(rng)
    month = _month(rng)
    day = _day(rng, month, year)
    if rng.random() < 0.06:
        # The end of February, in leap and common years.
        if rng.random() < 0.6:
            year = _digits(rng, rng.choice(LEAP_YEARS + COMMON_YEARS), 4)
        month = rng.choice(['2', '02'])
        day = rng.choice(['28', '29', '29', '29', '30'])
    if roll < 0.4:
        fields = (year, month, day)
    elif roll < 0.65:
        fields = (month, day, year)
    elif roll < 0.85:
        fields = (day, month, year)
    elif roll < 0.92:
        fields = (year, day, month)
    else:
        fields = tuple(
            _digits(rng, rng.randint(0, 10 ** rng.randint(1, 6)))
            for _ in range(3)
        )
    text = sep.join(fields)
    if rng.random() < 0.3:
        text += ' ' + _time(rng)
    return text


def date_literals(seed: int, n: int) -> Iterator[str]:
    rng = random.Random(seed)
    for _ in range(n):
        yield date_literal(rng)
