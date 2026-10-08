# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P6.view-expr: the safety check on a view's expression columns gives the
same answer and reason from the Rust core as from ``rfm/view_expr.py``.

50,000 expressions are generated (seed 20261006), half from templates of
the SQL a view writes and half as soups of tokens, both drawing on what the
check's regular expressions are sensitive to: Python's Unicode ``\w`` and
``\s`` (combining marks, other scripts' digits, ``\x1c`` to ``\x1f``,
``\x85``), case folding (``ſ``, the Kelvin sign, ``İ``), every quoting form,
unsafe tokens, keywords in and out of keyword position, type names, qualified
and quoted calls, and session values. Each is checked as the engine checks
it: ``_unsafe_expr_reason`` with no dialect, ``databricks`` and
``snowflake``; then, where that passes, ``_untrusted_call_reason`` for the
same dialect. Every answer, ``None`` or the reason's text, must be equal.
"""

from __future__ import annotations

import random
import re
from collections import Counter

from _dialects import rust_call

SEED = 20261006
NUM_EXPRESSIONS = 50_000

_IDENTS = (
    'x',
    'amount',
    'price',
    'qty',
    'o.total',
    'ORDERS.MERGE',
    'a.select',
    't.x.y',
    '$col',
    'col$1',
    '_y',
    'x1',
    'naïve',
    'été',
    'x́',
    'ſelect',
    'K',
    'İd',
    'straße',
    'ﬁle',
    'σΣ',
    'ǅ',
    '٣٤',
    'x²',
    '½',
    'ª',
    'x‍y',
    '',
    '中文',
    '\U0001d400',
)

_KEYWORDS = (
    'select',
    'SELECT',
    'Select',
    'table',
    'update',
    'delete',
    'merge',
    'create',
    'alter',
    'drop',
    'grant',
    'revoke',
    'execute',
    'insert',
    'truncate',
    'from',
    'FROM',
    'not',
    'NOT',
    'case',
    'when',
    'then',
    'else',
    'end',
    'is',
    'distinct',
    'like',
    'ilike',
    'rlike',
    'regexp',
    'escape',
    'in',
    'between',
    'and',
    'or',
    'as',
    'AS',
    'interval',
    'over',
    'any',
    'some',
    'all',
    'exists',
    'for',
    'both',
    'leading',
    'trailing',
    'placing',
    'by',
    'div',
    'collate',
    'current_user',
    'CURRENT_DATE',
    'current_timestamp',
    'localtime',
    'localtimestamp',
    'session_user',
    'current_catalog',
    'identifier',
)

_FUNCTIONS = (
    'upper',
    'UPPER',
    'Upper',
    'coalesce',
    'iff',
    'IFF',
    'if',
    'div0',
    'named_struct',
    'xxhash64',
    'md5_number_lower64',
    'regexp_like',
    'date_trunc',
    'to_timestamp_ltz',
    'array_construct',
    'concat_ws',
    'cast',
    'try_cast',
    'round',
    'abs',
    'secret',
    'http_request',
    'SYSTEM$WAIT',
    'my_udf',
    'udf$1',
    'sum',
    'count',
    'current_date',
    'escape',
    'exists',
    'filter',
    'insert',
    'truncate',
    'identifier',
    'decimal',
    'varchar',
    'VARCHAR',
    'timestamp_ltz',
    'number',
    'map',
    'struct',
    'ſecret',
    'uppеr',
    'transform',
    'get',
    'any',
    'all',
    'not',
    'case',
    'in',
    'distinct',
)

_TYPES = (
    'VARCHAR',
    'varchar',
    'DECIMAL',
    'number',
    'timestamp_ntz',
    'string',
    'char',
    'vector',
    'int',
    'my_type',
)

_LITERALS = (
    "'abc'",
    "'it''s'",
    "'a\\'b'",
    "'select'",
    "'--'",
    "'é'",
    "''",
    '"name"',
    '"na""me"',
    '"a\\b"',
    '"select"',
    '`col`',
    '`a``b`',
    '`a\\`',
    '$$x$$',
    '$$select 1$$',
    '$$$$',
    '1',
    '2.5',
    '1e3',
    '0x1F',
    "'unterminated",
    '"unterminated',
    '`unterminated',
    '$$unterminated',
    "X'00'",
)

_PUNCTUATION = (
    '+',
    '-',
    '*',
    '/',
    '%',
    '=',
    '<',
    '>',
    '!=',
    '<>',
    '||',
    '|',
    '&',
    '^',
    '~',
    ',',
    '(',
    ')',
    '(',
    ')',
    '[',
    ']',
    '?',
    ':',
    '::',
    '.',
    ';',
    '--',
    '/*',
    '*/',
    '|>',
    '->>',
    '->',
    '$',
    '\\',
    '!',
)

_SPACES = (
    ' ',
    ' ',
    ' ',
    ' ',
    '',
    '',
    '  ',
    '\t',
    '\n',
    '\r\n',
    '\x0b',
    '\x0c',
    '\x1c',
    '\x1f',
    '\x85',
    ' ',
    ' ',
    ' ',
    '　',
    '​',
)


def _space(rng: random.Random) -> str:
    return rng.choice(_SPACES)


def _token(rng: random.Random) -> str:
    pool = rng.choice(
        (_IDENTS, _KEYWORDS, _FUNCTIONS, _LITERALS, _PUNCTUATION, _PUNCTUATION)
    )
    return rng.choice(pool)


def _operand(rng: random.Random, depth: int) -> str:
    kind = rng.randrange(7 if depth < 3 else 3)
    if kind == 0:
        return rng.choice(_IDENTS)
    if kind == 1:
        return rng.choice(_LITERALS[:10] + _LITERALS[16:21])
    if kind == 2:
        return rng.choice(('1', '0', '2.5', 'NULL', 'TRUE'))
    if kind == 3:
        return _call(rng, depth + 1)
    if kind == 4:
        return f'({_expression(rng, depth + 1)})'
    if kind == 5:
        return (
            f'CASE WHEN {_condition(rng, depth + 1)} THEN '
            f'{_operand(rng, depth + 1)} ELSE {_operand(rng, depth + 1)} END'
        )
    return rng.choice(
        (
            f'CAST({_operand(rng, depth + 1)} AS {rng.choice(_TYPES)}'
            f'({rng.randint(1, 38)}))',
            f'{_operand(rng, depth + 1)}::{rng.choice(_TYPES)}'
            f'({rng.randint(1, 38)})',
            f'{rng.choice(_TYPES)}({rng.randint(1, 38)})',
            f'({_operand(rng, depth + 1)})({_operand(rng, depth + 1)})',
            f'{rng.choice(_IDENTS)}[{rng.randint(0, 3)}]({_operand(rng, depth + 1)})',
        )
    )


def _call(rng: random.Random, depth: int) -> str:
    name = rng.choice(_FUNCTIONS)
    if rng.random() < 0.1:
        name = rng.choice(('a.', '"q".', '`q`.', '')) + name
    args = ', '.join(_operand(rng, depth + 1) for _ in range(rng.randint(0, 3)))
    return f'{name}{_space(rng)}({args})'


def _condition(rng: random.Random, depth: int) -> str:
    left = _operand(rng, depth + 1)
    kind = rng.randrange(8)
    if kind == 0:
        return f'{left} = {_operand(rng, depth + 1)}'
    if kind == 1:
        return f'{left} NOT LIKE ({rng.choice(_LITERALS[:7])})'
    if kind == 2:
        return f'{left} IS DISTINCT FROM ({_operand(rng, depth + 1)})'
    if kind == 3:
        return f'{left} IN ({_operand(rng, depth + 1)}, 2)'
    if kind == 4:
        return f'NOT {_call(rng, depth + 1)}'
    if kind == 5:
        return f"{left} LIKE {_operand(rng, depth + 1)} ESCAPE ('!')"
    if kind == 6:
        return f'{left} BETWEEN 1 AND {_operand(rng, depth + 1)}'
    return f'{left} {rng.choice(_KEYWORDS)} ({_operand(rng, depth + 1)})'


def _expression(rng: random.Random, depth: int = 0) -> str:
    out = _operand(rng, depth)
    for _ in range(rng.randint(0, 2)):
        out += f' {rng.choice(("+", "-", "*", "/", "||", "AND", "OR"))} '
        out += _operand(rng, depth + 1)
    return out


def _soup(rng: random.Random) -> str:
    return ''.join(
        _space(rng) + _token(rng) for _ in range(rng.randint(1, 14))
    ) + rng.choice(('', '', ' ', '\n'))


def expressions() -> list[str]:
    rng = random.Random(SEED)
    out = []
    for i in range(NUM_EXPRESSIONS):
        expr = _expression(rng) if i % 2 == 0 else _soup(rng)
        if rng.random() < 0.05:
            at = rng.randint(0, len(expr))
            expr = expr[:at] + _token(rng) + expr[at:]
        out.append(expr)
    return out


def _python(expr: str) -> list[tuple[str, str, str | None]]:
    r"""``(function, dialect, answer)`` for every check the engine makes of
    ``expr``."""
    from sdm_relational_engine.rfm.view_expr import (
        _unsafe_expr_reason,
        _untrusted_call_reason,
    )

    out: list[tuple[str, str, str | None]] = []
    for dialect in (None, 'databricks', 'snowflake'):
        unsafe = _unsafe_expr_reason(expr, dialect)
        out.append(('_unsafe_expr_reason', dialect, unsafe))
        if unsafe is None and dialect is not None:
            out.append(
                (
                    '_untrusted_call_reason',
                    dialect,
                    _untrusted_call_reason(expr, dialect, columns=()),
                )
            )
    return out


def _shape(reason: str) -> str:
    return re.sub(r"'[^']*'|\"[^\"]*\"", 'Q', reason)


def test_view_expressions_match_python() -> None:
    exprs = expressions()
    answers = {expr: _python(expr) for expr in exprs}
    accepted: Counter[tuple[str, str | None]] = Counter()
    shapes: Counter[str] = Counter()
    for found in answers.values():
        for function, dialect, answer in found:
            if answer is None:
                accepted[(function, dialect)] += 1
            else:
                shapes[_shape(answer)] += 1
    for dialect in ('databricks', 'snowflake'):
        assert accepted[('_untrusted_call_reason', dialect)] >= 3_000, (
            f'too few generated expressions pass on {dialect}: {accepted}'
        )
    # Every reason the check gives, each many times over.
    assert len(shapes) >= 15, f'too few kinds of refusal: {sorted(shapes)}'
    scarce = {shape: n for shape, n in shapes.items() if n < 20}
    assert not scarce, f'kinds of refusal seldom generated: {scarce}'
    failures, checked = [], 0
    for expr, found in answers.items():
        for function, dialect, answer in found:
            checked += 1
            backend = 'snow' if dialect == 'snowflake' else 'databricks'
            args = {'expr': expr, 'dialect': dialect}
            if function == '_untrusted_call_reason':
                args['columns'] = []
            try:
                got = rust_call(backend, f'view_expr.{function}', args)
            except Exception as error:  # noqa: BLE001 - reported per case
                failures.append(f'{function}({expr!r}, {dialect!r}): {error!r}')
                continue
            if got != {'result': answer}:
                failures.append(
                    f'{function}({expr!r}, {dialect!r}): Python {answer!r}, '
                    f'Rust {got.get("result", got)!r}'
                )
    assert not failures, (
        f'{len(failures)} of {checked} checks differ:\n'
        + '\n'.join(failures[:40])
    )
