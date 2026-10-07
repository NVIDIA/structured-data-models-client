# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Which functions a view's expressions may call.

A metric or semantic view is authored by someone else, and its expressions run
under the caller's credentials every time a graph samples. Only pure built-in
scalar functions are trusted by default.
"""

from __future__ import annotations

import pytest
from sdm_relational_engine.rfm.view_expr import (
    _DATABRICKS_FUNCTIONS,
    _DATABRICKS_KEYWORDS,
    _GRAMMAR_WORDS,
    _NEVER_BEFORE_AN_EXPRESSION,
    _SNOWFLAKE_FUNCTIONS,
    _SNOWFLAKE_RESERVED,
    _unsafe_expr_reason,
    _untrusted_call_reason,
)

_DIALECTS = ['databricks', 'snowflake']
# The bare names these expressions use for columns.
_COLUMNS = ('a', 'col', 'd', 'name', 'order_date', 'x', 'y', 'z')


def _call(expr: str, dialect: str) -> str | None:
    return _untrusted_call_reason(expr, dialect, columns=_COLUMNS)


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'expr',
    [
        "CASE WHEN x > 1 THEN 'a' ELSE 'b' END",
        'CAST(x AS DECIMAL(10,2))',
        'cast(x as varchar(20))',
        'TRY_CAST(x AS NUMERIC(5, 1))',
        'x::DECIMAL(10, 2)',
        'EXTRACT(YEAR FROM d)',
        "TRIM(BOTH ' ' FROM x)",
        "SPLIT(x,'-')[0]",
        'IFF(x > 1, 1, 0)',
        'x::int',
        'transform(a, x -> x + 1)',
        "CONCAT('evil(', x)",
        'CONCAT(`evil(`, x)',
        'x IN (1, 2) AND NOT (y OR z)',
        'COALESCE(x, 0) BETWEEN (1) AND (2)',
        'UPPER (name)',
        "DATE_TRUNC('MONTH', order_date)",
        'x',
        'orders.amount * 2',
    ],
)
def test_a_pure_built_in_expression_is_trusted(dialect: str, expr: str) -> None:
    assert _unsafe_expr_reason(expr) is None
    assert _call(expr, dialect) is None


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    ('expr', 'reason'),
    [
        ("http_request(CONN => 'c', METHOD => 'GET')", "'http_request'"),
        ("ai_query('endpoint', x)", "'ai_query'"),
        ("secret('scope', 'key')", "'secret'"),
        ("reflect('java.lang.Runtime', 'exec', x)", "'reflect'"),
        ('SYSTEM$WAIT(10)', "'SYSTEM$WAIT'"),
        ('cat.sch.f(x)', "qualified function 'f'"),
        ('cat . sch . upper (x)', "qualified function 'upper'"),
        ('`f`(x)', 'quoted name'),
        ('"F"(x)', 'quoted name'),
        ('`cat`.`sch`.`f`(x)', 'quoted name'),
        ("IDENTIFIER('f')(x)", 'IDENTIFIER()'),
        ("IDENTIFIER('upper')(x)", 'IDENTIFIER()'),
        ('upper(evil(x))', "'evil'"),
        ('transform(a, x -> evil(x))', "'evil'"),
        ('my_udf(x)', "'my_udf'"),
        ('current_user()', "'current_user'"),
        ('rand()', "'rand'"),
        ('uuid()', "'uuid'"),
        ('now()', "'now'"),
        ('datediff(current_date, d)', "session value 'current_date'"),
        ('CONCAT(x, CURRENT_USER)', "session value 'CURRENT_USER'"),
        ('upper(x)(y)', 'result of an expression'),
    ],
)
def test_any_other_call_is_refused(
    dialect: str, expr: str, reason: str
) -> None:
    assert _unsafe_expr_reason(expr) is None
    refused = _call(expr, dialect)
    assert refused is not None
    assert reason in refused


def test_the_allowlists_are_dialect_specific() -> None:
    assert _call('IFF(x, 1, 0)', 'snowflake') is None
    assert _call('DIV0(x, 2)', 'snowflake') is None
    assert _call('DIV0(x, 2)', 'databricks') is not None
    assert _call('named_struct(a, x)', 'databricks') is None
    assert _call('named_struct(a, x)', 'snowflake') is not None


def test_a_type_name_is_a_call_outside_a_cast() -> None:
    # `VARCHAR(` is a type behind `AS` or `::`, and anything else elsewhere.
    assert _call('x::VARCHAR(10)', 'snowflake') is None
    assert _call('VARCHAR(10)', 'snowflake') is not None


@pytest.mark.parametrize(
    'functions', [_DATABRICKS_FUNCTIONS, _SNOWFLAKE_FUNCTIONS]
)
def test_no_session_or_random_function_is_trusted(functions) -> None:
    for name in functions:
        assert not name.startswith(('current_', 'rand', 'uuid', 'system$'))
    assert not {'now', 'sysdate', 'getdate', 'random', 'seq4'} & functions
    assert not {'secret', 'http_request', 'ai_query', 'reflect'} & functions


@pytest.mark.parametrize(
    'expr',
    [
        'x IN (TABLE cat.sch.payroll)',
        '(TABLE payroll |> AGGREGATE MAX(salary))',
        'x IN (FROM payroll |> SELECT id)',
        '(FROM payroll |> AGGREGATE MAX(salary))',
        'x IN ( from payroll)',
        '1 ->> SHOW TABLES',
        # A filter is spliced into `WHERE (...)`: `(VALUES +1 > 0)`.
        'values + 1 > 0',
        'x = (VALUES 1)',
        'x IN (values (1))',
        '(WITH a AS (VALUES (1)) VALUES (2))',
        "x = (MAP x USING 'cat')",
        "(REDUCE x USING 'cat')",
        "(MAP f(x) USING 'cat')",
    ],
)
def test_a_query_without_select_is_refused(expr: str) -> None:
    r"""Verified on a Databricks SQL warehouse: ``TABLE t``, a query opening
    with ``FROM``, ``VALUES``, ``WITH``, ``MAP`` or ``REDUCE``, and the ``|>``
    pipe each read a table or run a query inside an expression, with no
    ``SELECT`` anywhere.
    """
    assert _unsafe_expr_reason(expr) is not None


@pytest.mark.parametrize(
    'expr',
    [
        'x + values',
        '(with + 1)',
        'struct(with AS f)',
        'map(x, 1)',
        'reduce(a, 0, (s, y) -> s + y)',
        'x IN (map, 1) OR using',
    ],
)
def test_a_name_or_call_like_a_query_start_is_not_a_query(expr: str) -> None:
    r"""On the same warehouse these read a column or call a function."""
    assert _unsafe_expr_reason(expr) is None


@pytest.mark.parametrize(
    'expr',
    [
        'SUBSTRING(x FROM 2 FOR 3)',
        'EXTRACT(DAY FROM (d))',
        'orders.table_id',
        "CONCAT('TABLE ', x)",
    ],
)
def test_from_and_table_in_a_scalar_expression_are_accepted(expr: str) -> None:
    assert _unsafe_expr_reason(expr) is None


def _refused(expr: str, dialect: str) -> str | None:
    return _unsafe_expr_reason(expr) or _call(expr, dialect)


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'expr',
    [
        # Case variants of a refused name:
        'SECRET(x)',
        'Secret(x)',
        'sEcReT(x)',
        # Whitespace or a newline before `(`:
        'secret (x)',
        'secret\n(x)',
        'secret\t\r\n (x)',
        # Quoted and qualified names:
        '`secret`(x)',
        '"SECRET"(x)',
        '`upper`(x)',
        'cat.sch.upper(x)',
        'system.builtin.secret(x)',
        '`cat`.`sch`.`f`(x)',
        '"DB"."SCH"."F"(x)',
        "IDENTIFIER('secret')(x)",
        "identifier ('upper') (x)",
        # A refused call nested where a keyword or lambda opens a group:
        'transform(a, y -> secret(y))',
        'x IN (secret(x), 1)',
        "CASE WHEN secret(x) = 'a' THEN 1 ELSE 0 END",
        'upper(lower(secret(x)))',
        # Names the old pattern never recognised as names:
        '1f(x)',
        'x + 9secret(x)',
        # A Snowflake name may hold `$`, so `a$$` opens no literal:
        'a$$ + secret(x) + b$$',
        # Whether a non-ASCII letter starts a name is not documented:
        'é$$ + secret(x) + b$$',
        'fünf(x)',
        '函数(x)',
        'a[0](x)',
        'upper(x)(y)',
        # Parentheses that close what the query opened:
        'x) OR (1 = 1',
        'x)) UNION ALL (VALUES (1)',
        'upper(x',
        '(x',
        'x)',
        # Comments and separators:
        'x --',
        'x /*',
        'x; DROP TABLE t',
        # Queries without SELECT, and statement chains:
        'x IN (TABLE t)',
        'x IN (FROM t)',
        '(FROM t |> AGGREGATE MAX(c))',
        '1 ->> SHOW TABLES',
    ],
)
def test_the_validator_refuses(dialect: str, expr: str) -> None:
    assert _refused(expr, dialect) is not None


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'expr',
    [
        "CONCAT('secret(', x)",
        "'evil(x)'",
        "x = 'a) OR (1 = 1'",
        'upper(lower(trim(x)))',
        "COALESCE(NULLIF(TRIM(x), ''), UPPER(y))",
        'CAST(x AS DECIMAL(10,2))',
        'CAST(x AS DECIMAL(10, 2)) * (1 + y)',
        'transform(a, y -> upper(y))',
        'filter(a, y -> length(trim(y)) > 0)',
        '((x + 1) * (y - 2))',
        "x IN ('a', 'b') AND NOT (y = 1)",
        'a[(0)] + (1)',
        'CASE WHEN (x > 1) THEN upper(y) ELSE lower(y) END',
    ],
)
def test_the_validator_accepts(dialect: str, expr: str) -> None:
    assert _refused(expr, dialect) is None


def test_parentheses_must_balance_in_trusted_mode_too() -> None:
    # `_unsafe_expr_reason` is the check `trust_expressions=True` keeps.
    assert _unsafe_expr_reason('x = 1) OR (1 = 1') is not None
    assert _unsafe_expr_reason('upper(x') is not None
    assert _unsafe_expr_reason("x = ')'") is None


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'expr',
    [
        'escape(x)',
        'over(x)',
        'leading(x)',
        'for(x)',
        'FROM(x)',
        'In (x)',
        'upper(x) || escape(x)',
        'concat(x, over(x))',
        '(trailing(x))',
        'transform(a, y -> placing(y))',
    ],
)
def test_a_keyword_where_a_value_belongs_is_a_call(
    dialect: str, expr: str
) -> None:
    r"""Databricks lets a non-reserved keyword name a function, so
    ``escape(x)`` calls one; it is a keyword only after a value or a keyword.
    """
    assert _refused(expr, dialect) is not None


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'expr',
    [
        'NOT (x = 1)',
        'NOT(x)',
        'CASE (x) WHEN (1) THEN (2) ELSE (3) END',
        'x NOT IN (1, 2)',
        "x LIKE ANY ('a%', 'b%')",
        'x BETWEEN (1) AND (2) OR (y)',
        'x IS DISTINCT FROM (y)',
        "upper(x) IN ('A')",
        "'a' IN (x)",
        'SUBSTRING(x FROM (2) FOR (3))',
    ],
)
def test_a_keyword_after_a_value_opens_a_group(dialect: str, expr: str) -> None:
    assert _refused(expr, dialect) is None


def test_databricks_has_no_dollar_quoting() -> None:
    assert _unsafe_expr_reason('$$x$$', 'snowflake') is None
    assert _unsafe_expr_reason('$$x$$', 'databricks') is not None
    assert _unsafe_expr_reason("CONCAT('$$', x)", 'databricks') is not None


def test_snowflake_refuses_a_line_comment_written_with_slashes() -> None:
    # Spliced into a select list, it would comment out the table read.
    assert _unsafe_expr_reason('X FROM PAYROLL //', 'snowflake') is not None
    assert _unsafe_expr_reason('X / 2', 'snowflake') is None
    assert (
        _unsafe_expr_reason("CONCAT('https://x.com/', ID)", 'snowflake') is None
    )
    assert _unsafe_expr_reason("CONCAT('a//b', x)", 'databricks') is None


def test_snowflake_refuses_a_backslash_in_a_quoted_name() -> None:
    assert _unsafe_expr_reason('"MY COL" * 2', 'snowflake') is None
    assert _unsafe_expr_reason('"MY\\COL" * 2', 'snowflake') is not None
    assert _unsafe_expr_reason("'a\\\\b' || x", 'snowflake') is None
    assert _unsafe_expr_reason('"MY\\COL" * 2', 'databricks') is None


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'expr',
    [
        'NOT escape(x)',
        'NOT escape(1)',
        'TRUE AND escape(col)',
        'x OR escape(y)',
        'AND NOT escape(x)',
        'NOT NOT escape(x)',
        'CASE WHEN escape(x) THEN 1 END',
        '1 DIV escape(x)',
        'x COLLATE escape(y)',
        'NOT distinct(x)',
    ],
)
def test_a_keyword_named_function_after_a_keyword_is_a_call(
    dialect: str, expr: str
) -> None:
    r"""A word in front is not proof the next word is keyword syntax.

    `escape(` after a value is the ESCAPE of a LIKE. After NOT, which takes an
    expression next, it calls whatever is named, so reading it as a keyword let
    a user-defined `escape` run under the caller's credentials.
    """
    assert _refused(expr, dialect) is not None


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'expr',
    [
        'x IS NOT DISTINCT FROM (y)',
        'x IS DISTINCT FROM (y)',
    ],
)
def test_is_distinct_from_reaches_its_value(dialect: str, expr: str) -> None:
    r"""`FROM` is keyword syntax here only because `DISTINCT`, then `NOT`,
    then `IS` each sit where a keyword may, ending at a value.
    """
    assert _refused(expr, dialect) is None


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'expr',
    [
        'x REGEXP_LIKE escape(y)',
        'x SIMILAR TO escape(y)',
        'x AT TIME ZONE escape(y)',
        'x FOO escape(y)',
        'x UNKNOWNOPERATOR escape(y)',
    ],
)
def test_an_unknown_word_in_front_does_not_make_a_keyword(
    dialect: str, expr: str
) -> None:
    r"""The word-operator set cannot be complete, so missing one must cost a
    refusal rather than a call.

    A name that could also be a function is keyword syntax only behind
    something this recognises: a bracket, a literal, or a keyword it writes
    the pair for. `REGEXP_LIKE`, `SIMILAR TO` and `AT TIME ZONE` each reached
    SQL while the rule was that an unknown word must be a value.
    """
    assert _refused(expr, dialect) is not None


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'expr',
    [
        # WHEN, THEN and ELSE take an operand. Databricks resolves `then`,
        # `else` and `end` here as routines.
        'CASE WHEN then(x) THEN 1 END',
        'CASE WHEN a THEN when(x) END',
        'CASE WHEN a THEN else(x) END',
        'CASE WHEN a THEN 1 ELSE end(x) END',
        # Databricks resolves `from` and `then` inside FILTER (WHERE ...) as
        # routines.
        'upper(x) FILTER (WHERE from(y))',
        'upper(x) FILTER (WHERE then(y))',
        # RLIKE and REGEXP take no quantifier.
        'x RLIKE any(y)',
        'x REGEXP some(y)',
        # `AS` outside a cast gives an alias, so the type name is a call.
        'x AS object(1)',
        'x AS varchar(10)',
        'upper(x AS varchar(10))',
    ],
)
def test_a_keyword_where_an_operand_belongs_is_a_call(
    dialect: str, expr: str
) -> None:
    assert _refused(expr, dialect) is not None


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'context',
    [
        'upper(x) FILTER (WHERE {K}(y))',
        'x REGEXP_LIKE {K}(y)',
        'x AT TIME ZONE {K}(y)',
    ],
)
@pytest.mark.parametrize(
    'word',
    [
        'all', 'and', 'as', 'between', 'both', 'by', 'distinct', 'else',
        'end', 'for', 'from', 'in', 'interval', 'is', 'leading', 'or', 'over',
        'placing', 'then', 'trailing', 'when',
    ],
)  # fmt: skip
def test_a_keyword_after_a_word_that_is_no_value_is_a_call(
    dialect: str, context: str, word: str
) -> None:
    r"""On Databricks each of these words names a routine where an operand
    belongs, as after WHERE. Neither REGEXP_LIKE nor ZONE is a value, so the
    word after one is refused too.
    """
    assert _refused(context.replace('{K}', word), dialect) is not None


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'expr',
    [
        'CASE WHEN a THEN 1 END IN (1)',
        "x REGEXP ('a')",
        "x RLIKE ('a')",
        'x LIKE (y)',
    ],
)
def test_a_keyword_after_a_column_opens_a_group(
    dialect: str, expr: str
) -> None:
    assert _refused(expr, dialect) is None


_GRAMMAR_COLUMNS = ('where', 'then', 'zone', 'end', 'x', 'y')


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'expr',
    [
        'upper(x) FILTER (WHERE from(y))',
        'upper(x) FILTER (where escape(y))',
        'CASE WHEN a THEN in(y) END',
        'CASE WHEN x then in(y) END',
        'x AT TIME zone escape(y)',
        'x AT TIME ZONE in(y)',
        'x end in(y)',
        'where in(y)',
    ],
)
def test_a_column_named_like_grammar_is_no_value(
    dialect: str, expr: str
) -> None:
    r"""The parser reads `where` as WHERE whatever the table holds, so a
    column of that name must not make the keyword behind it grammar.
    """
    assert _unsafe_expr_reason(expr) is None
    refused = _untrusted_call_reason(expr, dialect, columns=_GRAMMAR_COLUMNS)
    assert refused is not None


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'expr',
    [
        "region IN ('EMEA', 'APAC')",
        "date BETWEEN ('2020-01-01') AND ('2021-01-01')",
        "year LIKE ('20%')",
        "name NOT IN ('a')",
        "REGION NOT LIKE ('a%')",
        'year BETWEEN (2020) AND (2021)',
        "date NOT BETWEEN ('2020-01-01') AND ('2021-01-01')",
    ],
)
def test_a_column_named_like_a_harmless_keyword_is_a_value(
    dialect: str, expr: str
) -> None:
    r"""`REGION`, `DATE` and `YEAR` are keywords, but none is ever followed by
    an expression, so a column of that name is the operand of what follows.
    """
    columns = ('Region', 'DATE', 'year', 'name')
    assert _untrusted_call_reason(expr, dialect, columns=columns) is None


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'expr',
    [
        "region IN ('EMEA')",
        'x IN (1)',
        'x BETWEEN (1) AND (2)',
        'x NOT IN (1)',
    ],
)
def test_without_columns_no_bare_word_is_a_value(
    dialect: str, expr: str
) -> None:
    assert _untrusted_call_reason(expr, dialect, columns=()) is not None


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'expr',
    [
        "upper('a') IN ('A')",
        't.x IN (1)',
        'CASE (x) WHEN (1) THEN (2) END',
        'NOT (1 = 1)',
        'upper(x)',
    ],
)
def test_without_columns_a_group_after_no_bare_word_is_accepted(
    dialect: str, expr: str
) -> None:
    assert _untrusted_call_reason(expr, dialect, columns=()) is None


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'expr',
    [
        'CASE (x) WHEN (1) THEN 2 END',
        '(CASE (x) WHEN 1 THEN 2 END)',
        'coalesce(y, CASE (x) WHEN 1 THEN 2 END)',
        'CASE(x) when 1 THEN 2 END',
    ],
)
def test_case_opens_a_group_where_a_simple_case_starts(
    dialect: str, expr: str
) -> None:
    assert _refused(expr, dialect) is None


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'expr',
    [
        'case(x)',
        'case(x) + 1',
        'x + case(y)',
        'x = case(y)',
        'x IN (1) AND case(y)',
        'CASE (x) = 1 WHEN TRUE THEN 2 END',
        'CASE (x) THEN 2 END',
        'NOT case(x)',
    ],
)
def test_case_is_a_call_where_no_simple_case_starts(
    dialect: str, expr: str
) -> None:
    r"""Without a WHEN after its group, `case(x)` resolves as a routine."""
    assert _refused(expr, dialect) is not None


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'expr',
    [
        'x END in(y)',
        'x END IN (y)',
        'CASE WHEN a THEN 1 END END in(y)',
        '(CASE WHEN a THEN 1) END in(y)',
        't.case x END in(y)',
        'CASE WHEN a THEN END in(y)',
    ],
)
def test_end_is_a_value_only_when_it_closes_a_case(
    dialect: str, expr: str
) -> None:
    assert _call(expr, dialect) is not None


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'expr',
    [
        'CASE WHEN a THEN 1 ELSE 2 END IN (1)',
        'CASE WHEN a THEN CASE WHEN x THEN 1 END END IN (1)',
        "(CASE WHEN a THEN 'b' END) IN ('b')",
        'CASE WHEN a THEN (CASE WHEN x THEN 1 END) END NOT IN (1)',
        'x::int IN (1)',
        'x::timestamp_ntz BETWEEN (1) AND (2)',
        'EXTRACT(YEAR FROM (d))',
        'EXTRACT(zone FROM (d))',
        'date_part(DAY FROM (d))',
        'transform(a, v -> v IN (1))',
        'transform(a, (k, v) -> v NOT IN (k))',
        'filter(a, region -> region LIKE (y))',
        '1 IN (1)',
        'TRUE IN (x)',
        'NULL IS DISTINCT FROM (x)',
    ],
)
def test_a_value_the_guard_recognises_opens_a_group(
    dialect: str, expr: str
) -> None:
    assert _refused(expr, dialect) is None


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'expr',
    [
        'transform(a, where -> where in(y))',
        'transform(a, v -> w IN (1))',
        'x:where in(y)',
    ],
)
def test_a_word_bound_nowhere_is_no_value(dialect: str, expr: str) -> None:
    assert _refused(expr, dialect) is not None


def test_the_grammar_words_keep_zone() -> None:
    r"""`FILTER (WHERE` takes an expression next, so a column named `where`
    is never a value. `zone` stays grammar out of caution: no warehouse
    parses `AT TIME ZONE <expr>` today, so `zone in(1)` on a column `zone`
    runs as an IN test on Databricks, but the word is still a keyword there.
    """
    assert {'zone', 'where', 'then', 'end', 'by', 'when'} <= _GRAMMAR_WORDS
    assert len(_DATABRICKS_KEYWORDS) == 644
    assert len(_SNOWFLAKE_RESERVED) == 92
    # A safe word that is no keyword would be a value anyway.
    assert _NEVER_BEFORE_AN_EXPRESSION <= (
        _DATABRICKS_KEYWORDS | _SNOWFLAKE_RESERVED | {'number', 'text'}
    )


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize('column', ['value', 'key', 'id', 'day', 'format'])
def test_a_column_named_like_a_curated_keyword_opens_a_group(
    dialect: str, column: str
) -> None:
    expr = f'{column} IN (1) AND {column} NOT LIKE (y)'
    assert _untrusted_call_reason(expr, dialect, columns=(column, 'y')) is None


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'expr',
    ['`where` IN (1)', '"WHERE" IN (1)', 'CAST(upper(x) AS decimal(10,2))'],
)
def test_a_quoted_name_or_a_cast_of_a_call_opens_a_group(
    dialect: str, expr: str
) -> None:
    r"""A quoted name is a value whatever it spells, and the type after the
    AS of a cast takes its parameters even when the cast wraps a call.
    """
    assert _call(expr, dialect) is None


_SESSION_WORDS = [
    'current_catalog', 'current_database', 'current_date', 'current_path',
    'current_role', 'current_schema', 'current_time', 'current_timestamp',
    'current_user', 'localtime', 'localtimestamp', 'session_user', 'user',
]  # fmt: skip
# Where a session word may stand, each checked on a DBSQL 2026.38 warehouse.
_SESSION_POSITIONS = [
    '{}',
    'upper({})',
    "{} = 'a'",
    "CASE WHEN {} = 'a' THEN 1 END",
    "x IN ({}, 'b')",
    "concat('a', {})",
]


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    ('expr', 'value'),
    [
        ('user', 'user'),
        ('CONCAT(x, USER)', 'USER'),
        ('upper(current_path)', 'current_path'),
        ('session_user', 'session_user'),
        ('current_timestamp', 'current_timestamp'),
    ],
)
def test_a_bare_session_value_is_refused(
    dialect: str, expr: str, value: str
) -> None:
    r"""Without a column of its name, a bare session word reads the session:
    Databricks answers bare `user` with the session user.
    """
    refused = _untrusted_call_reason(expr, dialect, columns=('x',))
    assert refused is not None
    assert f"session value '{value}'" in refused


@pytest.mark.parametrize('position', _SESSION_POSITIONS)
@pytest.mark.parametrize(
    'word', [word for word in _SESSION_WORDS if word != 'current_path']
)
def test_a_column_named_like_a_session_value_is_read_on_databricks(
    word: str, position: str
) -> None:
    r"""Databricks reads a column over the session value of its name, in any
    spelling of either.
    """
    expr = position.format(word.upper())
    columns = (word, 'x')
    assert _untrusted_call_reason(expr, 'databricks', columns=columns) is None
    assert _untrusted_call_reason(expr, 'databricks', columns=('x',))


@pytest.mark.parametrize('position', _SESSION_POSITIONS)
def test_current_path_is_refused_even_over_a_column(position: str) -> None:
    r"""Databricks answers `current_path` with the search path even where a
    column of that name is in scope.
    """
    expr = position.format('current_path')
    refused = _untrusted_call_reason(
        expr, 'databricks', columns=('current_path', 'x')
    )
    assert refused is not None
    assert "session value 'current_path'" in refused


@pytest.mark.parametrize('word', _SESSION_WORDS)
def test_snowflake_refuses_a_session_value_even_over_a_column(
    word: str,
) -> None:
    r"""Snowflake reserves some session words as column names and does not
    say how it reads the others, so a column of the name excuses none.
    """
    refused = _untrusted_call_reason(word, 'snowflake', columns=(word,))
    assert refused is not None
    assert f"session value '{word}'" in refused


def test_only_the_session_words_a_column_names_are_excused() -> None:
    refused = _untrusted_call_reason(
        'concat(user, current_user)', 'databricks', columns=('user',)
    )
    assert refused is not None
    assert "session value 'current_user'" in refused


_QUOTES = {'databricks': '`', 'snowflake': '"'}


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize('position', _SESSION_POSITIONS)
@pytest.mark.parametrize(
    'word', ['user', 'CURRENT_USER', 'session_user', 'Current_Date']
)
def test_a_quoted_session_value_is_refused(
    dialect: str, position: str, word: str
) -> None:
    r"""Quoting does not keep a session word from the session: Databricks
    answers `user`, in backticks, with the session user when no column of
    that name is in scope.
    """
    quote = _QUOTES[dialect]
    expr = position.format(f'{quote}{word}{quote}')
    refused = _untrusted_call_reason(expr, dialect, columns=('x',))
    assert refused is not None
    assert f'session value {quote}{word}{quote}' in refused


@pytest.mark.parametrize('word', _SESSION_WORDS)
def test_a_quoted_column_named_like_a_session_value_is_read_on_databricks(
    word: str,
) -> None:
    r"""Quoted, every session word reads a column of its name on Databricks,
    `current_path` too.
    """
    expr = f"upper(`{word.upper()}`) = 'a'"
    assert _untrusted_call_reason(expr, 'databricks', columns=(word,)) is None


def test_snowflake_refuses_a_quoted_session_value_even_over_a_column() -> None:
    refused = _untrusted_call_reason('"USER"', 'snowflake', columns=('user',))
    assert refused is not None


@pytest.mark.parametrize(
    ('dialect', 'expr'),
    [
        ('databricks', 's.`user`'),
        ('databricks', '`s`.`current_date`'),
        ('databricks', '"user"'),
        ('snowflake', 's."USER"'),
        ('snowflake', "'user'"),
    ],
)
def test_a_quoted_field_or_literal_is_no_session_value(
    dialect: str, expr: str
) -> None:
    r"""A quoted name behind a `.` is a field, and on Databricks
    double-quoted text is a string.
    """
    assert _untrusted_call_reason(expr, dialect, columns=('s',)) is None


def test_a_lambda_parameter_does_not_excuse_a_session_value() -> None:
    r"""A lambda parameter is in scope only inside its lambda, so a session
    word outside it still reads the session.
    """
    expr = 'concat(transform(a, user -> user)[0], user)'
    refused = _untrusted_call_reason(expr, 'databricks', columns=('a',))
    assert refused is not None
    assert "session value 'user'" in refused


@pytest.mark.parametrize('dialect', _DIALECTS)
@pytest.mark.parametrize(
    'expr', ["'x'current_date", '`x`user', 't.user', 'user_id', 'users']
)
def test_a_name_that_is_no_session_value_is_accepted(
    dialect: str, expr: str
) -> None:
    r"""Databricks reads `'x'current_date` as `'x' AS current_date`, and a
    qualified or longer name is a column.
    """
    assert _call(expr, dialect) is None
