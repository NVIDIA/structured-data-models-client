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
    _SNOWFLAKE_FUNCTIONS,
    _unsafe_expr_reason,
    _untrusted_call_reason,
)

_DIALECTS = ['databricks', 'snowflake']


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
    assert _untrusted_call_reason(expr, dialect) is None


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
    refused = _untrusted_call_reason(expr, dialect)
    assert refused is not None
    assert reason in refused


def test_the_allowlists_are_dialect_specific() -> None:
    assert _untrusted_call_reason('IFF(x, 1, 0)', 'snowflake') is None
    assert _untrusted_call_reason('DIV0(x, 2)', 'snowflake') is None
    assert _untrusted_call_reason('DIV0(x, 2)', 'databricks') is not None
    assert _untrusted_call_reason('named_struct(a, x)', 'databricks') is None
    assert _untrusted_call_reason('named_struct(a, x)', 'snowflake') is not None


def test_a_type_name_is_a_call_outside_a_cast() -> None:
    # `VARCHAR(` is a type behind `AS` or `::`, and anything else elsewhere.
    assert _untrusted_call_reason('x::VARCHAR(10)', 'snowflake') is None
    assert _untrusted_call_reason('VARCHAR(10)', 'snowflake') is not None


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
    ],
)
def test_a_query_without_select_is_refused(expr: str) -> None:
    r"""Verified on a Databricks SQL warehouse: ``TABLE t``, a query opening
    with ``FROM`` and the ``|>`` pipe each read a table inside an expression,
    with no ``SELECT`` anywhere.
    """
    assert _unsafe_expr_reason(expr) is not None


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
