# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Which names a view's expression reads as columns.

A column may take a keyword's name, so where a word stands, not the word,
decides whether it is a column. Each case below reads what the analyzed plan
of `SELECT (<expr>) FROM t AS source` referenced on a DBSQL 2026.39
warehouse, for a table `t` whose columns are named like keywords.
"""

from __future__ import annotations

import pytest
from sdm_relational_engine.rfm.view_expr import (
    UnreadableExpressionError,
    name_references,
)

_PROBE = '(CASE WHEN x THEN not END)'


@pytest.mark.parametrize(
    'expr, columns',
    [
        # A window's clauses take any expression.
        ('SUM(x) OVER (PARTITION BY range)', {'x', 'range'}),
        (
            'SUM(x) OVER (PARTITION BY partition ORDER BY y)',
            {'x', 'partition', 'y'},
        ),
        ('SUM(x) OVER (PARTITION BY region ORDER BY y)', {'x', 'region', 'y'}),
        ('SUM(x) OVER (PARTITION BY source.range)', {'x', 'source.range'}),
        (
            'SUM(x) OVER (ORDER BY rows ROWS BETWEEN 1 PRECEDING AND '
            '1 FOLLOWING)',
            {'x', 'rows'},
        ),
        ('first_value(desc) IGNORE NULLS OVER (ORDER BY desc)', {'desc'}),
        (
            'percentile_cont(0.5) WITHIN GROUP (ORDER BY window DESC)',
            {'window'},
        ),
        # A word right after an operand is an operator or a clause's word.
        ("col REGEXP 'x'", {'col'}),
        ('SUBSTRING(col FROM 1 FOR 2)', {'col'}),
        ('OVERLAY(order PLACING order FROM 1 FOR 2)', {'order'}),
        ('TRIM(BOTH struct FROM using)', {'struct', 'using'}),
        ('from COLLATE UTF8_LCASE = from', {'from'}),
        ('else IS UNKNOWN', {'else'}),
        ('end IS NOT DISTINCT FROM between', {'end', 'between'}),
        ('rows NOT LIKE rows', {'rows'}),
        # Where an operand stands, a word is a column unless it opens a form
        # after which the rest parses.
        ('placing OR NOT rlike', {'placing', 'rlike'}),
        ('NOT div', {'div'}),
        ("DATE '2020-01-01' + ilike", {'ilike'}),
        ('COUNT(*) FILTER (WHERE between)', {'between'}),
        ('COUNT(DISTINCT from, from)', {'from'}),
        ('CASE from WHEN from THEN from END', {'from'}),
        ('CASE WHEN NOT is THEN is END', {'is'}),
        ('CASE CASE WHEN both THEN 1 END WHEN 1 THEN both END', {'both'}),
        ('CASE case WHEN when THEN end END', {'case', 'when', 'end'}),
        ('CASE WHEN x THEN not END', {'x', 'not'}),
        # However long or deep the expression around it.
        pytest.param('NOT (' * 63 + _PROBE + ')' * 63, {'x', 'not'}, id='deep'),
        pytest.param(
            'NOT (' * 34 + 'y' + ')' * 34 + ' AND ' + _PROBE,
            {'x', 'not', 'y'},
            id='deep-then-shallow',
        ),
        pytest.param(
            ' AND '.join(f'NOT c{i}' for i in range(1000)) + ' AND ' + _PROBE,
            {f'c{i}' for i in range(1000)} | {'x', 'not'},
            id='long',
        ),
        # A JSON path's base, and a column in a lambda.
        ('all:a', {'all'}),
        ('transform(arr, (z, i) -> all || i)', {'arr', 'all'}),
        # A type ends where the cast grammar ends it, so the form around
        # the cast still parses: past its parameters, fields or words.
        ('CAST(CASE WHEN a THEN b END AS DECIMAL(38,2))', {'a', 'b'}),
        ('CAST(CASE WHEN a THEN b END AS INT)', {'a', 'b'}),
        ('CAST(COALESCE(b, c) AS DECIMAL(38,2))', {'b', 'c'}),
        (
            'TRY_CAST(CASE case WHEN end THEN end END AS CHAR(3))',
            {'case', 'end'},
        ),
        ('CAST(NOT not AS DECIMAL(3,1))', {'not'}),
        ('COUNT(DISTINCT CAST(distinct AS VARCHAR(10)))', {'distinct'}),
        ('COUNT(ALL all::DECIMAL(10))', {'all'}),
        ('CASE WHEN when THEN then::DECIMAL(10) END', {'when', 'then'}),
        (
            'CAST(CASE WHEN case THEN CASE WHEN when THEN end END END '
            'AS DECIMAL(38,2)) + end',
            {'case', 'when', 'end'},
        ),
        (
            'CAST(CASE WHEN interval THEN day END AS INTERVAL DAY TO SECOND)',
            {'interval', 'day'},
        ),
        (
            'CAST(CASE WHEN with THEN local END '
            'AS TIMESTAMP WITH LOCAL TIME ZONE)',
            {'with', 'local'},
        ),
        (
            'CAST(CASE WHEN zone THEN without END '
            'AS TIMESTAMP WITHOUT TIME ZONE)',
            {'zone', 'without'},
        ),
        ('CASE WHEN with THEN time::timestamp END', {'with', 'time'}),
        ('CAST(CASE WHEN case THEN NULL END AS STRUCT<>) IS NULL', {'case'}),
        (
            'COUNT(x) OVER (PARTITION BY CAST(CASE WHEN end THEN end END '
            'AS DECIMAL(10)) ORDER BY y)',
            {'x', 'end', 'y'},
        ),
        (
            'SUM(decimal) FILTER (WHERE NOT decimal::decimal(38, 2) < int)',
            {'decimal', 'int'},
        ),
        # Only ARRAY, MAP and STRUCT take `<...>`: after any other type or a
        # collation, `<` compares.
        ('x::int < y AND n > 1', {'x', 'y', 'n'}),
        ('map COLLATE UTF8_LCASE < string AND n > 1', {'map', 'string', 'n'}),
    ],
)
def test_a_name_is_a_column_where_databricks_reads_one(
    expr: str, columns: set[str]
) -> None:
    r"""Where a word stands decides whether it names a column: a column may
    take a keyword's name, and a keyword after an operand is grammar.
    """
    assert {'.'.join(name.parts) for name in name_references(expr)} == columns


@pytest.mark.parametrize('depth', [64, 200])
@pytest.mark.parametrize('opener', ['NOT (', 'f(DISTINCT '])
def test_an_expression_nested_too_deep_is_refused(
    opener: str, depth: int
) -> None:
    r"""Past the deepest trial the scan keeps, it says so rather than guess,
    which would read `END` as a column and lose `not`.
    """
    with pytest.raises(UnreadableExpressionError):
        name_references(opener * depth + _PROBE + ')' * depth)
