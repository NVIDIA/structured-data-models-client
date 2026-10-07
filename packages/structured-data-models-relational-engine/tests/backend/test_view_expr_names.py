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
