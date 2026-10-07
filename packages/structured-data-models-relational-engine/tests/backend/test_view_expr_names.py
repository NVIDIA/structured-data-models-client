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
from sdm_relational_engine.rfm.view_expr import name_references


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
        ('row_number() OVER (PARTITION BY row ORDER BY row)', {'row'}),
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
        ('range DIV range', {'range'}),
        ('from COLLATE UTF8_LCASE = from', {'from'}),
        ('else IS UNKNOWN', {'else'}),
        ('end IS NOT DISTINCT FROM between', {'end', 'between'}),
        ('rows NOT LIKE rows', {'rows'}),
        # Where an operand stands, a word is a column unless it opens a form.
        ('placing OR NOT rlike', {'placing', 'rlike'}),
        ('NOT div', {'div'}),
        ("DATE '2020-01-01' + ilike", {'ilike'}),
        ('COUNT(*) FILTER (WHERE between)', {'between'}),
        ('COUNT(DISTINCT from, from)', {'from'}),
        ('CASE from WHEN from THEN from END', {'from'}),
        ('CASE WHEN NOT is THEN is END', {'is'}),
        ('CASE CASE WHEN both THEN 1 END WHEN 1 THEN both END', {'both'}),
        # A CASE is a form only if the rest then parses: no END is left here.
        ('CASE case WHEN when THEN end END', {'case', 'when', 'end'}),
        ('CASE WHEN case THEN end ELSE not END', {'case', 'end', 'not'}),
        ('CASE WHEN x THEN not END', {'x', 'not'}),
        ('all:a', {'all'}),
        ('transform(arr, (z, i) -> all || i)', {'arr', 'all'}),
    ],
)
def test_a_name_is_a_column_where_databricks_reads_one(
    expr: str, columns: set[str]
) -> None:
    r"""A set of keywords decided which words were columns, so a column
    named `range` was lost in `PARTITION BY range`, and `REGEXP` and `FOR`
    were read as columns no view names.
    """
    assert {'.'.join(name.parts) for name in name_references(expr)} == columns
