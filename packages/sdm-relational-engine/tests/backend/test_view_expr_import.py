# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1
r"""Column expressions lifted out of a semantic/metric view definition run as
SQL under the caller's warehouse role, so a hostile view definition must not
reach the sampler.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from predictive_query.schema.typing import Dtype
from sdm_relational_engine.rfm.base import SourceColumn, SourceForeignKey
from sdm_relational_engine.rfm.graph import _unsafe_expr_reason

_EXFILTRATION = '(SELECT MAX(SALARY) FROM PAYROLL)'


@pytest.mark.parametrize(
    'expr',
    [
        _EXFILTRATION,
        'NAME; DROP TABLE USERS',
        'NAME -- trailing comment',
        'NAME /* block */',
        'CASE WHEN 1=1 THEN (SELECT 1) ELSE NAME END',
        'name) AS x, (SELECT 1',
    ],
)
def test_unsafe_expressions_are_reported(expr):
    assert _unsafe_expr_reason(expr) is not None


@pytest.mark.parametrize(
    'expr',
    [
        "'x' UNION SELECT SECRET FROM PAYROLL",
        "COALESCE(NAME, 'none') || (SELECT MAX(SALARY) FROM PAYROLL)",
        '"drop" || (SELECT 1)',
        "CASE WHEN NOTE = 'select' THEN (SELECT 1) ELSE 0 END",
    ],
)
def test_keywords_outside_quotes_are_still_reported(expr):
    r"""Ignoring quoted text must not become a way to smuggle a sub-query in:
    an expression that also carries a quoted literal is still refused when a
    keyword sits outside the quotes.
    """
    assert _unsafe_expr_reason(expr) is not None


@pytest.mark.parametrize(
    'expr',
    [
        'NAME',
        '"my name"',
        'DB.PUBLIC.USERS.NAME',
        'UPPER(NAME)',
        'CAST(CREATED_AT AS DATE)',
        'QUANTITY * UNIT_PRICE_USD',
        "CASE WHEN SEGMENT = 'SMB' THEN 1 ELSE 0 END",
        "CONCAT(FIRST_NAME, ' ', LAST_NAME)",
        'DATE_TRUNC(month, CREATED_AT)',
        'COALESCE(OFFSET_DAYS, 0)',
    ],
)
def test_ordinary_scalar_expressions_are_accepted(expr):
    assert _unsafe_expr_reason(expr) is None


@pytest.mark.parametrize(
    'expr',
    [
        "CASE WHEN STATUS = 'select' THEN 1 ELSE 0 END",
        "CONCAT('drop', NAME)",
        "COALESCE(REASON, 'no merge')",
        "CASE WHEN NOTE = 'update pending' THEN 1 ELSE 0 END",
        '"drop"',
        '`delete`',
        "IFF(ACTION = 'delete', 1, 0)",
    ],
)
def test_keywords_inside_quoted_text_are_accepted(expr):
    r"""A keyword inside a string literal or a quoted identifier is data, not
    SQL, so refusing it would drop a legitimate column.
    """
    assert _unsafe_expr_reason(expr) is None


@pytest.mark.parametrize(
    'expr',
    [
        f"$$'$$ || {_EXFILTRATION}::VARCHAR || $$'$$",
        f"'\\'' || {_EXFILTRATION}::VARCHAR || '\\''",
        f"'\\' || {_EXFILTRATION} || '\\'",
        f'"\\" || {_EXFILTRATION} || "\\"',
        f'"\\"" || {_EXFILTRATION} || "b"',
        f'$$x$$ || {_EXFILTRATION}',
        f"'''' || {_EXFILTRATION} || ''''",
        f"'\\\\' || {_EXFILTRATION} || '\\\\'",
        f'`a` || {_EXFILTRATION} || `b`',
        f"'a\\' || {_EXFILTRATION} || 'b\\'",
        f'$$ {_EXFILTRATION}',
    ],
)
def test_dialect_escaping_cannot_hide_a_sub_query(expr):
    r"""Snowflake dollar-quoting and the backslash escapes both warehouses
    honour must not be able to shift where the guard believes a literal ends.
    Verified live: none of these evaluates its sub-query on Snowflake or
    Databricks once refused here. See the ``$$``/backslash guard-bypass report
    under
    """
    assert _unsafe_expr_reason(expr) is not None


@pytest.mark.parametrize(
    'expr',
    [
        'TRUNCATE(AMOUNT, 2)',
        'TRUNCATE (AMOUNT, 2)',
        'truncate(amount)',
        "INSERT(NAME, 1, 2, '**')",
        "INSERT(name, 1, 0, 'Dr. ')",
        'ORDERS.MERGE',
        'ORDERS.CREATE',
        'orders.update',
        '"ORDERS".DELETE',
        "REGEXP_REPLACE(NAME, '\\\\s+', ' ')",
        "SPLIT_PART(PATH, '\\\\', 1)",
        "CONCAT(NAME, '\\'')",
        '$$plain text$$',
        "$$it's$$",
        'AMOUNT$USD * 2',
        'T$1.COL',
    ],
)
def test_scalar_functions_and_qualified_names_are_accepted(expr):
    r"""``TRUNCATE`` and ``INSERT`` are also Snowflake scalar functions, and a
    keyword behind a ``.`` is the tail of a qualified name; neither can begin a
    statement, so refusing them only drops legitimate columns.
    """
    assert _unsafe_expr_reason(expr) is None


@pytest.mark.parametrize(
    'expr',
    [
        'INSERT INTO T VALUES (1)',
        'TRUNCATE TABLE T',
        'TRUNCATE  TABLE  T',
        'SELECT(SECRET)FROM PAYROLL',
        '(SELECT(1))',
    ],
)
def test_statements_that_look_like_calls_are_still_reported(expr):
    r"""``SELECT`` may legally be followed by ``(``, so only ``INSERT`` and
    ``TRUNCATE``, which as statements are always followed by a keyword,
    are exempt when they are called.
    """
    assert _unsafe_expr_reason(expr) is not None


# Snowflake semantic views ####################################################

snowflake_connector = pytest.importorskip('snowflake.connector')
pytest.importorskip('yaml')

SnowflakeConnection = snowflake_connector.SnowflakeConnection

_SEMANTIC_VIEW = """
tables:
  - name: USERS
    base_table:
      database: DB
      schema: PUBLIC
      table: USERS
    primary_key:
      columns:
        - USER_ID
    dimensions:
      - name: USER_ID
        expr: USER_ID
        data_type: NUMBER
      - name: NAME_UPPER
        expr: UPPER(NAME)
        data_type: VARCHAR
{extra}"""


def _leaked_dimension(expr: str = _EXFILTRATION) -> str:
    quoted = expr.replace("'", "''")
    return (
        f'      - name: LEAKED\n'
        f"        expr: '{quoted}'\n"
        f'        data_type: NUMBER\n'
    )


_LEAKED_DIMENSION = _leaked_dimension()

_SOURCE_COLUMNS = [
    SourceColumn(
        name='USER_ID',
        dtype=Dtype.int64,
        is_primary_key=True,
        is_unique_key=True,
        is_nullable=False,
    ),
    SourceColumn(
        name='NAME',
        dtype=Dtype.string,
        is_primary_key=False,
        is_unique_key=False,
        is_nullable=True,
    ),
]


class _Cursor:
    def __init__(self, connection: _Connection) -> None:
        self._connection = connection

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, sql: str, parameters: Any = None) -> None:
        self._connection.calls.append(sql)

    def fetchone(self) -> tuple:
        return (self._connection.scalar,)

    def fetchall(self) -> list:
        return []


class _Connection(SnowflakeConnection):
    def __init__(self, scalar: Any = None) -> None:
        self.calls: list[str] = []
        self.scalar = scalar
        self._paramstyle = 'pyformat'

    def cursor(self) -> _Cursor:  # type: ignore[override]
        return _Cursor(self)


def _stub_table_class(base):
    r"""A table class that answers the source-metadata probes offline, so the
    view-to-graph conversion runs without a warehouse.
    """

    class _StubTable(base):  # type: ignore[valid-type,misc]
        def _get_source_columns(self) -> list[SourceColumn]:
            return list(_SOURCE_COLUMNS)

        def _get_source_foreign_keys(self) -> list[SourceForeignKey]:
            return []

        def _get_source_sample_df(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    'USER_ID': [1, 2, 3],
                    'NAME': ['a', 'b', 'c'],
                }
            )

        def _get_expr_sample_df(self, columns) -> pd.DataFrame:
            return pd.DataFrame(
                {column.name: ['x', 'y', 'z'] for column in columns}
            )

        def _get_num_rows(self) -> int | None:
            return 3

    return _StubTable


@pytest.fixture()
def snow_graph(monkeypatch):
    import sdm_relational_engine.rfm.backend.snow as snow

    monkeypatch.setattr(snow, 'SnowTable', _stub_table_class(snow.SnowTable))

    from sdm_relational_engine.rfm import Graph

    def build(extra: str = ''):
        return Graph.from_snowflake_semantic_view(
            'MY_VIEW',
            connection=_Connection(scalar=_SEMANTIC_VIEW.format(extra=extra)),
            verbose=False,
        )

    return build


def test_semantic_view_expression_is_not_executed(snow_graph):
    with pytest.warns(UserWarning, match='LEAKED'):
        graph = snow_graph(_LEAKED_DIMENSION)

    table = graph['USERS']
    assert not table.has_column('LEAKED')
    assert [
        str(column.expr) for column in table.columns if column.expr is not None
    ] == ['UPPER(NAME)']


@pytest.mark.parametrize(
    'expr',
    [
        f"$$'$$ || {_EXFILTRATION}::VARCHAR || $$'$$",
        f"'\\'' || {_EXFILTRATION}::VARCHAR || '\\''",
    ],
)
def test_dialect_escaped_expression_is_not_executed(snow_graph, expr):
    r"""End-to-end through the public constructor: a dimension that hides its
    quotes in ``$$`` or ``\'`` is dropped rather than spliced into the sampler
    query.
    """
    with pytest.warns(UserWarning, match='LEAKED'):
        graph = snow_graph(_leaked_dimension(expr))

    table = graph['USERS']
    assert not table.has_column('LEAKED')
    assert [
        str(column.expr) for column in table.columns if column.expr is not None
    ] == ['UPPER(NAME)']


def test_semantic_view_keeps_ordinary_expression_columns(snow_graph):
    graph = snow_graph()

    assert str(graph['USERS']['NAME_UPPER'].expr) == 'UPPER(NAME)'


def test_semantic_view_keeps_a_scalar_function_column(snow_graph):
    r"""The guard must not drop a ``TRUNCATE``/``INSERT`` dimension; see"""
    extra = (
        '      - name: NAME_MASKED\n'
        '        expr: "INSERT(NAME, 1, 2, \'**\')"\n'
        '        data_type: VARCHAR\n'
        '      - name: NAME_TRUNC\n'
        "        expr: 'TRUNCATE(USER_ID, 2)'\n"
        '        data_type: NUMBER\n'
    )
    graph = snow_graph(extra)

    table = graph['USERS']
    assert table.has_column('NAME_MASKED')
    assert table.has_column('NAME_TRUNC')


def test_a_literal_holding_the_table_name_survives_conversion(snow_graph):
    r"""End to end through the public constructor."""
    graph = snow_graph("""      - name: TAGGED
        expr: CONCAT('USERS.', NAME)
        data_type: VARCHAR
""")
    assert str(graph['USERS']['TAGGED'].expr) == "CONCAT('USERS.', NAME)"
