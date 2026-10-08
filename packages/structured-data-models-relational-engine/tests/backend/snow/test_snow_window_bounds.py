# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Window bounds on Snowflake name UTC instants for zoned time columns.

A bound is naive text, and Snowflake converts a ``TIMESTAMP_NTZ`` to a
``TIMESTAMP_LTZ`` or ``TIMESTAMP_TZ`` in the session's ``TIMEZONE``,
America/Los_Angeles by default, so a zoned column's window was shifted by that
zone's offset. Such a column now reads every bound, in the key list and
outside it, through ``TO_TIMESTAMP_LTZ`` / ``TO_TIMESTAMP_TZ`` with an
explicit format, and the bound carries ``+00:00``. A naive or ``DATE`` column
keeps today's statement exactly.
"""

from __future__ import annotations

import datetime

import pandas as pd
import pyarrow as pa
import pytest
from predictive_query.schema.typing import Dtype
from sdm_relational_engine.rfm.base import ColumnSpec

pytest.importorskip(
    'snowflake.connector',
    reason="requires the 'snowflake' extra",
)

from sdm_relational_engine.rfm.backend.snow import dialect

_FORMAT = 'YYYY-MM-DD"T"HH24:MI:SSTZH:TZM'


@pytest.mark.parametrize(
    ('spelled', 'zone'),
    [
        ('TIMESTAMP_LTZ(9)', 'LTZ'),
        ('timestamp_ltz', 'LTZ'),
        ('TIMESTAMP_TZ(9)', 'TZ'),
        ('TIMESTAMP_NTZ(9)', None),
        ('DATE', None),
        ('VARCHAR(16777216)', None),
        (None, None),
    ],
)
def test_the_zone_follows_the_declared_type(
    spelled: str | None, zone: str | None
) -> None:
    assert dialect.zone_of(spelled) == zone


def _window(zone: str | None) -> tuple[str, list]:
    return dialect.window(
        '"DB"."S"."ORDERS"',
        ['"ORDER_ID"'],
        key='"USER_ID"',
        dtype=Dtype.int,
        rows=[(1, 'a', 'b')],
        with_start=True,
        time='"AT"',
        bounds=['e', 's'],
        order_by='"AT"',
        zone=zone,
    )


def _neighbors(zone: str | None) -> tuple[str, list]:
    return dialect.neighbors(
        '"DB"."S"."ORDERS"',
        ['"ORDER_ID"'],
        key='"USER_ID"',
        dtype=Dtype.int,
        rows=[(1, 'a', 'b')],
        windowed=True,
        time='"AT"',
        bounds=['e', 's'],
        order_by='"AT"',
        num_neighbors=3,
        zone=zone,
    )


def test_a_naive_column_keeps_todays_statement() -> None:
    sql, params = _window(None)
    assert 'f.value[1]::TIMESTAMP_NTZ as __KUMO_END_TIME__' in sql
    assert 'f.value[2]::TIMESTAMP_NTZ as __KUMO_START_TIME__' in sql
    assert 'WHERE "AT" <= ?\nAND "AT" > ?\n' in sql
    assert 'TO_TIMESTAMP' not in sql
    assert params == ['[[1, "a", "b"]]', 'e', 's']
    assert 'TO_TIMESTAMP' not in _neighbors(None)[0]


@pytest.mark.parametrize('zone', ['LTZ', 'TZ'])
@pytest.mark.parametrize('statement', [_window, _neighbors])
def test_a_zoned_column_reads_every_bound_as_an_instant(
    zone: str, statement: object
) -> None:
    sql, _ = statement(zone)  # type: ignore[operator]
    function = f'TO_TIMESTAMP_{zone}'
    assert f"{function}(f.value[1]::VARCHAR, '{_FORMAT}')" in sql
    assert f"{function}(f.value[2]::VARCHAR, '{_FORMAT}')" in sql
    assert sql.count(f"{function}(?, '{_FORMAT}')") == 2
    assert 'TIMESTAMP_NTZ' not in sql
    # The key list is the one bare placeholder left.
    assert sql.replace(f"{function}(?, '{_FORMAT}')", '').count('?') == 1


def test_a_zoned_bound_is_written_as_the_format_reads_it() -> None:
    anchor = pd.Series([pd.Timestamp('2024-03-15 12:00:00')])
    written = anchor.dt.strftime(dialect.ZONED_BOUND_FORMAT)[0]
    assert written == '2024-03-15T12:00:00+00:00'
    assert datetime.datetime.fromisoformat(written) == datetime.datetime(
        2024, 3, 15, 12, tzinfo=datetime.timezone.utc
    )
    naive = anchor.dt.strftime(dialect.NAIVE_BOUND_FORMAT)[0]
    assert naive == '2024-03-15 12:00:00'


# Expression columns ##########################################################


class _ExpressionTable:
    r"""What ``SnowTable._get_expr_sample_df`` reads: one sample, as the
    connector reads ``TIMESTAMP_LTZ``, ``TIMESTAMP_TZ`` and ``TIMESTAMP_NTZ``
    expressions (the first two in the session's zone).
    """

    source_name = 'DB.S.ORDERS'

    def __init__(self) -> None:
        self._zoned_expr_columns: set[str] = {'WAS_ZONED'}

    def _read_sample(self, projections: object) -> pa.Table:
        instant = pd.Timestamp('2024-03-15 12:00:00')
        zone = 'America/Los_Angeles'
        return pa.table(
            {
                'LTZ': pa.array([instant], pa.timestamp('ns', tz=zone)),
                'TZ': pa.array([instant], pa.timestamp('ns', tz=zone)),
                'NTZ': pa.array([instant], pa.timestamp('ns')),
                'WAS_ZONED': pa.array([instant], pa.timestamp('ns')),
                'LABEL': pa.array(['a']),
            }
        )

    def _sanitize(self, df: pd.DataFrame, **_: object) -> pd.DataFrame:
        return df


def test_an_expression_read_as_a_zoned_timestamp_is_zoned() -> None:
    from sdm_relational_engine.rfm.backend.snow.table import SnowTable

    table = _ExpressionTable()
    columns = [
        ColumnSpec(name=name, expr=f'EXPR_{name}')
        for name in ('LTZ', 'TZ', 'NTZ', 'WAS_ZONED', 'LABEL')
    ]

    SnowTable._get_expr_sample_df(table, columns)  # type: ignore[arg-type]

    assert table._zoned_expr_columns == {'LTZ', 'TZ'}


class _Sampler:
    r"""What ``SnowSampler._zone`` reads: one source ``TIMESTAMP_LTZ``
    column, and the expression columns the table read as zoned.
    """

    def __init__(self) -> None:
        self.source_table_dict = {'ORDERS': {'AT': object()}}
        self._snowflake_type_dict = {'ORDERS': {'AT': 'TIMESTAMP_LTZ(9)'}}
        table = _ExpressionTable()
        table._zoned_expr_columns = {'AT_EXPR'}
        self._table_dict = {'ORDERS': table}


@pytest.mark.parametrize(
    ('column', 'zone'),
    [
        ('AT', 'LTZ'),
        # An expression's zone type isn't known; TIMESTAMP_TZ bounds compare
        # with either as instants.
        ('AT_EXPR', 'TZ'),
        # Naive, or never sampled because it was given its types.
        ('NAIVE_EXPR', None),
    ],
)
def test_an_expression_column_takes_the_zone_its_sample_was_read_with(
    column: str, zone: str | None
) -> None:
    from sdm_relational_engine.rfm.backend.snow.sampler import SnowSampler

    assert SnowSampler._zone(_Sampler(), 'ORDERS', column) == zone  # type: ignore[arg-type]
