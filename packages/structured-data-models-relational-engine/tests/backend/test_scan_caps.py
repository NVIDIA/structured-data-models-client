# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The inference sample of a large table reads less than all of it.

The sample takes the 1,000 rows with the lowest hash, which reads every
block. On a large table each warehouse reads part of it first:
PostgreSQL and Snowflake a repeatable sample of its blocks, a power-of-two
fraction of it, and Databricks a bounded subset of its rows. A small table is
read as before.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pyarrow as pa
import pytest
from sdm_relational_engine.rfm.base.utils import (
    percent_literal,
    prefilter_fraction,
)

TIB = 1 << 40
GIB = 1 << 30
MIB = 1 << 20


@pytest.mark.parametrize(
    ('wanted', 'fraction'),
    [
        (Fraction(2048, 16384), Fraction(1, 8)),
        (Fraction(2048, 16385), Fraction(1, 8)),
        (Fraction(2048, 32767), Fraction(1, 8)),
        (Fraction(2048, 32768), Fraction(1, 16)),
        (Fraction(2048, 142720), Fraction(1, 64)),
        (Fraction(1, 7), None),
        (Fraction(1, 2), None),
        (Fraction(0), None),
    ],
)
def test_the_fraction_is_the_power_of_two_at_or_above(
    wanted: Fraction, fraction: Fraction | None
) -> None:
    assert prefilter_fraction(wanted) == fraction


def test_the_literal_is_exact() -> None:
    assert percent_literal(Fraction(1, 8)) == '12.5'
    assert percent_literal(Fraction(1, 64)) == '1.5625'
    assert percent_literal(Fraction(1, 512)) == '0.1953125'


# Snowflake ###################################################################


class _Listed:
    r"""What :meth:`SnowTable._prefilter` reads: the table's listing."""

    _SAMPLE_PREFILTER_PARTITIONS = 128
    _MICRO_PARTITION_BYTES = 16 << 20
    _SAMPLED_KINDS = frozenset({'TABLE', 'TRANSIENT', 'TEMPORARY'})

    def __init__(self, listing: dict[str, Any] | None) -> None:
        self._listing = listing


def _listing(rows: int, bytes_: int, **flags: str) -> dict[str, Any]:
    return {
        'rows': rows,
        'bytes': bytes_,
        'kind': 'TABLE',
        'is_external': 'N',
        'is_hybrid': 'N',
        'is_iceberg': 'N',
        'is_dynamic': 'N',
        **flags,
    }


@pytest.mark.parametrize(
    ('listing', 'clause'),
    [
        (_listing(10_000_000, TIB), 'SAMPLE SYSTEM (1.5625) SEED (0)'),
        (_listing(800_000, TIB), 'SAMPLE SYSTEM (12.5) SEED (0)'),
        (_listing(799_999, TIB), None),
        (_listing(10**9, 4 * GIB), None),
        (_listing(10**9, 64 * GIB), 'SAMPLE SYSTEM (3.125) SEED (0)'),
        (_listing(60, MIB), None),
        (_listing(10**9, TIB, kind='TRANSIENT'), 'SAMPLE SYSTEM (0.1953125) '),
        (_listing(10**9, TIB, is_hybrid='Y'), None),
        (_listing(10**9, TIB, is_external='Y'), None),
        (_listing(10**9, TIB, is_iceberg='Y'), None),
        (_listing(10**9, TIB, is_dynamic='Y'), None),
        (_listing(10**9, TIB, kind='VIEW'), None),
        (None, None),
    ],
)
def test_a_large_standard_snowflake_table_samples_its_partitions(
    listing: dict[str, Any] | None, clause: str | None
) -> None:
    pytest.importorskip('snowflake.connector')
    from sdm_relational_engine.rfm.backend.snow import SnowTable

    got = SnowTable._prefilter(_Listed(listing), 100_000, seed=0)  # type: ignore[arg-type]
    if clause is None:
        assert got is None
    else:
        assert got is not None and got.startswith(clause.strip())


def test_a_seeded_draw_keeps_its_seed_below_two_to_the_31() -> None:
    pytest.importorskip('snowflake.connector')
    from sdm_relational_engine.rfm.backend.snow import SnowTable

    got = SnowTable._prefilter(
        _Listed(_listing(10**9, TIB)),  # type: ignore[arg-type]
        10_000_000,
        seed=(1 << 31) + 5,
    )
    assert got == 'SAMPLE SYSTEM (1.5625) SEED (5)'


# Databricks ##################################################################


class _Described:
    r"""What :meth:`DatabricksTable._is_large` reads, answered from a map of
    statement to result, or to the error it raises.
    """

    _SAMPLE_BOUND_MIN_BYTES = 1 << 30
    _catalog = 'main'
    _schema = 'shop'
    _source_name = 'orders'

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.asked: list[str] = []

    def _run(self, statement: tuple[str, Any]) -> pa.Table:
        sql, _ = statement
        self.asked.append(sql)
        answer = self.answers.get(sql.split(' `', 1)[0])
        if answer is None or isinstance(answer, Exception):
            raise answer or RuntimeError(f'unsupported: {sql}')
        return answer

    def _rows(self, statement: tuple[str, Any]) -> list[tuple[Any, ...]]:
        table = self._run(statement)
        return list(zip(*[c.to_pylist() for c in table.columns], strict=True))


def _detail(size: int | None) -> pa.Table:
    return pa.table(
        {
            'format': ['delta'],
            'sizeInBytes': pa.array([size], pa.int64()),
        }
    )


def _extended(**rows: str) -> pa.Table:
    return pa.table(
        {
            'col_name': ['id', '', *rows],
            'data_type': ['bigint', '', *rows.values()],
            'comment': [None] * (2 + len(rows)),
        }
    )


@pytest.mark.parametrize(
    ('answers', 'large'),
    [
        ({'DESCRIBE DETAIL': _detail(10 * TIB)}, True),
        ({'DESCRIBE DETAIL': _detail(GIB)}, True),
        ({'DESCRIBE DETAIL': _detail(GIB - 1)}, False),
        (
            {
                'DESCRIBE DETAIL': RuntimeError('[EXPECT_TABLE_NOT_VIEW]'),
                'DESCRIBE TABLE EXTENDED': _extended(Type='VIEW'),
            },
            True,
        ),
        (
            {
                'DESCRIBE DETAIL': _detail(None),
                'DESCRIBE TABLE EXTENDED': _extended(
                    Type='EXTERNAL', Provider='parquet'
                ),
            },
            False,
        ),
        (
            {
                'DESCRIBE DETAIL': _detail(None),
                'DESCRIBE TABLE EXTENDED': _extended(
                    Type='EXTERNAL', Statistics=f'{2 * GIB} bytes, 7 rows'
                ),
            },
            True,
        ),
        ({}, False),
    ],
)
def test_databricks_sizes_a_table_from_its_metadata(
    answers: dict[str, Any], large: bool
) -> None:
    pytest.importorskip('databricks.sql')
    from sdm_relational_engine.rfm.backend.databricks import DatabricksTable

    table = _Described(answers)
    assert DatabricksTable._is_large(table) is large  # type: ignore[arg-type]
    assert table.asked[0] == 'DESCRIBE DETAIL `main`.`shop`.`orders`'


def test_a_bounded_subset_is_todays_statement_over_a_limit() -> None:
    from sdm_relational_engine.rfm.backend.databricks import dialect

    sql, params = dialect.sample(
        dialect.bounded('`main`.`shop`.`orders`', 100_000),
        ['`amount`'],
        ['xxhash64(`order_id`)'],
        1000,
    )
    assert params is None
    assert sql == (
        'SELECT xxhash64(`order_id`) AS __kumo_order_0__, `amount` '
        'FROM (SELECT * FROM `main`.`shop`.`orders` LIMIT 100000) '
        'ORDER BY 1 LIMIT 1000'
    )
