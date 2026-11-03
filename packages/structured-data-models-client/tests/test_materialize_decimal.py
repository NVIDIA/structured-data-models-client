# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact decimal aggregation through public warehouse materialization."""

from __future__ import annotations

import json
import warnings
from fractions import Fraction
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import PredictiveQuery, Schema

ANCHOR = pd.Timestamp('2024-03-01')

DECIMAL_AVERAGE = (
    'CREATE TABLE users (user_id BIGINT PRIMARY KEY)',
    'INSERT INTO users SELECT i FROM range(4) t(i)',
    'CREATE TABLE orders (order_id BIGINT PRIMARY KEY, user_id BIGINT, price DECIMAL(38, 18), ts TIMESTAMP)',
    "INSERT INTO orders VALUES (1, 1, 60000000000000000000, '2024-02-15'), (2, 1, 60000000000000000000, '2024-02-15'), (3, 2, 1, '2024-01-01')",
)


def _float32_once(exact: Fraction) -> float:
    """The float32 nearest ``exact``, ties to even: one rounding, never
    through a float64 first."""
    first = np.float32(float(exact))
    candidates = {
        first,
        np.nextafter(first, np.float32(np.inf)),
        np.nextafter(first, np.float32(-np.inf)),
    }

    def key(c: Any) -> tuple[Fraction, int]:
        bits = int(np.array(c, dtype=np.float32).view(np.uint32))
        return (abs(Fraction(float(c)) - exact), bits & 1)

    return float(min(candidates, key=key))


def _decimal_average(query: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        connection = duckdb.connect()
        for statement in DECIMAL_AVERAGE:
            connection.execute(statement)
        schema = Schema.from_source(DuckDBSource(connection))
        return PredictiveQuery(query, schema).materialize(
            indices=[1], anchor_time=ANCHOR
        )


def test_a_decimal_average_past_38_digits_is_not_the_caller_s_error() -> None:
    average = 'PREDICT AVG(orders.price, 0, 1, months) FOR EACH users.user_id'
    total = 'PREDICT SUM(orders.price, 0, 1, months) FOR EACH users.user_id'

    def rows_of(query: str) -> list[list[Any]]:
        request = _decimal_average(query)
        rows = request._payloads[0]['context']['instance_table']['rows']
        return json.loads(json.dumps(rows))

    six = 60000000000000000000
    assert rows_of(average) == [
        [0, 1, '2024-02-01T00:00:00.000000Z', _float32_once(Fraction(six))]
    ]
    summed = rows_of(total)
    labels = {(row[1], row[2]): row[3] for row in summed}
    assert labels.pop((1, '2024-02-01T00:00:00.000000Z')) == _float32_once(
        Fraction(2 * six)
    )
    assert labels and set(labels.values()) == {0.0}, labels


# Recorded decimal-sum rows from reference b413e9a (pandas 3.0.6,
# pyarrow 25.0.1), using the public materialize API. Unlike the average
# oracle above, these expected sums and entity/anchor orders are historical
# recordings, preserved in the external frozen-client archive.
SIX = '60000000000000000000'

DECIMAL_SUMS = {
    'running-past-38-digits': (
        'DECIMAL(38, 18)',
        [SIX, SIX, '-' + SIX],
        6.000000208206194e19,
    ),
    'back-to-zero': ('DECIMAL(38, 18)', [SIX, SIX, '-' + SIX, '-' + SIX], 0.0),
    'another-order': (
        'DECIMAL(38, 18)',
        [SIX, '-' + SIX, SIX],
        6.000000208206194e19,
    ),
    'scale-0': (
        'DECIMAL(38, 0)',
        ['6' + '0' * 37] * 2 + ['-6' + '0' * 37],
        6.000000213819608e37,
    ),
    'running-past-i128': (
        'DECIMAL(38, 0)',
        ['9' + '0' * 37] * 2 + ['-9' + '0' * 37],
        8.999999813669171e37,
    ),
}

DECIMAL_SUM_EXAMPLES = [
    (0, 0, '2024-02-01'),
    (1, 1, '2024-02-01'),
    (2, 2, '2024-02-01'),
    (3, 3, '2024-02-01'),
    (4, 0, '2024-01-01'),
    (5, 1, '2024-01-01'),
    (6, 3, '2024-01-01'),
    (7, 2, '2024-01-01'),
]


def _decimal_sum_rows(label: float) -> list[list[Any]]:
    return [
        [
            i,
            entity,
            f'{day}T00:00:00.000000Z',
            label if (entity, day) == (1, '2024-02-01') else 0.0,
        ]
        for i, entity, day in DECIMAL_SUM_EXAMPLES
    ]


def _decimal_sum(dtype: str, prices: list[str]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        connection = duckdb.connect()
        connection.execute('CREATE TABLE users (user_id BIGINT PRIMARY KEY)')
        connection.execute('INSERT INTO users SELECT i FROM range(4) t(i)')
        connection.execute(
            f'CREATE TABLE orders (order_id BIGINT PRIMARY KEY, user_id BIGINT, price {dtype}, ts TIMESTAMP)'
        )
        connection.execute("INSERT INTO orders VALUES (0, 2, 1, '2024-01-01')")
        for i, price in enumerate(prices):
            connection.execute(
                f"INSERT INTO orders VALUES ({i + 1}, 1, {price}, TIMESTAMP '2024-02-10' + to_days({i}))"
            )
        schema = Schema.from_source(DuckDBSource(connection))
        return PredictiveQuery(
            'PREDICT SUM(orders.price, 0, 1, months) FOR EACH users.user_id',
            schema,
        ).materialize(indices=[1], anchor_time=ANCHOR)


@pytest.mark.parametrize('name', sorted(DECIMAL_SUMS))
def test_a_decimal_sum_s_total_alone_is_checked(name: str) -> None:
    dtype, prices, label = DECIMAL_SUMS[name]
    request = _decimal_sum(dtype, prices)
    rows = request._payloads[0]['context']['instance_table']['rows']
    assert rows == _decimal_sum_rows(label)
