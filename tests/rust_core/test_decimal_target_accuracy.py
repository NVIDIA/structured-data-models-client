# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal, localcontext

import pandas as pd
import pyarrow as pa
import pytest
from sdm_client import PredictiveQuery, Schema, StructuredDataError
from sdm_connectors import DuckDBSource

duckdb = pytest.importorskip('duckdb')
pytest.importorskip('sdm_core._native')


def materialize_target(values, precision, scale, *, aggregate=False):
    connection = duckdb.connect()
    try:
        if aggregate:
            users = pd.DataFrame({'user_id': range(1, 7)})
            source_values = values * 70
            orders = pd.DataFrame(
                {
                    'order_id': range(len(source_values) * 6),
                    'user_id': [
                        user for user in range(1, 7) for _ in source_values
                    ],
                    'ordered_at': [
                        pd.Timestamp('2024-01-01') + pd.Timedelta(days=day)
                        for day in range(70)
                        for _ in values
                    ]
                    * 6,
                    'amount': pd.Series(
                        [
                            None if value is None else value * user
                            for user in range(1, 7)
                            for value in source_values
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(precision, scale)),
                    ),
                }
            )
            for name, frame in [('users', users), ('orders', orders)]:
                connection.register('input_frame', frame)
                connection.execute(
                    f'CREATE TABLE {name} AS SELECT * FROM input_frame'
                )
                connection.unregister('input_frame')
            schema = Schema.from_source(
                DuckDBSource(connection), time_columns={'orders': 'ordered_at'}
            )
            query = (
                'PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.user_id'
            )
            options = {
                'anchor_time': pd.Timestamp('2024-03-01'),
                'num_context_samples': 12,
            }
        else:
            frame = pd.DataFrame(
                {
                    'id': range(1, len(values) + 1),
                    'label': pd.Series(
                        values,
                        dtype=pd.ArrowDtype(pa.decimal128(precision, scale)),
                    ),
                }
            )
            connection.register('input_frame', frame)
            connection.execute(
                'CREATE TABLE items AS SELECT * FROM input_frame'
            )
            schema = Schema.from_source(DuckDBSource(connection))
            schema['items'].primary_key = 'id'
            query = 'PREDICT items.label FOR EACH items.id'
            options = {}
        request = PredictiveQuery(query, schema).materialize(
            indices=[6], random_seed=7, **options
        )
        return (request, orders) if aggregate else request
    finally:
        connection.close()


@pytest.mark.parametrize(
    'texts,precision,scale',
    [
        (['100', '200', None], 38, 0),
        (['1.25', '-0.25', '0.01', None], 18, 2),
        (
            [
                '10000000000000000000000000000',
                '1',
                '-10000000000000000000000000000',
            ],
            38,
            0,
        ),
    ],
)
def test_arrow_decimal_sum_matches_exact_values_before_float32(
    texts, precision, scale
):
    values = [None if value is None else Decimal(value) for value in texts]
    request, orders = materialize_target(
        values, precision, scale, aggregate=True
    )
    for payload in request.payloads:
        table = payload['context']['instance_table']
        target = table['columns'].index('TARGET')
        entity = table['columns'].index('__kumo_entity_ref_0')
        anchor = table['columns'].index('__kumo_anchor_time')
        assert table['rows']
        for row in table['rows']:
            time = pd.Timestamp(row[anchor]).tz_localize(None)
            selected = orders.loc[
                (orders.user_id == int(row[entity]))
                & (orders.ordered_at > time)
                & (orders.ordered_at <= time + pd.Timedelta(days=30)),
                'amount',
            ]
            with localcontext() as context:
                context.prec = 100
                exact = sum(
                    (value for value in selected if not pd.isna(value)),
                    Decimal(0),
                )
            assert row[target] == pytest.approx(float(exact), rel=1e-7)


def test_distinct_decimal_classes_cannot_collapse_in_numeric_wire_values():
    values = list(
        map(
            Decimal,
            [
                '0.100000000000000001',
                '0.100000000000000002',
                '-0.1',
                '1',
                '2',
                '0',
            ],
        )
    )
    with pytest.raises(
        StructuredDataError, match='Distinct decimal target classes'
    ) as refused:
        materialize_target(values, 38, 18)
    assert refused.value.code == 'INVALID_REQUEST'


@pytest.mark.parametrize(
    'texts,precision,scale',
    [
        (['100', '200', '300', '400', None, '0'], 38, 0),
        (
            [
                '9007199254740992',
                '9007199254740993',
                '-9007199254740993',
                '1',
                None,
                '0',
            ],
            38,
            0,
        ),
        (
            [
                '9223372036854775808',
                '9223372036854775809',
                '-9223372036854775809',
                '1',
                None,
                '0',
            ],
            38,
            0,
        ),
        (['1.00', '1.00', '2.00', '3.00', None, '0.00'], 18, 2),
    ],
)
def test_whole_decimal_targets_remain_exact(texts, precision, scale):
    values = [None if value is None else Decimal(value) for value in texts]
    request = materialize_target(values, precision, scale)
    for payload in request.payloads:
        table = payload['context']['instance_table']
        target = table['columns'].index('TARGET')
        entity = table['columns'].index('__kumo_entity_ref_0')
        assert table['rows']
        for row in table['rows']:
            assert Decimal(str(row[target])) == values[int(row[entity]) - 1]
