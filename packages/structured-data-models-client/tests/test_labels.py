# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Known answers calculated directly from the shared tiny historical dataset."""

import json
from pathlib import Path

import pandas as pd
import pytest

from sdm_client import PredictiveQuery, Schema
from sdm_client.errors import StructuredDataError

DATA = json.loads(
    (
        Path(__file__).resolve().parents[3]
        / 'tests/fixtures/labels/ground-truth.json'
    ).read_text()
)


def schema():
    tables = {
        name: pd.DataFrame(DATA[name]) for name in ('users', 'items', 'orders')
    }
    tables['orders']['time'] = pd.to_datetime(tables['orders']['time'])
    result = Schema.from_data(
        tables,
        primary_keys={
            'users': 'user_id',
            'items': 'item_id',
            'orders': 'order_id',
        },
        time_columns={'orders': 'time'},
    )
    result['users']['value'].stype = 'numerical'
    result['users']['segment'].stype = 'categorical'
    return result


@pytest.mark.parametrize('case', DATA['cases'], ids=lambda case: case['name'])
def test_shared_ground_truth(case):
    query = PredictiveQuery(
        f'PREDICT {case["target"]} FOR EACH users.user_id', schema()
    )
    frame = query.labels(100, anchor_time=pd.Timestamp(DATA['anchor']))
    assert list(frame.columns) == ['ENTITY', 'ANCHOR_TIMESTAMP', 'TARGET']
    actual = {
        str(row.ENTITY): row.TARGET.tolist()
        if hasattr(row.TARGET, 'tolist')
        else row.TARGET
        for row in frame.itertuples()
    }
    assert actual == case['expected']
    assert (frame.ANCHOR_TIMESTAMP == pd.Timestamp(DATA['anchor'])).all()
    pd.testing.assert_frame_equal(
        frame, query.labels(100, anchor_time=pd.Timestamp(DATA['anchor']))
    )


def test_forecast_is_one_entity_at_successive_anchors():
    expected = DATA['forecast']
    query = PredictiveQuery(expected['query'], schema())
    frame = query.labels(3, anchor_time=pd.Timestamp(DATA['anchor']))
    assert frame.ENTITY.tolist() == expected['entity']
    assert frame.TARGET.tolist() == expected['target']
    assert (
        frame.ANCHOR_TIMESTAMP.tolist()
        == pd.to_datetime(expected['anchors']).tolist()
    )
    assert len(query.labels(1, anchor_time=pd.Timestamp(DATA['anchor']))) == 1


@pytest.mark.parametrize(
    'size,iterations', [(-1, 10), (True, 10), (1, 0), (1, -1), (1, False)]
)
def test_invalid_controls(size, iterations):
    with pytest.raises(StructuredDataError):
        PredictiveQuery(
            'PREDICT users.value FOR EACH users.user_id', schema()
        ).labels(size, max_iterations=iterations)


def test_zero_and_no_qualifying_labels_are_empty():
    data = schema()
    query = PredictiveQuery('PREDICT users.value FOR EACH users.user_id', data)
    empty = query.labels(0)
    assert list(empty.columns) == ['ENTITY', 'ANCHOR_TIMESTAMP', 'TARGET']
    assert empty.empty
    filtered = PredictiveQuery(
        'PREDICT users.value FOR EACH users.user_id WHERE users.value > 100',
        data,
    ).labels(100)
    assert filtered.empty
    assert filtered.TARGET.dtype == empty.TARGET.dtype


def test_size_and_seed_bound_selection():
    query = PredictiveQuery(
        'PREDICT users.value FOR EACH users.user_id', schema()
    )
    first = query.labels(2, random_seed=7)
    assert len(first) == 2
    assert set(first.ENTITY) <= {1, 2, 3}
    pd.testing.assert_frame_equal(first, query.labels(2, random_seed=7))


@pytest.mark.parametrize('connector', ['sqlite', 'duckdb'])
def test_provider_ground_truth(connector):
    import sqlite3

    import duckdb
    from sdm_connectors import DuckDBSource, SQLiteSource

    connection = (
        duckdb.connect()
        if connector == 'duckdb'
        else sqlite3.connect(':memory:')
    )
    try:
        for name, declarations in (
            ('users', 'user_id BIGINT, value BIGINT, segment VARCHAR'),
            ('items', 'item_id BIGINT'),
            (
                'orders',
                'order_id BIGINT, user_id BIGINT, item_id BIGINT, price DOUBLE, time '
                + ('TIMESTAMP' if connector == 'duckdb' else 'TEXT'),
            ),
        ):
            columns = DATA[name]
            connection.execute(f'CREATE TABLE {name} ({declarations})')
            connection.executemany(
                f'INSERT INTO {name} VALUES ({", ".join("?" for _ in columns)})',
                list(zip(*columns.values(), strict=True)),
            )
        source = (
            DuckDBSource(connection)
            if connector == 'duckdb'
            else SQLiteSource(connection)
        )
        data = Schema.from_source(source)
        for name, key in [
            ('users', 'user_id'),
            ('items', 'item_id'),
            ('orders', 'order_id'),
        ]:
            data[name].primary_key = key
        data['orders'].time_column = 'time'
        data['users']['value'].stype = 'numerical'
        data['users']['segment'].stype = 'categorical'
        from sdm_client import Relationship

        data.relationships = [
            Relationship('orders', 'user_id', 'users', 'user_id'),
            Relationship('orders', 'item_id', 'items', 'item_id'),
        ]
        for case in DATA['cases']:
            output = PredictiveQuery(
                f'PREDICT {case["target"]} FOR EACH users.user_id', data
            ).labels(100, anchor_time=pd.Timestamp(DATA['anchor']))
            actual = {
                str(row.ENTITY): row.TARGET.tolist()
                if hasattr(row.TARGET, 'tolist')
                else row.TARGET
                for row in output.itertuples()
            }
            assert actual == case['expected']
        forecast = PredictiveQuery(DATA['forecast']['query'], data).labels(
            3, anchor_time=pd.Timestamp(DATA['anchor'])
        )
        assert forecast.TARGET.tolist() == DATA['forecast']['target']
    finally:
        connection.close()


def test_entity_time_anchors_are_static_only():
    data = schema()
    data['users'].data['created'] = pd.to_datetime(
        ['2024-01-01', '2024-01-02', '2024-01-03']
    )
    # Admission observes the added column before declaring it as the table time.
    tables = {name: table.data for name, table in data.tables.items()}
    data = Schema.from_data(
        tables,
        primary_keys={
            'users': 'user_id',
            'items': 'item_id',
            'orders': 'order_id',
        },
        time_columns={'users': 'created', 'orders': 'time'},
    )
    data['users']['value'].stype = 'numerical'
    output = PredictiveQuery(
        'PREDICT users.value FOR EACH users.user_id', data
    ).labels(10, anchor_time='entity')
    expected = {
        1: pd.Timestamp('2024-01-01'),
        2: pd.Timestamp('2024-01-02'),
        3: pd.Timestamp('2024-01-03'),
    }
    assert (
        dict(zip(output.ENTITY, output.ANCHOR_TIMESTAMP, strict=True))
        == expected
    )
    with pytest.raises(StructuredDataError, match='static'):
        PredictiveQuery(
            'PREDICT COUNT(orders.*, 0, 1, days) FOR EACH users.user_id', data
        ).labels(10, anchor_time='entity')


def test_null_targets_are_omitted_and_empty_link_schema_is_preserved():
    data = schema()
    data['users'].data['value'] = pd.array([10, None, 30], dtype='Int64')
    output = PredictiveQuery(
        'PREDICT users.value FOR EACH users.user_id', data
    ).labels(10)
    assert set(output.ENTITY) == {1, 3}
    empty = PredictiveQuery(
        'PREDICT LIST_DISTINCT(orders.item_id, 0, 2, days) FOR EACH users.user_id',
        data,
    ).labels(0)
    assert empty.empty
    assert list(empty.columns) == ['ENTITY', 'ANCHOR_TIMESTAMP', 'TARGET']


def test_default_forecast_anchor_uses_full_horizon_even_when_truncated():
    query = PredictiveQuery(DATA['forecast']['query'], schema())
    # Latest historical event is Jan 5; full three-day horizon starts Jan 2.
    output = query.labels(1)
    assert output.ENTITY.tolist() == [1]
    assert output.ANCHOR_TIMESTAMP.tolist() == [pd.Timestamp('2024-01-02')]
    assert output.TARGET.tolist() == [3.0]


def test_decimal_class_identity_and_value_dependent_empty_dtype():
    from decimal import Decimal

    import pyarrow as pa

    values = pd.Series(
        pa.array(
            [Decimal('16777217.01'), Decimal('16777217.02')],
            type=pa.decimal128(12, 2),
        ),
        dtype=pd.ArrowDtype(pa.decimal128(12, 2)),
    )
    data = Schema.from_data(
        {'users': pd.DataFrame({'user_id': [1, 2], 'category': values})},
        primary_keys={'users': 'user_id'},
    )
    data['users']['category'].stype = 'categorical'
    query = PredictiveQuery(
        'PREDICT users.category FOR EACH users.user_id', data
    )
    output = query.labels(2)
    assert dict(zip(output.ENTITY, output.TARGET, strict=True)) == {
        1: '16777217.01',
        2: '16777217.02',
    }
    # Empty conversion has no observed class collision and does not read rows.
    empty = query.labels(0)
    assert empty.empty
    assert str(empty.TARGET.dtype) == 'float32'


def test_decimal_classes_without_collisions_keep_numeric_targets():
    from decimal import Decimal

    import numpy as np
    import pyarrow as pa

    dtype = pa.decimal128(8, 2)
    values = pd.Series(
        pa.array([Decimal('1.01'), Decimal('2.02')], type=dtype),
        dtype=pd.ArrowDtype(dtype),
    )
    data = Schema.from_data(
        {'users': pd.DataFrame({'user_id': [1, 2], 'category': values})},
        primary_keys={'users': 'user_id'},
    )
    data['users']['category'].stype = 'categorical'
    output = (
        PredictiveQuery('PREDICT users.category FOR EACH users.user_id', data)
        .labels(2)
        .sort_values('ENTITY')
    )
    assert output.TARGET.dtype == np.dtype('float32')
    assert (
        output.TARGET.tolist()
        == np.array([1.01, 2.02], dtype=np.float32).tolist()
    )
