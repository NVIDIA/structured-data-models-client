# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unsupported keys and unreadable times are refused without invented facts."""

import sqlite3

import pandas as pd
import pyarrow as pa
import pytest
from sdm_connectors import SQLiteSource

from sdm_client import PredictiveQuery, Schema
from sdm_client.errors import StructuredDataError


@pytest.mark.parametrize(
    'dtype',
    [
        pd.ArrowDtype(pa.duration('ms')),
        pd.ArrowDtype(pa.duration('us')),
        pd.ArrowDtype(pa.time32('ms')),
        pd.ArrowDtype(pa.time64('us')),
        'timedelta64[ns]',
    ],
)
def test_unsupported_temporal_keys_are_not_reported_as_duplicates(dtype):
    if isinstance(dtype, pd.ArrowDtype):
        keys = pd.Series(
            pa.array([1, 2, 3, 4], type=dtype.pyarrow_dtype), dtype=dtype
        )
    else:
        keys = pd.Series([1, 2, 3, 4], dtype=dtype)
    frame = pd.DataFrame({'product_id': keys, 'category': ['a', 'b', 'a', 'b']})
    with pytest.raises(StructuredDataError) as caught:
        Schema.from_data(
            {'products': frame}, primary_keys={'products': 'product_id'}
        )
    assert caught.value.code == 'UNSUPPORTED_REQUEST'
    message = str(caught.value)
    assert 'products.product_id' in message
    assert str(dtype) in message
    assert 'Cast' in message
    assert 'repeat a value' not in message


@pytest.mark.parametrize('valid', [False, True, 'mixed'])
def test_unreadable_time_ranges_do_not_invent_observed_dates(tmp_path, valid):
    dbapi = pytest.importorskip('adbc_driver_sqlite.dbapi')
    users = pd.DataFrame(
        {'user_id': [1, 2, 3, 4], 'country': ['a', 'b', 'a', 'b']}
    )
    if valid is True:
        values = ['2024-01-01 00:00:00'] * 8
    elif valid == 'mixed':
        values = [
            None,
            'unreadable',
            '2024-01-01 00:00:00',
            '2024-01-02 00:00:00',
        ] * 2
    else:
        values = ['one', 'two'] * 4
    orders = pd.DataFrame(
        {
            'order_id': list(range(1, 9)),
            'user_id': [1, 2, 3, 4] * 2,
            'ordered_at': values,
        }
    )
    path = tmp_path / 'shop.db'
    with sqlite3.connect(path) as written:
        users.to_sql('users', written, index=False)
        orders.to_sql('orders', written, index=False)
    connection = dbapi.connect(str(path))
    try:
        schema = Schema.from_source(
            SQLiteSource(connection),
            primary_keys={'users': 'user_id', 'orders': 'order_id'},
            time_columns={'orders': 'ordered_at'},
        )
        query = PredictiveQuery(
            'PREDICT users.country FOR EACH users.user_id', schema
        )
        if valid:
            request = query.materialize(indices=[1, 2], random_seed=7)
            assert request.payloads
        else:
            with pytest.raises(StructuredDataError) as caught:
                query.materialize(indices=[1, 2], random_seed=7)
            assert caught.value.code == 'INVALID_REQUEST'
            message = str(caught.value)
            assert 'No readable timestamps' in message
            assert 'ordered_at' in message
            assert '2262' not in message and '1677' not in message
    finally:
        connection.close()
