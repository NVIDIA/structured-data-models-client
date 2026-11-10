# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native payload attachments preserve typed source values as valid JSON."""

from __future__ import annotations

import json
import warnings

import duckdb
import numpy as np
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import PredictiveQuery, Schema
from sdm_client._rust import run as _rust_run


@pytest.mark.parametrize(
    'text_keys', [False, True], ids=['integer-key', 'text-key']
)
@pytest.mark.parametrize(
    'price_type',
    ['SMALLINT', 'INTEGER', 'BIGINT', 'FLOAT', 'DOUBLE', 'DECIMAL(10,2)'],
)
def test_payload_bytes_preserve_source_types_and_labels(
    text_keys, price_type, monkeypatch
):
    taken = []
    take = _rust_run.Finished.take

    def capture(self, attachment):
        result = take(self, attachment)
        state = self.result.get('state')
        if isinstance(state, dict) and attachment == state.get('payload'):
            taken.append(bytes(result))
        return result

    monkeypatch.setattr(_rust_run.Finished, 'take', capture)
    connection = duckdb.connect()
    key_type = 'VARCHAR' if text_keys else 'BIGINT'
    key = "'u' || i::VARCHAR" if text_keys else 'i'
    connection.execute(f"""CREATE TABLE users (
        user_id {key_type} PRIMARY KEY, age INTEGER, flag BOOLEAN,
        small SMALLINT, large BIGINT, single FLOAT, double_value DOUBLE,
        text_value VARCHAR, day DATE, moment TIMESTAMP, missing DOUBLE)""")
    connection.execute(f"""INSERT INTO users SELECT {key}, 20+i, i%2=0,
        i::SMALLINT, (1000+i)::BIGINT, 0.1::FLOAT, 0.125::DOUBLE,
        'café "quoted" ' || i::VARCHAR, DATE '2023-01-02',
        TIMESTAMP '2023-01-02 03:04:05.123456', NULL FROM range(8) t(i)""")
    connection.execute(f"""CREATE TABLE orders (
        order_id BIGINT PRIMARY KEY, user_id {key_type} REFERENCES users(user_id),
        price {price_type}, created_at TIMESTAMP)""")
    order_key = "'u' || (i%8)::VARCHAR" if text_keys else 'i%8'
    connection.execute(f"""INSERT INTO orders SELECT i, {order_key}, (i%5)+0.25,
        TIMESTAMP '2023-01-01' + (i//8)*INTERVAL '15 days' FROM range(96) t(i)""")
    indices = ['u0', 'u1'] if text_keys else [0, 1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(connection))
        query = PredictiveQuery(
            'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
            schema,
        )
        request = query.materialize(
            indices=indices,
            anchor_time=pd.Timestamp('2023-07-01'),
            random_seed=42,
        )
    assert len(taken) == 1

    # Reject non-JSON numeric constants rather than accepting Python's extension.
    def invalid_constant(value):
        raise AssertionError(f'non-JSON number: {value}')

    payload = json.loads(taken[0], parse_constant=invalid_constant)
    assert payload == request._payloads[0]
    assert payload['client_generated_keys'] == {'orders': '__node_id'}
    predicted = payload['predict']['instance_table']
    assert [row[1] for row in predicted['rows']] == indices
    context = payload['context']['instance_table']
    assert context['rows']
    for row in context['rows']:
        cells = dict(zip(context['columns'], row, strict=True))
        start = pd.Timestamp(cells['__kumo_anchor_time']).tz_localize(None)
        expected = connection.execute(
            'SELECT coalesce(sum(price), 0) FROM orders WHERE user_id = ? AND created_at > ? AND created_at <= ?',
            [
                cells['__kumo_entity_ref_0'],
                start,
                start + pd.Timedelta(days=30),
            ],
        ).fetchone()[0]
        assert cells['TARGET'] == float(np.float32(expected))
    for side in ['context', 'predict']:
        users = payload[side]['related_tables']['users']
        assert users['rows']
        for row in users['rows']:
            cells = dict(zip(users['columns'], row, strict=True))
            identity = cells['user_id']
            number = int(identity[1:]) if text_keys else identity
            assert type(identity) is (str if text_keys else int)
            for name in ['age', 'small', 'large']:
                assert type(cells[name]) is int
            for name in ['single', 'double_value']:
                assert type(cells[name]) is float
            assert cells['age'] == 20 + number
            assert cells['flag'] is (number % 2 == 0)
            assert cells['small'] == number
            assert cells['large'] == 1000 + number
            assert cells['single'] == float(np.float32(0.1))
            assert cells['double_value'] == 0.125
            assert cells['text_value'] == f'café "quoted" {number}'
            assert cells['day'] == '2023-01-02T00:00:00.000000Z'
            assert pd.Timestamp(cells['moment']).tz_localize(
                None
            ) == pd.Timestamp('2023-01-02 03:04:05.123456')
            assert cells['missing'] is None
