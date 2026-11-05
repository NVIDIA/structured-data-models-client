# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A materialized warehouse request cannot pickle its open source connection."""

from __future__ import annotations

import pickle
import warnings

import duckdb
import pandas as pd
import pytest


def test_a_materialized_request_refuses_pickle_by_its_source() -> None:
    from sdm_client import PredictiveQuery, Schema
    from sdm_connectors import DuckDBSource

    connection = duckdb.connect()
    connection.execute(
        'CREATE TABLE users AS SELECT i::BIGINT AS user_id FROM range(8) t(i)'
    )
    connection.execute(
        'CREATE TABLE orders AS SELECT i::BIGINT AS order_id, '
        '(i % 8)::BIGINT AS user_id, '
        "TIMESTAMP '2024-01-01' + to_days(i::INTEGER) AS ts FROM range(80) t(i)"
    )
    schema = Schema.from_source(DuckDBSource(connection))
    with warnings.catch_warnings():
        # The sampler's advice on indices, which this test does not read.
        warnings.simplefilter('ignore')
        request = PredictiveQuery(
            'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id',
            schema,
        ).materialize(indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01'))
    with pytest.raises(TypeError, match='cannot be pickled'):
        pickle.dumps(request)
