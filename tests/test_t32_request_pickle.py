# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The client test claiming ``state.pickle/request/any``
(``docs/design/rust-core/phase-0/scope-python.yaml``'s ``tests``, read by
``scripts/rust_core/coverage.py``; T3.2, carried row P2-T2.16-3): what
pickling a materialized request does, which no contract case can ask, since
the case format has no pickle step.

A request holds the source it was materialized from, and a source holds an
open connection, which ``sdm_connectors.Source.__reduce__`` refuses to
pickle (``TypeError``), as the reference does: a request is moved by its
payload, never by pickling the object.
"""

from __future__ import annotations

import pickle
import warnings

import pytest

duckdb = pytest.importorskip('duckdb')
pd = pytest.importorskip('pandas')


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
