# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import numpy as np
import pandas as pd
import pytest
from predictive_query.ast import (
    Aggregation,
    Column,
    DateOffsetRange,
)
from predictive_query.query import ValidatedPredictiveQuery
from predictive_query.schema.typing import AggregationType
from sdm_relational_engine.rfm import Graph, KumoRelational

try:
    from sdm_relational_engine.rfm.backend.databricks import DatabricksSampler
except ImportError:
    pytest.skip("'databricks' extension not installed", allow_module_level=True)


def test_sample_subgraph(graph: Graph) -> None:
    model = KumoRelational(graph)
    sampler = cast(DatabricksSampler, model._sampler)

    anchor = sampler.get_max_time(['order_lines'])
    entity = sampler._sample_entity_table(
        'customers', {'customer_id', 'segment'}, num_rows=5, random_seed=42
    )
    pkey = entity['customer_id'].reset_index(drop=True)

    subgraph = sampler.sample_subgraph(
        entity_table_names=('customers',),
        entity_pkey=pkey,
        anchor_time=pd.Series([anchor] * len(pkey)),
        num_neighbors=[16, 8],
    )

    assert 'customers' in subgraph.tables
    assert 'order_lines' in subgraph.tables
    assert subgraph.tables['customers'].df.shape[0] == len(pkey)
    assert ('order_lines', 'customer_id', 'customers') in subgraph.links


def test_sample_target(graph: Graph) -> None:
    model = KumoRelational(graph)
    sampler = cast(DatabricksSampler, model._sampler)

    query = ValidatedPredictiveQuery(
        target_ast=Aggregation(
            aggr=AggregationType.COUNT,
            target=Column(fqn='order_lines.*'),
            aggr_time_range=DateOffsetRange(0, 30),
        ),
        entity_ast=Column(fqn='customers.customer_id'),
    )

    train, test = sampler.sample_target(
        query=query,
        num_train_examples=20,
        train_anchor_time=cast(pd.Timestamp, pd.Timestamp('2026-04-01')),
        num_train_trials=200,
        num_test_examples=10,
        test_anchor_time=cast(pd.Timestamp, pd.Timestamp('2026-05-01')),
        num_test_trials=100,
        random_seed=42,
    )

    assert len(train.target) == 20
    assert len(test.target) == 10
    # `COUNT` targets are non-negative integers:
    assert (train.target >= 0).all()
    assert (train.target == train.target.round()).all()


def test_by_fkey_count_parity(graph: Graph) -> None:
    r"""Unbounded foreign-key sampling must return exactly the per-entity row
    counts reported by a direct ``GROUP BY``.
    """
    model = KumoRelational(graph)
    sampler = cast(DatabricksSampler, model._sampler)

    entity = sampler._sample_entity_table(
        'customers', {'customer_id'}, num_rows=15, random_seed=7
    )
    pkey = entity['customer_id'].reset_index(drop=True)

    _, batch = sampler._by_fkey(
        table_name='order_lines',
        foreign_key='customer_id',
        index=pkey,
        num_neighbors=10**9,
        anchor_time=None,
        columns={'customer_id', 'order_id'},
    )
    counts = pd.Series(batch).value_counts().to_dict()

    source_name = sampler.source_name_dict['order_lines']
    ids = ', '.join(f"'{value}'" for value in pkey)
    with sampler._connection.cursor() as cursor:
        cursor.execute(
            f'SELECT customer_id, COUNT(*) FROM {source_name} '
            f'WHERE customer_id IN ({ids}) GROUP BY customer_id'
        )
        expected = {row[0]: row[1] for row in cursor.fetchall()}

    for i, value in enumerate(pkey):
        assert counts.get(i, 0) == expected.get(value, 0)


def test_by_pkey_dedup(graph: Graph) -> None:
    r"""Duplicate primary keys in the input are de-duplicated and mapped back
    to the correct batch indices.
    """
    model = KumoRelational(graph)
    sampler = cast(DatabricksSampler, model._sampler)

    entity = sampler._sample_entity_table(
        'customers', {'customer_id'}, num_rows=3, random_seed=1
    )
    base = entity['customer_id'].tolist()
    # Repeat the keys to exercise duplicate handling:
    pkey = pd.Series(base + base)

    df, batch = sampler._by_pkey('customers', pkey, {'customer_id'})

    assert len(df) == len(pkey)
    assert np.array_equal(np.sort(batch), np.arange(len(pkey)))
    out = df['customer_id'].to_numpy()[np.argsort(batch)]
    assert list(out) == [str(v) for v in pkey]


def test_chunked_batch_offset(graph: Graph, monkeypatch) -> None:
    r"""When a large batch is split into multiple chunks (to respect the 1 MB
    parameter limit), per-chunk batch indices must be stitched back into a
    single global batch space. We force one chunk per row and assert results
    are identical to the single-chunk path.
    """
    model = KumoRelational(graph)
    sampler = cast(DatabricksSampler, model._sampler)

    entity = sampler._sample_entity_table(
        'customers', {'customer_id'}, num_rows=12, random_seed=11
    )
    pkey = entity['customer_id'].reset_index(drop=True)

    # Baseline: single chunk.
    df0, batch0 = sampler._by_pkey('customers', pkey, {'customer_id'})
    baseline = dict(zip(batch0.tolist(), df0['customer_id'].tolist()))

    # Force maximal chunking: one row per chunk.
    monkeypatch.setattr(
        sampler,
        '_chunk_rows',
        lambda rows, max_bytes=0: [[row] for row in rows] or [[]],
    )

    df1, batch1 = sampler._by_pkey('customers', pkey, {'customer_id'})
    chunked = dict(zip(batch1.tolist(), df1['customer_id'].tolist()))

    assert sorted(batch1.tolist()) == list(range(len(pkey)))
    assert chunked == baseline

    # Also verify the foreign-key path stitches chunks correctly:
    _, fk_batch = sampler._by_fkey(
        table_name='order_lines',
        foreign_key='customer_id',
        index=pkey,
        num_neighbors=10**9,
        anchor_time=None,
        columns={'customer_id', 'order_id'},
    )
    counts = pd.Series(fk_batch).value_counts().to_dict()

    source_name = sampler.source_name_dict['order_lines']
    ids = ', '.join(f"'{value}'" for value in pkey)
    with sampler._connection.cursor() as cursor:
        cursor.execute(
            f'SELECT customer_id, COUNT(*) FROM {source_name} '
            f'WHERE customer_id IN ({ids}) GROUP BY customer_id'
        )
        expected = {row[0]: row[1] for row in cursor.fetchall()}

    for i, value in enumerate(pkey):
        assert counts.get(i, 0) == expected.get(value, 0)
