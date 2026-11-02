# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public sampled-context ordering is independent of NumPy binary profiles."""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from sdm_client import PredictiveQuery
from sdm_client._rust import tabular_index


def key(value):
    if isinstance(value, tuple):
        return tuple(key(part) for part in value)
    if pd.isna(value):
        return (1, 0)
    if isinstance(value, np.generic):
        value = value.item()
    return (0, value)


@pytest.mark.parametrize(
    'labels',
    [
        [2, 1, 1, 0],
        [np.uint64(2**63 + 1), np.int64(2**63 - 1), 2**63],
        [np.int64(2**53 + 1), np.float64(2**53), 2**53],
        [np.float16(1), np.float32(1), np.float64(1), 0],
        [np.float32(2**24), float(2**24 + 1), np.int64(2**24 + 1)],
        [np.float64(float('inf')), 10**400, -(10**400), float('-inf')],
        [np.float64(1), None, float('nan'), np.int64(0)],
        [Decimal('1.0'), Decimal('1.00'), Decimal('0')],
    ],
)
def test_sampled_numeric_context_uses_stable_value_order(labels, monkeypatch):
    def producer_not_needed(*args):
        raise AssertionError(
            'portable ordering must not inspect NumPy producers'
        )

    monkeypatch.setattr(tabular_index, '_profile', producer_not_needed)
    monkeypatch.setattr(
        tabular_index, '_comparison_semantics', producer_not_needed
    )
    index = pd.Index([labels[i % len(labels)] for i in range(81)], dtype=object)
    assert_materialization(index)


def test_multiindex_context_keeps_equal_tuple_draw_order():
    index = pd.MultiIndex.from_tuples(
        [(i % 3, (i // 3) % 2) for i in range(81)]
    )
    assert_materialization(index)


def assert_materialization(index):
    data = pd.DataFrame(
        {
            'id': range(81),
            'feature': np.arange(81) * 0.5,
            'target': np.arange(81) + 100.0,
        },
        index=index,
    )
    sampled = data.iloc[:-1].sample(n=40, random_state=42)
    positions = sorted(range(len(sampled)), key=lambda i: key(sampled.index[i]))
    expected = sampled.iloc[positions]['id'].tolist()
    request = PredictiveQuery(
        'PREDICT data.target FOR EACH data.id', {'data': data}
    ).materialize(indices=[80], num_context_samples=40, task='regression')
    assert request.context.instance_table['id'].tolist() == expected
    assert set(expected) == set(sampled['id'])
    assert request.predict.instance_table['id'].tolist() == [80]


@pytest.mark.parametrize('seed', range(12))
def test_generated_context_matches_independent_row_identity_oracle(seed):
    rng = np.random.default_rng(seed)
    count = int(rng.integers(24, 100))
    labels = [None, -1, 0, 0, 2**53 + 1, np.float64(2**53)]
    data = pd.DataFrame(
        {
            'id': range(count),
            'feature': rng.normal(size=count),
            'target': rng.normal(size=count),
        },
        index=pd.Index(
            [labels[int(i)] for i in rng.integers(len(labels), size=count)],
            dtype=object,
        ),
    )
    # Null features are valid; null targets must never enter the context.
    data.iloc[::5, data.columns.get_loc('feature')] = np.nan
    data.iloc[::7, data.columns.get_loc('target')] = np.nan
    requested = rng.choice(count, size=3, replace=False).tolist()
    candidates = data.loc[~data['id'].isin(requested) & data['target'].notna()]
    size = int(rng.integers(3, len(candidates) + 1))
    if size == len(candidates):
        expected = candidates['id'].tolist()
    else:
        draw = candidates.sample(n=size, random_state=42)
        order = sorted(range(size), key=lambda i: key(draw.index[i]))
        expected = draw.iloc[order]['id'].tolist()
    request = PredictiveQuery(
        'PREDICT data.target FOR EACH data.id', {'data': data}
    ).materialize(
        indices=requested, num_context_samples=size, task='regression'
    )
    assert request.context.instance_table['id'].tolist() == expected
    assert request.predict.instance_table['id'].tolist() == requested
