# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public frame schema inference through the shared Rust policy."""

import pandas as pd
import pytest

from sdm_client import Schema, _routing
from sdm_client.errors import StructuredDataError


@pytest.fixture(autouse=True)
def frame_schema_route(monkeypatch):
    monkeypatch.setenv(_routing.ENVIRONMENT, ','.join(_routing.FRAME_SCHEMA))


def test_inferred_relationship_keeps_original_frames():
    customers = pd.DataFrame({'customer_id': [1, 2], 'age': [30, 40]})
    orders = pd.DataFrame(
        {
            'order_id': [10, 11, 12],
            'customer_id': [1, 1, 2],
            'amount': [2.0, 3.0, 4.0],
        }
    )
    schema = Schema.from_data({'customers': customers, 'orders': orders})
    assert schema['customers'].data is customers
    assert schema['orders'].data is orders
    assert schema['customers'].primary_key == 'customer_id'
    assert [
        (r.source_table, r.source_column, r.target_table, r.target_column)
        for r in schema.relationships
    ] == [
        ('orders', 'customer_id', 'customers', 'customer_id'),
    ]


def test_explicit_empty_relationships_remain_empty():
    schema = Schema.from_data(
        {
            'parent': pd.DataFrame({'parent_id': [1, 2]}),
            'child': pd.DataFrame(
                {'child_id': [1, 2, 3], 'parent_id': [1, 1, 2]}
            ),
        },
        relationships=[],
    )
    assert schema.relationships == []


def test_composite_override_counts_whole_key():
    frame = pd.DataFrame({'a': [1, 1, 2], 'b': [1, 2, 1]})
    schema = Schema.from_data(
        {'data': frame}, primary_keys={'data': ['a', 'b']}
    )
    assert schema['data'].primary_key == ('a', 'b')
    assert schema['data'].duplicate_keys == 0


@pytest.mark.parametrize(
    'frame',
    [
        pd.DataFrame({'id': [1, 1]}),
        pd.DataFrame({'id': pd.Series([1, None], dtype='Int64')}),
    ],
)
def test_declared_key_rejects_duplicates_and_missing_values(frame):
    with pytest.raises(StructuredDataError):
        Schema.from_data({'data': frame}, primary_keys={'data': 'id'})
