# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
from pathlib import Path

import pandas as pd
import pytest

from sdm_client import Relationship, Schema
from sdm_client.errors import StructuredDataError

LINK = Relationship('orders', 'customer_id', 'customers', 'customer_id')


def schema(relationships=None):
    return Schema.from_data(
        {
            name: pd.read_csv(
                Path(__file__).resolve().parents[3]
                / 'tests/fixtures/schema-refinement'
                / f'{name}.csv'
            )
            for name in ('customers', 'orders')
        },
        primary_keys={'customers': 'customer_id', 'orders': 'order_id'},
        relationships=relationships,
    )


def test_infer_links_records_actual_origin_and_is_detached():
    original = schema([])
    inferred = original.infer_links()
    assert original.relationships == []
    assert inferred.relationships == [LINK]
    assert inferred.link_origins[0].origin == 'inferred'
    assert inferred.infer_links().link_origins == inferred.link_origins
    inferred['customers']['value'].stype = 'categorical'
    assert original['customers']['value'].stype != 'categorical'
    assert inferred['customers'].data is original['customers'].data
    assert (
        pickle.loads(pickle.dumps(inferred)).link_origins
        == inferred.link_origins
    )


def test_initial_inference_and_explicit_links_have_distinct_origins():
    assert schema().link_origins[0].origin == 'inferred'
    assert schema([LINK]).link_origins[0].origin == 'declared'
    unknown = schema()
    del unknown._inference
    restored = pickle.loads(pickle.dumps(unknown))
    assert restored.link_origins[0].origin == 'unknown'


def test_table_and_link_edits_are_atomic_and_keep_original():
    original = schema([LINK])
    assert (
        original.with_relationship(LINK).link_origins == original.link_origins
    )
    removed = original.without_table('orders')
    assert list(removed.tables) == ['customers']
    assert removed.relationships == []
    assert removed.link_origins == ()
    assert original.relationships == [LINK]
    assert original.without_relationship(LINK).relationships == []
    for action in (
        lambda: original.without_table('absent'),
        lambda: original.without_relationship(Relationship('x', 'a', 'y', 'b')),
    ):
        with pytest.raises(StructuredDataError) as error:
            action()
        assert error.value.code == 'INVALID_REQUEST'
    assert original.relationships == [LINK]


def test_explicit_none_and_types_survive_metadata_rerun():
    original = schema([])
    original['customers'].primary_key = None
    original['customers'].time_column = None
    original['customers']['value'].stype = 'categorical'
    changed = original.infer_metadata()
    assert changed['customers'].primary_key is None
    assert changed['customers'].time_column is None
    assert changed['customers']['value'].stype == 'categorical'
    assert changed.relationships == []


def test_missing_metadata_can_use_fresh_data_without_overwriting_declared_keys():
    frame = pd.DataFrame({'user_id': [0] * 30, 'value': [1.0] * 30})
    original = Schema.from_data({'users': frame}, relationships=[])
    assert original['users'].primary_key is None
    original['users'].data['user_id'] = range(30)
    inferred = original.infer_metadata()
    assert inferred['users'].primary_key == 'user_id'
    assert original['users'].primary_key is None
