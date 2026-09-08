# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema inference: what it gets right, and what it refuses to guess."""

from __future__ import annotations

import pandas as pd
import pytest

from sdm_client import PredictiveQuery, Relationship, Schema
from sdm_client.errors import StructuredDataError


def users() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'user_id': [1, 2, 3, 4],
            'age': [31.0, 44.5, 28.0, 51.0],
            'segment': ['north', 'south', 'north', 'east'],
            'joined': pd.to_datetime(
                ['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01']
            ),
        }
    )


def orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'order_id': [10, 11, 12],
            'user_id': [1, 2, 1],
            'price': [10.0, 20.0, 30.0],
        }
    )


def test_a_single_table_schema_is_the_ordinary_case():
    schema = Schema.from_data({'users': users()})
    assert len(schema) == 1
    assert schema.relationships == []
    assert 'users' in schema


def test_dtypes_and_semantic_types_are_inferred_per_column():
    table = Schema.from_data({'users': users()})['users']
    assert table['age'].dtype == 'float64'
    assert table['age'].stype == 'numerical'
    assert table['segment'].stype == 'categorical'
    assert table['joined'].stype == 'timestamp'
    assert table['joined'].dtype.startswith('timestamp')


def test_a_key_shaped_unique_column_becomes_the_primary_key_and_an_id():
    table = Schema.from_data({'users': users()})['users']
    assert table.primary_key == 'user_id'
    # A key typed from its dtype alone becomes numerical, is stored as a float,
    # and then fails to join against an integer key.
    assert table['user_id'].stype == 'ID'


def test_a_unique_column_without_a_key_shaped_name_is_not_a_key():
    frame = pd.DataFrame({'measurement': [1.5, 2.5, 3.5], 'n': [1, 2, 3]})
    assert Schema.from_data({'t': frame})['t'].primary_key is None


def test_a_key_shaped_column_that_repeats_is_not_a_key():
    frame = pd.DataFrame({'user_id': [1, 1, 1, 2], 'v': [1, 2, 3, 4]})
    assert Schema.from_data({'t': frame})['t'].primary_key is None


def test_the_single_timestamp_column_becomes_the_time_column():
    assert Schema.from_data({'users': users()})['users'].time_column == 'joined'


def test_two_timestamp_columns_are_ambiguous_so_neither_is_chosen():
    frame = users().assign(last_seen=lambda f: f['joined'])
    assert Schema.from_data({'users': frame})['users'].time_column is None


def test_a_foreign_key_is_linked_to_the_table_whose_key_it_names():
    schema = Schema.from_data({'users': users(), 'orders': orders()})
    assert schema.relationships == [
        Relationship('orders', 'user_id', 'users', 'user_id')
    ]
    # The join key must be an identifier on both sides.
    assert schema['orders']['user_id'].stype == 'ID'


def test_relationships_can_be_asserted_empty():
    schema = Schema.from_data(
        {'users': users(), 'orders': orders()}, relationships=()
    )
    assert schema.relationships == []


def test_a_table_does_not_link_to_itself():
    schema = Schema.from_data({'users': users()})
    assert schema.relationships == []


def test_overrides_replace_inference():
    # `age` rather than `segment`: an override is an assertion that the column
    # identifies a row, and `segment` repeats. Any unique column inference
    # would not have picked makes the same point.
    schema = Schema.from_data(
        {'users': users()},
        primary_keys={'users': 'age'},
        time_columns={'users': 'joined'},
    )
    assert schema['users'].primary_key == 'age'
    assert schema['users'].time_column == 'joined'


def test_a_semantic_type_can_be_corrected_after_the_fact():
    schema = Schema.from_data({'users': users()})
    schema['users']['age'].stype = 'categorical'
    assert schema['users']['age'].stype == 'categorical'


def test_an_unknown_semantic_type_is_refused_when_it_is_set():
    schema = Schema.from_data({'users': users()})
    with pytest.raises(StructuredDataError) as error:
        schema['users']['age'].stype = 'numerological'
    assert error.value.code == 'INVALID_REQUEST'


@pytest.mark.parametrize(
    ('tables', 'fragment'),
    [
        ({}, 'at least one table'),
        ({'t': [1, 2, 3]}, 'must be a pandas DataFrame'),
    ],
)
def test_bad_input_is_named_rather_than_crashing_later(tables, fragment):
    with pytest.raises(StructuredDataError) as error:
        Schema.from_data(tables)
    assert fragment in str(error.value)


def test_duplicate_column_names_are_refused():
    frame = pd.DataFrame([[1, 2]], columns=['a', 'a'])
    with pytest.raises(StructuredDataError) as error:
        Schema.from_data({'t': frame})
    assert 'duplicate column' in str(error.value)


def test_an_override_naming_something_absent_is_refused():
    with pytest.raises(StructuredDataError) as error:
        Schema.from_data({'users': users()}, primary_keys={'users': 'nope'})
    assert 'no column' in str(error.value)


def test_indexing_a_missing_table_or_column_lists_what_is_there():
    schema = Schema.from_data({'users': users()})
    with pytest.raises(StructuredDataError) as error:
        schema['orders']
    assert 'users' in str(error.value)
    with pytest.raises(StructuredDataError) as error:
        schema['users']['nope']
    assert 'age' in str(error.value)


def test_an_all_null_object_column_does_not_claim_a_numeric_type():
    frame = pd.DataFrame({'k_id': [1, 2], 'maybe': [None, None]})
    assert Schema.from_data({'t': frame})['t']['maybe'].dtype == 'string'


def test_a_high_cardinality_string_column_is_text_not_categorical():
    frame = pd.DataFrame({'note': [f'a unique note {i}' for i in range(200)]})
    assert Schema.from_data({'t': frame})['t']['note'].stype == 'text'


def test_a_contradictory_column_is_refused_before_anything_is_sent():
    """The pair was checked from one side only.

    Assigning a meaning the values cannot carry was refused, but reaching the
    same pair by assigning the storage instead was not, and nothing asked
    again before the request was built -- so a column could say it holds
    timestamps and means a quantity, and be sent.

    Checked when the caller has finished rather than as they assign, because
    re-declaring storage takes two assignments and either order passes
    through a pair that does not hold.
    """
    schema = Schema.from_data(
        {'users': pd.DataFrame({'user_id': [1, 2], 'age': [20.0, 30.0]})},
        relationships=[],
    )
    schema['users'].primary_key = 'user_id'
    schema['users']['age'].dtype = 'timestamp[us]'

    with pytest.raises(StructuredDataError) as error:
        PredictiveQuery(
            'PREDICT users.age FOR EACH users.user_id', schema
        ).materialize(indices=[1])
    assert error.value.code == 'INVALID_REQUEST'


def test_re_declaring_storage_and_meaning_together_is_allowed():
    """Either assignment order passes through a pair that does not hold, so
    neither may be refused as it happens."""
    schema = Schema.from_data(
        {'orders': pd.DataFrame({'order_id': [1, 2], 'ship_via': [1, 2]})},
        relationships=[],
    )
    column = schema['orders']['ship_via']

    column.dtype = 'string'
    column.stype = 'categorical'

    column.check_consistent()
    assert (column.dtype, column.stype) == ('string', 'categorical')


def test_the_tabular_path_refuses_a_contradictory_column_too():
    """Both request paths start from a schema and both must ask the same
    question; putting the check on the relational one alone left this open."""
    from sdm_client import Request
    from sdm_client.adapters.tabular import TabularAdapter

    context = pd.DataFrame({'x': [1, 2, 3], 'y': [0, 1, 0]})
    predict = pd.DataFrame({'x': [4]})
    schema = Schema.from_data({'instance': context}, relationships=[])
    schema['instance']['x'].dtype = 'string'

    request = Request(
        context=context,
        predict=predict,
        target='y',
        task='classification',
        schema=schema,
    )

    with pytest.raises(StructuredDataError) as error:
        next(TabularAdapter().prediction_plan(request, ('prediction',)))
    assert error.value.code == 'INVALID_REQUEST'


def test_the_tabular_path_accepts_a_column_re_declared_in_full():
    from sdm_client import Request
    from sdm_client.adapters.tabular import TabularAdapter

    context = pd.DataFrame({'x': [1, 2, 3], 'y': [0, 1, 0]})
    predict = pd.DataFrame({'x': [4]})
    schema = Schema.from_data({'instance': context}, relationships=[])
    schema['instance']['x'].dtype = 'string'
    schema['instance']['x'].stype = 'categorical'

    request = Request(
        context=context,
        predict=predict,
        target='y',
        task='classification',
        schema=schema,
    )
    body = next(TabularAdapter().prediction_plan(request, ('prediction',)))

    assert body['schema']['instance_table']['columns']['x'] == {
        'dtype': 'string',
        'stype': 'categorical',
    }
