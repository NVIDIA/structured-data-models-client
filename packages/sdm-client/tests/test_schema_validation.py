# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A schema edit is checked when it is made, not several cells later.

The engine this client is built on already validates these on assignment --
`rfm/base/table.py` and `rfm/base/column.py` -- and the client's `Table` was a
plain dataclass, so a wrong edit was stored and only failed at `materialize()`,
as a bare `KeyError` or a `ValueError` outside the documented error surface.

Nothing is refused here that the engine would have accepted. The rule is the
engine's own.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sdm_client import Relationship, Schema
from sdm_client.errors import StructuredDataError


def users(rows: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'user_id': range(rows),
            'age': [20.0 + index for index in range(rows)],
            'segment': [
                'north' if index % 2 else 'south' for index in range(rows)
            ],
            'joined': pd.date_range('2024-01-01', periods=rows, freq='D'),
        }
    )


def schema(rows: int = 6) -> Schema:
    return Schema.from_data({'users': users(rows)})


# --- a column has to exist ---------------------------------------------------


def test_a_primary_key_that_is_not_a_column_is_refused_at_assignment():
    """It used to be stored, and surface as a KeyError from a dict lookup."""
    with pytest.raises(StructuredDataError) as error:
        schema()['users'].primary_key = 'NoSuchColumn'
    assert 'no column' in str(error.value)


def test_a_time_column_that_is_not_a_column_is_refused_at_assignment():
    with pytest.raises(StructuredDataError):
        schema()['users'].time_column = 'NoSuchColumn'


def test_an_empty_composite_key_is_refused():
    with pytest.raises(StructuredDataError) as error:
        schema()['users'].primary_key = []
    assert 'empty primary key' in str(error.value)


# --- the two roles are exclusive ---------------------------------------------


def test_the_time_column_cannot_also_be_the_primary_key():
    built = schema()
    built['users'].time_column = 'joined'
    with pytest.raises(StructuredDataError) as error:
        built['users'].primary_key = 'joined'
    assert 'time column' in str(error.value)


def test_the_primary_key_cannot_also_be_the_time_column():
    built = schema()
    with pytest.raises(StructuredDataError) as error:
        built['users'].time_column = 'user_id'
    assert 'primary key' in str(error.value)


# --- a meaning the values cannot carry ---------------------------------------


def test_text_on_a_numeric_column_is_refused():
    with pytest.raises(StructuredDataError) as error:
        schema()['users']['age'].stype = 'text'
    assert 'cannot mean' in str(error.value)


def test_a_numeric_column_cannot_be_the_time_column():
    with pytest.raises(StructuredDataError):
        Schema.from_data({'users': users()}, time_columns={'users': 'age'})


def test_a_correction_the_docs_invite_still_works():
    """`age` is numeric and re-typing it categorical is a legal correction."""
    built = schema()
    built['users']['age'].stype = 'categorical'
    assert built['users']['age'].stype == 'categorical'


def test_a_string_column_may_still_be_the_time_column():
    """A string can hold an ISO timestamp, and the language allows it.

    The rule is the engine's, so this stays legal even though it looks odd.
    """
    frame = users()
    frame['when'] = [f'2024-01-0{index + 1}' for index in range(len(frame))]
    built = Schema.from_data({'users': frame})
    built['users'].time_column = 'when'
    assert built['users'].time_column == 'when'


def test_an_unknown_semantic_type_is_still_refused_by_name():
    with pytest.raises(StructuredDataError) as error:
        schema()['users']['age'].stype = 'nonsense'
    assert 'not a semantic type' in str(error.value)


# --- a key has to identify a row --------------------------------------------


def duplicated(rows: int = 1_000, repeats: int = 20) -> pd.DataFrame:
    frame = users(rows)
    return pd.concat([frame, frame.head(repeats)], ignore_index=True)


def test_an_explicit_key_that_repeats_is_refused():
    with pytest.raises(StructuredDataError) as error:
        Schema.from_data(
            {'users': duplicated()}, primary_keys={'users': 'user_id'}
        )
    message = str(error.value)
    assert 'not unique enough' in message
    assert '20 of 1020' in message


def test_assigning_a_repeating_key_is_refused_too():
    """Inference declined this column; assignment must not put it back."""
    built = Schema.from_data({'users': duplicated()})
    assert built['users'].primary_key is None
    with pytest.raises(StructuredDataError):
        built['users'].primary_key = 'user_id'


def test_the_tolerance_inference_allows_is_the_tolerance_assignment_allows():
    """One duplicate in a thousand is what the 99% threshold exists for."""
    built = Schema.from_data({'users': duplicated(repeats=5)})
    assert built['users'].primary_key == 'user_id'
    built['users'].primary_key = 'user_id'


def test_a_unique_key_is_accepted():
    built = schema()
    built['users'].primary_key = 'age'
    assert built['users'].primary_key == 'age'


def test_a_composite_key_is_checked_across_its_columns():
    frame = pd.DataFrame(
        {'a': [1, 1, 2, 2], 'b': [1, 2, 1, 2], 'v': [1.0, 2.0, 3.0, 4.0]}
    )
    built = Schema.from_data({'t': frame}, relationships=[])
    built['t'].primary_key = ['a', 'b']
    assert built['t'].primary_key == ['a', 'b']
    with pytest.raises(StructuredDataError):
        built['t'].primary_key = ['a']


def test_an_empty_table_has_no_key_to_check():
    frame = pd.DataFrame({'user_id': pd.Series(dtype='int64')})
    built = Schema.from_data({'users': frame}, relationships=[])
    built['users'].primary_key = 'user_id'


# --- the constructor and the setter agree ------------------------------------


def test_the_constructor_and_the_setter_refuse_the_same_things():
    frame = duplicated()
    with pytest.raises(StructuredDataError):
        Schema.from_data({'users': frame}, primary_keys={'users': 'user_id'})
    built = Schema.from_data({'users': frame})
    with pytest.raises(StructuredDataError):
        built['users'].primary_key = 'user_id'


def test_an_inferred_schema_is_unaffected():
    built = Schema.from_data({'users': users(), 'orders': orders()})
    assert built['users'].primary_key == 'user_id'
    assert built['orders'].primary_key == 'order_id'
    assert built['users'].time_column == 'joined'


def orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'order_id': range(8),
            'user_id': [index % 6 for index in range(8)],
            'price': [10.0 + index for index in range(8)],
            'placed': pd.date_range('2024-02-01', periods=8, freq='D'),
        }
    )


def test_an_explicit_relationship_still_marks_both_ends_as_ids():
    built = Schema.from_data(
        {'users': users(), 'orders': orders()},
        relationships=[Relationship('orders', 'user_id', 'users', 'user_id')],
    )
    assert built['orders']['user_id'].stype == 'ID'
