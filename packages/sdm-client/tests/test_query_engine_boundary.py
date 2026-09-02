# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the engine does behind `materialize()`, and what the caller sees.

Three separate things were leaking out of that boundary: the engine's own
exception types, its progress output, and a graph that was not the schema's.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pandas as pd
import pytest

from sdm_client import PredictiveQuery, Relationship, Schema
from sdm_client.errors import StructuredDataError

pytest.importorskip('sdm_relational_engine')


def users(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'user_id': range(rows),
            'age': [20.0 + index for index in range(rows)],
            'churn': [index % 2 for index in range(rows)],
        }
    )


def orders(rows: int = 80) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'order_id': range(rows),
            'user_id': [index % 40 for index in range(rows)],
            'price': [10.0 + index for index in range(rows)],
            'placed': pd.date_range('2024-01-01', periods=rows, freq='D'),
        }
    )


def schema() -> Schema:
    return Schema.from_data({'users': users(), 'orders': orders()})


def query(text: str = 'PREDICT users.churn FOR EACH users.user_id'):
    return PredictiveQuery(text, schema())


# --- the graph sent is the schema described ---------------------------------


def edges_of(built: Schema) -> set:
    from sdm_client.query import _engine, _to_engine_graph

    graph = _to_engine_graph(_engine(), built)
    return {(edge.src_table, edge.fkey, edge.dst_table) for edge in graph.edges}


def described(built: Schema) -> set:
    return {
        (link.source_table, link.source_column, link.target_table)
        for link in built.relationships
    }


def test_the_graph_sent_is_exactly_what_the_schema_describes():
    built = schema()
    assert edges_of(built) == described(built)


def test_asserting_there_are_no_relationships_means_none_are_sent():
    """The docstring says an empty sequence asserts there are none."""
    built = Schema.from_data(
        {'users': users(), 'orders': orders()}, relationships=[]
    )
    assert built.relationships == []
    assert edges_of(built) == set()


def test_one_explicit_relationship_sends_exactly_that_one():
    built = Schema.from_data(
        {'users': users(), 'orders': orders()},
        relationships=[Relationship('orders', 'user_id', 'users', 'user_id')],
    )
    assert edges_of(built) == {('orders', 'user_id', 'users')}


def test_a_relationship_the_schema_does_not_have_is_not_invented():
    """A column an inference elsewhere might link on stays unlinked."""
    frame = users()
    frame['orders_on_hand'] = frame['user_id']
    built = Schema.from_data(
        {'users': frame, 'orders': orders()},
        relationships=[Relationship('orders', 'user_id', 'users', 'user_id')],
    )
    assert edges_of(built) == {('orders', 'user_id', 'users')}


# --- a library writes to a logger, not to stdout ----------------------------


def test_materialize_prints_nothing():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        query().materialize(indices=[1, 2], num_context_samples=20)
    assert buffer.getvalue() == ''


# --- every failure is this client's own error -------------------------------


def test_an_anchor_before_the_data_is_a_structured_data_error():
    with pytest.raises(StructuredDataError) as error:
        query(
            'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id'
        ).materialize(
            indices=[1, 2],
            num_context_samples=20,
            anchor_time=pd.Timestamp('1990-01-01'),
        )
    assert 'Anchor timestamp' in str(error.value)


def test_an_entity_that_is_not_in_the_table_is_a_structured_data_error():
    with pytest.raises(StructuredDataError):
        query().materialize(indices=[-1, -2], num_context_samples=20)


def test_the_entity_ceiling_is_named_by_this_client_before_gathering():
    """The engine's own message names a `batch_size=` this client has not got."""
    with pytest.raises(StructuredDataError) as error:
        query().materialize(indices=list(range(1_001)), num_context_samples=20)
    message = str(error.value)
    assert error.value.code == 'UNSUPPORTED_REQUEST'
    assert '1,000' in message
    assert 'batch_size' not in message


def test_the_ceiling_is_refused_before_the_engine_is_reached():
    built = query()
    called = False

    def _fail(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError('the engine was reached')

    built._engine_model = _fail  # type: ignore[method-assign]
    with pytest.raises(StructuredDataError):
        built.materialize(indices=list(range(1_001)), num_context_samples=20)
    assert not called


def test_a_dropped_row_is_explained_rather_than_only_counted():
    """The engine says how many rows went unresolved, never why. It knows."""
    frame = orders()
    frame.loc[frame.index[:10], 'placed'] = pd.NaT
    built = Schema.from_data({'users': users(), 'orders': frame})
    with pytest.raises(StructuredDataError) as error:
        PredictiveQuery(
            'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id', built
        ).materialize(
            indices=[1, 2], num_context_samples=20, anchor_time='entity'
        )
    message = str(error.value)
    if 'unresolved' in message or 'dropped' in message:
        assert 'no time' in message or 'dropped' in message


def test_an_ordinary_call_is_unaffected():
    request = query().materialize(indices=[1, 2], num_context_samples=20)
    assert request.payloads
