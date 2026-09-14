# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
import pytest

from sdm_client import PredictiveQuery
from sdm_client.errors import StructuredDataError


def users(n: int = 50, *, unknown: int = 0) -> pd.DataFrame:
    """``n`` rows, the last ``unknown`` of them without a target."""
    churn: list[str | None] = ['yes' if i % 2 else 'no' for i in range(n)]
    for position in range(n - unknown, n):
        churn[position] = None
    return pd.DataFrame(
        {
            'user_id': range(n),
            'age': [20.0 + i for i in range(n)],
            'churn': churn,
        }
    )


def query(frame: pd.DataFrame | None = None) -> PredictiveQuery:
    return PredictiveQuery(
        'PREDICT users.churn FOR EACH users.user_id',
        {'users': users() if frame is None else frame},
    )


# --- the resulting request --------------------------------------------------


def test_one_table_and_indices_builds_a_flat_request():
    request = query().materialize(indices=[1, 2, 3])
    assert not request.is_relational
    assert request.payloads is None
    assert request.predict_rows == 3


# --- the rows being scored --------------------------------------------------


def test_the_rows_are_looked_up_by_the_column_the_query_names():
    """``FOR EACH users.user_id`` means the indices are ``user_id`` values."""
    request = query().materialize(indices=[7, 9])
    scored = request.predict.instance_table
    assert list(scored['user_id']) == [7, 9]
    assert list(scored['age']) == [27.0, 29.0]


def test_the_rows_come_back_in_the_order_they_were_asked_for():
    request = query().materialize(indices=[9, 1, 5])
    assert list(request.predict.instance_table['user_id']) == [9, 1, 5]


def test_all_rows_matching_an_entity_value_are_scored():
    frame = users(200)
    frame.loc[[7, 8], 'user_id'] = 6

    request = query(frame).materialize(indices=[6])

    assert list(request.predict.instance_table['user_id']) == [6, 6, 6]
    assert 6 not in set(request.context.instance_table['user_id'])


def test_the_target_is_dropped_from_the_rows_being_scored():
    """Those are the rows whose answer is being asked for."""
    scored = query().materialize(indices=[1, 2]).predict.instance_table
    assert 'churn' not in scored.columns
    assert 'age' in scored.columns


def test_every_other_column_ships_as_it_stands():
    request = query().materialize(indices=[1])
    assert set(request.context.instance_table.columns) == {
        'user_id',
        'age',
        'churn',
    }
    assert set(request.predict.instance_table.columns) == {'user_id', 'age'}


def test_a_key_the_table_does_not_hold_is_refused():
    with pytest.raises(StructuredDataError) as error:
        query().materialize(indices=[1, 4_000, 4_001])
    message = str(error.value)
    assert '4000' in message and '4001' in message
    assert 'users.user_id' in message
    assert error.value.code == 'INVALID_REQUEST'


# --- the context ------------------------------------------------------------


def test_the_context_is_the_rows_whose_target_is_known():
    request = query(users(50, unknown=10)).materialize(indices=[0])
    context = request.context.instance_table
    assert len(context) == 39  # 50 rows, 10 unlabelled, 1 being scored
    assert context['churn'].notna().all()


def test_the_rows_being_scored_are_never_in_the_context():
    """A row must not be both the question and an answered example.

    Every row here has a known target, so nothing but the exclusion keeps them
    apart -- which is exactly the case that leaks if it is not written down.
    """
    scored = [3, 8, 11, 40]
    request = query().materialize(indices=scored)
    context = request.context.instance_table
    assert set(context['user_id']).isdisjoint(scored)
    assert len(context) == 46


def test_the_same_call_twice_builds_the_same_context():
    frame = users(4_000)
    first = query(frame).materialize(indices=[1, 2], num_context_samples=100)
    second = query(frame).materialize(indices=[1, 2], num_context_samples=100)
    assert first.context_rows == second.context_rows
    pd.testing.assert_frame_equal(
        first.context.instance_table, second.context.instance_table
    )


def test_the_sample_is_a_sample_and_not_the_first_rows():
    """Uniform over the table, so the head of it is not the answer."""
    request = query(users(4_000)).materialize(
        indices=[0], num_context_samples=100
    )
    drawn = list(request.context.instance_table['user_id'])
    assert drawn != sorted(range(1, 101))
    assert max(drawn) > 1_000


def test_the_count_caps_the_context():
    request = query(users(4_000)).materialize(
        indices=[0], num_context_samples=250
    )
    assert request.context_rows == 250


def test_omitting_the_count_uses_all_eligible_context_rows():
    request = query(users(4_000)).materialize(indices=[0])
    assert request.context_rows == 3_999


def test_asking_for_more_than_the_table_holds_takes_what_there_is():
    request = query().materialize(indices=[0], num_context_samples=900)
    assert request.context_rows == 49


# --- the same packaging, whichever way the rows were found ------------------


def test_the_flat_paths_agree_on_the_target_and_the_task():
    """Only the frames differ. The question they are packaged into does not."""
    frame = users()
    from_indices = query(frame).materialize(indices=[1, 2])
    from_rows = query(frame).materialize(
        predict=pd.DataFrame({'user_id': [900, 901], 'age': [31.0, 45.0]})
    )
    assert from_indices.task == from_rows.task
    assert from_indices.task.kind == 'classification'
    assert from_indices.task.target.column_name == 'churn'
    assert from_indices.entity_table == from_rows.entity_table
    assert list(from_indices.schema) == list(from_rows.schema) == ['users']
    assert (
        from_indices.task.target.dtype
        == from_rows.task.target.dtype
        == from_rows.schema['users']['churn'].dtype
    )
    assert from_indices.is_relational == from_rows.is_relational is False


def test_the_task_kind_is_not_narrowed_from_the_targets_cardinality():
    """A two-class target must not become binary here either."""
    kind = query().materialize(indices=[1]).task.kind
    assert kind == 'classification'
    assert kind not in {'binary_classification', 'multiclass_classification'}


def test_the_contexts_differ_by_design():
    """Disjoint on one path; the table itself on the other."""
    frame = users()
    from_indices = query(frame).materialize(indices=[1, 2])
    from_rows = query(frame).materialize(
        predict=pd.DataFrame({'user_id': [900], 'age': [31.0]})
    )
    assert from_indices.context_rows == 48
    assert from_rows.context_rows == 50
