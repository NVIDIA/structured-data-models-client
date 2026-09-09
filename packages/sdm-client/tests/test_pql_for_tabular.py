# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A predictive query that builds a single-table request.

The source design listed five reasons this could not work, one of them marked
fatal: a query resolves against a schema, so it can only ever name rows that
are already there, while a single-table model exists to score rows that are
not. The resolution is that the rows are an argument. The query still earns its
place by saying which column is the target and what kind of question it is.
"""

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest
from predictive_query import RELATIONAL, TABULAR

from sdm_client import PredictiveQuery
from sdm_client.errors import StructuredDataError


def users(n: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'user_id': range(n),
            'age': [20.0 + i for i in range(n)],
            'churn': [i % 2 for i in range(n)],
        }
    )


def new_rows() -> pd.DataFrame:
    return pd.DataFrame({'user_id': [900, 901], 'age': [31.0, 45.0]})


def query(text: str = 'PREDICT users.churn FOR EACH users.user_id'):
    return PredictiveQuery(text, {'users': users()})


def test_a_query_can_score_rows_the_tables_never_contained():
    """The blocker the source design called fatal, and how it is answered."""
    request = query().materialize(predict=new_rows())
    assert request.predict_rows == 2
    assert request.context_rows == 30
    assert not request.is_relational


def test_the_query_supplies_the_target_and_the_task():
    request = query().materialize(predict=new_rows())
    assert request.task.target.column_name == 'churn'
    assert request.task.kind == 'classification'


def test_the_task_kind_is_not_narrowed_from_the_targets_cardinality():
    """A two-class target must not become multiclass.

    A tabular model takes only the coarse kind and decides binary against
    multiclass itself, so narrowing here would be both a guess and a rejection.
    """
    request = query().materialize(predict=new_rows())
    assert request.task.kind == 'classification'
    assert request.task.kind not in {
        'binary_classification',
        'multiclass_classification',
    }


def test_the_targets_dtype_still_comes_from_the_schema():
    request = query().materialize(predict=new_rows())
    column = request.schema[request.entity_table]['churn']
    assert request.task.target.dtype == column.dtype


def test_indices_and_predict_are_mutually_exclusive():
    with pytest.raises(StructuredDataError) as error:
        query().materialize(indices=[1], predict=new_rows())
    assert 'not both' in str(error.value)


def test_neither_one_is_refused_rather_than_scoring_everything():
    with pytest.raises(StructuredDataError) as error:
        query().materialize()
    assert 'indices' in str(error.value)


def test_supplying_rows_for_a_multi_table_schema_is_refused():
    """Scoring supplied rows is the single-table path, and says so."""
    schema = {
        'users': users(),
        'orders': pd.DataFrame({'order_id': [1], 'user_id': [0]}),
    }
    with pytest.raises(StructuredDataError) as error:
        PredictiveQuery(
            'PREDICT users.churn FOR EACH users.user_id', schema
        ).materialize(predict=new_rows())
    assert 'single-table' in str(error.value)


def test_a_query_the_tabular_profile_rejects_is_refused_by_the_language():
    """A capability may only reject, and an aggregation needs a second table."""
    with pytest.raises((StructuredDataError, ValueError)) as error:
        PredictiveQuery(
            'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
            {
                'users': users(),
                'orders': pd.DataFrame(
                    {
                        'order_id': [1, 2],
                        'user_id': [0, 1],
                        'ts': pd.to_datetime(['2024-01-01', '2024-02-01']),
                    }
                ),
            },
        ).materialize(predict=new_rows())
    assert str(error.value)


@pytest.mark.parametrize(
    ('argument', 'expected_profile', 'expected_materializer'),
    [
        ('predict', TABULAR, 'tabular'),
        ('indices', RELATIONAL, 'relational'),
    ],
)
def test_materialize_selects_the_profile_from_the_call_shape(
    monkeypatch, argument, expected_profile, expected_materializer
):
    predictive_query = query()
    request = object()
    resolve = Mock(return_value=object())
    tabular = Mock(return_value=request)
    relational = Mock(return_value=request)
    monkeypatch.setattr(predictive_query, '_resolve', resolve)
    monkeypatch.setattr(
        predictive_query._tabular_materializer, 'table', tabular
    )
    monkeypatch.setattr(
        predictive_query._relational_materializer, 'materialize', relational
    )

    arguments = (
        {'predict': new_rows()} if argument == 'predict' else {'indices': [1]}
    )
    assert predictive_query.materialize(**arguments) is request
    resolve.assert_called_once_with(expected_profile)

    selected = tabular if expected_materializer == 'tabular' else relational
    skipped = relational if expected_materializer == 'tabular' else tabular
    selected.assert_called_once()
    skipped.assert_not_called()


def test_a_query_predicting_a_column_that_is_not_there_is_refused():
    with pytest.raises((StructuredDataError, ValueError)):
        PredictiveQuery(
            'PREDICT users.nope FOR EACH users.user_id',
            {'users': users()},
        ).materialize(predict=new_rows())


def test_a_schema_translates_into_the_language_once():
    """One translation, so a query and the engine cannot read it differently."""
    from sdm_client.schema import Schema, to_graph_definition

    schema = Schema.from_data({'users': users()})
    definition = to_graph_definition(schema)
    assert 'users' in definition.tables
    assert definition.tables['users'].pkey == 'user_id'
    names = {column.name for column in definition.tables['users'].cols}
    assert {'user_id', 'age', 'churn'} <= names


@pytest.mark.parametrize('bad', [0, -1, 5.5, '5', True])
def test_a_bad_context_count_is_refused_on_this_path_too(bad):
    """Validation runs before the branch, so it cannot be routed around.

    This path returned early, ahead of the check, so an invalid count was
    silently accepted here while being refused on the other one.
    """
    with pytest.raises(StructuredDataError):
        query().materialize(predict=new_rows(), num_context_samples=bad)


@pytest.mark.parametrize('count', [500, 10_001])
def test_predict_does_not_accept_a_context_count(count):
    with pytest.raises(StructuredDataError) as error:
        query().materialize(
            predict=new_rows(),
            num_context_samples=count,
        )

    message = str(error.value)
    assert 'cannot be used with predict=<frame>' in message
    assert 'schema table is used as context in full' in message


def test_predict_does_not_accept_relational_sampling_options():
    with pytest.raises(StructuredDataError) as error:
        query().materialize(
            predict=new_rows(),
            num_neighbors=[8, 8],
        )

    message = str(error.value)
    assert 'predict=<frame> does not accept options' in message
    assert 'num_neighbors' in message


def test_the_ordinary_call_still_works():
    assert query().materialize(predict=new_rows()).predict_rows == 2


def test_predict_does_not_accept_unknown_options():
    with pytest.raises(StructuredDataError) as error:
        query().materialize(
            predict=new_rows(),
            num_neighbours=[8, 8],
        )

    message = str(error.value)
    assert 'predict=<frame> does not accept options' in message
    assert 'num_neighbours' in message
