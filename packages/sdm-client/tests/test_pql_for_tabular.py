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
            'churn': ['yes' if i % 2 else 'no' for i in range(n)],
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


@pytest.mark.parametrize(
    ('target', 'expected_task'),
    [
        pytest.param(
            ['yes' if i % 2 else 'no' for i in range(30)],
            'classification',
            id='classification',
        ),
        pytest.param(
            [float(i) for i in range(30)],
            'regression',
            id='regression',
        ),
    ],
)
@pytest.mark.parametrize('path', ['table', 'indices'])
def test_each_tabular_path_infers_the_task(target, expected_task, path):
    frame = users().assign(churn=target)
    predictive_query = PredictiveQuery(
        'PREDICT users.churn FOR EACH users.user_id', {'users': frame}
    )
    arguments = (
        {'predict': new_rows()} if path == 'table' else {'indices': [1, 2]}
    )

    request = predictive_query.materialize(**arguments)

    assert request.task.kind == expected_task
    assert request.task.target.column_name == 'churn'


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


@pytest.mark.parametrize(
    ('argument', 'tables', 'expected_profile', 'expected_materializer'),
    [
        ('predict', 1, TABULAR, 'table'),
        ('indices', 1, TABULAR, 'indices'),
        ('indices', 2, RELATIONAL, 'relational'),
    ],
)
def test_materialize_selects_the_profile_from_the_call_shape(
    monkeypatch, argument, tables, expected_profile, expected_materializer
):
    data = {'users': users()}
    if tables == 2:
        data['orders'] = pd.DataFrame({'order_id': [1], 'user_id': [0]})
    predictive_query = PredictiveQuery(
        'PREDICT users.churn FOR EACH users.user_id', data
    )
    request = object()
    resolved = predictive_query._resolve(expected_profile)
    resolve = Mock(return_value=resolved)
    table = Mock(return_value=request)
    indices = Mock(return_value=request)
    relational = Mock(return_value=request)
    monkeypatch.setattr(predictive_query, '_resolve', resolve)
    monkeypatch.setattr(predictive_query._tabular_materializer, 'table', table)
    monkeypatch.setattr(
        predictive_query._tabular_materializer, 'indices', indices
    )
    monkeypatch.setattr(
        predictive_query._relational_materializer, 'materialize', relational
    )

    arguments = (
        {'predict': new_rows()} if argument == 'predict' else {'indices': [1]}
    )
    assert predictive_query.materialize(**arguments) is request
    resolve.assert_called_once_with(expected_profile)

    materializers = {
        'table': table,
        'indices': indices,
        'relational': relational,
    }
    materializers.pop(expected_materializer).assert_called_once()
    for skipped in materializers.values():
        skipped.assert_not_called()


def test_relational_options_are_forwarded_to_the_relational_materializer(
    monkeypatch,
):
    predictive_query = PredictiveQuery(
        'PREDICT users.churn FOR EACH users.user_id',
        {
            'users': users(),
            'orders': pd.DataFrame({'order_id': [1], 'user_id': [0]}),
        },
    )
    resolved = object()
    request = object()
    monkeypatch.setattr(
        predictive_query, '_resolve', Mock(return_value=resolved)
    )
    materialize = Mock(return_value=request)
    monkeypatch.setattr(
        predictive_query._relational_materializer,
        'materialize',
        materialize,
    )

    assert (
        predictive_query.materialize(indices=[1], num_neighbors=[8, 8])
        is request
    )
    materialize.assert_called_once_with(
        query=resolved,
        indices=[1],
        num_context_samples=None,
        options={'num_neighbors': [8, 8]},
    )


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


@pytest.mark.parametrize('path', ['predict', 'indices'])
@pytest.mark.parametrize('bad', [0, -1, 5.5, '5', True])
def test_a_bad_context_count_is_refused_before_dispatch(path, bad):
    arguments = (
        {'predict': new_rows()} if path == 'predict' else {'indices': [1]}
    )
    with pytest.raises(StructuredDataError):
        query().materialize(**arguments, num_context_samples=bad)


@pytest.mark.parametrize('count', [1, 500, 10_000, 10_001])
def test_a_valid_context_count_reaches_the_tabular_materializer(count):
    predictive_query = query()
    expected = object()

    def _materialize(**kwargs):
        assert kwargs['num_context_samples'] == count
        return expected

    predictive_query._tabular_materializer.indices = _materialize  # type: ignore[method-assign]
    assert (
        predictive_query.materialize(indices=[1], num_context_samples=count)
        is expected
    )


@pytest.mark.parametrize('count', [500, 10_001])
def test_predict_does_not_accept_a_context_count(count):
    with pytest.raises(StructuredDataError) as error:
        query().materialize(
            predict=new_rows(),
            num_context_samples=count,
        )

    message = str(error.value)
    assert 'cannot be used with predict=<frame>' in message
    assert 'build the schema from the rows you want' in message


@pytest.mark.parametrize('path', ['predict', 'indices'])
@pytest.mark.parametrize('option', ['num_neighbors', 'num_neighbours'])
def test_single_table_requests_do_not_accept_options(path, option):
    arguments = (
        {'predict': new_rows()} if path == 'predict' else {'indices': [1]}
    )
    with pytest.raises(StructuredDataError) as error:
        query().materialize(**arguments, **{option: [8, 8]})

    message = str(error.value)
    assert 'apply only to relational materialization' in message
    assert option in message
    assert 'KumoRelational' in message


@pytest.mark.parametrize(
    'text',
    [
        'PREDICT users.churn FOR EACH users.user_id WHERE users.age > 45',
        'PREDICT users.churn FOR EACH users.user_id ASSUMING users.age > 65',
        'PREDICT users.age > 30 FOR EACH users.user_id',
        'PREDICT COUNT(users.*, 0, 30, days) FOR EACH users.user_id',
        'PREDICT SUM(users.age, 0, 30, days) FOR EACH users.user_id',
    ],
)
@pytest.mark.parametrize('path', ['predict', 'indices'])
def test_a_query_the_tabular_path_would_ignore_is_refused(text, path):
    arguments = (
        {'predict': new_rows()} if path == 'predict' else {'indices': [1]}
    )
    with pytest.raises(StructuredDataError) as error:
        query(text).materialize(**arguments)

    assert error.value.code == 'INVALID_REQUEST'
    assert 'a tabular model' in str(error.value)


# --- a quoted name resolves to the column the data has ----------------------


def airfoil(n: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'Row ID': range(n),
            'attack-angle': [float(i) for i in range(n)],
            'scaled-sound-pressure': [100.0 + i for i in range(n)],
        }
    )


QUOTED = (
    'PREDICT `air foil`.`scaled-sound-pressure` FOR EACH `air foil`.`Row ID`'
)


def test_a_quoted_target_is_dropped_from_the_rows_to_score():
    """`target_column` once returned the query's spelling, backticks and all,
    and the drop of the target from the scored rows missed the column. The
    table and entity are quoted too, so the whole name path is exercised."""
    request = PredictiveQuery(QUOTED, {'air foil': airfoil()}).materialize(
        indices=[28, 29]
    )
    assert request.task.target.column_name == 'scaled-sound-pressure'
    assert 'scaled-sound-pressure' not in request.predict.instance_table.columns
    assert 'attack-angle' in request.predict.instance_table.columns


def test_a_quoted_target_names_the_column_rows_supplied_separately_lack():
    rows = airfoil().iloc[:2].drop(columns=['scaled-sound-pressure'])
    request = PredictiveQuery(QUOTED, {'air foil': airfoil()}).materialize(
        predict=rows
    )
    assert request.task.target.column_name == 'scaled-sound-pressure'
    assert request.task.kind == 'regression'
