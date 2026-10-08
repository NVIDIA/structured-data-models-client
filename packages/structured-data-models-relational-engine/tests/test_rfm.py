# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from predictive_query.ast import Column, Condition, Constant
from predictive_query.query import ValidatedPredictiveQuery
from predictive_query.schema.task import TaskType
from predictive_query.schema.typing import Dtype, Stype
from sdm_relational_engine.exceptions import NoEligibleEntityError
from sdm_relational_engine.rfm import (
    Explanation,
    Graph,
    KumoRelational,
    LocalTable,
    TaskTable,
)
from sdm_relational_engine.rfm.base.utils import Timestamp
from sdm_relational_engine.rfm.inference.context import (
    REVERSE_RELATION_PREFIX,
    SampledEdgeLayout,
)
from sdm_relational_engine.rfm.inference.requests import RFMPredictResponse
from sdm_relational_engine.utils.progress_logger import PlainProgressLogger


class MockAPI:
    def predict(
        self,
        request: dict[str, Any],
        *,
        entity_ids: list[Any] | None = None,
        instance_ids: list[Any] | None = None,
        anchor_times: list[Any] | None = None,
    ) -> RFMPredictResponse:
        return RFMPredictResponse(
            prediction={
                'columns': ['ENTITY', 'TRUE_PROB'],
                'data': [[0, 0.15]],
            }
        )

    def create_session(self, request: dict[str, Any]) -> str:
        return 'sess_mock'

    def session_predict(
        self,
        session_id: str,
        request: dict[str, Any],
        *,
        entity_ids: list[Any] | None = None,
        instance_ids: list[Any] | None = None,
        anchor_times: list[Any] | None = None,
    ) -> RFMPredictResponse:
        return self.predict(
            request,
            entity_ids=entity_ids,
            instance_ids=instance_ids,
            anchor_times=anchor_times,
        )

    def delete_session(self, session_id: str) -> None:
        pass


@pytest.mark.parametrize('query_fixture', ['ltv'])
@pytest.mark.parametrize('anchor_time', [Timestamp('2025-01-01')])
@pytest.mark.parametrize('evaluate', [False, True])
def test_temporal_rfm(
    user_store_graph: Graph,
    query_fixture: str,
    anchor_time: pd.Timestamp,
    evaluate: bool,
    request: pytest.FixtureRequest,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    assert str(model) == 'KumoRelational()'

    query = request.getfixturevalue(query_fixture)

    train_table = model.get_train_table(
        query,
        size=4,
        anchor_time=anchor_time,
    )
    pd.testing.assert_frame_equal(
        train_table.sort_values('ENTITY').reset_index(drop=True),
        pd.DataFrame(
            {
                'ENTITY': [0, 1, 2, 3],
                'ANCHOR_TIMESTAMP': pd.Series(
                    [anchor_time] * 4, dtype='datetime64[ns]'
                ),
                'TARGET': pd.Series([0.0, 75.0, 0.0, 95.0], dtype='float32'),
            }
        ),
    )

    with pytest.warns(UserWarning, match='form proper input data'):
        task_table = model._get_task_table(
            query,
            indices=None if evaluate else query.get_rfm_entity_id_list(),
            anchor_time=anchor_time,
        )
        context = model._get_context(task_table)

    assert context.task_type == TaskType.REGRESSION
    assert context.entity_table_names == ('USERS',)

    assert np.array_equal(
        context.sampled_subgraph.anchor_time,
        np.array(
            [(anchor_time - pd.DateOffset(days=7)).value] * 4
            + [anchor_time.value] * 4
        ),
    )

    assert set(context.sampled_subgraph.tables) == {'USERS', 'ORDERS', 'STORES'}

    table = context.sampled_subgraph.tables['USERS']
    pd.testing.assert_frame_equal(
        table.df,
        pd.DataFrame(
            {
                'USER_ID': [0, 1, 2, 3],
                'AGE': [20.0, 30.0, 40.0, None],
                'GENDER': ['male', 'female', 'female', None],
                'STATUS': ['A', 'B', 'A', 'C'],
            }
        ),
        check_like=True,
    )
    assert table.row is not None
    if evaluate:
        assert np.array_equal(table.row, np.array([1, 3, 0, 2, 1, 3, 0, 2]))
    else:
        assert np.array_equal(table.row, np.array([1, 3, 0, 2, 0, 1, 2, 3]))
    assert np.array_equal(table.batch, np.arange(8))
    assert table.num_sampled_nodes == [8, 0, 0]
    assert table.stype_dict == {
        'AGE': Stype.numerical,
        'GENDER': Stype.categorical,
        'STATUS': Stype.categorical,
    }
    assert table.primary_key == 'USER_ID'

    table = context.sampled_subgraph.tables['ORDERS']
    pd.testing.assert_frame_equal(
        table.df,
        pd.DataFrame(
            {
                'AMOUNT': [10.0, 15.0, 10.0],
                'CAT': [10.0, 15.0, 28.0],
                'TIME': pd.to_datetime(
                    [
                        '2025-01-01',
                        '2024-12-20',
                        '2025-01-01',
                    ]
                ).astype('datetime64[ns]'),
            }
        ),
        check_like=True,
    )
    assert table.row is not None
    if evaluate:
        assert np.array_equal(table.row, np.array([1, 2, 1, 0]))
        assert np.array_equal(table.batch, np.array([2, 5, 6, 6]))
    else:
        assert np.array_equal(table.row, np.array([1, 1, 0, 2]))
        assert np.array_equal(table.batch, np.array([2, 4, 4, 7]))
    assert table.num_sampled_nodes == [0, 4, 0]
    assert table.stype_dict == {
        'AMOUNT': Stype.numerical,
        'CAT': Stype.categorical,
        'TIME': Stype.timestamp,
    }
    assert table.primary_key is None

    table = context.sampled_subgraph.tables['STORES']
    pd.testing.assert_frame_equal(
        table.df,
        pd.DataFrame({'CAT': ['burger', 'pizza']}),
    )
    assert table.row is not None
    if evaluate:
        assert np.array_equal(table.row, np.array([1, 0, 1, 0]))
        assert np.array_equal(table.batch, np.array([2, 5, 6, 6]))
    else:
        assert np.array_equal(table.row, np.array([1, 1, 0, 0]))
        assert np.array_equal(table.batch, np.array([2, 4, 4, 7]))
    assert table.num_sampled_nodes == [0, 0, 4]
    assert table.stype_dict == {'CAT': Stype.categorical}
    assert table.primary_key is None

    assert set(context.sampled_subgraph.links) == {
        ('ORDERS', 'USER_ID', 'USERS'),
        ('USERS', f'{REVERSE_RELATION_PREFIX}USER_ID', 'ORDERS'),
        ('STORES', f'{REVERSE_RELATION_PREFIX}STORE_ID', 'ORDERS'),
    }

    link = context.sampled_subgraph.links[('ORDERS', 'USER_ID', 'USERS')]
    assert link.layout == SampledEdgeLayout.COO
    assert link.row is None
    assert link.col is not None
    if evaluate:
        assert np.array_equal(link.col, np.array([2, 5, 6, 6]))
    else:
        assert np.array_equal(link.col, np.array([2, 4, 4, 7]))
    assert link.num_sampled_edges == [4, 0]

    link = context.sampled_subgraph.links[
        ('USERS', f'{REVERSE_RELATION_PREFIX}USER_ID', 'ORDERS')
    ]
    assert link.layout == SampledEdgeLayout.REV
    assert link.row is None
    assert link.col is None
    assert link.num_sampled_edges == [0, 4]

    link = context.sampled_subgraph.links[
        ('STORES', f'{REVERSE_RELATION_PREFIX}STORE_ID', 'ORDERS')
    ]
    assert link.layout == SampledEdgeLayout.COO
    assert link.row is None
    assert link.col is None
    assert link.num_sampled_edges == [0, 4]

    if evaluate:
        pd.testing.assert_series_equal(
            context.y_train,
            pd.Series([0.0, 10.0, 10.0, 0.0], dtype='float32'),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            context.y_test,
            pd.Series([75.0, 95.0, 0.0, 0.0], dtype='float32'),
            check_names=False,
        )
    else:
        pd.testing.assert_series_equal(
            context.y_train,
            pd.Series([0.0, 10.0, 10.0, 0.0], dtype='float32'),
            check_names=False,
        )
        assert context.y_test is None


@pytest.mark.parametrize('query_fixture', ['assuming'])
@pytest.mark.parametrize('anchor_time', [Timestamp('2025-01-01')])
@pytest.mark.parametrize('evaluate', [False, True])
def test_assuming(
    user_store_graph: Graph,
    query_fixture: str,
    anchor_time: pd.Timestamp,
    evaluate: bool,
    request: pytest.FixtureRequest,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    assert str(model) == 'KumoRelational()'

    query = request.getfixturevalue(query_fixture)

    with pytest.warns(UserWarning, match='form proper input data'):
        task_table = model._get_task_table(
            query,
            indices=None if evaluate else query.get_rfm_entity_id_list(),
            anchor_time=anchor_time,
        )
        context = model._get_context(task_table)

    assert context.task_type == TaskType.REGRESSION
    assert context.entity_table_names == ('USERS',)

    n_predict = 2 if evaluate else 4
    n_context = 2
    assert np.array_equal(
        context.sampled_subgraph.anchor_time,
        np.array(
            [(anchor_time - pd.DateOffset(days=7)).value] * n_context
            + [anchor_time.value] * n_predict
        ),
    )

    table = context.sampled_subgraph.tables['USERS']
    expected_df = pd.DataFrame(
        {
            'USER_ID': [0, 1, 2, 3],
            'AGE': [20.0, 30.0, 40.0, None],
            'GENDER': ['male', 'female', 'female', None],
            'STATUS': ['A', 'B', 'A', 'C'],
        }
    )
    if evaluate:
        # 2 has no transactions and is thus filtered out everywhere
        expected_df = expected_df[expected_df['USER_ID'] != 2]
        expected_df = expected_df.reset_index(drop=True)
    pd.testing.assert_frame_equal(table.df, expected_df, check_like=True)

    pd.testing.assert_series_equal(
        context.y_train.reset_index(drop=True),
        pd.Series([10.0, 10.0], dtype='float32'),
        check_names=False,
    )

    if evaluate:
        assert context.y_test is not None
        pd.testing.assert_series_equal(
            context.y_test.reset_index(drop=True),
            pd.Series([75.0, 95.0], dtype='float32'),
            check_names=False,
        )
    else:
        assert context.y_test is None


@pytest.mark.parametrize('query_fixture', ['age'])
@pytest.mark.parametrize('anchor_time', [None])
@pytest.mark.parametrize('evaluate', [False, True])
def test_static_rfm(
    user_store_graph: Graph,
    query_fixture: str,
    anchor_time: pd.Timestamp | None,
    evaluate: bool,
    request: pytest.FixtureRequest,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    assert str(model) == 'KumoRelational()'

    query = request.getfixturevalue(query_fixture)

    train_table = model.get_train_table(
        query,
        size=4,
        anchor_time=anchor_time,
    )
    pd.testing.assert_frame_equal(
        train_table.sort_values('ENTITY').reset_index(drop=True),
        pd.DataFrame(
            {
                'ENTITY': [0, 1, 2],
                'ANCHOR_TIMESTAMP': pd.Series(
                    [Timestamp('2025-01-09')] * 3,
                    dtype='datetime64[ns]',
                ),
                'TARGET': pd.Series([20.0, 30.0, 40.0], dtype='float32'),
            }
        ),
    )

    task_table = model._get_task_table(
        query,
        indices=None if evaluate else query.get_rfm_entity_id_list(),
        anchor_time=anchor_time,
    )
    context = model._get_context(
        task_table,
        exclude_cols_dict=query.get_exclude_cols_dict(),
    )

    assert context.task_type == TaskType.REGRESSION
    assert context.entity_table_names == ('USERS',)

    size = 3 if evaluate else 5
    assert np.array_equal(
        context.sampled_subgraph.anchor_time,
        np.array([Timestamp('2025-01-09').value] * size),
    )

    assert set(context.sampled_subgraph.tables) == {'USERS', 'ORDERS', 'STORES'}

    table = context.sampled_subgraph.tables['USERS']
    assert table.df.shape == (3 if evaluate else 4, 3)
    if evaluate:
        assert table.row is None
    else:
        assert table.row is not None
    assert np.array_equal(table.batch, np.arange(size))
    assert table.num_sampled_nodes == [size, 0, 0]
    assert table.stype_dict == {
        'GENDER': Stype.categorical,
        'STATUS': Stype.categorical,
    }
    assert table.primary_key == 'USER_ID'

    if not evaluate:
        table = context.sampled_subgraph.tables['ORDERS']
        assert table.df.shape == (13, 3)
        assert table.row is None
        assert table.num_sampled_nodes == [0, 13, 0]
        assert table.stype_dict == {
            'AMOUNT': Stype.numerical,
            'CAT': Stype.categorical,
            'TIME': Stype.timestamp,
        }
        assert table.primary_key is None

        table = context.sampled_subgraph.tables['STORES']
        assert table.df.shape == (3, 1)
        assert table.row is not None
        assert len(table.row) == 8
        assert table.num_sampled_nodes == [0, 0, 8]
        assert table.stype_dict == {'CAT': Stype.categorical}
        assert table.primary_key is None

        assert set(context.sampled_subgraph.links) == {
            ('ORDERS', 'USER_ID', 'USERS'),
            ('USERS', f'{REVERSE_RELATION_PREFIX}USER_ID', 'ORDERS'),
            ('STORES', f'{REVERSE_RELATION_PREFIX}STORE_ID', 'ORDERS'),
        }

        link = context.sampled_subgraph.links[('ORDERS', 'USER_ID', 'USERS')]
        assert link.layout == SampledEdgeLayout.CSC
        assert link.row is None
        assert link.col is not None
        assert link.num_sampled_edges == [13, 0]

        link = context.sampled_subgraph.links[
            (
                'USERS',
                f'{REVERSE_RELATION_PREFIX}USER_ID',
                'ORDERS',
            )
        ]
        assert link.layout == SampledEdgeLayout.REV
        assert link.row is None
        assert link.col is None
        assert link.num_sampled_edges == [0, 13]

        link = context.sampled_subgraph.links[
            (
                'STORES',
                f'{REVERSE_RELATION_PREFIX}STORE_ID',
                'ORDERS',
            )
        ]
        assert link.layout == SampledEdgeLayout.COO
        assert link.row is not None
        assert link.col is None
        assert link.num_sampled_edges == [0, 13]

        pd.testing.assert_series_equal(
            context.y_train,
            pd.Series([30.0, 20.0, 40.0], dtype='float32'),
            check_names=False,
        )
        assert context.y_test is None
    else:
        pd.testing.assert_series_equal(
            context.y_train,
            pd.Series([20.0, 40.0], dtype='float32'),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            context.y_test,
            pd.Series([30.0], dtype='float32'),
            check_names=False,
        )


@pytest.mark.parametrize('evaluate', [False])
def test_entity_anchor_time(
    user_store_graph: Graph,
    evaluate: bool,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)

    query = ValidatedPredictiveQuery(
        target_ast=Column(fqn='ORDERS.AMOUNT', stype_maybe=Stype.numerical),
        entity_ast=Column(fqn='ORDERS.ORDER_ID'),
        rfm_entity_ids=Condition(
            target=Column(fqn='ORDERS.ORDER_ID'),
            op='=',
            value=Constant(value='0', dtype_maybe=Dtype.int),
        ),
    )

    task_table = model._get_task_table(
        query,
        indices=None if evaluate else query.get_rfm_entity_id_list(),
        anchor_time='entity',
    )
    context = model._get_context(task_table)

    df = context.sampled_subgraph.tables['ORDERS'].df
    df = df.iloc[context.sampled_subgraph.tables['ORDERS'].row]
    df = df.iloc[: context.sampled_subgraph.batch_size]

    df = pd.merge(
        df.drop(columns='TIME'),
        cast(LocalTable, user_store_graph['ORDERS'])._data,
        how='left',
        on='ORDER_ID',
    )
    batch_time = pd.to_datetime(df['TIME']).astype('datetime64[ns]')
    batch_time = batch_time.astype(int).to_numpy()
    anchor_time = context.sampled_subgraph.anchor_time
    assert np.array_equal(anchor_time, batch_time)


@pytest.mark.parametrize('query_fixture', ['forecast'])
@pytest.mark.parametrize('anchor_time', [Timestamp('2025-01-05')])
@pytest.mark.parametrize('evaluate', [False, True])
def test_forecasting(
    user_store_graph: Graph,
    query_fixture: str,
    anchor_time: pd.Timestamp,
    evaluate: bool,
    request: pytest.FixtureRequest,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    query = request.getfixturevalue(query_fixture)

    task_table = model._get_task_table(
        query,
        indices=None if evaluate else query.get_rfm_entity_id_list(),
        anchor_time=anchor_time,
    )
    context = model._get_context(task_table)

    assert context.task_type == TaskType.FORECASTING
    assert context.step_size == 86400 * 1_000_000_000  # 1 day in nanoseconds
    assert context.num_forecasts == 4

    assert context.y_train.dtype == np.float32

    if evaluate:
        pd.testing.assert_series_equal(
            context.y_train.reset_index(drop=True),
            pd.Series(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    10.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype='float32',
            ),
            check_names=False,
        )
        assert context.y_test is not None
        assert context.y_test.dtype == np.float32
        assert len(context.y_test) == 4  # One per forecast step.
        train_times = context.sampled_subgraph.anchor_time[
            : len(context.y_train)
        ]
        test_times = context.sampled_subgraph.anchor_time[
            len(context.y_train) :
        ]
        assert len(np.intersect1d(train_times, test_times)) == 0
    else:
        pd.testing.assert_series_equal(
            context.y_train.reset_index(drop=True),
            pd.Series(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    10.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype='float32',
            ),
            check_names=False,
        )
        assert context.y_test is None


@pytest.mark.parametrize('evaluate', [False, True])
def test_forecasting_single_step(
    user_store_graph: Graph,
    forecast_single_step: ValidatedPredictiveQuery,
    evaluate: bool,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    anchor_time = Timestamp('2025-01-05')

    task_table = model._get_task_table(
        forecast_single_step,
        indices=None
        if evaluate
        else forecast_single_step.get_rfm_entity_id_list(),
        anchor_time=anchor_time,
    )
    context = model._get_context(task_table)

    assert context.task_type == TaskType.FORECASTING
    assert context.step_size == 86400 * 1_000_000_000  # 1 day in nanoseconds
    assert context.num_forecasts == 1
    assert context.y_train.dtype == np.float32

    if evaluate:
        assert context.y_test is not None
        assert context.y_test.dtype == np.float32
        assert len(context.y_test) == 1  # One per forecast step.
    else:
        assert context.y_test is None


def test_validation(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:

    model = KumoRelational(user_store_graph, verbose=False)

    with pytest.raises(ValueError, match='is before the earliest timestamp'):
        model._get_task_table(
            ltv,
            indices=ltv.get_rfm_entity_id_list(),
            anchor_time=Timestamp('2024-12-19'),
        )

    with pytest.raises(ValueError, match='too early or aggregation'):
        model._get_task_table(
            ltv,
            indices=ltv.get_rfm_entity_id_list(),
            anchor_time=Timestamp('2024-12-26'),
        )

    with pytest.warns(UserWarning, match='later date than the prediction'):
        model._get_task_table(
            ltv,
            indices=ltv.get_rfm_entity_id_list(),
            anchor_time=Timestamp('2024-12-26'),
            context_anchor_time=Timestamp('2024-12-31'),
        )

    with pytest.warns(UserWarning, match='will leak information'):
        model._get_task_table(
            ltv,
            indices=ltv.get_rfm_entity_id_list(),
            anchor_time=Timestamp('2024-12-24'),
            context_anchor_time=Timestamp('2024-12-23'),
        )

    with pytest.warns(UserWarning, match='form proper input data'):
        model._get_task_table(
            ltv,
            indices=ltv.get_rfm_entity_id_list(),
            anchor_time=Timestamp('2025-01-02'),
        )

    with pytest.warns(UserWarning, match='is after the latest timestamp'):
        model._get_task_table(
            ltv,
            indices=ltv.get_rfm_entity_id_list(),
            anchor_time=Timestamp('2025-01-11'),
        )

    with pytest.raises(ValueError, match='is after the latest supported'):
        model._get_task_table(
            ltv,
            indices=None,
            anchor_time=Timestamp('2025-01-03'),
        )

    task_table = model._get_task_table(
        ltv,
        indices=ltv.get_rfm_entity_id_list(),
    )
    with pytest.raises(ValueError, match='more than 6 hops'):
        model._get_context(task_table, num_neighbors=[10] * 7)


def test_get_train_table(
    string_user_graph: Graph,
    churn: ValidatedPredictiveQuery,
) -> None:
    model = KumoRelational(string_user_graph, verbose=False)
    train_table = model.get_train_table(churn, size=3)

    assert len(train_table) == 3
    assert set(train_table['ENTITY']) == {'user_a', 'user_b', 'user_d'}


def test_get_train_table_with_no_labels_is_an_empty_table() -> None:
    r"""No entity passing the filter is an answer of no labels. The sampler's
    error became a ValueError, and the RuntimeError caught here no longer
    matched it, so the caller got the error instead.
    """
    from sdm_relational_engine.rfm.backend.local import LocalTable

    movies = LocalTable(
        df=pd.DataFrame({'MID': range(10), 'GENRES': ['Comedy'] * 10}),
        name='MOVIES',
        primary_key='MID',
    )
    ratings = LocalTable(
        df=pd.DataFrame(
            {
                'RID': range(40),
                'MID': [mid % 10 for mid in range(40)],
                'STARS': [1.0] * 40,
                'TIME': pd.date_range('2020-01-01', periods=40, freq='D'),
            }
        ),
        name='RATING',
        primary_key='RID',
        time_column='TIME',
    )
    graph = Graph([movies, ratings])
    graph.link('RATING', 'MID', 'MOVIES')
    model = KumoRelational(graph, verbose=False)

    def train_table(genre: str) -> pd.DataFrame:
        return model.get_train_table(
            f'PREDICT COUNT(RATING.STARS, 0, 7, days) FOR EACH MOVIES.MID '
            f"WHERE MOVIES.GENRES = '{genre}'",
            size=3,
            anchor_time=pd.Timestamp('2020-01-10'),
        )

    assert len(train_table('Comedy')) == 3
    empty = train_table('comedy')
    assert len(empty) == 0
    assert list(empty.columns) == ['ENTITY', 'ANCHOR_TIMESTAMP', 'TARGET']


def test_get_train_table_with_no_entity_at_the_anchor_raises() -> None:
    r"""An entity table whose time column starts after the anchor has no
    entity to ask about, which is a misconfigured column rather than an
    answer of no labels, so it reaches the caller instead of reading as an
    empty table.
    """
    from sdm_relational_engine.rfm.backend.local import LocalTable

    movies = LocalTable(
        df=pd.DataFrame(
            {
                'MID': range(10),
                'ADDED': pd.date_range('2021-01-01', periods=10, freq='D'),
            }
        ),
        name='MOVIES',
        primary_key='MID',
        time_column='ADDED',
    )
    ratings = LocalTable(
        df=pd.DataFrame(
            {
                'RID': range(40),
                'MID': [mid % 10 for mid in range(40)],
                'STARS': [1.0] * 40,
                'TIME': pd.date_range('2020-01-01', periods=40, freq='D'),
            }
        ),
        name='RATING',
        primary_key='RID',
        time_column='TIME',
    )
    graph = Graph([movies, ratings])
    graph.link('RATING', 'MID', 'MOVIES')
    model = KumoRelational(graph, verbose=False)

    with pytest.raises(NoEligibleEntityError, match='No row of'):
        model.get_train_table(
            'PREDICT COUNT(RATING.STARS, 0, 7, days) FOR EACH MOVIES.MID',
            size=3,
            anchor_time=pd.Timestamp('2020-01-10'),
        )


def test_context_anchor_time(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)

    with pytest.warns(UserWarning, match='form proper input data'):
        task_table = model._get_task_table(
            ltv,
            indices=ltv.get_rfm_entity_id_list(),
            anchor_time=Timestamp('2025-01-01'),
            context_anchor_time=Timestamp('2024-12-24'),
        )
        context = model._get_context(task_table)

    pd.testing.assert_series_equal(
        context.y_train,
        pd.Series([0.0, 0, 0, 0.0], dtype='float32'),
        check_names=False,
    )
    assert context.y_test is None


def test_indices(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    model._client = MockAPI()  # type: ignore
    df = model.predict(ltv, indices=[0, 1, 2, 3], verbose=False)
    assert len(df) == 1


def test_batch_mode(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    model._client = MockAPI()  # type: ignore
    with model.batch_mode(batch_size=3):
        df = model.predict(ltv, indices=[0, 1, 2, 3], verbose=True)
    assert len(df) == 2


def test_optimize_warns_on_a_backend_that_ignores_it(
    user_store_graph: Graph,
) -> None:
    with pytest.warns(UserWarning, match="'optimize=True' has no effect"):
        KumoRelational(user_store_graph, verbose=False, optimize=True)


def test_optimize_left_unset_does_not_warn(user_store_graph: Graph) -> None:
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        KumoRelational(user_store_graph, verbose=False)

    assert [
        str(item.message) for item in caught if 'optimize' in str(item.message)
    ] == []


def test_batch_mode_restores_state_after_an_exception(
    user_store_graph: Graph,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    with pytest.raises(RuntimeError, match='user code blew up'):
        with model.batch_mode(batch_size=25, num_retries=3):
            raise RuntimeError('user code blew up')
    assert model._batch_size.get() is None
    assert model._num_retries.get() == 0


def test_retry_restores_state_after_an_exception(
    user_store_graph: Graph,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    with pytest.raises(RuntimeError), model.retry(7):
        raise RuntimeError('boom')
    assert model._num_retries.get() == 0


def test_nested_batch_mode_honours_the_inner_retry_count(
    user_store_graph: Graph,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    with model.retry(2):
        assert model._num_retries.get() == 2
        with model.batch_mode(batch_size=3, num_retries=9):
            assert model._num_retries.get() == 9
        assert model._num_retries.get() == 2
    assert model._num_retries.get() == 0


def test_batch_mode_rejects_a_non_max_string(user_store_graph: Graph) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    with pytest.raises(ValueError, match="'batch_size' must be a positive int"):
        with model.batch_mode(batch_size='MAX'):
            pass


@pytest.mark.parametrize('anchor_time', ['2024-10-01', 1_700_000_000])
def test_predict_rejects_a_non_timestamp_anchor_time(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
    anchor_time: Any,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    model._client = MockAPI()  # type: ignore
    with pytest.raises(
        TypeError, match="'anchor_time' must be a pandas.Timestamp"
    ):
        model.predict(
            ltv, indices=[0, 1], anchor_time=anchor_time, verbose=False
        )


def test_predict_anchor_time_string_hint_names_the_timestamp_call(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    model._client = MockAPI()  # type: ignore
    with pytest.raises(TypeError, match="pd.Timestamp\\('2024-10-01'\\)"):
        model.predict(
            ltv, indices=[0, 1], anchor_time='2024-10-01', verbose=False
        )


@pytest.mark.parametrize('max_pq_iterations', [0, -5])
def test_predict_rejects_non_positive_max_pq_iterations(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
    max_pq_iterations: int,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    model._client = MockAPI()  # type: ignore
    with pytest.raises(
        ValueError, match="'max_pq_iterations' must be greater than zero"
    ):
        model.predict(
            ltv,
            indices=[0, 1],
            max_pq_iterations=max_pq_iterations,
            verbose=False,
        )


@pytest.mark.parametrize('num_hops', [-1, 0, 100])
def test_predict_rejects_out_of_range_num_hops(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
    num_hops: int,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    model._client = MockAPI()  # type: ignore
    with pytest.raises(ValueError, match="'num_hops' must be between 1 and"):
        model.predict(ltv, indices=[0, 1], num_hops=num_hops, verbose=False)


def test_regression_quantile_output_config(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    columns = ['ENTITY', 'q_0.005', 'q_0.025', 'q_0.5', 'q_0.975', 'q_0.995']
    captured_config = None

    class MockQuantileAPI:
        def predict(
            self,
            request: dict[str, Any],
            *,
            entity_ids: list[Any] | None = None,
            instance_ids: list[Any] | None = None,
            anchor_times: list[Any] | None = None,
        ) -> RFMPredictResponse:
            nonlocal captured_config
            captured_config = request['inference']['inference_config']
            return RFMPredictResponse(
                prediction={
                    'columns': columns,
                    'data': [[0, 0.5, 2.5, 50.0, 97.5, 99.5]],
                }
            )

    model = KumoRelational(user_store_graph, verbose=False)
    model._client = MockQuantileAPI()  # type: ignore

    df = model.predict(
        ltv,
        indices=[0],
        inference_config={'output_type': 'quantiles'},
        verbose=False,
    )

    assert captured_config['kind'] == 'regression'
    assert captured_config['output_type'] == 'quantiles'
    assert list(df.columns) == columns


def test_forecasting_single_entity_validation(
    user_store_graph: Graph,
    forecast: ValidatedPredictiveQuery,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    with pytest.raises(ValueError, match='single entity ID'):
        model._get_task_table(
            forecast,
            indices=[0, 1],
            anchor_time=Timestamp('2025-01-05'),
        )


def test_end_time_column() -> None:
    df = pd.DataFrame(
        {
            'USER_ID': [0, 1, 2, 3, 4, 5],
            'END': pd.date_range('2023-01-01', periods=6),
            'Y': [0, 1, 0, 1, 0, 1],
        }
    )
    graph = Graph.from_data({'USERS': df}, verbose=False)
    graph['USERS'].time_column = None
    graph['USERS'].end_time_column = 'END'

    model = KumoRelational(graph, verbose=False)

    query = ValidatedPredictiveQuery(
        target_ast=Condition(
            target=Column(fqn='USERS.Y'),
            op='=',
            value=Constant(value='1', dtype_maybe=Dtype.int),
        ),
        entity_ast=Column(fqn='USERS.USER_ID'),
    )

    task_table = model._get_task_table(
        query,
        indices=[0],
        anchor_time=Timestamp('2023-01-04'),
    )
    context = model._get_context(
        task_table,
        exclude_cols_dict=query.get_exclude_cols_dict(),
    )

    assert context.sampled_subgraph.tables['USERS'].num_rows == 3
    assert context.sampled_subgraph.tables['USERS'].row is None
    df = context.sampled_subgraph.tables['USERS'].df
    df = df.sort_values('USER_ID').reset_index(drop=True)
    pd.testing.assert_frame_equal(
        df,
        pd.DataFrame(
            {
                'USER_ID': [0, 4, 5],
                'END': pd.Series(
                    [Timestamp('2023-01-01'), pd.NaT, pd.NaT],
                    dtype='datetime64[ns]',
                ),
            }
        ),
        check_like=True,
    )


@pytest.mark.parametrize(
    'ltv_offset', [(1, 1), (2, 1), (1, 2), (2, 2)], indirect=True
)
@pytest.mark.parametrize('anchor_time', [Timestamp('2025-01-01')])
@pytest.mark.parametrize('evaluate', [False, True])
def test_query_start(
    user_store_graph: Graph,
    ltv_offset: ValidatedPredictiveQuery,
    anchor_time: pd.Timestamp,
    evaluate: bool,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)

    task_table = model._get_task_table(
        ltv_offset,
        indices=None if evaluate else ltv_offset.get_rfm_entity_id_list(),
        anchor_time=anchor_time,
    )
    context = model._get_context(task_table)

    earliest_timestamp = pd.to_datetime(
        cast(LocalTable, user_store_graph['ORDERS'])._data['TIME']
    ).min()
    anchor_times = pd.to_datetime(context.sampled_subgraph.anchor_time)
    context_anchor_times = anchor_times[: len(context.y_train)]
    unique_anchor_times = context_anchor_times.unique().sort_values()
    target_timeframe = ltv_offset.target_timeframe
    assert target_timeframe is not None
    assert target_timeframe.start is not None
    assert target_timeframe.end is not None
    steps = target_timeframe.end - target_timeframe.start
    expected_anchor_times = pd.date_range(
        start=anchor_time - pd.DateOffset(days=target_timeframe.end),
        end=earliest_timestamp,
        freq=f'-{steps}D',
        unit='ns',
    )[::-1]

    pd.testing.assert_series_equal(
        pd.Series(unique_anchor_times),
        pd.Series(expected_anchor_times),
    )


def _make_forecast_task(num_context: int, num_forecasts: int) -> TaskTable:
    context_df = pd.DataFrame(
        {
            'ENTITY': [0] * num_context,
            'TARGET': pd.Series([10.0] * num_context, dtype='float32'),
            'ANCHOR_TIMESTAMP': pd.date_range(
                '2025-01-01', periods=num_context, freq='D'
            ),
        }
    )
    pred_df = pd.DataFrame(
        {
            'ENTITY': [0],
            'ANCHOR_TIMESTAMP': [Timestamp('2025-06-01')],
        }
    )
    return TaskTable(
        task_type=TaskType.FORECASTING,
        context_df=context_df,
        pred_df=pred_df,
        entity_table_name='USERS',
        entity_column='ENTITY',
        target_column='TARGET',
        time_column='ANCHOR_TIMESTAMP',
        num_forecasts=num_forecasts,
    )


def test_num_forecasts_exceeds_context_predict(user_store_graph: Graph) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    model._client = MockAPI()  # type: ignore
    task = _make_forecast_task(num_context=3, num_forecasts=5)
    with pytest.raises(ValueError, match='number of forecast steps'):
        model.predict_task(task, verbose=False, use_prediction_time=True)


def test_custom_task_features(user_store_graph: Graph) -> None:
    context_df = pd.DataFrame(
        {
            'ENTITY': [0, 0, 0],
            'TARGET': pd.Series([10.0, 15.0, 20.0], dtype='float32'),
            'ANCHOR_TIMESTAMP': pd.date_range(
                '2025-01-01', periods=3, freq='D'
            ),
            'FEAT_A': ['A', 'B', 'C'],
        }
    )
    pred_df = pd.DataFrame(
        {
            'ENTITY': [0, 0, 0, 0, 0],
            'TARGET': pd.Series([5.0, 8.0, 12.0, 6.0, 9.0], dtype='float32'),
            'ANCHOR_TIMESTAMP': pd.date_range(
                '2025-02-01', periods=5, freq='D'
            ),
            'FEAT_A': ['D', 'E', 'F', 'G', 'H'],
        }
    )
    task = TaskTable(
        task_type=TaskType.FORECASTING,
        context_df=context_df,
        pred_df=pred_df,
        entity_table_name='USERS',
        entity_column='ENTITY',
        target_column='TARGET',
        time_column='ANCHOR_TIMESTAMP',
    )

    model = KumoRelational(user_store_graph, verbose=False)
    context = model._get_context(task)
    assert context.task_table is not None
    pd.testing.assert_frame_equal(
        context.task_table.df,
        pd.DataFrame({'FEAT_A': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']}),
    )
    assert context.task_table.row is None
    assert np.array_equal(context.task_table.batch, np.arange(8))
    assert context.task_table.num_sampled_nodes == []
    assert context.task_table.stype_dict == {'FEAT_A': Stype.categorical}
    assert context.task_table.primary_key is None


def test_lag_timesteps(
    user_store_graph: Graph,
    churn: ValidatedPredictiveQuery,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)

    task_table = model._get_task_table(
        churn,
        indices=churn.get_rfm_entity_id_list(),
        anchor_time=Timestamp('2025-01-05'),
        lag_timesteps=4,
    )
    pred_df = task_table._pred_df.sort_values('ENTITY').reset_index(drop=True)

    churn.target_ast = churn.target_ast.target  # type: ignore
    for i in range(1, 5):
        df = model.get_train_table(
            churn,
            size=4,
            anchor_time=Timestamp('2025-01-05') - pd.DateOffset(days=i),
        )
        df = df.sort_values('ENTITY').reset_index(drop=True)
        pd.testing.assert_series_equal(
            df['TARGET'],
            pred_df[f'__kumo_arl{i - 1}__'],
            check_names=False,
        )


@pytest.mark.parametrize('lag_timesteps', [-1, -4])
def test_a_negative_lag_is_refused_by_get_task_table(
    user_store_graph: Graph,
    churn: ValidatedPredictiveQuery,
    lag_timesteps: int,
) -> None:
    r"""A negative count reached the sampler and produced empty lag columns
    rather than an error, so the caller got a silently featureless task.
    """
    model = KumoRelational(user_store_graph, verbose=False)

    with pytest.raises(ValueError, match='cannot be negative'):
        model._get_task_table(
            churn,
            indices=churn.get_rfm_entity_id_list(),
            anchor_time=Timestamp('2025-01-05'),
            lag_timesteps=lag_timesteps,
        )


def test_zero_lags_is_accepted_by_get_task_table(
    user_store_graph: Graph,
    churn: ValidatedPredictiveQuery,
) -> None:
    r"""Zero means "no lagged features", which is a valid request."""
    model = KumoRelational(user_store_graph, verbose=False)

    task_table = model._get_task_table(
        churn,
        indices=churn.get_rfm_entity_id_list(),
        anchor_time=Timestamp('2025-01-05'),
        lag_timesteps=0,
    )
    assert not [c for c in task_table._pred_df.columns if 'arl' in c]


@pytest.mark.parametrize('lag_timesteps', [0, -1])
def test_add_lagged_target_requires_a_positive_count(
    user_store_graph: Graph,
    churn: ValidatedPredictiveQuery,
    lag_timesteps: int,
) -> None:
    r"""Unlike ``_get_task_table``, asking to add lags needs at least one:
    adding zero of them is a no-op the caller did not intend.
    """
    model = KumoRelational(user_store_graph, verbose=False)
    task_table = model._get_task_table(
        churn,
        indices=churn.get_rfm_entity_id_list(),
        anchor_time=Timestamp('2025-01-05'),
    )

    with pytest.raises(ValueError, match='needs to be positive'):
        model.add_lagged_target(task_table, churn, lag_timesteps)


# --- Explanation warning tests -------------------------------------------


def test_explanation_warning_display():
    """Warning appears in all display paths when set; absent when None."""
    warning_msg = 'Cross-region fallback used.'
    prediction = pd.DataFrame({'ENTITY': [1], 'SCORE': [0.9]})

    with_warning = Explanation(
        prediction=prediction,
        summary='s',
        details=MagicMock(),
        warning=warning_msg,
    )
    no_warning = Explanation(
        prediction=prediction, summary='s', details=MagicMock(), warning=None
    )

    assert warning_msg in str(with_warning)
    assert 'Warning' not in str(no_warning)

    with (
        patch(
            'sdm_relational_engine.rfm.explain.result.in_notebook',
            return_value=True,
        ),
        patch(
            'sdm_relational_engine.rfm.explain.result.display'
        ) as mock_display,
    ):
        with_warning.print()
        calls = [str(c) for c in mock_display.message.call_args_list]
        assert any(warning_msg in c for c in calls)

        mock_display.reset_mock()
        no_warning.print()
        calls = [str(c) for c in mock_display.message.call_args_list]
        assert not any('Warning' in c for c in calls)


def test_explanation_warning_flows_from_api_response(
    user_store_graph: Graph, ltv: ValidatedPredictiveQuery
) -> None:
    """Warning from predictions[].explanation is surfaced in Explanation."""
    mock_resp = RFMPredictResponse(
        prediction={
            'columns': ['ENTITY', 'SCORE', 'EXPLANATION'],
            'data': [
                [
                    1,
                    0.9,
                    {
                        'format': 'natural_language_summary',
                        'summary': 'Summary.',
                        'warning': 'Cross-region fallback used.',
                    },
                ]
            ],
        }
    )

    class MockPredictAPI:
        def predict(
            self,
            request: dict[str, Any],
            *,
            entity_ids: list[Any] | None = None,
            instance_ids: list[Any] | None = None,
            anchor_times: list[Any] | None = None,
        ) -> RFMPredictResponse:
            return mock_resp

    model = KumoRelational(user_store_graph, verbose=False)
    model._client = MockPredictAPI()  # type: ignore
    result = model.predict(ltv, indices=[0], explain=True, verbose=False)
    assert isinstance(result, Explanation)
    assert result.summary == 'Summary.'
    assert result.details['format'] == 'natural_language_summary'
    assert result.warning == 'Cross-region fallback used.'


class MockMulticlassAPI(MockAPI):
    r"""Returns a multi-class frame whose CLASS values are strings, exactly as
    the wire format delivers them.
    """

    def predict(
        self,
        request: dict[str, Any],
        *,
        entity_ids: list[Any] | None = None,
        instance_ids: list[Any] | None = None,
        anchor_times: list[Any] | None = None,
    ) -> RFMPredictResponse:
        return RFMPredictResponse(
            prediction={
                'columns': ['ENTITY', 'CLASS', 'SCORE', 'PREDICTED'],
                'data': [
                    [0, '2', 0.7, True],
                    [0, '1', 0.3, False],
                ],
            }
        )


def test_multiclass_class_column_keeps_the_target_dtype(
    user_store_graph: Graph,
) -> None:
    # Regression: CLASS came back as str while ENTITY in the same frame was
    # int64, so `df['CLASS'] == 2` was False everywhere and a merge back to the
    # source table raised.
    task = TaskTable(
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        context_df=pd.DataFrame(
            {
                'ENTITY': [0, 1, 3, 0],
                'TARGET': pd.Series([1, 2, 1, 2], dtype='int64'),
                'ANCHOR_TIMESTAMP': pd.to_datetime(['2025-01-05'] * 4),
            }
        ),
        pred_df=pd.DataFrame(
            {
                'ENTITY': [0],
                'ANCHOR_TIMESTAMP': pd.to_datetime(['2025-01-05']),
            }
        ),
        entity_table_name='USERS',
        entity_column='ENTITY',
        target_column='TARGET',
        time_column='ANCHOR_TIMESTAMP',
    )

    model = KumoRelational(user_store_graph, verbose=False)
    model._client = MockMulticlassAPI()  # type: ignore
    df = model.predict_task(task, verbose=False)

    assert df['CLASS'].dtype == df['ENTITY'].dtype
    assert df['CLASS'].tolist() == [2, 1]
    assert (df['CLASS'] == 2).sum() == 1


class MockRankingAPI(MockAPI):
    def predict(
        self,
        request: dict[str, Any],
        *,
        entity_ids: list[Any] | None = None,
        instance_ids: list[Any] | None = None,
        anchor_times: list[Any] | None = None,
    ) -> RFMPredictResponse:
        return RFMPredictResponse(
            prediction={
                'columns': ['ENTITY', 'CLASS', 'SCORE'],
                'data': [[0, '1', 0.9], [0, '2', 0.5]],
            }
        )


def test_link_prediction_class_column_keeps_the_target_key_dtype(
    user_store_graph: Graph,
) -> None:
    # Regression: ranking ids arrive as strings, so a merge back to the
    # destination table raised 'You are trying to merge on str and int64
    # columns'.
    task = TaskTable(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        context_df=pd.DataFrame(
            {
                'ENTITY': [0, 1, 3, 0],
                'TARGET': [[0, 1], [1, 2], [0, 2], [1]],
                'ANCHOR_TIMESTAMP': pd.to_datetime(['2025-01-05'] * 4),
            }
        ),
        pred_df=pd.DataFrame(
            {
                'ENTITY': [0],
                'ANCHOR_TIMESTAMP': pd.to_datetime(['2025-01-05']),
            }
        ),
        entity_table_name=('USERS', 'STORES'),
        entity_column='ENTITY',
        target_column='TARGET',
        time_column='ANCHOR_TIMESTAMP',
    )

    model = KumoRelational(user_store_graph, verbose=False)
    model._client = MockRankingAPI()  # type: ignore
    df = model.predict_task(task, verbose=False)

    assert df['CLASS'].dtype == user_store_graph['STORES']['STORE_ID'].dtype
    assert df['CLASS'].tolist() == [1, 2]


def test_over_cap_batch_size_suggests_a_value_under_the_cap() -> None:
    """The message hard-coded ``batch_size=500`` as its worked example. Temporal
    link prediction caps at 200, so following the suggestion verbatim
    reproduced the identical error including the identical suggestion. Checked
    for every task type, since the cap is per task type.
    """
    from sdm_relational_engine.rfm.rfm import _MAX_PRED_SIZE

    for task_type in TaskType:
        limit = _MAX_PRED_SIZE[task_type]
        message = _over_cap_message(task_type, limit + 1)
        suggested = int(re.search(r'`batch_size=(\d+)`', message).group(1))
        assert suggested <= limit, task_type
        assert f'more than {limit:,} entities' in message
        assert "`batch_size='max'`" in message


def _over_cap_message(task_type: TaskType, batch_size: int) -> str:
    r"""Drive the real raise site, so the assertion is about shipped text."""
    graph = Graph.from_data(
        {
            'users': pd.DataFrame({'user_id': [0, 1], 'age': [30, 40]}),
        },
        verbose=False,
    )
    model = KumoRelational(graph, verbose=False)
    model._validate_task_references = lambda _task: None
    model._batch_size.set(batch_size)

    task = MagicMock()
    task.task_type = task_type
    task.num_context_examples = 8
    task.num_prediction_examples = batch_size
    task.num_forecasts = 1
    task.narrow_context.return_value = task

    with pytest.raises(ValueError) as excinfo:
        next(
            model._iter_task_requests(
                task,
                explain=False,
                return_embeddings=False,
                run_mode='fast',
                num_neighbors=[8, 8],
                inference_config=MagicMock(),
                logger=PlainProgressLogger('Predicting', verbose=False),
                exclude_cols_dict=None,
                use_prediction_time=False,
                top_k=None,
                random_seed=42,
            )
        )
    return str(excinfo.value)
