# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

from __future__ import annotations

import copy
from collections.abc import Generator
from typing import TYPE_CHECKING

import pandas as pd
import pytest
import requests_mock
from predictive_query.ast import (
    Aggregation,
    Column,
    Condition,
    Constant,
    DateOffsetRange,
)
from predictive_query.query import ValidatedPredictiveQuery
from predictive_query.schema.typing import (
    AggregationType,
    Dtype,
    MemberOp,
    ProblemType,
    RelOp,
    Stype,
)
from sdm_api.endpoints import Endpoint, HTTPMethod

if TYPE_CHECKING:
    import sdm_relational_engine.rfm as rfm

# Not mock:// due to https://stackoverflow.com/a/76056002
MOCK_URL = 'https://nim.test'


def pytest_addoption(parser):
    parser.addoption(
        '--runintegration',
        action='store_true',
        default=False,
        help='run integration tests',
    )


def pytest_collection_modifyitems(config, items):
    # check if you got an option like --key=snowflake
    if not config.getoption('--runintegration'):
        skip_integ = pytest.mark.skip(reason='integration test')
        for item in items:
            if 'integration' in item.keywords:
                item.add_marker(skip_integ)
    else:
        skip_integ = pytest.mark.skip(reason='no integration test')
        for item in items:
            if 'integration' not in item.keywords:
                item.add_marker(skip_integ)


@pytest.fixture(scope='class')
def mock_api() -> Generator[requests_mock.Mocker, None, None]:
    with requests_mock.Mocker() as m:
        yield m


@pytest.fixture(scope='session')
def _user_store_graph_template() -> rfm.Graph:
    import sdm_relational_engine.rfm as rfm

    df_dict = {}
    df_dict['USERS'] = pd.DataFrame(
        {
            'USER_ID': [0, 1, 2, 3],
            'AGE': [20, 30, 40, float('NaN')],
            'GENDER': ['male', 'female', 'female', None],
            'STATUS': ['A', 'B', 'A', 'C'],
        }
    )
    df_dict['ORDERS'] = pd.DataFrame(
        {
            'ORDER_ID': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'USER_ID': [0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3],
            'STORE_ID': [0, 1, 0, 1, 2, 2, 0, 1, 2, 0, 1, 1, 2],
            'AMOUNT': [
                10,
                15,
                float('NaN'),
                20,
                25,
                30,
                10,
                25,
                20,
                10,
                15,
                15,
                20,
            ],
            'CAT': [
                10,
                15,
                float('NaN'),
                20,
                25,
                30,
                11,
                26,
                27,
                28,
                29,
                32,
                33,
            ],
            'TIME': [
                '2025-01-01',
                '2024-12-20',
                '2025-01-03',
                '2025-01-02',
                '2025-01-03',
                '2025-01-04',
                '2025-01-09',
                '2025-01-02',
                '2025-01-02',
                '2025-01-01',
                '2025-01-02',
                '2025-01-03',
                '2025-01-04',
            ],
        }
    )
    df_dict['STORES'] = pd.DataFrame(
        {
            'STORE_ID': [0, 1, 2],
            'CAT': ['burger', 'pizza', 'fries'],
        }
    )

    graph = rfm.Graph.from_data(df_dict, verbose=False)
    graph['USERS']['AGE'].stype = Stype.numerical
    graph['ORDERS']['AMOUNT'].stype = Stype.numerical
    return graph


@pytest.fixture(scope='session')
def _string_user_graph_template() -> rfm.Graph:
    import sdm_relational_engine.rfm as rfm

    df_dict = {}
    df_dict['USERS'] = pd.DataFrame(
        {
            'USER_ID': ['user_a', 'user_b', 'user_c', 'user_d'],
            'NAME': ['Alice', 'Bob', 'Charlie', 'David'],
        }
    )
    df_dict['ORDERS'] = pd.DataFrame(
        {
            'ORDER_ID': [0, 1, 2, 3, 4],
            'USER_ID': ['user_a', 'user_a', 'user_b', 'user_c', 'user_d'],
            'TIME': [
                '2024-01-01',
                '2025-01-02',
                '2025-01-03',
                '2025-01-04',
                '2025-01-05',
            ],
        }
    )

    return rfm.Graph.from_data(df_dict, verbose=False)


@pytest.fixture()
def user_store_graph(_user_store_graph_template: rfm.Graph) -> rfm.Graph:
    r"""A copy of the graph, which is built once for the session.

    Building it costs about thirty times what copying it does, and 139 cases
    ask for one. Every test still gets its own, which is why this is
    function-scoped: several of them mutate what they are given.
    """
    return copy.deepcopy(_user_store_graph_template)


@pytest.fixture()
def string_user_graph(_string_user_graph_template: rfm.Graph) -> rfm.Graph:
    r"""A copy of the graph, for the reason above."""
    return copy.deepcopy(_string_user_graph_template)


@pytest.fixture()
def churn() -> ValidatedPredictiveQuery:
    return ValidatedPredictiveQuery(
        target_ast=Condition(
            target=Aggregation(
                aggr=AggregationType.COUNT,
                target=Column(fqn='ORDERS.*'),
                aggr_time_range=DateOffsetRange(0, 1),
            ),
            op=RelOp.EQ,
            value=0,
        ),
        entity_ast=Column(fqn='USERS.USER_ID'),
        rfm_entity_ids=Condition(
            target=Column(fqn='USERS.USER_ID'),
            op=MemberOp.IN,
            value=Constant.from_value([0, 1, 2, 3]),
        ),
    )


@pytest.fixture()
def ltv() -> ValidatedPredictiveQuery:
    return ValidatedPredictiveQuery(
        target_ast=Aggregation(
            aggr=AggregationType.SUM,
            target=Column(fqn='ORDERS.AMOUNT'),
            aggr_time_range=DateOffsetRange(0, 7),
        ),
        entity_ast=Column(fqn='USERS.USER_ID'),
        rfm_entity_ids=Condition(
            target=Column(fqn='USERS.USER_ID'),
            op=MemberOp.IN,
            value=Constant.from_value([0, 1, 2, 3]),
        ),
    )


@pytest.fixture()
def forecast() -> ValidatedPredictiveQuery:
    q = ValidatedPredictiveQuery(
        target_ast=Aggregation(
            aggr=AggregationType.SUM,
            target=Column(fqn='ORDERS.AMOUNT'),
            aggr_time_range=DateOffsetRange(0, 1),
        ),
        entity_ast=Column(fqn='USERS.USER_ID'),
        rfm_entity_ids=Condition(
            target=Column(fqn='USERS.USER_ID'),
            op=MemberOp.IN,
            value=Constant.from_value([0]),
        ),
        problem_type=ProblemType.FORECAST,
    )
    q.num_forecasts = 4
    return q


@pytest.fixture()
def forecast_single_step() -> ValidatedPredictiveQuery:
    q = ValidatedPredictiveQuery(
        target_ast=Aggregation(
            aggr=AggregationType.SUM,
            target=Column(fqn='ORDERS.AMOUNT'),
            aggr_time_range=DateOffsetRange(0, 1),
        ),
        entity_ast=Column(fqn='USERS.USER_ID'),
        rfm_entity_ids=Condition(
            target=Column(fqn='USERS.USER_ID'),
            op=MemberOp.IN,
            value=Constant.from_value([0]),
        ),
        problem_type=ProblemType.FORECAST,
    )
    q.num_forecasts = 1
    return q


@pytest.fixture()
def age() -> ValidatedPredictiveQuery:
    return ValidatedPredictiveQuery(
        target_ast=Column(fqn='USERS.AGE', stype_maybe=Stype.numerical),
        entity_ast=Column(fqn='USERS.USER_ID'),
        rfm_entity_ids=Condition(
            target=Column(fqn='USERS.USER_ID'),
            op=MemberOp.IN,
            value=Constant.from_value([2, 3]),
        ),
    )


@pytest.fixture()
def assuming() -> ValidatedPredictiveQuery:
    return ValidatedPredictiveQuery(
        target_ast=Aggregation(
            aggr=AggregationType.SUM,
            target=Column(fqn='ORDERS.AMOUNT'),
            aggr_time_range=DateOffsetRange(0, 7),
        ),
        entity_ast=Column(fqn='USERS.USER_ID'),
        rfm_entity_ids=Condition(
            target=Column(fqn='USERS.USER_ID'),
            op=MemberOp.IN,
            value=Constant.from_value([0, 1, 2, 3]),
        ),
        whatif_ast=Condition(
            target=Aggregation(
                aggr=AggregationType.SUM,
                target=Column(fqn='ORDERS.AMOUNT'),
                aggr_time_range=DateOffsetRange(0, 7),
            ),
            op=RelOp.GEQ,
            value=Constant(value='10', dtype_maybe=Dtype.int),
        ),
    )


@pytest.fixture()
def ltv_offset(request: pytest.FixtureRequest) -> ValidatedPredictiveQuery:
    start, steps = request.param if hasattr(request, 'param') else (0, 1)
    return ValidatedPredictiveQuery(
        target_ast=Aggregation(
            aggr=AggregationType.SUM,
            target=Column(fqn='ORDERS.AMOUNT'),
            aggr_time_range=DateOffsetRange(start, start + steps),
        ),
        entity_ast=Column(fqn='USERS.USER_ID'),
        rfm_entity_ids=Condition(
            target=Column(fqn='USERS.USER_ID'),
            op=MemberOp.IN,
            value=Constant.from_value([0, 1, 2, 3]),
        ),
    )


def get_mock_method(mock_api, endpoint: Endpoint):
    method_map = {
        HTTPMethod.GET: mock_api.get,
        HTTPMethod.POST: mock_api.post,
        HTTPMethod.DELETE: mock_api.delete,
    }
    if endpoint.method not in method_map:
        raise ValueError(f'Unsupported HTTP method: {endpoint.method}')

    return method_map[endpoint.method]
