# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a model refuses is the profile's answer, not a surprise downstream.

Every query here is good PQL. The only thing that decides whether it is
accepted is which profile is asked, which is the whole point of keeping
capability a pass of its own.
"""

from __future__ import annotations

import pytest

from predictive_query import RELATIONAL, TABULAR, validate
from predictive_query.schema import codes
from predictive_query.schema.graph import (
    ColumnKey,
    ColumnKeyGroup,
    GraphDefinition,
)
from predictive_query.schema.table import Column, TableDefinition
from predictive_query.schema.typing import Dtype, Stype


def one_table() -> GraphDefinition:
    """One table, so nothing here is refused for reaching a second one."""
    return GraphDefinition(
        tables={
            'users': TableDefinition(
                cols=[
                    Column('user_id', Stype.ID, Dtype.int64),
                    Column('age', Stype.numerical, Dtype.float32),
                    Column('churn', Stype.categorical, Dtype.string),
                ],
                pkey='user_id',
            )
        },
        col_groups=[],
    )


def two_tables() -> GraphDefinition:
    """The same users, plus the events an aggregation needs.

    An aggregation reaches across a foreign key by definition, and ASSUMING
    is only allowed over a temporal target, so the two-table graph is the only
    one those queries can resolve against at all. A tabular model refuses them
    for more than one reason, and the reason under test is one of them.
    """
    graph = one_table()
    graph.tables['orders'] = TableDefinition(
        cols=[
            Column('order_id', Stype.ID, Dtype.int64),
            Column('user_id', Stype.ID, Dtype.int64),
            Column('price', Stype.numerical, Dtype.float32),
            Column('ordered_at', Stype.timestamp, Dtype.date),
        ],
        pkey='order_id',
        time_col='ordered_at',
    )
    graph.col_groups.append(
        ColumnKeyGroup(
            columns=(
                ColumnKey('orders', 'user_id'),
                ColumnKey('users', 'user_id'),
            )
        )
    )
    return graph


def codes_for(text: str, profile, graph: GraphDefinition) -> set[str]:
    resolved, response = validate(text, graph, profile)
    if response.ok:
        assert resolved is not None
    return {error.code for error in response.errors}


NARROWED = [
    pytest.param(
        'PREDICT users.churn FOR EACH users.user_id WHERE users.age > 45',
        one_table,
        codes.MODEL_NO_ENTITY_FILTER,
        id='where',
    ),
    pytest.param(
        'PREDICT users.age > 30 FOR EACH users.user_id',
        one_table,
        codes.MODEL_NO_DERIVED_TARGET,
        id='condition-target',
    ),
    pytest.param(
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
        two_tables,
        codes.MODEL_NO_DERIVED_TARGET,
        id='count-target',
    ),
    pytest.param(
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
        two_tables,
        codes.MODEL_NO_DERIVED_TARGET,
        id='sum-target',
    ),
    pytest.param(
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id '
        'ASSUMING SUM(orders.price, 0, 30, days) > 10',
        two_tables,
        codes.MODEL_NO_COUNTERFACTUAL,
        id='assuming',
    ),
]


@pytest.mark.parametrize(('text', 'graph', 'code'), NARROWED)
def test_a_tabular_model_refuses_what_it_would_otherwise_ignore(
    text, graph, code
):
    """Each of these resolves, and each is dropped on the floor downstream.

    The tabular materializer reads an entity column, a target column and its
    semantic type. A filter, an ASSUMING clause and a computed target are all
    invisible to it, so a profile that accepted them would answer a different
    question from the one that was asked.
    """
    assert code in codes_for(text, TABULAR, graph())


@pytest.mark.parametrize(('text', 'graph', 'code'), NARROWED)
def test_a_relational_model_takes_all_of_them(text, graph, code):
    """Same text, same graph: a capability narrows a model, not the language."""
    assert codes_for(text, RELATIONAL, graph()) == set()


def test_a_bare_column_is_still_the_query_a_tabular_model_answers():
    text = 'PREDICT users.churn FOR EACH users.user_id'
    assert codes_for(text, TABULAR, one_table()) == set()


def test_a_window_and_a_derived_target_are_reported_as_two_reasons():
    """An aggregation is both, and each names a different thing to fix."""
    reported = codes_for(
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
        TABULAR,
        two_tables(),
    )
    assert codes.MODEL_NO_TEMPORAL_TARGET in reported
    assert codes.MODEL_NO_DERIVED_TARGET in reported
