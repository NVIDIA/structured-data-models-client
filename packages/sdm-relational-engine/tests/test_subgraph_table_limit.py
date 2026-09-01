# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""The subgraph table ceiling restates a server limit, so it must agree with
it exactly: a subgraph of precisely ``_MAX_SUBGRAPH_TABLES`` tables is what
the NIM accepts, and refusing it locally would deny a supported graph.
"""

from __future__ import annotations

import pandas as pd
import pytest
from predictive_query.ast import (
    Aggregation,
    Column,
    Condition,
    Constant,
    DateOffsetRange,
)
from predictive_query.query import ValidatedPredictiveQuery
from predictive_query.schema.typing import AggregationType, MemberOp, RelOp
from sdm_relational_engine.rfm import Graph, KumoRelational
from sdm_relational_engine.rfm.rfm import _MAX_SUBGRAPH_TABLES

_ENTITIES = ['user_a', 'user_b', 'user_c', 'user_d']


def _star_graph(num_satellites: int) -> Graph:
    r"""``USERS`` plus ``num_satellites`` tables that each point straight at
    it, so a single hop pulls every one of them into the subgraph.

    Keys and edges are declared rather than inferred: inference warns about a
    table it cannot find a key for, and this suite runs warnings as errors.
    """
    from sdm_relational_engine.rfm.backend.local import LocalTable

    tables = [
        LocalTable(
            pd.DataFrame({'USER_ID': _ENTITIES, 'NAME': list('ABCD')}),
            'USERS',
            primary_key='USER_ID',
        )
    ]
    edges = []
    for i in range(num_satellites):
        name = f'EVENTS_{i}'
        tables.append(
            LocalTable(
                pd.DataFrame(
                    {
                        'EVENT_ID': list(range(4)),
                        'USER_ID': _ENTITIES,
                        'TIME': pd.to_datetime(
                            [
                                '2025-01-01',
                                '2025-01-02',
                                '2025-01-03',
                                '2025-01-04',
                            ]
                        ),
                    }
                ),
                name,
                primary_key='EVENT_ID',
                time_column='TIME',
            )
        )
        edges.append((name, 'USER_ID', 'USERS'))

    graph = Graph(tables=tables, edges=edges)
    graph.infer_metadata(verbose=False)
    return graph


def _churn_on(table: str) -> ValidatedPredictiveQuery:
    return ValidatedPredictiveQuery(
        target_ast=Condition(
            target=Aggregation(
                aggr=AggregationType.COUNT,
                target=Column(fqn=f'{table}.*'),
                aggr_time_range=DateOffsetRange(0, 1),
            ),
            op=RelOp.EQ,
            value=0,
        ),
        entity_ast=Column(fqn='USERS.USER_ID'),
        rfm_entity_ids=Condition(
            target=Column(fqn='USERS.USER_ID'),
            op=MemberOp.IN,
            value=Constant.from_value(_ENTITIES),
        ),
    )


def _context(num_satellites: int) -> None:
    graph = _star_graph(num_satellites)
    model = KumoRelational(graph, verbose=False)
    query = _churn_on('EVENTS_0')
    task_table = model._get_task_table(
        query, indices=query.get_rfm_entity_id_list()
    )
    model._get_context(task_table, num_neighbors=[8])


def test_a_subgraph_at_the_limit_is_accepted() -> None:
    _context(_MAX_SUBGRAPH_TABLES - 1)


def test_a_subgraph_over_the_limit_is_refused() -> None:
    with pytest.raises(
        ValueError, match=f'more than {_MAX_SUBGRAPH_TABLES} tables'
    ):
        _context(_MAX_SUBGRAPH_TABLES)
