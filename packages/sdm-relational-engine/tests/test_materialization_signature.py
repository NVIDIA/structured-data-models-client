# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

import pandas as pd
from sdm_relational_engine.rfm import Graph
from sdm_relational_engine.rfm.backend.local import LocalTable


def _graph() -> tuple[Graph, pd.DataFrame]:
    users = pd.DataFrame({'user_id': [1, 2], 'age': [20, 30]})
    table = LocalTable(df=users, name='users', primary_key='user_id')
    return Graph([table]), users


def test_an_untouched_graph_keeps_its_signature() -> None:
    graph, _ = _graph()

    assert graph._materialization_signature() == (
        graph._materialization_signature()
    )


def test_appending_rows_changes_the_signature() -> None:
    r"""Rows added after a materialization would otherwise be invisible to a
    caller reusing it.
    """
    graph, _ = _graph()
    before = graph._materialization_signature()

    graph['users']._data.loc[2] = {'user_id': 3, 'age': 40}

    assert graph._materialization_signature() != before


def test_adding_a_table_changes_the_signature() -> None:
    graph, _ = _graph()
    before = graph._materialization_signature()

    graph.add_table(
        LocalTable(
            df=pd.DataFrame({'order_id': [1], 'user_id': [1]}),
            name='orders',
            primary_key='order_id',
        )
    )

    assert graph._materialization_signature() != before


def test_editing_a_value_in_place_does_not_change_the_signature() -> None:
    r"""The documented limit of the guard, pinned so it is a known gap rather
    than a surprise: only a count and the schema are compared, because reading
    every value on every prediction would cost more than the reuse saves.
    """
    graph, users = _graph()
    before = graph._materialization_signature()

    users.loc[0, 'age'] = 999

    assert graph._materialization_signature() == before


def test_a_backend_that_cannot_count_cheaply_contributes_nothing() -> None:
    r"""A remote table must not be made to run a query once per prediction."""
    from sdm_relational_engine.rfm.base.table import Table

    assert Table._local_row_count(object()) is None
