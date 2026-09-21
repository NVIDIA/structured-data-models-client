# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from sdm_relational_engine.rfm import Graph
from sdm_relational_engine.rfm.backend.local import LocalTable
from sdm_relational_engine.rfm.query_parser import parse_query_locally


@pytest.fixture
def two_links_to_customer() -> Graph:
    r"""The shape rel-salt has: one table reaching another by several keys."""
    customer = LocalTable(
        df=pd.DataFrame({'cust_id': [1, 2, 3], 'tier': ['a', 'b', 'a']}),
        name='customer',
        primary_key='cust_id',
    )
    item = LocalTable(
        df=pd.DataFrame(
            {
                'item_id': [1, 2, 3],
                'sold_to': [1, 2, 3],
                'ship_to': [2, 3, 1],
                'amount': [10.0, 20.0, 30.0],
                'ts': pd.to_datetime(
                    ['2024-01-01', '2024-01-02', '2024-01-03']
                ),
            }
        ),
        name='item',
        primary_key='item_id',
        time_column='ts',
    )
    graph = Graph([customer, item])
    graph.link('item', 'sold_to', 'customer')
    graph.link('item', 'ship_to', 'customer')
    return graph


def test_the_ambiguous_link_error_names_the_call_that_fixes_it(
    two_links_to_customer: Graph,
) -> None:
    r"""Telling the caller the link is ambiguous without naming the repair
    leaves them to find `unlink` themselves, and the keys are qualified
    internally, so a suggestion built from them verbatim would not run.
    """
    with pytest.raises(Exception) as raised:
        parse_query_locally(
            'PREDICT SUM(item.amount, 0, 30, days) FOR EACH customer.cust_id',
            two_links_to_customer._to_api_graph_definition(),
        )

    message = str(raised.value)
    assert 'not unique' in message
    # The keys are listed qualified as a diagnosis, which is useful, but the
    # call has to name the bare column or it will not run.
    assert "schema.unlink('item', 'sold_to', 'customer')" in message
    assert "schema.unlink('item', 'item." not in message
