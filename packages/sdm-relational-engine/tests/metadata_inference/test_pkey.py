# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
import pytest
from sdm_relational_engine.rfm.metadata_inference.pkey import infer_primary_key


def test_ambiguous_primary_key_warning_names_the_tied_candidates() -> None:
    """``max_score`` used to be bound to a ``(name, score)`` tuple and compared
    against a float, so the list of tied candidates was always empty and the
    message told the user to choose without saying from what.
    """
    df = pd.DataFrame(
        {
            'user_id': [1, 2, 3, 4],
            'user_key': [9, 8, 7, 6],
            'age': [10, 20, 30, 40],
        }
    )

    with pytest.warns(
        UserWarning, match='multiple potential primary keys'
    ) as record:
        assert infer_primary_key('users', df, ['user_id', 'user_key']) is None

    message = str(record[0].message)
    assert "'user_id'" in message
    assert "'user_key'" in message


def test_unambiguous_primary_key_is_returned_without_warning() -> None:
    df = pd.DataFrame({'user_id': [1, 2, 3], 'age': [10, 20, 30]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert infer_primary_key('users', df, ['user_id']) == 'user_id'

    assert caught == []


def _star_schema() -> dict[str, pd.DataFrame]:
    r"""Textbook naming in which the fact tables' keys do not echo their table
    names, so ``infer_primary_key`` declines every one of them.
    """
    n = 20
    return {
        'customers': pd.DataFrame({'customer_id': range(n)}),
        'products': pd.DataFrame({'product_id': range(n)}),
        'sales_orders': pd.DataFrame(
            {
                'order_id': range(n),
                'customer_id': [i % 5 for i in range(n)],
            }
        ),
        'order_lines': pd.DataFrame(
            {
                'line_id': range(n),
                'order_id': [i % 7 for i in range(n)],
                'product_id': [i % 4 for i in range(n)],
            }
        ),
    }


def test_declined_primary_key_is_reported_after_link_inference() -> None:
    """Uniqueness contributes at most 3.0 against a ``>= 4`` cut, so only a name
    matching the table's can carry a candidate. ``sales_orders.order_id`` is
    unique, uncontested and still declined, which costs the graph every edge
    into that table.
    """
    from sdm_relational_engine.rfm import Graph

    with pytest.warns(UserWarning, match='No primary key was inferred') as rec:
        graph = Graph.from_data(_star_schema(), verbose=False)

    reported = {str(warning.message) for warning in rec}
    assert any(
        "'sales_orders'" in msg and "['order_id']" in msg for msg in reported
    )
    assert any(
        "'order_lines'" in msg and "['line_id']" in msg for msg in reported
    )
    assert graph['sales_orders'].primary_key is None


def test_a_unique_foreign_key_is_not_reported_as_a_missing_primary_key() -> (
    None
):
    r"""The reason this diagnosis waits for link inference.

    ``RETURNS.ORDER_ID`` is unique -- one return per order -- and is an
    ordinary foreign key. Reported at inference time it is indistinguishable
    from a declined primary key, so warning there fires on every table that
    holds only foreign keys.
    """
    from sdm_relational_engine.rfm import Graph

    df_dict = {
        'USERS': pd.DataFrame({'USER_ID': [0, 1, 2, 3]}),
        'ORDERS': pd.DataFrame(
            {
                'ORDER_ID': [str(i) for i in range(4)],
                'USER_ID': [0, 1, 1, 3],
            }
        ),
        'RETURNS': pd.DataFrame({'ORDER_ID': ['1', '2']}),
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        graph = Graph.from_data(df_dict, verbose=False)

    assert [str(warning.message) for warning in caught] == []
    assert graph['RETURNS'].primary_key is None
    assert ('RETURNS', 'ORDER_ID', 'ORDERS') in [
        tuple(edge) for edge in graph.edges
    ]


def test_a_local_declined_key_claims_uniqueness_outright() -> None:
    """A ``LocalTable``'s sample is the whole frame, so the claim is exact and the
    wording stays unqualified.
    """
    from sdm_relational_engine.rfm import Graph

    with pytest.warns(UserWarning, match='No primary key was inferred') as rec:
        Graph.from_data(_star_schema(), verbose=False)

    reported = ' '.join(str(warning.message) for warning in rec)
    assert 'hold a unique value per row' in reported
    assert 'sampled' not in reported


def test_a_sampled_declined_key_says_so(tmp_path: Any) -> None:
    """Every SQL backend caps the sample at ``Table._NUM_SAMPLE_ROWS``, so a
    foreign key that merely happens to be distinct in the rows read looks
    exactly like a key candidate. ``basket_id`` here is unique across the first
    1,000 rows and repeats afterwards -- 1,500 distinct values in 5,000 rows --
    and the diagnostic must not state its uniqueness as a fact about the table.
    """
    pytest.importorskip('adbc_driver_sqlite')

    import sqlite3

    from sdm_relational_engine.rfm import Graph

    path = tmp_path / 'events.db'
    connection = sqlite3.connect(path)
    n = 5_000
    basket = (list(range(1_000)) + [1_000 + i // 8 for i in range(n - 1_000)])[
        :n
    ]
    frame = pd.DataFrame({'basket_id': basket, 'qty': [1] * n})
    frame.to_sql('events', connection, index=False)
    connection.commit()
    connection.close()
    assert frame['basket_id'].nunique() < n

    with pytest.warns(UserWarning, match='No primary key was inferred') as rec:
        graph = Graph.from_sqlite(str(path), verbose=False)

    reported = ' '.join(str(warning.message) for warning in rec)
    assert "['basket_id']" in reported
    assert '1,000 rows sampled' in reported
    assert 'may or may not be unique overall' in reported
    assert graph['events'].primary_key is None
