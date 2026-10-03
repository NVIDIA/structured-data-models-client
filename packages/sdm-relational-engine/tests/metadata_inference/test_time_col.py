# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The engine's time column inference, over the rows it reads of a table."""

from __future__ import annotations

import pandas as pd
import pytest
from predictive_query.time_columns import RejectedTimeColumn
from sdm_relational_engine.rfm import Graph, LocalTable
from sdm_relational_engine.rfm.metadata_inference import infer_time_column

duckdb = pytest.importorskip('duckdb')
from sdm_relational_engine.rfm.backend.duckdb import DuckDBTable  # noqa: E402

ROWS = 1_200
HOURS = pd.Series(pd.date_range('2023-01-01', periods=ROWS, freq='h'))
LATER = HOURS + pd.Timedelta(days=3)


def _duckdb_table(frame: pd.DataFrame) -> DuckDBTable:
    connection = duckdb.connect()
    connection.register('frame', frame)
    connection.execute('CREATE TABLE t AS SELECT * FROM frame')
    return DuckDBTable(connection=connection, name='t')


def test_an_edit_stamp_no_longer_wins_a_tie_with_the_order_date():
    """`sales_order_details` holds the two with the same earliest value, and
    the first of them was taken.
    """
    frame = pd.DataFrame({'ModifiedDate': HOURS, 'OrderDate': HOURS})
    assert infer_time_column(frame, ['ModifiedDate', 'OrderDate']) == (
        'OrderDate'
    )


def test_a_lone_edit_stamp_is_taken_as_a_last_resort():
    frame = pd.DataFrame({'LastModified': HOURS})
    assert infer_time_column(frame, ['LastModified']) == 'LastModified'


def test_of_several_stamps_the_oldest_is_still_taken():
    frame = pd.DataFrame(
        {'shipped': HOURS + pd.Timedelta(days=1), 'ordered': HOURS}
    )
    assert infer_time_column(frame, ['shipped', 'ordered']) == 'ordered'


def test_the_oldest_is_read_over_the_whole_table():
    """The rows read decide nothing a later row could overturn."""
    ordered = HOURS + pd.Timedelta(days=1)
    shipped = pd.Series(
        list(HOURS[:1_000] + pd.Timedelta(days=2))
        + list(pd.date_range('2020-01-01', periods=ROWS - 1_000, freq='D'))
    )
    table = _duckdb_table(
        pd.DataFrame(
            {'id': range(ROWS), 'ordered': ordered, 'shipped': shipped}
        )
    )
    sample = table._get_sample_df()
    assert sample['ordered'].min() < sample['shipped'].min()

    table.infer_time_column(verbose=False)

    assert table.time_column is not None
    assert table.time_column.name == 'shipped'


def test_a_table_says_which_columns_it_refused_and_why():
    frame = pd.DataFrame(
        {
            'id': range(4),
            'ModifiedDate': pd.date_range('2024-01-01', periods=4),
            'updated_at': pd.date_range('2024-01-01', periods=4),
            'joined': [pd.Timestamp('2025-09-11')] * 4,
        }
    )
    table = LocalTable(frame, name='customers').infer_metadata(verbose=False)
    assert table.time_column is None
    assert table.time_column_choice is not None
    assert table.time_column_choice.rejected == (
        RejectedTimeColumn('ModifiedDate', 'edit_stamp'),
        RejectedTimeColumn('updated_at', 'edit_stamp'),
        RejectedTimeColumn('joined', 'constant', pd.Timestamp('2025-09-11')),
    )


def test_a_column_constant_in_the_rows_read_is_checked_against_the_table():
    """rel-hm's `transactions.t_dat` holds one date in its first 1000 rows."""
    late = [pd.Timestamp('2023-01-01')] * 1_000 + list(
        pd.date_range('2023-02-01', periods=ROWS - 1_000)
    )
    table = _duckdb_table(pd.DataFrame({'id': range(ROWS), 't_dat': late}))
    asked: list[str] = []
    fetch = table._fetch_rows

    def recorded(sql: str) -> list[tuple]:
        asked.append(sql)
        return fetch(sql)

    table._fetch_rows = recorded  # type: ignore[method-assign]
    assert table._get_sample_df()['t_dat'].nunique() == 1

    table.infer_time_column(verbose=False)

    assert table.time_column is not None
    assert table.time_column.name == 't_dat'
    assert [sql for sql in asked if 'MIN(' in sql] == [
        'SELECT COUNT(*), MIN("t_dat"), MAX("t_dat"), COUNT("t_dat")\nFROM "t"'
    ]


def test_a_column_constant_in_every_row_is_refused_after_checking():
    table = _duckdb_table(
        pd.DataFrame(
            {'id': range(ROWS), 'stamp': [pd.Timestamp('2025-09-11')] * ROWS}
        )
    )
    table.infer_time_column(verbose=False)
    assert table.time_column is None
    assert table.time_column_choice is not None
    assert [r.why for r in table.time_column_choice.rejected] == [
        'every row holds 2025-09-11 00:00:00'
    ]


def test_every_column_is_checked_in_one_query():
    """AdventureWorks `sales_order_headers` holds four timestamps."""
    table = _duckdb_table(
        pd.DataFrame(
            {
                'id': range(ROWS),
                'OrderDate': HOURS,
                'DueDate': LATER,
                'ShipDate': LATER,
                'ModifiedDate': LATER,
                '_etl_loaded_at': HOURS,
            }
        )
    )
    asked: list[str] = []
    fetch = table._fetch_rows

    def recorded(sql: str) -> list[tuple]:
        asked.append(sql)
        return fetch(sql)

    table._fetch_rows = recorded  # type: ignore[method-assign]
    table.infer_time_column(verbose=False)

    assert table.time_column is not None
    assert table.time_column.name == 'OrderDate'
    assert [sql for sql in asked if 'MIN(' in sql] == [
        'SELECT COUNT(*), '
        'MIN("OrderDate"), MAX("OrderDate"), COUNT("OrderDate"), '
        'MIN("DueDate"), MAX("DueDate"), COUNT("DueDate"), '
        'MIN("ShipDate"), MAX("ShipDate"), COUNT("ShipDate"), '
        'MIN("ModifiedDate"), MAX("ModifiedDate"), COUNT("ModifiedDate")\n'
        'FROM "t"'
    ]


def test_a_failed_check_keeps_the_column():
    """Only evidence refuses a column, and a warehouse error is none."""
    late = [pd.Timestamp('2023-01-01')] * ROWS
    table = _duckdb_table(pd.DataFrame({'id': range(ROWS), 't_dat': late}))

    def refused(sql: str) -> list[tuple]:
        raise RuntimeError('PERMISSION_DENIED')

    table._fetch_rows = refused  # type: ignore[method-assign]
    table.infer_time_column(verbose=False)

    assert table.time_column is not None
    assert table.time_column.name == 't_dat'
    assert table.time_column_choice is not None
    assert table.time_column_choice.rejected == ()


def test_a_graph_refuses_an_edit_stamp_on_a_table_an_edge_points_at():
    """`users` is an entity once `orders` links to it, and its
    `last_login` is no user's time. A log nothing links to keeps its own.
    """

    def tables() -> list[LocalTable]:
        return [
            LocalTable(
                pd.DataFrame({'user_id': range(4), 'last_login': HOURS[:4]}),
                name='users',
            ),
            LocalTable(
                pd.DataFrame({'order_id': range(8), 'user_id': [0, 1] * 4}),
                name='orders',
            ),
            LocalTable(
                pd.DataFrame({'user_id': [0, 1] * 4, 'updated_at': HOURS[:8]}),
                name='user_updates',
            ),
        ]

    alone = Graph(tables=tables()).infer_metadata(verbose=False)
    assert alone['users'].time_column is not None
    assert alone['users'].time_column.name == 'last_login'

    linked = Graph(
        tables=tables(),
        edges=[
            ('orders', 'user_id', 'users'),
            ('user_updates', 'user_id', 'users'),
        ],
    ).infer_metadata(verbose=False)
    assert linked['users'].time_column is None
    assert linked['user_updates'].time_column is not None
    assert linked['user_updates'].time_column.name == 'updated_at'

    inferred = Graph(tables=tables()).infer_metadata(verbose=False)
    inferred.infer_links(verbose=False)
    assert inferred['users'].time_column is None


def _users_and_orders() -> dict[str, pd.DataFrame]:
    """No link can be inferred: `orders.buyer` names no table."""
    return {
        'users': pd.DataFrame({'user_id': range(4), 'last_login': HOURS[:4]}),
        'orders': pd.DataFrame({'order_id': range(8), 'buyer': [0, 1] * 4}),
    }


def test_linking_a_table_settles_its_edit_stamp_again():
    graph = Graph.from_data(_users_and_orders(), verbose=False)
    assert graph.edges == []
    assert graph['users'].time_column is not None
    assert graph['users'].time_column.name == 'last_login'

    graph.link('orders', 'buyer', 'users')
    assert graph['users'].time_column is None

    graph.unlink('orders', 'buyer', 'users')
    assert graph['users'].time_column is not None
    assert graph['users'].time_column.name == 'last_login'


@pytest.mark.parametrize('given', ['last_login', None])
def test_a_time_column_set_by_hand_is_never_settled(given):
    """Set to the column inference took, or to none, it is the caller's."""
    graph = Graph.from_data(_users_and_orders(), verbose=False)
    graph['users'].time_column = given

    graph.link('orders', 'buyer', 'users')
    held = graph['users'].time_column
    assert (held and held.name) == given

    graph.unlink('orders', 'buyer', 'users')
    held = graph['users'].time_column
    assert (held and held.name) == given
