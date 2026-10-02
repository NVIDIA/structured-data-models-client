# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which timestamps inference may take as a table's time column.

AdventureWorks stamps every row of `customers` with one `ModifiedDate`, later
than every order, and that column was taken as the time column because it was
the only timestamp. Every windowed question about a customer was then refused.
"""

from __future__ import annotations

import pandas as pd
import pytest

from predictive_query.time_columns import (
    RejectedTimeColumn,
    is_edit_stamp,
    screen_time_columns,
)

DAYS = pd.Series(pd.date_range('2024-01-01', periods=4, freq='D'))
ONE_DAY = pd.Series([pd.Timestamp('2025-09-11')] * 4)
NO_DAY = pd.Series([pd.NaT] * 4)


@pytest.mark.parametrize(
    'name',
    [
        'ModifiedDate',
        'LastModified',
        'LASTMODIFIEDDATE',
        'updated_at',
        '_fivetran_synced',
        'last_seen',
        'lastLogin',
        'editedAt',
        'etl_loaded_at',
        'ETLLoadedAt',
        'refreshed_on',
        'DeletedDate',
        'deletion_date',
        'ingested_at',
        '_sdc_batched_at',
        '_airbyte_extracted_at',
        'refresh_date',
        'REFRESHDATE',
        'reload_ts',
        'reloadedAt',
        'LASTLOGIN',
        'LASTSEEN',
        'LASTORDERDATE',
        'LOADDATE',
        'SYNCEDAT',
    ],
)
def test_a_stamp_of_the_last_change_is_refused(name):
    assert is_edit_stamp(name)


@pytest.mark.parametrize(
    'name',
    [
        'created_at',
        'CreationDate',
        'OrderDate',
        'change_date',
        'inserted_at',
        'shipped_at',
        'SellStartDate',
        'download_time',
        'lastingDate',
        'LASTINGDATE',
        'ORDERDATE',
        'CREATEDAT',
        'SYNCHRONOUSDATE',
    ],
)
def test_a_stamp_of_when_a_row_began_is_kept(name):
    assert not is_edit_stamp(name)


def test_a_lone_edit_stamp_is_refused_and_says_why():
    screen = screen_time_columns(
        pd.DataFrame({'ModifiedDate': DAYS}), ['ModifiedDate']
    )
    assert screen.chosen is None
    assert screen.rejected == (
        RejectedTimeColumn('ModifiedDate', 'edit_stamp'),
    )
    assert screen.rejected[0].why == (
        'it looks like when a row was last changed, not when it began'
    )


def test_a_column_every_row_shares_is_refused_with_its_value():
    screen = screen_time_columns(pd.DataFrame({'joined': ONE_DAY}), ['joined'])
    assert screen.chosen is None
    assert screen.rejected[0].why == 'every row holds 2025-09-11 00:00:00'


def test_an_empty_column_is_refused():
    screen = screen_time_columns(pd.DataFrame({'joined': NO_DAY}), ['joined'])
    assert screen.chosen is None
    assert screen.rejected[0].why == 'no row holds a value'


def test_a_column_constant_only_in_the_rows_read_is_kept():
    """rel-hm's `transactions.t_dat` holds one date in its first 1000 rows."""
    asked = []

    def full_range(name):
        asked.append(name)
        return pd.Timestamp('2018-09-20'), pd.Timestamp('2020-09-22')

    screen = screen_time_columns(
        pd.DataFrame({'t_dat': ONE_DAY, 'other': DAYS}),
        ['t_dat', 'other'],
        full_range=full_range,
    )
    assert screen.rejected == ()
    assert screen.eligible == ('t_dat', 'other')
    # Only a column that looked constant pays for the query.
    assert asked == ['t_dat']


def test_a_column_constant_in_every_row_is_refused_after_asking():
    screen = screen_time_columns(
        pd.DataFrame({'joined': ONE_DAY}),
        ['joined'],
        full_range=lambda name: (pd.Timestamp('2025-09-11'),) * 2,
    )
    assert screen.rejected == (
        RejectedTimeColumn('joined', 'constant', pd.Timestamp('2025-09-11')),
    )


def test_a_table_that_cannot_be_asked_keeps_its_column():
    screen = screen_time_columns(
        pd.DataFrame({'joined': ONE_DAY}),
        ['joined'],
        full_range=lambda name: None,
    )
    assert screen.chosen == 'joined'


def test_the_order_date_beside_an_edit_stamp_is_chosen():
    """`sales_order_details` took `ModifiedDate` over `OrderDate` on a tie."""
    screen = screen_time_columns(
        pd.DataFrame({'ModifiedDate': DAYS, 'OrderDate': DAYS}),
        ['ModifiedDate', 'OrderDate'],
    )
    assert screen.chosen == 'OrderDate'


def test_a_creation_stamp_is_preferred_over_another_timestamp():
    screen = screen_time_columns(
        pd.DataFrame({'shipped_at': DAYS, 'CreationDate': DAYS}),
        ['shipped_at', 'CreationDate'],
    )
    assert screen.chosen == 'CreationDate'


def test_two_ordinary_timestamps_are_both_left_eligible():
    screen = screen_time_columns(
        pd.DataFrame({'shipped_at': DAYS, 'delivered_at': DAYS}),
        ['shipped_at', 'delivered_at'],
    )
    assert screen.chosen is None
    assert screen.eligible == ('shipped_at', 'delivered_at')
