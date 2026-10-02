# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which timestamp inference takes as a table's time column.

AdventureWorks stamps every row of `customers` with one `ModifiedDate`, later
than every order, and that column was taken as the time column because it was
the only timestamp. Every windowed question about a customer was then refused.
An append-only log of changes, though, has only its `updated_at`, and that is
when each of its rows began.
"""

from __future__ import annotations

import pandas as pd
import pytest

from predictive_query.time_columns import (
    RejectedTimeColumn,
    choose_time_column,
    is_birth_date,
    is_edit_stamp,
    is_load_stamp,
)

DAYS = pd.Series(pd.date_range('2024-01-01', periods=4, freq='D'))
LATER = DAYS + pd.Timedelta(days=3)
ONE_DAY = pd.Series([pd.Timestamp('2025-09-11')] * 4)
NO_DAY = pd.Series([pd.NaT] * 4)


def _choose(full_range=None, **columns):
    return choose_time_column(
        pd.DataFrame(columns), list(columns), full_range=full_range
    )


@pytest.mark.parametrize(
    'name',
    [
        '_fivetran_synced',
        'etl_loaded_at',
        'ETLLoadedAt',
        'refreshed_on',
        'ingested_at',
        '_sdc_batched_at',
        '_airbyte_extracted_at',
        'refresh_date',
        'REFRESHDATE',
        'reload_ts',
        'reloadedAt',
        'LOADDATE',
        'SYNCEDAT',
    ],
)
def test_a_stamp_of_a_load_is_read_as_one(name):
    assert is_load_stamp(name)
    assert not is_edit_stamp(name)


@pytest.mark.parametrize(
    'name',
    [
        'ModifiedDate',
        'LastModified',
        'LASTMODIFIEDDATE',
        'updated_at',
        'last_seen',
        'lastLogin',
        'editedAt',
        'DeletedDate',
        'deletion_date',
        'LASTLOGIN',
        'LASTSEEN',
        'LASTORDERDATE',
    ],
)
def test_a_stamp_of_the_last_change_is_read_as_one(name):
    assert is_edit_stamp(name)
    assert not is_load_stamp(name)


@pytest.mark.parametrize(
    'name',
    [
        'BirthDate',
        'birth_date',
        'BIRTHDATE',
        'birthday',
        'date_of_birth',
        'DATEOFBIRTH',
        'dob',
        'DOB',
        'born_at',
        'BORNAT',
    ],
)
def test_a_date_of_birth_is_read_as_one(name):
    assert is_birth_date(name)


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
    assert not is_load_stamp(name)
    assert not is_birth_date(name)


def test_a_load_stamp_is_never_taken():
    choice = _choose(_fivetran_synced=DAYS)
    assert choice.column is None
    assert choice.rejected == (
        RejectedTimeColumn('_fivetran_synced', 'load_stamp'),
    )
    assert choice.rejected[0].why == (
        'it looks like when a row was loaded, not when it began'
    )


def test_a_date_of_birth_is_never_taken():
    """Northwind's `Employees` holds `BirthDate` and `HireDate`, and the
    earliest values are the birth dates."""
    choice = _choose(BirthDate=DAYS, HireDate=LATER)
    assert choice.column == 'HireDate'
    assert choice.how == 'only'
    alone = _choose(dob=DAYS)
    assert alone.column is None
    assert alone.rejected[0].why == (
        'it looks like a date of birth, not when a row began'
    )


def test_a_lone_edit_stamp_is_taken_as_a_last_resort():
    """`order_status_updates(order_id, status, updated_at)` is a log of
    changes, and each row began when it was written."""
    choice = _choose(updated_at=DAYS)
    assert choice.column == 'updated_at'
    assert choice.how == 'edit_stamp'
    assert choice.rejected == ()


def test_an_edit_stamp_is_never_taken_over_another_timestamp():
    """`sales_order_details` holds the two with the same earliest value."""
    choice = _choose(ModifiedDate=DAYS, OrderDate=LATER)
    assert choice.column == 'OrderDate'
    assert choice.how == 'only'
    assert choice.rejected == (
        RejectedTimeColumn('ModifiedDate', 'edit_stamp'),
    )
    assert choice.rejected[0].why == (
        'it looks like when a row was last changed, not when it began'
    )


def test_two_edit_stamps_alone_are_neither_taken():
    choice = _choose(updated_at=DAYS, last_login=LATER)
    assert choice.column is None
    assert [r.column for r in choice.rejected] == ['updated_at', 'last_login']


def test_a_constant_edit_stamp_is_refused_for_its_value():
    """AdventureWorks `customers.ModifiedDate`."""
    choice = _choose(ModifiedDate=ONE_DAY)
    assert choice.column is None
    assert choice.rejected == (
        RejectedTimeColumn('ModifiedDate', 'constant', ONE_DAY[0]),
    )


def test_a_column_every_row_shares_is_refused_with_its_value():
    choice = _choose(joined=ONE_DAY)
    assert choice.column is None
    assert choice.rejected[0].why == 'every row holds 2025-09-11 00:00:00'


def test_an_empty_column_is_refused():
    choice = _choose(joined=NO_DAY)
    assert choice.column is None
    assert choice.rejected[0].why == 'no row holds a value'


def test_a_single_value_is_no_evidence_of_a_constant():
    """A one-row frame keeps its timestamp."""
    assert _choose(ordered=ONE_DAY[:1]).column == 'ordered'
    held_once = pd.Series([pd.NaT, pd.Timestamp('2025-09-11'), pd.NaT])
    assert _choose(ordered=held_once).column == 'ordered'


def test_a_column_constant_only_in_the_rows_read_is_kept():
    """rel-hm's `transactions.t_dat` holds one date in its first 1000 rows."""
    asked = []

    def full_range(name):
        asked.append(name)
        return pd.Timestamp('2018-09-20'), pd.Timestamp('2020-09-22'), 9

    choice = _choose(full_range, t_dat=ONE_DAY, other=LATER)
    assert choice.rejected == ()
    assert choice.among == ('t_dat', 'other')
    assert choice.column == 't_dat'
    # A column that looked constant pays for the check, and every column a
    # choice by earliest value is made among is read over the whole table.
    assert asked == ['t_dat', 'other']


def test_a_column_constant_in_every_row_is_refused_after_asking():
    choice = _choose(
        lambda name: (pd.Timestamp('2025-09-11'),) * 2 + (9,),
        joined=ONE_DAY,
    )
    assert choice.rejected == (
        RejectedTimeColumn('joined', 'constant', pd.Timestamp('2025-09-11')),
    )


def test_a_value_held_once_in_the_rows_read_is_checked_against_the_table():
    held_once = pd.Series([pd.NaT, pd.Timestamp('2025-09-11'), pd.NaT])
    stamp = pd.Timestamp('2025-09-11')
    assert (
        _choose(lambda name: (stamp, stamp, 2), joined=held_once).column is None
    )
    assert (
        _choose(lambda name: (stamp, stamp, 1), joined=held_once).column
        == 'joined'
    )


def test_a_table_that_cannot_be_asked_keeps_its_column():
    assert _choose(lambda name: None, joined=ONE_DAY).column == 'joined'


def test_a_creation_stamp_is_preferred_over_another_timestamp():
    choice = _choose(shipped_at=DAYS, CreationDate=LATER)
    assert choice.column == 'CreationDate'
    assert choice.how == 'created'


def test_of_several_timestamps_the_earliest_is_taken():
    """AdventureWorks `sales_order_headers` holds OrderDate, DueDate and
    ShipDate, and the order comes first."""
    choice = _choose(
        DueDate=LATER + pd.Timedelta(days=9),
        ShipDate=LATER,
        OrderDate=DAYS,
    )
    assert choice.column == 'OrderDate'
    assert choice.how == 'earliest'
    assert choice.among == ('DueDate', 'ShipDate', 'OrderDate')


def test_the_earliest_is_read_over_the_whole_table_when_it_can_be():
    """The rows read are not the first rows of the table on every source."""
    starts = {
        'shipped': pd.Timestamp('2020-01-01'),
        'ordered': pd.Timestamp('2021-01-01'),
    }
    choice = _choose(
        lambda name: (starts[name], pd.Timestamp('2030-01-01'), 9),
        shipped=LATER,
        ordered=DAYS,
    )
    assert choice.column == 'shipped'


def test_the_earliest_compares_times_in_any_zone_and_text():
    choice = _choose(
        lambda name: {
            'shipped': ('2024-01-01 06:00:00', '2025-01-01', 9),
            'ordered': (
                pd.Timestamp('2024-01-01 09:00', tz='Asia/Tokyo'),
                pd.Timestamp('2025-01-01', tz='Asia/Tokyo'),
                9,
            ),
        }[name],
        shipped=DAYS,
        ordered=LATER,
    )
    assert choice.column == 'ordered'


def test_a_tie_on_the_earliest_value_takes_the_first_column():
    assert _choose(shipped=DAYS, delivered=DAYS).column == 'shipped'
