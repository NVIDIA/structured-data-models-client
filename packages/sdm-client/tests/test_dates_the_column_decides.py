# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A slash-separated date column is often not ambiguous at all.

`02/01/2024` is two dates and nothing in that value says which. But a column
holding `15/03/2024` has said: no month is 15, so the first component is the
day and every other row reads the same way. Refusing the whole shape treated a
column that answers the question as one that does not, and sent a caller to
convert data that needed no converting.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from sdm_client.core import dates
from sdm_client.errors import StructuredDataError


@pytest.mark.parametrize(
    ('values', 'decided'),
    [
        (['02/01/2024', '15/03/2024', '06/04/2024'], True),
        (['01/02/2024', '03/15/2024', '04/06/2024'], False),
        (['31/12/2023'], True),
        (['12/31/2023'], False),
        (['02/01/2024', '03/05/2024', '06/04/2024'], None),
        (['15/03/2024', '03/15/2024'], None),
        (['2024-01-02', '2024-03-15'], None),
        ([], None),
    ],
)
def test_which_way_round_the_column_reads(values, decided):
    assert dates.day_comes_first(pd.Series(values, dtype=object)) is decided


@pytest.mark.parametrize(
    ('values', 'unambiguous'),
    [
        (['2024-01-02', '2024-03-15'], True),
        (['02/01/2024', '15/03/2024'], True),
        (['01/02/2024', '03/15/2024'], True),
        (['02/01/2024', '03/05/2024'], False),
        (['15/03/2024', '03/15/2024'], False),
    ],
)
def test_a_column_that_says_is_not_refused(values, unambiguous):
    assert dates.looks_unambiguous(pd.Series(values)) is unambiguous


@pytest.mark.parametrize(
    ('values', 'read_as'),
    [
        (['02/01/2024', '15/03/2024'], ['2024-01-02', '2024-03-15']),
        (['01/02/2024', '03/15/2024'], ['2024-01-02', '2024-03-15']),
        (['2024-01-02', '2024-03-15'], ['2024-01-02', '2024-03-15']),
    ],
)
def test_the_column_is_read_the_way_it_said(values, read_as):
    """The same three days written three ways, all landing on the same days.
    Reading a day-first column month-first is the silent wrong answer this
    exists to prevent, and it is wrong by up to eleven months."""
    got = dates.read(
        pd.Series(values), dates.conversion_that_works(pd.Series(values))
    )

    assert list(got.dt.strftime('%Y-%m-%d')) == read_as


def test_an_undecidable_column_is_still_refused():
    """Where the values genuinely do not say, guessing is what was wrong."""
    assert not dates.looks_unambiguous(
        pd.Series(['02/01/2024', '03/05/2024', '06/04/2024'])
    )


def test_a_part_code_that_opens_like_a_date_is_not_one():
    assert dates.day_comes_first(pd.Series(['15/03/2024-REV2'])) is None


def test_the_deciding_value_is_found_wherever_it_is():
    """One deciding value is enough and it can be anywhere. Sampling the front
    of the column would refuse one that answers, a thousand rows later."""
    late = pd.Series(['01/02/2024'] * 1000 + ['03/15/2024'])

    assert dates.day_comes_first(late) is False
    assert dates.looks_unambiguous(late)


def test_a_column_that_never_decides_is_still_refused_however_long():
    assert dates.day_comes_first(pd.Series(['01/02/2024'] * 2000)) is None


@pytest.mark.parametrize(
    ('values', 'decided'),
    [
        (['03/15/2024 00:00:00', '04/16/2024 00:00:00'], False),
        (['15/03/2024 08:30', '16/04/2024 09:00'], True),
        (['15/03/2024T08:30:00+01:00'], True),
        (['01/02/2024 00:00:00', '03/05/2024 00:00:00'], None),
        (['15/03/2024 batch7'], None),
    ],
)
def test_a_time_after_the_date_does_not_hide_what_it_says(values, decided):
    """A warehouse hands back both shapes. Reading only the date-only one
    refused a timestamp column that says perfectly well which way it reads."""
    assert dates.day_comes_first(pd.Series(values, dtype=object)) is decided


def test_a_timestamp_column_is_read_the_way_it_said():
    got = dates.read(
        pd.Series(['15/03/2024 08:30', '02/01/2024 09:00']),
        dates.Conversion(dayfirst=True),
    )

    assert list(got.dt.strftime('%Y-%m-%d %H:%M')) == [
        '2024-03-15 08:30',
        '2024-01-02 09:00',
    ]


# --- the year window is for deciding what a column is, not for overruling ---


def test_a_declared_time_column_is_not_refused_for_its_century():
    """A table of nineteenth century records is a real table. The plausible
    year window exists to tell a part code from a date on a column nobody has
    described; a caller who named their time column has already said."""
    old = pd.Series(['1899-01-01', '1899-02-14', '1875-06-30'])

    assert dates.conversion_that_works(old) is None
    assert dates.conversion_that_works(old, only_plausible_years=False) == (
        dates.Conversion()
    )


def test_a_part_code_is_still_not_a_date_where_nothing_said_it_was():
    codes = pd.Series([f'{4711 + i}-01-02' for i in range(5)])

    assert dates.conversion_that_works(codes) is None


# --- the conversion is carried, not worked out again -----------------------


@pytest.mark.parametrize(
    ('conversion', 'written'),
    [
        (dates.Conversion(), ''),
        (dates.Conversion(utc=True), ', utc=True'),
        (dates.Conversion(dayfirst=True), ', dayfirst=True'),
        (dates.Conversion(dayfirst=False), ', dayfirst=False'),
    ],
)
def test_a_conversion_writes_itself_as_the_call_it_is(conversion, written):
    """The message that tells someone to make the call and the call itself
    come from one place, so they cannot drift apart."""
    assert str(conversion) == written
    assert list(conversion.arguments) == (
        [k.split('=')[0].strip() for k in written.split(', ') if k]
    )


# --- a column that says one thing is not called ambiguous -------------------


def time_column(values):
    from sdm_client import PredictiveQuery, Schema
    from sdm_client.schema import Relationship

    users = pd.DataFrame({'uid': range(40)})
    events = pd.DataFrame(
        {'uid': [i % 40 for i in range(len(values))], 'when': values}
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'users': users, 'events': events},
            primary_keys={'users': 'uid'},
            time_columns={'events': 'when'},
            relationships=[Relationship('events', 'uid', 'users', 'uid')],
        )
        return PredictiveQuery(
            'PREDICT COUNT(events.*, 0, 30, days) > 0 FOR EACH users.uid',
            schema,
        ).materialize(indices=[1, 2], anchor_time=pd.Timestamp('2021-06-01'))


def test_a_year_first_column_with_a_named_zone_is_read():
    """The shape test took `Z` and `+00:00` and not `UTC`, so a column that
    says exactly what its dates are was refused for saying two things."""
    assert (
        time_column(
            [f'2021-02-{(i % 28) + 1:02d} 13:24:26 UTC' for i in range(120)]
        )
        is not None
    )


@pytest.mark.parametrize(
    'values',
    [
        [f'2021-02-{(i % 28) + 1:02d} 13:24:26 CST' for i in range(120)],
        [f'2021-02-{(i % 28) + 1:02d}' for i in range(119)] + ['not a date'],
    ],
    ids=['an ambiguous zone', 'one value that is not a date'],
)
def test_a_column_that_will_not_read_is_not_called_ambiguous(values):
    """`dayfirst=True`, what the ambiguity message prescribes, either raises on
    these or reads every date it can as a different one."""
    with pytest.raises(StructuredDataError) as raised:
        time_column(values)

    said = str(raised.value)
    assert 'do not read as a date' in said
    assert 'dayfirst' not in said


def test_a_date_written_the_other_way_is_still_called_ambiguous():
    """One value in another format is the ambiguity the message is for, and it
    has to stay told apart from a value that is not a date at all."""
    values = [f'2020-01-{1 + i % 28:02d}' for i in range(1500)] + ['02/01/2020']

    with pytest.raises(StructuredDataError, match='day-first'):
        time_column(values)


def test_the_remedy_for_a_column_that_will_not_read_works():
    values = [f'2021-02-{(i % 28) + 1:02d}' for i in range(119)] + [
        'not a date'
    ]

    assert (
        time_column(pd.to_datetime(pd.Series(values), errors='coerce'))
        is not None
    )
