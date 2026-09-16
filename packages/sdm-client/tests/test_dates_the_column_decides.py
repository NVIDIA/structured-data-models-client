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

import pandas as pd
import pytest

from sdm_client.core import dates


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
    got = dates.read(pd.Series(values), '')

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
    got = dates.read(pd.Series(['15/03/2024 08:30', '02/01/2024 09:00']), '')

    assert list(got.dt.strftime('%Y-%m-%d %H:%M')) == [
        '2024-03-15 08:30',
        '2024-01-02 09:00',
    ]
