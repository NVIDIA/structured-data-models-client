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
    pytest.importorskip('sdm_relational_engine')

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
    ('values', 'said'),
    [
        (
            [f'2021-02-{(i % 28) + 1:02d} 13:24:26 CST' for i in range(120)],
            'none of its 120 values read as a date',
        ),
        (
            [f'2021-02-{(i % 28) + 1:02d}' for i in range(119)]
            + ['not a date'],
            'do not read as a date',
        ),
    ],
    ids=['a zone pandas will not read', 'one value that is not a date'],
)
def test_a_column_that_will_not_read_is_not_called_ambiguous(values, said):
    """`dayfirst=True`, what the ambiguity message prescribes, either raises on
    these or reads every date it can as a different one.

    A column where nothing reads is told apart from one where something does:
    coercing the second leaves the rows that read, and coercing the first
    leaves a column of NaT, which this client refuses for a reason of its own.
    """
    with pytest.raises(StructuredDataError) as raised:
        time_column(values)

    message = str(raised.value)
    assert said in message
    assert 'dayfirst' not in message


def test_a_column_written_two_ways_is_not_called_ambiguous():
    """Every row of it says what it is; what it has no answer to is which one
    call reads it, and `dayfirst=True` re-dates the year-first half."""
    values = [f'2020-01-{1 + i % 28:02d}' for i in range(1500)] + ['02/01/2020']

    with pytest.raises(StructuredDataError) as raised:
        time_column(values)

    message = str(raised.value)
    assert 'written two ways' in message
    assert 'dayfirst' not in message


def test_the_row_that_makes_a_column_two_shapes_is_found_wherever_it_is():
    """A column long enough to outrun a sample holds the same thing a short
    one does, so the shape test reads past the front."""
    import pandas as pd

    from sdm_client.core import dates

    values = pd.Series(
        [f'2020-01-{1 + i % 28:02d}' for i in range(5000)] + ['02/01/2020']
    )
    assert sorted(dates.shapes_written(values)) == [
        'day-or-month-first',
        'year-first',
    ]


def test_the_remedy_for_a_column_that_will_not_read_works():
    values = [f'2021-02-{(i % 28) + 1:02d}' for i in range(119)] + [
        'not a date'
    ]

    assert (
        time_column(pd.to_datetime(pd.Series(values), errors='coerce'))
        is not None
    )


@pytest.mark.parametrize(
    'zone',
    ['-0800', '-08:00', '-08', '+0530', 'Z', '+00:00', 'UTC', 'America/Denver'],
)
def test_a_year_first_column_is_read_however_its_zone_is_written(zone):
    """The shape test took a named zone and not a numeric offset, and a
    space-separated numeric offset is what a warehouse writes by default."""
    from sdm_client.core import dates

    values = pd.Series(
        [f'2021-02-{(i % 28) + 1:02d} 13:24:26.000 {zone}' for i in range(120)]
    )
    assert dates.looks_unambiguous(values), zone


def test_the_offset_a_warehouse_writes_by_default_is_read_end_to_end():
    read = time_column(
        [f'2021-02-{(i % 28) + 1:02d} 13:24:26.000 -0800' for i in range(120)]
    )
    assert read is not None


@pytest.mark.parametrize(
    'value', ['02/01/2020', '2020-01-02-REV-7', 'not a date at all']
)
def test_widening_the_zone_does_not_widen_what_counts_as_year_first(value):
    from sdm_client.core import dates

    assert not dates.YEAR_FIRST.fullmatch(value)


def test_a_column_where_nothing_reads_is_not_told_to_coerce():
    """Coercing every value leaves a column of NaT, which this client refuses
    for a reason of its own, so the remedy swaps one refusal for another."""
    with pytest.raises(StructuredDataError) as raised:
        time_column(['2021-13-45'] * 120)

    message = str(raised.value)
    assert 'none of its 120 values read as a date' in message
    assert 'coerce' not in message


def test_a_column_where_something_reads_is_still_told_to_coerce():
    with pytest.raises(StructuredDataError) as raised:
        time_column([f'2021-02-{(i % 28) + 1:02d}' for i in range(119)] + ['x'])

    assert "errors='coerce'" in str(raised.value)


@pytest.mark.parametrize(
    ('odd_one_out', 'expected'),
    [
        ('13/02/2021', 'written two ways'),
        ('99/99/2020', 'do not read as a date'),
        ('not a date', 'do not read as a date'),
    ],
    ids=['a real date the other way', 'slash-shaped junk', 'plain junk'],
)
def test_a_second_shape_has_to_hold_dates_to_be_a_second_shape(
    odd_one_out, expected
):
    """`99/99/2020` is shaped like a slash date and is not one, so reporting
    the column as written two ways sends its owner to reconcile two formats
    when what they have is one bad value."""
    values = [f'2021-02-{(i % 28) + 1:02d}' for i in range(119)] + [odd_one_out]

    with pytest.raises(StructuredDataError) as raised:
        time_column(values)

    assert expected in str(raised.value)


def test_a_column_of_nothing_but_slash_shaped_junk_says_nothing_reads():
    with pytest.raises(StructuredDataError) as raised:
        time_column(['99/99/2020'] * 20)

    assert 'none of its 20 values read as a date' in str(raised.value)


def test_a_year_first_shape_has_to_hold_dates_too():
    """The same rule on the other shape: `2021-13-45` leads with its year and
    is not a date, so a column holding it and a real slash date is one bad
    half, not two formats."""
    with pytest.raises(StructuredDataError) as raised:
        time_column(['2021-13-45'] * 10 + ['13/02/2021'] * 10)

    assert 'do not read as a date' in str(raised.value)


@pytest.mark.parametrize(
    'zone', ['+0200', ' +0200', ' UTC', 'Z', ' -08:00', '']
)
def test_a_slash_date_is_a_shape_however_its_zone_is_written(zone):
    """A column is not written in two shapes because this client reads a zone
    on one of them and not the other."""
    from sdm_client.core import dates

    year_first = f'2020-01-01 00:00{zone}'
    slash = f'13/02/2021 00:00{zone}'
    assert dates.YEAR_FIRST.fullmatch(year_first), year_first
    assert dates.DAY_OR_MONTH_FIRST.match(slash), slash
    assert sorted(dates.shapes_written(pd.Series([year_first, slash]))) == [
        'day-or-month-first',
        'year-first',
    ]


def test_a_zone_on_every_row_is_still_one_shape():
    from sdm_client.core import dates

    values = pd.Series([f'2020-01-{d:02d} 00:00 +0100' for d in range(1, 29)])
    assert sorted(dates.shapes_written(values)) == ['year-first']


@pytest.mark.parametrize(
    'values',
    [
        ['2020-01-01', '2020-01-02 00:00 +0100'],
        ['2020-01-01 00:00 +0100', '2020-01-02 00:00 -0800'],
        ['2020-01-01', '2020-01-02T00:00:00Z'],
    ],
    ids=['some rows zoned', 'two offsets', 'a bare date and a Z'],
)
def test_a_year_first_column_reads_when_only_some_rows_carry_a_zone(values):
    """A warehouse writes a zone on the rows that have one. Every row here
    leads with its year, so calling the column ambiguous describes none of
    them – it needs a per-row format, not a choice of reading."""
    assert time_column(values * 45) is not None


def test_reading_each_value_on_its_own_terms_is_not_offered_to_inference():
    """It also reads part codes, which is the guess the plausible-year window
    exists to refuse."""
    from sdm_client.core import dates

    codes = pd.Series(['4711-01-02'] * 20)
    assert dates.conversion_that_works(codes) is None
    assert dates.conversion_that_works(codes, only_plausible_years=False)


def test_a_column_that_says_two_things_is_still_refused():
    with pytest.raises(StructuredDataError, match='does not say unambiguously'):
        time_column(['02/01/2020', '03/01/2020'] * 5)


ISO = [f'2021-02-{d:02d}' for d in range(1, 29)]
ZONED = [f'2021-02-{d:02d} 00:00 +0100' for d in range(1, 29)]
DAY_FIRST = [f'{d:02d}/02/2021' for d in range(1, 29)]
MONTH_FIRST = [f'02/{d:02d}/2021' for d in range(1, 29)]
UNDECIDED = ['02/01/2020', '03/01/2020'] * 14
JUNK = ['not a date']
SLASH_JUNK = ['99/99/2020']
UNREADABLE_ISO = ['2021-13-45']


def which_arm(values) -> str:
    """Which account `_to_time` gives of a column, or that it read it."""
    from sdm_client.materializers.relational import _to_time

    try:
        _to_time(pd.Series(values), 't.when')
        return 'read'
    except StructuredDataError as raised:
        said = str(raised)
    for key, arm in (
        ('written two ways', 'two shapes'),
        ('none of its', 'none read'),
        ('do not read as a date', 'some unread'),
        ('does not say unambiguously', 'ambiguous'),
    ):
        if key in said:
            return arm
    return said


@pytest.mark.parametrize(
    ('values', 'arm'),
    [
        (ISO, 'read'),
        (ZONED, 'read'),
        (ISO[:14] + ZONED[:14], 'read'),
        (DAY_FIRST, 'read'),
        (MONTH_FIRST, 'read'),
        (UNDECIDED, 'ambiguous'),
        (ISO[:14] + DAY_FIRST[:14], 'two shapes'),
        (ISO[:14] + UNDECIDED[:14], 'two shapes'),
        (ISO + JUNK, 'some unread'),
        (ISO + SLASH_JUNK, 'some unread'),
        (ISO + UNREADABLE_ISO, 'some unread'),
        (DAY_FIRST + JUNK, 'some unread'),
        (DAY_FIRST + SLASH_JUNK, 'some unread'),
        (MONTH_FIRST + JUNK, 'some unread'),
        (UNREADABLE_ISO * 14 + DAY_FIRST[:14], 'some unread'),
        (JUNK * 28, 'none read'),
        (SLASH_JUNK * 28, 'none read'),
        (UNREADABLE_ISO * 28, 'none read'),
    ],
    ids=lambda v: v if isinstance(v, str) else '',
)
def test_every_shape_of_column_gets_the_account_that_is_true_of_it(values, arm):
    """`_to_time` has five arms and their order is load-bearing. A column
    written day-first with one value that is not a date was called ambiguous,
    where the same column written year-first was reported correctly."""
    assert which_arm(values) == arm


MIXED_OFFSET_SLASH = [
    '13/02/2020 00:00 +0100',
    '14/02/2020 00:00 -0800',
    '15/02/2020',
]
TRAILING_SPACE = ['2024-09-01 00:00:00 ', '2024-09-02 00:00:00 ']


def test_a_slash_column_carrying_a_zone_on_some_rows_reads():
    """It decides what its dates are and still needs a format per value, so
    one shape is enough to reach the per-row reading."""
    assert which_arm(MIXED_OFFSET_SLASH * 10) == 'read'


def test_asking_whether_a_column_reads_cannot_itself_fail():
    """Pandas refuses to build a column of mixed offsets without `utc`, and
    this is the one question that has to have an answer."""
    from sdm_client.core.dates import reads_as_a_date

    assert reads_as_a_date(pd.Series(MIXED_OFFSET_SLASH)).all()


@pytest.mark.parametrize(
    'values',
    [TRAILING_SPACE, ['  2024-09-01', '  2024-09-02'], ['2024-09-01\t']],
    ids=['trailing', 'leading', 'a tab'],
)
def test_space_around_a_value_is_not_part_of_its_shape(values):
    """A column of these matched no shape, so it was refused by the arm that
    counts what does not read – which then said `0 of 20` of them did not."""
    assert which_arm(values * 14) == 'read'


@pytest.mark.parametrize(
    ('values', 'arm', 'names'),
    [
        (
            ['13/02/2020 00:00 +0100', '14/02/2020 00:00 -0800', 'not a date'],
            'some unread',
            'not a date',
        ),
        (
            ['2020-01-01', '2020-01-02 00:00 +0100', 'not a date'],
            'some unread',
            'not a date',
        ),
        (
            [f'{d:02d}/02/2021' for d in range(1, 29)] + ['not a date'],
            'some unread',
            'not a date',
        ),
    ],
    ids=['mixed-offset slash', 'partial zone', 'day-first'],
)
def test_what_does_not_read_is_counted_by_the_one_reading_that_answers(
    values, arm, names
):
    """Three places asked whether a value reads and answered three ways: one
    raised on mixed offsets, one counted valid day-first dates as unread, and
    one sent the column to the ambiguity arm."""
    from sdm_client.materializers.relational import _to_time

    padded = values if len(values) > 20 else values * 10
    assert which_arm(padded) == arm

    with pytest.raises(StructuredDataError) as raised:
        _to_time(pd.Series(padded), 't.when')
    said = str(raised.value)
    assert names in said
    assert f'{padded.count("not a date")} of {len(padded)}' in said


def _account_the_facts_call_for(values) -> str:
    """Which arm a column's shape and readability say it belongs on.

    Written from the facts rather than from the branch order, so it disagrees
    with `_to_time` whenever the order is wrong. It shares the two primitives,
    so it checks the routing rather than re-deciding what reads.
    """
    from sdm_client.core import dates

    column = pd.Series(values)
    reads = dates.reads_as_a_date(column)
    shapes = dates.shapes_written(column)
    if len(shapes) > 1:
        return 'two shapes'
    if not reads.any():
        return 'none read'
    if not reads.all():
        return 'some unread'
    if 'year-first' in shapes or dates.day_comes_first(column) is not None:
        return 'read'
    return 'ambiguous'


ZONES = ['', 'Z', '+0100', ' +0100', ' -08:00', ' UTC', ' ']


def _written(shape: str, day: int, zone: str) -> str:
    stem = {
        'year-first': f'2021-02-{day:02d}',
        'day-first': f'{day:02d}/02/2021',
        'month-first': f'02/{day:02d}/2021',
    }[shape]
    return f'{stem} 00:00:00{zone}' if zone else stem


@pytest.mark.parametrize('shape', ['year-first', 'day-first', 'month-first'])
@pytest.mark.parametrize('zone', ZONES)
@pytest.mark.parametrize(
    'junk', [[], ['not a date'], ['99/99/2020'], ['2021-13-45']]
)
def test_the_arm_a_column_lands_on_follows_from_what_it_holds(
    shape, zone, junk
):
    """The branch order is load-bearing and nothing but the order says so.

    Every defect found in this file was a column routed to an arm whose
    message was not true of it, so this asserts the routing against the facts
    for every combination of shape, zone spelling and bad value.
    """
    values = [_written(shape, day, zone) for day in range(1, 29)] + junk
    assert which_arm(values) == _account_the_facts_call_for(values)


def test_a_column_with_both_faults_is_told_about_both():
    """Coercing is what the unreadable-values message offers, and coercing a
    column written two ways throws away every row of the shape it does not
    settle on – measured at 10 of 30 here – so the shapes are named first,
    and the values that do not read are named too."""
    from sdm_client.materializers.relational import _to_time

    values = ['2021-02-01', '13/02/2021', 'not a date'] * 10

    with pytest.raises(StructuredDataError) as raised:
        _to_time(pd.Series(values), 't.when')

    said = str(raised.value)
    assert 'written two ways' in said
    assert '10 of its values do not read as a date either' in said


def test_a_column_written_two_ways_and_nothing_else_says_only_that():
    from sdm_client.materializers.relational import _to_time

    with pytest.raises(StructuredDataError) as raised:
        _to_time(pd.Series(['2021-02-01', '13/02/2021'] * 10), 't.when')

    said = str(raised.value)
    assert 'written two ways' in said
    assert 'do not read as a date either' not in said


def test_coercing_a_column_written_two_ways_would_lose_the_other_half():
    """Why the shapes are named first. This is the remedy the other arm
    offers, measured on the column that has both faults."""
    values = pd.Series(['2021-02-01', '13/02/2021', 'not a date'] * 10)

    coerced = pd.to_datetime(values, errors='coerce')

    assert int(coerced.isna().sum()) == 20
    assert coerced[1::3].isna().all()
