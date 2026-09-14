# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A schema that describes the data it was given, and says what it declined.

Three faults that compose into one bad sequence. Inference never reads a string
as a date, so a warehouse `DATE` column arrives as text and the table gets no
time column -- silently. The caller who spots that and reaches for the repair
the SDK's own error messages teach, `schema[t][c].stype = 'timestamp'`, found it
set the stype and not the time column, so the schema then said the column was a
timestamp and that the table had none. And a table keyed on a domain code got no
key at all, which costs every relationship pointing at it, also silently.
"""

from __future__ import annotations

import warnings
from functools import partial

import pandas as pd
import pytest

from sdm_client import Schema
from sdm_client.schema import Column, Table


def warnings_from(build) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        built = build()
    return built, [str(w.message) for w in caught]


# --- naming a timestamp orders the table -----------------------------------


def test_setting_stype_to_timestamp_sets_the_time_column():
    frame = pd.DataFrame(
        {'rid': [1, 2, 3], 'date': ['2021-01-01', '2021-02-01', '2021-03-01']}
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'review': frame})

    assert schema['review'].time_column is None

    schema['review']['date'].stype = 'timestamp'

    assert schema['review'].time_column == 'date'


def test_a_second_timestamp_does_not_displace_the_first():
    """Filling an absent time column is not the same as overriding a choice.

    The other column must not be key-shaped: a column the table is keyed by is
    refused as a time column by a different guard, which would pass this test
    without the one it is written for ever running.
    """
    frame = pd.DataFrame(
        {
            'seq': [1, 2],
            'created': pd.to_datetime(['2021-01-01', '2021-02-01']),
        }
    )
    schema = Schema.from_data({'t': frame})
    assert schema['t'].primary_key is None
    assert schema['t'].time_column == 'created'

    schema['t']['seq'].dtype = 'timestamp[us]'
    schema['t']['seq'].stype = 'timestamp'

    assert schema['t'].time_column == 'created'


def test_naming_one_of_two_timestamps_resolves_what_inference_declined():
    """Two candidates make inference decline. Saying which is the way back."""
    frame = pd.DataFrame(
        {
            'a': pd.to_datetime(['2021-01-01', '2021-02-01']),
            'b': pd.to_datetime(['2022-01-01', '2022-02-01']),
        }
    )
    schema = Schema.from_data({'t': frame})
    assert schema['t'].time_column is None

    schema['t']['b'].stype = 'timestamp'

    assert schema['t'].time_column == 'b'


def test_the_primary_key_is_not_taken_as_the_time_column():
    """The checked route refuses this pairing; the stype route must not make
    it by another door."""
    frame = pd.DataFrame({'k': [1, 2], 'v': [1.0, 2.0]})
    schema = Schema.from_data({'t': frame}, primary_keys={'t': 'k'})

    schema['t']['k'].dtype = 'timestamp[us]'
    schema['t']['k'].stype = 'timestamp'

    assert schema['t'].time_column is None


def test_a_column_is_not_compared_by_the_table_it_belongs_to():
    """The back-reference is not a field, so it must stay out of equality."""
    one = Schema.from_data({'a': pd.DataFrame({'v': [1.0]})})
    other = Schema.from_data({'b': pd.DataFrame({'v': [1.0]})})
    assert one['a']['v'] == other['b']['v']
    assert '_table' not in repr(one['a']['v'])


def test_a_column_handed_to_two_tables_becomes_two_columns():
    """One `Column` can only point at one table, so setting the stype through
    the first would have ordered the second."""
    shared = Column('when', 'string', 'categorical')
    first = Table('a', pd.DataFrame({'when': ['2021-01-01']}), {'when': shared})
    second = Table(
        'b', pd.DataFrame({'when': ['2021-01-01']}), {'when': shared}
    )

    first['when'].stype = 'timestamp'

    assert first.time_column == 'when'
    assert second.time_column is None


def test_a_caller_dict_is_not_written_into():
    columns = {'when': Column('when', 'string', 'categorical')}
    Table('a', pd.DataFrame({'when': ['2021-01-01']}), columns)
    Table('b', pd.DataFrame({'when': ['2021-01-01']}), columns)
    assert columns['when'] is not None
    assert list(columns) == ['when']


# --- a date stored as text is said out loud --------------------------------


@pytest.mark.parametrize(
    'values',
    [
        ['2013-01-01', '2013-06-15', '2014-02-20'],
        ['2013-01-01T04:05:06', '2013-06-15T00:00:00', '2014-02-20T23:59:59'],
        ['2013-01-01 04:05:06', '2013-06-15 00:00:00', '2014-02-20 23:59:59'],
        ['2013/01/01', '2013/06/15', '2014/02/20'],
        [
            '2013-01-01T04:05:06Z',
            '2013-06-15T00:00:00Z',
            '2014-02-20T00:00:00Z',
        ],
        [
            '2013-01-01T04:05:06+05:30',
            '2013-06-15T00:00:00-08:00',
            '2014-02-20T00:00:00+0000',
        ],
    ],
)
def test_a_text_date_column_says_it_is_not_the_time_column(values):
    frame = pd.DataFrame({'date': values * 4, 'v': [1.0] * 12})
    table, said = warnings_from(lambda: Schema.from_data({'t': frame})['t'])

    assert table.time_column is None
    assert any('t.date holds dates but is stored as text' in s for s in said)
    assert any('pd.to_datetime' in s for s in said)


def test_the_remedy_the_warning_names_actually_works():
    """The one thing that makes this warning worth emitting."""
    frame = pd.DataFrame(
        {'date': ['2013-01-01', '2013-06-15'] * 6, 'v': [1.0] * 12}
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'t': frame})

    schema['t']['date'].stype = 'timestamp'

    assert schema['t'].time_column == 'date'


def test_a_code_that_is_not_a_date_is_not_reported_as_one():
    """The reason strings are not parsed at all: a loose reader turns part
    codes into timestamps and dies inside materialization."""
    frame = pd.DataFrame(
        {'code': ['4711-01-02', 'A-12-9', 'ZX-0001'] * 4, 'v': [1.0] * 12}
    )
    _, said = warnings_from(lambda: Schema.from_data({'t': frame}))
    assert not any('stored as text' in s for s in said)


def test_a_code_that_merely_starts_with_a_date_is_not_reported_as_one():
    """A part code can open with something date-shaped and continue. Matching
    a prefix rather than the whole value is how a loose reader gets here."""
    frame = pd.DataFrame(
        {'part': ['2013-01-01-REV-7', '2013-01-01-REV-8'] * 6, 'v': [1.0] * 12}
    )
    _, said = warnings_from(lambda: Schema.from_data({'t': frame}))
    assert not any('stored as text' in s for s in said)


def test_a_partly_date_column_is_not_reported_as_one():
    frame = pd.DataFrame(
        {'mixed': ['2013-01-01'] * 11 + ['not a date'], 'v': [1.0] * 12}
    )
    _, said = warnings_from(lambda: Schema.from_data({'t': frame}))
    assert not any('stored as text' in s for s in said)


def test_a_date_column_with_one_bad_value_past_the_sample_is_not_reported():
    """The cheap check reads the first 1000 values. Stopping there would make
    the warning recommend a conversion that raises on value 1001."""
    frame = pd.DataFrame(
        {'d': ['2024-01-01'] * 1000 + ['not a date'], 'v': [1.0] * 1001}
    )
    _, said = warnings_from(lambda: Schema.from_data({'t': frame}))
    assert not any('stored as text' in s for s in said)


def test_an_impossible_date_is_not_reported_as_a_date():
    """`2024-99-99` is date-shaped and is not a date. The remedy the warning
    names is `pd.to_datetime`, which raises on it."""
    frame = pd.DataFrame(
        {'d': ['2024-02-31', '2024-99-99', '2024-00-00'] * 4, 'v': [1.0] * 12}
    )
    _, said = warnings_from(lambda: Schema.from_data({'t': frame}))
    assert not any('stored as text' in s for s in said)


def test_the_conversion_the_warning_names_is_checked_before_it_is_named():
    """Whenever it warns, `pd.to_datetime` must succeed on that column."""
    frame = pd.DataFrame(
        {'d': ['2024-01-01', '2024-06-15T10:00:00Z'] * 6, 'v': [1.0] * 12}
    )
    for values in (
        ['2024-01-01', '2024-06-15'],
        ['2024-01-01T10:00:00Z', '2024-06-15T11:00:00Z'],
        ['2024-01-01T10:00:00+05:30', '2024-06-15T11:00:00-08:00'],
    ):
        frame = pd.DataFrame({'d': values * 6, 'v': [1.0] * 12})
        _, said = warnings_from(partial(Schema.from_data, {'t': frame}))
        named = [s for s in said if 'stored as text' in s]
        assert named, values
        call = named[0].split('pd.to_datetime(')[1].split(') --')[0]
        arguments = {'utc': True} if 'utc=True' in call else {}
        assert pd.to_datetime(frame['d'], **arguments).notna().all()


def test_a_part_code_shaped_like_a_date_is_not_reported_as_one():
    """`4711-01-02` is a part number, matches the date form exactly, and
    pandas reads it as the year 4711 without complaint. Telling its owner to
    convert it would turn part numbers into timestamps."""
    frame = pd.DataFrame(
        {'part': ['4711-01-02', '4711-01-03'] * 6, 'v': [1.0] * 12}
    )
    _, said = warnings_from(partial(Schema.from_data, {'t': frame}))
    assert not any('stored as text' in s for s in said)


def test_a_column_of_more_than_one_date_format_is_not_reported():
    """Every value must be the same unambiguous shape. `pd.to_datetime` will
    read a mixture, and reading mixtures is how a part code became a date."""
    frame = pd.DataFrame(
        {'d': ['2024-01-01'] * 11 + ['March 5, 2024'], 'v': [1.0] * 12}
    )
    _, said = warnings_from(partial(Schema.from_data, {'t': frame}))
    assert not any('stored as text' in s for s in said)


def test_a_table_that_found_its_time_column_is_not_told_about_text():
    frame = pd.DataFrame(
        {
            'when': pd.to_datetime(['2021-01-01'] * 4),
            'other': ['2021-01-01'] * 4,
            'v': [1.0] * 4,
        }
    )
    table, said = warnings_from(lambda: Schema.from_data({'t': frame})['t'])
    assert table.time_column == 'when'
    assert not any('stored as text' in s for s in said)


# --- a table nothing can link to says so --------------------------------


def test_a_keyless_table_another_table_names_says_it_cannot_be_linked_to():
    """`IATA_CODE` and `SKU` are keys no suffix rule will cover, so the tables
    get none -- and a table with no key is not a candidate target for any
    edge, so the whole graph silently collapses to one table."""
    frames = {
        'airports': pd.DataFrame(
            {
                'IATA_CODE': ['ATL', 'ORD', 'LAX'],
                'CITY': ['Atlanta', 'Chicago', 'LA'],
            }
        ),
        'sales': pd.DataFrame(
            {
                'SALE_ID': [1, 2, 3, 4],
                'IATA_CODE': ['ATL', 'ORD', 'ATL', 'LAX'],
                'AMOUNT': [10.0, 20.0, 30.0, 40.0],
            }
        ),
    }
    schema, said = warnings_from(lambda: Schema.from_data(frames))

    assert schema['airports'].primary_key is None
    assert schema.relationships == []
    complaint = [s for s in said if 'nothing can link to a table' in s]
    assert complaint, said
    assert "primary_keys={'airports': 'IATA_CODE'}" in complaint[0]


def test_saying_the_key_is_what_the_warning_asks_for_builds_the_graph():
    """The one thing that makes the warning worth emitting."""
    frames = {
        'airports': pd.DataFrame(
            {'IATA_CODE': ['ATL', 'ORD'], 'CITY': ['Atlanta', 'Chicago']}
        ),
        'sales': pd.DataFrame({'SALE_ID': [1, 2], 'IATA_CODE': ['ATL', 'ORD']}),
    }
    schema, said = warnings_from(
        lambda: Schema.from_data(frames, primary_keys={'airports': 'IATA_CODE'})
    )
    assert [
        (r.source_table, r.source_column, r.target_table)
        for r in schema.relationships
    ] == [('sales', 'IATA_CODE', 'airports')]
    assert not any('nothing can link to a table' in s for s in said)


def test_a_unique_foreign_key_is_not_made_into_a_parent():
    """Uniqueness in the rows to hand does not distinguish a key from a
    foreign key that happens not to repeat. Nominating `line_items.ORDER_NO`
    turns a child into a parent and invents `shipments -> line_items`."""
    frames = {
        'line_items': pd.DataFrame(
            {'ORDER_NO': ['O1', 'O2'], 'LINE_NO': [1, 1], 'sku': ['A', 'B']}
        ),
        'shipments': pd.DataFrame(
            {'SHIPMENT_ID': [1, 2, 3], 'ORDER_NO': ['O1', 'O1', 'O2']}
        ),
    }
    schema, _ = warnings_from(lambda: Schema.from_data(frames))

    assert schema['line_items'].primary_key is None
    assert schema.relationships == []


def test_a_declared_parent_key_is_not_undone_by_a_unique_child_column():
    """The costlier half of the same mistake. A second table keyed on a name
    makes it ambiguous, and ambiguity deletes *every* edge using it -- so
    guessing here can destroy a graph whose keys were declared correctly."""
    frames = {
        'orders': pd.DataFrame({'ORDER_NO': ['O1', 'O2'], 'total': [1.0, 2.0]}),
        'line_items': pd.DataFrame(
            {'ORDER_NO': ['O1', 'O2'], 'LINE_NO': [1, 1], 'sku': ['A', 'B']}
        ),
        'shipments': pd.DataFrame(
            {'SHIPMENT_ID': [1, 2, 3], 'ORDER_NO': ['O1', 'O1', 'O2']}
        ),
    }
    schema, _ = warnings_from(
        lambda: Schema.from_data(frames, primary_keys={'orders': 'ORDER_NO'})
    )
    assert {(r.source_table, r.target_table) for r in schema.relationships} == {
        ('line_items', 'orders'),
        ('shipments', 'orders'),
    }


def test_a_column_no_other_table_names_costs_no_edge_and_is_not_mentioned():
    schema, said = warnings_from(
        lambda: Schema.from_data(
            {
                'airports': pd.DataFrame(
                    {'IATA_CODE': ['ATL', 'ORD'], 'CITY': ['Atlanta', 'Chi']}
                )
            }
        )
    )
    assert schema['airports'].primary_key is None
    assert not any('nothing can link to a table' in s for s in said)


def test_a_repeated_column_is_not_offered_as_a_key():
    _, said = warnings_from(
        lambda: Schema.from_data(
            {
                'parent': pd.DataFrame(
                    {'code': ['a', 'a', 'b'], 'n': [1.0, 2.0, 3.0]}
                ),
                'child': pd.DataFrame({'cid': [1, 2], 'code': ['a', 'b']}),
            }
        )
    )
    assert not any('nothing can link to a table' in s for s in said)


def test_a_float_measure_is_not_offered_however_unique():
    _, said = warnings_from(
        lambda: Schema.from_data(
            {
                'parent': pd.DataFrame({'amount': [1.5, 2.5, 3.5]}),
                'child': pd.DataFrame({'cid': [1, 2], 'amount': [1.5, 2.5]}),
            }
        )
    )
    assert not any('nothing can link to a table' in s for s in said)


def test_a_null_bearing_column_is_not_offered():
    _, said = warnings_from(
        lambda: Schema.from_data(
            {
                'parent': pd.DataFrame({'code': ['a', None, 'b']}),
                'child': pd.DataFrame({'cid': [1, 2], 'code': ['a', 'b']}),
            }
        )
    )
    assert not any('nothing can link to a table' in s for s in said)


def test_a_table_that_has_a_key_is_not_asked_about_one():
    _, said = warnings_from(
        lambda: Schema.from_data(
            {
                'parent': pd.DataFrame({'code': ['a', 'b'], 'n': [1, 2]}),
                'child': pd.DataFrame({'cid': [1, 2], 'code': ['a', 'b']}),
            },
            primary_keys={'parent': 'n'},
        )
    )
    assert not any('nothing can link to a table' in s for s in said)


def test_the_time_column_is_not_offered_as_a_key():
    """A string date declared as the time column is unique, is named by the
    child, and is a string -- it clears every other test, and a column cannot
    be both."""
    frames = {
        'parent': pd.DataFrame(
            {'day': ['2021-01-01', '2021-01-02'], 'n': [1.0, 2.0]}
        ),
        'child': pd.DataFrame(
            {'cid': [1, 2], 'day': ['2021-01-01', '2021-01-02']}
        ),
    }
    _, said = warnings_from(
        lambda: Schema.from_data(frames, time_columns={'parent': 'day'})
    )
    assert not any('nothing can link to a table' in s for s in said)


# --- a warning only where something was actually lost ----------------------


def test_a_child_of_a_declared_parent_is_not_told_to_claim_the_key():
    """`line_items.ORDER_NO` is unique in these rows and is a foreign key.
    Telling its owner to name it the primary key would make `ORDER_NO`
    ambiguous and delete the two edges that are working -- the warning would
    talk the caller into the failure it exists to prevent."""
    frames = {
        'orders': pd.DataFrame({'ORDER_NO': ['O1', 'O2'], 'total': [1.0, 2.0]}),
        'line_items': pd.DataFrame(
            {'ORDER_NO': ['O1', 'O2'], 'LINE_NO': [1, 1]}
        ),
        'shipments': pd.DataFrame(
            {'SHIPMENT_ID': [1, 2, 3], 'ORDER_NO': ['O1', 'O1', 'O2']}
        ),
    }
    schema, said = warnings_from(
        partial(Schema.from_data, frames, primary_keys={'orders': 'ORDER_NO'})
    )
    assert {(r.source_table, r.target_table) for r in schema.relationships} == {
        ('line_items', 'orders'),
        ('shipments', 'orders'),
    }
    assert not any('nothing can link to a table' in s for s in said)


def test_declaring_the_relationships_means_none_were_dropped():
    frames = {
        'airports': pd.DataFrame(
            {'IATA_CODE': ['ATL', 'ORD'], 'CITY': ['Atlanta', 'Chicago']}
        ),
        'sales': pd.DataFrame({'SALE_ID': [1, 2], 'IATA_CODE': ['ATL', 'ORD']}),
    }
    _, said = warnings_from(partial(Schema.from_data, frames, relationships=[]))
    assert not any('nothing can link to a table' in s for s in said)

    _, inferring = warnings_from(partial(Schema.from_data, frames))
    assert any('nothing can link to a table' in s for s in inferring)


def test_naming_the_time_column_means_it_is_not_missing():
    frame = pd.DataFrame(
        {'date': ['2024-01-01', '2024-01-02'], 'v': [1.0, 2.0]}
    )
    schema, said = warnings_from(
        partial(Schema.from_data, {'t': frame}, time_columns={'t': 'date'})
    )
    assert schema['t'].time_column == 'date'
    assert not any('stored as text' in s for s in said)


def test_the_two_warnings_never_name_the_same_column():
    """A column cannot be both the time column and the primary key, so asking
    for both would leave a caller who followed the advice in order with
    neither."""
    frames = {
        'calendar': pd.DataFrame(
            {'day': ['2024-01-01', '2024-01-02'], 'holiday': ['n', 'y']}
        ),
        'sales': pd.DataFrame(
            {
                'sale_id': [1, 2],
                'day': ['2024-01-01', '2024-01-02'],
                'amount': [10.0, 20.0],
            }
        ),
    }
    _, said = warnings_from(partial(Schema.from_data, frames))

    assert any('calendar.day holds dates' in s for s in said)
    assert not any('nothing can link to a table' in s for s in said)
