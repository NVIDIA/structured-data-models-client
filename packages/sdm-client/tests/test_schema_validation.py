# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A schema edit is checked when it is made, not several cells later.

The engine this client is built on already validates these on assignment
(`rfm/base/table.py` and `rfm/base/column.py`), and the client's `Table` was a
plain dataclass, so a wrong edit was stored and only failed at `materialize()`,
as a bare `KeyError` or a `ValueError` outside the documented error surface.

Nothing is refused here that the engine would have accepted. The rule is the
engine's own.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from sdm_client import Relationship, Schema
from sdm_client.errors import StructuredDataError


def users(rows: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'user_id': range(rows),
            'age': [20.0 + index for index in range(rows)],
            'segment': [
                'north' if index % 2 else 'south' for index in range(rows)
            ],
            'joined': pd.date_range('2024-01-01', periods=rows, freq='D'),
        }
    )


def schema(rows: int = 6) -> Schema:
    return Schema.from_data({'users': users(rows)})


# --- a column has to exist ---------------------------------------------------


def test_a_primary_key_that_is_not_a_column_is_refused_at_assignment():
    """It used to be stored, and surface as a KeyError from a dict lookup."""
    with pytest.raises(StructuredDataError) as error:
        schema()['users'].primary_key = 'NoSuchColumn'
    assert 'no column' in str(error.value)


def test_a_time_column_that_is_not_a_column_is_refused_at_assignment():
    with pytest.raises(StructuredDataError):
        schema()['users'].time_column = 'NoSuchColumn'


def test_an_empty_composite_key_is_refused():
    with pytest.raises(StructuredDataError) as error:
        schema()['users'].primary_key = []
    assert 'empty primary key' in str(error.value)


# --- the two roles are exclusive ---------------------------------------------


def test_the_time_column_cannot_also_be_the_primary_key():
    # A string column, so it *could* be a key, otherwise the dtype check
    # fires first and this would not reach the question being asked.
    frame = users()
    frame['when'] = [f'2024-01-0{index + 1}' for index in range(len(frame))]
    built = Schema.from_data({'users': frame}, relationships=[])
    built['users'].time_column = 'when'
    with pytest.raises(StructuredDataError) as error:
        built['users'].primary_key = 'when'
    assert 'time column' in str(error.value)


def test_a_timestamp_cannot_be_a_primary_key_at_all():
    built = schema()
    with pytest.raises(StructuredDataError) as error:
        built['users'].primary_key = 'joined'
    assert 'cannot identify a row' in str(error.value)


def test_the_primary_key_cannot_also_be_the_time_column():
    built = schema()
    with pytest.raises(StructuredDataError) as error:
        built['users'].time_column = 'user_id'
    assert 'primary key' in str(error.value)


# --- a meaning the values cannot carry ---------------------------------------


def test_text_on_a_numeric_column_is_refused():
    with pytest.raises(StructuredDataError) as error:
        schema()['users']['age'].stype = 'text'
    assert 'cannot mean' in str(error.value)


def test_a_numeric_column_cannot_be_the_time_column():
    with pytest.raises(StructuredDataError):
        Schema.from_data({'users': users()}, time_columns={'users': 'age'})


def test_a_correction_the_docs_invite_still_works():
    """`age` is numeric and re-typing it categorical is a legal correction."""
    built = schema()
    built['users']['age'].stype = 'categorical'
    assert built['users']['age'].stype == 'categorical'


def test_a_string_column_may_still_be_the_time_column():
    """A string can hold an ISO timestamp, and the language allows it.

    The rule is the engine's, so this stays legal even though it looks odd.
    """
    frame = users()
    frame['when'] = [f'2024-01-0{index + 1}' for index in range(len(frame))]
    built = Schema.from_data({'users': frame})
    built['users'].time_column = 'when'
    assert built['users'].time_column == 'when'


def test_an_unknown_semantic_type_is_still_refused_by_name():
    with pytest.raises(StructuredDataError) as error:
        schema()['users']['age'].stype = 'nonsense'
    assert 'not a semantic type' in str(error.value)


# --- a key has to identify a row --------------------------------------------


def duplicated(rows: int = 1_000, repeats: int = 20) -> pd.DataFrame:
    frame = users(rows)
    return pd.concat([frame, frame.head(repeats)], ignore_index=True)


def test_an_explicit_key_that_repeats_is_refused():
    with pytest.raises(StructuredDataError) as error:
        Schema.from_data(
            {'users': duplicated()}, primary_keys={'users': 'user_id'}
        )
    message = str(error.value)
    assert 'not unique enough' in message
    assert '20 of 1020' in message


def test_assigning_a_repeating_key_is_refused_too():
    """Inference declined this column; assignment must not put it back."""
    built = Schema.from_data({'users': duplicated()})
    assert built['users'].primary_key is None
    with pytest.raises(StructuredDataError):
        built['users'].primary_key = 'user_id'


def test_the_tolerance_inference_allows_is_the_tolerance_assignment_allows():
    """One duplicate in a thousand is what the 99% threshold exists for."""
    with pytest.warns(UserWarning):
        built = Schema.from_data({'users': duplicated(repeats=5)})
    assert built['users'].primary_key == 'user_id'
    with pytest.warns(UserWarning):
        built['users'].primary_key = 'user_id'


def test_a_unique_key_is_accepted():
    built = schema()
    built['users'].primary_key = 'age'
    assert built['users'].primary_key == 'age'


def test_a_composite_key_is_checked_across_its_columns():
    frame = pd.DataFrame(
        {'a': [1, 1, 2, 2], 'b': [1, 2, 1, 2], 'v': [1.0, 2.0, 3.0, 4.0]}
    )
    built = Schema.from_data({'t': frame}, relationships=[])
    built['t'].primary_key = ['a', 'b']
    assert built['t'].primary_key == ('a', 'b')
    with pytest.raises(StructuredDataError):
        built['t'].primary_key = ['a']


def test_an_empty_table_has_no_key_to_check():
    frame = pd.DataFrame({'user_id': pd.Series(dtype='int64')})
    built = Schema.from_data({'users': frame}, relationships=[])
    built['users'].primary_key = 'user_id'


# --- the constructor and the setter agree ------------------------------------


def test_the_constructor_and_the_setter_refuse_the_same_things():
    frame = duplicated()
    with pytest.raises(StructuredDataError):
        Schema.from_data({'users': frame}, primary_keys={'users': 'user_id'})
    built = Schema.from_data({'users': frame})
    with pytest.raises(StructuredDataError):
        built['users'].primary_key = 'user_id'


def test_an_inferred_schema_is_unaffected():
    built = Schema.from_data({'users': users(), 'orders': orders()})
    assert built['users'].primary_key == 'user_id'
    assert built['orders'].primary_key == 'order_id'
    assert built['users'].time_column == 'joined'


def orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'order_id': range(8),
            'user_id': [index % 6 for index in range(8)],
            'price': [10.0 + index for index in range(8)],
            'placed': pd.date_range('2024-02-01', periods=8, freq='D'),
        }
    )


def test_an_explicit_relationship_still_marks_both_ends_as_ids():
    built = Schema.from_data(
        {'users': users(), 'orders': orders()},
        relationships=[Relationship('orders', 'user_id', 'users', 'user_id')],
    )
    assert built['orders']['user_id'].stype == 'ID'


# --- what the threshold tolerated is said out loud ---------------------------


def test_a_tolerated_duplicate_is_counted_and_shown():
    """Inference accepts a nearly-unique key on purpose. It should say so."""
    frame = duplicated(repeats=5)
    with pytest.warns(UserWarning, match='already been used'):
        built = Schema.from_data({'users': frame})
    table = built['users']
    assert table.primary_key == 'user_id'
    assert table.duplicate_keys == 5
    assert '5 repeated keys' in repr(table)


def test_a_unique_key_says_nothing():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        built = Schema.from_data({'users': users()})
    assert built['users'].duplicate_keys == 0
    assert 'repeated' not in repr(built['users'])


# --- assignment and inference describe the same table -----------------------


def test_assigning_a_key_marks_its_columns_as_identifiers():
    """The engine's setter does this, so both routes have to."""
    built = Schema.from_data({'users': users()}, relationships=[])
    assert built['users']['age'].stype != 'ID'
    built['users'].primary_key = 'age'
    assert built['users']['age'].stype == 'ID'


def test_assigning_a_key_records_and_reports_what_it_tolerated():
    frame = duplicated(repeats=5).rename(columns={'user_id': 'k'})
    built = Schema.from_data({'users': frame}, relationships=[])
    assert built['users'].duplicate_keys == 0
    with pytest.warns(UserWarning, match='already been used'):
        built['users'].primary_key = 'k'
    assert built['users'].duplicate_keys == 5
    assert '5 repeated keys' in repr(built['users'])


def test_a_composite_key_marks_every_column_it_names():
    frame = pd.DataFrame({'a': [1, 1, 2], 'b': [1, 2, 1], 'v': [1.0, 2.0, 3.0]})
    built = Schema.from_data({'t': frame}, relationships=[])
    built['t'].primary_key = ['a', 'b']
    assert built['t']['a'].stype == 'ID'
    assert built['t']['b'].stype == 'ID'


def test_the_constructor_takes_a_composite_key_as_the_setter_does():
    frame = pd.DataFrame({'a': [1, 1, 2], 'b': [1, 2, 1], 'v': [1.0, 2.0, 3.0]})
    built = Schema.from_data(
        {'t': frame}, primary_keys={'t': ['a', 'b']}, relationships=[]
    )
    assert built['t'].primary_key == ('a', 'b')


def test_a_composite_key_naming_a_missing_column_is_refused():
    frame = pd.DataFrame({'a': [1, 2], 'v': [1.0, 2.0]})
    with pytest.raises(StructuredDataError):
        Schema.from_data(
            {'t': frame}, primary_keys={'t': ['a', 'nope']}, relationships=[]
        )


# --- a relationship has to be one an edge can express -----------------------


def test_a_relationship_must_point_at_the_primary_key():
    with pytest.raises(StructuredDataError) as error:
        Schema.from_data(
            {'users': users(), 'orders': orders()},
            relationships=[Relationship('orders', 'user_id', 'users', 'age')],
        )
    assert 'primary key' in str(error.value)


def test_a_relationship_joining_incomparable_types_is_refused():
    frame = orders()
    frame['user_id'] = frame['user_id'].astype(str)
    with pytest.raises(StructuredDataError) as error:
        Schema.from_data(
            {'users': users(), 'orders': frame},
            relationships=[
                Relationship('orders', 'user_id', 'users', 'user_id')
            ],
        )
    assert 'cannot match' in str(error.value)


def test_a_relationship_into_a_composite_key_is_refused():
    """An engine edge joins one column to one key; a pair is not expressible."""
    frame = pd.DataFrame({'a': [1, 1, 2], 'b': [1, 2, 1], 'v': [1.0, 2.0, 3.0]})
    child = pd.DataFrame({'child_id': [1, 2], 'a': [1, 2]})
    with pytest.raises(StructuredDataError) as error:
        Schema.from_data(
            {'t': frame, 'c': child},
            primary_keys={'t': ['a', 'b'], 'c': 'child_id'},
            relationships=[Relationship('c', 'a', 't', 'a')],
        )
    assert 'composite key' in str(error.value)


def test_a_key_column_that_cannot_identify_a_row_is_refused():
    """A bool has two values, so it cannot tell rows apart.

    Left as a non-ID primary key it produced a raw engine ValueError at
    materialize, which is the leak bug 3 closed everywhere else.
    """
    frame = pd.DataFrame({'flag': [False, True], 'y': [0, 1]})
    built = Schema.from_data({'t': frame}, relationships=[])
    with pytest.raises(StructuredDataError) as error:
        built['t'].primary_key = 'flag'
    assert 'cannot identify a row' in str(error.value)


def test_a_tolerated_duplicate_is_reported_once():
    """Inference assigns the key, and assignment is what reports."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        built = Schema.from_data({'users': duplicated(repeats=5)})
    assert len(caught) == 1
    assert built['users'].duplicate_keys == 5


def test_clearing_the_key_clears_what_described_it():
    with pytest.warns(UserWarning):
        built = Schema.from_data({'users': duplicated(repeats=5)})
    built['users'].primary_key = None
    assert built['users'].duplicate_keys == 0
    assert 'repeated' not in repr(built['users'])


def test_inference_marks_the_key_it_nominates():
    built = Schema.from_data({'users': users()}, relationships=[])
    assert built['users']['user_id'].stype == 'ID'


def test_inference_does_not_nominate_a_key_it_would_then_refuse():
    """A bool column called `id` is key-shaped and unique, and cannot be a key.

    Nominating it made `Schema.from_data` refuse a frame for a key nobody
    asked for.
    """
    frame = pd.DataFrame({'id': [False, True], 'y': [0, 1]})
    built = Schema.from_data({'t': frame}, relationships=[])
    assert built['t'].primary_key is None
    with pytest.raises(StructuredDataError):
        built['t'].primary_key = 'id'


def test_a_timestamp_named_like_a_key_is_not_nominated():
    frame = pd.DataFrame(
        {
            'when_id': pd.date_range('2024-01-01', periods=3),
            'v': [1.0, 2.0, 3.0],
        }
    )
    assert (
        Schema.from_data({'t': frame}, relationships=[])['t'].primary_key
        is None
    )


def test_an_ordinary_key_is_still_nominated():
    frame = pd.DataFrame({'user_id': [1, 2, 3], 'v': [1.0, 2.0, 3.0]})
    built = Schema.from_data({'t': frame}, relationships=[])
    assert built['t'].primary_key == 'user_id'
    assert built['t']['user_id'].stype == 'ID'


# --- a key with no value is its own failure, not a near-miss on uniqueness ---


def _table(frame: pd.DataFrame):
    from sdm_client.schema import _infer_table

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return _infer_table('t', frame)


def test_a_single_null_key_is_refused_rather_than_passing_silently():
    """`duplicated()` does not flag one null, so this used to be accepted.

    The engine then drops the row as `null_primary_key_rows` and nobody hears
    about it, which is the quietest way to lose a row.
    """
    frame = pd.DataFrame({'id': [1, 2, None, 4], 'v': [1.0, 2.0, 3.0, 4.0]})
    table = _table(frame)
    with pytest.raises(StructuredDataError) as error:
        table.primary_key = 'id'
    assert 'no value' in str(error.value)


def test_many_null_keys_are_reported_as_nulls_not_as_duplicates():
    """`duplicated()` treats null as equal to null, so these looked like
    repeats, and 'repeats a value already used, e.g. nan' sends the reader
    to fix the wrong thing."""
    frame = pd.DataFrame({'id': [None, None, None], 'v': [1.0, 2.0, 3.0]})
    table = _table(frame)
    with pytest.raises(StructuredDataError) as error:
        table.primary_key = 'id'
    message = str(error.value)
    assert 'no value' in message
    assert 'already used' not in message


def test_inference_tolerates_a_null_bearing_key_and_says_so():
    """Inference is a guess and a caller's key is an assertion.

    A nearly-unique column is tolerated with a warning; a null-bearing one is
    now treated the same way, because refusing it made `Schema.from_data`
    raise on frames that previously worked. The engine drops those rows
    without a word, so the guess is said out loud.
    """
    frame = pd.DataFrame(
        {'user_id': list(range(999)) + [None], 'v': [1.0] * 1000}
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        built = Schema.from_data({'users': frame})
    assert built['users'].primary_key == 'user_id'
    assert any('no value for it' in str(w.message) for w in caught)


def test_duplicate_keys_is_recorded_when_a_table_is_built_with_a_key():
    """The failing case was the constructor, not assignment.

    `duplicate_keys` was an init field declared after `primary_key`, so the
    generated `__init__` assigned 0 over whatever `_adopt_key` had just
    computed. Every table built this way reported none.
    """
    from sdm_client.schema import Column, Table

    frame = pd.DataFrame({'id': list(range(299)) + [0], 'v': [1.0] * 300})
    columns = {
        'id': Column(name='id', dtype='int64', stype='ID'),
        'v': Column(name='v', dtype='float64', stype='numerical'),
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        table = Table(name='t', data=frame, columns=columns, primary_key='id')
    assert table.duplicate_keys == 1


def test_duplicate_keys_cannot_be_asserted_by_a_caller():
    """A settable count could disagree with the frame it describes."""
    import dataclasses

    from sdm_client.schema import Table

    names = {f.name for f in dataclasses.fields(Table) if f.init}
    assert 'duplicate_keys' not in names


def test_the_duplicate_key_warning_points_at_the_caller():
    """A fixed stacklevel is right for assignment and blames this library on
    the inference path, which is the path most callers take."""
    frame = pd.DataFrame({'id': list(range(299)) + [0], 'v': [1.0] * 300})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        Schema.from_data({'t': frame})
    assert caught
    assert caught[0].filename == __file__


def test_a_null_bearing_inferred_key_warns_rather_than_breaking_a_relationship():
    """Refusing here would break `Schema.from_data` on frames that worked.

    A relationship pointing at the table needs it to have a primary key, so
    declining to infer one turned a silently dropped row into a hard failure
    with a message that did not mention nulls at all.
    """
    users = pd.DataFrame(
        {'user_id': list(range(999)) + [None], 'age': [1.0] * 1000}
    )
    orders = pd.DataFrame({'order_id': list(range(50)), 'user_id': [1.0] * 50})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        built = Schema.from_data(
            {'users': users, 'orders': orders},
            relationships=[
                Relationship('orders', 'user_id', 'users', 'user_id')
            ],
        )
    assert built['users'].primary_key == 'user_id'
    assert any('no value for it' in str(w.message) for w in caught)


def test_a_hand_built_table_refuses_a_list_cell():
    """`Schema.from_data` checked this; constructing a Table directly did not,
    and the shape surfaced much later as an unhashable-type crash."""
    from sdm_client.schema import Column, Table

    frame = pd.DataFrame({'row_id': [1, 2], 'tags': [['a', 'b'], ['c']]})
    with pytest.raises(StructuredDataError) as error:
        Table(
            name='t',
            data=frame,
            columns={
                'row_id': Column('row_id', 'int64', 'ID'),
                'tags': Column('tags', 'string', 'categorical'),
            },
        )
    assert 'container' in str(error.value)


def test_an_inferred_relationship_is_held_to_the_join_contract():
    """Inference matches on name alone, so it can pair columns that cannot
    join; an explicit link with the same dtypes is refused outright."""
    users = pd.DataFrame({'user_id': ['a', 'b', 'c'], 'v': [1.0, 2.0, 3.0]})
    orders = pd.DataFrame({'order_id': [1, 2, 3], 'user_id': [1, 2, 3]})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        built = Schema.from_data({'users': users, 'orders': orders})
    assert built.relationships == []


def test_a_compatible_relationship_is_still_inferred():
    users = pd.DataFrame({'user_id': [1, 2, 3], 'v': [1.0, 2.0, 3.0]})
    orders = pd.DataFrame({'order_id': [1, 2, 3], 'user_id': [1, 2, 3]})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        built = Schema.from_data({'users': users, 'orders': orders})
    assert [str(r) for r in built.relationships] == [
        'orders.user_id -> users.user_id'
    ]


def test_naming_a_dtype_clears_the_untyped_flag():
    """`untyped` records that inference had nothing to go on.

    A caller naming the dtype supplies exactly the information that was
    missing, so the adapter must stop substituting UNTYPED; otherwise an
    explicit `int64` on an all-null column still went out as `string`.
    """
    frame = pd.DataFrame({'row_id': [1, 2], 'v': [None, None]})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        column = Schema.from_data({'t': frame})['t']['v']
    assert column.untyped is True
    column.dtype = 'int64'
    assert column.untyped is False


def test_a_column_cannot_be_built_claiming_both_a_dtype_and_no_type():
    """`untyped` is what makes the adapter discard the declared dtype, so a
    caller must not be able to construct that contradiction directly."""
    import dataclasses

    from sdm_client.schema import Column

    names = {f.name for f in dataclasses.fields(Column) if f.init}
    assert 'untyped' not in names


def test_untyped_cannot_be_set_by_a_caller():
    """The adapter reads it as authority over `dtype`, so setting it is
    saying a column both has a type and has none."""
    from sdm_client.schema import Column

    column = Column('v', 'int64', 'numerical')
    with pytest.raises(StructuredDataError) as error:
        column.untyped = True
    assert 'cannot be set' in str(error.value)


# --- a name a query could never spell is refused at the door ----------------


@pytest.mark.parametrize(
    'name',
    ['sound`pressure', 'sound\npressure', 'sound\rpressure', 'sound.pressure'],
    ids=['backtick', 'newline', 'carriage-return', 'dot'],
)
@pytest.mark.parametrize('where', ['column', 'table'])
def test_a_name_no_query_can_spell_is_refused(name, where):
    frame = pd.DataFrame({name if where == 'column' else 'value': [1.0, 2.0]})
    with pytest.raises(StructuredDataError) as error:
        Schema.from_data({name if where == 'table' else 'records': frame})
    assert error.value.code == 'INVALID_REQUEST'
    assert repr(name) in str(error.value)


@pytest.mark.parametrize(
    'name', ['scaled-sound-pressure', 'Customer ID', 'ünïcode', '1st_reading']
)
def test_a_name_quoting_can_carry_is_accepted(name):
    built = Schema.from_data({'records': pd.DataFrame({name: [1.0, 2.0]})})
    assert name in built['records']
