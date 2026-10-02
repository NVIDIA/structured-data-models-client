# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A key name two tables share names no one table, so it links nothing.

`_infer_relationships` matched a column against a dict keyed by primary-key
*column name*. Tables whose keys are spelled alike, `ID`, `business_id`,
`LOAN_ID`, overwrote one another, so every foreign key of that name pointed
at whichever table the mapping happened to hold last. The result moved with
dict insertion order, joined nothing in the cases that mattered, and invented
edges between tables that have no relationship at all.

The function's own docstring says a wrong link is harder to notice than a
missing one. Ambiguity is therefore resolved by declining to guess.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sdm_client import Schema
from sdm_client.errors import StructuredDataError


def edges(schema: Schema) -> set[tuple[str, str, str, str]]:
    return {
        (r.source_table, r.source_column, r.target_table, r.target_column)
        for r in schema.relationships
    }


# --- a name two tables share is not a name at all ---------------------------


def test_a_shared_key_name_infers_nothing_for_that_name():
    """Both orderings, because the defect was that they disagreed."""
    frames = {
        'alpha': pd.DataFrame({'ID': [1, 2, 3], 'a': [1.0, 2.0, 3.0]}),
        'beta': pd.DataFrame({'ID': [10, 20, 30], 'b': [1.0, 2.0, 3.0]}),
        'child': pd.DataFrame({'cid': [1, 2, 3], 'ID': [1, 2, 3]}),
    }
    keys = {'alpha': 'ID', 'beta': 'ID', 'child': 'cid'}
    forward = Schema.from_data(frames, primary_keys=keys)
    backward = Schema.from_data(
        {name: frames[name] for name in reversed(list(frames))},
        primary_keys=keys,
    )
    assert edges(forward) == set()
    assert edges(backward) == set()


def test_two_parents_sharing_a_key_name_are_not_linked_to_each_other():
    """`airlines.IATA_CODE -> airports.IATA_CODE` joined 0 of 14 rows."""
    schema = Schema.from_data(
        {
            'airlines': pd.DataFrame(
                {'IATA_CODE': ['AA', 'DL'], 'name': ['American', 'Delta']}
            ),
            'airports': pd.DataFrame(
                {'IATA_CODE': ['ATL', 'ORD'], 'city': ['Atlanta', 'Chicago']}
            ),
        },
        primary_keys={'airlines': 'IATA_CODE', 'airports': 'IATA_CODE'},
    )
    assert edges(schema) == set()


# --- an unambiguous name still links --------------------------------------


def test_an_unambiguous_key_still_links():
    schema = Schema.from_data(
        {
            'users': pd.DataFrame({'user_id': [1, 2, 3], 'seg': list('abc')}),
            'orders': pd.DataFrame(
                {'order_id': [1, 2, 3], 'user_id': [1, 2, 3]}
            ),
        },
        primary_keys={'users': 'user_id', 'orders': 'order_id'},
    )
    assert edges(schema) == {('orders', 'user_id', 'users', 'user_id')}


def test_ambiguity_in_one_name_does_not_suppress_another():
    """Only the contested name goes unlinked; the rest of the graph stands."""
    schema = Schema.from_data(
        {
            'alpha': pd.DataFrame({'ID': [1, 2], 'x': [1.0, 2.0]}),
            'beta': pd.DataFrame({'ID': [3, 4], 'y': [1.0, 2.0]}),
            'users': pd.DataFrame({'user_id': [1, 2], 'seg': ['a', 'b']}),
            'events': pd.DataFrame(
                {'ev_id': [1, 2], 'user_id': [1, 2], 'ID': [1, 2]}
            ),
        },
        primary_keys={
            'alpha': 'ID',
            'beta': 'ID',
            'users': 'user_id',
            'events': 'ev_id',
        },
    )
    assert edges(schema) == {('events', 'user_id', 'users', 'user_id')}


def test_a_table_keyed_like_its_own_column_is_not_linked_to_itself():
    schema = Schema.from_data(
        {
            'users': pd.DataFrame({'user_id': [1, 2, 3], 'seg': list('abc')}),
        },
        primary_keys={'users': 'user_id'},
    )
    assert edges(schema) == set()


def test_a_composite_key_is_not_a_target_for_inference():
    """A composite key names no single column, so nothing can point at it."""
    schema = Schema.from_data(
        {
            'pairs': pd.DataFrame(
                {'a': [1, 1, 2], 'b': [1, 2, 1], 'v': [1.0, 2.0, 3.0]}
            ),
            'child': pd.DataFrame({'cid': [1, 2, 3], 'a': [1, 1, 2]}),
        },
        primary_keys={'pairs': ['a', 'b'], 'child': 'cid'},
    )
    assert edges(schema) == set()


def test_ambiguity_is_scoped_to_the_column_and_not_the_table():
    """A table whose own key name is contested still links its other columns.

    Distinguishes column-scoped ambiguity from a coarser "skip this table"
    rule, which would satisfy every test above while quietly losing edges.
    """
    schema = Schema.from_data(
        {
            'users': pd.DataFrame({'ID': [1, 2], 'seg': ['a', 'b']}),
            'orders': pd.DataFrame(
                {'ID': [1, 2], 'account_id': [7, 8], 'v': [1.0, 2.0]}
            ),
            'accounts': pd.DataFrame(
                {'account_id': [7, 8], 'tier': ['x', 'y']}
            ),
        },
        primary_keys={
            'users': 'ID',
            'orders': 'ID',
            'accounts': 'account_id',
        },
    )
    assert edges(schema) == {('orders', 'account_id', 'accounts', 'account_id')}


def test_a_composite_key_component_does_not_contest_a_real_key():
    """`a` inside a composite key is not a claim on the name `a`.

    Only a table keyed by a single column claims that name, so a composite
    holding `a` must not suppress a genuine `a`-keyed parent elsewhere. Both
    the composite table and the child hold an `a` column, and both are ordinary
    foreign keys into the one table that is actually keyed by it.
    """
    schema = Schema.from_data(
        {
            'pairs': pd.DataFrame(
                {'a': [1, 1, 2], 'b': [1, 2, 1], 'v': [1.0, 2.0, 3.0]}
            ),
            'owner': pd.DataFrame({'a': [1, 2], 'name': ['x', 'y']}),
            'child': pd.DataFrame({'cid': [1, 2], 'a': [1, 2]}),
        },
        primary_keys={'pairs': ['a', 'b'], 'owner': 'a', 'child': 'cid'},
    )
    assert edges(schema) == {
        ('child', 'a', 'owner', 'a'),
        ('pairs', 'a', 'owner', 'a'),
    }


# --- a tolerated duplicate says what would have worked ----------------------


def test_a_repeated_key_names_the_composite_that_is_unique(recwarn):
    """A repeat is the one visible sign the column identifies a parent rather
    than a row. Saying only that the rows are indistinguishable describes the
    symptom; naming the composite is what the caller can act on."""
    frame = pd.DataFrame(
        {'user_id': list(range(200)), 'line_no': [1] * 200, 'v': [1.0] * 200}
    )
    frame.loc[[7, 8], 'user_id'] = 6
    frame.loc[[7, 8], 'line_no'] = [2, 3]

    schema = Schema.from_data({'t': frame})

    assert schema['t'].primary_key == 'user_id'
    said = ' '.join(str(w.message) for w in recwarn)
    assert 'identifies a row 2 time(s)' in said
    assert "unique together with 'line_no'" in said


def test_a_repeated_key_with_no_composite_says_so_instead(recwarn):
    """No column completes it, so there is nothing to suggest and the message
    must not invent one."""
    frame = pd.DataFrame({'user_id': list(range(200)), 'v': [1.0] * 200})
    frame.loc[[7, 8], 'user_id'] = 6

    schema = Schema.from_data({'t': frame})

    assert schema['t'].primary_key == 'user_id'
    said = ' '.join(str(w.message) for w in recwarn)
    assert 'unique together with' not in said
    assert 'foreign key rather than an identifier' in said


def test_a_name_match_that_cannot_join_says_why_it_was_dropped():
    """Dropping it is right, because joining int64 to string matches nothing.
    Doing it silently left the query refused for a foreign key that looks as
    though it should exist."""
    import warnings

    import pandas as pd

    from sdm_client import Schema

    movies = pd.DataFrame(
        {
            'movieID': [str(i) for i in range(60)],
            'title': [f't{i}' for i in range(60)],
        }
    )
    ratings = pd.DataFrame(
        {
            'ratingID': range(400),
            'movieID': [i % 60 for i in range(400)],
            'when': pd.date_range('2020-01-01', periods=400, freq='D'),
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        built = Schema.from_data(
            {'movies': movies, 'ratings': ratings},
            primary_keys={'movies': 'movieID', 'ratings': 'ratingID'},
            time_columns={'ratings': 'when'},
        )

    assert built.relationships == []
    assert any('cannot match' in str(item.message) for item in caught)


def a_child_of_a_declared_parent() -> dict[str, pd.DataFrame]:
    """A fact table whose foreign key is nearly unique, as a sample makes it."""
    orders = pd.DataFrame({'order_id': range(200), 'total': [1.0] * 200})
    line_items = pd.DataFrame(
        {
            'order_id': [*range(192), *range(8)],
            'qty': [1] * 200,
        }
    )
    return {'orders': orders, 'line_items': line_items}


def test_a_child_of_a_declared_parent_is_not_told_its_key_repeats(recwarn):
    """The repeats are what a foreign key is, the edge through it works, and
    the printed remedy deletes it."""
    schema = Schema.from_data(a_child_of_a_declared_parent())

    assert schema['line_items'].primary_key is None
    assert edges(schema) == {('line_items', 'order_id', 'orders', 'order_id')}
    said = ' '.join(str(w.message) for w in recwarn)
    assert 'does not identify a row' not in said


def test_the_remedy_that_was_printed_here_costs_the_edge():
    """Why the warning above must not fire: de-duplicating makes the foreign
    key unique, which makes it this table's key too, which makes the name
    ambiguous, which deletes the edge that was working."""
    tables = a_child_of_a_declared_parent()
    thinned = {
        'orders': tables['orders'],
        'line_items': tables['line_items'].drop_duplicates('order_id'),
    }
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        followed = Schema.from_data(thinned)

    assert len(thinned['line_items']) == 192
    assert edges(followed) == set()


def test_no_table_is_told_both_that_it_has_a_key_shaped_column_and_that_it_has_none(
    recwarn,
):
    """One call printed both, with remedies that cannot both be followed."""
    account = pd.DataFrame(
        {
            'ACCOUNT_ID': [*range(190), *range(10)],
            'balance': [1.0] * 200,
            'code': [f'c{i}' for i in range(200)],
        }
    )
    entries = pd.DataFrame(
        {
            'ENTRY_ID': range(400),
            'ACCOUNT_ID': [i % 190 for i in range(400)],
            'code': [f'c{i % 200}' for i in range(400)],
        }
    )
    schema = Schema.from_data({'account': account, 'entries': entries})

    assert schema['account'].primary_key is None
    said = [str(w.message) for w in recwarn if 'account' in str(w.message)]
    assert any('is named like a key' in line for line in said), said
    assert not any('has no primary key' in line for line in said), said


def test_no_warning_says_a_keyless_table_reaches_the_model_unjoined(recwarn):
    """It does not reach the model at all through an edge, and where it names
    another table it is sent as that table's child."""
    Schema.from_data(
        {
            'account': pd.DataFrame(
                {'code': [f'c{i}' for i in range(200)], 'v': [1.0] * 200}
            ),
            'entries': pd.DataFrame(
                {'eid': range(200), 'code': [f'c{i}' for i in range(200)]}
            ),
        }
    )
    assert 'unjoined' not in ' '.join(str(w.message) for w in recwarn)


def test_the_shared_key_name_warning_does_not_leave_out_the_entity_table(
    recwarn,
):
    """The table a question is about is always in the request."""
    Schema.from_data(
        {
            'alpha': pd.DataFrame({'ID': range(20), 'a': [1.0] * 20}),
            'beta': pd.DataFrame({'ID': range(20), 'b': [1.0] * 20}),
        },
        primary_keys={'alpha': 'ID', 'beta': 'ID'},
    )
    said = ' '.join(str(w.message) for w in recwarn)
    assert 'each of them is left out of the request' not in said
    assert 'answered from that table alone' in said


def test_the_composite_key_offered_is_one_that_would_be_accepted(recwarn):
    """The recommender omitted the null check the acceptor enforces, so a
    column with one null was offered and refused on the very next call."""
    frame = pd.DataFrame(
        {'uid': list(range(200)), 'line': [1] * 200, 'v': [1.0] * 200}
    )
    frame.loc[[7, 8], 'uid'] = 6
    frame.loc[[7, 8], 'line'] = [2, 3]
    frame['maybe'] = [None] + [f'x{i}' for i in range(199)]

    Schema.from_data({'t': frame})

    said = ' '.join(str(item.message) for item in recwarn)
    assert "'maybe'" not in said
    if 'unique together with' in said:
        offered = said.split('unique together with')[1].split("'")[1]
        Schema.from_data({'t': frame}, primary_keys={'t': ['uid', offered]})


def test_a_composite_key_is_named_as_the_columns_it_is():
    """Joining the parts with a dot rendered ['a','b'] of table 't' as
    't.a.b', which reads as one column named 'a.b'."""
    frame = pd.DataFrame(
        {'a': [1, 2, None, 4] * 5, 'b': range(20), 'v': [1.0] * 20}
    )

    with pytest.raises(StructuredDataError) as raised:
        Schema.from_data({'t': frame}, primary_keys={'t': ['a', 'b']})

    said = str(raised.value)
    assert 't.a.b' not in said
    assert "the primary key of 't'" in said
    assert "in 'a'" in said


def test_a_single_column_key_is_still_named_plainly():
    frame = pd.DataFrame({'a': [1, 2, None, 4] * 5, 'v': [1.0] * 20})

    with pytest.raises(StructuredDataError) as raised:
        Schema.from_data({'t': frame}, primary_keys={'t': 'a'})

    assert 't.a has' in str(raised.value)
