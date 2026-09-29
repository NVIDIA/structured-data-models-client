# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A schema asks the question its present state describes, not its history.

Two schemas with the same tables and no relationships asked different
questions -- one a regression, the other a multiclass classification --
because one of them had once had a link. Key columns were typed `ID` when the
schema was built and never re-derived, and the marking went through the same
assignment a caller uses, so it set `stype_declared` and became
indistinguishable from a meaning the caller chose, with nothing recorded to
restore.
"""

from __future__ import annotations

import copy
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from sdm_client import Relationship, Schema
from sdm_client.errors import StructuredDataError

RNG = np.random.default_rng(0)
USERS = pd.DataFrame({'user_id': range(60), 'age': RNG.integers(20, 70, 60)})
ORDERS = pd.DataFrame(
    {
        'order_id': range(300),
        'user_id': RNG.integers(0, 60, 300),
        'price': RNG.random(300) * 100,
    }
)
LINK = Relationship('orders', 'user_id', 'users', 'user_id')


def build(links: list[Relationship]) -> Schema:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data(
            {'users': USERS, 'orders': ORDERS},
            primary_keys={'users': 'user_id', 'orders': 'order_id'},
            relationships=links,
        )


def test_a_link_that_is_gone_no_longer_types_the_column_it_joined_on() -> None:
    linked = build([LINK])
    assert linked['orders']['user_id'].stype == 'ID'

    linked.relationships = []
    assert (
        linked['orders']['user_id'].stype
        == build([])['orders']['user_id'].stype
    )


@pytest.mark.parametrize(
    'drop',
    [
        lambda links: links.clear(),
        lambda links: links.pop(),
        lambda links: links.remove(LINK),
        lambda links: links.__delitem__(0),
    ],
    ids=['clear', 'pop', 'remove', 'del'],
)
def test_every_way_of_dropping_a_link_re_derives_the_key_columns(drop) -> None:
    schema = build([LINK])
    drop(schema.relationships)
    assert not schema.relationships
    assert schema['orders']['user_id'].stype == 'numerical'


def test_a_link_added_later_types_its_column_as_a_key() -> None:
    schema = build([])
    assert schema['orders']['user_id'].stype == 'numerical'

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema.relationships.append(LINK)

    assert schema['orders']['user_id'].stype == 'ID'


def test_the_schema_s_own_marking_is_not_recorded_as_the_caller_s_choice() -> (
    None
):
    linked = build([LINK])
    assert linked['orders']['user_id'].stype == 'ID'
    assert not linked['orders']['user_id'].stype_declared


def test_a_meaning_the_caller_chose_survives_a_link_coming_and_going() -> None:
    schema = build([])
    schema['orders']['user_id'].stype = 'ID'
    assert schema['orders']['user_id'].stype_declared

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema.relationships = [LINK]
    schema.relationships = []

    assert schema['orders']['user_id'].stype == 'ID'
    assert schema['orders']['user_id'].stype_declared


def test_a_primary_key_stays_an_identifier_with_no_relationships_at_all() -> (
    None
):
    schema = build([])
    assert schema['users']['user_id'].stype == 'ID'
    assert schema['orders']['order_id'].stype == 'ID'


def test_a_copy_carries_what_the_marking_had_to_restore() -> None:
    linked = build([LINK])

    for copied in (copy.deepcopy(linked), pickle.loads(pickle.dumps(linked))):
        assert copied['orders']['user_id'].stype == 'ID'
        copied.relationships = []
        assert copied['orders']['user_id'].stype == 'numerical'


ALT = pd.DataFrame(
    {
        'order_id': range(300),
        'alt_id': range(1_000, 1_300),
        'user_id': RNG.integers(0, 60, 300),
    }
)


def with_alt(links: list[Relationship]) -> Schema:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data(
            {'users': USERS, 'orders': ALT},
            primary_keys={'users': 'user_id', 'orders': 'order_id'},
            relationships=links,
        )


def test_a_primary_key_that_was_replaced_stops_typing_its_old_column() -> None:
    schema = with_alt([])
    assert schema['orders']['order_id'].stype == 'ID'

    schema['orders'].primary_key = 'alt_id'

    assert schema['orders']['alt_id'].stype == 'ID'
    assert schema['orders']['order_id'].stype == 'numerical'


def test_a_replaced_key_a_link_still_joins_on_stays_an_identifier() -> None:
    schema = with_alt([Relationship('orders', 'order_id', 'users', 'user_id')])

    schema['orders'].primary_key = 'alt_id'

    assert schema['orders']['order_id'].stype == 'ID'


def test_clearing_the_primary_key_releases_the_column_it_named() -> None:
    schema = with_alt([])
    schema['orders'].primary_key = None
    assert schema['orders']['order_id'].stype == 'numerical'


def test_a_table_on_its_own_answers_for_its_own_key() -> None:
    schema = with_alt([])
    table = schema['orders']
    object.__setattr__(table, '_schema', None)
    assert table['order_id'].stype == 'ID'

    table.primary_key = 'alt_id'

    assert table['alt_id'].stype == 'ID'
    assert table['order_id'].stype == 'numerical'


def test_a_declared_meaning_does_not_survive_becoming_a_key() -> None:
    """What a key column means is settled by its being a key. Honouring a
    declared `numerical` there sent a join key as a float, which is the
    failure the marking exists to prevent."""
    schema = with_alt([])
    schema['orders']['alt_id'].stype = 'numerical'

    schema['orders'].primary_key = 'alt_id'
    assert schema['orders']['alt_id'].stype == 'ID'

    schema['orders'].primary_key = 'order_id'
    assert schema['orders']['alt_id'].stype == 'numerical'


def test_a_declared_meaning_does_not_survive_a_link_either() -> None:
    schema = with_alt([])
    schema['orders']['user_id'].stype = 'numerical'

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema.relationships = [
            Relationship('orders', 'user_id', 'users', 'user_id')
        ]
    assert schema['orders']['user_id'].stype == 'ID'

    schema.relationships = []
    assert schema['orders']['user_id'].stype == 'numerical'


def test_a_key_column_replaced_through_the_columns_dict_is_asked_again() -> (
    None
):
    """`tables` and a table's `columns` are ordinary dictionaries, so a table
    or a column replaced through one arrives never having been asked, and
    `check_consistent()` is where a caller asks whether their schema holds."""
    from sdm_client.schema import Column

    schema = with_alt([Relationship('orders', 'user_id', 'users', 'user_id')])
    schema['orders'].columns['user_id'] = Column(
        'user_id', 'int64', 'numerical'
    )
    assert schema['orders']['user_id'].stype == 'numerical'

    schema.check_consistent()

    assert schema['orders']['user_id'].stype == 'ID'


def test_a_key_this_cannot_read_is_refused_before_the_table_takes_it() -> None:
    """The refusal ran after the assignment, leaving a table that could not be
    copied or sent and a `check_consistent()` that passed anyway."""
    schema = with_alt([])
    before = schema['orders'].primary_key

    with pytest.raises(StructuredDataError):
        schema['orders'].primary_key = {'alt_id'}

    assert schema['orders'].primary_key == before
    schema.snapshot()


def test_declaring_a_meaning_on_a_live_key_does_not_change_what_it_is() -> None:
    """A key's meaning is settled by its being a key, and the schema said so
    wherever it was asked -- but only re-derived when the keys changed, so
    between declaring a meaning on a live key and the next such moment the
    column described itself as whatever was declared."""
    schema = with_alt([Relationship('orders', 'user_id', 'users', 'user_id')])

    schema['orders']['user_id'].stype = 'categorical'
    assert schema['orders']['user_id'].stype == 'ID'

    schema['orders']['order_id'].stype = 'categorical'
    assert schema['orders']['order_id'].stype == 'ID'

    schema.relationships = []
    assert schema['orders']['user_id'].stype == 'categorical'


def test_a_meaning_declared_on_an_ordinary_column_is_kept() -> None:
    schema = with_alt([])
    schema['orders']['alt_id'].stype = 'categorical'
    assert schema['orders']['alt_id'].stype == 'categorical'
    assert schema['orders']['alt_id'].stype_declared


def test_declaring_id_on_a_key_leaves_nothing_to_hand_back() -> None:
    """The declaration replaces what there was to give back. Recorded beside
    it, the meaning inference had found was handed back over the caller's own
    word when the key moved elsewhere."""
    schema = with_alt([])
    schema['orders']['order_id'].stype = 'ID'

    schema['orders'].primary_key = 'alt_id'

    assert schema['orders']['order_id'].stype == 'ID'


def test_a_key_the_caller_never_spoke_for_is_still_handed_back() -> None:
    schema = with_alt([])
    schema['orders'].primary_key = 'alt_id'
    assert schema['orders']['order_id'].stype == 'numerical'


def test_a_replacement_column_learns_which_table_it_is_in() -> None:
    """`_mark_key_columns` bound each table to its schema but not each column
    to its table, so a column replaced through the ordinary dictionary could
    be repaired once and never asked again."""
    from sdm_client.schema import Column

    schema = with_alt([])
    schema['orders'].columns['order_id'] = Column(
        'order_id', 'int64', 'numerical'
    )
    schema.check_consistent()
    assert schema['orders']['order_id'].stype == 'ID'

    schema['orders']['order_id'].stype = 'categorical'
    assert schema['orders']['order_id'].stype == 'ID'


def test_a_shallow_copy_is_a_schema_of_its_own() -> None:
    """`copy.copy` shared the tables and their columns, which was always a
    trap and became a destructive one once key columns were re-derived:
    clearing the copy's relationships retyped a column of a schema that still
    had them."""
    import copy as copying

    original = build([LINK])
    copied = copying.copy(original)

    copied.relationships = []

    assert original.relationships
    assert original['orders']['user_id'].stype == 'ID'
    assert copied['orders']['user_id'].stype == 'numerical'


def test_editing_a_shallow_copy_leaves_the_original_alone() -> None:
    import copy as copying

    original = build([])
    copied = copying.copy(original)

    copied['orders']['price'].stype = 'categorical'

    assert original['orders']['price'].stype != 'categorical'
