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
