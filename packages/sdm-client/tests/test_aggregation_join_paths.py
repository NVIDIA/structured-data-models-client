# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which tables an aggregation may be asked about.

A star schema is the commonest relational shape there is, and only one child of
a parent could be aggregated: whichever relationship happened to be declared
last. The refusal said "there is no foreign key from bureau to application" at a
moment when there was exactly one, and reversing the declaration order swapped
which child worked -- same tables, same rows, same query.

A table two hops away cannot be aggregated either, and that one is a real limit
rather than a defect: an aggregation joins on a single key and the language has
no form for a path. But the refusal was the same "there is no foreign key"
sentence, which sends the caller to add a key that already exists one hop away.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from sdm_client import KumoRelational, PredictiveQuery, Relationship, Schema
from sdm_client.errors import StructuredDataError

DAYS = pd.to_datetime('1997-01-01') + pd.to_timedelta(
    [i % 300 for i in range(800)], unit='D'
)


def frames() -> dict[str, pd.DataFrame]:
    return {
        'A': pd.DataFrame(
            {'a_id': range(200), 'a_val': [i % 5 for i in range(200)]}
        ),
        'B': pd.DataFrame(
            {
                'b_id': range(400),
                'a_id': [i % 200 for i in range(400)],
                'b_amt': [float(i) for i in range(400)],
                'b_when': DAYS[:400],
            }
        ),
        'C': pd.DataFrame(
            {
                'c_id': range(800),
                'b_id': [i % 400 for i in range(800)],
                'c_amt': [float(i) for i in range(800)],
                'c_when': DAYS,
            }
        ),
        'D': pd.DataFrame(
            {
                'd_id': range(400),
                'a_id': [i % 200 for i in range(400)],
                'd_amt': [float(i) for i in range(400)],
                'd_when': DAYS[:400],
            }
        ),
        'E': pd.DataFrame(
            {
                'e_id': range(200),
                'a_id': [i % 200 for i in range(200)],
                'e_amt': [float(i) for i in range(200)],
                'e_when': DAYS[:200],
            }
        ),
    }


KEYS = {'A': 'a_id', 'B': 'b_id', 'C': 'c_id', 'D': 'd_id', 'E': 'e_id'}
TIMES = {'B': 'b_when', 'C': 'c_when', 'D': 'd_when', 'E': 'e_when'}


def build(tables, links) -> Schema:
    data = {name: frames()[name] for name in tables}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data(
            data,
            relationships=links,
            primary_keys={k: v for k, v in KEYS.items() if k in data},
            time_columns={k: v for k, v in TIMES.items() if k in data},
        )


def materialize(query: str, tables, links):
    return PredictiveQuery(query, build(tables, links)).materialize(
        indices=[1, 2], model=KumoRelational()
    )


# --- every child of a parent, not just the last one declared ---------------


ONE = Relationship('B', 'a_id', 'A', 'a_id')
TWO = Relationship('D', 'a_id', 'A', 'a_id')
THREE = Relationship('E', 'a_id', 'A', 'a_id')


@pytest.mark.parametrize('order', [[ONE, TWO], [TWO, ONE]], ids=['B,D', 'D,B'])
@pytest.mark.parametrize('child', ['B.b_amt', 'D.d_amt'])
def test_any_child_of_a_parent_can_be_aggregated(order, child):
    """Which one worked used to be decided by declaration order, silently."""
    request = materialize(
        f'PREDICT SUM({child}, 0, 30, days) FOR EACH A.a_id',
        ['A', 'B', 'D'],
        order,
    )
    assert request.predict_rows == 2


@pytest.mark.parametrize('child', ['B.b_amt', 'D.d_amt', 'E.e_amt'])
def test_three_children_all_work(child):
    request = materialize(
        f'PREDICT SUM({child}, 0, 30, days) FOR EACH A.a_id',
        ['A', 'B', 'D', 'E'],
        [ONE, TWO, THREE],
    )
    assert request.predict_rows == 2


def test_two_keys_from_one_table_are_still_ambiguous():
    """The check this must not lose: a table holding two foreign keys into the
    same parent has no unique join, and guessing would change the question."""
    users = pd.DataFrame(
        {'user_id': range(120), 'seg': [i % 3 for i in range(120)]}
    )
    messages = pd.DataFrame(
        {
            'm_id': range(240),
            'sender_id': [i % 120 for i in range(240)],
            'recipient_id': [(i * 7) % 120 for i in range(240)],
            'amt': [float(i) for i in range(240)],
            'when': DAYS[:240],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'users': users, 'messages': messages},
            relationships=[
                Relationship('messages', 'sender_id', 'users', 'user_id'),
                Relationship('messages', 'recipient_id', 'users', 'user_id'),
            ],
            primary_keys={'users': 'user_id', 'messages': 'm_id'},
            time_columns={'messages': 'when'},
        )
    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery(
            'PREDICT SUM(messages.amt, 0, 30, days) FOR EACH users.user_id',
            schema,
        ).materialize(indices=[1, 2], model=KumoRelational())
    message = str(raised.value)
    assert 'is not unique' in message, message
    assert 'messages.sender_id' in message
    assert 'messages.recipient_id' in message


def test_two_keys_in_separate_groups_are_both_seen():
    """They arrive as two column groups, one per relationship, so the keys
    have to be accumulated across groups: taking the last group's answer sees
    one key and calls the join unique when it is not."""
    users = pd.DataFrame(
        {'user_id': range(120), 'seg': [i % 3 for i in range(120)]}
    )
    messages = pd.DataFrame(
        {
            'm_id': range(240),
            'sender_id': [i % 120 for i in range(240)],
            'recipient_id': [(i * 7) % 120 for i in range(240)],
            'amt': [float(i) for i in range(240)],
            'when': DAYS[:240],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'users': users, 'messages': messages},
            relationships=[
                Relationship('messages', 'sender_id', 'users', 'user_id'),
                Relationship('messages', 'recipient_id', 'users', 'user_id'),
            ],
            primary_keys={'users': 'user_id', 'messages': 'm_id'},
            time_columns={'messages': 'when'},
        )
    from sdm_client.schema import to_graph_definition

    groups = [
        {(c.table_name, c.col_name) for c in group.columns}
        for group in to_graph_definition(schema).col_groups
    ]
    holding = [g for g in groups if ('users', 'user_id') in g]
    assert len(holding) == 2, groups


def test_one_key_from_one_table_is_not_ambiguous():
    request = materialize(
        'PREDICT SUM(B.b_amt, 0, 30, days) FOR EACH A.a_id', ['A', 'B'], [ONE]
    )
    assert request.predict_rows == 2


# --- a table further away is refused for the right reason ------------------


CHAIN = [
    Relationship('B', 'a_id', 'A', 'a_id'),
    Relationship('C', 'b_id', 'B', 'b_id'),
]


def test_each_single_hop_of_a_chain_works():
    assert (
        materialize(
            'PREDICT SUM(B.b_amt, 0, 30, days) FOR EACH A.a_id',
            ['A', 'B', 'C'],
            CHAIN,
        ).predict_rows
        == 2
    )
    assert (
        materialize(
            'PREDICT SUM(C.c_amt, 0, 30, days) FOR EACH B.b_id',
            ['A', 'B', 'C'],
            CHAIN,
        ).predict_rows
        == 2
    )


def test_two_hops_are_refused_by_naming_the_route():
    """An aggregation joins on one key, so this cannot be expressed. Saying
    "there is no foreign key from C to A" was false and sent the caller to add
    one that already exists a hop away."""
    with pytest.raises(StructuredDataError) as raised:
        materialize(
            'PREDICT SUM(C.c_amt, 0, 30, days) FOR EACH A.a_id',
            ['A', 'B', 'C'],
            CHAIN,
        )
    message = str(raised.value)
    assert 'C reaches A through C -> B -> A' in message
    assert 'there is no foreign key' not in message.lower()
    assert 'Aggregate C for each B instead' in message


def test_the_remedy_the_refusal_names_actually_works():
    """The one thing that makes naming the route worth doing."""
    assert (
        materialize(
            'PREDICT SUM(C.c_amt, 0, 30, days) FOR EACH B.b_id',
            ['A', 'B', 'C'],
            CHAIN,
        ).predict_rows
        == 2
    )


def test_a_table_with_no_route_keeps_the_original_message():
    """When there really is no foreign key, that is the right thing to say."""
    unrelated = pd.DataFrame(
        {
            'z_id': range(200),
            'z_amt': [float(i) for i in range(200)],
            'z_when': DAYS[:200],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'A': frames()['A'], 'B': frames()['B'], 'Z': unrelated},
            relationships=[ONE],
            primary_keys={'A': 'a_id', 'B': 'b_id', 'Z': 'z_id'},
            time_columns={'B': 'b_when', 'Z': 'z_when'},
        )
    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery(
            'PREDICT SUM(Z.z_amt, 0, 30, days) FOR EACH A.a_id', schema
        ).materialize(indices=[1, 2], model=KumoRelational())
    message = str(raised.value)
    assert 'There is no foreign key from table Z to table A' in message
    assert 'reaches' not in message


# --- a route that cannot be travelled is not offered ------------------------


def test_aggregating_a_parent_for_each_child_keeps_the_plain_message():
    """`A -> B` is the reverse of `B -> A`. An undirected walk finds it and
    advises aggregating A for each B, which is the query that was just
    refused."""
    A = frames()['A'].assign(a_when=DAYS[:200])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'A': A, 'B': frames()['B']},
            relationships=[ONE],
            primary_keys={'A': 'a_id', 'B': 'b_id'},
            time_columns={'A': 'a_when', 'B': 'b_when'},
        )
    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery(
            'PREDICT SUM(A.a_val, 0, 30, days) FOR EACH B.b_id', schema
        ).materialize(indices=[1, 2], model=KumoRelational())
    message = str(raised.value)
    assert 'reaches' not in message
    assert 'There is no foreign key from table A to table B' in message


def test_two_children_of_one_parent_are_not_routed_through_it():
    """`C -> A <- D` joins each child to the parent and to nothing else, so
    `C reaches D through C -> A -> D` is a join that does not exist."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {name: frames()[name] for name in ('A', 'B', 'D')},
            relationships=[ONE, TWO],
            primary_keys={'A': 'a_id', 'B': 'b_id', 'D': 'd_id'},
            time_columns={'B': 'b_when', 'D': 'd_when'},
        )
    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery(
            'PREDICT SUM(B.b_amt, 0, 30, days) FOR EACH D.d_id', schema
        ).materialize(indices=[1, 2], model=KumoRelational())
    message = str(raised.value)
    assert 'reaches' not in message
    assert 'There is no foreign key from table B to table D' in message


def test_the_route_is_only_offered_upward():
    """Directly, on the helper, over the three shapes at once."""
    from sdm_client.schema import to_graph_definition

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {name: frames()[name] for name in ('A', 'B', 'C', 'D')},
            relationships=[ONE, TWO, Relationship('C', 'b_id', 'B', 'b_id')],
            primary_keys={'A': 'a_id', 'B': 'b_id', 'C': 'c_id', 'D': 'd_id'},
            time_columns={'B': 'b_when', 'C': 'c_when', 'D': 'd_when'},
        )

    from predictive_query.validator.join_validator import JoinValidator

    route = JoinValidator(to_graph_definition(schema))._route_between

    assert route('C', 'A') == ['C', 'B', 'A']
    assert route('B', 'A') == ['B', 'A']
    assert route('A', 'C') == []
    assert route('A', 'B') == []
    assert route('C', 'D') == []
    assert route('D', 'C') == []


def test_a_one_to_one_link_names_no_direction():
    """An extension table shares its parent's key, so the group holds two
    primary keys and neither table is the parent. Picking one would make the
    direction depend on set order, which is the same coin-flip that made a
    child's aggregation depend on declaration order."""
    from predictive_query.validator.join_validator import JoinValidator

    from sdm_client.schema import to_graph_definition

    users = pd.DataFrame(
        {'user_id': range(100), 'seg': [i % 3 for i in range(100)]}
    )
    profiles = pd.DataFrame(
        {'user_id': range(100), 'bio_len': [float(i) for i in range(100)]}
    )
    events = pd.DataFrame(
        {
            'e_id': range(200),
            'user_id': [i % 100 for i in range(200)],
            'amt': [float(i) for i in range(200)],
            'when': DAYS[:200],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'users': users, 'profiles': profiles, 'events': events},
            relationships=[
                Relationship('profiles', 'user_id', 'users', 'user_id'),
                Relationship('events', 'user_id', 'users', 'user_id'),
            ],
            primary_keys={
                'users': 'user_id',
                'profiles': 'user_id',
                'events': 'e_id',
            },
            time_columns={'events': 'when'},
        )
    route = JoinValidator(to_graph_definition(schema))._route_between

    assert route('profiles', 'users') == []
    assert route('users', 'profiles') == []
    assert route('profiles', 'events') == []
    assert route('events', 'users') == ['events', 'users']


def test_a_column_two_hops_away_gets_a_column_repair():
    """The same distance and a different question. Reading a column is not
    aggregating rows, so "an aggregation joins on one key" describes neither
    what was asked nor what to do instead."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {name: frames()[name] for name in ('A', 'B', 'C')},
            relationships=CHAIN,
            primary_keys={'A': 'a_id', 'B': 'b_id', 'C': 'c_id'},
            time_columns={'B': 'b_when', 'C': 'c_when'},
        )
    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery('PREDICT A.a_val FOR EACH C.c_id', schema).materialize(
            indices=[1, 2], model=KumoRelational()
        )
    message = str(raised.value)
    assert 'C reaches through C -> B -> A' in message
    assert 'an aggregation joins on one key' not in message
    assert 'Aggregate C for each B' not in message
    assert 'Ask for a column of B instead' in message
