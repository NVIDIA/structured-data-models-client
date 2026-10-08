# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One table can reach another by more than one column, and often should.

A sales line names the customer it was sold to and the one it ships to, and
those are different facts. But an aggregation across a pair of tables has to
resolve to one column, so a query crossing an ambiguous pair is refused, and
it was refused at the moment the query was asked, which reads as a problem
with the query rather than with the graph it was written against. Worse, the
one instruction that would fix it named `graph.unlink`, which a caller holding
a `Schema` does not have.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sdm_client import KumoRelational, PredictiveQuery, Relationship, Schema
from sdm_client.errors import StructuredDataError

RNG = np.random.default_rng(0)
QUERY = (
    'PREDICT COUNT(salesdocumentitem.*, 0, 30, days) FOR EACH customer.CUSTOMER'
)


def schema_reaching_by(columns: list[str], *, quiet: bool = True) -> Schema:
    """`quiet` off for the tests that read what building it says: suppressing
    warnings inside the helper is what makes such a test catch nothing."""
    customer = pd.DataFrame(
        {'CUSTOMER': range(60), 'seg': RNG.choice([*'xy'], 60)}
    )
    item = {
        'ID': range(400),
        'amt': RNG.gamma(2.0, 10.0, 400),
        'ts': pd.to_datetime('2023-01-01')
        + pd.to_timedelta(RNG.integers(0, 150, 400), 'D'),
    }
    for column in columns:
        item[column] = RNG.integers(0, 60, 400)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore' if quiet else 'always')
        return Schema.from_data(
            {'customer': customer, 'salesdocumentitem': pd.DataFrame(item)},
            primary_keys={'customer': 'CUSTOMER', 'salesdocumentitem': 'ID'},
            time_columns={'salesdocumentitem': 'ts'},
            relationships=[
                Relationship('salesdocumentitem', c, 'customer', 'CUSTOMER')
                for c in columns
            ],
        )


def said_building(columns: list[str]) -> str:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        schema_reaching_by(columns, quiet=False)
    return ' '.join(
        str(w.message) for w in caught if 'reaches' in str(w.message)
    )


def test_it_is_said_when_the_graph_is_built_not_when_a_query_runs():
    said = said_building(['SOLDTOPARTY', 'SHIPTOPARTY'])

    assert "'salesdocumentitem' reaches 'customer'" in said
    assert "'SHIPTOPARTY' and 'SOLDTOPARTY'" in said


def test_one_way_to_reach_a_table_says_nothing():
    assert said_building(['SOLDTOPARTY']) == ''


def test_every_competing_column_is_named():
    said = said_building(
        ['SOLDTOPARTY', 'SHIPTOPARTY', 'BILLTOPARTY', 'PAYERPARTY']
    )

    for column in ('SOLDTOPARTY', 'SHIPTOPARTY', 'BILLTOPARTY', 'PAYERPARTY'):
        assert repr(column) in said
    assert 'They are all kept' in said


def test_nothing_is_dropped_on_the_caller_s_behalf():
    """Which link a question means decides what the model sees, so it stays
    the caller's to make."""
    schema = schema_reaching_by(['SOLDTOPARTY', 'SHIPTOPARTY'])

    assert len(schema.relationships) == 2


def test_the_refusal_names_something_a_schema_holder_can_do():
    pytest.importorskip('sdm_relational_engine')

    schema = schema_reaching_by(['SOLDTOPARTY', 'SHIPTOPARTY'])

    with pytest.raises(StructuredDataError) as refused:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            PredictiveQuery(QUERY, schema).materialize(
                indices=[1, 2], model=KumoRelational()
            )

    said = str(refused.value)
    assert (
        'schema.relationships = [link for link in schema.relationships' in said
    )
    assert "'SOLDTOPARTY'" in said


def test_following_the_advice_makes_the_query_resolve():
    """The whole point: the instruction has to work."""
    pytest.importorskip('sdm_relational_engine')

    schema = schema_reaching_by(['SOLDTOPARTY', 'SHIPTOPARTY'])
    schema.relationships = [
        link
        for link in schema.relationships
        if link.source_column not in ['SOLDTOPARTY']
    ]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = PredictiveQuery(QUERY, schema).materialize(
            indices=[1, 2], model=KumoRelational()
        )

    assert request.predict_rows == 2


# --- the advice has to fix this pair, and only this pair -------------------


def two_ambiguous_pairs() -> Schema:
    customers = pd.DataFrame({'id': range(40), 'seg': RNG.choice([*'xy'], 40)})
    users = pd.DataFrame({'uid': range(40), 'k': RNG.choice([*'xy'], 40)})
    when = pd.to_datetime('2023-01-01') + pd.to_timedelta(
        RNG.integers(0, 90, 200), 'D'
    )
    line = pd.DataFrame(
        {
            'lid': range(200),
            'sold_to': RNG.integers(0, 40, 200),
            'ship_to': RNG.integers(0, 40, 200),
            'amt': RNG.gamma(2.0, 5.0, 200),
            'ts': when,
        }
    )
    message = pd.DataFrame(
        {
            'mid': range(200),
            'sender_id': RNG.integers(0, 40, 200),
            'recipient_id': RNG.integers(0, 40, 200),
            'ts': when,
        }
    )
    ticket = pd.DataFrame(
        {'tid': range(50), 'ship_to': RNG.integers(0, 40, 50)}
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data(
            {
                'customers': customers,
                'users': users,
                'line': line,
                'message': message,
                'ticket': ticket,
            },
            primary_keys={
                'customers': 'id',
                'users': 'uid',
                'line': 'lid',
                'message': 'mid',
                'ticket': 'tid',
            },
            time_columns={'line': 'ts', 'message': 'ts'},
            relationships=[
                Relationship('line', 'sold_to', 'customers', 'id'),
                Relationship('line', 'ship_to', 'customers', 'id'),
                Relationship('message', 'sender_id', 'users', 'uid'),
                Relationship('message', 'recipient_id', 'users', 'uid'),
                Relationship('ticket', 'ship_to', 'customers', 'id'),
            ],
        )


def test_the_advice_names_the_pair_the_query_crossed():
    """Two pairs are ambiguous. Advising the first one found sends a caller to
    edit a relationship their query never touched."""
    pytest.importorskip('sdm_relational_engine')

    with pytest.raises(StructuredDataError) as refused:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            PredictiveQuery(
                'PREDICT COUNT(message.*, 0, 30, days) FOR EACH users.uid',
                two_ambiguous_pairs(),
            ).materialize(indices=[1, 2], model=KumoRelational())

    advice = str(refused.value)
    advice = advice[advice.index('In this schema') :]
    assert "'message' reaches 'users'" in advice
    assert 'line' not in advice


def test_following_the_advice_leaves_another_table_s_column_alone():
    """`ticket` also has a `ship_to`. Dropping by column name everywhere would
    take it as well, and nothing about the query asked for that."""
    schema = two_ambiguous_pairs()

    kept = [
        link
        for link in schema.relationships
        if not (
            link.source_table == 'line'
            and link.target_table == 'customers'
            and link.source_column in ['sold_to']
        )
    ]

    assert any(link.source_table == 'ticket' for link in kept)
    assert not any(
        link.source_table == 'line' and link.source_column == 'sold_to'
        for link in kept
    )


def test_the_remedy_is_written_scoped_to_the_pair():
    said = said_building(['SOLDTOPARTY', 'SHIPTOPARTY'])

    assert 'link.source_table ==' in said
    assert 'link.target_table ==' in said


def caught(build) -> list[str]:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        build()
    return [str(item.message) for item in record]


def test_a_table_that_points_at_itself_is_one_link_and_not_a_pair():
    """A manager, a parent comment, a parent category: an ordinary shape, and
    the test for a reverse edge is satisfied by the edge itself."""
    users = pd.DataFrame(
        {'uid': range(20), 'manager_id': [max(0, i - 1) for i in range(20)]}
    )
    said = caught(
        lambda: Schema.from_data(
            {'users': users},
            primary_keys={'users': 'uid'},
            relationships=[Relationship('users', 'manager_id', 'users', 'uid')],
        )
    )
    assert not [line for line in said if 'name each other' in line]


def test_two_tables_that_name_each_other_are_still_reported():
    users = pd.DataFrame({'uid': range(20), 'top_order': range(20)})
    orders = pd.DataFrame({'oid': range(20), 'uid': range(20)})
    said = caught(
        lambda: Schema.from_data(
            {'users': users, 'orders': orders},
            primary_keys={'users': 'uid', 'orders': 'oid'},
            relationships=[
                Relationship('orders', 'uid', 'users', 'uid'),
                Relationship('users', 'top_order', 'orders', 'oid'),
            ],
        )
    )
    assert [line for line in said if 'name each other' in line]


def test_appending_the_reverse_edge_is_reported_too():
    """The route the client's own advice prints ran no check at all."""
    users = pd.DataFrame({'uid': range(20), 'top_order': range(20)})
    orders = pd.DataFrame({'oid': range(20), 'uid': range(20)})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'users': users, 'orders': orders},
            primary_keys={'users': 'uid', 'orders': 'oid'},
            relationships=[Relationship('orders', 'uid', 'users', 'uid')],
        )

    said = caught(
        lambda: schema.relationships.append(
            Relationship('users', 'top_order', 'orders', 'oid')
        )
    )
    assert [line for line in said if 'name each other' in line]


@pytest.mark.parametrize(
    'add',
    [
        lambda links, link: links.append(link),
        lambda links, link: links.extend([link]),
        lambda links, link: links.insert(0, link),
        lambda links, link: links.__iadd__([link]),
        lambda links, link: links.__setitem__(slice(0, 0), [link]),
    ],
    ids=['append', 'extend', 'insert', 'in-place add', 'slice assignment'],
)
def test_a_duplicate_link_is_dropped_however_it_is_added(add):
    users = pd.DataFrame({'uid': range(20)})
    orders = pd.DataFrame({'oid': range(20), 'uid': range(20)})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'users': users, 'orders': orders},
            primary_keys={'users': 'uid', 'orders': 'oid'},
            relationships=[Relationship('orders', 'uid', 'users', 'uid')],
        )

    add(schema.relationships, Relationship('orders', 'uid', 'users', 'uid'))
    assert len(schema.relationships) == 1


def test_repeating_the_list_in_place_does_not_repeat_the_edge():
    users = pd.DataFrame({'uid': range(20)})
    orders = pd.DataFrame({'oid': range(20), 'uid': range(20)})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'users': users, 'orders': orders},
            primary_keys={'users': 'uid', 'orders': 'oid'},
            relationships=[Relationship('orders', 'uid', 'users', 'uid')],
        )

    schema.relationships.__imul__(2)
    assert len(schema.relationships) == 1
