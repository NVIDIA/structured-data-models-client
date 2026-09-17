# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a request describes, and what it was built from.

Four ways the object a caller holds did not say what it appeared to. The same
relationship was refused when passed to `from_data` and accepted when assigned
afterwards, and `check_consistent()` -- which a caller runs to ask exactly this
-- passed it. A materialized request kept the caller's schema by reference, so
editing the schema afterwards changed what was sent. A relational request named
its entity table by dict insertion order. And the `predict=<frame>` route left
rows the table already held in the context, with their answers, where the
`indices=` route removed them.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sdm_client import KumoRelational, PredictiveQuery, Relationship, Schema
from sdm_client.errors import StructuredDataError
from sdm_client.request import Request

RNG = np.random.default_rng(7)


def single(rows: int = 500) -> Schema:
    frame = pd.DataFrame(
        {
            'rid': range(rows),
            'a': RNG.normal(0, 1, rows),
            'label': RNG.integers(0, 2, rows).astype(str),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data({'t': frame}, primary_keys={'t': 'rid'})


def graph(order=('stores', 'events')) -> Schema:
    codes = [f'S{i}' for i in range(20)]
    stores = pd.DataFrame({'sid': codes, 'r': RNG.choice(list('NS'), 20)})
    events = pd.DataFrame(
        {
            'eid': range(400),
            'sid': RNG.choice(codes, 400),
            'amt': RNG.gamma(2.0, 10.0, 400),
            'ts': pd.to_datetime('2023-01-01')
            + pd.to_timedelta(RNG.integers(0, 200, 400), 'D'),
        }
    )
    tables = {'stores': stores, 'events': events}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data(
            {name: tables[name] for name in order},
            primary_keys={'stores': 'sid', 'events': 'eid'},
            time_columns={'events': 'ts'},
            relationships=[Relationship('events', 'sid', 'stores', 'sid')],
        )


# --- a relationship is a relationship however it arrives -------------------


def mismatched() -> tuple[Schema, Relationship]:
    users = pd.DataFrame({'user_id': [1, 2, 3], 'seg': list('abc')})
    orders = pd.DataFrame({'order_id': [1, 2, 3], 'user_id': list('xyz')})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'users': users, 'orders': orders}, relationships=[]
        )
    return schema, Relationship('orders', 'user_id', 'users', 'user_id')


def test_an_assigned_relationship_is_checked_like_a_passed_one():
    """`from_data` refused this link; assignment took it, and
    `check_consistent()` said the schema was fine."""
    schema, link = mismatched()
    with pytest.raises(StructuredDataError) as raised:
        schema.relationships = [link]
    assert 'joins' in str(raised.value)


def test_an_assigned_relationship_naming_nothing_is_refused():
    schema, _ = mismatched()
    with pytest.raises(StructuredDataError) as raised:
        schema.relationships = [
            Relationship('orders', 'nope', 'users', 'user_id')
        ]
    assert "no column 'nope'" in str(raised.value)


def test_a_good_relationship_still_assigns():
    users = pd.DataFrame({'user_id': [1, 2, 3], 'seg': list('abc')})
    orders = pd.DataFrame({'order_id': [1, 2, 3], 'user_id': [1, 2, 3]})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'users': users, 'orders': orders}, relationships=[]
        )
    link = Relationship('orders', 'user_id', 'users', 'user_id')
    schema.relationships = [link]
    assert schema.relationships == [link]


def test_clearing_the_relationships_is_allowed():
    schema, _ = mismatched()
    schema.relationships = []
    assert schema.relationships == []


def test_the_two_routes_now_agree():
    users = pd.DataFrame({'user_id': [1, 2, 3], 'seg': list('abc')})
    orders = pd.DataFrame({'order_id': [1, 2, 3], 'user_id': list('xyz')})
    link = Relationship('orders', 'user_id', 'users', 'user_id')
    with pytest.raises(StructuredDataError):
        Schema.from_data(
            {'users': users, 'orders': orders}, relationships=[link]
        )
    schema, _ = mismatched()
    with pytest.raises(StructuredDataError):
        schema.relationships = [link]


# --- the request keeps what it gathered ------------------------------------


def test_editing_the_schema_afterwards_does_not_change_the_request():
    schema = single()
    request = PredictiveQuery(
        'PREDICT t.label FOR EACH t.rid', schema
    ).materialize(indices=[1, 2])

    schema['t']['a'].stype = 'categorical'

    assert request.schema['t']['a'].stype == 'numerical'
    assert schema['t']['a'].stype == 'categorical'


def test_the_snapshot_shares_the_rows_rather_than_copying_them():
    """The rows are the data and duplicating a warehouse extract per question
    would cost more than the defect it guards against."""
    schema = single()
    copied = schema.snapshot()
    assert copied['t'].data is schema['t'].data


def test_a_snapshot_keeps_keys_time_columns_and_relationships():
    schema = graph()
    copied = schema.snapshot()
    assert copied['stores'].primary_key == 'sid'
    assert copied['events'].time_column == 'ts'
    assert copied.relationships == schema.relationships


def test_a_snapshot_does_not_re_judge_a_schema_mid_edit():
    """A schema is allowed to be inconsistent between two assignments, which
    is what `check_consistent` exists to ask about later."""
    schema = single()
    schema['t']['a'].dtype = 'timestamp[us]'

    copied = schema.snapshot()

    assert copied['t']['a'].dtype == 'timestamp[us]'
    assert copied['t']['a'].stype == 'numerical'


def test_a_snapshot_keeps_what_was_recorded_about_a_column():
    frame = pd.DataFrame(
        {'rid': range(20), 'unknown': [None] * 20, 'v': [1.0] * 20}
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'t': frame}, primary_keys={'t': 'rid'})
    schema['t']['unknown'].stype = 'text'

    copied = schema.snapshot()

    assert copied['t']['unknown'].untyped
    assert copied['t']['unknown'].stype_declared


# --- the entity table is what the query asked about ------------------------


@pytest.mark.parametrize(
    'order', [('stores', 'events'), ('events', 'stores')], ids=['s,e', 'e,s']
)
def test_a_relational_request_names_the_entity_the_query_named(order):
    """It was read from dict insertion order, so it answered correctly or
    incorrectly depending on how the caller built their dict."""
    pytest.importorskip('sdm_relational_engine')

    schema = graph(order)
    request = PredictiveQuery(
        'PREDICT SUM(events.amt, 0, 30, days) FOR EACH stores.sid', schema
    ).materialize(indices=['S0', 'S1'], model=KumoRelational())

    assert request.entity_table == 'stores'
    assert request.task.entity_table == 'stores'


def test_a_single_table_request_still_names_its_table():
    schema = single()
    request = PredictiveQuery(
        'PREDICT t.label FOR EACH t.rid', schema
    ).materialize(indices=[1, 2])
    assert request.entity_table == 't'


# --- both single-table routes agree about what may be shown ----------------


def test_rows_supplied_for_scoring_are_taken_out_of_the_context():
    """`indices=` removed them and this route did not, so scoring rows you
    already hold -- which is how anyone checks a model against known answers --
    handed the model those answers and returned what reads like accuracy."""
    schema = single()
    frame = schema['t'].data
    rows = frame[frame['rid'].isin([1, 2, 3])].drop(columns=['label'])

    request = PredictiveQuery(
        'PREDICT t.label FOR EACH t.rid', schema
    ).materialize(predict=rows)

    context = request.context.instance_table
    assert not context['rid'].isin([1, 2, 3]).any()
    assert len(context) == len(frame) - 3


def test_rows_the_table_does_not_hold_remove_nothing():
    """What the route is for."""
    schema = single()
    request = PredictiveQuery(
        'PREDICT t.label FOR EACH t.rid', schema
    ).materialize(predict=pd.DataFrame({'rid': [9001, 9002], 'a': [0.1, 0.2]}))
    assert len(request.context.instance_table) == len(schema['t'].data)


def test_the_indices_route_is_unchanged():
    schema = single()
    request = PredictiveQuery(
        'PREDICT t.label FOR EACH t.rid', schema
    ).materialize(indices=[1, 2, 3])
    assert not request.context.instance_table['rid'].isin([1, 2, 3]).any()


def test_a_predict_frame_without_the_key_column_removes_nothing():
    """Nothing to match on, so nothing is guessed at."""
    schema = single()
    request = PredictiveQuery(
        'PREDICT t.label FOR EACH t.rid', schema
    ).materialize(predict=pd.DataFrame({'a': [0.1, 0.2]}))
    assert len(request.context.instance_table) == len(schema['t'].data)


def test_editing_the_schema_does_not_change_a_relational_request_either():
    """Both materializers hand the request a schema, and both had to stop
    handing over the caller's live one."""
    pytest.importorskip('sdm_relational_engine')

    schema = graph()
    request = PredictiveQuery(
        'PREDICT SUM(events.amt, 0, 30, days) FOR EACH stores.sid', schema
    ).materialize(indices=['S0', 'S1'], model=KumoRelational())

    schema['stores']['r'].stype = 'text'

    assert request.schema['stores']['r'].stype == 'categorical'
    assert schema['stores']['r'].stype == 'text'


def test_a_predict_frame_request_keeps_its_schema_too():
    schema = single()
    frame = schema['t'].data
    request = PredictiveQuery(
        'PREDICT t.label FOR EACH t.rid', schema
    ).materialize(predict=frame.head(2).drop(columns=['label']))

    schema['t']['a'].stype = 'categorical'

    assert request.schema['t']['a'].stype == 'numerical'


# --- the values are read once ----------------------------------------------


def test_a_column_is_typed_from_one_reading_of_its_values():
    """`infer_tfm_dtype` walks every cell of an object column, and it was
    asked three times per column -- for the dtype, again inside the stype, and
    again to record that there was no type. On a million rows of warehouse
    `Decimal` that was five seconds before anything was sent."""
    import sdm_client.schema as schema_module

    reads: list[str] = []
    real = schema_module.infer_tfm_dtype

    def counting(series):
        reads.append(getattr(series, 'name', '?'))
        return real(series)

    frame = pd.DataFrame(
        {
            'rid': range(50),
            'amount': [float(i) for i in range(50)],
            'label': ['a', 'b'] * 25,
            'when': pd.to_datetime(['2024-01-01'] * 50),
        }
    )
    schema_module.infer_tfm_dtype = counting
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Schema.from_data({'t': frame}, primary_keys={'t': 'rid'})
    finally:
        schema_module.infer_tfm_dtype = real

    assert sorted(reads) == sorted(frame.columns)


def test_a_list_cell_is_still_refused_before_the_column_is_typed():
    """The check that had to stay: typing a column asks for its distinct
    values, and a list cell makes that raise from inside pandas."""
    frame = pd.DataFrame({'rid': [1, 2], 'tags': [['a'], ['b']]})
    with pytest.raises(StructuredDataError) as raised:
        Schema.from_data({'t': frame})
    assert 'tags' in str(raised.value)


def test_typing_still_answers_the_same_for_every_shape():
    frame = pd.DataFrame(
        {
            'i': [1, 2, 3],
            'f': [1.5, 2.5, 3.5],
            'b': [True, False, True],
            's': ['a', 'b', 'c'],
            't': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'empty': [None, None, None],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'t': frame})
    typed = {
        name: (c.dtype, c.stype) for name, c in schema['t'].columns.items()
    }

    assert typed['i'] == ('int64', 'numerical')
    assert typed['f'] == ('float64', 'numerical')
    assert typed['b'] == ('bool', 'categorical')
    assert typed['s'] == ('string', 'categorical')
    assert typed['t'] == ('timestamp[us]', 'timestamp')
    assert typed['empty'] == ('string', 'categorical')
    assert schema['t']['empty'].untyped


# --- a copy judges nothing --------------------------------------------------


def test_a_snapshot_keeps_a_key_inference_tolerated():
    """Inference accepts a key with a null in it -- it warns and goes on,
    because refusing would break frames that worked -- and assignment refuses
    one. Copying through the setter turned a schema `from_data` had accepted
    into a request that could not be materialized."""
    frame = pd.DataFrame({'user_id': [*range(199), None], 'v': [1.0] * 200})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'t': frame})
    assert schema['t'].primary_key == 'user_id'

    copied = schema.snapshot()

    assert copied['t'].primary_key == 'user_id'


def test_a_snapshot_keeps_what_the_key_tolerated():
    frame = pd.DataFrame({'user_id': list(range(200)), 'v': [1.0] * 200})
    frame.loc[[7, 8], 'user_id'] = 6
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'t': frame})
    assert schema['t'].duplicate_keys == 2

    assert schema.snapshot()['t'].duplicate_keys == 2


def test_a_snapshot_does_not_walk_the_rows():
    """`Table.__post_init__` checks every cell of every column, which on a
    warehouse extract is a second full scan added to every request."""
    import sdm_client.schema as schema_module

    frame = pd.DataFrame({'rid': range(500), 'v': [1.0] * 500})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'t': frame}, primary_keys={'t': 'rid'})

    walked: list[str] = []
    real = schema_module.require_scalar_cells

    def counting(series, where):
        walked.append(where)
        return real(series, where)

    schema_module.require_scalar_cells = counting
    try:
        schema.snapshot()
    finally:
        schema_module.require_scalar_cells = real

    assert walked == []


def test_a_snapshot_says_nothing():
    """Assigning a key re-reports whatever its uniqueness tolerated, so
    copying through the setter replayed the warning on every request."""
    frame = pd.DataFrame({'user_id': list(range(200)), 'v': [1.0] * 200})
    frame.loc[[7, 8], 'user_id'] = 6
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'t': frame})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        schema.snapshot()

    assert [str(w.message) for w in caught] == []


def test_a_snapshot_re_points_its_columns_at_itself():
    """A column makes itself the time column of the table it belongs to, so a
    copy whose columns still point at the original would order the wrong
    table."""
    frame = pd.DataFrame(
        {'rid': range(20), 'when': ['2024-01-01'] * 20, 'v': [1.0] * 20}
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'t': frame}, primary_keys={'t': 'rid'})
    copied = schema.snapshot()

    copied['t']['when'].stype = 'timestamp'

    assert copied['t'].time_column == 'when'
    assert schema['t'].time_column is None


# --- the question a caller asks is the one that answers --------------------


def test_appending_a_bad_link_is_caught_by_check_consistent():
    """`relationships` is an ordinary list, so `append` reaches it without
    passing the setter -- and this is the call a caller makes to find out
    whether their schema holds together."""
    schema, link = mismatched()
    schema.relationships.append(link)

    with pytest.raises(StructuredDataError) as raised:
        schema.check_consistent()
    assert 'joins' in str(raised.value)


def test_check_consistent_still_passes_a_good_schema():
    schema = graph()
    schema.check_consistent()


@pytest.mark.parametrize(
    'route', ['assign', 'from_data'], ids=['assignment', 'from_data']
)
def test_something_that_is_not_a_relationship_is_named(route):
    """A tuple of the right four strings is the obvious mistake, and it came
    back as `AttributeError: 'tuple' object has no attribute 'source_table'`."""
    users = pd.DataFrame({'user_id': [1, 2, 3], 'seg': list('abc')})
    orders = pd.DataFrame({'order_id': [1, 2, 3], 'user_id': [1, 2, 3]})
    wrong = ('orders', 'user_id', 'users', 'user_id')

    with pytest.raises(StructuredDataError) as raised:
        if route == 'assign':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                schema = Schema.from_data(
                    {'users': users, 'orders': orders}, relationships=[]
                )
            schema.relationships = [wrong]
        else:
            Schema.from_data(
                {'users': users, 'orders': orders}, relationships=[wrong]
            )
    assert 'must be a Relationship' in str(raised.value)
    assert 'tuple' in str(raised.value)


def test_materialized_without_an_entity_table_reads_the_query_not_the_dict():
    """The argument did not exist before, so a caller who predates it must
    still get an answer, and not the insertion order it replaced. The
    envelopes carry the query's own answer, which is what is read.

    `events` is given first so that insertion order and the query disagree,
    which is the only arrangement that can tell them apart.
    """
    pytest.importorskip('sdm_relational_engine')

    rng = np.random.default_rng(0)
    users = pd.DataFrame({'uid': range(60), 'seg': rng.choice([*'xy'], 60)})
    events = pd.DataFrame(
        {
            'eid': range(240),
            'uid': rng.integers(0, 60, 240),
            'amt': rng.gamma(2.0, 10.0, 240),
            'ts': pd.to_datetime('2023-01-01')
            + pd.to_timedelta(rng.integers(0, 100, 240), 'D'),
        }
    )
    query = 'PREDICT SUM(events.amt, 0, 30, days) FOR EACH users.uid'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'events': events, 'users': users},
            primary_keys={'users': 'uid', 'events': 'eid'},
            time_columns={'events': 'ts'},
            relationships=[Relationship('events', 'uid', 'users', 'uid')],
        )
        gathered = PredictiveQuery(query, schema).materialize(
            indices=[1, 2], model=KumoRelational()
        )

    assert list(schema.tables)[0] == 'events'

    without = Request.materialized(
        payloads=gathered.payloads,
        schema=schema,
        query=query,
        indices=[1, 2],
    )

    assert without.entity_table == 'users'
