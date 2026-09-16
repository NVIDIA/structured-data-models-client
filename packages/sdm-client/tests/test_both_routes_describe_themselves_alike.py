# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A request describes what it holds, whichever route gathered it.

`context` and `predict` are documented as the rows where the answer is known
and the rows being asked about. The relational route keeps its data as
finished wire envelopes, and these two were left empty for it, so the same
object reported several hundred context rows through `context_rows` and none
at all through `context`. One question, answered twice and differently.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sdm_client import (
    KumoRelational,
    KumoTabular,
    PredictiveQuery,
    Relationship,
    Schema,
)
from sdm_client.errors import StructuredDataError
from sdm_client.request import Request, Split, _split_from

RNG = np.random.default_rng(3)


def one_table() -> Schema:
    frame = pd.DataFrame(
        {
            'rid': range(400),
            'x': RNG.normal(0, 1, 400),
            'amount': RNG.gamma(2.0, 20.0, 400).round(2),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data({'t': frame}, primary_keys={'t': 'rid'})


def envelope(scored: list[int], known: list[int]) -> dict:
    return {
        'task': {
            'kind': 'regression',
            'target': {'column_name': 'a', 'dtype': 'float64'},
        },
        'context': {
            'instance_table': {
                'columns': ['id', 'v'],
                'rows': [[i, 1.0] for i in known],
            },
            'related_tables': {
                'e': {'columns': ['id'], 'rows': [[i] for i in known]}
            },
        },
        'predict': {
            'instance_table': {
                'columns': ['id'],
                'rows': [[i] for i in scored],
            }
        },
    }


def test_the_two_routes_report_the_same_halves_for_the_same_question():
    pytest.importorskip('sdm_relational_engine')

    schema = one_table()
    query = 'PREDICT t.amount FOR EACH t.rid'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tabular = PredictiveQuery(query, schema).materialize(
            indices=[1, 2], model=KumoTabular()
        )
        relational = PredictiveQuery(query, schema).materialize(
            indices=[1, 2], model=KumoRelational()
        )

    assert len(relational.context.instance_table) == len(
        tabular.context.instance_table
    )
    assert len(relational.predict.instance_table) == len(
        tabular.predict.instance_table
    )


@pytest.mark.parametrize('model', ['tabular', 'relational'])
def test_a_request_agrees_with_its_own_counts(model):
    """The halves and the counts are two readings of one thing."""
    pytest.importorskip('sdm_relational_engine')

    chosen = KumoTabular() if model == 'tabular' else KumoRelational()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = PredictiveQuery(
            'PREDICT t.amount FOR EACH t.rid', one_table()
        ).materialize(indices=[1, 2], model=chosen)

    assert len(request.context.instance_table) == request.context_rows
    assert len(request.predict.instance_table) == request.predict_rows
    assert request.context_rows > 0


def test_a_relational_request_shows_the_tables_it_carries():
    pytest.importorskip('sdm_relational_engine')

    users = pd.DataFrame({'uid': range(300), 'seg': RNG.choice([*'xy'], 300)})
    events = pd.DataFrame(
        {
            'eid': range(1200),
            'uid': RNG.integers(0, 300, 1200),
            'amt': RNG.gamma(2.0, 10.0, 1200),
            'ts': pd.to_datetime('2023-01-01')
            + pd.to_timedelta(RNG.integers(0, 150, 1200), 'D'),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'users': users, 'events': events},
            primary_keys={'users': 'uid', 'events': 'eid'},
            time_columns={'events': 'ts'},
            relationships=[Relationship('events', 'uid', 'users', 'uid')],
        )
        request = PredictiveQuery(
            'PREDICT SUM(events.amt, 0, 30, days) FOR EACH users.uid', schema
        ).materialize(indices=list(range(8)), model=KumoRelational())

    assert set(request.context.related_tables) == {'users', 'events'}
    assert len(request.context.instance_table) == request.context_rows


# --- batching is transport, so the halves are the whole of each half --------


def test_several_envelopes_read_as_one_question():
    payloads = (envelope([1, 2], [10, 11, 12]), envelope([3], [13, 14]))

    context = _split_from(payloads, 'context')
    predict = _split_from(payloads, 'predict')

    assert len(context.instance_table) == 5
    assert len(predict.instance_table) == 3
    assert len(context.related_tables['e']) == 5
    assert list(context.instance_table.index) == list(range(5))


@pytest.mark.parametrize(
    'broken',
    [
        {'context': 'not a mapping'},
        {},
        {'context': {}},
        {'context': {'instance_table': 'not a mapping'}},
        {'context': {'instance_table': {'rows': [[1]]}}},
        {'context': {'instance_table': {'columns': ['id']}}},
        {'context': {'instance_table': {'columns': 'id', 'rows': [[1]]}}},
        {'context': {'instance_table': {'columns': ['id'], 'rows': 'nope'}}},
    ],
)
def test_an_envelope_that_cannot_be_read_contributes_nothing(broken):
    """Read defensively, the way the counts beside them are. A half that
    cannot be read is not a half carrying no rows, and neither is a related
    table: reporting either as empty would describe the request wrongly
    rather than decline to describe it."""
    context = _split_from((envelope([1], [10]), broken), 'context')

    assert len(context.instance_table) == 1
    assert len(context.related_tables['e']) == 1


def test_an_unreadable_related_table_does_not_cost_the_rows_beside_it():
    """The halves are read per table, so one table that cannot be read leaves
    the instance rows and the other tables where they are."""
    broken = {
        'context': {
            'instance_table': {'columns': ['id', 'v'], 'rows': [[9, 1.0]]},
            'related_tables': {'e': 'not a mapping'},
        }
    }

    context = _split_from((envelope([1], [10]), broken), 'context')

    assert len(context.instance_table) == 2
    assert len(context.related_tables['e']) == 1


def test_envelopes_that_disagree_on_columns_keep_every_column():
    """Dropping to the intersection would lose a column silently; the union
    with gaps says which rows lacked it."""
    first = envelope([1], [10])
    second = envelope([2], [11])
    second['context']['instance_table']['columns'] = ['id', 'v', 'extra']
    second['context']['instance_table']['rows'] = [[11, 1.0, 'x']]

    context = _split_from((first, second), 'context')

    assert set(context.instance_table.columns) == {'id', 'v', 'extra'}
    assert len(context.instance_table) == 2
    assert context.instance_table['extra'].isna().sum() == 1


# --- built when read, not while materializing -------------------------------


def test_the_halves_are_not_built_until_they_are_read(monkeypatch):
    """The adapter sends the envelopes as they are, so a request that is
    materialized and sent must not pay to read rows nobody looks at."""
    import sdm_client.request as module

    built: list[str] = []
    real = module._split_from
    monkeypatch.setattr(
        module,
        '_split_from',
        lambda payloads, half: built.append(half) or real(payloads, half),
    )

    request = Request.materialized(
        payloads=(envelope([1, 2], [10, 11]),),
        schema=one_table(),
        query='PREDICT t.amount FOR EACH t.rid',
        indices=[1, 2],
        entity_table='t',
    )

    assert built == []
    assert request.context_rows == 2
    assert built == [], 'counting rows must not build the frames either'

    assert len(request.context.instance_table) == 2
    assert built == ['context']


def test_a_half_reflects_an_edit_that_kept_the_row_count():
    """A count is not a fingerprint. Rows are edited in place as well as
    removed, and a half held between reads would describe the old ones."""
    payloads = [envelope([1], [10, 11, 12])]
    request = Request.materialized(
        payloads=tuple(payloads),
        schema=one_table(),
        query='PREDICT t.amount FOR EACH t.rid',
        indices=[1],
        entity_table='t',
    )

    assert request.context.instance_table['id'].tolist() == [10, 11, 12]

    payloads[0]['context']['instance_table']['rows'][1] = [99, 1.0]
    request._payloads = tuple(payloads)

    assert request.context.instance_table['id'].tolist() == [10, 99, 12]


def test_an_object_with_no_payloads_attribute_still_answers():
    """`__new__` and unpickling both produce one, and the attribute beside
    this one already reads defensively for that reason."""
    bare = Request.__new__(Request)

    assert len(bare.context.instance_table) == 0
    assert len(bare.predict.instance_table) == 0


def test_a_related_tables_that_is_not_a_mapping_is_ignored():
    payload = {
        'context': {
            'instance_table': {'columns': ['id'], 'rows': [[1]]},
            'related_tables': 'bad',
        }
    }

    context = _split_from((payload,), 'context')

    assert len(context.instance_table) == 1
    assert context.related_tables == {}


# --- what a half may be replaced with, and where ----------------------------


def test_the_direct_route_takes_a_replacement():
    """The route that keeps its rows as these frames is the one where putting
    a frame here is meaningful, and everything reading it follows."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = PredictiveQuery(
            'PREDICT t.amount FOR EACH t.rid', one_table()
        ).materialize(indices=[1, 2], model=KumoTabular())

    request.predict = Split(instance_table=pd.DataFrame({'rid': [7, 8, 9]}))

    assert request.predict_rows == 3
    assert request.predict.instance_table['rid'].tolist() == [7, 8, 9]
    assert not request.nothing_to_score


def test_emptying_the_direct_route_is_seen_as_empty():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = PredictiveQuery(
            'PREDICT t.amount FOR EACH t.rid', one_table()
        ).materialize(indices=[1, 2], model=KumoTabular())

    request.predict = Split(
        instance_table=pd.DataFrame({'rid': pd.Series([], dtype='int64')})
    )

    assert request.nothing_to_score
    assert request.predict_rows == 0


@pytest.mark.parametrize('half', ['context', 'predict'])
def test_a_gathered_request_refuses_a_replacement(half):
    """Its rows are already encoded, and the encoding carries what a frame
    does not: the reference each row is addressed by and the moment it is
    asked at. Accepting one would change what the request reports and not
    what it sends, so the object would describe rows the model never sees."""
    request = Request.materialized(
        payloads=(envelope([1], [10, 11]),),
        schema=one_table(),
        query='PREDICT t.amount FOR EACH t.rid',
        indices=[1],
        entity_table='t',
    )

    with pytest.raises(StructuredDataError) as refused:
        setattr(request, half, Split(instance_table=pd.DataFrame({'id': [7]})))

    assert refused.value.code == 'INVALID_REQUEST'
    assert 'already encoded' in str(refused.value)
    assert 'Materialize the query again' in str(refused.value)
    assert request.predict_rows == 1


def test_a_stored_half_does_not_outrank_the_envelopes():
    """The setter refuses this state, but an object restored from a pickle
    written before it carries one. The envelopes are what is counted and what
    is sent, so they are what is read."""
    request = Request.materialized(
        payloads=(envelope([1], [10, 11]),),
        schema=one_table(),
        query='PREDICT t.amount FOR EACH t.rid',
        indices=[1],
        entity_table='t',
    )
    request.__dict__['_predict'] = Split(
        instance_table=pd.DataFrame({'id': [7, 8, 9]})
    )

    assert len(request.predict.instance_table) == request.predict_rows == 1
    assert not request.nothing_to_score
