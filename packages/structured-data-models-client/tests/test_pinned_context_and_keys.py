# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three ways a request was answered about rows it should not have been.

A session pins the context the `fit()` request gathered, which excluded the rows
*that* request named. Every later call names different rows, and nothing
excluded those, so they sit in the pinned context with their target values and
the model is asked about rows it has been shown the answers for. Measured on a
target that is pure noise: a session scored 0.595 where the same rows scored
one-shot came back 0.455, against a chance rate of 0.5.

A key the caller set to `None` was let through and then discarded, so the
engine's own guess (made from a column *name*, without checking uniqueness)
survived, and linking that column was refused for being a primary key the
caller had explicitly removed.

And the engine's node lookup cast keys to the Python type of its index's first
element, which for an object index is a class rather than a dtype.
"""

from __future__ import annotations

import decimal
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
from sdm_client.wire.keys import _named_by, keys_in, scored_keys

D = decimal.Decimal


# --- the keys a pinned context holds ---------------------------------------


def envelope(half: str, columns, rows, primary_key=None) -> dict:
    payload = {half: {'instance_table': {'columns': columns, 'rows': rows}}}
    if primary_key is not None:
        payload['schema'] = {'instance_table': {'primary_key': primary_key}}
    return payload


def test_a_tabular_context_is_read_by_its_declared_key():
    payload = envelope(
        'context',
        ['rid', 'x', 'label'],
        [[1, 0.5, 'a'], [2, 0.6, 'b']],
        primary_key=['rid'],
    )
    assert keys_in(payload, 'context') == {_named_by([1]), _named_by([2])}


def test_a_relational_context_is_read_by_its_entity_reference():
    """The two models name the key differently, and the reference wins because
    the caller's own key never travels on that path."""
    payload = envelope(
        'context',
        ['instance_id', '__kumo_entity_ref_0', 'TARGET'],
        [[0, 7, 1.0], [1, 11, 2.0]],
        primary_key=['instance_id'],
    )
    assert keys_in(payload, 'context') == {_named_by([7]), _named_by([11])}


@pytest.mark.parametrize(
    'payload',
    [
        None,
        {},
        {'context': {}},
        {'context': {'instance_table': {}}},
        envelope('context', ['a'], [[1]]),
        envelope('context', ['a'], [[1]], primary_key=['b']),
        envelope('context', 'not a list', [[1]], primary_key=['a']),
    ],
    ids=[
        'none',
        'empty',
        'no instance table',
        'no rows',
        'no key named',
        'key not a column',
        'columns not a list',
    ],
)
def test_a_context_this_cannot_read_yields_no_keys(payload):
    """Reporting a key it did not find would be worse than reporting none:
    the warning it feeds is about rows, and a wrong row set is a wrong claim."""
    assert keys_in(payload, 'context') == frozenset()


def test_a_row_too_short_to_hold_the_key_is_skipped():
    payload = envelope(
        'context', ['x', 'rid'], [[0.5, 1], [0.6]], primary_key=['rid']
    )
    assert keys_in(payload, 'context') == {_named_by([1])}


# --- scoring rows the context already answers ------------------------------


def frame(rows: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(31)
    return pd.DataFrame(
        {
            'rid': np.arange(rows),
            'x1': rng.normal(0, 1, rows),
            'label': rng.integers(0, 2, rows).astype(str),
        }
    )


def query(data: pd.DataFrame) -> PredictiveQuery:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'t': data}, primary_keys={'t': 'rid'})
    return PredictiveQuery('PREDICT t.label FOR EACH t.rid', schema)


def test_the_rows_being_scored_are_read_from_the_request():
    data = frame()
    request = query(data).materialize(indices=[1, 2, 3])
    assert scored_keys(request) == {_named_by([k]) for k in (1, 2, 3)}


# --- a key set to None is taken back from the engine -----------------------


def test_a_child_keyed_id_can_be_linked():
    """The engine infers a key from a column *name* without checking
    uniqueness, so `ID` holding [1, 1, 2, 2, 3] became that table's primary
    key. The client let the caller set it to None and never passed that on."""

    severity = pd.DataFrame({'ID': [1, 1, 2, 2, 3], 'sev': list('abcde')})
    location = pd.DataFrame({'ID': [1, 2, 3], 'loc': list('xyz')})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'location': location, 'severity_type': severity},
            primary_keys={'location': 'ID'},
            relationships=[
                Relationship('severity_type', 'ID', 'location', 'ID')
            ],
        )
    schema['severity_type'].primary_key = None

    request = PredictiveQuery(
        'PREDICT location.loc FOR EACH location.ID', schema
    ).materialize(indices=[1, 2], model=KumoRelational())

    assert request.predict_rows == 2
    assert 'severity_type' in request.payloads[0]['schema']['related_tables']


# --- the engine casts to a dtype, not to a class ---------------------------


# --- a key written the same way on both sides ------------------------------


def test_a_wide_key_matches_across_the_wire():
    """Past 2**53 a key travels as a base-10 string by contract, so the pinned
    key is text and the scored key is an int. Compared as they arrive they
    never meet, and the warning goes quiet on exactly the warehouse keys it
    matters most for."""
    wide = 9007199254740993
    assert _named_by([str(wide)]) == _named_by([wide])


def test_a_composite_key_is_a_tuple_rather_than_nothing():
    """Reading only single-column keys meant a session over a composite-keyed
    entity could score pinned rows in silence."""
    payload = envelope(
        'context',
        ['a', 'b', 'v'],
        [[1, 'x', 0.5], [2, 'y', 0.6]],
        primary_key=['a', 'b'],
    )
    assert keys_in(payload, 'context') == {
        _named_by([1, 'x']),
        _named_by([2, 'y']),
    }


def test_a_composite_key_distinguishes_its_parts():
    assert _named_by([1, 2]) != _named_by([2, 1])
    assert _named_by([1, 2]) != _named_by(['12'])


# --- a window makes the moment part of the question -------------------------


def test_a_frame_backed_request_has_no_moment_to_compare():
    """The frame-backed half belongs to the single-table route, and a
    single-table model is not asked for a window over time: the query is
    refused before it reaches here. So there is no moment in the frame to put
    in the key, and `with_anchor` has nothing to do there."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            'rid': range(60),
            'x': rng.normal(0, 1, 60),
            'amount': rng.gamma(2.0, 5.0, 60),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'t': frame}, primary_keys={'t': 'rid'})
        request = PredictiveQuery(
            'PREDICT t.amount FOR EACH t.rid', schema
        ).materialize(indices=[1, 2])

    assert request.payloads is None
    assert not request.target_moves_with_time
    assert '__kumo_anchor_time' not in request.predict.instance_table
    assert scored_keys(request) == scored_keys(request, with_anchor=True)

    with pytest.raises(StructuredDataError):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            PredictiveQuery(
                'PREDICT AVG(t.amount, 0, 7, days) FOR EACH t.rid', schema
            ).materialize(indices=[1], model=KumoTabular())


class SessionEndpoint:
    url = 'http://localhost'
    serves_sessions = True

    def __init__(self, model):
        self.model = model
        self.created = []
        self.scored = []
        self.deleted = []

    def create_session(self, payload):
        self.created.append(payload)
        return {'session_id': 'pinned'}

    def session_predict(self, session_id, payload):
        self.scored.append(payload)
        table = payload['predict']['instance_table']
        identity = (
            table['columns'].index('instance_id')
            if 'instance_id' in table['columns']
            else None
        )
        return {
            'model': self.model,
            'id': 'answer',
            'metadata': {},
            'predictions': [
                {
                    'row_index': i,
                    'id': str(row[identity]) if identity is not None else None,
                    'prediction': 0.0
                    if self.model == 'kumo-relational'
                    else '0',
                }
                for i, row in enumerate(table['rows'])
            ],
        }

    def delete_session(self, session_id):
        self.deleted.append(session_id)

    def close(self):
        pass


def direct(context, scores, keys='rid'):
    from sdm_client import Request

    schema = Schema.from_data({'t': context}, primary_keys={'t': keys})
    return Request(
        context,
        scores,
        target='label',
        task='classification',
        schema=schema,
        entity_table='t',
    )


def session_answer(opening, scoring, model='kumo-tabular', holdout=None):
    from sdm_client import SDMClient

    endpoint = SessionEndpoint(model)
    with SDMClient('http://localhost') as client:
        client._transport = endpoint
        with client.fit(opening, model=model, holdout=holdout) as fitted:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                result = fitted.predict(scoring)
    assert len(endpoint.created) == 1
    assert len(endpoint.scored) == 1
    assert endpoint.deleted == ['pinned']
    assert len(result.items) == scoring.predict_rows
    return result, [str(w.message) for w in caught], endpoint


@pytest.mark.parametrize(
    ('context_rows', 'scored_rows', 'overlap'),
    [
        (range(600), range(200, 260), 60),
        (range(20, 600), range(20), 0),
        (range(20), range(10, 30), 10),
    ],
)
def test_public_session_reports_the_actual_context_overlap(
    context_rows, scored_rows, overlap
):
    data = frame()
    context = data.iloc[list(context_rows)]
    scores = data.iloc[list(scored_rows)].drop(columns=['label'])
    opening = direct(context, scores)
    _, said, endpoint = session_answer(opening, opening)
    overlap_notices = [message for message in said if 'pinned' in message]
    if overlap:
        assert any(
            f'{overlap} of the {len(scores)} rows' in message
            for message in overlap_notices
        )
        assert any(
            'before calling fit()' in message for message in overlap_notices
        )
        assert any('use predict()' in message for message in overlap_notices)
    else:
        assert overlap_notices == []
    table = endpoint.created[0]['context']['instance_table']
    key = table['columns'].index('rid')
    assert {row[key] for row in table['rows']} == set(context_rows)


@pytest.mark.parametrize('overlap', [True, False])
def test_public_composite_session_preserves_key_parts(overlap):
    data = frame(40).rename(columns={'rid': 'a'})
    data['a'] = [100 + i // 4 for i in range(40)]
    data['b'] = [i % 4 for i in range(40)]
    scores = data.head(4).drop(columns=['label']).copy()
    scores.loc[1, 'b'] = 99  # Same first component, different second.
    scores.loc[2, ['a', 'b']] = [2, 100]  # Swapped components.
    scores.loc[3, 'a'] = 9999  # Same second component, different first.
    context = data if overlap else data.assign(a=data['a'] + 1000)
    request = direct(context, scores, ['a', 'b'])
    _, said, endpoint = session_answer(request, request)
    assert any('1 of the 4 rows' in message for message in said) is overlap
    assert endpoint.created[0]['schema']['instance_table']['primary_key'] == [
        'a',
        'b',
    ]


def test_public_session_retains_wide_key_identity():
    data = frame(40)
    data['rid'] += 2**53 + 1
    request = direct(data, data.head(3).drop(columns=['label']))
    result, said, _ = session_answer(request, request)
    assert result.to_pandas()['id'].tolist() == data['rid'].head(3).tolist()
    assert any('3 of the 3 rows' in message for message in said)


def test_empty_context_is_refused_instead_of_creating_an_unreadable_session():
    with pytest.raises(StructuredDataError, match='context has no rows'):
        direct(frame(0), frame(2).drop(columns=['label']))


@pytest.mark.parametrize(
    ('held', 'asked'),
    [
        ([1, 2, 3], ['2']),
        ([1, 2, 3], [2]),
        ([D(1), D(2), D(3)], [D(2)]),
        (['a', 'b', 'c'], ['b']),
    ],
)
def test_public_materialization_resolves_object_keys(held, asked):
    people = pd.DataFrame(
        {'pid': pd.Series(held, dtype=object), 'v': [1.0, 2.0, 3.0]}
    )
    schema = Schema.from_data(
        {'people': people}, primary_keys={'people': 'pid'}
    )
    request = PredictiveQuery(
        'PREDICT people.v FOR EACH people.pid', schema
    ).materialize(
        indices=asked,
        model=KumoRelational(),
        num_context_samples=3,
    )
    assert request.predict_rows == 1
    table = request.payloads[0]['predict']['instance_table']
    at = table['columns'].index('__kumo_entity_ref_0')
    assert str(table['rows'][0][at]) == str(held[1])


@pytest.mark.parametrize('key', ['not a key', [1], {'a': 1}])
def test_public_materialization_refuses_invalid_keys(key):
    people = pd.DataFrame({'pid': [1, 2, 3], 'v': [1.0, 2.0, 3.0]})
    schema = Schema.from_data(
        {'people': people}, primary_keys={'people': 'pid'}
    )
    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery(
            'PREDICT people.v FOR EACH people.pid', schema
        ).materialize(
            indices=[key],
            model=KumoRelational(),
        )
    assert raised.value.code == 'INVALID_REQUEST'


@pytest.mark.parametrize(
    ('windowed', 'moment', 'holdout', 'warns'),
    [
        (True, '2023-06-01', None, False),
        (True, '2023-05-01', None, True),
        (False, '2023-06-01', None, True),
        (True, '2023-05-01', [1], False),
    ],
)
def test_public_session_compares_the_moment_only_for_windowed_targets(
    windowed, moment, holdout, warns
):
    from test_context_that_answers_the_question import STATIC, WINDOWED, graph

    asked = PredictiveQuery(WINDOWED if windowed else STATIC, graph())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        opening = asked.materialize(
            indices=[299],
            model=KumoRelational(),
            num_context_samples=300,
            anchor_time=pd.Timestamp('2023-06-01'),
            context_anchor_time=pd.Timestamp('2023-05-01'),
        )
        scoring = asked.materialize(
            indices=[1],
            model=KumoRelational(),
            num_context_samples=300,
            anchor_time=pd.Timestamp(moment),
            context_anchor_time=pd.Timestamp('2023-04-01'),
        )
    _, said, endpoint = session_answer(
        opening, scoring, 'kumo-relational', holdout
    )
    assert any('pinned' in message for message in said) is warns
    context = endpoint.created[0]['context']['instance_table']
    ref = context['columns'].index('__kumo_entity_ref_0')
    assert any(row[ref] == 1 for row in context['rows']) is (holdout is None)
    if windowed and holdout is None:
        anchor = context['columns'].index(
            endpoint.created[0]['task']['anchor_time_column']
        )
        assert any(
            row[ref] == 1 and str(row[anchor]).startswith('2023-05-01')
            for row in context['rows']
        )
