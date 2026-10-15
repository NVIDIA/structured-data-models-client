# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A ranking and a forecast have no task on the endpoint, so they are asked
as questions it does serve.

The models answer classification and regression and nothing else, so a
request for either kind came back `422 UNSUPPORTED_TASK` after being built and
sent. Both are the same question asked repeatedly, though: a ranking is, per
candidate, *will this entity reach this one?*, whose probability orders them;
a forecast is, per step, *what does this window hold?*, which is the question
already being asked with its window moved along.

Each part needs its own context because the labels differ. What was true in
the second window is not what was true in the first, so the split happens
where contexts are gathered.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
import pytest

from sdm_client import KumoRelational, PredictiveQuery, Relationship, Schema
from sdm_client.errors import StructuredDataError

RNG = np.random.default_rng(7)
RANK = (
    'PREDICT LIST_DISTINCT(buys.iid, 0, 30, days) RANK TOP 2 FOR EACH users.uid'
)
FORECAST = (
    'PREDICT COUNT(buys.*, 0, 30, days) FORECAST 4 TIMEFRAMES '
    'FOR EACH users.uid'
)


def shop() -> Schema:
    users = pd.DataFrame({'uid': range(60), 'seg': RNG.choice([*'xy'], 60)})
    items = pd.DataFrame({'iid': range(6), 'cat': RNG.choice([*'pq'], 6)})
    pairs = [
        (int(u), int(RNG.integers(0, 6))) for u in RNG.integers(0, 60, 3000)
    ]
    buys = pd.DataFrame(
        {
            'bid': range(len(pairs)),
            'uid': [p[0] for p in pairs],
            'iid': [p[1] for p in pairs],
            'ts': pd.to_datetime('2023-01-01')
            + pd.to_timedelta(RNG.integers(0, 300, len(pairs)), 'D'),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data(
            {'users': users, 'items': items, 'buys': buys},
            primary_keys={'users': 'uid', 'items': 'iid', 'buys': 'bid'},
            time_columns={'buys': 'ts'},
            relationships=[
                Relationship('buys', 'uid', 'users', 'uid'),
                Relationship('buys', 'iid', 'items', 'iid'),
            ],
        )


def built(query: str, **how) -> object:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return PredictiveQuery(query, shop()).materialize(
            model=KumoRelational(), **how
        )


def test_a_ranking_is_one_question_per_candidate():
    pytest.importorskip('sdm_relational_engine')

    request = built(RANK, indices=[1, 2], candidates=[0, 1, 2, 3])

    assert len(request.parts) == 4
    assert all(
        payload['task']['kind'] == 'binary_classification'
        for part in request.parts
        for payload in part.payloads
    )


@pytest.mark.parametrize(
    ('query', 'how'),
    [(RANK, {'candidates': [0, 1]}), (FORECAST, {})],
    ids=['ranking', 'forecast'],
)
def test_more_entities_than_one_call_scores_is_refused_before_gathering(
    monkeypatch, query, how
):
    """A part is one call, which is what makes the fan-out countable. The
    refusal reads the count it was given, so it costs no gathering."""
    pytest.importorskip('sdm_relational_engine')
    from sdm_client.materializers.relational import RelationalMaterializer
    from sdm_client.models import RELATIONAL_LIMITS

    monkeypatch.setitem(RELATIONAL_LIMITS, 'predict_rows_max', 2)
    monkeypatch.setattr(
        RelationalMaterializer,
        '_get_model',
        lambda self: pytest.fail('the engine was reached'),
    )
    with pytest.raises(StructuredDataError) as refused:
        built(query, indices=[1, 2, 3, 4, 5], **how)
    assert refused.value.code == 'UNSUPPORTED_REQUEST'
    assert refused.value.details['entities'] == 5
    assert 'fewer entities' in str(refused.value)


def test_a_smaller_batch_lowers_what_a_part_may_ask_for(monkeypatch):
    """`rows_per_call` only goes down, and a part is one call, so asking for a
    smaller one asks for fewer entities rather than for more calls."""
    pytest.importorskip('sdm_relational_engine')

    with pytest.raises(StructuredDataError) as refused:
        built(FORECAST, indices=[1, 2, 3], rows_per_call=2)
    assert refused.value.details['rows_per_call'] == 2
    assert built(FORECAST, indices=[1, 2], rows_per_call=2).parts


def test_the_whole_decomposed_question_is_held_to_the_memory_ceiling(
    monkeypatch,
):
    """Each part gathers the whole population with a context of its own, so a
    ranking over 200 candidates holds 200 times what one of them does. Bounding
    each part on its own left the question itself unbounded."""
    pytest.importorskip('sdm_relational_engine')
    from sdm_client.materializers import relational as materializer

    monkeypatch.setattr(materializer, 'MOST_ENTITIES', 5)
    built(RANK, indices=[1, 2], candidates=[0, 1])
    with pytest.raises(StructuredDataError) as refused:
        built(RANK, indices=[1, 2, 3], candidates=[0, 1])
    assert refused.value.details['total'] == 6
    assert refused.value.details['gathered_times'] == 2


def test_a_step_that_fails_keeps_the_steps_already_paid_for():
    """A question asked in parts pays per part, and the steps answered before
    a failure are as bought as the batches of one are.

    Two steps of two entities, told apart by the step rather than by a row
    number, which each step counts from its own zero.
    """
    pytest.importorskip('sdm_relational_engine')
    from sdm_client import KumoRelational, SDMClient

    class FailsOnTheThirdCall:
        calls = 0

        def predict(self, payload: dict) -> dict:
            type(self).calls += 1
            if self.calls == 3:
                raise RuntimeError('the warehouse token could not be renewed')
            table = payload['predict']['instance_table']
            at = table['columns'].index('instance_id')
            return {
                'id': 'echo',
                'model': 'kumo-relational',
                'metadata': {},
                'predictions': [
                    {'row_index': i, 'id': str(row[at]), 'prediction': 1.0}
                    for i, row in enumerate(table['rows'])
                ],
            }

    request = built(FORECAST, indices=[1, 2])
    client = SDMClient(url='http://localhost:8000')
    client._transport = FailsOnTheThirdCall()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(RuntimeError) as failed:
            client.predict(request, model=KumoRelational())
    assert failed.value.details['parts_answered'] == 2
    kept = failed.value.details['answered'].to_pandas()
    assert len(kept) == 4
    assert sorted(kept['forecast_step'].unique()) == [1, 2]


@pytest.fixture(scope='module')
def _forecast_template():
    pytest.importorskip('sdm_relational_engine')
    return built(FORECAST, indices=[3])


@pytest.fixture()
def forecast_request(_forecast_template):
    """A copy of the gathered forecast both checks below are about.

    Gathering it is the expensive part of each, and they were each doing it.
    Copied rather than shared, so neither can reach the other's.
    """
    return copy.deepcopy(_forecast_template)


def test_a_forecast_is_one_question_per_step(forecast_request):
    assert len(forecast_request.parts) == 4
    windows = [part.query for part in forecast_request.parts]
    assert '0, 30, days' in windows[0]
    assert '30, 60, days' in windows[1]
    assert '90, 120, days' in windows[3]


def test_each_part_gathers_its_own_context(forecast_request):
    """The labels differ between parts, so a shared context would answer the
    wrong question for all but one of them."""
    contexts = [part.payloads[0]['context'] for part in forecast_request.parts]

    assert all(context is not contexts[0] for context in contexts[1:])


def test_the_candidates_have_to_be_named():
    """There is no ranking task to choose them with, and one call is made per
    candidate, so assuming a set would be assuming the cost too."""
    pytest.importorskip('sdm_relational_engine')

    with pytest.raises(StructuredDataError) as refused:
        built(RANK, indices=[1, 2])

    assert 'candidates' in str(refused.value)
    assert 'One call is made per candidate' in str(refused.value)


def test_candidates_are_refused_where_nothing_is_ranked():
    pytest.importorskip('sdm_relational_engine')

    with pytest.raises(StructuredDataError) as refused:
        built(
            'PREDICT COUNT(buys.*, 0, 30, days) FOR EACH users.uid',
            indices=[1],
            candidates=[1, 2],
        )

    assert 'does not rank anything' in str(refused.value)


def test_a_repeated_candidate_is_asked_about_once():
    pytest.importorskip('sdm_relational_engine')

    request = built(RANK, indices=[1], candidates=[1, 1, 2, 2, 2])

    assert len(request.parts) == 2


def test_too_many_parts_is_refused_before_the_waiting_starts():
    """The cost of this is a call per part, and a caller should meet that as a
    sentence rather than as a wait."""
    pytest.importorskip('sdm_relational_engine')

    with pytest.raises(StructuredDataError) as refused:
        built(RANK, indices=[1], candidates=list(range(500)))

    assert '500 candidates means 500 calls' in str(refused.value)
    assert 'MOST_PARTS' in str(refused.value)


def test_an_ordinary_query_is_not_split():
    pytest.importorskip('sdm_relational_engine')

    request = built(
        'PREDICT COUNT(buys.*, 0, 30, days) FOR EACH users.uid', indices=[1, 2]
    )

    assert request.parts is None
