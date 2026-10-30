# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A query that starts with `EXPLAIN` is answered as if `outputs` named
``'explanation'``.

What is held here is what reaches the endpoint: the output fields each call
asks for. The relational engine's gathering is stood in for by the payloads
``test_relational_explanations`` builds, so the real ``materialize()`` runs
without the engine; everything after gathering is the client's own.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pytest
from sample_tables import graph, people
from test_relational_explanations import Transport, gathered

from sdm_client import (
    KumoRelational,
    KumoTabular,
    PredictiveQuery,
    Relationship,
    Request,
    Schema,
    SDMClient,
)
from sdm_client.core.backend import NimBackend
from sdm_client.errors import StructuredDataError, UnsupportedRequestError
from sdm_client.materializers.relational import RelationalMaterializer

CHURN = 'PREDICT people.churn FOR EACH people.person_id'
TIER = 'PREDICT users.tier FOR EACH users.user_id'
FORECAST = (
    'EXPLAIN PREDICT COUNT(orders.*, 0, 30, days) FORECAST 2 TIMEFRAMES '
    'FOR EACH users.user_id'
)
RANK = (
    'EXPLAIN PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) RANK TOP 2 '
    'FOR EACH users.user_id'
)
DEFAULTS = ['prediction', 'probabilities']
CLASSES = [
    {'kind': 'probability', 'class': '0'},
    {'kind': 'probability', 'class': '1'},
]


def tabular_asked(text: str) -> Request:
    """``text`` over ``people``, scoring persons 50 and 51."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'people': people()}, primary_keys={'people': 'person_id'}
        )
    return PredictiveQuery(text, schema).materialize(
        indices=[50, 51], model=KumoTabular(), task='classification'
    )


def relational_asked(
    monkeypatch, text: str, *, kind: str = 'classification', **how: Any
) -> Request:
    """``text`` over the sample graph, each gather answered by users 5 and 7.

    ``labels``, when given, replaces the context's target column.
    """
    labels = how.pop('labels', None)

    def gather(self, query, indices, num_context_samples, options):
        payload = gathered(5, 7, task=kind)
        if labels is not None:
            rows = payload['context']['instance_table']['rows']
            for row, label in zip(rows, labels, strict=True):
                row[2] = label
        return (payload,)

    monkeypatch.setattr(RelationalMaterializer, '_gather', gather)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            graph(),
            primary_keys={
                'users': 'user_id',
                'items': 'item_id',
                'orders': 'order_id',
            },
            time_columns={'orders': 'ts'},
            relationships=[
                Relationship('orders', 'user_id', 'users', 'user_id'),
                Relationship('orders', 'item_id', 'items', 'item_id'),
            ],
        )
        return PredictiveQuery(text, schema).materialize(
            indices=[5, 7], model=KumoRelational(), **how
        )


def fields_sent(sent: list[dict]) -> list[list[str]]:
    """The output fields each call asked for, in order."""
    return [payload['output']['fields'] for payload in sent]


class TabularEndpoint:
    """Answers ``kumo-tabular`` calls, explaining each row when asked;
    records every scoring payload."""

    def __init__(self, gradient_explanation: Callable[..., dict]) -> None:
        self.build = gradient_explanation
        self.sent: list[dict] = []

    def predict(self, payload: dict) -> dict:
        self.sent.append(payload)
        explain = 'explanation' in payload['output']['fields']
        records = []
        for index, _ in enumerate(payload['predict']['instance_table']['rows']):
            record = {
                'row_index': index,
                'prediction': 1,
                'probabilities': {'0': 0.4, '1': 0.6},
            }
            if explain:
                record['explanation'] = self.build(
                    outputs=CLASSES,
                    row=index,
                    columns=['age'],
                    gradients=[[-1.0], [1.0]],
                )
            records.append(record)
        return {
            'id': 'r',
            'model': 'kumo-tabular',
            'predictions': records,
            'metadata': {},
        }

    def create_session(self, payload: dict) -> dict:
        return {'session_id': 's1'}

    def session_predict(self, session_id: str, payload: dict) -> dict:
        return self.predict(payload)

    def delete_session(self, session_id: str) -> None:
        return None


class RelationalEndpoint(Transport):
    """``Transport``, answering a regression with a number, and leaving the
    explanation off the calls numbered in ``bare``."""

    def __init__(self, gradient_explanation, bare: frozenset[int]) -> None:
        super().__init__(gradient_explanation)
        self.bare = bare

    def predict(self, payload: dict) -> dict:
        answer = super().predict(payload)
        for record in answer['predictions']:
            if payload['task']['kind'] == 'regression':
                record['prediction'] = 1.0
                del record['probabilities']
            if len(self.sent) - 1 in self.bare:
                del record['explanation']
        return answer


@pytest.fixture
def tabular_endpoint(gradient_explanation) -> tuple[SDMClient, TabularEndpoint]:
    client = SDMClient(url='http://localhost:8000')
    endpoint = TabularEndpoint(gradient_explanation)
    client._backend = NimBackend(endpoint)
    return client, endpoint


@pytest.fixture
def relational_endpoint(
    gradient_explanation,
) -> Callable[..., tuple[SDMClient, RelationalEndpoint]]:
    """A client on a ``RelationalEndpoint``; ``bare`` as it takes."""

    def build(bare: frozenset[int] = frozenset()):
        client = SDMClient(url='http://localhost:8000')
        endpoint = RelationalEndpoint(gradient_explanation, bare)
        client._backend = NimBackend(endpoint)
        return client, endpoint

    return build


# --- the prefix survives materialize() --------------------------------------


def test_tabular_asks_for_explanation(tabular_endpoint):
    """An `EXPLAIN` query materialized for tabular asks for the defaults and
    ``'explanation'``, and its answer carries one per row.

    Assumptions:
    - Two persons scored; the endpoint explains when asked.
    """
    client, endpoint = tabular_endpoint

    result = client.predict(
        tabular_asked(f'EXPLAIN {CHURN}'), model=KumoTabular()
    )

    assert fields_sent(endpoint.sent) == [[*DEFAULTS, 'explanation']]
    assert len(result.explanations) == len(result) == 2


def test_relational_asks_for_explanation(monkeypatch, relational_endpoint):
    """An `EXPLAIN` query materialized for relational asks for the defaults
    and ``'explanation'``, and its answer carries one per entity.

    Assumptions:
    - The engine's gathering is stood in for; users 5 and 7 are scored.
    """
    client, endpoint = relational_endpoint()

    result = client.predict(
        relational_asked(monkeypatch, f'EXPLAIN {TIER}'),
        model=KumoRelational(),
    )

    assert fields_sent(endpoint.sent) == [[*DEFAULTS, 'explanation']]
    assert sorted(result.explanations) == [5, 7]


# --- what outputs become ----------------------------------------------------


@pytest.mark.parametrize(
    ('outputs', 'expected'),
    [
        (None, [*DEFAULTS, 'explanation']),
        (['prediction'], ['prediction', 'explanation']),
        (['explanation'], ['explanation']),
        (['prediction', 'explanation'], ['prediction', 'explanation']),
    ],
    ids=['defaults', 'appended', 'already named', 'named among others'],
)
def test_explain_adds_explanation(tabular_endpoint, outputs, expected):
    """`EXPLAIN` adds ``'explanation'`` to the defaults or the caller's list,
    once.

    Assumptions:
    - A tabular classification, whose defaults are prediction and
      probabilities.
    """
    client, endpoint = tabular_endpoint

    client.predict(
        tabular_asked(f'EXPLAIN {CHURN}'), model=KumoTabular(), outputs=outputs
    )

    assert fields_sent(endpoint.sent) == [expected]


@pytest.mark.parametrize(
    ('outputs', 'expected'),
    [(None, DEFAULTS), (['prediction'], ['prediction'])],
    ids=['defaults', 'named'],
)
def test_no_prefix_changes_nothing(tabular_endpoint, outputs, expected):
    """Without `EXPLAIN` the call asks for exactly what it did before.

    Assumptions:
    - A tabular classification, whose defaults are prediction and
      probabilities.
    """
    client, endpoint = tabular_endpoint

    client.predict(tabular_asked(CHURN), model=KumoTabular(), outputs=outputs)

    assert fields_sent(endpoint.sent) == [expected]


def test_empty_outputs_still_refused(tabular_endpoint):
    """`EXPLAIN` does not turn ``outputs=[]`` into a request; it is refused
    as before, and nothing is sent.

    Assumptions:
    - A tabular classification.
    """
    client, endpoint = tabular_endpoint

    with pytest.raises(StructuredDataError, match='at least one field'):
        client.predict(
            tabular_asked(f'EXPLAIN {CHURN}'), model=KumoTabular(), outputs=[]
        )
    assert endpoint.sent == []


# --- rankings and forecasts -------------------------------------------------


def test_ranking_refused_before_sending(monkeypatch, relational_endpoint):
    """An `EXPLAIN` ranking is refused by the ranking-outputs check, naming
    ``'explanation'``, before any call is sent.

    Assumptions:
    - Built by the real ``materialize()`` with two candidates; gathering is
      stood in for.
    """
    client, endpoint = relational_endpoint()
    request = relational_asked(
        monkeypatch, RANK, kind='binary_classification', candidates=[0, 1]
    )

    with pytest.raises(StructuredDataError) as refused:
        client.predict(request, model=KumoRelational())

    assert len(request.parts) == 2
    assert "'explanation' is not among them" in str(refused.value)
    assert endpoint.sent == []


def test_forecast_explains_every_step(monkeypatch, relational_endpoint):
    """Every step of an `EXPLAIN` forecast asks for ``'explanation'``, and
    each step's points are explained under ``(id, step)``.

    Assumptions:
    - Two steps, built by the real ``materialize()``; gathering is stood in
      for, each step a regression over users 5 and 7.
    """
    client, endpoint = relational_endpoint()

    result = client.predict(
        relational_asked(monkeypatch, FORECAST, kind='regression'),
        model=KumoRelational(),
    )

    assert fields_sent(endpoint.sent) == [['prediction', 'explanation']] * 2
    assert sorted(result.explanations) == [(5, 1), (5, 2), (7, 1), (7, 2)]


def test_forecast_names_an_unexplained_step(monkeypatch, relational_endpoint):
    """An `EXPLAIN` forecast whose second step comes back unexplained says
    so, naming the step.

    Assumptions:
    - Two steps; the endpoint leaves the explanation off the second call.
    """
    client, _ = relational_endpoint(bare=frozenset({1}))
    request = relational_asked(monkeypatch, FORECAST, kind='regression')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        client.predict(request, model=KumoRelational())

    said = [str(item.message) for item in caught]
    assert any('step 2 carries none' in message for message in said), said


# --- sessions ---------------------------------------------------------------


def test_session_explains_when_asked(tabular_endpoint):
    """A session fitted without `EXPLAIN` accepts the same question asked
    with it, and that call asks for ``'explanation'``.

    Assumptions:
    - The session pinned the plain question with the default outputs.
    """
    client, endpoint = tabular_endpoint
    fitted = client.fit(tabular_asked(CHURN), model=KumoTabular())

    result = fitted.predict(tabular_asked(f'EXPLAIN {CHURN}'))

    assert fields_sent(endpoint.sent) == [[*DEFAULTS, 'explanation']]
    assert len(result.explanations) == 2


def test_session_keeps_its_question(tabular_endpoint):
    """After an `EXPLAIN` call, a plain call on the same session is accepted
    and asks for the defaults only.

    Assumptions:
    - The session pinned the plain question with the default outputs.
    """
    client, endpoint = tabular_endpoint
    fitted = client.fit(tabular_asked(CHURN), model=KumoTabular())
    fitted.predict(tabular_asked(f'EXPLAIN {CHURN}'))

    fitted.predict(tabular_asked(CHURN))

    assert fields_sent(endpoint.sent)[-1] == DEFAULTS


def test_explain_fit_pins_explanation(tabular_endpoint):
    """A session fitted with `EXPLAIN` explains a later plain call.

    Assumptions:
    - No outputs were named at fit.
    """
    client, endpoint = tabular_endpoint
    fitted = client.fit(tabular_asked(f'EXPLAIN {CHURN}'), model=KumoTabular())

    result = fitted.predict(tabular_asked(CHURN))

    assert fields_sent(endpoint.sent) == [[*DEFAULTS, 'explanation']]
    assert len(result.explanations) == 2


@pytest.mark.parametrize('prefix', ['', 'EXPLAIN '], ids=['plain', 'explain'])
def test_session_refuses_another_question(tabular_endpoint, prefix):
    """A different question is refused, `EXPLAIN` or not, before it is
    sent.

    Assumptions:
    - The session pinned the churn question; the other asks about segment.
    """
    client, endpoint = tabular_endpoint
    fitted = client.fit(tabular_asked(CHURN), model=KumoTabular())
    other = tabular_asked(
        f'{prefix}PREDICT people.segment FOR EACH people.person_id'
    )

    with pytest.raises(StructuredDataError, match='pinned a different'):
        fitted.predict(other)
    assert endpoint.sent == []


# --- in process -------------------------------------------------------------


def test_in_process_tabular_explains(tabular):
    """An `EXPLAIN` query answered in process carries an explanation per row.

    Assumptions:
    - A small random-weight tabular model.
    """
    with SDMClient.in_process(tabular) as client:
        result = client.predict(
            tabular_asked(f'EXPLAIN {CHURN}'), model=KumoTabular()
        )

    assert len(result.explanations) == len(result) == 2
    assert {item.explanation['format'] for item in result} == {
        'sdm_gradient_v1'
    }


def test_in_process_relational_explains(relational):
    """An `EXPLAIN` query gathered by the engine and answered in process
    carries an explanation per entity.

    Assumptions:
    - A small random-weight relational model; a binary question over two
      entities of the sample graph.
    """
    request = PredictiveQuery(
        'EXPLAIN PREDICT COUNT(orders.*, 0, 30, days) > 2 '
        'FOR EACH users.user_id',
        graph(),
    ).materialize(indices=[0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with SDMClient.in_process(relational) as client:
            result = client.predict(request, model=KumoRelational())

    assert len(result.explanations) == len(result) == 2


# --- the relational preflight -----------------------------------------------


def test_one_class_refusal_names_the_column(monkeypatch, relational_endpoint):
    """A one-class context is refused naming the predicted column, not the
    wire's ``TARGET``, when the query starts with `EXPLAIN`.

    Assumptions:
    - Every context row is labelled ``'no'``; gathering is stood in for.
    """
    client, endpoint = relational_endpoint()
    request = relational_asked(
        monkeypatch, f'EXPLAIN {TIER}', labels=['no', 'no', 'no']
    )

    with pytest.raises(UnsupportedRequestError) as refused:
        client.predict(request, model=KumoRelational())

    assert 'every context row has the same users.tier' in str(refused.value)
    assert endpoint.sent == []
