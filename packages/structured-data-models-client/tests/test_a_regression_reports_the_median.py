# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A relational regression reports the median of its predicted distribution.

The endpoint reports the mean unless told otherwise, while the engine and
kumo-relational-client report the median. On a right-skewed target such as a
count the two differ, so the same question answered higher here than there.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import pandas as pd
import pytest

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Relationship,
    Schema,
    SDMClient,
    UnsupportedRequestError,
)
from sdm_client.errors import StructuredDataError

pytest.importorskip('sdm_relational_engine')

URL = 'http://127.0.0.1:9'
COUNT = 'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH people.user_id'


def shop() -> Schema:
    people = pd.DataFrame(
        {'user_id': range(12), 'churn': [i % 2 for i in range(12)]}
    )
    orders = pd.DataFrame(
        {
            'order_id': range(240),
            'user_id': [i % 12 for i in range(240)],
            'item': [i % 5 for i in range(240)],
            'ordered_at': pd.date_range('2026-01-01', periods=240, freq='D'),
        }
    )
    items = pd.DataFrame({'item': range(5), 'kind': list('abcab')})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data(
            {'people': people, 'orders': orders, 'items': items},
            primary_keys={
                'people': 'user_id',
                'orders': 'order_id',
                'items': 'item',
            },
            time_columns={'orders': 'ordered_at'},
            relationships=[
                Relationship('orders', 'user_id', 'people', 'user_id'),
                Relationship('orders', 'item', 'items', 'item'),
            ],
        )


def asked(query: str = COUNT, **options: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return PredictiveQuery(query, shop()).materialize(
            model=KumoRelational(), indices=[1, 2], **options
        )


def answer(sent: dict[str, Any]) -> dict[str, Any]:
    rows = sent['predict']['instance_table']['rows']
    return {
        'object': 'prediction.response',
        'id': 'pred',
        'model': 'kumo-relational',
        'metadata': {},
        'predictions': [
            {'id': str(row[0]), 'row_index': i, 'prediction': 1.0}
            for i, row in enumerate(rows)
        ],
    }


def sent_to_a_nim(request: Any, requests_mock: Any) -> list[dict[str, Any]]:
    route = requests_mock.post(
        f'{URL}/v1/predictions',
        json=lambda call, _: answer(call.json()),
    )
    with SDMClient(URL) as client:
        client.predict(request, model=KumoRelational())
    return [call.json() for call in route.request_history]


def sent_to_databricks(request: Any) -> list[dict[str, Any]]:
    bodies: list[dict[str, Any]] = []

    class Endpoints:
        def query(self, *, name: str, dataframe_records: list) -> Any:
            body = json.loads(dataframe_records[0]['request_json'])
            bodies.append(body)
            return {
                'predictions': [{'response_json': json.dumps(answer(body))}]
            }

    workspace = type('Workspace', (), {'serving_endpoints': Endpoints()})()
    client = SDMClient.for_databricks_serving(
        'kumo-rfm-serving', workspace_client=workspace
    )
    client.predict(request, model=KumoRelational())
    return bodies


def without_request_id(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, 'metadata': {}}


def test_a_regression_asks_for_the_median_by_default(requests_mock):
    request = asked()
    (body,) = sent_to_a_nim(request, requests_mock)

    assert body['task']['kind'] == 'regression'
    assert body['output']['prediction_statistic'] == 'median'
    assert 'output_type' not in body['inference']['inference_config']
    # Both transports wrap the same payload, so both ask for the same thing.
    (wrapped,) = sent_to_databricks(request)
    assert without_request_id(wrapped) == without_request_id(body)


def test_the_mean_is_sent_when_it_is_asked_for(requests_mock):
    (body,) = sent_to_a_nim(asked(prediction_statistic='mean'), requests_mock)
    assert body['output']['prediction_statistic'] == 'mean'


@pytest.mark.parametrize(
    'options, statistic',
    [({}, 'median'), ({'prediction_statistic': 'mean'}, 'mean')],
    ids=['by default', 'when asked'],
)
def test_a_session_scores_with_the_statistic_asked_for(
    options, statistic, requests_mock
):
    """A session pins the context and leaves the output to each scoring call,
    so that call is where the statistic has to travel."""
    requests_mock.post(f'{URL}/v1/sessions', json={'session_id': 's1'})
    scored = requests_mock.post(
        f'{URL}/v1/sessions/s1/predictions',
        json=lambda call, _: answer(call.json()),
    )
    requests_mock.delete(f'{URL}/v1/sessions/s1', status_code=204)

    with SDMClient(URL) as client:
        with client.fit(asked(**options), model=KumoRelational()) as fitted:
            fitted.predict(asked(**options))

    (body,) = [call.json() for call in scored.request_history]
    assert body['output']['prediction_statistic'] == statistic


def test_every_step_of_a_forecast_reports_the_statistic_asked_for():
    request = asked(
        'PREDICT COUNT(orders.*, 0, 30, days) FORECAST 2 TIMEFRAMES '
        'FOR EACH people.user_id',
        prediction_statistic='mean',
    )
    bodies = sent_to_databricks(request)
    assert len(bodies) == 2
    assert {body['output']['prediction_statistic'] for body in bodies} == {
        'mean'
    }


def test_output_type_is_refused_and_points_at_the_option(requests_mock):
    route = requests_mock.post(f'{URL}/v1/predictions', json={})
    request = asked(inference_config={'output_type': 'mean'})

    with pytest.raises(UnsupportedRequestError) as error:
        with SDMClient(URL) as client:
            client.predict(request, model=KumoRelational())

    assert 'prediction_statistic=' in str(error.value)
    assert not route.called


@pytest.mark.parametrize(
    'query, statistic, said',
    [
        (COUNT, 'quantiles', "prediction_statistic is 'quantiles'"),
        (
            'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH people.user_id',
            'mean',
            'this question is a binary_classification',
        ),
        (
            'PREDICT LIST_DISTINCT(orders.item, 0, 30, days) RANK TOP 2 '
            'FOR EACH people.user_id',
            'mean',
            'applies only to a regression, and this question is a ranking',
        ),
    ],
    ids=['an unknown statistic', 'a classification', 'a ranking'],
)
def test_a_statistic_that_cannot_be_reported_is_refused(query, statistic, said):
    with pytest.raises(StructuredDataError, match=said) as error:
        asked(query, prediction_statistic=statistic)
    assert error.value.code == 'INVALID_REQUEST'


CHURN = 'PREDICT COUNT(orders.*, 0, 30, days) > 2 FOR EACH people.user_id'
RANKING = (
    'PREDICT LIST_DISTINCT(orders.item, 0, 30, days) RANK TOP 2 '
    'FOR EACH people.user_id'
)


def ranked(sent: dict[str, Any]) -> dict[str, Any]:
    answered = answer(sent)
    for prediction in answered['predictions']:
        prediction['probabilities'] = {'0': 0.4, '1': 0.6}
    return answered


@pytest.mark.parametrize(
    'query, options',
    [(CHURN, {}), (RANKING, {'candidates': [0, 1]})],
    ids=['classification', 'ranking'],
)
def test_a_question_with_no_distribution_sends_no_statistic(
    query, options, requests_mock
):
    r"""Only a regression has a distribution to summarise, so nothing else
    carries the field, and a deployment reading it cannot be told a statistic
    for an answer that has none.
    """
    route = requests_mock.post(
        f'{URL}/v1/predictions',
        json=lambda call, _: ranked(call.json()),
    )
    with SDMClient(URL) as client:
        client.predict(asked(query, **options), model=KumoRelational())

    sent = [call.json() for call in route.request_history]
    assert sent
    for body in sent:
        assert body['task']['kind'] != 'regression'
        assert 'prediction_statistic' not in body['output']
