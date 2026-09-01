# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end against a running ``kumo-relational`` NIM.

Excluded by default. Point ``SDM_RELATIONAL_URL`` at the endpoint and run with
``-m live_nim``. Needs the ``[relational]`` extra, because the gathering step
runs in the engine.

    SDM_RELATIONAL_URL=http://localhost:8001 pytest -m live_nim

What these prove that a mock cannot: that the payload the engine gathers is one
the endpoint accepts, that the answer maps back to the entity that was asked
about, and that point-in-time correctness is enforced rather than assumed.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from sdm_client import (
    KumoRelational,
    KumoTabular,
    PredictiveQuery,
    StructuredDataClient,
    UnsupportedRequestError,
)
from sdm_client.errors import StructuredDataError

pytestmark = pytest.mark.live_nim

RELATIONAL_URL = os.environ.get('SDM_RELATIONAL_URL')

requires_relational = pytest.mark.skipif(
    not RELATIONAL_URL, reason='set SDM_RELATIONAL_URL to a kumo-relational NIM'
)

pytest.importorskip(
    'sdm_relational_engine', reason='needs the [relational] extra'
)


N_USERS = 40


def tables() -> dict[str, pd.DataFrame]:
    """A graph where the answer is legible: every other user is busy.

    Order counts differ sharply between the two halves, so a prediction that
    tracks them is evidence the whole path works rather than that something
    came back.
    """
    users = pd.DataFrame(
        {
            'user_id': range(N_USERS),
            'segment': ['north' if i % 3 else 'south' for i in range(N_USERS)],
        }
    )
    rows = []
    order_id = 0
    for user in range(N_USERS):
        busy = user % 2 == 0
        for k in range(40 if busy else 4):
            rows.append(
                {
                    'order_id': order_id,
                    'user_id': user,
                    'price': 10.0 + (user % 7) * 3 + k,
                    # Spread over months: a 30-day aggregation needs 30 days of
                    # history before the anchor, and the engine enforces it.
                    'ts': pd.Timestamp('2024-01-01')
                    + pd.Timedelta(
                        days=(k * 5 if busy else k * 50) + (user % 9)
                    ),
                }
            )
            order_id += 1
    return {'users': users, 'orders': pd.DataFrame(rows)}


COUNT_QUERY = 'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id'


@pytest.fixture
def client():
    with StructuredDataClient(url=RELATIONAL_URL, timeout=600.0) as opened:
        yield opened


@pytest.fixture(scope='module')
def query():
    return PredictiveQuery(COUNT_QUERY, tables())


@requires_relational
def test_the_endpoint_reports_the_relational_model(client):
    assert 'kumo-relational' in client.models()


@requires_relational
def test_materializing_gathers_without_sending_anything(query):
    """The gathering step is explicit because it is the expensive one."""
    request = query.materialize(indices=[0, 1, 2])
    assert request.payloads, 'nothing was gathered'
    payload = request.payloads[0]
    for section in ('task', 'schema', 'context', 'predict'):
        assert section in payload, f'missing {section}'
    assert request.is_relational


@requires_relational
def test_a_relational_question_is_answered_per_entity(client, query):
    indices = [7, 12, 3, 20, 15]
    request = query.materialize(indices=indices)
    frame = client.predict(request, model=KumoRelational()).to_pandas()

    assert len(frame) == len(indices)
    # The entity is what was asked about, not the engine's transport key.
    assert [int(value) for value in frame['id']] == indices


@requires_relational
def test_the_predictions_track_the_data_they_were_given(client, query):
    """Even users order ten times as often, and the answers should say so."""
    busy, quiet = [0, 2, 4, 6], [1, 3, 5, 7]
    request = query.materialize(indices=busy + quiet)
    frame = client.predict(request, model=KumoRelational()).to_pandas()
    frame['entity'] = [int(value) for value in frame['id']]

    busy_mean = frame[frame['entity'].isin(busy)]['prediction'].mean()
    quiet_mean = frame[frame['entity'].isin(quiet)]['prediction'].mean()
    assert busy_mean > quiet_mean, (
        f'busy users predicted {busy_mean:.2f}, quiet {quiet_mean:.2f}'
    )


@requires_relational
def test_a_query_naming_a_column_that_does_not_exist_fails_before_sending():
    with pytest.raises((StructuredDataError, ValueError)) as error:
        PredictiveQuery(
            'PREDICT COUNT(orders.nope, 0, 30, days) FOR EACH users.user_id',
            tables(),
        ).materialize(indices=[1])
    assert 'nope' in str(error.value).lower()


@requires_relational
def test_an_aggregation_wider_than_the_history_is_refused_locally(client):
    """Point-in-time correctness: a 90-day window needs 90 days before it."""
    short = tables()
    short['orders'] = short['orders'][
        short['orders']['ts'] < pd.Timestamp('2024-01-20')
    ]
    with pytest.raises((StructuredDataError, ValueError)) as error:
        PredictiveQuery(
            'PREDICT COUNT(orders.*, 0, 90, days) FOR EACH users.user_id',
            short,
        ).materialize(indices=[0, 2])
    assert 'data' in str(error.value).lower()


@requires_relational
def test_materialize_refuses_to_score_a_whole_table_by_default(query):
    with pytest.raises(StructuredDataError) as error:
        query.materialize()
    assert 'indices' in str(error.value)


@requires_relational
def test_the_tabular_model_refuses_this_request_without_sending_it(
    client, query
):
    """A relational request needs relationships, which tabular rejects."""
    request = query.materialize(indices=[0, 1])
    with pytest.raises(UnsupportedRequestError):
        client.predict(request, model=KumoTabular())


@requires_relational
def test_an_inference_option_the_endpoint_refuses_is_named_locally(query):
    with pytest.raises(UnsupportedRequestError) as error:
        request = query.materialize(
            indices=[0, 1], inference_config={'output_type': 'quantiles'}
        )
        adapter_outputs = ('prediction',)
        from sdm_client.adapters.relational import RelationalAdapter

        list(RelationalAdapter().prediction_plan(request, adapter_outputs))
    assert 'output_type' in str(error.value)


@requires_relational
def test_asking_for_a_field_the_model_does_not_produce_is_refused(
    client, query
):
    request = query.materialize(indices=[0])
    with pytest.raises(UnsupportedRequestError) as error:
        client.predict(request, model=KumoRelational(), outputs=['scores_x'])
    assert 'scores_x' in str(error.value)


@requires_relational
def test_a_binary_question_comes_back_with_probabilities(client):
    request = PredictiveQuery(
        'PREDICT COUNT(orders.*, 0, 30, days) > 2 FOR EACH users.user_id',
        tables(),
    ).materialize(indices=[0, 1, 2, 3])
    frame = client.predict(request, model=KumoRelational()).to_pandas()
    assert len(frame) == 4
    probability_columns = [
        column for column in frame.columns if column.startswith('probability_')
    ]
    assert probability_columns, f'no probabilities in {list(frame.columns)}'
