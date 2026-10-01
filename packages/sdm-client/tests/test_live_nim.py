# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""End-to-end against a running NIM.

Excluded by default. Point ``SDM_TABULAR_URL`` at a NIM serving
``kumo-tabular`` and run with ``-m live_nim``:

    SDM_TABULAR_URL=http://localhost:8000 pytest -m live_nim

These are the tests that catch what a mock cannot: that the envelope this
client builds is one the endpoint actually accepts, and that what comes back
parses.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from sdm_client import (
    KumoTabular,
    Request,
    SDMClient,
    UnsupportedRequestError,
)

pytestmark = pytest.mark.live_nim

TABULAR_URL = os.environ.get('SDM_TABULAR_URL')

requires_tabular = pytest.mark.skipif(
    not TABULAR_URL, reason='set SDM_TABULAR_URL to a kumo-tabular NIM'
)


@pytest.fixture
def client():
    with SDMClient(url=TABULAR_URL, timeout=180.0) as opened:
        yield opened


def labelled(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'row_id': range(n),
            'age': [20.0 + i * 0.7 for i in range(n)],
            'spend': [100.0 + 9 * i for i in range(n)],
            'segment': ['north' if i % 3 else 'south' for i in range(n)],
            'churn': [i % 2 for i in range(n)],
        }
    )


def unlabelled() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'row_id': [900, 901, 902],
            'age': [25.0, 44.0, 31.5],
            'spend': [150.0, 430.0, 210.0],
            'segment': ['north', 'south', 'north'],
        }
    )


@requires_tabular
def test_the_endpoint_reports_the_model_it_serves(client):
    """Read from /v1/models, not from the client's own list of models."""
    assert 'kumo-tabular' in client.models()


@requires_tabular
def test_the_endpoint_is_ready(client):
    assert client.health_ready() is True


@requires_tabular
def test_a_classification_request_scores_every_row_it_was_given(client):
    request = Request(
        context=labelled(),
        predict=unlabelled(),
        target='churn',
        task='classification',
    )
    result = client.predict(request, model=KumoTabular())
    frame = result.to_pandas()

    assert len(frame) == 3
    assert list(frame['row_index']) == [0, 1, 2]
    assert 'prediction' in frame.columns
    # Probabilities are asked for by default on a classification task.
    assert {'probability_0', 'probability_1'} <= set(frame.columns)
    for column in ('probability_0', 'probability_1'):
        assert frame[column].between(0.0, 1.0).all()
    assert (
        (frame['probability_0'] + frame['probability_1'])
        .sub(1.0)
        .abs()
        .lt(1e-5)
        .all()
    )


@requires_tabular
def test_a_regression_request_returns_a_number_per_row(client):
    context = labelled().drop(columns=['churn'])
    request = Request(
        context=context,
        predict=unlabelled().drop(columns=['spend']),
        target='spend',
        task='regression',
    )
    result = client.predict(request, model=KumoTabular())
    frame = result.to_pandas()
    assert len(frame) == 3
    assert pd.api.types.is_numeric_dtype(frame['prediction'])


@requires_tabular
def test_asking_for_only_a_prediction_returns_no_probability_columns(client):
    request = Request(
        context=labelled(),
        predict=unlabelled(),
        target='churn',
        task='classification',
    )
    frame = client.predict(
        request, model=KumoTabular(), outputs=['prediction']
    ).to_pandas()
    assert 'prediction' in frame.columns
    assert not [c for c in frame.columns if c.startswith('probability_')]


@requires_tabular
def test_a_fitted_context_is_reused_across_calls(client):
    """`fit` pins the context; scoring sends only the rows."""
    request = Request(
        context=labelled(),
        predict=unlabelled(),
        target='churn',
        task='classification',
    )
    with client.fit(request, model=KumoTabular()) as fitted:
        assert fitted.model == 'kumo-tabular'
        first = fitted.predict(request).to_pandas()
        assert len(first) == 3

        other = Request(
            context=labelled(),
            predict=unlabelled().head(1),
            target='churn',
            task='classification',
        )
        second = fitted.predict(other).to_pandas()
        assert len(second) == 1


@requires_tabular
def test_a_single_row_scores(client):
    request = Request(
        context=labelled(),
        predict=unlabelled().head(1),
        target='churn',
        task='classification',
    )
    assert len(client.predict(request, model=KumoTabular()).to_pandas()) == 1


@requires_tabular
def test_a_relational_request_is_refused_without_reaching_the_endpoint(client):
    """Refused where the tables are offered, which is earlier than the profile
    check that used to catch it, so no request is built and none is sent."""
    orders = pd.DataFrame({'order_id': [1, 2], 'row_id': [0, 1]})

    with pytest.raises(UnsupportedRequestError):
        Request(
            context=labelled(),
            predict=unlabelled(),
            target='churn',
            task='classification',
            related_tables={'orders': orders},
        )


@requires_tabular
def test_predictions_join_on_the_key_the_result_carries(client):
    """The documented way to use a result: join, never rely on row order."""
    to_score = unlabelled()
    request = Request(
        context=labelled(),
        predict=to_score,
        target='churn',
        task='classification',
    )
    frame = client.predict(request, model=KumoTabular()).to_pandas()
    joined = to_score.reset_index(names='row_index').merge(
        frame, on='row_index', validate='one_to_one'
    )
    assert len(joined) == len(to_score)
    assert joined['prediction'].notna().all()


@requires_tabular
def test_a_tabular_question_runs_from_predictive_query_text(client):
    """Phase 7: the query says what to predict, the rows say what to score.

    Both halves come from different places on purpose. A query resolves against
    a schema and so can only name rows that are already there; these are not.
    """
    pytest.importorskip('predictive_query')
    from sdm_client import PredictiveQuery

    request = PredictiveQuery(
        'PREDICT instance.churn FOR EACH instance.row_id',
        {'instance': labelled()},
    ).materialize(predict=unlabelled())

    assert request.task.target.column_name == 'churn'
    # Never narrowed: the model decides binary against multiclass itself.
    assert request.task.kind == 'classification'

    frame = client.predict(request, model=KumoTabular()).to_pandas()
    assert len(frame) == 3
    assert 'prediction' in frame.columns
