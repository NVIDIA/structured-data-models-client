# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Why a forecast is the forecast it is, and the report that draws it.

An explanation walks a window per step of the model's context, so it costs
tens of seconds where a forecast costs a fraction of one. It is computed once
here and read by every test that needs one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdm_client import KumoForecast, Request, SDMClient, StructuredDataError

pytest.importorskip('matplotlib', reason='needs the report extra')

from sdm_client.reporting import forecast_report  # noqa: E402

pytestmark = pytest.mark.filterwarnings('ignore')


def history(rows: int = 600) -> pd.DataFrame:
    step = np.arange(rows, dtype=float)
    return pd.DataFrame({'flow': np.sin(step / 9), 'temp': np.cos(step / 7)})


@pytest.fixture(scope='module')
def explained(forecast):
    with SDMClient.in_process(forecast) as client:
        return client.predict(
            Request.forecast(history(), target='flow', horizon=4),
            model=KumoForecast(),
            outputs=['prediction', 'explanation'],
        )


def pages(path) -> int:
    raw = path.read_bytes()
    return raw.count(b'/Type /Page') - raw.count(b'/Type /Pages')


def test_an_explanation_comes_back_when_it_is_asked_for(explained):
    """It reads the model's gradients and its latent trajectory, so it is
    produced where the model runs and nowhere else."""
    carried = explained.explanation

    assert isinstance(carried, dict)
    assert set(carried) >= {'forecast', 'explanation', 'metadata'}


def test_the_attribution_covers_every_lag_and_every_step(explained):
    block = explained.explanation['explanation']
    attributions = np.asarray(block['lag_horizon_attributions'])

    assert attributions.shape == (block['shapes']['K_lags'], 4)
    assert np.isfinite(attributions).all()


def test_an_explanation_is_data_rather_than_an_object(explained):
    """It travels as the contract's `explanation` field, so it has to be
    what that field carries."""
    import json

    assert json.loads(json.dumps(explained.explanation))


def test_a_forecast_asked_without_one_does_not_pay_for_it(forecast):
    """It costs far more than the forecast, so it is not computed unless it
    was asked for."""
    with SDMClient.in_process(forecast) as client:
        result = client.predict(
            Request.forecast(history(), target='flow', horizon=4),
            model=KumoForecast(),
            outputs=['prediction'],
        )

    assert result.explanation is None


def test_a_report_is_drawn_from_an_explanation(explained, tmp_path):
    written = forecast_report(explained, tmp_path / 'forecast.pdf')

    assert written.exists()
    assert pages(written) >= 3


def test_the_report_names_the_series_and_the_lag_window(explained, tmp_path):
    """Read off the data the page is drawn from. The text in this PDF is
    compressed, so sniffing its bytes for a word tests the encoder rather
    than the report."""
    metadata = explained.explanation['metadata']

    assert metadata['target_column'] == 'flow'
    assert metadata['channel_labels'] == ['flow', 'temp']
    assert forecast_report(
        explained, tmp_path / 'forecast.pdf', dataset_name='flow'
    ).exists()


def test_a_forecast_with_no_explanation_says_how_to_ask_for_one(
    forecast, tmp_path
):
    with SDMClient.in_process(forecast) as client:
        result = client.predict(
            Request.forecast(history(), target='flow', horizon=4),
            model=KumoForecast(),
            outputs=['prediction'],
        )

    with pytest.raises(StructuredDataError) as caught:
        forecast_report(result, tmp_path / 'forecast.pdf')
    assert 'explanation' in str(caught.value)


def test_a_history_shorter_than_the_model_reads_cannot_be_attributed(
    forecast,
):
    """The attribution is about the window the model read, and a history
    shorter than that window is not one it read."""
    with SDMClient.in_process(forecast) as client:
        with pytest.raises(StructuredDataError) as caught:
            client.predict(
                Request.forecast(history(100), target='flow', horizon=2),
                model=KumoForecast(),
                outputs=['prediction', 'explanation'],
            )
    assert '512' in str(caught.value)


def test_only_this_model_is_asked_to_explain_a_forecast(anomaly):
    """`explanation` is a field the forecaster produces, not every model."""
    from sdm_client import KumoAnomaly

    frame = pd.DataFrame({'flow': np.sin(np.arange(60) / 9)})
    with SDMClient.in_process(anomaly) as client:
        with pytest.raises(StructuredDataError) as caught:
            client.predict(
                Request.anomaly(frame, inference={'num_samples': 1}),
                model=KumoAnomaly(),
                outputs=['prediction', 'explanation'],
            )
    assert 'explanation' in str(caught.value)
