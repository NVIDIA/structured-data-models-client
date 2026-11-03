# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``kumo-forecast`` asked to continue a series it is shown the history of.

A small random-weight model stands in for the published one, so nothing is
downloaded and no GPU is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdm_client import (
    KumoForecast,
    KumoTabular,
    Request,
    SDMClient,
    StructuredDataError,
    UnsupportedRequestError,
)
from sdm_client.core.warnings import SDMWarning

pytestmark = pytest.mark.filterwarnings('ignore')

#: The published model reads 512 steps and forecasts 72, so retrieval needs a
#: history longer than both put together before it has anything to retrieve.
RETRIEVAL_NEEDS = 512 + 72


def history(rows: int = 200) -> pd.DataFrame:
    step = np.arange(rows, dtype=float)
    return pd.DataFrame({'flow': np.sin(step / 9), 'temp': np.cos(step / 7)})


def forecast_of(
    model,
    frame: pd.DataFrame,
    horizon: int = 6,
    target: str = 'flow',
    **inference,
):
    with SDMClient.in_process(model) as client:
        request = Request.forecast(
            frame,
            target=target,
            horizon=horizon,
            inference=inference or None,
        )
        return client.predict(request, model=KumoForecast()).to_pandas()


def test_one_row_per_step_comes_back_in_order(forecast):
    answer = forecast_of(forecast, history(), horizon=6)

    assert list(answer.columns) == ['row_index', 'forecast_step', 'prediction']
    # One row is scored, and the step counts from one, which is what the
    # contract declares `forecast_step` as.
    assert answer['row_index'].tolist() == [0] * 6
    assert answer['forecast_step'].tolist() == list(range(1, 7))
    assert np.isfinite(answer['prediction']).all()


@pytest.mark.parametrize('horizon', [1, 6, 80])
def test_the_horizon_is_how_many_steps_are_asked_for(forecast, horizon):
    """Beyond the model's own 72 the forecast is extended autoregressively,
    which is still the horizon that was asked for."""
    assert len(forecast_of(forecast, history(), horizon=horizon)) == horizon


def test_a_horizon_longer_than_the_history_is_answered_in_full(forecast):
    """How many steps were asked for does not depend on how many were seen.
    Reading the horizon off the history capped a long forecast at its length.
    """
    answer = forecast_of(forecast, history(10), horizon=15)

    assert len(answer) == 15
    assert answer['forecast_step'].tolist() == list(range(1, 16))


def test_a_series_with_nothing_measured_alongside_it_is_forecast(forecast):
    """The ordinary case. Reading the covariates as a shape of no columns
    left nothing to take a length from, and the request failed."""
    frame = history()[['flow']]

    assert len(forecast_of(forecast, frame, horizon=4)) == 4


def test_a_series_whose_values_are_unique_is_still_the_series(forecast):
    """Schema inference reads a column of unique values as the table's key,
    and a key names a step rather than measuring anything. Naming it as the
    target settles what it is, and dropping it left nothing to forecast."""
    frame = pd.DataFrame({'grid': np.arange(40, dtype=float) * 1.5})

    answer = forecast_of(forecast, frame, horizon=3, target='grid')

    assert len(answer) == 3


def test_a_key_that_is_not_the_target_is_still_a_key(forecast):
    """The premise of the test above: only the named series is rescued."""
    frame = history().assign(sensor_id=np.arange(len(history())))

    assert forecast_of(forecast, frame)['prediction'].tolist() == (
        pytest.approx(forecast_of(forecast, history())['prediction'].tolist())
    )


@pytest.mark.parametrize(
    'order', [['grid', 'sensor_id'], ['sensor_id', 'grid']]
)
def test_an_identifier_beside_the_target_is_the_key_instead(forecast, order):
    """Both columns are unique, so either could be read as the key. Taking
    the target out of that choice must leave the other one holding it;
    clearing the key outright let an identifier in as history."""
    rows = len(history())
    frame = pd.DataFrame(
        {
            'grid': np.arange(rows, dtype=float) * 1.5,
            'sensor_id': np.arange(100, 100 + rows),
        }
    )[order]

    request = Request.forecast(frame, target='grid', horizon=3)

    assert request.schema['instance'].primary_key == 'sensor_id'
    assert request.schema['instance']['grid'].stype == 'numerical'


def test_a_series_alone_has_no_key_to_fall_back_to(forecast):
    """The other end of the rule above: nothing beside it means no key."""
    frame = pd.DataFrame({'grid': np.arange(40, dtype=float) * 1.5})

    request = Request.forecast(frame, target='grid', horizon=3)

    assert request.schema['instance'].primary_key is None


def test_the_supervised_constructor_will_not_build_this_question(forecast):
    """Its scored half is the future, and this model reads nothing about the
    future: every value in those rows would be discarded and the answer come
    from the count alone."""
    frame = history()
    future = pd.DataFrame({'temp': [0.0] * 4})

    with SDMClient.in_process(forecast) as client:
        with pytest.raises(
            (UnsupportedRequestError, StructuredDataError)
        ) as caught:
            client.predict(
                Request(
                    context=frame,
                    predict=future,
                    target='flow',
                    task='forecasting',
                ),
                model=KumoForecast(),
            )
    assert 'Request.forecast' in str(caught.value)


def test_the_answer_is_the_one_the_model_gives_when_called_directly(
    sdm, forecast
):
    """Nothing is computed on the way through: the client shapes the request
    and reads the answer back, and the numbers are the model's own."""
    import torch

    frame = history()
    answer = forecast_of(forecast, frame, horizon=6)
    target = sdm.TableTensor.from_pandas(frame[['flow']], {'flow': 'numerical'})
    alongside = sdm.TableTensor.from_pandas(
        frame[['temp']], {'temp': 'numerical'}
    )
    direct = forecast(alongside, target, alongside[:6, ..., :0])

    torch.testing.assert_close(
        torch.tensor(answer['prediction'].to_numpy()),
        direct['flow'].numerical[..., 0].double(),
    )


def test_the_same_history_forecasts_the_same_twice(forecast):
    frame = history()

    assert forecast_of(forecast, frame)['prediction'].tolist() == (
        pytest.approx(forecast_of(forecast, frame)['prediction'].tolist())
    )


def test_a_key_column_is_not_read_as_history(forecast):
    """`reading_id` names the step. Forecasting from it would read a counter
    that rises for ever as though it were something measured."""
    frame = history()
    keyed = frame.assign(reading_id=np.arange(len(frame)))

    assert forecast_of(forecast, keyed)['prediction'].tolist() == (
        pytest.approx(forecast_of(forecast, frame)['prediction'].tolist())
    )


@pytest.mark.parametrize('horizon', [0, -1, 1.5, True, 10_001])
def test_a_horizon_that_is_not_a_count_of_steps_is_refused(forecast, horizon):
    """The upper bound is the contract's: `num_forecasts`, `forecast_step`
    and the number of predictions are each capped at ten thousand, so a
    longer horizon is one no deployment may answer."""
    with pytest.raises(StructuredDataError) as caught:
        Request.forecast(history(), target='flow', horizon=horizon)
    assert 'horizon' in str(caught.value)


def test_the_longest_horizon_the_contract_carries_is_accepted(forecast):
    """The premise of the bound above: ten thousand is inside it."""
    request = Request.forecast(history(), target='flow', horizon=10_000)

    assert request.horizon == 10_000


def test_a_history_without_the_series_is_refused(forecast):
    with pytest.raises(StructuredDataError) as caught:
        Request.forecast(history(), target='missing', horizon=4)
    assert 'missing' in str(caught.value)


def test_a_history_with_no_rows_is_refused(forecast):
    with pytest.raises(StructuredDataError) as caught:
        Request.forecast(history().iloc[:0], target='flow', horizon=4)
    assert 'no rows' in str(caught.value)


def test_a_column_the_model_cannot_forecast_from_is_refused(forecast):
    frame = history()
    frame['site'] = 'north'

    with pytest.raises(StructuredDataError) as caught:
        forecast_of(forecast, frame)
    assert 'site' in str(caught.value)


def test_retrieval_that_has_nothing_to_retrieve_from_says_so(forecast):
    """Below one window plus a horizon the model ignores `alpha` without
    saying anything, so a caller who set it would read a direct forecast and
    believe it was blended."""
    with pytest.warns(SDMWarning, match='retrieval'):
        forecast_of(forecast, history(200), alpha=0.2)


def test_retrieval_that_can_run_is_not_warned_about(forecast):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter('error', SDMWarning)
        forecast_of(forecast, history(RETRIEVAL_NEEDS), alpha=0.2)


def test_retrieval_changes_the_forecast_where_it_runs(forecast):
    """The premise of the warning above: with enough history it does
    something, so a run that is silently skipped is worth saying."""
    frame = history(RETRIEVAL_NEEDS)
    direct = forecast_of(forecast, frame)['prediction'].tolist()
    blended = forecast_of(forecast, frame, alpha=0.2)['prediction'].tolist()

    assert direct != pytest.approx(blended)


@pytest.mark.parametrize(
    ('option', 'value'),
    [('alpha', -0.1), ('alpha', 1.5), ('k', 0), ('k', 1.5)],
)
def test_a_retrieval_knob_outside_its_range_is_refused(forecast, option, value):
    with pytest.raises(StructuredDataError) as caught:
        forecast_of(forecast, history(), **{option: value})
    assert option in str(caught.value)


def test_there_is_no_session_to_pin(forecast):
    with SDMClient.in_process(forecast) as client:
        with pytest.raises(
            (UnsupportedRequestError, StructuredDataError)
        ) as caught:
            client.fit(
                Request.forecast(history(), target='flow', horizon=4),
                model=KumoForecast(),
            )
    assert 'session' in str(caught.value)


def test_another_model_is_not_asked_this_question(tabular):
    with SDMClient.in_process(tabular) as client:
        with pytest.raises(StructuredDataError):
            client.predict(
                Request.forecast(history(), target='flow', horizon=4),
                model=KumoTabular(),
            )


def test_a_structured_data_models_without_this_model_still_answers(
    sdm, monkeypatch
):
    monkeypatch.delattr(sdm.models, 'KumoForecast', raising=False)

    with pytest.raises(StructuredDataError) as caught:
        SDMClient.in_process(object())
    assert caught.value.code == 'INVALID_CONFIGURATION'


def test_a_model_that_is_not_installed_is_not_offered(sdm, monkeypatch):
    monkeypatch.delattr(sdm.models, 'KumoForecast', raising=False)

    with SDMClient.in_process() as client:
        assert 'kumo-forecast' not in client.models()
