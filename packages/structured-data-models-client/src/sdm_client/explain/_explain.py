# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Running the attribution over a history, and saying what it found.

The ported pipeline wants the context as channels by steps and the model
behind a protocol; the client holds a history as a frame and a loaded
structured-data-models forecaster. This is what stands between them, and the
dict it returns is what travels as the ``explanation`` output field.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from sdm_client.errors import StructuredDataError

#: How far back the lag attribution looks, and how finely. Upstream's
#: defaults; both cost time roughly in proportion.
_LAGS = 128


def explanation_of(
    model: Any,
    history: np.ndarray,
    *,
    channels: list[str],
    target: str,
    horizon: int,
    forecast: np.ndarray,
    feature_axis: bool = False,
) -> dict[str, Any]:
    r"""Attribute ``forecast`` to the history it was made from.

    Args:
        model: The loaded ``sdm.models.KumoForecast``.
        history: The context as ``[C, R]``, channels by steps, in the order
            ``channels`` names them.
        channels: What each channel is called.
        target: Which of them was forecast.
        horizon: How many steps were asked for.
        forecast: What came back, one value per step.
        feature_axis: Also attribute across channels rather than only across
            lags. It costs several times the rest put together.

    Raises:
        StructuredDataError: If the history is shorter than the model reads.
    """

    from sdm_client.explain._kumo_ts.interpretability import explain_forecast
    from sdm_client.explain._model import Forecaster

    forecaster = Forecaster(model)
    if history.shape[-1] < forecaster.context_size:
        raise StructuredDataError(
            f'an explanation attributes a forecast to the window the model '
            f'reads, which is {forecaster.context_size} steps, and this '
            f'history is {history.shape[-1]}',
            code='UNSUPPORTED_SHAPE',
        )
    # Standardised first, as the model's own path standardises before it
    # reads anything. The attribution then works in the space the model works
    # in; handing it raw readings attributed a model being fed numbers it
    # never sees.
    if len(channels) > 1 and not forecaster.mixes_channels:
        # Without channel mixing and without retrieval, a covariate cannot
        # reach the forecast at all. The embedding still pools every channel,
        # so an attribution would move when one of them moves and report an
        # influence the forecast does not have.
        raise StructuredDataError(
            'this forecaster does not mix channels, so nothing measured '
            'alongside the series can reach the forecast, and an attribution '
            'over them would report an influence there is none of. Explain a '
            'history of the series alone, or use a forecaster that mixes '
            'channels',
            code='UNSUPPORTED_REQUEST',
        )
    window_only = history[..., -forecaster.context_size :]
    if feature_axis and not np.isfinite(window_only).all():
        # Feature-axis attribution perturbs a channel towards its own mean
        # over the window, and that mean is NaN if any reading in it is
        # missing. The channel then reads as inactive and is given no
        # influence at all, which is a confident answer about a column
        # nothing was learnt about. Refused rather than reported.
        raise StructuredDataError(
            f'feature-axis attribution replaces a channel with its own '
            f'average over the window it reads, which a gap in that window '
            f'makes undefined, and the channel is then reported as having no '
            f'influence. The window is the last {forecaster.context_size} '
            f'steps; fill its gaps, or ask for the explanation without the '
            f'feature axis',
            code='UNSUPPORTED_SHAPE',
        )
    if feature_axis and horizon > forecaster.prediction_size:
        raise StructuredDataError(
            f'feature-axis attribution reads one forward pass, which covers '
            f'{forecaster.prediction_size} steps, and this forecast is '
            f'{horizon}. Ask for it over a horizon the model reaches in one '
            f'pass, or ask for the explanation without it',
            code='UNSUPPORTED_SHAPE',
        )
    window = _standardised(model, window_only)
    explanation = explain_forecast(
        forecaster,
        x_context_ct=np.ascontiguousarray(window, dtype='float32'),
        input_mask_l=np.ones(forecaster.context_size, dtype='float32'),
        model_horizon=forecaster.prediction_size,
        forecast_horizon=horizon,
        device=next(model.parameters()).device,
        n_lags=min(_LAGS, forecaster.context_size),
        surrogate=False,
        channel_axis=feature_axis,
        channel_target=channels.index(target) if feature_axis else None,
        channel_output_aware=feature_axis,
    )
    return _as_dict(model, explanation, channels, target, horizon, forecast)


def _standardised(model: Any, window: np.ndarray) -> np.ndarray:
    r"""``window`` through the model's own frozen standardisation."""
    import pandas as pd
    from sdm import TableTensor

    frame = pd.DataFrame(
        {str(i): row for i, row in enumerate(np.asarray(window, dtype=float))}
    )
    table = TableTensor.from_pandas(
        frame, dict.fromkeys(frame.columns, 'numerical')
    )
    standardised = model.default_recipe().target.transform(table)
    return standardised.numerical.detach().cpu().numpy().T


def _in_original_scale(model: Any, values: np.ndarray) -> np.ndarray:
    r"""The inverse of :func:`_standardised`, for what is drawn."""
    import pandas as pd
    from sdm import TableTensor

    array = np.asarray(values, dtype=float)
    frame = pd.DataFrame({str(i): row for i, row in enumerate(array)})
    table = TableTensor.from_pandas(
        frame, dict.fromkeys(frame.columns, 'numerical')
    )
    restored = model.default_recipe().target.inverse_transform(table)
    return restored.numerical.detach().cpu().numpy().T


def _as_dict(
    model: Any,
    explanation: Any,
    channels: list[str],
    target: str,
    horizon: int,
    forecast: np.ndarray,
) -> dict[str, Any]:
    r"""The explanation as the contract carries one: plain JSON data.

    Rendered by the code that drew it upstream, so a report built from this
    and one built there say the same things in the same words.
    """
    from sdm_client.explain._kumo_ts.pages import _explanation_to_dict

    # The attribution works standardised, so what it calls the baseline
    # forecast is standardised too. The report draws it beside readings, so
    # it is put back in their scale before it travels.
    explanation = replace(
        explanation,
        baseline_forecast=_in_original_scale(
            model, np.asarray(explanation.baseline_forecast)
        ),
    )
    frame = pd.DataFrame(
        {
            'forecast_step': range(1, horizon + 1),
            f'{target}_forecast': np.asarray(forecast, dtype='float32'),
        }
    )
    rendered = _explanation_to_dict(
        frame,
        explanation,
        target_column=target,
        timestamp_column='forecast_step',
        channel_labels=list(channels),
    )
    # Upstream keeps the labels only where feature-axis attribution ran, and
    # they name the channels of every explanation. The client knows them
    # either way, and a report that cannot say which channel is which is
    # harder to read than one that can.
    rendered.setdefault('metadata', {})['channel_labels'] = list(channels)
    # The feature axis perturbs each channel and watches the forecast move.
    # Perturbing the series being forecast moves it by construction, so that
    # channel's own effect is set aside rather than measured, and it reaches
    # the page as a zero beside measured numbers. Saying so is what keeps a
    # reader from taking it for the least influential channel.
    if rendered.get('explanation', {}).get('feature_axis'):
        rendered['metadata']['feature_axis_excludes'] = target
    return rendered
