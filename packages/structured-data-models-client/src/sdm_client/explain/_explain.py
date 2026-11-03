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
    window = history[..., -forecaster.context_size :]
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
    return _as_dict(explanation, channels, target, horizon, forecast)


def _as_dict(
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
    return rendered
