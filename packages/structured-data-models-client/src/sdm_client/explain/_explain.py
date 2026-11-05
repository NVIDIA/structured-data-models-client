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

#: How many times each channel is nudged when the stability of the embedding
#: is measured. Upstream's default, named here because the memory the probes
#: take is worked out from it before they are built.
_STABILITY_TRIALS = 30

#: The most the stability probes may take. They are one window per channel
#: per trial and a copy of each, so the memory grows with the square of the
#: number of channels: forty channels of a 512-step window is already most of
#: a gigabyte once both stacks are counted.
_STABILITY_BYTES = 256 * 1024 * 1024

#: How finely the path from a reference window to the real one is walked
#: when the embedding is attributed step by step, and how many of those
#: points are embedded at once. Upstream's step count; the batch bounds the
#: activations rather than the answer, which does not depend on it.
_GRADIENT_STEPS = 64
_GRADIENT_BATCH = 8

#: What of the embedding the attribution explains the movement of. Upstream
#: explains the length of it, which cannot be the answer here: this
#: forecaster's embedding is a retrieval key and is scaled to unit length
#: before it is returned, so its length is one whatever it was given, and
#: attributing a constant attributed nothing. Measured on the published
#: weights: the length moved by 2e-07 between a real window and a reference
#: one and every reading came back a billionth, where the sum moves and is
#: explained.
_GRADIENT_OBJECTIVE = 'sum'


def explanation_of(
    model: Any,
    history: np.ndarray,
    *,
    channels: list[str],
    target: str,
    horizon: int,
    forecast: np.ndarray,
    feature_axis: bool = False,
    integrated_gradients: bool = False,
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
        integrated_gradients: Also attribute the model's own reading of the
            window, reading by reading, by walking the path from a reference
            window to this one. It is the only attribution here that reads
            the model's gradients rather than watching its output move.

    Raises:
        StructuredDataError: If the history is shorter than the model reads,
            if the forecaster cannot mix channels while more than one
            travelled, or, for the feature axis, if the window it reads has
            a gap in it or the horizon runs past one forward pass.

    Without channel mixing and without retrieval, a column measured
    alongside the series cannot reach the forecast at all, while the
    embedding still pools it, so an attribution would move when it moves and
    report an influence the forecast does not have. Feature-axis attribution
    replaces a channel with its own average over the window it reads, and a
    gap makes that average undefined, which reads as a channel with no
    influence: a confident answer about a column nothing was learnt about.
    Both are refused rather than reported.

    The window is standardised first, as the model's own path standardises
    before it reads anything, so the attribution works in the space the
    model works in. Handing it raw readings attributed a model being fed
    numbers it never sees.
    """

    from sdm_client.explain._attribution.interpretability import (
        explain_forecast,
    )
    from sdm_client.explain._model import Forecaster

    forecaster = Forecaster(model)
    if history.shape[-1] < forecaster.context_size:
        raise StructuredDataError(
            f'an explanation attributes a forecast to the window the model '
            f'reads, which is {forecaster.context_size} steps, and this '
            f'history is {history.shape[-1]}',
            code='UNSUPPORTED_SHAPE',
        )
    if len(channels) > 1 and not forecaster.mixes_channels:
        raise StructuredDataError(
            'this forecaster does not mix channels, so nothing measured '
            'alongside the series can reach the forecast, and an attribution '
            'over them would report an influence there is none of. Explain a '
            'history of the series alone, or use a forecaster that mixes '
            'channels',
            code='UNSUPPORTED_REQUEST',
        )
    window_only = history[..., -forecaster.context_size :]
    over_the_window = [
        name
        for name, asked in (
            ('the feature axis', feature_axis),
            ('integrated gradients', integrated_gradients),
        )
        if asked
    ]
    if over_the_window and not np.isfinite(window_only).all():
        raise StructuredDataError(
            f'this explanation was asked for '
            f'{" and ".join(over_the_window)}, and each of those measures '
            f'against a baseline worked out from the window it reads -- a '
            f'channel replaced by its own average, or the window replaced by '
            f'noise. A gap makes that baseline undefined, and every '
            f'attribution taken against it is then NaN, which reads as no '
            f'influence at all. The window is the last '
            f'{forecaster.context_size} steps; fill its gaps, or ask for the '
            f'explanation without them',
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
    device = next(model.parameters()).device
    window = _standardised(model, window_only)
    explanation = explain_forecast(
        forecaster,
        x_context_ct=np.ascontiguousarray(window, dtype='float32'),
        input_mask_l=np.ones(forecaster.context_size, dtype='float32'),
        model_horizon=forecaster.prediction_size,
        forecast_horizon=horizon,
        device=device,
        n_lags=min(_LAGS, forecaster.context_size),
        surrogate=False,
        channel_axis=feature_axis,
        channel_target=channels.index(target) if feature_axis else None,
        channel_output_aware=feature_axis,
    )
    return _as_dict(
        model,
        explanation,
        channels,
        target,
        horizon,
        forecast,
        smoothness=_smoothness_of(explanation, forecaster.context_size),
        stability=(
            _stability_of(forecaster, window, device) if feature_axis else None
        ),
        gradients=(
            _gradients_of(forecaster, window, channels, device)
            if integrated_gradients
            else None
        ),
    )


def _smoothness_of(explanation: Any, context_size: int) -> Any:
    r"""How smoothly the model's latent state moved over the history.

    Read off the trajectory the attribution already walked rather than walked
    again. That trajectory runs the history and then the forecast, one state
    per step, and its first ``context_size`` states are exactly the states a
    second walk over the history alone would produce, because each one is the
    window ending at its own step and the forecast is not in any of them.
    """
    from sdm_client.explain._attribution.interpretability import (
        summarize_trajectory_stability,
    )

    trajectory = np.asarray(explanation.latent_trajectory)
    history = trajectory[:context_size]
    if history.shape[0] < 2:
        return None
    return summarize_trajectory_stability(history)


def _stability_of(forecaster: Any, window: np.ndarray, device: Any) -> Any:
    r"""How much the embedding moves when one channel is nudged.

    Asked for alongside the feature axis, because it answers the question the
    feature axis raises: an influence attributed to a channel the embedding
    treats erratically is an influence measured through a derivative that is
    not there.
    """
    import warnings

    from sdm_client.core.warnings import SDMWarning
    from sdm_client.explain._attribution.channel_stability import (
        compute_per_channel_embedding_stability,
    )

    channels, steps = window.shape
    probes = 2 * (1 + _STABILITY_TRIALS * channels) * channels * steps * 4
    if probes > _STABILITY_BYTES:
        warnings.warn(
            f'the stability of the embedding is measured by nudging each of '
            f'these {channels} channels {_STABILITY_TRIALS} times, which '
            f'would take {probes // (1024 * 1024)} MiB of probes at once. '
            f'That page is left out of this explanation; the rest of the '
            f'feature axis is not',
            SDMWarning,
            stacklevel=2,
        )
        return None
    return compute_per_channel_embedding_stability(
        forecaster,
        np.ascontiguousarray(window, dtype='float32'),
        seq_len=int(steps),
        input_mask_t=np.ones(steps, dtype='float32'),
        device=device,
        n_trials=_STABILITY_TRIALS,
    )


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


def _gradients_of(
    forecaster: Any, window: np.ndarray, channels: list[str], device: Any
) -> Any:
    r"""What the model read, reading by reading, by integrated gradients.

    Held inside the forecaster's differentiable normalisation, because a
    gradient taken outside it describes a forward pass with each window's own
    statistics frozen, which is a different pass from the one that answered.
    What that costs is measurable and was measured, on the published
    weights because an untrained forecaster's embedding barely moves at all
    and the measure divides by how far it moved: an attribution has to add
    up to the movement it explains, and taken outside this it accounted for
    none of that movement and overshot it by half again, where taken inside
    it accounts for all but a twenty-fifth of a percent. The ported code offers
    a patch of its own for the same purpose and finds nothing to patch here,
    since it looks for a method this forecaster spells another way; the flag
    it records is true either way, and true.
    """
    from sdm_client.explain._attribution.pages import (
        _compute_integrated_gradients_report,
    )

    with forecaster.differentiable_normalisation():
        return _compute_integrated_gradients_report(
            model=forecaster,
            x_context_ct=np.ascontiguousarray(window, dtype='float32'),
            input_mask_l=np.ones(window.shape[-1], dtype='float32'),
            device=device,
            channel_labels=list(channels),
            baseline='noise',
            n_steps=_GRADIENT_STEPS,
            n_baselines=1,
            internal_batch_size=_GRADIENT_BATCH,
            reduce=_GRADIENT_OBJECTIVE,
            seed=0,
            grad_through_norm=True,
        )


def _as_dict(
    model: Any,
    explanation: Any,
    channels: list[str],
    target: str,
    horizon: int,
    forecast: np.ndarray,
    *,
    smoothness: Any = None,
    stability: Any = None,
    gradients: Any = None,
) -> dict[str, Any]:
    r"""The explanation as the contract carries one: plain JSON data.

    Rendered by the code that drew it, so a report built from this and one
    built where that code came from say the same things in the same words.

    The attribution works standardised, so what it calls the baseline
    forecast is standardised too, and the report draws it beside readings;
    it is put back in their scale before it travels. The channel labels
    travel with every explanation rather than only with the feature axis,
    which is where that code keeps them: the client knows them either way,
    and a report that cannot say which channel is which is harder to read
    than one that can.

    The feature axis perturbs each channel and watches the forecast move.
    Perturbing the series being forecast moves it by construction, so that
    channel's own effect is set aside rather than measured and reaches the
    page as a zero beside measured numbers. Saying which channel that was is
    what keeps a reader from taking it for the least influential one.
    """
    from sdm_client.explain._attribution.pages import _explanation_to_dict

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
        trajectory_report=smoothness,
        feature_axis_embedding_stability_report=stability,
        integrated_gradients_report=gradients,
    )
    rendered.setdefault('metadata', {})['channel_labels'] = list(channels)
    if rendered.get('explanation', {}).get('feature_axis'):
        rendered['metadata']['feature_axis_excludes'] = target
    return rendered
