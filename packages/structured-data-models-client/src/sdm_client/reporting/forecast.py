# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A forecast explanation, drawn as it was written to be drawn.

What is new is that the pages are drawn from an explanation that travelled as
data: the client carries one as the contract's ``explanation`` field, plain
JSON, because that is what an output field is. The arrays the pages read are
all in it, so the drawing reads them back.

A page whose data is not there is not drawn. Feature-axis attribution is the
one that is usually absent, because it costs several times the rest and is
asked for separately.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from sdm_client.errors import MissingExtraError, StructuredDataError

if TYPE_CHECKING:  # pragma: no cover - imported for typing only.
    from sdm_client.result import PredictionResult


def forecast_report(
    result: PredictionResult,
    output_path: str | Path,
    *,
    dataset_name: str | None = None,
    top_k: int = 5,
    artifacts_dir: str | Path | None = None,
) -> Path:
    r"""Draw why ``result`` is the forecast it is.

    Args:
        result: What :meth:`SDMClient.predict <sdm_client.SDMClient.predict>`
            answered for a forecast asked with ``'explanation'`` among its
            output fields.
        output_path: Where to write the PDF.
        dataset_name: A name for the run, shown on the first page.
        top_k: How many lags to list per horizon.
        artifacts_dir: Where to keep the CSVs and images the pages are drawn
            from. They are written either way, because the attribution
            heatmap is a page drawn from an image on disk; naming a
            directory is what keeps them afterwards.

    Raises:
        MissingExtraError: If matplotlib is not installed.
        StructuredDataError: If the answer carries no explanation.

    The feature axis sets the forecast series' own effect aside rather than
    measuring it, and it arrives as a zero. Left in, the page ranks it among
    measured channels and has called it the most influential of them, so it
    is taken out along with its label: a row that is not a measurement does
    not belong in a matrix of measurements.

    The lags are ranked from the scores rather than from the attributions.
    That helper differences a cumulative score into a per-step weight, and
    handing it the normalised attribution ranked the lags in very nearly the
    opposite order. It returns those per-step weights, which are what the
    heatmap's own caption describes; drawn from the cumulative scores
    instead, the heatmap read a run of flat history as the most influential
    part of the window while the table beside it ranked the newest step
    first.

    Each artifact is written where the data behind it was computed, because
    the pages name those files and a report that draws the pages and writes
    none of them points a reader at nothing.

    >>> result = client.predict(
    ...     request, model=KumoForecast(),
    ...     outputs=['prediction', 'explanation'],
    ... )
    >>> forecast_report(result, 'forecast.pdf')
    """
    carried = result.explanation
    if not carried:
        raise StructuredDataError(
            'this answer carries no explanation. Ask for one with '
            "outputs=['prediction', 'explanation']; it is not computed "
            'unless it is asked for, because it costs far more than the '
            'forecast does',
            code='INVALID_REQUEST',
        )
    pages = _pages()
    frame, target, step = _forecast_frame(carried)
    explanation = _explanation(carried)
    explanation, channels = _without_the_unmeasured(
        explanation, _channels(carried), carried
    )
    attributions = np.asarray(explanation.lag_horizon_attributions)
    marginal, topk_rows = pages._topk_lag_steps_per_horizon(
        np.asarray(explanation.lag_horizon_scores), top_k=top_k
    )
    per_step = attributions if marginal is None else np.asarray(marginal)
    smoothness = _smoothness(carried)
    stability, stability_channels = _stability(carried)
    gradients = (carried.get('explanation') or {}).get('integrated_gradients')
    with _artifacts(artifacts_dir) as out_dir:
        _saved_as_json(out_dir, carried)
        heatmap = pages._save_lag_horizon_artifacts(
            out_dir,
            per_step,
            np.asarray(explanation.lag_horizon_scores),
        )
        horizon = attributions.shape[1]
        context = _context_len(carried, horizon)
        if explanation.flow_magnitudes is not None and context:
            pages._save_semantic_flow_csv(
                out_dir,
                np.asarray(explanation.flow_magnitudes),
                context_len=context,
                forecast_horizon=horizon,
            )
        if explanation.channel_horizon_attributions is not None:
            pages._save_channel_axis_artifacts(
                out_dir,
                np.asarray(explanation.channel_horizon_attributions),
                channel_labels=channels,
            )
        if gradients is not None:
            pages._save_integrated_gradients_artifacts(out_dir, gradients)
        if stability is not None:
            pages._save_feature_axis_embedding_stability_csv(
                out_dir, stability, stability_channels
            )
        drawn = pages._build_pdf_report(
            Path(output_path),
            dataset_name=dataset_name,
            forecast_df=frame,
            explanation=explanation,
            target_column=target,
            timestamp_column=step,
            heatmap_path=heatmap,
            topk_rows=topk_rows,
            top_k=top_k,
            context_len=_context_len(carried, attributions.shape[1]),
            forecast_horizon=attributions.shape[1],
            channel_labels=channels,
            trajectory_report=smoothness,
            feature_axis_embedding_stability_report=stability,
            feature_axis_stability_channel_labels=stability_channels,
            integrated_gradients_report=gradients,
        )
    if drawn is None:  # pragma: no cover - matplotlib is present by here
        raise MissingExtraError('report', 'matplotlib')
    return drawn


def _pages() -> Any:
    try:
        from sdm_client.explain._attribution import pages
    except ImportError as error:  # pragma: no cover - the extra is absent
        raise MissingExtraError('report', 'matplotlib') from error
    return pages


def _explanation(carried: dict[str, Any]) -> Any:
    r"""The explanation as the drawing expects it, read back from the data.

    Only the arrays the pages read are restored. A field the explanation did
    not carry stays absent, and the page that would have drawn it is not
    drawn, which is what "this was not computed" looks like on paper.

    The channel attribution is serialised under ``feature_axis`` rather than
    beside the lag arrays. Read flat it was found nowhere, and the page that
    draws it was skipped although it had been computed, which costs minutes.
    """
    from sdm_client.explain._attribution.interpretability import (
        ForecastExplanation,
    )

    block = carried.get('explanation') or {}
    feature_axis = block.get('feature_axis') or {}
    diagnostics = block.get('diagnostics') or {}

    def array(source: dict[str, Any], name: str) -> Any:
        value = source.get(name)
        return None if value is None else np.asarray(value, dtype=float)

    def scalar(name: str) -> float | None:
        value = diagnostics.get(name)
        return None if value is None else float(value)

    return ForecastExplanation(
        baseline_forecast=array(block, 'baseline_forecast'),
        lag_horizon_scores=array(block, 'lag_horizon_scores'),
        lag_horizon_attributions=array(block, 'lag_horizon_attributions'),
        flow_magnitudes=array(block, 'flow_magnitudes'),
        latent_trajectory=array(block, 'latent_trajectory'),
        channel_horizon_attributions=array(
            feature_axis, 'channel_horizon_attributions'
        ),
        per_channel_flow=array(feature_axis, 'per_channel_flow'),
        channel_flow_method=feature_axis.get('method'),
        flow_ratio_forecast_vs_history=scalar('flow_ratio_forecast_vs_history'),
        flow_variance_ratio_forecast_vs_history=scalar(
            'flow_variance_ratio_forecast_vs_history'
        ),
        curvature_ratio_forecast_vs_history=scalar(
            'curvature_ratio_forecast_vs_history'
        ),
        latent_diag_mahalanobis_ratio_forecast_vs_history=scalar(
            'latent_diag_mahalanobis_ratio_forecast_vs_history'
        ),
    )


def _smoothness(carried: dict[str, Any]) -> Any:
    r"""How smoothly the latent state moved, read back from the data."""
    from sdm_client.explain._attribution.interpretability import (
        TrajectoryStabilityReport,
    )

    block = (carried.get('explanation') or {}).get('trajectory_stability')
    if not block:
        return None
    rates = (
        'zero_crossing_rate_mean',
        'zero_crossing_rate_p95',
        'direction_flip_rate_mean',
        'direction_flip_rate_p95',
        'relative_jitter_mean',
        'relative_jitter_p95',
        'occupancy_positive_mean',
        'occupancy_negative_mean',
    )
    if any(block.get(name) is None for name in rates):
        return None
    read = {name: float(block[name]) for name in rates}
    return TrajectoryStabilityReport(
        zero_crossing_rate_mean=read['zero_crossing_rate_mean'],
        zero_crossing_rate_p95=read['zero_crossing_rate_p95'],
        direction_flip_rate_mean=read['direction_flip_rate_mean'],
        direction_flip_rate_p95=read['direction_flip_rate_p95'],
        relative_jitter_mean=read['relative_jitter_mean'],
        relative_jitter_p95=read['relative_jitter_p95'],
        occupancy_positive_mean=read['occupancy_positive_mean'],
        occupancy_negative_mean=read['occupancy_negative_mean'],
        n_time_steps=int(block.get('n_time_steps') or 0),
        n_dimensions=int(block.get('n_dimensions') or 0),
    )


def _stability(carried: dict[str, Any]) -> tuple[Any, list[str] | None]:
    r"""How much the embedding moved when a channel was nudged, and the
    channels it is about.

    The labels it carries are its own and are handed back beside it. There
    are as many of them as there are channels, and that count includes the
    series being forecast, which the channel attribution beside it sets
    aside. Drawn under the explanation's labels instead, the two lists were
    different lengths, and the page fell back to numbering the channels.
    """
    from sdm_client.explain._attribution.channel_stability import (
        PerChannelEmbeddingStabilityReport,
    )

    feature_axis = (carried.get('explanation') or {}).get('feature_axis') or {}
    block = feature_axis.get('embedding_stability')
    if not block:
        return None, None
    labels = [str(name) for name in block.get('channel_labels') or []]
    ratios = {
        name: np.asarray(block.get(name), dtype=float)
        for name in (
            'lip_ratio_mean',
            'lip_ratio_max',
            'lip_ratio_p50',
            'lip_ratio_p95',
        )
    }
    reference = block.get('step_delta_norm_mean')
    return PerChannelEmbeddingStabilityReport(
        **ratios,
        n_trials_per_channel=np.asarray(
            block.get('n_trials_per_channel'), dtype='int64'
        ),
        n_unique_windows=int(block.get('n_unique_windows') or 0),
        step_delta_norm_mean=float('nan')
        if reference is None
        else float(reference),
        n_channels=len(labels),
    ), labels


def _forecast_frame(carried: dict[str, Any]) -> tuple[Any, str, str]:
    import pandas as pd

    rows = carried.get('forecast') or []
    frame = pd.DataFrame(rows)
    target = str((carried.get('metadata') or {}).get('target_column') or '')
    step = 'forecast_step'
    if f'{target}_forecast' not in frame.columns:
        raise StructuredDataError(
            f'this explanation names {target!r} as the series forecast and '
            f'carries {sorted(frame.columns)}, which does not include it',
            code='INVALID_RESPONSE',
        )
    return frame, target, step


def _channels(carried: dict[str, Any]) -> list[str] | None:
    labels = (carried.get('metadata') or {}).get('channel_labels')
    return None if labels is None else [str(name) for name in labels]


def _context_len(carried: dict[str, Any], horizon: int) -> int | None:
    r"""How many of the trajectory's steps are history.

    The trajectory runs the context and then the horizon, so its length is
    both put together. Reading it as the context alone told the page every
    transition was history and none was forecast.
    """
    block = (carried.get('explanation') or {}).get('diagnostics') or {}
    shape = block.get('latent_trajectory_shape')
    if not shape:
        return None
    steps = int(shape[0]) - horizon
    return steps if steps > 0 else None


@contextmanager
def _artifacts(named: str | Path | None) -> Iterator[Path]:
    r"""Where the images and CSVs the pages read are written.

    The attribution heatmap is a page drawn from a file, so there is always
    somewhere for it to be. A caller who named a directory keeps what was
    written there; one who did not gets a temporary one that goes away.
    """
    if named is not None:
        out_dir = Path(named)
        out_dir.mkdir(parents=True, exist_ok=True)
        yield out_dir
        return
    with tempfile.TemporaryDirectory(prefix='sdm-forecast-') as temporary:
        yield Path(temporary)


def _saved_as_json(out_dir: Path, carried: dict[str, Any]) -> Path:
    r"""The explanation these pages were drawn from, written out beside them.

    Three of the pages send the reader to ``explanation.json`` for the
    diagnostics block and for the values they only summarise, and nothing
    wrote one, so the reader was sent to a file that was never there.

    What is written is the answer's own explanation, unchanged: the pages
    read nothing else, so anything derived here could disagree with them.
    """
    out_path = out_dir / 'explanation.json'
    with out_path.open('w', encoding='utf-8') as handle:
        json.dump(carried, handle, indent=2, allow_nan=False)
    return out_path


def _without_the_unmeasured(
    explanation: Any, channels: list[str] | None, carried: dict[str, Any]
) -> tuple[Any, list[str] | None]:
    r"""``explanation`` with any channel it did not measure removed."""
    from dataclasses import replace

    excluded = (carried.get('metadata') or {}).get('feature_axis_excludes')
    attributions = explanation.channel_horizon_attributions
    if excluded is None or attributions is None or not channels:
        return explanation, channels
    if excluded not in channels:
        return explanation, channels
    keep = [name != excluded for name in channels]
    return (
        replace(
            explanation,
            channel_horizon_attributions=np.asarray(attributions)[keep],
            per_channel_flow=(
                None
                if explanation.per_channel_flow is None
                else np.asarray(explanation.per_channel_flow)[keep]
            ),
        ),
        [name for name in channels if name != excluded],
    )
