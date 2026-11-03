# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A forecast explanation, drawn as Kumo-TS draws it.

The pages come from that package. What is new is that they are drawn from an
explanation that travelled as data: the client carries one as the contract's
``explanation`` field, plain JSON, because that is what an output field is.
The arrays the pages read are all in it, so the drawing reads them back.

A page whose data is not there is not drawn. Feature-axis attribution is the
one that is usually absent, because it costs several times the rest and is
asked for separately.
"""

from __future__ import annotations

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
    attributions = np.asarray(explanation.lag_horizon_attributions)
    # Scores, not attributions. That helper differences a cumulative score
    # into a per-step weight, and handing it the normalised attribution
    # ranked the lags in very nearly the opposite order.
    _, topk_rows = pages._topk_lag_steps_per_horizon(
        np.asarray(explanation.lag_horizon_scores), top_k=top_k
    )
    with _artifacts(artifacts_dir) as out_dir:
        heatmap = pages._save_lag_horizon_artifacts(
            out_dir,
            attributions,
            np.asarray(explanation.lag_horizon_scores),
        )
        # The pages name these files, so a report that draws the pages and
        # writes none of them points a reader at nothing. Each is written
        # where the data behind it was computed.
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
                channel_labels=_channels(carried),
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
            channel_labels=_channels(carried),
        )
    if drawn is None:  # pragma: no cover - matplotlib is present by here
        raise MissingExtraError('report', 'matplotlib')
    return drawn


def _pages() -> Any:
    try:
        from sdm_client.explain._kumo_ts import pages
    except ImportError as error:  # pragma: no cover - the extra is absent
        raise MissingExtraError('report', 'matplotlib') from error
    return pages


def _explanation(carried: dict[str, Any]) -> Any:
    r"""The explanation as the drawing expects it, read back from the data.

    Only the arrays the pages read are restored. A field the explanation did
    not carry stays absent, and the page that would have drawn it is not
    drawn, which is what "this was not computed" looks like on paper.
    """
    from sdm_client.explain._kumo_ts.interpretability import (
        ForecastExplanation,
    )

    block = carried.get('explanation') or {}
    # Serialised under `feature_axis`, not beside the lag arrays. Reading it
    # flat found nothing and the page that draws it was skipped although it
    # had been computed, which costs minutes.
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
