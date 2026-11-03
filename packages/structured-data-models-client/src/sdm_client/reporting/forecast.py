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
) -> Path:
    r"""Draw why ``result`` is the forecast it is.

    Args:
        result: What :meth:`SDMClient.predict <sdm_client.SDMClient.predict>`
            answered for a forecast asked with ``'explanation'`` among its
            output fields.
        output_path: Where to write the PDF.
        dataset_name: A name for the run, shown on the first page.
        top_k: How many lags to list per horizon.

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
    _, topk_rows = pages._topk_lag_steps_per_horizon(attributions, top_k=top_k)
    drawn = pages._build_pdf_report(
        Path(output_path),
        dataset_name=dataset_name,
        forecast_df=frame,
        explanation=explanation,
        target_column=target,
        timestamp_column=step,
        heatmap_path=None,
        topk_rows=topk_rows,
        top_k=top_k,
        context_len=_context_len(carried),
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

    def array(name: str) -> Any:
        value = block.get(name)
        return None if value is None else np.asarray(value, dtype=float)

    return ForecastExplanation(
        baseline_forecast=array('baseline_forecast'),
        lag_horizon_scores=array('lag_horizon_scores'),
        lag_horizon_attributions=array('lag_horizon_attributions'),
        flow_magnitudes=array('flow_magnitudes'),
        latent_trajectory=array('latent_trajectory'),
        channel_horizon_attributions=array('channel_horizon_attributions'),
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


def _context_len(carried: dict[str, Any]) -> int | None:
    block = (carried.get('explanation') or {}).get('diagnostics') or {}
    shape = block.get('latent_trajectory_shape')
    return None if not shape else int(shape[0])
