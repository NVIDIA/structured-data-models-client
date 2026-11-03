# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""An anomaly run, put back together and drawn.

The answer says how anomalous each row was. It does not say what was in the
row, because the client sent that and does not send it back, so the readings
have to be rejoined to their scores before there is anything to plot.

**That join is on ``row_index`` and on nothing else.** It is the position of
the row in the frame that was sent, so it is the one thing that identifies a
row whatever the readings held. A key would be the obvious alternative and is
wrong here: this client accepts a key that is nearly unique, so joining on one
can match a score to several readings and quietly multiply the series being
drawn. Row order is wronger still, because answers may arrive in any order.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from sdm_client.errors import MissingExtraError, StructuredDataError

if TYPE_CHECKING:  # pragma: no cover - imported for typing only.
    from sdm_client.result import PredictionResult

#: The thresholders an anomaly answer carries, under the names
#: :mod:`sdm_client.in_process.anomaly` puts them back together as.
THRESHOLDERS = ('scs', 'macs')

_SCORE = 'prediction'
_DETECTED = 'detected'


def anomaly_report(
    result: PredictionResult,
    readings: pd.DataFrame,
    output_path: str | Path,
    *,
    detected_by: str = 'scs',
    feature_columns: Sequence[Hashable] | None = None,
    timestamp_column: str | None = None,
    ground_truth_column: str | None = None,
    title: str = 'Kumo-Anomaly report',
    max_report_pages: int = 10,
) -> Path:
    r"""Draw ``result`` over the ``readings`` it scored, as a PDF.

    Args:
        result: What :meth:`SDMClient.predict <sdm_client.SDMClient.predict>`
            answered for an anomaly request. Must carry the thresholder's
            labels, which means ``probabilities`` was among the output fields;
            it is by default.
        readings: The frame that was scored. Rows are matched to their
            scores by position in it, so it is that frame and not a filtered
            or reordered copy of it.

            It may carry **columns** the model was not sent, and usually
            does: the model reconstructs numbers and refuses a timestamp, so
            a caller scores the channels and keeps the timestamp to plot
            against. Dropping a column does not move a row, so the positions
            still line up.
        output_path: Where to write the PDF.
        detected_by: Which thresholder's labels mark a row as detected,
            ``'scs'`` or ``'macs'``. They disagree, and a report has to mark
            one of them, so it says which.
        feature_columns: The signals to draw. By default every numerical
            column of ``readings`` that is not the timestamp or the ground
            truth, which is what the model scored.
        timestamp_column: The column to plot against. Inferred from
            conventional names when not given; row numbers when there is none.
        ground_truth_column: Known anomalies to draw alongside the detected
            ones, when the caller has them. Inferred from conventional names.
        title: The title on the first page.
        max_report_pages: The most pages to draw. Signals are grouped to fit.

    Raises:
        MissingExtraError: If matplotlib is not installed.
        StructuredDataError: If the answer is not an anomaly answer, does not
            carry the labels asked for, or does not line up with ``readings``.

    >>> result = client.predict(Request.anomaly(readings), model=KumoAnomaly())
    >>> anomaly_report(result, readings, 'readings.pdf')
    """
    if detected_by not in THRESHOLDERS:
        raise StructuredDataError(
            f'detected_by is {detected_by!r}; it names one of '
            f'{list(THRESHOLDERS)}, whose labels disagree, so a report marks '
            f'one of them and says which',
            code='INVALID_REQUEST',
        )
    pages = _pages()
    frame = _rejoined(result, readings, detected_by)
    resolved = _signals(
        readings,
        feature_columns,
        timestamp_column=timestamp_column
        or pages.infer_timestamp_column(readings),
        ground_truth_column=ground_truth_column
        or pages.infer_ground_truth_column(readings),
    )
    return pages.generate_anomaly_detection_report(
        frame,
        output_path,
        feature_columns=resolved,
        timestamp_column=timestamp_column,
        ground_truth_column=ground_truth_column,
        anomaly_column=_DETECTED,
        score_column=_SCORE,
        title=title,
        max_report_pages=max_report_pages,
    )


def _pages() -> Any:
    try:
        from sdm_client.reporting import _pages as drawn
    except ImportError as error:  # pragma: no cover - the extra is absent
        raise MissingExtraError('report', 'matplotlib') from error
    return drawn


def _rejoined(
    result: PredictionResult, readings: pd.DataFrame, detected_by: str
) -> pd.DataFrame:
    r"""``readings`` with each row's score and label put back on it."""
    answer = result.to_pandas()
    label = f'probability_{detected_by}'
    if _SCORE not in answer.columns or label not in answer.columns:
        raise StructuredDataError(
            f'this answer carries {sorted(answer.columns)}, and a report '
            f'needs {_SCORE!r} and {label!r}. Ask for them with '
            f"outputs=['prediction', 'probabilities']",
            code='INVALID_REQUEST',
        )
    if 'row_index' not in answer.columns:
        raise StructuredDataError(
            'this answer names no row_index, so its scores cannot be matched '
            'to the readings they are about',
            code='INVALID_RESPONSE',
        )

    positions = answer['row_index'].to_numpy()
    if len(answer) != len(readings) or sorted(positions) != list(
        range(len(readings))
    ):
        raise StructuredDataError(
            f'the answer is about {len(answer)} row(s) and these readings '
            f'are {len(readings)}, so they are not the frame that was '
            f'scored. Pass the frame the request was built from, unfiltered '
            f'and in the order it was sent',
            code='INVALID_REQUEST',
        )

    # `row_index` is a position in the frame that was sent, not a label of
    # this frame's index, so the readings are taken by position and given a
    # fresh index. A frame indexed 100..199 would otherwise line up with
    # nothing.
    drawn = readings.reset_index(drop=True).copy()
    ordered = answer.sort_values('row_index', kind='stable')
    drawn[_SCORE] = ordered[_SCORE].to_numpy()
    drawn[_DETECTED] = ordered[label].to_numpy().astype(int)
    return drawn


def _signals(
    readings: pd.DataFrame,
    feature_columns: Sequence[Hashable] | None,
    *,
    timestamp_column: str | None,
    ground_truth_column: str | None,
) -> list[Hashable]:
    r"""The columns to draw as signals.

    Named rather than left to be worked out from the joined frame: that frame
    carries the score and the label, and a default that reads every numerical
    column off it would plot those as though they were readings.
    """
    if feature_columns is not None:
        missing = [
            name for name in feature_columns if name not in readings.columns
        ]
        if missing:
            raise StructuredDataError(
                f'the readings carry no column(s) {missing} to draw',
                code='INVALID_REQUEST',
            )
        return list(feature_columns)

    excluded = {timestamp_column, ground_truth_column, _SCORE, _DETECTED}
    resolved = [
        name
        for name in readings.select_dtypes(include=[np.number]).columns
        if name not in excluded
    ]
    if not resolved:
        raise StructuredDataError(
            'these readings carry no numerical column to draw',
            code='INVALID_REQUEST',
        )
    return resolved
