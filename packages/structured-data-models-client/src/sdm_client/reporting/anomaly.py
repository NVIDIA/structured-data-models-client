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

import numbers
from collections.abc import Hashable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from sdm_client.errors import MissingExtraError, StructuredDataError
from sdm_client.models import TFM_MODEL_KUMO_ANOMALY

if TYPE_CHECKING:  # pragma: no cover - imported for typing only.
    from sdm_client.result import PredictionResult

#: The thresholders an anomaly answer carries, under the names
#: :mod:`sdm_client.in_process.anomaly` puts them back together as.
THRESHOLDERS = ('scs', 'macs')

_SCORE = 'prediction'
_DETECTED = 'detected'


def _free(name: str, taken: Any) -> str:
    r"""``name``, or a spelling of it the readings have not already used.

    The score and the label are put onto the frame to draw it. A reading
    already called one of those would be replaced by it, and the report would
    then plot the score as though it were that reading, or read the
    detections as the ground truth they were supposed to be judged against.
    """
    while name in taken:
        name = f'_{name}'
    return name


def anomaly_report(
    result: PredictionResult,
    readings: pd.DataFrame,
    output_path: str | Path,
    *,
    detected_by: str | Sequence[Any] | np.ndarray = 'scs',
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
        detected_by: What marks a row as detected. Either the name of a
            thresholder the answer carries, ``'scs'`` or ``'macs'`` -- they
            disagree, and a report has to mark one of them, so it says which
            -- or the detections themselves, one per row of ``readings`` and
            in its order, for a caller who drew the line somewhere else.
            :func:`~sdm_client.percentile_thresholds` draws one such line.
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
    if isinstance(detected_by, str) and detected_by not in THRESHOLDERS:
        raise StructuredDataError(
            f'detected_by is {detected_by!r}; it names one of '
            f'{list(THRESHOLDERS)}, whose labels disagree, so a report marks '
            f'one of them and says which, or it is the detections themselves',
            code='INVALID_REQUEST',
        )
    pages = _pages()
    frame, score, detected = _rejoined(result, readings, detected_by)
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
        anomaly_column=detected,
        score_column=score,
        title=_titled(title, detected_by),
        detections_supplied=not isinstance(detected_by, str),
        max_report_pages=max_report_pages,
    )


def _pages() -> Any:
    try:
        from sdm_client.reporting import _pages as drawn
    except ImportError as error:  # pragma: no cover - the extra is absent
        raise MissingExtraError('report', 'matplotlib') from error
    return drawn


def _rejoined(
    result: PredictionResult,
    readings: pd.DataFrame,
    detected_by: str | Sequence[Any] | np.ndarray,
) -> tuple[pd.DataFrame, str, str]:
    r"""``readings`` with each row's score and label put back on it, and
    what those two columns ended up being called.

    The readings keep their own index: a DatetimeIndex is an axis the pages
    plot against, and resetting it drew dates a month apart as evenly spaced
    steps. ``row_index`` is a position in the frame that was sent rather
    than a label of this index, so the values are assigned as arrays, which
    is positional and ignores the index entirely.

    A label is not converted to a whole number here. 0.9 is not a label, and
    truncating it would turn it into one before anything checked it.
    """
    if result.model != TFM_MODEL_KUMO_ANOMALY:
        raise StructuredDataError(
            f'this answer came from {result.model!r} and every page here '
            f'reads it as {TFM_MODEL_KUMO_ANOMALY!r}: the score it draws is '
            f'a reconstruction error and it says so on the page. Another '
            f"model's answer would be drawn under that label and read as "
            f'something it is not',
            code='INVALID_REQUEST',
        )
    answer = result.to_pandas()
    named = isinstance(detected_by, str)
    label = f'probability_{detected_by}' if named else None
    wanted = [_SCORE] + ([label] if label is not None else [])
    if any(name not in answer.columns for name in wanted):
        raise StructuredDataError(
            f'this answer carries {sorted(answer.columns)}, and a report '
            f'needs {wanted}. Ask for them with '
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

    drawn = readings.copy()
    ordered = answer.sort_values('row_index', kind='stable')
    score = _free(_SCORE, drawn.columns)
    detected = _free(_DETECTED, drawn.columns)
    drawn[score] = ordered[_SCORE].to_numpy()
    drawn[detected] = (
        ordered[label].to_numpy()
        if label is not None
        else _supplied(detected_by, len(readings))
    )
    return drawn, score, detected


def _supplied(detections: Sequence[Any] | np.ndarray, rows: int) -> np.ndarray:
    r"""The caller's own detections, checked against the rows they are about.

    Taken as an array rather than as a labelled series: they are one per row
    of the readings in the order the readings are in, which is the same thing
    the scores are matched by, and aligning them on an index instead would
    match a detection to whichever row happened to share a label.

    A score is refused rather than read as a detection. Drawn as one it
    marks every row whose score rounds up, which is a different report from
    the one the caller asked for and looks exactly like the one they asked
    for. What the values are is read before they are converted, for the same
    reason: converting a complex number to a float keeps the real part and
    drops the rest without a word, so ``2j`` arrived as a detection. An
    array of objects is read element by element, because its own dtype says
    only that the elements were not all one thing.
    """
    values = np.asarray(
        detections.to_numpy()
        if isinstance(detections, (pd.Series, pd.Index))
        else detections
    ).reshape(-1)
    if values.size != rows:
        raise StructuredDataError(
            f'detected_by carries {values.size} detection(s) and these '
            f'readings are {rows} row(s); it marks one row each',
            code='INVALID_REQUEST',
        )
    if values.dtype == bool:
        return values
    if values.dtype.kind not in 'biufO' or (
        values.dtype.kind == 'O'
        and not all(_is_a_real_number(value) for value in values)
    ):
        raise StructuredDataError(
            f'detected_by carries {values.dtype} values, which are not a '
            f'decision about a row. Pass true and false, one per row',
            code='INVALID_REQUEST',
        )
    try:
        numeric = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise StructuredDataError(
            'detected_by carries values that are not true or false, so '
            'nothing in it says which rows were detected',
            code='INVALID_REQUEST',
        ) from error
    if not bool(np.isin(numeric, (0.0, 1.0)).all()):
        raise StructuredDataError(
            'detected_by carries values other than true and false, so it is '
            'a score rather than a decision about one. Compare it against a '
            'threshold first; percentile_thresholds finds one',
            code='INVALID_REQUEST',
        )
    return numeric.astype(bool)


def _is_a_real_number(value: Any) -> bool:
    r"""Whether ``value`` is a number that reads as a float unchanged.

    Asked of the numeric tower where it answers, and of the types by name
    where it does not: ``numpy.bool_`` and :class:`~decimal.Decimal` are
    neither :class:`numbers.Real` nor :class:`numbers.Complex`, and both are
    ordinary spellings of a detection.
    """
    if isinstance(value, (bool, np.bool_, Decimal)):
        return True
    return isinstance(value, numbers.Real)


def _titled(title: str, detected_by: str | Sequence[Any] | np.ndarray) -> str:
    r"""The title, saying what decided the detections it draws."""
    if isinstance(detected_by, str):
        return f'{title} (detections by {detected_by.upper()})'
    return f'{title} (detections supplied by the caller)'


def _signals(
    readings: pd.DataFrame,
    feature_columns: Sequence[Hashable] | None,
    *,
    timestamp_column: str | None,
    ground_truth_column: str | None,
) -> list[Hashable]:
    r"""The columns to draw as signals.

    Read off the readings rather than the frame that is drawn: that frame
    carries the score and the label as well, and a default worked out from it
    would plot those as though they were readings.
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

    excluded = {timestamp_column, ground_truth_column}
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
