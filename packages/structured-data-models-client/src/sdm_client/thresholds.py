# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Turning anomaly scores into a decision about which rows are anomalous.

The model answers with a score per row and with two thresholders' labels
beside it. Both of those read the whole series and decide for themselves
where the line is, which is what makes them useful and also what makes them
impossible to move: a caller who wants a stricter or a looser line has no way
to ask for one.

This is that way. It works on the scores the answer already carries, so it
costs no inference and can be re-run as often as a reader wants to argue with
where the line fell.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sdm_client.errors import StructuredDataError


def percentile_thresholds(
    scores: Sequence[float] | np.ndarray,
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.0,
) -> tuple[float, float]:
    r"""Where a run of anomaly scores stops being ordinary, by percentile.

    The upper threshold is the upper percentile of the scores, widened by how
    scattered they are: a run whose scores vary a great deal has to clear a
    higher bar than a run whose scores sit together, because in a scattered
    run the same percentile is reached by ordinary rows.

    Args:
        scores: What the model answered, one score per row. For a
            ``kumo-anomaly`` answer that is its ``prediction`` column.
        lower_percentile: The percentile below which a score is quiet.
        upper_percentile: The percentile the upper threshold is widened from.

    Returns:
        The lower and the upper threshold. A score above the upper one is the
        detection; the lower one is the floor an ordinary score sits above.

    Raises:
        StructuredDataError: If the percentiles are not between 0 and 100 and
            in that order, if the scores are empty, not all finite or not all
            positive, or if a threshold worked out from them is not finite. A
            threshold that is NaN or infinite compares false against every
            score and silently detects nothing at all, which is the one
            failure this must not have; a single NaN score carries into every
            percentile and into the mean, and scores near the largest float
            overflow when the mean of them is taken.

            Scores are the distance between what was observed and what the
            model reconstructed, so a negative one is not a score this knows
            how to read. The widening below is a multiplication, and
            multiplying a negative threshold by more than one lowers the bar
            rather than raising it, so a scattered run of them would be held
            to a *looser* line than a settled one.

    >>> result = client.predict(Request.anomaly(readings), model=KumoAnomaly())
    >>> scores = result.to_pandas()['prediction']
    >>> _, upper = percentile_thresholds(scores, upper_percentile=99.5)
    >>> anomaly_report(result, readings, 'readings.pdf',
    ...                detected_by=scores > upper)
    """
    for name, value in (
        ('lower_percentile', lower_percentile),
        ('upper_percentile', upper_percentile),
    ):
        if not 0.0 <= float(value) <= 100.0:
            raise StructuredDataError(
                f'{name} is {value!r}; a percentile is between 0 and 100',
                code='INVALID_REQUEST',
            )
    if float(lower_percentile) > float(upper_percentile):
        raise StructuredDataError(
            f'lower_percentile is {lower_percentile!r} and upper_percentile '
            f'is {upper_percentile!r}, so the quiet line sits above the '
            f'loud one and every score would be both',
            code='INVALID_REQUEST',
        )

    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 0:
        raise StructuredDataError(
            'there are no scores to find a threshold in',
            code='INVALID_REQUEST',
        )
    if not bool(np.isfinite(values).all()):
        raise StructuredDataError(
            'these scores are not all finite, and a threshold worked out '
            'from them detects nothing at all',
            code='INVALID_REQUEST',
        )
    if bool((values < 0.0).any()):
        raise StructuredDataError(
            'these scores include a negative one, and a score is how far '
            'the reading sat from what was reconstructed, which has no '
            'direction. Pass the distances themselves',
            code='INVALID_REQUEST',
        )

    lower = float(np.percentile(values, float(lower_percentile)))
    upper = float(np.percentile(values, float(upper_percentile)))
    widened = upper * (1.0 + 0.5 * min(_spread(values), 1.0))
    if not np.isfinite(widened):
        raise StructuredDataError(
            f'the upper threshold these scores give is {widened}, which no '
            f'score is above, so the report would detect nothing at all. '
            f'Scale them down before asking where the line falls',
            code='INVALID_REQUEST',
        )
    return lower, widened


def _spread(values: np.ndarray) -> float:
    r"""How scattered ``values`` are, as a share of how large they are.

    The coefficient of variation, worked out on the values divided by the
    largest of them. That ratio is what the coefficient of variation is, so
    dividing first changes nothing about the answer, and it keeps every sum
    taken on the way within the range of a float: scores near the largest
    float sum to infinity, and the spread of them came back NaN, which
    widened the threshold to NaN and detected nothing.
    """
    largest = float(values.max())
    if largest <= 0.0:
        return 1.0
    scaled = values / largest
    mean = float(scaled.mean())
    return float(scaled.std()) / mean if mean > 0 else 1.0
