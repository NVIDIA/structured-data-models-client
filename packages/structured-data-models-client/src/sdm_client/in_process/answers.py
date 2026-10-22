# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What a model returned, as one answer per scored row."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from sdm_api.generated.tfm_api import PredictionItem

from sdm_client.errors import StructuredDataError

#: The levels a NIM reports when the request names none.
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

#: The model predicts quantiles q001 through q999; q500 is the median.
_QUANTILES = 999
_MEDIAN = 499


def classified(
    output: Any,
    encoded: Sequence[Any],
    labels: Sequence[Any],
    ids: Sequence[str | None],
    fields: Sequence[str],
) -> list[PredictionItem]:
    r"""One answer per row from a classification's class probabilities.

    The model names each probability column after a class as it encoded it;
    ``labels`` is what the caller calls each of ``encoded``, in order.
    """
    named = {name: index for index, name in enumerate(_columns(output))}
    try:
        order = [named[str(label)] for label in encoded]
    except KeyError as error:
        raise StructuredDataError(
            f'the model answered for classes {sorted(named)}, not the '
            f'{len(encoded)} the context holds',
            code='INTERNAL_ERROR',
        ) from error
    probabilities = _numbers(output)[:, order]
    return [
        PredictionItem(
            row_index=index,
            id=ids[index],
            prediction=labels[int(np.argmax(row))],
            probabilities=(
                {
                    str(label): float(row[position])
                    for position, label in enumerate(labels)
                }
                if 'probabilities' in fields
                else None
            ),
        )
        for index, row in enumerate(probabilities)
    ]


def regressed(
    output: Any,
    ids: Sequence[str | None],
    fields: Sequence[str],
    statistic: str,
    levels: Sequence[float],
) -> list[PredictionItem]:
    r"""One answer per row from a regression's predicted quantiles.

    The point prediction is the distribution's mean or median, and each
    quantile asked for is interpolated between the two predicted around it.
    """
    if 'quantiles' in fields and not all(0.001 <= x <= 0.999 for x in levels):
        raise StructuredDataError(
            'quantile levels must be between 0.001 and 0.999',
            code='UNSUPPORTED_OUTPUT',
        )
    quantiles = _numbers(output)
    if quantiles.shape[1] != _QUANTILES:
        raise StructuredDataError(
            f'the model answered with {quantiles.shape[1]} quantiles, not '
            f'{_QUANTILES}',
            code='INTERNAL_ERROR',
        )
    points = (
        quantiles.mean(axis=1) if statistic == 'mean' else quantiles[:, _MEDIAN]
    )
    return [
        PredictionItem(
            row_index=index,
            id=ids[index],
            prediction=float(points[index]),
            quantiles=(
                {format(level, '.15g'): _at(row, level) for level in levels}
                if 'quantiles' in fields
                else None
            ),
        )
        for index, row in enumerate(quantiles)
    ]


def _at(row: Any, level: float) -> float:
    position = level * 1000.0 - 1.0
    lower, upper = math.floor(position), math.ceil(position)
    weight = position - lower
    return float(row[lower] * (1.0 - weight) + row[upper] * weight)


def _columns(output: Any) -> tuple[str, ...]:
    from sdm import Stype

    return tuple(output.columns[Stype.numerical])


def _numbers(output: Any) -> Any:
    array = np.asarray(output.numerical.detach().cpu().numpy())
    if array.ndim != 2 or not np.isfinite(array).all():
        raise StructuredDataError(
            'the model answered with a malformed output', code='INTERNAL_ERROR'
        )
    return array
