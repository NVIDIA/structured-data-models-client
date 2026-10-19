# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What a model returned, written as the answers a NIM sends back."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from sdm_client.in_process.refusal import refusal
from sdm_client.in_process.tables import values

#: The levels a NIM reports when the request names none.
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

#: The model predicts quantiles q001 through q999; q500 is the median.
_QUANTILES = 999
_MEDIAN = 499


def row_ids(
    table: Mapping[str, Any], primary_key: str | Sequence[str] | None
) -> list[str | None]:
    r"""The id a NIM answers each row under.

    A key named as one column is the value itself; a key named as a list is
    the list of its values, written as JSON. Either is the value as it was
    sent, so an integer sent as its digits stays a string.
    """
    columns = table['columns']
    if isinstance(primary_key, str) and primary_key in columns:
        return [str(value) for value in values(table, primary_key)]
    if (
        isinstance(primary_key, list)
        and primary_key
        and all(name in columns for name in primary_key)
    ):
        cells = [values(table, name) for name in primary_key]
        return [
            json.dumps(list(key), separators=(',', ':'), default=str)
            for key in zip(*cells, strict=True)
        ]
    return [None] * len(table['rows'])


def classified(
    output: Any,
    encoded: Sequence[Any],
    labels: Sequence[Any],
    ids: Sequence[str | None],
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    r"""One answer per row from a classification's class probabilities.

    The model names each probability column after a class as it encoded it;
    ``labels`` is what the caller calls each of ``encoded``, in order.
    """
    named = {name: index for index, name in enumerate(_columns(output))}
    try:
        order = [named[str(label)] for label in encoded]
    except KeyError as error:
        raise refusal(
            'INTERNAL_ERROR',
            f'the model answered for classes {sorted(named)}, not the '
            f'{len(encoded)} the context holds',
        ) from error
    probabilities = _numbers(output)[:, order]
    answers = []
    for index, row in enumerate(probabilities):
        answer: dict[str, Any] = {
            'row_index': index,
            'id': ids[index],
            'prediction': labels[int(np.argmax(row))],
        }
        if 'probabilities' in fields:
            answer['probabilities'] = {
                str(label): float(row[position])
                for position, label in enumerate(labels)
            }
        answers.append(answer)
    return answers


def regressed(
    output: Any,
    ids: Sequence[str | None],
    fields: Sequence[str],
    statistic: str,
    levels: Sequence[float],
) -> list[dict[str, Any]]:
    r"""One answer per row from a regression's predicted quantiles.

    The point prediction is the distribution's mean or median, and each
    quantile asked for is interpolated between the two predicted around it.
    """
    if 'quantiles' in fields and not all(0.001 <= x <= 0.999 for x in levels):
        raise refusal(
            'UNSUPPORTED_OUTPUT',
            'quantile levels must be between 0.001 and 0.999',
        )
    quantiles = _numbers(output)
    if quantiles.shape[1] != _QUANTILES:
        raise refusal(
            'INTERNAL_ERROR',
            f'the model answered with {quantiles.shape[1]} quantiles, not '
            f'{_QUANTILES}',
        )
    points = (
        quantiles.mean(axis=1) if statistic == 'mean' else quantiles[:, _MEDIAN]
    )
    answers = []
    for index, row in enumerate(quantiles):
        answer: dict[str, Any] = {
            'row_index': index,
            'id': ids[index],
            'prediction': float(points[index]),
        }
        if 'quantiles' in fields:
            answer['quantiles'] = {
                format(level, '.15g'): _at(row, level) for level in levels
            }
        answers.append(answer)
    return answers


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
        raise refusal(
            'INTERNAL_ERROR', 'the model answered with a malformed output'
        )
    return array
