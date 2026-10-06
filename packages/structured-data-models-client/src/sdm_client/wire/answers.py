# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Giving an answer back in the type the question was asked in.

Both models answer with a class label the column already held, and both
spell it on the wire as text. Restoring the column's own type is therefore
about the response rather than about either model, and lives here rather
than with one of them.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from sdm_client.wire.dtypes import as_declared


def as_declared_target(items: list[Any], payload: dict[str, Any] | None):
    r"""Give `prediction` back as the type the target column holds.

    One endpoint renders a class label as a string and the other sends the
    number, for the same question on the same integer-coded target. Both are
    legal and the values agree; only their types differ, so

        sum(1 for row in result if row.prediction == 0)

    This counted one of five on one model and none of five on the other. There
    was no exception, only a plausible-looking answer wrong in the direction
    of "this class never occurs". The target's dtype is on the wire, so it is
    read back with it.

    Class labels only, and a `> 0` target's labels are a bool, which one
    endpoint spells `'false'`. They are values the column held, so giving them
    back as the column's own type is restoring something. A regression's answer
    is a number the model computed. Rounding it to the target's dtype would
    make this client decide how precise the answer is.

    The same labels key `probabilities` and are restored there too.
    """

    task = (payload or {}).get('task', {})
    if 'classification' not in str(task.get('kind', '')):
        return items
    dtype = task.get('target', {}).get('dtype')
    if not dtype or not dtype.startswith(('int', 'float', 'bool')):
        return items
    read = []
    for item in items:
        changed: dict[str, Any] = {}
        value = getattr(item, 'prediction', None)
        restored = as_declared(value, dtype)
        if restored is not value:
            changed['prediction'] = restored
        given = getattr(item, 'probabilities', None)
        if given:
            keyed = _as_the_column_spells_them(given, dtype)
            if keyed != given:
                changed['probabilities'] = keyed
        read.append(dataclasses.replace(item, **changed) if changed else item)
    return read


def _as_the_column_spells_them(
    probabilities: dict[str, Any], dtype: str
) -> dict[str, Any]:
    r"""The probability mapping, keyed by each class as the column spells it.

    One endpoint keys a bool target `'false'` and the other `'False'`, so
    `frame['probability_true']` raises depending on which answered.

    Left alone where restoring would collapse two classes into one.
    """
    renamed = {}
    for key, value in probabilities.items():
        restored = as_declared(key, dtype)
        renamed[key if restored is key else str(restored)] = value
    return renamed if len(renamed) == len(probabilities) else probabilities
