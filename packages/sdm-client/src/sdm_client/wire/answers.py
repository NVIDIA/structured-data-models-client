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

    counted one of five on one model and none of five on the other -- no
    exception, a plausible-looking answer, and wrong in the direction of "this
    class never occurs". The target's dtype is on the wire, so it is read back
    with it.

    Class labels only. They are values the column held, so giving them back as
    the column's own type is restoring something. A regression's answer is a
    number the model computed, and rounding that to the target's dtype would be
    this client deciding how precise the answer is.
    """

    task = (payload or {}).get('task', {})
    if 'classification' not in str(task.get('kind', '')):
        return items
    dtype = task.get('target', {}).get('dtype')
    if not dtype or not (dtype.startswith('int') or dtype.startswith('float')):
        return items
    read = []
    for item in items:
        value = getattr(item, 'prediction', None)
        restored = as_declared(value, dtype)
        read.append(
            item
            if restored is value
            else dataclasses.replace(item, prediction=restored)
        )
    return read
