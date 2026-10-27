# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native observations for the core's closed object-Decimal constructor plan.

The native source owns method invocation and its exceptions.
The core owns which operation is reached, which row follows, and the meaning
of its results, including short-circuiting before a later native override.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _observation(kind: str, value: Any) -> bool:
    """Perform only the selected native operation, including truth coercion."""
    if kind == 'missing':
        return _missing(value)
    if kind == 'is_decimal':
        return isinstance(value, Decimal)
    if kind == 'finite':
        return bool(value.is_finite())
    if kind == 'negative_exponent':
        return bool(value.as_tuple().exponent < 0)
    if kind == 'fractional':
        return bool(value != value.to_integral_value())
    if kind == 'wide':
        return bool(value.adjusted() >= 18)
    if kind == 'at_least_minimum':
        return bool(-(2**63) <= value)
    if kind == 'at_most_maximum':
        return bool(value <= 2**63 - 1)
    raise AssertionError('an unknown native Decimal observation was selected')


def meaning(series: pd.Series) -> dict[str, Any] | None:
    """Consume actual native inference reads in the core-selected order.

    Retain one native iterator just as the original inference does.
    Re-resolving a row with ``iloc`` after an earlier method mutates the
    source could observe different storage from the original iterator.
    """
    from sdm_client import _native

    inference = _native.ObjectDecimalConstructor(len(series))
    content = pd.api.types.infer_dtype(series, skipna=True)
    inference.content(content)
    if not inference.is_decimal():
        return None
    current_row = -1
    current = None
    iterator = iter(series)
    while True:
        kind, row = inference.next_read()
        if kind == 'complete':
            dtype, untyped = inference.meaning()
            return {'dtype': dtype, 'untyped': untyped}
        if row != current_row:
            current = next(iterator)
            current_row = row
        inference.observe(_observation(kind, current))


def storage(series: pd.Series) -> pa.StructArray:
    """Export underlying native Decimal tuples without replaying overrides.

    This is typed source data, independent of inferred meaning.
    Calling the built-in descriptor reads the underlying Decimal storage;
    user-defined ``as_tuple`` is invoked only by a reached inference read.
    """
    negative = []
    digits = []
    exponent = []
    special = []
    missing = []
    for value in series:
        if isinstance(value, Decimal):
            parts = Decimal.as_tuple(value)
            code = (
                0
                if isinstance(parts.exponent, int)
                else {'F': 1, 'n': 2, 'N': 3}[parts.exponent]
            )
            negative.append(bool(parts.sign))
            digits.append(''.join(str(digit) for digit in parts.digits))
            exponent.append(parts.exponent if code == 0 else 0)
            special.append(code)
            missing.append(False)
        else:
            # Native content classification already established this closed
            # Decimal/empty family. Its absent cells carry no Decimal tuple.
            negative.append(False)
            digits.append('')
            exponent.append(0)
            special.append(0)
            missing.append(True)
    return pa.StructArray.from_arrays(
        [
            pa.array(negative, type=pa.bool_()),
            pa.array(digits, type=pa.string()),
            pa.array(exponent, type=pa.int64()),
            pa.array(special, type=pa.uint8()),
        ],
        fields=[
            pa.field('negative', pa.bool_(), nullable=False),
            pa.field('digits', pa.string(), nullable=False),
            pa.field('exponent', pa.int64(), nullable=False),
            pa.field('special', pa.uint8(), nullable=False),
        ],
        mask=pa.array(missing, type=pa.bool_()),
    )
