# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from sdm_client.errors import StructuredDataError

JSON_SAFE_INT_MAX = 9007199254740991
JSON_SAFE_INT_MIN = -9007199254740991

# Ordered from narrowest to widest; anything outside this map (timestamps,
# strings) has no numeric widening and falls back to 'string'.
_NUMERIC_WIDTH = {
    'bool': 0,
    'int32': 1,
    'int64': 2,
    'float32': 3,
    'float64': 4,
}

# `infer_tfm_dtype`'s answer for a column that carries no type information at
# all, so that it contributes nothing to a widening. Distinct from every real
# wire dtype, and never emitted into a payload.
UNTYPED = ''

# What `pandas.api.types.infer_dtype` reports for the contents of an `object`
# column, mapped to the wire dtype that represents those values. `object` is a
# container, not a type: classifying it by the container alone declares numbers
# as strings, which changes what the model is asked to predict on.
_OBJECT_CONTENT_DTYPES = {
    'integer': 'int64',
    'floating': 'float64',
    'mixed-integer-float': 'float64',
    'decimal': 'float64',
    'boolean': 'bool',
    'date': 'timestamp[us]',
    'datetime': 'timestamp[us]',
    'datetime64': 'timestamp[us]',
}


def infer_tfm_dtype(series: pd.Series) -> str:
    r"""The wire dtype that represents ``series``' values.

    An ``object`` column is classified by what it holds, not by the fact that
    it is ``object``: ``pd.read_sql`` over an all-null integer column, a frame
    built from records, and a column that has been through ``fillna`` are all
    ``object`` while holding perfectly ordinary numbers.

    An ``object`` column with no non-null values is reported as
    :data:`UNTYPED` rather than guessed at, so that a frame which happens to be
    missing a feature entirely cannot re-type the same column in another frame.

    Every integer width is reported as ``int64``, narrow ones included. The
    two model paths used to disagree here -- one reported the true width, the
    other always ``int64`` -- which meant identical data was described two
    ways. The server decodes both through ``int()`` and cannot tell them
    apart, so the choice is about which description to standardize on, and
    ``int64`` is the one that is never wrong: every narrower value fits it,
    while the reverse is not true of a column that is widened after its dtype
    is taken.

    Being consistent matters more than being narrow. The server requires a
    task's declared target dtype to equal the schema's dtype for that column
    exactly, and answers 422 when they differ, so a second rule anywhere is a
    rejected request rather than a cosmetic difference.
    """
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return 'bool'
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return 'timestamp[us]'
    if pd.api.types.is_integer_dtype(dtype):
        return 'int64'
    if pd.api.types.is_float_dtype(dtype):
        return 'float64' if dtype.itemsize > 4 else 'float32'
    if pd.api.types.is_object_dtype(dtype):
        content = pd.api.types.infer_dtype(series, skipna=True)
        if content == 'empty':
            return UNTYPED
        return _OBJECT_CONTENT_DTYPES.get(content, 'string')
    return 'string'


def widen_tfm_dtype(dtypes: Iterable[str]) -> str:
    r"""The narrowest wire dtype that can hold every one of ``dtypes``.

    A column typed differently in two frames must travel under a dtype that
    represents both, otherwise the values of one frame are coerced to the other
    frame's dtype and silently corrupted.

    :data:`UNTYPED` candidates are dropped: a column that carries no type
    information cannot make the union any wider, and letting it widen the union
    to ``'string'`` would re-type the other frame's real values.
    """
    candidates = [dtype for dtype in dict.fromkeys(dtypes) if dtype]
    if not candidates:
        return 'string'
    if len(candidates) == 1:
        return candidates[0]
    if any(dtype not in _NUMERIC_WIDTH for dtype in candidates):
        return 'string'
    widest = max(candidates, key=_NUMERIC_WIDTH.__getitem__)
    if widest == 'float32' and 'int64' in candidates:
        return 'float64'
    return widest


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False  # Array-likes (lists, ndarrays) are values, not nulls.


def _require_finite(value: Any) -> None:
    r"""``inf`` has no JSON literal; emitting it produces invalid JSON."""
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise StructuredDataError(
            f'value {value!r} is not JSON-representable; replace non-finite '
            'numbers with a finite value or a null',
            code='INVALID_REQUEST',
        )


def serialize_cell(value: Any, dtype: str) -> Any:
    if _is_missing(value):
        return None
    if dtype == 'timestamp[us]':
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize('UTC')
        else:
            timestamp = timestamp.tz_convert('UTC')
        return timestamp.isoformat(timespec='microseconds').replace(
            '+00:00', 'Z'
        )
    if dtype in ('int64', 'int32'):
        _require_finite(value)
        integer_value = int(value)
        if isinstance(value, (float, np.floating)) and integer_value != value:
            raise StructuredDataError(
                f'value {value!r} cannot be sent as {dtype} without losing its '
                'fractional part',
                code='INVALID_REQUEST',
            )
        if (
            integer_value > JSON_SAFE_INT_MAX
            or integer_value < JSON_SAFE_INT_MIN
        ):
            return str(integer_value)
        return integer_value
    if dtype in ('float64', 'float32'):
        _require_finite(value)
        number = float(value)
        if isinstance(value, Integral) and number != value:
            raise StructuredDataError(
                f'value {value!r} cannot be sent as {dtype} without losing '
                f'precision, because it is outside the range a {dtype} '
                'represents exactly; give the column the same integer dtype '
                'in every frame, or convert it to strings',
                code='INVALID_REQUEST',
            )
        return number
    if dtype == 'bool':
        return bool(value)
    return str(value)


def serialize_column(series: pd.Series, dtype: str) -> list[Any]:
    return [serialize_cell(value, dtype) for value in series.tolist()]
