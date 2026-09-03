# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from decimal import Decimal
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from sdm_client.errors import StructuredDataError

JSON_SAFE_INT_MAX = 9007199254740991
JSON_SAFE_INT_MIN = -9007199254740991

# The range the wire's `int64` represents. Beyond it a value cannot be sent
# under that dtype whatever its Python type, so the declaration and the cells
# would disagree.
INT64_MAX = 2**63 - 1
INT64_MIN = -(2**63)

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
    'boolean': 'bool',
    'date': 'timestamp[us]',
    'datetime': 'timestamp[us]',
    'datetime64': 'timestamp[us]',
}


def _holds_only_scalars(dtype: Any) -> bool:
    r"""Whether a column of ``dtype`` cannot contain a sequence.

    Asked of the dtype rather than by elimination: ``is_object_dtype`` alone
    would call an Arrow-backed list column scalar, and that is what
    ``pd.read_parquet(dtype_backend='pyarrow')`` gives for one.
    """
    arrow_type = getattr(dtype, 'pyarrow_dtype', None)
    if arrow_type is not None:
        return _arrow_holds_only_scalars(arrow_type)
    return bool(
        pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_datetime64_any_dtype(dtype)
        or pd.api.types.is_timedelta64_dtype(dtype)
        or isinstance(dtype, pd.StringDtype)
        or dtype == 'str'
    )


def _arrow_holds_only_scalars(arrow_type: Any) -> bool:
    r"""The same question of an Arrow type, through its wrappers.

    ``pa.types.is_nested`` answers for the containers -- list, its view and
    large variants, fixed-size list, struct, map, union -- but not for the
    types that *wrap* one. A dictionary or a run-end encoding of a list, and an
    extension type whose storage is a list, are all still columns of lists, so
    each is unwrapped and the question asked again. Enumerating the nested
    types instead was how an earlier attempt let a list view through.

    ``BaseExtensionType`` rather than ``ExtensionType``: pyarrow's own
    canonical extensions -- ``fixed_shape_tensor``, whose storage is a
    fixed-size list -- are the former and not the latter.
    """
    import pyarrow as pa

    while True:
        if isinstance(arrow_type, pa.BaseExtensionType):
            arrow_type = arrow_type.storage_type
            continue
        if pa.types.is_dictionary(arrow_type) or pa.types.is_run_end_encoded(
            arrow_type
        ):
            arrow_type = arrow_type.value_type
            continue
        # Binary is not nested, but pandas hands its cells back as `bytes`,
        # which `_is_container` refuses -- so the two questions have to agree
        # or the fast path would skip a column the cell rule would reject.
        if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(
            arrow_type
        ):
            return False
        if pa.types.is_fixed_size_binary(arrow_type):
            return False
        return not pa.types.is_nested(arrow_type)


def require_scalar_cells(series: pd.Series, described: str) -> None:
    r"""Refuse a column whose cells are lists or mappings.

    The contract carries a `stringlist` dtype, and says what it is for: array
    cells exist "so that the stringlist dtype can carry JSON arrays **without
    permitting array-valued prediction scalars**" -- temporal link prediction
    labels, which no model reachable from here serves.

    Nothing in this client sends one. `pandas.api.types.infer_dtype` reports a
    list column as `mixed`, which falls through to `string`, and `serialize_cell`
    ends in `str(value)` -- so a column of lists used to travel as the *repr* of
    a Python list, `"['a', 'b']"`, as an ordinary category. That is silent, and
    the model is asked about text that was never in the data.

    Raised here rather than only where a schema is built, because a value can
    reach the wire without one: the rows being scored are not part of the
    schema, and a caller may supply a schema of their own.
    """
    # Skipped only for a dtype that cannot hold a sequence, which is most of
    # them on the hot path for every table in every request. Asked the other way
    # round on purpose: `is_object_dtype` alone would let an Arrow-backed list
    # column through, and that is what `pd.read_parquet(dtype_backend='pyarrow')`
    # gives for one.
    if _holds_only_scalars(series.dtype):
        return
    for value in series:
        if is_container(value):
            shape = 'a mapping' if isinstance(value, Mapping) else 'a container'
            raise StructuredDataError(
                f'{described} holds {shape} ({type(value).__name__}), and a '
                f'nested column is not something this client can send. '
                f'Multi-label targets are not supported by these models; for a '
                f'feature, give one value per row',
                code='UNSUPPORTED_REQUEST',
            )


def is_container(value: Any) -> bool:
    r"""Whether ``value`` holds other values rather than being one.

    Asked as a property, not as a list of types. Enumerating them is how a
    tuple, then a mapping, then a `range`, then a nested Series each turned out
    to be missing -- and every one of them would have been written by
    :func:`serialize_cell`'s `str()` fallback as its *repr*: `"[1, 2]"`,
    `"{'a': 1}"`, `"range(0, 2)"`.

    A `str` is iterable and is exactly what a text column holds, so it is the
    one exception. Everything a scalar column legitimately carries -- a number,
    a `Decimal`, a `UUID`, a timestamp -- is not iterable at all.
    """
    if isinstance(value, str):
        return False
    return isinstance(value, (Iterable, Mapping, bytes, memoryview))


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
        if content == 'decimal':
            return _decimal_dtype(series)
        return _OBJECT_CONTENT_DTYPES.get(content, 'string')
    return 'string'


def _decimal_dtype(series: pd.Series) -> str:
    r"""``int64`` for a whole-number ``Decimal`` column, ``float64`` otherwise.

    ``DECIMAL`` and ``NUMBER`` are how a warehouse spells an exact number, and
    a driver hands them back as ``Decimal`` objects. Calling the column
    ``float64`` because that is what floats are made of loses every value past
    2**53 -- which is most warehouse ids -- so a column whose values are whole
    is reported as the integer it is, and travels under the rule that keeps
    those exact.

    A whole number too wide even for ``int64`` travels as ``string``, which
    keeps the digits exactly and is what the relational encoder does with the
    same input. Anything else is a ``float64``, and :func:`serialize_cell`
    refuses the values that cannot survive one.
    """
    integral = True
    for value in series:
        if _is_missing(value):
            continue
        if not isinstance(value, Decimal):
            return 'float64'
        if not value.is_finite() or value != value.to_integral_value():
            return 'float64'
        if not INT64_MIN <= int(value) <= INT64_MAX:
            integral = False
    return 'int64' if integral else 'string'


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


def json_ready(value: Any) -> Any:
    r"""``value`` with numpy scalars replaced by their Python equivalents.

    Anything a caller reads out of their own frame is a numpy scalar, and the
    JSON encoder has no rule for one -- so ``inference={'seed': df['seed'][0]}``
    used to die inside ``requests`` as a ``TypeError`` naming neither the key
    nor this client. Applied to the small hand-written parts of a payload;
    table cells go through :func:`serialize_cell`, which types them properly.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


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
    r"""``inf`` has no JSON literal; emitting it produces invalid JSON.

    Asked of the value rather than of its type: a ``Decimal('Infinity')`` from
    a warehouse is as unsendable as a ``float('inf')``, and testing only for
    ``float`` let it through to fail later inside the JSON encoder, as a
    transport error that named neither the column nor the value.
    """
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise StructuredDataError(
                f'value {value!r} is not JSON-representable; replace '
                'non-finite numbers with a finite value or a null',
                code='INVALID_REQUEST',
            )
        return
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise StructuredDataError(
            f'value {value!r} is not JSON-representable; replace non-finite '
            'numbers with a finite value or a null',
            code='INVALID_REQUEST',
        )


def _integral_value(value: Any) -> int | None:
    r"""``value`` as an ``int`` when it is a whole number, else ``None``.

    Asked of the value, not of its type. ``numbers.Integral`` covers ``int``
    and ``np.integer`` but not ``Decimal``, and a numpy integer compares equal
    to the ``float`` it rounds to -- so a type test and an ``!=`` each let a
    different unrepresentable value through the guard below.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Decimal):
        if value.is_finite() and value == value.to_integral_value():
            return int(value)
    return None


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
        if not isinstance(value, Integral) and integer_value != value:
            raise StructuredDataError(
                f'value {value!r} cannot be sent as {dtype} without losing its '
                'fractional part',
                code='INVALID_REQUEST',
            )
        if not INT64_MIN <= integer_value <= INT64_MAX:
            raise StructuredDataError(
                f'value {value!r} is outside the range {dtype} represents '
                f'({INT64_MIN} to {INT64_MAX}); send the column as strings',
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
        if not math.isfinite(number):
            # A finite value whose float conversion is not: too large for
            # the dtype to hold at all, rather than merely rounded by it.
            raise StructuredDataError(
                f'value {value!r} is outside the range {dtype} represents; '
                'send the column as strings to keep it exact',
                code='INVALID_REQUEST',
            )
        whole = _integral_value(value)
        if (
            whole is None
            and isinstance(value, Decimal)
            and abs(number) > (JSON_SAFE_INT_MAX)
        ):
            # Past 2**53 a float64 cannot even hold the integer part, so the
            # fractional value it becomes is not a rounding of the original in
            # any useful sense. Below that the loss is ordinary float
            # behaviour, which any float column already has.
            raise StructuredDataError(
                f'value {value!r} is too large for {dtype} to represent; '
                'send the column as strings to keep it exact',
                code='INVALID_REQUEST',
            )
        if whole is not None and int(number) != whole:
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
    # Every table reaches the wire through here, including the rows being
    # scored, which no schema describes. A list cell would otherwise be written
    # by `serialize_cell`'s `str(value)` fallback as the repr of a Python list.
    require_scalar_cells(series, f'column {series.name!r}')
    return [serialize_cell(value, dtype) for value in series.tolist()]
