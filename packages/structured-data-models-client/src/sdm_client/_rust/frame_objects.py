# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native observations for the core's closed object-Decimal constructor plan.

The native source owns method invocation and its exceptions.
The core owns which operation is reached, which row follows, and the meaning
of its results, including short-circuiting before a later native override.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from sdm_client._rust import errors


def _scalar_storage(dtype: Any) -> bool:
    """Whether the actual native storage can contain only scalar cells."""
    physical = getattr(dtype, 'pyarrow_dtype', None)
    if physical is not None:
        while True:
            if isinstance(physical, pa.BaseExtensionType):
                physical = physical.storage_type
            elif pa.types.is_dictionary(
                physical
            ) or pa.types.is_run_end_encoded(physical):
                physical = physical.value_type
            else:
                break
        return any(
            getattr(pa.types, name)(physical)
            for name in (
                'is_integer',
                'is_floating',
                'is_decimal',
                'is_boolean',
                'is_string',
                'is_large_string',
                'is_string_view',
                'is_date',
                'is_time',
                'is_timestamp',
                'is_duration',
                'is_null',
            )
            if hasattr(pa.types, name)
        )
    return bool(
        pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_datetime64_any_dtype(dtype)
        or pd.api.types.is_timedelta64_dtype(dtype)
        or isinstance(dtype, pd.StringDtype)
        or dtype == 'str'
    )


def validate_cells(frame: pd.DataFrame, table: str) -> None:
    """Present native storage/cell categories to the fixed core scalar rule."""
    from sdm_client import _native

    for name in frame.columns:
        series = frame[name]
        if _scalar_storage(series.dtype):
            continue
        for value in series:
            if isinstance(value, str) or not isinstance(
                value, (Iterable, Mapping, bytes, memoryview)
            ):
                kind = 'scalar'
            elif isinstance(value, (bytes, bytearray, memoryview)):
                kind = 'bytes'
            elif isinstance(value, Mapping):
                kind = 'mapping'
            else:
                kind = 'container'
            try:
                _native.check_frame_cell(
                    f'{table}.{name}',
                    kind,
                    type(value).__name__ if kind != 'scalar' else '',
                )
            except _native.CoreError as error:
                errors.raise_from_core(
                    error.control, error.attachments, internal_context=error
                )


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
    if kind == 'within_integer_range':
        try:
            return -(2**63) <= int(value) <= 2**63 - 1
        except (TypeError, ValueError, OverflowError):
            return False
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


def meaning(series: pd.Series) -> dict[str, Any]:
    """Consume actual native inference reads in the core-selected order.

    Retain one native iterator just as the original inference does.
    Re-resolving a row with ``iloc`` after an earlier method mutates the
    source could observe different storage from the original iterator.
    """
    from sdm_client import _native

    inference = _native.FrameObjectConstructor(len(series))
    content = (
        'categorical'
        if type(series.dtype) is pd.CategoricalDtype
        else pd.api.types.infer_dtype(series, skipna=True)
    )
    inference.content(content)
    current_row = -1
    current = None
    iterator = iter(series)
    while True:
        kind, row = inference.next_read()
        if kind == 'complete':
            return json.loads(inference.meaning())
        if kind == 'distinct':
            inference.distinct(int(series.nunique(dropna=True)))
            continue
        if row != current_row:
            current = next(iterator)
            current_row = row
        inference.observe(_observation(kind, current))


def storage(series: pd.Series) -> pa.StructArray:
    """Export neutral typed scalar facts without invoking user conversions.

    Native source positions remain in the caller-owned column.
    An unconverted custom scalar is explicitly present and unconverted;
    its actual selected conversion happens later in materialization.
    """
    import math
    import struct

    names = (
        'kind',
        'negative',
        'digits',
        'exponent',
        'special',
        'boolean',
        'integer',
        'float_bits',
        'text',
    )
    columns: dict[str, list[Any]] = {name: [] for name in names}
    native_integers = tuple(
        getattr(np, name)
        for name in (
            'int8',
            'int16',
            'int32',
            'int64',
            'uint8',
            'uint16',
            'uint32',
            'uint64',
        )
    )
    native_floats = (np.float16, np.float32, np.float64)
    for value in series:
        record: dict[str, Any] = dict.fromkeys(names)
        record['kind'] = 0
        if value is None or value is pd.NA or value is pd.NaT:
            record['kind'] = 1
        elif isinstance(value, Decimal):
            parts = Decimal.as_tuple(value)
            code = (
                0
                if isinstance(parts.exponent, int)
                else {'F': 1, 'n': 2, 'N': 3}[parts.exponent]
            )
            record.update(
                kind=2,
                negative=bool(parts.sign),
                digits=''.join(str(digit) for digit in parts.digits),
                exponent=parts.exponent if code == 0 else 0,
                special=code,
            )
        elif type(value) in (bool, np.bool_):
            record.update(kind=3, boolean=bool(value))
        elif isinstance(value, int):
            record.update(kind=4, integer=int.__repr__(value))
        elif type(value) in native_integers:
            record.update(kind=4, integer=repr(value.item()))
        elif isinstance(value, float) or type(value) in native_floats:
            number = (
                float.__float__(value)
                if isinstance(value, float)
                else value.item()
            )
            if math.isnan(number):
                record['kind'] = 1
            else:
                record.update(
                    kind=5,
                    float_bits=struct.unpack('<Q', struct.pack('<d', number))[
                        0
                    ],
                )
        elif isinstance(value, str):
            record.update(
                kind=6, text=str.encode(value, 'utf-8', 'surrogatepass')
            )
        for name in names:
            columns[name].append(record[name])
    types = (
        pa.uint8(),
        pa.bool_(),
        pa.string(),
        pa.int64(),
        pa.uint8(),
        pa.bool_(),
        pa.string(),
        pa.uint64(),
        pa.binary(),
    )
    return pa.StructArray.from_arrays(
        [
            pa.array(columns[name], type=dtype)
            for name, dtype in zip(names, types, strict=True)
        ],
        fields=[
            pa.field(name, dtype, nullable=name != 'kind')
            for name, dtype in zip(names, types, strict=True)
        ],
    )
