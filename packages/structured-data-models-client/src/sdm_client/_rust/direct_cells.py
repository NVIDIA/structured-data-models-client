# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reached native scalar conversions for direct inputs; Rust owns validity."""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Callable, Mapping
from decimal import Decimal
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from sdm_client._rust import errors
from sdm_client._rust.direct_io import DirectOperations
from sdm_client.wire.dtypes import (
    _exceeds_float64,
    _holds_only_scalars,
    _integral_value,
    _is_missing,
    is_container,
)


def admit(
    series: pd.Series,
    half: str,
    dtype: str,
    call: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> None:
    """Observe scalar admission at each independently reached native pass."""
    name = series.name
    if not _holds_only_scalars(series.dtype):
        for row, value in enumerate(series):
            if is_container(value):
                shape = (
                    'bytes'
                    if isinstance(value, (bytes, bytearray, memoryview))
                    else 'mapping'
                    if isinstance(value, Mapping)
                    else 'container'
                )
                call(
                    'check_direct_cell',
                    {
                        'identity': {
                            'table': half,
                            'column': name,
                            'row': row,
                            'dtype': dtype,
                        },
                        'observation': {
                            'kind': 'scalar_admission',
                            'shape': shape,
                            'native_type': type(value).__name__,
                        },
                    },
                )


def table(
    frame: pd.DataFrame,
    half: str,
    dtypes: dict[str, str],
    call: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> Any:
    """Finish one column before reading the next native column owner."""
    import pyarrow as pa

    names = list(frame.columns)
    arrays = []
    fields = []
    for name in names:
        series = frame[name]
        dtype = dtypes[name]
        admit(series, half, dtype, call)
        completed = (
            call.encode_column(series, dtype)
            if isinstance(call, DirectOperations)
            else None
        )
        if completed is None:
            values = series.tolist()
            completed = [
                convert(
                    value,
                    {
                        'table': half,
                        'column': name,
                        'row': row,
                        'dtype': dtype,
                    },
                    call,
                )
                for row, value in enumerate(values)
            ]
        encoded = [json.dumps(value) for value in completed]
        arrays.append(pa.array(encoded, type=pa.string()))
        fields.append(
            pa.field(
                name,
                pa.string(),
                metadata={
                    'sdm.direct.storage.v1': json.dumps(
                        {'family': 'encoded', 'dtype': dtype}
                    ),
                },
            )
        )
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def convert(
    value: Any,
    identity: dict[str, Any],
    call: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Observe one value sequentially, retaining it for native diagnostics."""

    def check(kind: str, **fields: Any) -> dict[str, Any]:
        with errors.holding(direct_cell=lambda fields: value):
            checked = call(
                'check_direct_cell',
                {
                    'identity': identity,
                    'observation': {'kind': kind, **fields},
                },
            )
        if checked['identity'] != identity:
            raise RuntimeError('the checked cell belongs to a different input')
        return checked

    def encoded(checked: dict[str, Any]) -> dict[str, Any]:
        result = checked['value']
        if not isinstance(result, dict):
            raise RuntimeError(
                'the native cell conversion has no encoded value'
            )
        return result

    missing = check('missingness', missing=_is_missing(value))
    if missing['value'] is not None:
        return encoded(missing)
    dtype = identity['dtype']
    if dtype in ('date', 'time'):
        timestamp = pd.Timestamp(value)
        timestamp = (
            timestamp.tz_localize('UTC')
            if timestamp.tzinfo is None
            else timestamp.tz_convert('UTC')
        )
        return encoded(
            check(
                'timestamp_coercion',
                value=timestamp.isoformat(timespec='microseconds').replace(
                    '+00:00', 'Z'
                ),
            )
        )
    if dtype in ('int32', 'int64', 'float32', 'float64'):
        finite = True
        if isinstance(value, Decimal):
            finite = value.is_finite()
        elif isinstance(value, (float, np.floating)):
            finite = math.isfinite(value)
        numeric = check('numeric_finiteness', finite=finite)
        if dtype in ('int32', 'int64'):
            integer = int(value)
            fractional = not isinstance(value, Integral) and bool(
                integer != value
            )
            within = (
                int(numeric['integer_min'])
                <= integer
                <= int(numeric['integer_max'])
            )
            return encoded(
                check(
                    'integer_coercion',
                    digits=str(integer) if within else None,
                    fractional=fractional,
                )
            )
        try:
            number = float(value)
        except OverflowError:
            check('float_coercion', bits=None)
            raise AssertionError(
                'the core accepted an overflowing conversion'
            ) from None
        bits = list(struct.pack('>d', number))
        check('float_coercion', bits=bits)
        # These helpers observe native representation/equality. They do not
        # raise the SDK's precision diagnostic; that decision stays in Rust.
        precision = check(
            'float_precision', exceeds=bool(_exceeds_float64(value))
        )
        whole = _integral_value(value)
        large_fraction = (
            whole is None
            and isinstance(value, Decimal)
            and bool(abs(value) > precision['exact_integer_limit'])
        )
        return encoded(
            check(
                'float_exactness',
                bits=bits,
                integral_matches=None
                if whole is None
                else int(number) == whole,
                large_fraction=large_fraction,
            )
        )
    if dtype == 'bool':
        return encoded(check('boolean_coercion', value=bool(value)))
    text = format(value, 'f') if isinstance(value, Decimal) else str(value)
    return encoded(check('text_coercion', value=text))
