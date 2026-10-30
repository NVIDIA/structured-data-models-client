# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reached native dtype observations, with inferred meaning owned by Rust."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import Any

import pandas as pd

from sdm_client.wire.dtypes import _is_missing, arrow_decimal_type
from sdm_client._rust.direct_io import DirectOperations


def observe(
    series: pd.Series,
    call: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    if isinstance(call, DirectOperations):
        batched = call.infer_column(series)
        if batched is not None:
            return batched
    dtype = series.dtype
    storage: dict[str, Any]
    if pd.api.types.is_bool_dtype(dtype):
        storage = {'kind': 'boolean'}
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        storage = {'kind': 'timestamp'}
    elif pd.api.types.is_integer_dtype(dtype):
        storage = {
            'kind': 'integer',
            'may_exceed_i64': pd.api.types.is_object_dtype(dtype) or 'uint64' in str(dtype).lower(),
        }
    elif pd.api.types.is_float_dtype(dtype):
        storage = {'kind': 'floating', 'bytes': dtype.itemsize}
    else:
        declared = arrow_decimal_type(dtype)
        if declared is not None:
            scale = declared.scale
            storage = {
                'kind': 'decimal', 'scale': scale,
                'precision': None if scale > 0 else declared.precision,
            }
        elif pd.api.types.is_object_dtype(dtype):
            storage = {'kind': 'object', 'content': pd.api.types.infer_dtype(series, skipna=True)}
        else:
            storage = {'kind': 'other'}
    finding = call('observe_direct_dtype', {'kind': 'storage', 'storage': storage})
    if finding['population'] == 'declared':
        return finding
    # These values are native outcomes, not Python-selected wire types. A
    # completed finding stops iteration before the next effectful scalar.
    for value in series:
        if _is_missing(value):
            continue
        if finding['population'] == 'whole_numbers':
            try:
                integer = int(value)
                within = int(finding['integer_min']) <= integer <= int(finding['integer_max'])
            except (TypeError, ValueError, OverflowError):
                within = None
            finding = call('observe_direct_dtype', {
                'kind': 'whole_number', 'previous': finding, 'within_range': within,
            })
        else:
            is_decimal = isinstance(value, Decimal)
            finite = bool(value.is_finite()) if is_decimal else False
            fractional = False
            outside = False
            if finite:
                exponent = value.as_tuple().exponent
                fractional = exponent < 0 and bool(value != value.to_integral_value())
                if not fractional and value.adjusted() >= finding['decimal_safe_precision']:
                    outside = not (
                        int(finding['integer_min']) <= value <= int(finding['integer_max'])
                    )
            finding = call('observe_direct_dtype', {
                'kind': 'decimal', 'previous': finding,
                'is_decimal': is_decimal, 'finite': finite,
                'fractional': fractional, 'outside_integer_range': outside,
            })
        if finding['population'] == 'declared':
            return finding
    return call('observe_direct_dtype', {'kind': 'exhausted', 'finding': finding})
