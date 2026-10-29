# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed native source construction, separate from source and task policy."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TypeAlias

import pandas as pd
import pyarrow as pa

from sdm_client._rust.codec import CodecError

NativeScalarColumn: TypeAlias = pd.Series | pd.api.extensions.ExtensionArray


def _string(value: object) -> str | None:
    if value is None or value is pd.NA:
        return None
    if not isinstance(value, (dict, list)) and bool(pd.isna(value)):
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(',', ':'))
    return str(value)


def scalar_column(
    values: pd.Series, meaning: Mapping[str, object]
) -> NativeScalarColumn:
    """Perform exactly one core-selected native construction or Decimal cast.

    The actual returned array/Series is assigned by the caller at the
    original DataFrame position. Returning a nullable array avoids an extra
    Series constructor. Native failures escape unchanged for the core's
    separate reached fallback or error assessment.
    """
    kind = meaning['meaning']
    if kind == 'original':
        return values
    physical = meaning['physical']
    if not isinstance(physical, Mapping):
        raise CodecError('native source scalar meaning lacks its physical type')
    scalar = physical['type']
    if not isinstance(scalar, str):
        raise CodecError('native source scalar type is not text')
    if kind == 'nullable':
        if scalar == 'utf8':
            return pd.array(values.map(_string), dtype='string')
        dtype = {'int64': 'Int64', 'float64': 'Float64', 'bool': 'boolean'}.get(
            scalar
        )
        if dtype is None:
            raise CodecError(
                'native nullable source meaning has no fixed constructor'
            )
        return pd.array(values, dtype=dtype)
    if kind == 'decimal':
        target = {
            'int64': pa.int64(),
            'float64': pa.float64(),
            'utf8': pa.string(),
        }.get(scalar)
        if target is None:
            raise CodecError('native Decimal source meaning has no fixed cast')
        return values.astype(pd.ArrowDtype(target))
    raise CodecError('unknown native source scalar meaning')
