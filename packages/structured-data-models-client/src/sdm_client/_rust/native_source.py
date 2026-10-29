# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed native source construction, separate from source and task policy."""

from __future__ import annotations

import json
import warnings
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
        if scalar == 'timestamp' and physical == {
            'type': 'timestamp',
            'unit': 'ns',
            'zone': None,
        }:
            return typed_timestamp(values)
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
    if kind in ('decimal', 'exact_decimal'):
        target = {
            'int64': pa.int64(),
            'float64': pa.float64(),
            'utf8': pa.string(),
        }.get(scalar)
        if target is None:
            raise CodecError('native Decimal source meaning has no fixed cast')
        return values.astype(pd.ArrowDtype(target))
    raise CodecError('unknown native source scalar meaning')


def typed_timestamp(values: pd.Series) -> pd.Series:
    """Fixed typed conversion; source completion retains any coercion outcome."""
    converted, _ = box_timestamp(values)
    return finish_timestamp(converted)


def box_timestamp(values: pd.Series) -> tuple[pd.Series, pd.Series | None]:
    """Actual typed boxing and reached native datetime conversion."""
    if isinstance(values.dtype, pd.ArrowDtype):
        values = pd.Series(
            values.to_numpy(), index=values.index, name=values.name
        )
    source = None
    if not pd.api.types.is_datetime64_any_dtype(values):
        source = values
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Could not infer format')
            values = pd.to_datetime(values, unit='ns', errors='coerce')
    return values, source


def finish_timestamp(values: pd.Series) -> pd.Series:
    """Finish only after any reached native coercion notice has been presented."""
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        values = values.dt.tz_convert('UTC').dt.tz_localize(None)
    if values.dtype != 'datetime64[ns]':
        values = values.astype('datetime64[ns]')
    return values
