#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Arrow-backed scalar encoding for prediction test inputs."""

from __future__ import annotations

import base64
import math
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any


def _missing(value: Any) -> bool:
    import pandas as pd

    return (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or (isinstance(value, float) and math.isnan(value))
    )


def _scalar(value: Any, arrow_type: Any) -> dict[str, Any]:
    r"""One value of an Arrow column as the contract's scalar
    (``sdm-contract``'s ``Scalar``, its ``t``-tagged JSON)."""
    import pyarrow as pa

    if value is None:
        return {'t': 'null'}
    integers = {
        pa.int8(): 'i8',
        pa.int16(): 'i16',
        pa.int32(): 'i32',
        pa.int64(): 'i64',
        pa.uint8(): 'u8',
        pa.uint16(): 'u16',
        pa.uint32(): 'u32',
        pa.uint64(): 'u64',
    }
    if arrow_type in integers:
        return {'t': integers[arrow_type], 'v': str(value)}
    if pa.types.is_boolean(arrow_type):
        return {'t': 'bool', 'v': value}
    if pa.types.is_floating(arrow_type):
        text = repr(float(value))
        text = {'nan': 'NaN', 'inf': 'inf', '-inf': '-inf'}.get(text, text)
        return {'t': 'f64' if arrow_type == pa.float64() else 'f32', 'v': text}
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return {'t': 'utf8', 'v': value}
    if pa.types.is_binary(arrow_type):
        return {'t': 'binary', 'v': base64.b64encode(value).decode()}
    if pa.types.is_decimal(arrow_type):
        return {
            't': 'd128' if pa.types.is_decimal128(arrow_type) else 'd256',
            'p': arrow_type.precision,
            's': arrow_type.scale,
            'v': str(unscaled(value, arrow_type.scale)),
        }
    if pa.types.is_date32(arrow_type):
        return {'t': 'date32', 'v': str((value - date(1970, 1, 1)).days)}
    # T2.12 (FR P3-8): a timedelta or a time of day, as pyarrow infers
    # them, a count of microseconds.
    one = timedelta(microseconds=1)
    if pa.types.is_duration(arrow_type) and arrow_type.unit == 'us':
        return {'t': 'duration', 'unit': 'us', 'v': str(value // one)}
    if pa.types.is_time64(arrow_type) and arrow_type.unit == 'us':
        since = datetime.combine(date(1970, 1, 1), value) - datetime(1970, 1, 1)
        return {'t': 'time64', 'unit': 'us', 'v': str(since // one)}
    raise TypeError(f'no scalar for {arrow_type}')


def unscaled(value: Decimal, scale: int) -> int:
    r"""``value`` times ``10**scale``, exactly, as an integer: from its
    digits and exponent, never through the caller's decimal context (whose
    precision, 28 digits by default, would round a 38-digit value)."""
    sign, digits, exponent = value.as_tuple()
    if not isinstance(exponent, int):
        raise ValueError(f'not a finite decimal: {value!r}')
    whole = int(''.join(map(str, digits)) or '0')
    shift = exponent + scale
    if shift >= 0:
        whole *= 10**shift
    else:
        whole, rest = divmod(whole, 10**-shift)
        if rest:
            raise ValueError(f'{value!r} has more than {scale} decimals')
    return -whole if sign else whole


def float_text(value: Any) -> str:
    r"""A float as the reference writes a float entity of a text key, the
    core's text of a float entity since T1.36 (the final review's P2-2):
    pyarrow's cast of the float, at its own width (NumPy's ``float32`` as a
    float32), to text, which pandas' cast to the key's ``ArrowDtype`` of
    text makes. The shortest digits that read back as it, plain from 1e-6
    to below 1e10, otherwise one digit, the rest after a point, ``e``, the
    exponent's sign and its digits, unpadded; a whole one with no point:
    ``1e16`` is ``1e+16``, ``1.0`` is ``1``, ``-0.0`` is ``-0``,
    ``1.5e-07`` is ``1.5e-7``, ``1e-06`` is ``0.000001``, and an infinity
    ``inf`` or ``-inf``."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    width = pa.float32() if isinstance(value, np.float32) else pa.float64()
    text = pc.cast(pa.array([value], width), pa.string())[0].as_py()
    assert isinstance(text, str)
    return text


def host_text(value: Any) -> str:
    r"""A value as the host's text column holds it where pyarrow infers no
    one type: a float (NumPy's too) as the reference writes a float entity
    (``float_text``; T1.32g's rule, a float in a mix written as a float
    alone is, with T1.36's text), any other value its ``str()``."""
    import numpy as np

    if isinstance(value, (float, np.floating)):
        return float_text(value)
    return str(value)


def host_arrow(values: list[Any]) -> dict[str, Any]:
    r"""The entities as one Arrow column by the host's natural inference,
    or as text when it fails: the column's type and each value as the
    contract's scalar. A timestamp is carried as its count of its column's
    unit (``us`` or ``ns``, the contract's units)."""
    import pyarrow as pa

    try:
        column = pa.array(values)
        rule = 'inferred'
    except (pa.ArrowException, OverflowError, TypeError, ValueError):
        if all(_missing(v) for v in values):
            column = pa.nulls(len(values))
        else:
            column = pa.array(
                [None if _missing(v) else host_text(v) for v in values],
                pa.string(),
            )
        rule = 'text'
    kind = column.type
    if pa.types.is_timestamp(kind):
        unit = kind.unit if kind.unit in ('us', 'ns') else 'us'
        counts = column.cast(pa.timestamp(unit, kind.tz)).cast(pa.int64())
        scalars = [
            {'t': 'null'}
            if v is None
            else {'t': 'ts', 'unit': unit, 'zone': kind.tz, 'v': str(v)}
            for v in counts.to_pylist()
        ]
        return {
            'rule': rule,
            'type': str(pa.timestamp(unit, kind.tz)),
            'values': scalars,
        }
    return {
        'rule': rule,
        'type': str(kind),
        'values': [_scalar(v, kind) for v in column.to_pylist()],
    }
