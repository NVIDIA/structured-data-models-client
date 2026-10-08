# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The encoded values of the dialect-call capture (``CONTRACT.md``,
"Encoded values"): a Python value as JSON, and back.

The fixture tags of ``tests/parity/FORMAT.md`` are extended with what a
dialect function takes or gives: Arrow arrays with their type, ``Dtype``,
fractions, records and the quoting function ``sample_order_terms`` takes.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import functools
import importlib
import json
import math
import re
import uuid
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1

_TEMPORAL = re.compile(r'(timestamp|date32|date64|time32|time64|duration)\b')


class Unencodable(TypeError):
    r"""A value the capture format has no form for."""


# --- Arrow types -------------------------------------------------------------


def arrow_type(spelled: str) -> pa.DataType:
    r"""The Arrow type ``str(type)`` spells as ``spelled``."""
    found = re.fullmatch(r'timestamp\[(s|ms|us|ns)(?:, tz=(.+))?\]', spelled)
    if found:
        return pa.timestamp(found.group(1), tz=found.group(2))
    if spelled == 'date32[day]':
        return pa.date32()
    if spelled == 'date64[ms]':
        return pa.date64()
    found = re.fullmatch(r'time32\[(s|ms)\]', spelled)
    if found:
        return pa.time32(found.group(1))
    found = re.fullmatch(r'time64\[(us|ns)\]', spelled)
    if found:
        return pa.time64(found.group(1))
    found = re.fullmatch(r'duration\[(s|ms|us|ns)\]', spelled)
    if found:
        return pa.duration(found.group(1))
    found = re.fullmatch(r'decimal(128|256)\((\d+), (-?\d+)\)', spelled)
    if found:
        make = pa.decimal128 if found.group(1) == '128' else pa.decimal256
        return make(int(found.group(2)), int(found.group(3)))
    simple = {
        'null': pa.null(),
        'bool': pa.bool_(),
        'int8': pa.int8(),
        'int16': pa.int16(),
        'int32': pa.int32(),
        'int64': pa.int64(),
        'uint8': pa.uint8(),
        'uint16': pa.uint16(),
        'uint32': pa.uint32(),
        'uint64': pa.uint64(),
        'halffloat': pa.float16(),
        'float': pa.float32(),
        'double': pa.float64(),
        'string': pa.string(),
        'large_string': pa.large_string(),
        'binary': pa.binary(),
        'large_binary': pa.large_binary(),
    }
    if spelled not in simple:
        raise Unencodable(f'the Arrow type {spelled!r} is not in the format')
    return simple[spelled]


def _storage_type(data_type: pa.DataType) -> pa.DataType:
    if pa.types.is_date32(data_type) or pa.types.is_time32(data_type):
        return pa.int32()
    return pa.int64()


def _encode_array(array: pa.Array) -> list[Any]:
    data_type = array.type
    arrow_type(str(data_type))  # Refuses a type outside the format.
    if _TEMPORAL.match(str(data_type)):
        storage = array.view(_storage_type(data_type))
        return [encode(v) for v in storage.to_pylist()]
    if pa.types.is_decimal(data_type):
        return [None if v is None else str(v) for v in array.to_pylist()]
    return [encode(v) for v in array.to_pylist()]


def _decode_array(data_type: pa.DataType, values: list[Any]) -> pa.Array:
    if _TEMPORAL.match(str(data_type)):
        storage = pa.array(
            [decode(v) for v in values], type=_storage_type(data_type)
        )
        return storage.view(data_type)
    if pa.types.is_decimal(data_type):
        return pa.array(
            [None if v is None else Decimal(v) for v in values],
            type=data_type,
        )
    return pa.array([decode(v) for v in values], type=data_type)


# --- Encoding ------------------------------------------------------------------


def _quote_char(quote: Any) -> str | None:
    r"""The quote character of a function that quotes one name, if it is one."""
    from sdm_connectors import quote_ident

    try:
        probe = quote('a')
    except Exception:  # noqa: BLE001 - not a quoting function
        return None
    if not isinstance(probe, str) or len(probe) != 3:
        return None
    char = probe[0]
    for name in ('a', f'x{char}y', 'two words', ''):
        if quote(name) != quote_ident(name, char):
            return None
    return char


def _is_record(value: Any) -> bool:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return True
    return isinstance(value, tuple) and hasattr(value, '_fields')


def _record_fields(value: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(value):
        return {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
        }
    return dict(value._asdict())


def encode(value: Any) -> Any:
    r"""``value`` as capture JSON; the inverse of :func:`decode`."""
    from predictive_query.schema.typing import Dtype

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Dtype):
        return {'$dtype': value.value}
    if isinstance(value, str):
        return str(value)
    if value is pd.NA:
        return {'$na': None}
    if value is pd.NaT:
        return {'$nat': None}
    if isinstance(value, np.generic):
        if isinstance(value, np.str_):
            return str(value)
        if isinstance(value, np.datetime64):
            return encode(pd.Timestamp(value))
        if isinstance(value, np.timedelta64):
            return encode(pd.Timedelta(value))
        return {
            '$np': {'type': value.dtype.name, 'value': encode(value.item())}
        }
    if isinstance(value, int):
        if INT64_MIN <= value <= INT64_MAX:
            return int(value)
        return {'$int': str(int(value))}
    if isinstance(value, float):
        if math.isnan(value):
            return {'$float': 'nan'}
        if math.isinf(value):
            return {'$float': 'inf' if value > 0 else '-inf'}
        return float(value)
    if isinstance(value, Decimal):
        return {'$decimal': str(value)}
    if isinstance(value, Fraction):
        return {
            '$fraction': [encode(value.numerator), encode(value.denominator)]
        }
    if isinstance(value, pd.Timestamp):
        return {'$timestamp': value.isoformat()}
    if isinstance(value, dt.datetime):
        return {'$datetime': value.isoformat()}
    if isinstance(value, dt.date):
        return {'$date': value.isoformat()}
    if isinstance(value, dt.time):
        return {'$time': value.isoformat()}
    if isinstance(value, (pd.Timedelta, dt.timedelta)):
        return {'$timedelta_us': int(pd.Timedelta(value).value // 1000)}
    if isinstance(value, (bytes, bytearray)):
        return {'$bytes': base64.b64encode(bytes(value)).decode('ascii')}
    if isinstance(value, uuid.UUID):
        return {'$uuid': str(value)}
    if isinstance(value, pa.ChunkedArray):
        values: list[Any] = []
        for chunk in value.chunks:
            values.extend(_encode_array(chunk))
        return {
            '$arrow': {
                'type': str(value.type),
                'values': values,
                'chunks': [len(chunk) for chunk in value.chunks],
            }
        }
    if isinstance(value, pa.Array):
        return {
            '$arrow': {'type': str(value.type), 'values': _encode_array(value)}
        }
    if isinstance(value, pa.DataType):
        arrow_type(str(value))
        return {'$arrow_type': str(value)}
    if isinstance(value, pd.Series):
        if not value.index.equals(pd.RangeIndex(len(value))):
            raise Unencodable('a series with an index of its own')
        return {
            '$series': {
                'dtype': str(value.dtype),
                'values': [encode(v) for v in value.tolist()],
                'name': encode(value.name),
            }
        }
    if _is_record(value):
        cls = type(value)
        return {
            '$record': {
                'class': f'{cls.__module__}.{cls.__qualname__}',
                'fields': {
                    key: encode(item)
                    for key, item in _record_fields(value).items()
                },
            }
        }
    if isinstance(value, tuple):
        return {'$tuple': [encode(item) for item in value]}
    if isinstance(value, np.ndarray):
        return {
            '$ndarray': {
                'dtype': 'str' if value.dtype.kind == 'U' else value.dtype.name,
                'values': [encode(item) for item in value.tolist()],
            }
        }
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise Unencodable(f'a mapping key {key!r} is not a string')
            out[key] = encode(item)
        if len(out) == 1 and next(iter(out)).startswith('$'):
            return {'$object': out}
        return out
    if callable(value) and not isinstance(value, type):
        char = _quote_char(value)
        if char is not None:
            return {'$quote': char}
    raise Unencodable(
        f'{type(value).__module__}.{type(value).__qualname__} has no form in '
        f'the capture format'
    )


# --- Decoding ------------------------------------------------------------------


def _import(qualified: str) -> Any:
    parts = qualified.split('.')
    for split in range(len(parts) - 1, 0, -1):
        try:
            found: Any = importlib.import_module('.'.join(parts[:split]))
        except ImportError:
            continue
        for name in parts[split:]:
            found = getattr(found, name)
        return found
    raise ImportError(f'cannot import {qualified}')


def decode(value: Any) -> Any:
    r"""The Python value :func:`encode` wrote."""
    if isinstance(value, list):
        return [decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if len(value) == 1:
        ((tag, body),) = value.items()
        if tag == '$int':
            return int(body)
        if tag == '$float':
            return float(body)
        if tag == '$decimal':
            return Decimal(body)
        if tag == '$timestamp':
            return pd.Timestamp(body)
        if tag == '$datetime':
            return dt.datetime.fromisoformat(body)
        if tag == '$date':
            return dt.date.fromisoformat(body)
        if tag == '$time':
            return dt.time.fromisoformat(body)
        if tag == '$timedelta_us':
            return dt.timedelta(microseconds=body)
        if tag == '$bytes':
            return base64.b64decode(body)
        if tag == '$uuid':
            return uuid.UUID(body)
        if tag == '$na':
            return pd.NA
        if tag == '$nat':
            return pd.NaT
        if tag == '$tuple':
            return tuple(decode(item) for item in body)
        if tag == '$np':
            return np.dtype(body['type']).type(decode(body['value']))
        if tag == '$ndarray':
            return np.array(decode(body['values']), dtype=body['dtype'])
        if tag == '$dtype':
            from predictive_query.schema.typing import Dtype

            return Dtype(body)
        if tag == '$fraction':
            return Fraction(decode(body[0]), decode(body[1]))
        if tag == '$arrow':
            data_type = arrow_type(body['type'])
            values = body['values']
            if 'chunks' not in body:
                return _decode_array(data_type, values)
            chunks, start = [], 0
            for length in body['chunks']:
                chunks.append(
                    _decode_array(data_type, values[start : start + length])
                )
                start += length
            assert start == len(values), 'chunk lengths do not add up'
            return pa.chunked_array(chunks, type=data_type)
        if tag == '$arrow_type':
            return arrow_type(body)
        if tag == '$series':
            return pd.Series(
                decode(body['values']),
                dtype=body['dtype'],
                name=decode(body['name']),
            )
        if tag == '$record':
            cls = _import(body['class'])
            fields = {key: decode(item) for key, item in body['fields'].items()}
            return cls(**fields)
        if tag == '$quote':
            from sdm_connectors import quote_ident

            return functools.partial(quote_ident, char=body)
        if tag == '$object':
            return {key: decode(item) for key, item in body.items()}
    return {key: decode(item) for key, item in value.items()}


# --- Comparing -------------------------------------------------------------------


def same(a: Any, b: Any) -> bool:
    r"""Equal as JSON, type included: ``1`` is not ``1.0`` or ``true``."""
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(map(same, a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, float):
        return a == b and math.copysign(1, a) == math.copysign(1, b)
    return a == b


def first_difference(a: Any, b: Any, path: str = '$') -> str | None:
    r"""Where two JSON values differ, by value or JSON type, or ``None``."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return f'{path}: keys {sorted(a)} != {sorted(b)}'
        for key in a:
            found = first_difference(a[key], b[key], f'{path}.{key}')
            if found:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f'{path}: length {len(a)} != {len(b)}'
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            found = first_difference(x, y, f'{path}[{i}]')
            if found:
                return found
        return None
    if not same(a, b):
        return f'{path}: {_short(a)} != {_short(b)}'
    return None


def _short(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False)
    kind = type(value).__name__
    if len(text) > 300:
        text = text[:300] + '...'
    return f'{text} ({kind})'
