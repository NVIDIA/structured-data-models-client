# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Values as language-neutral JSON, and rows as Arrow tables.

A database parameter or a value in a row is a Python object, and JSON has no
timestamp, decimal or bytes. Those are written as one-key objects whose key
starts with ``$``; ``tests/parity/FORMAT.md`` lists them. Everything is
encoded the same way on record and on replay, so comparing two encodings is
comparing the values byte for byte.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import math
import uuid
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

import pyarrow as pa

#: Field metadata marking a column stored as encoded JSON text.
JSON_COLUMN = {b'parity.encoding': b'json'}


class Unrepresentable(TypeError):
    r"""A value this format has no lossless form for."""


def encode(value: Any) -> Any:
    r"""``value`` as JSON, with a tagged object for what JSON lacks."""
    import numpy as np
    import pandas as pd

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return {'$float': 'nan'}
        if math.isinf(value):
            return {'$float': 'inf' if value > 0 else '-inf'}
        return value
    if isinstance(value, np.str_):
        return str(value)
    if value is pd.NaT:
        return {'$datetime': None}
    if isinstance(value, pd.Timestamp):
        return {'$datetime': value.isoformat()}
    if isinstance(value, dt.datetime):
        return {'$datetime': value.isoformat()}
    if isinstance(value, dt.date):
        return {'$date': value.isoformat()}
    if isinstance(value, dt.time):
        return {'$time': value.isoformat()}
    if isinstance(value, (pd.Timedelta, dt.timedelta)):
        return {'$timedelta_us': int(pd.Timedelta(value).value // 1000)}
    if isinstance(value, Decimal):
        return {'$decimal': str(value)}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {'$bytes': base64.b64encode(bytes(value)).decode('ascii')}
    if isinstance(value, uuid.UUID):
        return {'$uuid': str(value)}
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise Unrepresentable(f'a mapping key {key!r} is not a string')
            out[key] = encode(item)
        if len(out) == 1 and next(iter(out)).startswith('$'):
            return {'$object': out}
        return out
    if isinstance(value, np.ndarray):
        return [encode(item) for item in value.tolist()]
    if isinstance(value, Sequence):
        return [encode(item) for item in value]
    raise Unrepresentable(
        f'{type(value).__module__}.{type(value).__qualname__} has no form in '
        f'the parity format'
    )


def decode(value: Any) -> Any:
    r"""The Python value :func:`encode` wrote."""
    if isinstance(value, list):
        return [decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if len(value) == 1:
        ((tag, body),) = value.items()
        if tag == '$float':
            return float(body)
        if tag == '$datetime':
            return None if body is None else dt.datetime.fromisoformat(body)
        if tag == '$date':
            return dt.date.fromisoformat(body)
        if tag == '$time':
            return dt.time.fromisoformat(body)
        if tag == '$timedelta_us':
            return dt.timedelta(microseconds=body)
        if tag == '$decimal':
            return Decimal(body)
        if tag == '$bytes':
            return base64.b64decode(body)
        if tag == '$uuid':
            return uuid.UUID(body)
        if tag == '$object':
            return {key: decode(item) for key, item in body.items()}
    return {key: decode(item) for key, item in value.items()}


def canonical(value: Any) -> str:
    r"""One spelling of an encoded value, for comparing byte for byte."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def same(a: Any, b: Any) -> bool:
    r"""Equal, and of the same type all the way down.

    ``1 == 1.0`` and ``True == 1`` in Python; the wire and the database keep
    them apart, and so does this.
    """
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(map(same, a, b))
    if isinstance(a, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, dt.datetime):
        return a == b and a.utcoffset() == b.utcoffset()
    return bool(a == b)


def rows_to_table(
    rows: Sequence[Sequence[Any]], names: Sequence[str]
) -> pa.Table:
    r"""Rows a driver fetched, as an Arrow table that gives them back exactly.

    A column Arrow cannot hold exactly, such as one mixing types, is stored as
    encoded JSON text and marked so in its field metadata.
    """
    columns = [[row[i] for row in rows] for i in range(len(names))]
    arrays, fields = [], []
    for name, values in zip(names, columns, strict=True):
        try:
            array = pa.array(values)
            exact = same(array.to_pylist(), values)
        except (pa.ArrowException, TypeError, ValueError, OverflowError):
            exact = False
        if exact:
            arrays.append(array)
            fields.append(pa.field(name, array.type))
            continue
        encoded = [
            json.dumps(encode(value), sort_keys=True) for value in values
        ]
        if not same([decode(json.loads(text)) for text in encoded], values):
            raise Unrepresentable(
                f'column {name!r} holds values the parity format cannot give '
                f'back exactly: {values[:3]!r}'
            )
        arrays.append(pa.array(encoded, type=pa.string()))
        fields.append(pa.field(name, pa.string(), metadata=JSON_COLUMN))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def table_to_rows(table: pa.Table) -> list[tuple[Any, ...]]:
    r"""The rows :func:`rows_to_table` stored."""
    columns = []
    for field, column in zip(table.schema, table.columns, strict=True):
        values = column.to_pylist()
        if field.metadata and field.metadata.get(b'parity.encoding') == b'json':
            values = [decode(json.loads(text)) for text in values]
        columns.append(values)
    return list(zip(*columns, strict=True)) if columns else []


def encode_table(data: Any) -> dict[str, Any]:
    r"""A table the run wrote to the database, inline: names, types, rows."""
    table = as_arrow(data)
    return {
        'columns': table.column_names,
        'types': [str(field.type) for field in table.schema],
        'rows': [
            [encode(value) for value in row]
            for row in zip(*(c.to_pylist() for c in table.columns), strict=True)
        ],
    }


def as_arrow(data: Any) -> pa.Table:
    r"""An Arrow table, a record batch or a pandas frame, as an Arrow table."""
    if isinstance(data, pa.Table):
        return data
    if isinstance(data, pa.RecordBatch):
        return pa.Table.from_batches([data])
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        return pa.Table.from_pandas(data, preserve_index=False)
    raise Unrepresentable(
        f'cannot store {type(data).__name__} as a table in the parity format'
    )
