# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The host's describer: pandas frames as the core reads them (DV19).

The core decides what each column means; this only says what each column
holds. A column with an Arrow type is handed over as its Arrow array. An
``object`` column has none, so each of its cells is written as JSON, in the
encoding the parity fixtures use (``tests/parity/FORMAT.md``), with the
content ``pandas.api.types.infer_dtype`` reports. Every column's tag names
its pandas dtype.

Nothing here asks ``sdm_client``, ``sdm_relational_engine`` or
``predictive_query`` what a column is: the same frames give the same tables
and tags in any process.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import math
import uuid
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from ._containers import CyclicContainer, NestingTooDeep, ancestors

#: Integers an ``int64`` holds; wider ones are written as their digits.
_INT64 = (-(2**63), 2**63 - 1)


def timestamp_series_units() -> list[str] | None:
    import pandas as pd

    units = ['s', 'ms', 'us', 'ns']
    inferred = [
        str(pd.Series([pd.Timestamp(0).as_unit(unit)]).dtype)
        .removeprefix('datetime64[')
        .removesuffix(']')
        for unit in units
    ]
    return None if inferred == units else inferred


def numpy_text_prefix() -> bool:
    import warnings

    import numpy as np

    modes = []
    for dtype in (int, np.float32):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            try:
                values = np.fromstring('x', sep=',', dtype=dtype)
            except ValueError:
                if captured:
                    raise ValueError(
                        'unsupported NumPy text parser capability'
                    ) from None
                modes.append(False)
            else:
                expected = (
                    'string or file could not be read to its end due to '
                    'unmatched data; this will raise a ValueError in the future.'
                )
                if (
                    values.size != 0
                    or values.dtype != np.dtype(dtype)
                    or len(captured) != 1
                    or captured[0].category is not DeprecationWarning
                    or str(captured[0].message) != expected
                ):
                    raise ValueError('unsupported NumPy text parser capability')
                modes.append(True)
    if modes[0] != modes[1]:
        raise ValueError('unsupported NumPy text parser capability')
    return modes[0]


def _encode(value: Any) -> Any:
    r"""One cell as fixture JSON."""
    try:
        return _encode_at(value, frozenset())
    except (CyclicContainer, NestingTooDeep):
        shape = '$mapping' if isinstance(value, Mapping) else '$container'
        return {shape: type(value).__name__}


def _encode_at(value: Any, holding: frozenset[int]) -> Any:
    import numpy as np
    import pandas as pd

    if id(value) in holding:
        raise CyclicContainer('no plain column cell form for a cycle')
    holding = ancestors(value, holding)

    if value is None or isinstance(value, (bool, str)):
        return value
    if value is pd.NA:
        return {'$na': None}
    if value is pd.NaT:
        return {'$nat': None}
    if isinstance(value, np.generic) and not isinstance(value, np.str_):
        if isinstance(value, np.longdouble):
            # A long double, whose item is itself: read as float64, as
            # Schema.from_data reads its column.
            return _encode_at(float(value), holding)
        if isinstance(value, np.datetime64):
            return _encode_at(pd.Timestamp(value), holding)
        if isinstance(value, np.timedelta64):
            return _encode_at(pd.Timedelta(value), holding)
        converted = value.item()
        if isinstance(converted, np.generic):
            return {
                '$object': {'type': type(value).__name__, 'repr': repr(value)}
            }
        return {
            '$np': {
                'type': value.dtype.name,
                'value': _encode_at(converted, holding),
            }
        }
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, int):
        if _INT64[0] <= value <= _INT64[1]:
            return value
        return {'$int': str(value)}
    if isinstance(value, float):
        if math.isnan(value):
            return {'$float': 'nan'}
        if math.isinf(value):
            return {'$float': 'inf' if value > 0 else '-inf'}
        return value
    if isinstance(value, Decimal):
        return {'$decimal': str(value)}
    if isinstance(value, pd.Timestamp):
        return {'$timestamp': value.isoformat()}
    if isinstance(value, dt.datetime):
        return {'$datetime': value.isoformat()}
    if isinstance(value, dt.date):
        return {'$date': value.isoformat()}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {'$bytes': base64.b64encode(bytes(value)).decode('ascii')}
    if isinstance(value, uuid.UUID):
        return {'$uuid': str(value)}
    if isinstance(value, tuple):
        return {'$tuple': [_encode_at(item, holding) for item in value]}
    if isinstance(value, np.ndarray) and value.ndim == 1:
        return {
            '$ndarray': {
                'dtype': 'str' if value.dtype.kind == 'U' else value.dtype.name,
                'values': [
                    _encode_at(item, holding) for item in value.tolist()
                ],
            }
        }
    if type(value) is dict:
        # A plain dict's contents are what the request carries of it; keys
        # that are not text are written as pairs.
        if all(isinstance(key, str) for key in value):
            out = {
                key: _encode_at(item, holding) for key, item in value.items()
            }
            if len(out) == 1 and next(iter(out)).startswith('$'):
                return {'$dict': out}
            return out
        return {
            '$map': [
                [_encode_at(key, holding), _encode_at(item, holding)]
                for key, item in value.items()
            ]
        }
    if isinstance(value, Mapping):
        return {'$mapping': type(value).__name__}
    if type(value) is list:
        return [_encode_at(item, holding) for item in value]
    if isinstance(value, (list, set, frozenset, range)) or (
        hasattr(value, '__iter__') and not isinstance(value, str)
    ):
        return {'$container': type(value).__name__}
    return {'$object': {'type': type(value).__name__, 'repr': repr(value)}}


def _object_column(series: Any) -> tuple[Any, dict[str, Any]]:
    import pandas as pd
    import pyarrow as pa

    cells = [json.dumps(_encode(value), sort_keys=True) for value in series]
    tag: dict[str, Any] = {
        'dtype': str(series.dtype),
        'cells': 'json',
    }
    try:
        tag['content'] = pd.api.types.infer_dtype(series, skipna=True)
    except NotImplementedError as error:
        if not isinstance(series.dtype, pd.ArrowDtype):
            raise
        tag['infer_dtype_error'] = str(error)
    return pa.array(cells, pa.string()), tag


def _column(series: Any) -> tuple[Any, dict[str, Any]]:
    import pandas as pd
    import pyarrow as pa

    if not isinstance(
        series.dtype, pd.ArrowDtype
    ) and pd.api.types.is_object_dtype(series.dtype):
        return _object_column(series)
    physical_dtype = getattr(series.dtype, 'subtype', series.dtype)
    if series.dtype.kind == 'f' and physical_dtype.itemsize > 8:
        # numpy's long double, which Arrow has no type for: float64, as
        # Schema.from_data reads it.
        return pa.array(series.astype('float64'), from_pandas=True), {
            'dtype': 'float64'
        }
    try:
        array = pa.array(series, from_pandas=True)
    except (pa.ArrowException, TypeError, ValueError):
        if pd.api.types.is_float_dtype(series.dtype):
            bits = min(physical_dtype.itemsize * 8, 64)
            array = pa.array(
                series.to_numpy(dtype=f'float{bits}'), from_pandas=True
            )
        elif pd.api.types.is_integer_dtype(series.dtype):
            bits = physical_dtype.itemsize * 8
            prefix = (
                'uint'
                if pd.api.types.is_unsigned_integer_dtype(series.dtype)
                else 'int'
            )
            array = pa.array(
                series.to_numpy(dtype=f'{prefix}{bits}'), from_pandas=True
            )
        else:
            return _object_column(series)
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if isinstance(array, pa.ExtensionArray):
        # pandas' periods and intervals: each is one value, not the number
        # or the mapping its storage would read as.
        return _object_column(series)
    return array, {'dtype': str(series.dtype)}


def describe_frames(frames: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    r"""``frames``, a mapping of names to pandas ``DataFrame``s, as Arrow
    tables of the same names, and the JSON text of each column's tag: a
    label that is not text has its ``repr`` as ``name_repr``, and a column
    of a frame whose labels are a ``MultiIndex`` is tagged
    ``multi_index``."""
    import pandas as pd
    import pyarrow as pa

    tables: dict[str, Any] = {}
    tags: dict[str, dict[str, Any]] = {}
    for name, frame in frames.items():
        arrays = []
        fields = []
        table_tags: dict[str, Any] = {}
        # Columns labelled by a `MultiIndex` are each named by their tuple;
        # every one's tag says so, as nothing else in the table can.
        multi_index = isinstance(frame.columns, pd.MultiIndex)
        for position, column in enumerate(frame.columns):
            series = frame.iloc[:, position]
            array, tag = _column(series)
            if not isinstance(column, str):
                tag['name_repr'] = repr(column)
            if multi_index:
                tag['multi_index'] = True
            arrays.append(array)
            fields.append(pa.field(str(column), array.type))
            table_tags.setdefault(str(column), tag)
        if arrays:
            tables[name] = pa.Table.from_arrays(
                arrays, schema=pa.schema(fields)
            )
        else:
            # A frame of rows and no columns keeps its rows, which an Arrow
            # table of no arrays does not.
            rows = pa.array([{}] * len(frame), type=pa.struct([]))
            tables[name] = pa.Table.from_batches(
                [pa.RecordBatch.from_struct_array(rows)]
            )
        tags[str(name)] = table_tags
    return tables, json.dumps(tags)
