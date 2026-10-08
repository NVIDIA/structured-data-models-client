# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The operations the parity fixtures name, run against today's Python.

A fixture case is language-neutral: an operation name, a JSON input, and the
JSON output or error it gives. This module is the Python runner's side of
that contract: it turns an input into the arguments today's code takes, calls
it, and turns what comes back into JSON. ``tests/parity/FORMAT.md`` describes
the encoding; every other language's runner implements the same operations.

An error is ``{"class", "code", "message", "details_keys"}``: the error's
module-qualified class, its ``code`` where it has one (every
``StructuredDataError`` does), otherwise its class name, its message, and the
sorted keys of its ``details``. The warnings an operation gives its caller are
compared as well, in order.
"""

from __future__ import annotations

import base64
import contextlib
import datetime as dt
import json
import math
import re
import sys
import uuid
import warnings
from collections.abc import Callable, Iterator, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1
#: An Arrow-backed decimal, as pandas names its dtype.
_ARROW_DECIMAL = re.compile(r'decimal128\((\d+), (\d+)\)\[pyarrow\]')


class FixtureError(Exception):
    r"""What an operation raised: its class, code, message and the keys of
    its ``details``, and, for an operation that records them, the warnings it
    gave before it raised. The error itself is the ``__cause__``."""

    def __init__(
        self,
        code: str,
        message: str,
        cls: str = '',
        details_keys: list[str] | None = None,
        warnings: list[list[str]] | None = None,
    ) -> None:
        super().__init__(f'{code}: {message}')
        self.code = code
        self.message = message
        # Raised by an operation itself rather than from what it ran.
        self.cls = cls or qualname(type(self))
        self.details_keys = list(details_keys or [])
        self.warnings = warnings

    def described(self) -> dict[str, Any]:
        r"""The error as a fixture's ``expected_error`` records it."""
        described: dict[str, Any] = {
            'class': self.cls,
            'code': self.code,
            'message': self.message,
            'details_keys': self.details_keys,
        }
        if self.warnings is not None:
            described['warnings'] = self.warnings
        return described


# Values.


def decode(value: Any) -> Any:
    r"""A fixture's JSON value as the Python value it stands for."""
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
        if tag == '$datetime':
            return dt.datetime.fromisoformat(body)
        if tag == '$date':
            return dt.date.fromisoformat(body)
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
            if body['type'] == 'longdouble' and body['value'] == '-nan':
                return -np.longdouble('nan')
            # Through an array, so a unit such as datetime64[ns] is kept:
            # the scalar type alone would infer the unit from the value.
            return np.array([decode(body['value'])], dtype=body['type'])[0]
        if tag == '$ndarray':
            return np.array(decode(body['values']), dtype=body['dtype'])
        if tag == '$timestamp':
            return pd.Timestamp(body)
        if tag == '$pandas_timedelta':
            return pd.Timedelta(np.timedelta64(body['value'], body['unit']))
        if tag == '$complex':
            return complex(decode(body[0]), decode(body[1]))
        if tag == '$clongdouble':
            scalar = np.empty((), dtype=np.clongdouble)
            scalar.real = decode(body[0])
            scalar.imag = decode(body[1])
            return scalar[()]
        if tag == '$period':
            return pd.Period(ordinal=body['ordinal'], freq=body['freq'])
        if tag == '$interval':
            return pd.Interval(
                decode(body['left']),
                decode(body['right']),
                closed=body['closed'],
            )
    return {key: decode(item) for key, item in value.items()}


def encode(value: Any) -> Any:
    r"""A Python value as fixture JSON; the inverse of :func:`decode`."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if type(value) is np.longdouble:
        return {
            '$np': {
                'type': 'longdouble',
                'value': '-nan'
                if np.isnan(value) and np.signbit(value)
                else np.format_float_scientific(value, unique=True, trim='k'),
            }
        }
    if type(value) is np.clongdouble:
        return {'$clongdouble': [encode(value.real), encode(value.imag)]}
    if value is pd.NA:
        return {'$na': None}
    if value is pd.NaT:
        return {'$nat': None}
    if isinstance(value, np.generic) and not isinstance(value, np.str_):
        if isinstance(value, (np.datetime64, np.timedelta64)):
            return encode(
                pd.Timestamp(value)
                if isinstance(value, np.datetime64)
                else pd.Timedelta(value)
            )
        return {
            '$np': {'type': value.dtype.name, 'value': encode(value.item())}
        }
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, int):
        if INT64_MIN <= value <= INT64_MAX:
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
    if isinstance(value, complex):
        return {
            '$complex': [
                {'$float': '-nan'}
                if math.isnan(component) and math.copysign(1, component) < 0
                else encode(component)
                for component in (value.real, value.imag)
            ]
        }
    if isinstance(value, pd.Period):
        return {'$period': {'ordinal': value.ordinal, 'freq': value.freqstr}}
    if isinstance(value, pd.Interval):
        endpoints = {
            name: {
                '$pandas_timedelta': {
                    'value': int(endpoint.asm8.view('i8')),
                    'unit': endpoint.unit,
                }
            }
            if isinstance(endpoint, pd.Timedelta)
            else encode(endpoint)
            for name, endpoint in (('left', value.left), ('right', value.right))
        }
        return {
            '$interval': {
                **endpoints,
                'closed': value.closed,
            }
        }
    if isinstance(value, pd.Timestamp):
        return {'$timestamp': value.isoformat()}
    if isinstance(value, dt.datetime):
        return {'$datetime': value.isoformat()}
    if isinstance(value, dt.date):
        return {'$date': value.isoformat()}
    if isinstance(value, dt.time):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta, dt.timedelta)):
        return {'$timedelta_us': int(pd.Timedelta(value).value // 1000)}
    if isinstance(value, (bytes, bytearray)):
        return {'$bytes': base64.b64encode(bytes(value)).decode('ascii')}
    if isinstance(value, uuid.UUID):
        return {'$uuid': str(value)}
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
        return {str(key): encode(item) for key, item in value.items()}
    raise TypeError(f'no fixture form for {type(value).__name__}')


def build_series(spec: dict[str, Any]) -> pd.Series:
    r"""``{"dtype": D, "values": [...]}`` as a pandas series of dtype ``D``."""
    dtype = spec['dtype']
    values = decode(spec['values'])
    if dtype == 'category':
        return pd.Series(pd.Categorical(values))
    decimal = _ARROW_DECIMAL.fullmatch(dtype)
    if decimal:
        import pyarrow as pa

        precision, scale = int(decimal.group(1)), int(decimal.group(2))
        return pd.Series(
            values, dtype=pd.ArrowDtype(pa.decimal128(precision, scale))
        )
    if dtype.endswith('[pyarrow]'):
        import pyarrow as pa

        if dtype.startswith(('time32[', 'time64[')):
            values = [
                dt.time.fromisoformat(value)
                if isinstance(value, str)
                else value
                for value in values
            ]
        if (
            dtype
            == 'dictionary<values=string, indices=int32, ordered=0>[pyarrow]'
        ):
            return pd.Series(
                values,
                dtype=pd.ArrowDtype(pa.dictionary(pa.int32(), pa.string())),
            )
        return pd.Series(
            values, dtype=pd.ArrowDtype(pa.type_for_alias(dtype[:-9]))
        )
    if dtype.startswith('datetime64'):
        series = pd.Series(pd.to_datetime(values))
        return series if str(series.dtype) == dtype else series.astype(dtype)
    if dtype.startswith('timedelta64'):
        return pd.Series(pd.to_timedelta(values)).astype(dtype)
    if dtype == 'object':
        # Element by element, so a list or an array stays one cell.
        cells = np.empty(len(values), dtype=object)
        for index, value in enumerate(values):
            cells[index] = value
        return pd.Series(cells, dtype=object)
    return pd.Series(values, dtype=dtype)


def series_out(series: pd.Series) -> dict[str, Any]:
    r"""A series as ``{"dtype", "values"}``, nulls as pandas holds them."""
    return {
        'dtype': str(series.dtype),
        'values': [encode(value) for value in series.tolist()],
    }


def build_frame(spec: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {name: build_series(column) for name, column in spec['columns'].items()}
    )


def qualname(cls: type) -> str:
    return f'{cls.__module__}.{cls.__qualname__}'


def as_error(error: BaseException) -> FixtureError:
    code = getattr(error, 'code', None)
    if not isinstance(code, str) or not code:
        code = type(error).__name__
    message = getattr(error, 'message', None)
    if not isinstance(message, str):
        message = str(error)
    details = getattr(error, 'details', None)
    keys = (
        sorted(str(k) for k in details) if isinstance(details, Mapping) else []
    )
    return FixtureError(code, message, qualname(type(error)), keys)


def warned(caught: list[warnings.WarningMessage]) -> list[dict[str, str]]:
    r"""The warnings of a ``UserWarning`` class a caller saw, in order."""
    return [
        {'category': qualname(w.category), 'message': str(w.message)}
        for w in caught
        if issubclass(w.category, UserWarning)
    ]


# Dates: sdm_client.core.dates and the time-column reader.


def _strings(values: list[Any]) -> pd.Series:
    return pd.Series(values, dtype=object)


def _dates_day_comes_first(input: dict[str, Any]) -> Any:
    from sdm_client.core import dates

    return dates.day_comes_first(_strings(input['values']))


def _dates_looks_unambiguous(input: dict[str, Any]) -> Any:
    from sdm_client.core import dates

    return dates.looks_unambiguous(_strings(input['values']))


def _dates_shapes_written(input: dict[str, Any]) -> Any:
    from sdm_client.core import dates

    return sorted(dates.shapes_written(_strings(input['values'])))


def _dates_reads_as_a_date(input: dict[str, Any]) -> Any:
    from sdm_client.core import dates

    return [bool(v) for v in dates.reads_as_a_date(_strings(input['values']))]


def _dates_conversion_that_works(input: dict[str, Any]) -> Any:
    from sdm_client.core import dates

    found = dates.conversion_that_works(
        _strings(input['values']),
        only_plausible_years=input.get('only_plausible_years', True),
    )
    return None if found is None else encode(found.arguments)


def _dates_read(input: dict[str, Any]) -> Any:
    r"""The column read the way it says: ISO text, ``null`` for no date."""
    from sdm_client.core import dates

    values = _strings(input['values'])
    if 'conversion' in input:
        conversion = dates.Conversion(**input['conversion'])
    else:
        conversion = dates.conversion_that_works(values)
        if conversion is None:
            raise FixtureError(
                'NO_CONVERSION', 'no conversion reads the column'
            )
    read = dates.read(values, conversion)
    return [None if pd.isna(v) else v.isoformat() for v in read]


def _dates_pattern(input: dict[str, Any]) -> Any:
    from sdm_client.core import dates

    pattern = getattr(dates, input['pattern'])
    how = pattern.fullmatch if input.get('whole', True) else pattern.match
    return how(input['value']) is not None


def _dates_conversion_text(input: dict[str, Any]) -> Any:
    from sdm_client.core import dates

    conversion = dates.Conversion(**input)
    return {'text': str(conversion), 'arguments': list(conversion.arguments)}


def _time_column_read(input: dict[str, Any]) -> Any:
    r"""A string column named as a time column, read or refused."""
    from sdm_client.materializers.relational import _to_time

    read = _to_time(_strings(input['values']), input.get('where', 't.when'))
    return {
        'dtype': str(read.dtype),
        'values': [None if pd.isna(v) else v.isoformat() for v in read],
    }


# The tabular wire: sdm_client.wire.dtypes.


def _wire_infer_dtype(input: dict[str, Any]) -> Any:
    from sdm_client.wire.dtypes import infer_tfm_dtype

    return infer_tfm_dtype(build_series(input['series']))


def _wire_serialize_column(input: dict[str, Any]) -> Any:
    from sdm_client.wire.dtypes import infer_tfm_dtype, serialize_column

    series = build_series(input['series'])
    dtype = input.get('dtype') or infer_tfm_dtype(series)
    return {'dtype': dtype, 'cells': encode(serialize_column(series, dtype))}


def _wire_serialize_cell(input: dict[str, Any]) -> Any:
    from sdm_client.wire.dtypes import serialize_cell

    return encode(serialize_cell(decode(input['value']), input['dtype']))


def _wire_as_declared(input: dict[str, Any]) -> Any:
    from sdm_client.wire.dtypes import as_declared

    return encode(as_declared(decode(input['value']), input['dtype']))


def _wire_json_ready(input: dict[str, Any]) -> Any:
    from sdm_client.wire.dtypes import json_ready

    return encode(json_ready(decode(input['value'])))


# The relational wire: sdm_relational_engine.rfm.inference.payload.


def _payload_cell_json_value(input: dict[str, Any]) -> Any:
    from sdm_relational_engine.rfm.inference.payload import _cell_json_value

    return encode(_cell_json_value(decode(input['value']), input['dtype']))


def _payload_json_value(input: dict[str, Any]) -> Any:
    from sdm_relational_engine.rfm.inference.payload import _json_value

    return encode(_json_value(decode(input['value'])))


def _payload_class_label(input: dict[str, Any]) -> Any:
    from sdm_relational_engine.rfm.inference.payload import _class_label

    return _class_label(decode(input['value']), input['dtype'])


def _payload_decimal_dtype_name(input: dict[str, Any]) -> Any:
    from sdm_relational_engine.rfm.inference.payload import _decimal_dtype_name

    return _decimal_dtype_name(build_series(input['series']))


def _payload_dtype_name(input: dict[str, Any]) -> Any:
    from sdm_relational_engine.rfm.inference.payload import _dtype_name

    return _dtype_name(build_series(input['series']))


def _payload_target_json_value(input: dict[str, Any]) -> Any:
    from predictive_query.schema.task import TaskType
    from sdm_relational_engine.rfm.inference.payload import _target_json_value

    return encode(
        _target_json_value(TaskType(input['task_type']), decode(input['value']))
    )


def _payload_feature_column(input: dict[str, Any]) -> Any:
    r"""A ``FEATURE`` column on a users table, as a relational payload sends it.

    The graph is the one ``test_payload_dtype_fidelity.py`` builds: users with
    the feature, orders over a year, and a regression task of fixed rows.
    """
    from predictive_query.schema.task import TaskType
    from sdm_relational_engine.rfm import Graph, KumoRelational, TaskTable

    feature = build_series(input['feature'])
    users_count = len(feature)
    rng = np.random.default_rng(0)
    orders = pd.DataFrame(
        {
            'ORDER_ID': np.arange(6 * users_count, dtype='int64'),
            'USER_ID': np.tile(np.arange(users_count, dtype='int64'), 6),
            'AMOUNT': rng.uniform(5, 100, 6 * users_count).round(2),
            'TIME': pd.to_datetime('2024-01-01')
            + pd.to_timedelta(
                rng.integers(0, 420, 6 * users_count).astype('int64'), unit='D'
            ),
        }
    )
    users = pd.DataFrame({'USER_ID': np.arange(users_count, dtype='int64')})
    users['FEATURE'] = feature.values
    task = TaskTable(
        task_type=TaskType.REGRESSION,
        context_df=pd.DataFrame(
            {
                'ENTITY': np.arange(users_count - 2, dtype='int64'),
                'TARGET': np.arange(users_count - 2, dtype='float64'),
                'ANCHOR_TIMESTAMP': pd.to_datetime(
                    ['2025-02-20'] * (users_count - 2)
                ),
            }
        ),
        pred_df=pd.DataFrame(
            {
                'ENTITY': np.arange(
                    users_count - 2, users_count, dtype='int64'
                ),
                'ANCHOR_TIMESTAMP': pd.to_datetime(['2025-02-20'] * 2),
            }
        ),
        entity_table_name='USERS',
        entity_column='ENTITY',
        target_column='TARGET',
        time_column='ANCHOR_TIMESTAMP',
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        graph = Graph.from_data(
            {'USERS': users, 'ORDERS': orders}, verbose=False
        )
        payload = (
            KumoRelational(graph, verbose=False)
            .materialize_task(task, verbose=False)[0]
            .payload
        )
    table = payload['context']['related_tables']['USERS']
    index = table['columns'].index('FEATURE')
    return {
        'stype': str(graph['USERS']['FEATURE'].stype),
        'declared': payload['schema']['related_tables']['USERS']['columns'][
            'FEATURE'
        ],
        'cells': [row[index] for row in table['rows']],
    }


# Composite keys: sdm_relational_engine.rfm.base.composite_key.


def _composite_encode_values(input: dict[str, Any]) -> Any:
    from sdm_relational_engine.rfm.base.composite_key import encode_values

    return encode_values(decode(input['parts']))


def _composite_decode_value(input: dict[str, Any]) -> Any:
    from sdm_relational_engine.rfm.base.composite_key import decode_value

    return list(decode_value(input['value'], input['arity']))


def _composite_encode_identity(input: dict[str, Any]) -> Any:
    from sdm_relational_engine.rfm.base.composite_key import encode_identity

    return encode_identity(input['columns'])


def _composite_decode_identity(input: dict[str, Any]) -> Any:
    from sdm_relational_engine.rfm.base.composite_key import decode_identity

    found = decode_identity(input['derived'])
    return None if found is None else list(found)


def _composite_encode_frame(input: dict[str, Any]) -> Any:
    from sdm_relational_engine.rfm.base.composite_key import encode_frame

    return encode_frame(build_frame(input['frame']), input['columns']).tolist()


# Labels: sdm_relational_engine.rfm.query.pandas_executor.


def _executor() -> Any:
    from sdm_relational_engine.rfm.query.pandas_executor import (
        PQueryPandasExecutor,
    )

    return PQueryPandasExecutor()


def _constant(spec: dict[str, Any]) -> Any:
    from predictive_query.ast import Constant
    from predictive_query.schema.typing import Dtype

    value = spec['value']
    if isinstance(value, list):
        value = [_constant(item) for item in value]
    dtype = spec.get('dtype')
    return Constant(
        value=value, dtype_maybe=None if dtype is None else Dtype(dtype)
    )


def _labels_column(input: dict[str, Any]) -> Any:
    from predictive_query.ast import Column

    out, mask = _executor().execute_column(
        Column(fqn=input['fqn']),
        {name: build_frame(frame) for name, frame in input['tables'].items()},
        input['filter_na'],
    )
    return {'out': series_out(out), 'mask': [bool(m) for m in mask]}


def _labels_rel_op(input: dict[str, Any]) -> Any:
    from predictive_query.schema.typing import RelOp

    out = _executor().execute_rel_op(
        build_series(input['left']),
        RelOp(input['op']),
        _constant(input['right']),
    )
    return series_out(out)


def _labels_member_op(input: dict[str, Any]) -> Any:
    from predictive_query.schema.typing import MemberOp

    out = _executor().execute_member_op(
        left=build_series(input['left']),
        op=MemberOp(input['op']),
        right=_constant(input['right']),
    )
    return series_out(out)


def _labels_bool_op(input: dict[str, Any]) -> Any:
    from predictive_query.schema.typing import BoolOp

    right = input.get('right')
    out = _executor().execute_bool_op(
        build_series(input['left']),
        BoolOp(input['op']),
        None if right is None else build_series(right),
    )
    return series_out(out)


def _labels_aggregation_type(input: dict[str, Any]) -> Any:
    from predictive_query.schema.typing import AggregationType

    out, mask = _executor().execute_aggregation_type(
        op=AggregationType(input['op']),
        feat=build_series(input['feat']),
        batch=np.array(input['batch'], dtype='int64'),
        batch_size=input['batch_size'],
        filter_na=input['filter_na'],
    )
    return {'out': series_out(out), 'mask': [bool(m) for m in mask]}


def _labels_aggregation(input: dict[str, Any]) -> Any:
    from predictive_query.ast import Aggregation, Column, DateOffsetRange
    from predictive_query.schema.typing import AggregationType, TimeUnit

    spec = input['aggregation']
    aggr = Aggregation(
        target=Column(fqn=spec['target']),
        aggr=AggregationType(spec['aggr']),
        aggr_time_range=DateOffsetRange(
            spec['start'], spec['end'], TimeUnit(spec['unit'])
        ),
    )
    out, mask = _executor().execute_aggregation(
        aggr=aggr,
        feat_dict={n: build_frame(f) for n, f in input['tables'].items()},
        time_dict={n: build_series(s) for n, s in input['times'].items()},
        batch_dict={
            n: np.array(b, dtype='int64') for n, b in input['batches'].items()
        },
        anchor_time=build_series(input['anchor_time']),
        filter_na=input['filter_na'],
    )
    return {'out': series_out(out), 'mask': [bool(m) for m in mask]}


# Sampling's building blocks: relationallib's NeighborSampler, the local graph
# store, DateOffset arithmetic and the random stream of a seed.

#: The six dicts ``NeighborSampler.sample`` returns, in order.
_SAMPLED = (
    'row',
    'col',
    'node_id',
    'batch',
    'num_sampled_nodes',
    'num_sampled_edges',
)


def _sampling_neighbor_sample(input: dict[str, Any]) -> Any:
    r"""A ``relationallib.NeighborSampler`` made from ``sampler`` and sampled
    once per call, reseeded where a call says: each result's six dicts of
    lists. Arrays are ``$ndarray`` values; anything else is given as is."""
    import sdm_relational_engine.relationallib as relationallib

    given = input['sampler']
    args = [
        given['node_types'],
        [tuple(edge) for edge in given['edge_types']],
        decode(given['colptr']),
        decode(given['row']),
        decode(given['node_time']),
    ]
    sampler = (
        relationallib.NeighborSampler(*args, seed=decode(given['seed']))
        if 'seed' in given
        else relationallib.NeighborSampler(*args)
    )
    out = []
    for call in input['calls']:
        if 'reseed' in call:
            sampler.seed(decode(call['reseed']))
        result = sampler.sample(
            decode(call['num_neighbors']),
            decode(call['sampling_time_offsets']),
            call['entity_table'],
            decode(call['seed_node']),
            decode(call['seed_time']),
        )
        out.append(
            {
                name: {key: values.tolist() for key, values in part.items()}
                for name, part in zip(_SAMPLED, result, strict=True)
            }
        )
    return out


def _sampling_graph_store(input: dict[str, Any]) -> Any:
    r"""``LocalGraphStore`` of in-memory tables and a graph as
    ``tests/rust_core/sampling/CONTRACT.md`` (The graph store) describes it:
    its report, masks, key maps, times and CSC arrays."""
    from sdm_relational_engine.rfm import Edge, Graph, LocalTable
    from sdm_relational_engine.rfm.backend.local import LocalGraphStore

    frames = {name: build_frame(f) for name, f in input['tables'].items()}
    tables = []
    for spec in input['graph']['tables']:
        table = LocalTable(
            frames[spec['name']],
            spec['name'],
            primary_key=spec['primary_key'],
            time_column=spec['time_column'],
            end_time_column=spec['end_time_column'],
        )
        for column in spec['columns']:
            found = table[column['name']]
            if str(found.dtype) != column['dtype']:
                raise ValueError(
                    f"column '{column['name']}' reads as {found.dtype}, "
                    f'not {column["dtype"]}'
                )
            if str(found.stype) != column['stype']:
                found.stype = column['stype']
        tables.append(table)
    graph = Graph(
        tables,
        [
            Edge(e['src_table'], e['fkey'], e['dst_table'])
            for e in input['graph']['edges']
        ],
    )
    store = LocalGraphStore(graph, verbose=False)
    return {
        'report': {
            name: {
                'input_rows': r.input_rows,
                'output_rows': r.output_rows,
                'null_primary_key_rows': r.null_primary_key_rows,
                'duplicate_primary_key_rows': r.duplicate_primary_key_rows,
                'null_time_rows': r.null_time_rows,
            }
            for name, r in store.sanitization_report.tables.items()
        },
        'masks': {k: [bool(x) for x in v] for k, v in store.mask_dict.items()},
        'keys': {
            k: [encode(x) for x in v.index.tolist()]
            for k, v in store.pkey_map_dict.items()
        },
        'node_ids': {
            k: v['arange'].tolist() for k, v in store.pkey_map_dict.items()
        },
        'time': {k: v.tolist() for k, v in store.time_dict.items()},
        'time_range': {
            k: [int(lo.value), int(hi.value)]
            for k, (lo, hi) in store.min_max_time_dict.items()
        },
        'colptr': {
            '__'.join(k): v.tolist() for k, v in store.colptr_dict.items()
        },
        'row': {'__'.join(k): v.tolist() for k, v in store.row_dict.items()},
    }


def _date_offset(spec: dict[str, Any]) -> Any:
    return pd.DateOffset(n=spec['n'], **spec['kwds'])


def _sampling_date_offset_apply(input: dict[str, Any]) -> Any:
    r"""``pd.Timestamp(t, unit='ns') + offset`` (or minus) for each
    timestamp, as ``{"ns"}`` or the error it raises."""
    offset = _date_offset(input['offset'])
    out = []
    for t in input['timestamps']:
        try:
            ts = pd.Timestamp(t, unit='ns')
            moved = ts + offset if input['sign'] > 0 else ts - offset
            out.append({'ns': int(moved.value)})
        except Exception as error:  # noqa: BLE001 - every refusal is an outcome
            out.append(
                {'error': {'code': type(error).__name__, 'message': str(error)}}
            )
    return out


def _sampling_date_offset_to_seconds(input: dict[str, Any]) -> Any:
    from sdm_relational_engine.rfm.base.utils import date_offset_to_seconds

    return date_offset_to_seconds(_date_offset(input['offset']), input['bound'])


def _sampling_random_state(input: dict[str, Any]) -> Any:
    r"""``random_state(seed).randint(low, high, size=size)``."""
    from sdm_relational_engine.rfm.base.utils import random_state

    drawn = random_state(decode(input['seed'])).randint(
        input['low'], input['high'], size=input['size']
    )
    return drawn.tolist()


# Questions asked of frames: schema inference, task derivation, rewriting.


def _sampling_to_list(input: dict[str, Any]) -> Any:
    r"""``Table._sanitize`` of one column of text, held as pandas holds a
    SQL result's text (``arrow``), a frame's (``str``) or a driver's Python
    objects (``object``), with a list dtype or the ``sequence`` stype: the
    column's dtype after, and its values, a list as numpy's array."""
    import pyarrow as pa
    from predictive_query.schema.typing import Dtype, Stype
    from sdm_relational_engine.rfm.base.table import Table

    values = input['values']
    kind = input['series']
    if kind == 'arrow':
        series = pd.Series(
            pa.array(values, pa.string()), dtype=pd.ArrowDtype(pa.string())
        )
    elif kind == 'str':
        series = pd.Series(values, dtype='str')
    elif kind == 'object':
        series = pd.Series(values, dtype=object)
    else:
        raise ValueError(f'no series of kind {kind!r}')
    dtype, stype = input.get('dtype'), input.get('stype')
    out = Table._sanitize(
        pd.DataFrame({'x': series}),
        {'x': None if dtype is None else Dtype(dtype)},
        {'x': None if stype is None else Stype(stype)},
    )['x']
    return {
        'dtype': str(out.dtype),
        'values': [
            encode(value)
            if isinstance(value, (np.ndarray, list))
            else (None if pd.isna(value) else value)
            for value in out.tolist()
        ],
    }


def _relationships(given: Any) -> Any:
    from sdm_client import Relationship

    if given is None:
        return None
    return [Relationship(*link) for link in given]


def _keys(given: Any) -> Any:
    if given is None:
        return None
    return {
        table: tuple(key) if isinstance(key, list) else key
        for table, key in given.items()
    }


def _describe_schema(schema: Any) -> dict[str, Any]:
    tables = {}
    for name, table in schema.tables.items():
        key = table.primary_key
        tables[name] = {
            'primary_key': list(key) if isinstance(key, tuple) else key,
            'time_column': table.time_column,
            'columns': {
                column: {'dtype': str(spec.dtype), 'stype': str(spec.stype)}
                for column, spec in table.columns.items()
            },
        }
    return {
        'tables': tables,
        'relationships': [
            [r.source_table, r.source_column, r.target_table, r.target_column]
            for r in schema.relationships
        ],
    }


def _schema_from_data(input: dict[str, Any]) -> Any:
    r"""What ``Schema.from_data`` infers, after any edits, and what it said."""
    from sdm_client import Schema

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        schema = Schema.from_data(
            {name: build_frame(f) for name, f in input['tables'].items()},
            primary_keys=_keys(input.get('primary_keys')),
            time_columns=input.get('time_columns'),
            relationships=_relationships(input.get('relationships')),
        )
        for edit in input.get('edits', []):
            column = schema[edit['table']][edit['column']]
            setattr(column, edit['set'], edit['value'])
    out = _describe_schema(schema)
    out['warnings'] = [
        str(w.message) for w in caught if issubclass(w.category, UserWarning)
    ]
    return out


def _tabular_task(input: dict[str, Any]) -> Any:
    r"""The task a single-table question is gathered as, and the payload's
    target column, through ``PredictiveQuery(...).materialize(predict=...)``."""
    from sdm_client import PredictiveQuery
    from sdm_client.adapters.tabular import TabularAdapter

    frame = build_frame(input['frame'])
    scored = frame.head(input.get('scored', 2)).drop(columns=[input['target']])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        request = PredictiveQuery(
            input['query'], {input['table']: frame}
        ).materialize(predict=scored, **decode(input.get('options', {})))
        payload = next(
            iter(TabularAdapter().prediction_plan(request, ('prediction',)))
        )
    return {
        'kind': request.task.kind,
        'target_dtype': request.task.target.dtype,
        'declared': payload['schema']['instance_table']['columns'][
            input['target']
        ],
        'warnings': [
            str(w.message)
            for w in caught
            if issubclass(w.category, UserWarning)
        ],
    }


def _tabular_two_whole_numbers(input: dict[str, Any]) -> Any:
    from sdm_client.materializers.tabular import _holds_two_whole_numbers

    return _holds_two_whole_numbers(build_series(input['series']))


def _decompose_as_literal(input: dict[str, Any]) -> Any:
    from sdm_client.decompose import _as_literal

    return _as_literal(decode(input['value']))


def _decompose_parts(input: dict[str, Any]) -> Any:
    r"""The questions a ranking or forecast is asked as, over given frames."""
    from sdm_client import PredictiveQuery, Schema

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {name: build_frame(f) for name, f in input['tables'].items()},
            primary_keys=_keys(input.get('primary_keys')),
            time_columns=input.get('time_columns'),
        )
        request = PredictiveQuery(input['query'], schema).materialize(
            **decode(input.get('options', {}))
        )
    rewritten = request._decomposition
    if rewritten is None:
        return None
    return {
        'kind': rewritten.kind,
        'top': rewritten.top,
        'parts': [[text, encode(about)] for text, about in rewritten.parts],
    }


# The sampling driver, end to end: a question asked through the public API
# of tables in a database built for the case.

#: Where the scenarios' data generators and loaders are.
_PARITY_SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts' / 'parity'


def _run_setup(backend: str, connection: Any, statements: list[str]) -> None:
    r"""Run the case's own statements, such as keys and indexes, once the
    data is loaded."""
    for statement in statements:
        if backend == 'sqlite':
            with connection.cursor() as cursor:
                cursor.execute(statement)
            connection.commit()
        else:
            connection.execute(statement)


@contextlib.contextmanager
def _driver_schema(input: dict[str, Any]) -> Iterator[Any]:
    r"""The schema of the case's data, read as ``Schema.from_data`` or
    ``Schema.from_source`` reads it, with the stypes the case declares."""
    if str(_PARITY_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_PARITY_SCRIPTS))
    from _parity import data as datagen
    from _parity.runner import _source, real_connection
    from sdm_client import Schema

    backend = input['backend']
    data = dict(input['data'])
    tables = input['tables']
    options = {
        'primary_keys': _keys(input.get('primary_keys')),
        'time_columns': input.get('time_columns'),
        'relationships': _relationships(input.get('relationships')),
    }

    def declared(schema: Any) -> Any:
        for name, stype in (input.get('stypes') or {}).items():
            table, column = name.split('.', 1)
            schema[table][column].stype = stype
        return schema

    if backend == 'memory':
        frames = datagen.generate(data['generator'], data['seed'])
        yield declared(
            Schema.from_data({name: frames[name] for name in tables}, **options)
        )
        return
    if 'check_keys' in input:
        options['check_keys'] = input['check_keys']
    transcript = {'backend': backend, 'data': data}
    with real_connection(transcript) as connection:
        _run_setup(backend, connection, input.get('setup') or [])
        source = _source(transcript, connection, None)
        yield declared(
            Schema.from_source(
                source, tables={name: name for name in tables}, **options
            )
        )


def _user_warnings(caught: list[warnings.WarningMessage]) -> list[list[str]]:
    r"""Each ``UserWarning`` caught, in order, as ``[category, message]``."""
    return [
        [f'{w.category.__module__}.{w.category.__qualname__}', str(w.message)]
        for w in caught
        if issubclass(w.category, UserWarning)
    ]


def _driver_materialize(input: dict[str, Any]) -> Any:
    r"""What ``PredictiveQuery(query, schema).materialize(**options)`` sends
    and says: how many payloads, their SHA-256 as canonical JSON (keys
    sorted, no spaces, the request id fixed), and each ``UserWarning`` it
    gave, in order, as ``[category, message]``. Where it raises, the error
    carries the warnings it gave first.

    Each case starts as a fresh process would: the sampler's warning that it
    found too few examples has not been given yet.
    """
    import hashlib
    import json

    from sdm_client import PredictiveQuery, _impl

    if _impl.OLD:
        from sdm_relational_engine.rfm.base import sampler as base_sampler

        base_sampler._coverage_warned = False
    else:
        from sdm_client import _request_core

        _request_core._coverage_warned = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            with _driver_schema(input) as schema:
                request = PredictiveQuery(input['query'], schema).materialize(
                    **decode(input.get('options') or {})
                )
        except Exception as error:  # noqa: BLE001 - raised with its warnings
            raised = as_error(error)
            raised.warnings = _user_warnings(caught)
            raise raised from error
    payloads = json.loads(json.dumps(list(request._payloads)))
    for payload in payloads:
        metadata = payload.get('metadata')
        if isinstance(metadata, dict) and 'request_id' in metadata:
            metadata['request_id'] = 'request'
    text = json.dumps(payloads, sort_keys=True, separators=(',', ':'))
    return {
        'payloads': len(payloads),
        'sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
        'warnings': _user_warnings(caught),
    }


# The client end to end: public calls over frames, answered by given responses.


class _Response:
    def __init__(self, status: int, body: Any) -> None:
        self.status_code = status
        self.headers = {'Content-Type': 'application/json'}
        self._body = b'' if body is None else json.dumps(body).encode('utf-8')

    def iter_content(self, chunk_size: int = 1) -> Iterator[bytes]:
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _path_of(url: str) -> str:
    at = url.find('/v1/')
    return url[at:] if at >= 0 else url


class Answering:
    r"""What the client's transport sends through: each request is answered
    with the next of ``responses``, ``{"method", "path", "status", "body"}``,
    which must be for its method and path.

    ``requests`` notes each request's method and path, in order, and
    ``bodies`` the bodies posted.
    """

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.requests: list[list[str]] = []
        self.bodies: list[Any] = []

    def _serve(self, method: str, url: str) -> _Response:
        path = _path_of(url)
        at = len(self.requests)
        self.requests.append([method, path])
        if at >= len(self._responses):
            raise AssertionError(
                f'request {at} ({method} {path}) has no response in the fixture'
            )
        given = self._responses[at]
        if (given['method'], given['path']) != (method, path):
            raise AssertionError(
                f'request {at} is {method} {path}, and the fixture answers '
                f'{given["method"]} {given["path"]}'
            )
        return _Response(int(given['status']), given.get('body'))

    def post(self, url: str, data: bytes, **kwargs: Any) -> _Response:
        self.bodies.append(json.loads(data))
        return self._serve('POST', url)

    def delete(self, url: str, **kwargs: Any) -> _Response:
        return self._serve('DELETE', url)

    def close(self) -> None:
        return None


def _text_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _text_keys(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_text_keys(item) for item in value]
    return value


class Flow:
    r"""Makes public calls over the input's frames, in order.

    The calls are those of a parity transcript (``FORMAT.md``, ``calls``),
    plus ``Request.halves`` (read both halves of the request made last, which
    the output then holds), ``Schema.set`` (set an attribute of a schema
    column),
    ``Request.assign`` (assign a half of the request), and
    ``SDMClient.for_databricks_serving`` (make the client one of a serving
    endpoint). ``SDMClient.predict``, ``SDMClient.fit`` and
    ``FittedModel.predict`` take ``request`` to send something other than the
    request made last. A value
    ``{"$frame": {"table": T, "rows": [A, B], "drop": [C, ...]}}`` is frame
    ``T`` of the input's ``tables``; ``{"$frames": [T, ...]}`` is a mapping of
    them by name; ``{"$model": NAME}`` is an instance of the model handle
    ``sdm_client.models.NAME``; ``{"$schema": null}`` is the schema made last;
    ``{"$cyclic": KEY}`` is a mapping holding itself under ``KEY``;
    ``{"$inference_config": {"class": C, "fields": {...}}}`` is the engine's
    inference-config dataclass ``C`` with those fields. ``Schema.delete``
    leaves a column of the schema made last out.

    Some calls change what earlier calls made, in place, as a caller's
    statement between two calls does: ``Frame.set_columns`` relabels the
    columns of an input frame (a label given as a list makes the labels a
    ``MultiIndex`` of those tuples), ``Frame.drop`` drops columns of one,
    and ``Schema.put_source_table`` puts a ``SourceTable`` of a table's
    name, columns and keys into the ``tables`` of the schema made last.
    ``PredictionResult.absent`` asks the last result which of the names it
    is given it lacks, and ``Request.payloads`` reads the envelopes of the
    request made last, which the output then holds.
    """

    def __init__(self, tables: dict[str, pd.DataFrame], session: Any) -> None:
        self.tables = tables
        self.session = session
        self.schema: Any = None
        self.query: Any = None
        self.request: Any = None
        self.result: Any = None
        self.results: list[Any] = []
        self.fitted: Any = None
        self.frame: Any = None
        self.client: Any = None
        self.halves: dict[str, Any] | None = None
        self.absent: list[str] | None = None
        self.payloads: list[Any] | None = None

    def value(self, raw: Any) -> Any:
        if isinstance(raw, dict) and len(raw) == 1:
            ((tag, body),) = raw.items()
            if tag == '$frame':
                frame = self.tables[body['table']]
                if 'rows' in body:
                    start, stop = body['rows']
                    frame = frame.iloc[start:stop]
                if body.get('drop'):
                    frame = frame.drop(columns=body['drop'])
                return frame
            if tag == '$frames':
                return {name: self.tables[name] for name in body}
            if tag == '$model':
                import sdm_client.models as models

                return getattr(models, body)()
            if tag == '$schema':
                return self.schema
            if tag == '$inference_config':
                # The engine's dataclass, which a caller may hand over in
                # place of a mapping: ``{"class": NAME, "fields": {...}}``.
                import sdm_relational_engine.rfm.inference.config as config

                return getattr(config, body['class'])(**body.get('fields', {}))
            if tag == '$cyclic':
                # A mapping holding itself under the key given, which JSON
                # cannot write and a caller's Python can.
                cyclic: dict[str, Any] = {}
                cyclic[str(body)] = cyclic
                return cyclic
        if isinstance(raw, dict) and not (
            len(raw) == 1 and next(iter(raw)).startswith('$')
        ):
            return {key: self.value(item) for key, item in raw.items()}
        if isinstance(raw, list):
            return [self.value(item) for item in raw]
        return decode(raw)

    def run(self, calls: list[dict[str, Any]]) -> None:
        try:
            for call in calls:
                args = {
                    key: self.value(item)
                    for key, item in call.get('args', {}).items()
                }
                getattr(self, '_' + call['call'].replace('.', '_'))(**args)
        finally:
            if self.client is not None:
                client, self.client = self.client, None
                client.close()

    def _SDMClient_for_databricks_serving(self, endpoint: str) -> None:
        r"""A client of a Databricks serving endpoint, through a workspace
        client that is never called: what such a client refuses before
        sending anything."""
        from sdm_client import SDMClient

        self.client = SDMClient.for_databricks_serving(
            endpoint, workspace_client=object()
        )

    def _client(self) -> Any:
        if self.client is None:
            from sdm_client import SDMClient

            self.client = SDMClient(url='http://localhost:8000')
            self.client._transport._session.close()
            self.client._transport._session = self.session
        return self.client

    def _Schema_from_data(self, tables: Any = None, **args: Any) -> None:
        from sdm_client import Schema

        names = list(self.tables) if tables is None else tables
        if args.get('relationships') is not None:
            args['relationships'] = _relationships(args['relationships'])
        if args.get('primary_keys') is not None:
            args['primary_keys'] = _keys(args['primary_keys'])
        self.schema = Schema.from_data(
            {name: self.tables[name] for name in names}, **args
        )

    def _Schema_from_duckdb(
        self, load: list[str] | None = None, **args: Any
    ) -> None:
        r"""``Schema.from_source`` over an in-memory DuckDB holding the
        input's tables named in ``load``, all of them by default."""
        import duckdb
        from sdm_client import Schema
        from sdm_connectors import DuckDBSource

        connection = duckdb.connect()
        connection.execute('SET threads TO 1')
        for name in self.tables if load is None else load:
            connection.register(f'{name}_frame', self.tables[name])
            connection.execute(
                f'CREATE TABLE {name} AS SELECT * FROM {name}_frame'
            )
            connection.unregister(f'{name}_frame')
        if args.get('relationships') is not None:
            args['relationships'] = _relationships(args['relationships'])
        self.schema = Schema.from_source(DuckDBSource(connection), **args)

    def _Schema_link(self, link: list[str]) -> None:
        r"""Append a relationship to the schema made last."""
        self.schema.relationships.append(*_relationships([link]))

    def _Schema_set_table(self, table: str, attribute: str, value: Any) -> None:
        r"""Set an attribute of a table of the schema made last."""
        setattr(self.schema[table], attribute, value)

    def _Schema_set(
        self, table: str, column: str, attribute: str, value: Any
    ) -> None:
        setattr(self.schema[table][column], attribute, value)

    def _Schema_delete(self, table: str, column: str) -> None:
        r"""Leave a column of the schema made last out, as
        ``del schema[table][column]`` does."""
        del self.schema[table][column]

    def _Schema_put_source_table(self, table: str) -> None:
        r"""Put a ``SourceTable`` of the table's name, columns and keys into
        the ``tables`` of the schema made last, in place of the table."""
        from sdm_client.schema import SourceTable

        held = self.schema[table]
        self.schema.tables[table] = SourceTable(
            table,
            table,
            dict(held.columns),
            primary_key=held.primary_key,
            time_column=held.time_column,
        )

    def _Frame_set_columns(self, table: str, columns: list[Any]) -> None:
        r"""Relabel the columns of the input's frame ``table`` in place, as
        ``frame.columns = columns`` does; labels given as lists make the
        labels a ``MultiIndex`` of those tuples."""
        frame = self.tables[table]
        if any(isinstance(label, list) for label in columns):
            frame.columns = pd.MultiIndex.from_tuples(
                [tuple(label) for label in columns]
            )
        else:
            frame.columns = columns

    def _Frame_drop(self, table: str, columns: list[str]) -> None:
        r"""Drop columns of the input's frame ``table`` in place."""
        self.tables[table].drop(columns=columns, inplace=True)

    def _PredictiveQuery(self, query: Any, schema: Any = None) -> None:
        from sdm_client import PredictiveQuery

        self.query = PredictiveQuery(
            query, self.schema if schema is None else schema
        )

    def _PredictiveQuery_materialize(self, **args: Any) -> None:
        self.request = self.query.materialize(**args)

    def _Request(self, **args: Any) -> None:
        from sdm_client import Request

        self.request = Request(**args)

    def _Request_assign(self, half: str, value: Any) -> None:
        from sdm_client.request import Split

        setattr(self.request, half, Split(value))

    def _SDMClient_predict(
        self, model: Any, outputs: Any = None, **given: Any
    ) -> None:
        request = given.get('request', self.request)
        self.result = self._client().predict(
            request, model=model, outputs=outputs
        )
        self.results.append(self.result)

    def _SDMClient_fit(
        self,
        model: Any,
        outputs: Any = None,
        holdout: Any = None,
        **given: Any,
    ) -> None:
        request = given.get('request', self.request)
        self.fitted = self._client().fit(
            request, model=model, outputs=outputs, holdout=holdout
        )

    def _FittedModel_predict(self, **given: Any) -> None:
        self.result = self.fitted.predict(given.get('request', self.request))
        self.results.append(self.result)

    def _FittedModel_close(self) -> None:
        self.fitted.close()

    def _PredictionResult_merge(self, parts: list[int] | None = None) -> None:
        r"""Merge the results kept so far, or those at positions ``parts``."""
        from sdm_client import PredictionResult

        chosen = (
            self.results
            if parts is None
            else [self.results[index] for index in parts]
        )
        self.result = PredictionResult.merge(chosen)

    def _PredictionResult_to_pandas(self) -> None:
        self.frame = self.result.to_pandas()

    def _PredictionResult_absent(self, requested: list[str]) -> None:
        r"""Which of ``requested`` the last result lacks."""
        self.absent = list(self.result.absent(requested))

    def _Request_halves(self) -> None:
        r"""Read both halves of the request made last, as a caller does."""
        self.halves = {
            half: {
                'instance_table': _frame_out(split.instance_table),
                'related_tables': {
                    name: _frame_out(table)
                    for name, table in split.related_tables.items()
                },
            }
            for half, split in (
                ('context', self.request.context),
                ('predict', self.request.predict),
            )
        }

    def _Request_payloads(self) -> None:
        r"""Read the envelopes of the request made last, as a caller does,
        each with its ``metadata.request_id`` written ``<request-id>``."""
        self.payloads = [
            _without_request_id(json.loads(json.dumps(payload)))
            for payload in self.request.payloads
        ]

    def output(self) -> dict[str, Any]:
        out: dict[str, Any] = {'requests': self.session.requests}
        if self.halves is not None:
            out['halves'] = self.halves
        if self.absent is not None:
            out['absent'] = self.absent
        if self.payloads is not None:
            out['payloads'] = self.payloads
        if self.frame is not None:
            out['frame'] = {
                'columns': {
                    str(name): series_out(self.frame[name])
                    for name in self.frame.columns
                }
            }
        if self.result is not None:
            unscored = list(self.result.unscored_candidates)
            if unscored:
                out['unscored_candidates'] = encode(unscored)
            if self.result.explanation is not None:
                out['explanation'] = encode(_text_keys(self.result.explanation))
        return out


def _frame_out(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        'columns': {
            str(name): series_out(frame[name]) for name in frame.columns
        }
    }


def _client_ranked(input: dict[str, Any]) -> Any:
    r"""How a ranking orders its candidates, from the probability of the
    positive class each candidate's question was answered with.

    ``scores`` holds one list per candidate, in the order ``candidates`` names
    them, of the probability for each row asked about. What comes back is each
    row's ``prediction`` and ``rankings``, as the merged answer holds them.
    """
    from sdm_api.generated.tfm_api import PredictionItem
    from sdm_client.client import _ranked
    from sdm_client.decompose import RANKING, Decomposition
    from sdm_client.result import PredictionResult

    candidates = decode(input['candidates'])
    answers = [
        PredictionResult(
            model='kumo-relational',
            request_id='',
            items=[
                PredictionItem(
                    row_index=row,
                    id=str(row),
                    probabilities={'True': decode(score)},
                )
                for row, score in enumerate(scores)
            ],
            outputs=('prediction', 'probabilities'),
        )
        for scores in input['scores']
    ]
    rewritten = Decomposition(
        RANKING,
        tuple((f'part {i}', c) for i, c in enumerate(candidates)),
        input.get('top'),
    )
    merged = _ranked(answers, rewritten, input.get('outputs'))
    return [
        {
            'row_index': item.row_index,
            'prediction': encode(item.prediction),
            'rankings': encode([dict(entry) for entry in item.rankings or ()]),
        }
        for item in merged.items
    ]


def _client_run(input: dict[str, Any], session: Any = None) -> Any:
    r"""Public calls over the input's ``tables``, the model answered with its
    ``responses`` in order; what the last result's frame holds, what else it
    holds, and which requests were sent, with the bodies posted when
    ``show_bodies`` is true. ``session`` replaces the answering
    session (``make_fixtures.py`` records the responses with one)."""
    tables = {name: build_frame(spec) for name, spec in input['tables'].items()}
    if session is None:
        session = Answering(input.get('responses', []))
    flow = Flow(tables, session)
    flow.run(input['calls'])
    out = flow.output()
    if input.get('show_bodies'):
        out['bodies'] = [_without_request_id(body) for body in session.bodies]
    return out


def _without_request_id(body: Any) -> Any:
    r"""A posted body with its ``metadata.request_id``, which differs from
    run to run, set to ``<request-id>``."""
    metadata = body.get('metadata') if isinstance(body, dict) else None
    if isinstance(metadata, dict) and 'request_id' in metadata:
        metadata['request_id'] = '<request-id>'
    return body


OPERATIONS: dict[str, Callable[[dict[str, Any]], Any]] = {
    'dates.day_comes_first': _dates_day_comes_first,
    'dates.looks_unambiguous': _dates_looks_unambiguous,
    'dates.shapes_written': _dates_shapes_written,
    'dates.reads_as_a_date': _dates_reads_as_a_date,
    'dates.conversion_that_works': _dates_conversion_that_works,
    'dates.read': _dates_read,
    'dates.pattern': _dates_pattern,
    'dates.conversion_text': _dates_conversion_text,
    'time_column.read': _time_column_read,
    'wire.infer_dtype': _wire_infer_dtype,
    'wire.serialize_column': _wire_serialize_column,
    'wire.serialize_cell': _wire_serialize_cell,
    'wire.as_declared': _wire_as_declared,
    'wire.json_ready': _wire_json_ready,
    'payload.cell_json_value': _payload_cell_json_value,
    'payload.json_value': _payload_json_value,
    'payload.class_label': _payload_class_label,
    'payload.decimal_dtype_name': _payload_decimal_dtype_name,
    'payload.dtype_name': _payload_dtype_name,
    'payload.target_json_value': _payload_target_json_value,
    'payload.feature_column': _payload_feature_column,
    'composite.encode_values': _composite_encode_values,
    'composite.decode_value': _composite_decode_value,
    'composite.encode_identity': _composite_encode_identity,
    'composite.decode_identity': _composite_decode_identity,
    'composite.encode_frame': _composite_encode_frame,
    'labels.column': _labels_column,
    'labels.rel_op': _labels_rel_op,
    'labels.member_op': _labels_member_op,
    'labels.bool_op': _labels_bool_op,
    'labels.aggregation_type': _labels_aggregation_type,
    'labels.aggregation': _labels_aggregation,
    'sampling.neighbor_sample': _sampling_neighbor_sample,
    'sampling.graph_store': _sampling_graph_store,
    'sampling.date_offset_apply': _sampling_date_offset_apply,
    'sampling.date_offset_to_seconds': _sampling_date_offset_to_seconds,
    'sampling.random_state': _sampling_random_state,
    'sampling.to_list': _sampling_to_list,
    'schema.from_data': _schema_from_data,
    'tabular.task': _tabular_task,
    'tabular.two_whole_numbers': _tabular_two_whole_numbers,
    'decompose.as_literal': _decompose_as_literal,
    'decompose.parts': _decompose_parts,
    'driver.materialize': _driver_materialize,
    'client.run': _client_run,
    'client.ranked': _client_ranked,
}


def run(operation: str, input: Any) -> Any:
    r"""The output of ``operation`` on ``input``, as fixture JSON.

    Raises:
        FixtureError: What the operation raised, as a code and a message.
    """
    try:
        return OPERATIONS[operation](input)
    except FixtureError:
        raise
    except Exception as error:  # noqa: BLE001 - every refusal is an outcome
        raise as_error(error) from error


def same(a: Any, b: Any) -> bool:
    r"""Equal as JSON, type included: ``1`` is not ``1.0`` or ``true``."""
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(map(same, a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b
