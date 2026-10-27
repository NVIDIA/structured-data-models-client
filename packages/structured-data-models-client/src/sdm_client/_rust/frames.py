# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Bridge code: native conversion and public objects remain Python's;
# the routing seam is deleted in Phase 5 after the complete frame switch.

"""Closed native-family constructors, using fixed core Table/Schema stages."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from sdm_client._routing import RouteError
from sdm_client._rust import codec, errors, results
from sdm_client._rust import schema as schemas
from sdm_client._rust.run import Finished, run
from sdm_client.core.warnings import SDMWarning, caller_stacklevel
from sdm_client.errors import StructuredDataError

if TYPE_CHECKING:
    from sdm_client.schema import Relationship, Schema, Table


# Only successful Schema-stage final notices wait for the native pandas hint.
# Table/override/key/link diagnostics retain their immediate boundaries.
_FINAL_NOTICE_CODES = frozenset(
    {
        'FRAME_SCHEMA',
        'KEY_DECLINED',
        'TIME_COLUMNS_GUESSED',
        'TWO_WAYS_TO_A_TABLE',
    }
)


def _frame_example(
    fields: dict[str, Any], native_types: Mapping[str, Mapping[str, Any]]
) -> str:
    # Primitive scalar spelling is native presentation, including NumPy's
    # own version-dependent repr. The core supplies selected typed values.
    columns = fields['columns']
    types = native_types[fields['table']]
    # Let the actual native library traverse a selected composite scalar.
    # A mixed numeric pandas row can widen its components; typed Arrow
    # scalar repr alone would not reproduce that presentation.
    pieces = []
    for name, value in zip(columns, fields['example'], strict=True):
        dtype = types[name]
        if type(dtype) in (pd.Float32Dtype, pd.Float64Dtype):
            # The core selected a present scalar. Construct its native
            # representation without turning a valid NaN into pd.NA.
            native = pd.arrays.FloatingArray(
                np.array([value], dtype=dtype.numpy_dtype),
                np.zeros(1, dtype=bool),
            )
            pieces.append(pd.Series(native))
        else:
            pieces.append(pd.Series([value], dtype=dtype))
    selected = pd.concat(pieces, axis=1)
    selected.columns = columns
    example = (
        selected[columns[0]].iloc[0]
        if len(columns) == 1
        else tuple(selected.iloc[0])
    )
    return repr(example)


def _frame_key(fields: dict[str, Any]) -> str:
    from sdm_client.schema import _the_key_of

    return _the_key_of(fields['table'], fields['columns'])


def _frame_key_candidate(fields: dict[str, Any]) -> str:
    candidates = fields['candidates']
    return repr(candidates[0]) if len(candidates) == 1 else '...'


def _frame_timestamp(
    fields: dict[str, Any], native_types: Mapping[str, Mapping[str, Any]]
) -> str:
    # Only presentation of the core-selected scalar, in its original native
    # timezone implementation and unit. No original frame values are read.
    value = fields['value']
    dtype = native_types[fields['table']][fields['column']]
    if type(dtype) is pd.DatetimeTZDtype:
        stamp = pd.Timestamp(value.value, unit=value.unit, tz='UTC').tz_convert(
            dtype.tz
        )
        unit = dtype.unit
    else:
        stamp = pd.Timestamp(value.value, unit=value.unit)
        unit = np.datetime_data(dtype)[0]
    return str(stamp.as_unit(unit))


def _presentation(
    native_types: Mapping[str, Mapping[str, Any]],
) -> AbstractContextManager[None]:
    return errors.holding(
        frame_example=lambda fields: _frame_example(fields, native_types),
        frame_key=_frame_key,
        frame_key_candidate=_frame_key_candidate,
        frame_timestamp=lambda fields: _frame_timestamp(fields, native_types),
    )


def _show_held(notices: list[dict[str, Any]]) -> None:
    if notices:
        errors.warn_from_core(
            codec.encode({'message': 'warnings', 'warnings': notices}),
            {},
            presentation='frame_schema',
        )


def _final_notices(
    schema: Schema,
    notices: list[dict[str, Any]],
    native_types: Mapping[str, Mapping[str, Any]],
) -> None:
    # Owner-approved host-only diagnostic exception (2026-10-26): this
    # existing pandas hint never changes the core's result or its inputs.
    from sdm_client.schema import (
        _unlinkable_table_reads,
        _warn_dates_stored_as_text,
    )

    dates = {}
    for name, table in schema.tables.items():
        if (
            column := _warn_dates_stored_as_text(cast('Table', table))
        ) is not None:
            dates[name] = column
    pending = []
    with _presentation(native_types):
        for notice in notices:
            if notice['sentence'] != 'frame_keyless_shared':
                pending.append(notice)
                continue
            fields = errors.decoded(notice.get('fields', {}))
            name = fields['table']
            if dates.get(name) not in fields['candidates']:
                pending.append(notice)
                continue
            # The core selected the candidates. Remove only the column the
            # native date advice already named, to avoid contradictory advice.
            # This is presentation only; no fields return to the core.
            _show_held(pending)
            pending.clear()
            candidates = [c for c in fields['candidates'] if c != dates[name]]
            if candidates:
                elsewhere = {
                    other
                    for other, table in schema.tables.items()
                    if other != name
                    and any(c in table.columns for c in candidates)
                }
                warnings.warn(
                    _unlinkable_table_reads(name, candidates, len(elsewhere)),
                    category=SDMWarning,
                    stacklevel=caller_stacklevel(),
                )
        _show_held(pending)


def _convert(name: str, frame: pd.DataFrame) -> tuple[Any, dict[str, Any]]:
    # Validate one table before proceeding to the next. No preliminary
    # all-table scan or native-object conversion is permitted here.
    if type(frame.columns) not in (pd.Index, pd.RangeIndex) or any(
        type(column) not in (str, int, float, bool, type(None))
        for column in frame.columns
    ):
        raise RouteError(
            'the partial frame constructor requires ordinary built-in column names'
        )
    if frame.columns.has_duplicates:
        duplicates = sorted(
            {str(c) for c in frame.columns[frame.columns.duplicated()]}
        )
        raise StructuredDataError(
            f'table {name!r} has duplicate column name(s) {duplicates}',
            code='INVALID_REQUEST',
        )
    if any(isinstance(c, str) and type(c) is not str for c in frame.columns):
        raise RouteError(
            'the partial frame constructor requires ordinary built-in column names'
        )
    non_strings = [c for c in frame.columns if not isinstance(c, str)]
    if non_strings:
        raise StructuredDataError(
            f'table {name!r} column names must be strings; got {non_strings}',
            code='INVALID_REQUEST',
        )
    import pyarrow as pa

    arrays = []
    native_types = {}
    for column in frame.columns:
        series = frame[column]
        dtype = series.dtype
        primitive = (
            isinstance(dtype, np.dtype)
            and dtype.isnative
            and (
                (dtype.kind in 'biu' and dtype.itemsize <= 8)
                or (dtype.kind == 'f' and dtype.itemsize in (4, 8))
            )
        )
        masked = type(dtype) in (
            pd.Int8Dtype,
            pd.Int16Dtype,
            pd.Int32Dtype,
            pd.Int64Dtype,
            pd.UInt8Dtype,
            pd.UInt16Dtype,
            pd.UInt32Dtype,
            pd.UInt64Dtype,
            pd.Float32Dtype,
            pd.Float64Dtype,
            pd.BooleanDtype,
        )
        temporal = (
            isinstance(dtype, np.dtype)
            and dtype.isnative
            and dtype.kind == 'M'
            and np.datetime_data(dtype)[0] in ('s', 'ms', 'us', 'ns')
        ) or (
            type(dtype) is pd.DatetimeTZDtype
            and dtype.unit in ('s', 'ms', 'us', 'ns')
        )
        text = type(dtype) is pd.StringDtype and dtype.storage in (
            'python',
            'pyarrow',
        )
        if not (primitive or masked or temporal or text):
            raise RouteError(
                f'the partial frame constructor supports only NumPy or pandas masked numeric and Boolean columns, native datetimes, or built-in pandas strings: {name}.{column}'
            )
        # Convert values only. DataFrame export may add a non-Range index
        # column, which the public schema constructor never describes.
        # Native conversion supplies validity: NumPy NaNs become null;
        # a masked float's unmasked NaN remains a present Arrow value.
        arrays.append(pa.array(series, from_pandas=True))
        native_types[column] = dtype
    return pa.Table.from_arrays(arrays, names=list(frame.columns)), native_types


def _describe(
    stage: str,
    frames: Mapping[str, Any],
    spec: dict[str, Any],
    native_types: Mapping[str, Mapping[str, Any]],
    *,
    key_forms: dict[str, str] | None = None,
    relationships: list[dict[str, Any]] | None = None,
    held_notices: list[dict[str, Any]] | None = None,
) -> Finished:
    manifest: dict[str, Any] = {
        'stage': stage,
        'tables': [],
        'key_forms': key_forms or {},
        'override_order': {
            'primary_keys': list(spec['primary_keys']),
            'time_columns': list(spec['time_columns']),
        },
    }
    if relationships is not None:
        manifest['relationships'] = relationships
    rows: Any = None
    attached: dict[int, Any] = {1: codec.attachment(spec)}
    try:
        for ordinal, (name, rows) in enumerate(frames.items(), 3):
            manifest['tables'].append(
                {'source': codec.relation(name), 'rows': ordinal}
            )
            attached[ordinal] = rows
        attached[2] = codec.attachment(manifest)
        with _presentation(native_types):
            return run(
                {
                    'op': 'describe_source',
                    'provider': {'provider': 'frames', 'manifest': 2},
                    'source': 1,
                    'check_keys': True,
                },
                attached,
                quiet=_FINAL_NOTICE_CODES
                if held_notices is not None
                else frozenset(),
                kept=held_notices,
            )
    finally:
        rows = None
        attached.clear()


def _ordinary_builtin(value: Any, seen: set[int]) -> bool:
    # Only native representation facts are captured here. Never run repr
    # on a container until all nested values are known ordinary built-ins.
    kind = type(value)
    if kind in (
        type(None),
        bool,
        int,
        float,
        complex,
        str,
        bytes,
        bytearray,
        range,
        object,
    ):
        return True
    if kind is slice:
        return all(
            _ordinary_builtin(item, seen)
            for item in (value.start, value.stop, value.step)
        )
    if kind not in (list, tuple, dict, set, frozenset):
        return False
    if id(value) in seen:
        return True
    seen.add(id(value))
    if kind is dict:
        return all(
            _ordinary_builtin(key, seen) and _ordinary_builtin(item, seen)
            for key, item in value.items()
        )
    return all(_ordinary_builtin(item, seen) for item in value)


def from_data(
    cls: type[Schema],
    tables: Mapping[str, pd.DataFrame],
    *,
    primary_keys: Mapping[str, Any] | None = None,
    time_columns: Mapping[str, Any] | None = None,
    relationships: Sequence[Relationship] | None = None,
) -> Schema:
    from sdm_client.schema import Relationship

    if type(tables) is not dict:
        raise RouteError(
            'the partial frame constructor requires ordinary built-in table mappings'
        )
    if not tables:
        raise StructuredDataError(
            'a schema needs at least one table', code='INVALID_REQUEST'
        )
    snapshots: dict[str, Any] = {}
    originals: dict[str, pd.DataFrame] = {}
    native_types: dict[str, dict[str, Any]] = {}
    finished: Finished | None = None
    converted: Any = None
    held_notices: list[dict[str, Any]] = []
    try:
        for name, frame in tables.items():
            if isinstance(name, str) and type(name) is not str:
                raise RouteError(
                    'the partial frame constructor requires ordinary built-in table names'
                )
            if not isinstance(name, str) or not name:
                raise StructuredDataError(
                    f'table names must be non-empty strings; got {name!r}',
                    code='INVALID_REQUEST',
                )
            if not isinstance(frame, pd.DataFrame):
                raise StructuredDataError(
                    f'table {name!r} must be a pandas DataFrame, got {type(frame).__name__}',
                    code='INVALID_REQUEST',
                )
            if type(frame) is not pd.DataFrame:
                raise RouteError(
                    'the partial frame constructor requires ordinary built-in DataFrames'
                )
            converted, native_types[name] = _convert(name, frame)
            spec: dict[str, Any] = {
                'tables': [{'name': name, 'source': codec.relation(name)}],
                'primary_keys': {},
                'time_columns': {},
            }
            finished = _describe('table', {name: converted}, spec, native_types)
            result = results.expect(finished.result, results.Described)
            if result.frame_rows is None:
                raise StructuredDataError(
                    'Table stage returned no owned frame', code='INTERNAL_ERROR'
                )
            snapshots[name] = finished.take(result.frame_rows)
            originals[name] = frame
            finished = None
            converted = None
        # Native argument conversion is deliberately after all Table stages,
        # matching the constructor's override boundary.
        for option in (primary_keys, time_columns):
            if option is not None and type(option) is not dict:
                raise RouteError(
                    'the partial frame constructor requires ordinary built-in override mappings'
                )
            if option is not None and any(
                type(name) is not str for name in option
            ):
                raise RouteError(
                    'the partial frame constructor requires ordinary built-in override names'
                )
        for value in (primary_keys or {}).values():
            if not (
                value is None
                or type(value) is str
                or (
                    type(value) in (list, tuple)
                    and all(type(name) is str for name in value)
                )
            ):
                raise RouteError(
                    'the partial frame constructor requires ordinary built-in primary keys'
                )
        if any(
            value is not None and type(value) is not str
            for value in (time_columns or {}).values()
        ):
            raise RouteError(
                'the partial frame constructor requires ordinary built-in time column names'
            )
        if relationships is not None and type(relationships) not in (
            list,
            tuple,
        ):
            raise RouteError(
                'the partial frame constructor requires ordinary built-in relationship sequences'
            )
        spec = {
            'tables': [
                {'name': name, 'source': codec.relation(name)}
                for name in snapshots
            ],
            'primary_keys': {
                name: codec.given_key(value)[0]
                for name, value in (primary_keys or {}).items()
            },
            'time_columns': {
                name: codec.given_name_or_none(value)
                for name, value in (time_columns or {}).items()
            },
        }
        slots = None
        if relationships is not None:
            slots = []
            for link in relationships:
                if type(link) is Relationship:
                    if any(
                        type(value) is not str
                        for value in (
                            link.source_table,
                            link.source_column,
                            link.target_table,
                            link.target_column,
                        )
                    ):
                        raise RouteError(
                            'the partial frame constructor requires ordinary built-in relationship names'
                        )
                    slots.append(
                        {
                            'kind': 'link',
                            'source': {
                                'table': link.source_table,
                                'column': link.source_column,
                            },
                            'target': {
                                'table': link.target_table,
                                'column': link.target_column,
                            },
                        }
                    )
                elif _ordinary_builtin(link, set()):
                    slots.append(
                        {
                            'kind': 'invalid',
                            'native_type': type(link).__name__,
                            'native_repr': repr(link),
                        }
                    )
                else:
                    raise RouteError(
                        'the partial frame constructor requires ordinary built-in relationship arguments'
                    )
        try:
            finished = _describe(
                'schema',
                snapshots,
                spec,
                native_types,
                key_forms={
                    name: 'scalar' if type(value) is str else 'sequence'
                    for name, value in (primary_keys or {}).items()
                },
                relationships=slots,
                held_notices=held_notices,
            )
        except StructuredDataError:
            # Preserve preceding notices on a real core refusal. A warning
            # filter or presentation failure must propagate immediately:
            # held final advice has not been reached at that boundary.
            with _presentation(native_types):
                _show_held(held_notices)
            raise
        result = results.expect(finished.result, results.Described)
        if result.frame_rows is not None:
            raise StructuredDataError(
                'Schema stage returned a table snapshot', code='INTERNAL_ERROR'
            )
        described = json.loads(finished.take(result.description))
        finished = None
        schema = schemas.build(cls, None, described, frames=originals)
        _final_notices(schema, held_notices, native_types)
        return schema
    finally:
        # Tracebacks can retain these frames. Clear only call-owned snapshots;
        # the public tables retain their original DataFrames by reference.
        finished = None
        converted = None
        snapshots.clear()
        native_types.clear()
        held_notices.clear()
