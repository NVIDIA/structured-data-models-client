# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reached native boxing and dtype ownership for one live frame column.

The core resolves each column's current existing declaration independently.
Native pandas operations retain their original order and callback effects;
no future column declaration or immutable source snapshot is captured early.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from sdm_client._rust import codec
from sdm_client._rust.frame_decimals import native_decimal_type
from sdm_client._rust.frame_facts import native_declaration, observed_content
from sdm_client._rust.spec import held_column


def meaning(
    table: Any, name: str, column: Any, values: pd.Series, state: dict[str, Any]
) -> dict[str, Any]:
    import pyarrow as pa

    from sdm_client._rust.frame_facade import _adopt, _Bindings, _prepare

    dtype = values.dtype
    object_dtype = pd.api.types.is_object_dtype(dtype)
    string_dtype = pd.api.types.is_string_dtype(dtype)
    inferred = observed_content(
        pd.api.types.infer_dtype(values, skipna=True)
        if object_dtype or string_dtype
        else None
    )
    bindings = _Bindings()
    product = _adopt(
        _prepare(
            {
                'stage': 'column_meaning',
                'input': {
                    'table': table['name']
                    if type(table) is dict
                    else table.name,
                    'source': table['source']
                    if type(table) is dict
                    else codec.relation(
                        getattr(table, 'source_name', table.name)
                    ),
                    'column': column
                    if type(column) is dict
                    else held_column(name, column),
                    'designated_time': name
                    == (
                        table['time_column']
                        if type(table) is dict
                        else table.time_column
                    ),
                    'facts': {
                        'object': object_dtype,
                        'string': string_dtype,
                        'arrow_decimal': native_decimal_type(dtype) is not None,
                        'timestamp': pd.api.types.is_datetime64_any_dtype(dtype)
                        or type(dtype) is pd.ArrowDtype
                        and pa.types.is_timestamp(dtype.pyarrow_dtype),
                        'inferred': inferred,
                    },
                },
            },
            bindings,
            state,
        ),
        'column_meaning',
    )
    return product.data


def _time(
    values: pd.Series, resolved: dict[str, Any], state: dict[str, Any]
) -> pd.Series:
    import pyarrow as pa

    from sdm_client._rust.frame_facade import _adopt, _Bindings, _prepare
    from sdm_client._rust.frame_native import NativeColumn

    rows, encoding = NativeColumn(values).arrow()
    bindings = _Bindings()
    product = _adopt(
        _prepare(
            {
                'stage': 'time_column',
                'meaning': resolved,
                'values': {'values': bindings.add(rows), 'encoding': encoding},
            },
            bindings,
            state,
        ),
        'source_column',
    )
    try:
        array = pa.table(product.owners[product.data['values']]).column(0)
        # Arrow's whole-column result owns the actual typed values. Supplying
        # its array rather than a Series prevents native index realignment.
        return pd.Series(
            array.to_pandas().array, index=values.index, name=values.name
        )
    finally:
        product.owners.clear()


def _decimal(
    values: pd.Series, resolved: dict[str, Any], state: dict[str, Any]
) -> pd.Series:
    from sdm_client._rust.frame_facade import _adopt, _Bindings, _prepare

    kind = resolved['kind']
    if kind == 'decimal_float':
        return pd.to_numeric(values.astype(object), errors='coerce').astype(
            'float64'
        )
    if kind == 'decimal_string':
        return values.map(lambda value: None if pd.isna(value) else str(value))
    # The two complete native maps are separate original observation points.
    # In particular, pandas' actual uint64-to-int64 cast may wrap here.
    fractional = values.map(
        lambda value: not pd.isna(value) and value != int(value)
    )
    any_fractional = bool(fractional.any())
    example = repr(values[fractional].iloc[0]) if any_fractional else None
    bindings = _Bindings()
    _adopt(
        _prepare(
            {
                'stage': 'integer_column',
                'input': {
                    'meaning': resolved,
                    'fractional': any_fractional,
                    'example': example,
                },
            },
            bindings,
            state,
        ),
        'integer_column',
    )
    whole = values.map(lambda value: pd.NA if pd.isna(value) else int(value))
    return whole.astype('Int64' if whole.isna().any() else 'int64')


def source_meaning(
    schema: dict[str, Any],
    typed_tables: list[dict[str, Any]],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Bind final graph metadata to the tables' frozen dtype facts."""
    from sdm_client._rust.frame_facade import _adopt, _Bindings, _prepare

    bindings = _Bindings()
    result = _adopt(
        _prepare(
            {
                'stage': 'source_meaning',
                'schema': bindings.json(schema),
                'typed_tables': typed_tables,
            },
            bindings,
            state,
        ),
        'source_meaning',
    )
    return result.data['schema']


def table_meaning(
    table: dict[str, Any],
    frame: pd.DataFrame,
    state: dict[str, Any],
    *,
    empty: bool,
) -> dict[str, Any]:
    """Admit physical table types after all declared native casts."""
    from sdm_client._rust.frame_constructor import column as constructor_column
    from sdm_client._rust.frame_facade import _adopt, _Bindings, _prepare

    bindings = _Bindings()

    native = []
    if not empty:
        # The original physical table caches this one actual sample owner.
        # Dtype and stype inference, including later columns, share it.
        sample = frame.reset_index(drop=True)
        for name in frame.columns:
            values = sample[name]
            dtype = values.dtype
            facts = constructor_values(values)
            declared = native_declaration(dtype)
            native.append(
                {
                    'name': name,
                    'declared': declared,
                    'values': facts,
                    'meaning': constructor_column(
                        table['name'], name, sample, facts, declared
                    ),
                }
            )
    result = _adopt(
        _prepare(
            {
                'stage': 'table_meaning',
                'input': {
                    'table': table,
                    'native': native,
                    'empty': empty,
                },
            },
            bindings,
            state,
        ),
        'table_meaning',
    )
    return result.data


def constructor_values(values: pd.Series) -> dict[str, Any]:
    """Actual fixed constructor storage/content, before caller key metadata."""
    dtype = values.dtype
    if type(dtype) is pd.CategoricalDtype:
        return {'storage': 'enumerated'}
    if pd.api.types.is_object_dtype(dtype) and type(dtype) is not pd.ArrowDtype:
        sample = values.iloc[:1000]
        # Keep the original first-valid lookup before native scalar inference.
        index = sample.first_valid_index()
        if index is not None:
            pd.api.types.is_list_like(values[index])
        observed = pd.api.types.infer_dtype(sample, skipna=True)
        content = {
            'integer': 'integer',
            'floating': 'real',
            'mixed-integer-float': 'real',
            'decimal': 'real',
            'boolean': 'boolean',
            'timedelta': 'duration',
        }.get(observed, 'other')
        return {'storage': 'elements', 'content': content}
    declaration = native_declaration(dtype)
    if declaration is not None and declaration['declaration'] == 'physical':
        return {'storage': 'typed', 'physical': declaration['physical']}
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return {'storage': 'temporal'}
    if pd.api.types.is_timedelta64_dtype(dtype):
        return {'storage': 'duration'}
    if pd.api.types.is_bool_dtype(dtype):
        physical = {'type': 'bool'}
    elif pd.api.types.is_integer_dtype(dtype):
        width = getattr(dtype, 'itemsize') * 8
        prefix = (
            'uint' if pd.api.types.is_unsigned_integer_dtype(dtype) else 'int'
        )
        physical = {'type': prefix + str(width)}
    elif pd.api.types.is_float_dtype(dtype):
        physical = {'type': 'float' + str(getattr(dtype, 'itemsize') * 8)}
    else:
        # String values can contain native surrogate points, independently
        # of the later Arrow representation selected for their export.
        return {'storage': 'text'}
    return {'storage': 'typed', 'physical': physical}


def table_key(
    table: dict[str, Any], primary_key: list[str], state: dict[str, Any]
) -> dict[str, Any]:
    from sdm_client._rust.frame_facade import _adopt, _Bindings, _prepare

    return _adopt(
        _prepare(
            {'stage': 'table_key', 'table': table, 'primary_key': primary_key},
            _Bindings(),
            state,
        ),
        'key_meaning',
    ).data


def table_metadata(
    key: dict[str, Any],
    table: dict[str, Any],
    state: dict[str, Any],
    derived: dict[str, Any] | None,
) -> dict[str, Any]:
    from sdm_client._rust.frame_facade import _adopt, _Bindings, _prepare

    return _adopt(
        _prepare(
            {
                'stage': 'table_metadata',
                'input': {'key': key, 'table': table, 'derived': derived},
            },
            _Bindings(),
            state,
        ),
        'table_meaning',
    ).data


def typed_table(table: Any, state: dict[str, Any]) -> pd.DataFrame:
    """Native casts in the original schema-column iteration ownership."""
    frame = table.data
    if frame.empty:
        return frame
    cast = {}
    for name, column in table.columns.items():
        if name not in frame.columns:
            continue
        values = frame[name]
        resolved = meaning(table, name, column, values, state)
        kind = resolved['kind']
        if kind in ('decimal_float', 'decimal_integer', 'decimal_string'):
            cast[name] = _decimal(values, resolved, state)
        elif kind == 'text_time':
            cast[name] = _time(values, resolved, state)
        elif kind == 'native_time':
            cast[name] = pd.to_datetime(values, utc=True)
    return frame.assign(**cast) if cast else frame


def _typed_timestamp(values: pd.Series) -> pd.Series:
    """Actual whole native boxing, timezone and unit casts at sanitization.

    The core selected this for an actual typed timestamp source. String
    source formats use the fixed Reader at a different boundary.
    """
    if isinstance(values.dtype, pd.ArrowDtype):
        values = pd.Series(
            values.to_numpy(), index=values.index, name=values.name
        )
    if not pd.api.types.is_datetime64_any_dtype(values):
        # A native typed timestamp can box to real datetime objects. This
        # is their original native unit conversion, not string admission.
        values = pd.to_datetime(values, unit='ns', errors='coerce')
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        values = values.dt.tz_convert('UTC').dt.tz_localize(None)
    if values.dtype != 'datetime64[ns]':
        values = values.astype('datetime64[ns]')
    return values


def time_feature(
    table: Any, name: str, values: pd.Series, state: dict[str, Any]
) -> pd.Series:
    """Engine feature-time reading follows every earlier native source cast."""
    column = next(
        (column for column in table['columns'] if column['name'] == name), None
    )
    if column is None:
        return values
    resolved = meaning(table, name, column, values, state)
    if resolved['kind'] == 'text_feature_time':
        return _time(values, resolved, state)
    if resolved['kind'] == 'typed_time':
        return _typed_timestamp(values)
    return values
