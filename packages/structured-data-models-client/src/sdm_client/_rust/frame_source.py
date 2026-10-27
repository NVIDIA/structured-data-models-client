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
from sdm_client._rust.spec import held_column


def meaning(
    table: Any, name: str, column: Any, values: pd.Series, state: dict[str, Any]
) -> dict[str, Any]:
    from sdm_client._rust.frame_facade import _Bindings, _adopt, _prepare

    dtype = values.dtype
    object_dtype = pd.api.types.is_object_dtype(dtype)
    string_dtype = pd.api.types.is_string_dtype(dtype)
    inferred = (
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
                    'table': table.name,
                    'source': codec.relation(
                        getattr(table, 'source_name', table.name)
                    ),
                    'column': held_column(name, column),
                    'designated_time': name == table.time_column,
                    'facts': {
                        'object': object_dtype,
                        'string': string_dtype,
                        'arrow_decimal': native_decimal_type(dtype) is not None,
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

    from sdm_client._rust.frame_facade import _Bindings, _adopt, _prepare
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
    from sdm_client._rust.frame_facade import _Bindings, _adopt, _prepare

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


def time_feature(
    table: Any, name: str, values: pd.Series, state: dict[str, Any]
) -> pd.Series:
    """Engine feature-time reading follows every earlier native source cast."""
    column = table.columns.get(name)
    if column is None:
        return values
    resolved = meaning(table, name, column, values, state)
    return (
        _time(values, resolved, state)
        if resolved['kind'] == 'text_feature_time'
        else values
    )
