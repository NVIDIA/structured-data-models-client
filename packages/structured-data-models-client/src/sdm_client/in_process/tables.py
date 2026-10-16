# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Reading a wire table back as the types its schema declares."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any

import pyarrow as pa

from sdm_client.in_process.refusal import refusal

_ARROW_TYPES = {
    'bool': pa.bool_(),
    'int32': pa.int32(),
    'int64': pa.int64(),
    'float32': pa.float32(),
    'float64': pa.float64(),
    'string': pa.string(),
}
_INTEGERS = frozenset({'int32', 'int64'})
_NUMBERS = _INTEGERS | {'float32', 'float64'}


def values(table: Mapping[str, Any], column: str) -> list[Any]:
    r"""One column of an ``arrays`` table, in row order."""
    position = table['columns'].index(column)
    return [row[position] for row in table['rows']]


def read_cell(cell: Any, dtype: str | None) -> Any:
    r"""A cell as the value it stands for: an integer wider than a double
    travels as its base-10 digits."""
    if dtype in _INTEGERS and isinstance(cell, str):
        return int(cell)
    return cell


def to_arrow(
    table: Mapping[str, Any],
    columns: Sequence[str],
    dtypes: Mapping[str, str],
    keys: Collection[str] = (),
) -> pa.Table:
    r"""``columns`` of ``table`` as Arrow, each typed as ``dtypes`` declares.

    A numeric key is read as ``int64`` whatever its declared width, so both
    sides of a join agree. A fractional one is refused rather than truncated
    onto another row.
    """
    arrays = []
    for column in columns:
        cells = values(table, column)
        dtype = dtypes[column]
        if column in keys and dtype in _NUMBERS:
            arrays.append(pa.array(_whole(cells, dtype, column), pa.int64()))
        else:
            arrays.append(_array(cells, dtype, column))
    return pa.Table.from_arrays(arrays, names=list(columns))


def _array(cells: list[Any], dtype: str, column: str) -> pa.Array:
    try:
        if dtype == 'timestamp[us]':
            zoned = pa.array(cells, pa.string()).cast(
                pa.timestamp('us', tz='UTC')
            )
            return zoned.cast(pa.timestamp('us'))
        return pa.array(
            [read_cell(cell, dtype) for cell in cells], _ARROW_TYPES[dtype]
        )
    except (KeyError, pa.ArrowException, TypeError, ValueError) as error:
        raise refusal(
            'VALIDATION_FAILED',
            f'column {column!r} cannot be read as {dtype}',
        ) from error


def _whole(cells: list[Any], dtype: str, column: str) -> list[Any]:
    whole = []
    for cell in cells:
        value = read_cell(cell, dtype)
        if isinstance(value, float):
            if not value.is_integer():
                raise refusal(
                    'VALIDATION_FAILED',
                    f'key column {column!r} holds the fractional value '
                    f'{value!r}, which joins to no row',
                )
            value = int(value)
        whole.append(value)
    return whole
