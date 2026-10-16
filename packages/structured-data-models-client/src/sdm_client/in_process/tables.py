# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Reading a wire table back as the types its schema declares."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from sdm_client.in_process.refusal import refusal

if TYPE_CHECKING:
    import pyarrow as pa

#: Each dtype's Arrow type, by factory name. PyArrow comes with the extra and
#: is imported where it is used, so that a client without the extra reaches
#: the check that names it.
_ARROW_TYPES = {
    'bool': 'bool_',
    'int32': 'int32',
    'int64': 'int64',
    'float32': 'float32',
    'float64': 'float64',
    'string': 'string',
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
    import pyarrow as pa

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
    import pyarrow as pa

    try:
        if dtype == 'timestamp[us]':
            zoned = pa.array(cells, pa.string()).cast(
                pa.timestamp('us', tz='UTC')
            )
            return zoned.cast(pa.timestamp('us'))
        arrow_type = getattr(pa, _ARROW_TYPES[dtype])()
        return pa.array([read_cell(cell, dtype) for cell in cells], arrow_type)
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
                    f'identifier column {column!r} holds a fractional value',
                )
            value = int(value)
        whole.append(value)
    return whole
