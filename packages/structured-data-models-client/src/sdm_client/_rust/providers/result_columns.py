# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode the SQL data layer's explicit timestamp transport into native Arrow.

The descriptor supplies the position, label and timestamp meaning.
Both instant and wall-time microsecond counts retain their existing
timezone-free Arrow representation, without interpreting a session timezone.
Other port conversions remain with their concrete reader.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pyarrow as pa

_FIELDS = frozenset({'conversion', 'column', 'name', 'source_kind'})


def result_columns(
    table: pa.Table, conversions: Sequence[Mapping[str, object]]
) -> pa.Table:
    """Restore only explicitly named signed BIGINT microsecond columns.

    Validate every descriptor against the actual unmodified result before
    casting any column. Unconverted columns, labels, nullability and field
    and schema metadata remain the original result's.
    """
    positions = []
    seen = set()
    for conversion in conversions:
        if (
            not isinstance(conversion, Mapping)
            or set(conversion) != _FIELDS
            or type(conversion['conversion']) is not str
            or conversion['conversion'] != 'timestamp_micros'
            or type(conversion['source_kind']) is not str
            or conversion['source_kind'] not in ('wall_time', 'instant')
        ):
            raise ValueError('an unsupported SQL result conversion')
        column = conversion['column']
        name = conversion['name']
        if (
            type(column) is not int
            or not 0 <= column < table.num_columns
            or column in seen
            or type(name) is not str
            or name != table.column_names[column]
            or table.column(column).type != pa.int64()
        ):
            raise ValueError(
                'a SQL result conversion does not match its output column'
            )
        seen.add(column)
        positions.append(column)

    restored = table
    for column in positions:
        field = table.schema.field(column).with_type(pa.timestamp('us'))
        values = table.column(column).cast(field.type, safe=True)
        restored = restored.set_column(column, field, values)
    return restored
