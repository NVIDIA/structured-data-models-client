# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extend original recorded Arrow comparison only for ordered map arrays."""

from __future__ import annotations

from types import SimpleNamespace


def request_decoder(request):
    import pyarrow as arrow

    driver = request.driver_module()

    def has_map(dtype):
        if arrow.types.is_map(dtype):
            return True
        if arrow.types.is_dictionary(dtype):
            return has_map(dtype.value_type)
        if driver.type_name(dtype).startswith('list<'):
            return has_map(dtype.value_type)
        if arrow.types.is_struct(dtype):
            return any(has_map(field.type) for field in dtype)
        return False

    def values(array):
        array = driver._plain(array)
        dtype = array.type
        if not has_map(dtype):
            return driver._values(array)
        if arrow.types.is_map(dtype):
            result = []
            for item in array:
                if not item.is_valid:
                    result.append(None)
                else:
                    entries = item.values
                    keys = values(entries.field(0))
                    items = values(entries.field(1))
                    result.append(
                        [
                            [key, value]
                            for key, value in zip(keys, items, strict=True)
                        ]
                    )
            return result
        if driver.type_name(dtype).startswith('list<'):
            return [
                values(item.values) if item.is_valid else None for item in array
            ]
        columns = [
            values(array.field(index)) for index in range(dtype.num_fields)
        ]
        valid = array.is_valid().to_pylist()
        return [
            {
                field.name: columns[index][row]
                for index, field in enumerate(dtype)
            }
            if valid[row]
            else None
            for row in range(len(array))
        ]

    class MapEncoder(request.Encoder):
        def _arrow(self, value, tag):
            if isinstance(value, arrow.RecordBatch):
                table = arrow.Table.from_batches([value])
            elif isinstance(value, (arrow.Array, arrow.ChunkedArray)):
                table = arrow.table({'values': value})
            else:
                table = value
            if not any(has_map(field.type) for field in table.schema):
                return super()._arrow(value, tag)
            return {
                tag: {
                    'names': list(table.column_names),
                    'columns': [
                        {
                            'type': driver.type_name(
                                driver._plain(column).type
                            ),
                            'values': values(column),
                        }
                        for column in table.columns
                    ],
                }
            }

    def canonical_recorded(value, root):
        return MapEncoder(None).value(request.decode(value, root))

    return SimpleNamespace(
        canonical_recorded=canonical_recorded, encoder=MapEncoder
    )
