#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Share repeated fixture fields without computing any expected answers.

Each table has named fields, one dictionary of literal values per field,
and ordered rows of dictionary indexes. A -1 index denotes an absent field.
Only arrays of objects are factored; scalar arrays and original metadata
retain their ordinary JSON shapes. The decoder accepts ordinary documents
so recorders can still produce their original format before extraction.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

FORMAT = 'sdm-shared-fixture/1'


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=True
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def pack(value: Any) -> dict[str, Any]:
    tables: list[dict[str, Any]] = []

    def visit(node: Any) -> Any:
        if (
            isinstance(node, list)
            and len(node) >= 8
            and all(isinstance(v, dict) for v in node)
        ):
            fields = sorted({k for row in node for k in row})
            columns: list[list[Any]] = [[] for _ in fields]
            indexes: list[dict[bytes, int]] = [{} for _ in fields]
            rows = []
            for row in node:
                cells = []
                for i, field in enumerate(fields):
                    if field not in row:
                        cells.append(-1)
                        continue
                    key = canonical(row[field])
                    if key not in indexes[i]:
                        indexes[i][key] = len(columns[i])
                        columns[i].append(row[field])
                    cells.append(indexes[i][key])
                rows.append(cells)
            # Children are visited first, giving the table a topological order.
            columns = [[visit(v) for v in column] for column in columns]
            index = len(tables)
            tables.append({'fields': fields, 'values': columns, 'rows': rows})
            return {'$fixture_table': index}
        if isinstance(node, dict):
            if '$fixture_table' in node:
                raise ValueError('reserved fixture table key in source')
            return {k: visit(v) for k, v in node.items()}
        if isinstance(node, list):
            return [visit(v) for v in node]
        return node

    document = visit(value)
    return {
        'format': FORMAT,
        'expanded_sha256': digest(value),
        'document': document,
        'tables': tables,
    }


def unpack(value: Any) -> Any:
    if not isinstance(value, dict) or value.get('format') != FORMAT:
        return value
    tables = value['tables']

    def visit(node: Any, before: int) -> Any:
        if isinstance(node, dict) and '$fixture_table' in node:
            index = node['$fixture_table']
            if (
                type(index) is not int
                or not 0 <= index < before
                or len(node) != 1
            ):
                raise ValueError('invalid or cyclic fixture table reference')
            table = tables[index]
            fields, values = table['fields'], table['values']
            if len(fields) != len(values) or len(set(fields)) != len(fields):
                raise ValueError('invalid fixture table fields')
            result = []
            for row in table['rows']:
                if len(row) != len(fields):
                    raise ValueError('invalid fixture table row width')
                item = {}
                for field, column, cell in zip(
                    fields, values, row, strict=True
                ):
                    if (
                        type(cell) is not int
                        or cell < -1
                        or cell >= len(column)
                    ):
                        raise ValueError('invalid fixture table cell')
                    if cell != -1:
                        item[field] = visit(column[cell], index)
                result.append(item)
            return result
        if isinstance(node, dict):
            return {k: visit(v, before) for k, v in node.items()}
        if isinstance(node, list):
            return [visit(v, before) for v in node]
        return node

    result = visit(value['document'], len(tables))
    if digest(result) != value['expanded_sha256']:
        raise ValueError('expanded fixture digest differs from source')
    return result


def dumps(value: Any) -> str:
    """One named table per block, one dictionary value or case row per line."""
    if value.get('format') != FORMAT:
        return json.dumps(value, indent=2, ensure_ascii=True) + '\n'
    lines = ['{']
    for key in ('format', 'expanded_sha256', 'document'):
        lines.append(
            '  '
            + json.dumps(key)
            + ': '
            + json.dumps(value[key], ensure_ascii=True)
            + ','
        )
    lines.append('  "tables": [')
    for index, table in enumerate(value['tables']):
        lines.extend(
            [
                '    {',
                '      "fields": ' + json.dumps(table['fields']) + ',',
                '      "values": [',
            ]
        )
        for column_index, column in enumerate(table['values']):
            lines.append('        [')
            lines.extend(
                '          '
                + json.dumps(item, ensure_ascii=True)
                + (',' if i + 1 < len(column) else '')
                for i, item in enumerate(column)
            )
            lines.append(
                '        ]'
                + (',' if column_index + 1 < len(table['values']) else '')
            )
        lines.extend(['      ],', '      "rows": ['])
        lines.extend(
            '        '
            + json.dumps(row)
            + (',' if i + 1 < len(table['rows']) else '')
            for i, row in enumerate(table['rows'])
        )
        lines.extend(
            [
                '      ]',
                '    }' + (',' if index + 1 < len(value['tables']) else ''),
            ]
        )
    lines.extend(['  ]', '}'])
    return '\n'.join(lines) + '\n'
