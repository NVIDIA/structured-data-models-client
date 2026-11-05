# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared shop plus explicit extra user columns or uppercase identifiers."""

import json
from pathlib import Path


def load_fixture(path):
    path = Path(path)
    variant = json.loads(path.read_text())
    if 'base' not in variant:
        return variant
    if variant['base'] != 'shop.json':
        raise ValueError('shop variants must use shop.json')
    fixture = json.loads(path.with_name('shop.json').read_text())
    if set(variant) == {'base', 'uppercase_columns'}:
        if variant['uppercase_columns'] is not True:
            raise ValueError('uppercase_columns must be true')
        for table in fixture['tables'].values():
            table['columns'] = [name.upper() for name in table['columns']]
        fixture['primary_keys'] = {
            table: name.upper()
            for table, name in fixture['primary_keys'].items()
        }
        fixture['links'] = [
            [
                table + '.' + column.upper()
                for table, column in (name.split('.', 1) for name in link)
            ]
            for link in fixture['links']
        ]
        return fixture
    if set(variant) != {'base', 'user_columns'}:
        raise ValueError('unknown shop variant fields')
    users = fixture['tables']['users']
    for column in variant['user_columns']:
        if (
            not {'name', 'values'}
            <= column.keys()
            <= {'name', 'values', 'type'}
        ):
            raise ValueError('invalid shop column fields')
        name, values = column['name'], column['values']
        if name in users['columns'] or len(values) != len(users['rows']):
            raise ValueError(
                'shop columns need unique names and one value per row'
            )
        users['columns'].append(name)
        for row, value in zip(users['rows'], values, strict=True):
            row.append(value)
        if 'type' in column:
            users.setdefault('types', {})[name] = column['type']
    return fixture
