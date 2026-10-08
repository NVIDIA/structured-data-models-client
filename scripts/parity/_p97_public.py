# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seeded public recipes over the declared dtype and request templates."""

from __future__ import annotations

import copy
import datetime as dt
import math
import random
import sys
from decimal import Decimal


def values(dtype, series, rng, role):
    import pandas as pd

    family, _, name = dtype.partition('/')
    count = len(series)
    numbers = list(range(1, count + 1))
    if pd.api.types.is_numeric_dtype(series.dtype):
        numbers = [int(value) for value in series]
    if name == 'interval_int64':
        return [pd.Interval(value, value + 1) for value in numbers]
    if name == 'period_month':
        return [f'2024-{1 + ordinal % 12:02d}' for ordinal in range(count)]
    if name.startswith(('date32', 'date64')):
        return [
            dt.date(2024, 1, 1) + dt.timedelta(days=value) for value in numbers
        ]
    if name.startswith(('datetime64_', 'timestamp_')):
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            return list(series)
        return [
            pd.Timestamp('2024-01-01') + pd.Timedelta(days=value)
            for value in numbers
        ]
    if name.startswith(('timedelta64_', 'duration_', 'time32_', 'time64_')):
        return numbers
    if name.startswith('decimal'):
        return [Decimal(value * 100) for value in numbers]
    if name == 'null':
        return [None] * count
    if name in {'bool', 'boolean', 'bool8'}:
        return [bool(ordinal % 2) for ordinal in range(count)]
    if name.startswith(('list_', 'large_list_', 'fixed_list_')):
        if name == 'list_list_int64':
            return [[[value, value + 1]] for value in numbers]
        return [[value, value + 1] for value in numbers]
    if name.startswith('map_'):
        return [[('one', value), ('two', value + 1)] for value in numbers]
    if name == 'struct_int_string':
        return [{'number': value, 'label': 'one'} for value in numbers]
    if name == 'uuid':
        import uuid

        return [uuid.UUID(int=rng.getrandbits(128)) for _ in range(count)]
    if name == 'json':
        return [f'{{"value":{value}}}' for value in numbers]
    if name in {'binary', 'large_binary', 'fixed_binary_3', 'S6'}:
        if role == 'primary_key':
            return [f'{ordinal:03d}'.encode() for ordinal in range(count)]
        return [b'one' if ordinal % 2 else b'two' for ordinal in range(count)]
    if (
        'string' in name
        or name.startswith('category_')
        or name in {'U6', 'object'}
    ):
        if role == 'primary_key' and not name.startswith('category_'):
            return [str(ordinal) for ordinal in range(count)]
        return ['one' if ordinal % 2 else 'two' for ordinal in range(count)]
    if name.startswith(('complex', 'clongdouble')):
        return [str(value) + '+0j' for value in numbers]
    if family in {'numpy', 'pandas', 'arrow'} and name.startswith(
        (
            'int',
            'uint',
            'Int',
            'UInt',
            'float',
            'Float',
            'longdouble',
            'sparse_',
        )
    ):
        return numbers
    raise ValueError('unsupported public dtype: ' + dtype)


def frame_spec(frame, encode):
    return {
        'columns': {
            name: {
                'dtype': str(series.dtype),
                'values': encode(series.tolist()),
            }
            for name, series in frame.items()
        }
    }


def recipes(root, inventory, seed, seal):
    sys.path.insert(0, str(root / 'tests/parity'))
    from _fixture_ops import encode
    from _parity.data import generate

    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('public generation requires a uint32 seed')
    templates = inventory['matrix_templates']
    dtypes = inventory['dtype_ids']
    roles = inventory['roles']
    if not templates or not dtypes or not roles:
        raise ValueError(
            'public generation requires complete matrix dimensions'
        )
    variants = math.ceil(
        max(2000, inventory['minimum_unique_sweep_inputs'])
        / (len(dtypes) * len(roles))
    )
    rng = random.Random(seed)
    source = generate('shop', seed)
    source['users'] = source['users'].head(4).copy()
    source['products'] = source['products'].head(4).copy()
    source['orders'] = source['orders'].head(24).copy()
    for column in ('user_id', 'product_id'):
        source['orders'][column] = (source['orders'][column] - 1) % 4 + 1
    generated = []
    supplemental = []
    for dtype_ordinal, dtype in enumerate(dtypes):
        for role in roles:
            choices = sorted(
                name
                for name, template in templates.items()
                if template['role'] == role
                and dtype in template['dtype_ids']
                and 'constructor' not in template['input']
            )
            if not choices:
                raise ValueError('public dtype-role cell lacks a template')
            for variant in range(variants):
                template_name = choices[
                    (dtype_ordinal * variants + variant) % len(choices)
                ]
                template = templates[template_name]
                frames = {name: frame.copy() for name, frame in source.items()}
                scalar_name = dtype.partition('/')[2]
                if role == 'primary_key' and (
                    scalar_name in {'bool', 'boolean', 'bool8'}
                    or scalar_name.startswith('category_')
                ):
                    frames[template['table']] = (
                        frames[template['table']].head(2).copy()
                    )
                    foreign_key = template['column']
                    if foreign_key in frames['orders']:
                        frames['orders'][foreign_key] = (
                            frames['orders'][foreign_key] - 1
                        ) % 2 + 1
                frames['users']['country'] = [
                    ''.join(rng.choices('abcdefghijklmnopqrstuvwxyz', k=12))
                    for _ in range(len(frames['users']))
                ]
                column = frames[template['table']][template['column']]
                matrix_values = encode(values(dtype, column, rng, role))
                value = copy.deepcopy(template['input'])
                value['tables'] = {
                    name: frame_spec(frames[name], encode)
                    for name in value['table_names']
                }
                identity = f'sweep/{dtype}/{role}/{variant}'
                generated.append(
                    {
                        'id': identity,
                        'recipe': seal(
                            {
                                'entrypoint': template['entrypoint'],
                                'operation': template['operation'],
                                'input': value,
                                'matrix': {
                                    'template': template_name,
                                    'dtype': dtype,
                                    'values': matrix_values,
                                },
                                'seed': seed,
                            }
                        ),
                    }
                )
                if variant == 0:
                    declared_name = (
                        'declared-memory-' + role + '-' + template['task']
                    )
                    declared = templates[declared_name]
                    declared_input = copy.deepcopy(declared['input'])
                    declared_input['tables'] = copy.deepcopy(value['tables'])
                    supplemental.append(
                        {
                            'id': 'declared/' + identity,
                            'recipe': seal(
                                {
                                    'entrypoint': declared['entrypoint'],
                                    'operation': declared['operation'],
                                    'input': declared_input,
                                    'matrix': {
                                        'template': declared_name,
                                        'dtype': dtype,
                                        'values': copy.deepcopy(matrix_values),
                                    },
                                    'seed': seed,
                                }
                            ),
                        }
                    )
    return generated + supplemental
