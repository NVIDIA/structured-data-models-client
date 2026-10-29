# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless native builtin value storage shared by response hosts.

This codec performs representation conversion only, with no model admission,
restoration policy, native callbacks or reusable object identities.
"""

from __future__ import annotations

import struct
from decimal import Decimal
from typing import Any

from sdm_client._rust import codec


def _native(value: dict[str, Any]) -> Any:
    """Restore decoded builtin data without replaying numeric admission."""
    kind = value['kind']
    if kind == 'null':
        return None
    if kind == 'boolean':
        return value['value']
    if kind == 'integer':
        return codec.restored_integer(value['digits'])
    if kind == 'float':
        return struct.unpack('>d', bytes(value['bits']))[0]
    if kind == 'unicode':
        return ''.join(chr(point) for point in value['points'])
    if kind == 'array':
        return [_native(item) for item in value['items']]
    if kind == 'object':
        return {
            ''.join(chr(point) for point in member['key']['points']): _native(
                member['value']
            )
            for member in value['members']
        }
    if kind == 'tree':
        return _native_tree(value['nodes'])
    raise codec.CodecError('unknown decoded relational native value')


def _encoded_native(value: Any) -> dict[str, Any]:
    """Encode only decoded or restored builtins, with no native display."""
    if value is None:
        return {'kind': 'null'}
    if type(value) is bool:
        return {'kind': 'boolean', 'value': value}
    if type(value) is int:
        # Restoring an already reached int must not add a later decimal
        # display failure. Native probability-key str keeps its own boundary.
        return {'kind': 'integer', 'digits': str(Decimal(value))}
    if type(value) is float:
        return {'kind': 'float', 'bits': list(struct.pack('>d', value))}
    if type(value) is str:
        return {'kind': 'unicode', 'points': [ord(point) for point in value]}
    if type(value) in (list, dict):
        return _encoded_native_tree(value)
    raise codec.CodecError(
        'a decoded relational value ceased to be builtin data'
    )


def _encoded_native_tree(value: Any) -> dict[str, Any]:
    """Encode complete builtin containers as local postorder value data."""
    pending = [('value', value)]
    nodes: list[dict[str, Any]] = []
    roots: list[int] = []
    node: dict[str, Any]
    while pending:
        kind, item = pending.pop()
        if kind == 'value':
            if type(item) is list:
                pending.append(('array', len(item)))
                pending.extend(('value', child) for child in reversed(item))
                continue
            if type(item) is dict:
                pending.append(('object', list(item)))
                pending.extend(
                    ('value', child) for child in reversed(item.values())
                )
                continue
            node = {'kind': 'scalar', 'value': _encoded_native(item)}
        else:
            length = item if kind == 'array' else len(item)
            children = roots[len(roots) - length :] if length else []
            if length:
                del roots[len(roots) - length :]
            node = (
                {'kind': 'array', 'children': children}
                if kind == 'array'
                else {
                    'kind': 'object',
                    'members': [
                        {
                            'key': {
                                'kind': 'unicode',
                                'points': [ord(point) for point in key],
                            },
                            'value': child,
                        }
                        for key, child in zip(item, children, strict=True)
                    ],
                }
            )
        roots.append(len(nodes))
        nodes.append(node)
    return {'kind': 'tree', 'nodes': nodes}


def _native_tree(nodes: list[dict[str, Any]]) -> Any:
    """Restore a checked local value tree without recursive container calls."""
    if not nodes or nodes[-1]['kind'] == 'scalar':
        raise codec.CodecError('native tree has no final container root')
    consumed = object()
    values = []
    starts = []
    for ordinal, node in enumerate(nodes):
        kind = node['kind']
        if kind == 'scalar':
            if node['value']['kind'] in ('array', 'object', 'tree'):
                raise codec.CodecError(
                    'native tree scalar contains a container'
                )
            values.append(_native(node['value']))
            starts.append(ordinal)
            continue
        if kind == 'array':
            children = node['children']
        elif kind == 'object':
            children = [member['value'] for member in node['members']]
        else:
            raise codec.CodecError('unknown native tree node kind')
        if any(
            type(child) is not int or not 0 <= child < ordinal
            for child in children
        ) or any(a >= b for a, b in zip(children, children[1:], strict=False)):
            raise codec.CodecError('native tree child order is invalid')
        start = starts[children[0]] if children else ordinal
        next_ordinal = start
        for child in children:
            if starts[child] != next_ordinal:
                raise codec.CodecError(
                    'native tree interleaves child populations'
                )
            next_ordinal = child + 1
        if next_ordinal != ordinal:
            raise codec.CodecError('native tree omits intervening child data')
        items = []
        for child in children:
            item = values[child]
            if item is consumed:
                raise codec.CodecError('native tree repeats a consumed child')
            items.append(item)
            values[child] = consumed
        values.append(
            items
            if kind == 'array'
            else {
                ''.join(chr(point) for point in member['key']['points']): item
                for member, item in zip(node['members'], items, strict=True)
            }
        )
        starts.append(start)
    if not values:
        raise codec.CodecError('native tree has no root')
    root = values.pop()
    if root is consumed or any(value is not consumed for value in values):
        raise codec.CodecError('native tree contains unreachable values')
    return root
