# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native relational attribution after a decoded response has arrived.

The retained envelope supplies fresh entity cells, declared dtypes and
instances at their original read positions. Rust checks attribution and
owns result layout. Native converted entity owners stay local, addressed
only by source-row lineage bound to this particular model call.
"""

from __future__ import annotations

import copy
import json
import struct
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sdm_client._rust import codec, errors, results
from sdm_client._rust.relational_native import _points, vocabulary
from sdm_client._rust.run import Finished, run
from sdm_client.wire.dtypes import as_declared


def _run(
    value: dict[str, Any], attached: dict[int, bytes] | None = None
) -> Finished:
    return run(
        {'op': 'prepare_request', 'input': 1},
        {1: codec.attachment(value), **(attached or {})},
    )


def _call(
    value: dict[str, Any], attached: dict[int, bytes] | None = None
) -> dict[str, Any]:
    finished = _run(value, attached)
    prepared = results.expect(finished.result, results.Prepared)
    return json.loads(finished.take(prepared.value))


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


def _key_dtype(payload: dict[str, Any], reference: Any) -> Any:
    """One fresh native declared-dtype lookup, at each reached entity cast."""
    schema = payload.get('schema', {})
    for link in schema.get('relationships', []):
        if list(link.get('source_columns') or []) != [reference]:
            continue
        target = link.get('target_table')
        columns = link.get('target_columns') or []
        if not target or len(columns) != 1:
            return None
        return (
            schema.get('related_tables', {})
            .get(target, {})
            .get('columns', {})
            .get(columns[0], {})
            .get('dtype')
        )
    return None


@dataclass
class CompletedResponse:
    """A complete core response dataset and its local native entity column."""

    part: dict[str, Any]
    buffers: dict[str, bytes]
    entities: list[Any] | None

    @property
    def call_request_id(self) -> str:
        return self.part['data']['scope']['call_request_id']

    def attached(
        self, first: int = 2
    ) -> tuple[dict[str, Any], dict[int, bytes]]:
        """Bind these explicitly named dataset buffers for a core merge."""
        part = copy.deepcopy(self.part)
        attached = {}
        for at, name in enumerate(
            ('response', 'decoded', 'attributed', 'target'), start=first
        ):
            attached[at] = self.buffers[name]
            if name == 'response':
                part['data']['response'] = at
            elif name == 'decoded':
                part['data']['rows'] = at
            elif name == 'attributed':
                part['rows'] = at
            else:
                part['target_rows'] = at
        return part, attached

    def entity(self, lineage: dict[str, Any]) -> Any:
        """Present a core-selected source row from this exact call dataset."""
        if lineage['call_request_id'] != self.call_request_id:
            raise codec.CodecError(
                'relational lineage names another response dataset'
            )
        ordinal = lineage['source_row']
        if (
            self.entities is None
            or type(ordinal) is not int
            or not 0 <= ordinal < len(self.entities)
        ):
            raise codec.CodecError(
                'relational lineage names no native entity owner'
            )
        return self.entities[ordinal]


def present(finished: Finished, envelope: dict[str, Any]) -> CompletedResponse:
    """Finish one decoded call before the next envelope is sent or encoded."""
    prepared = results.expect(finished.result, results.Prepared)
    data = json.loads(finished.take(prepared.value))
    response_bytes = finished.take(data['response'])
    decoded_bytes = finished.take(data['rows'])
    rows = json.loads(decoded_bytes)

    table = envelope.get('predict', {}).get('instance_table', {})
    columns = table.get('columns') or []
    source = table.get('rows') or []
    names = []
    matches = []
    exhausted = True
    for name in columns:
        names.append(name)
        matched = bool(name.startswith(data['entity_reference_prefix']))
        matches.append(matched)
        if matched:
            exhausted = False
            break
    selected = _call(
        {
            'stage': 'relational_reference_column',
            'facts': {'prefix_matches': matches, 'exhausted': exhausted},
        }
    )
    reference = (
        None if selected['ordinal'] is None else names[selected['ordinal']]
    )
    entities = None
    instances = None
    forecasting = None
    if reference is not None:
        # The dtype expression is deliberately inside this comprehension.
        # Each actual cast can mutate the schema seen by the following row.
        entities = [
            as_declared(
                row[columns.index(reference)], _key_dtype(envelope, reference)
            )
            for row in source
        ]
        instances = (
            [row[columns.index('instance_id')] for row in source]
            if 'instance_id' in columns
            else None
        )
        # Preserve the comparison result itself. Its native truth is read
        # only after each reached core bounds check, and again if formatting
        # the reference's duplicate-row diagnostic requires it.
        forecasting = (
            envelope.get('task', {}).get('kind') == data['forecast_kind']
        )
    attribution = {
        'scope': data['scope'],
        'source_rows': None if entities is None else len(entities),
        'instances': instances is not None,
    }

    def response_value(fields: dict[str, Any]) -> Any:
        field = fields['native_field']
        if field == 'index':
            return str(codec.restored_integer(fields['index']))
        if field == 'kind':
            return repr(envelope.get('task', {}).get('kind'))
        if field == 'expected':
            assert instances is not None
            return repr(instances[fields['index']])
        if field == 'where':
            template = (
                fields['where_forecast']
                if forecasting
                else fields['where_static']
            )
            return template.format(
                index=fields['index'],
                step=codec.restored_integer(fields['step'])
                if fields['step']
                else None,
            )
        raise codec.CodecError(
            'unknown core-selected relational native presentation'
        )

    with errors.holding(relational_response_value=response_value):
        accepted = []
        at = 0
        while at < len(rows):
            # Keyed attribution reaches one native forecast/instance read at
            # a time. Sending an overlapping 256-row suffix would repeatedly
            # encode rows the core cannot yet consume.
            width = 256 if entities is None else 1
            checked = _call(
                {
                    'stage': 'check_relational_rows',
                    'attribution': attribution,
                    'rows': rows[at : at + width],
                }
            )
            accepted.extend(checked['rows'])
            at += len(checked['rows'])
            bound = checked['forecast']
            if bound is not None:
                checked_row = _call(
                    {
                        'stage': 'check_relational_forecast',
                        'bound': bound,
                        'forecasting': bool(forecasting),
                    }
                )
                if instances is None:
                    item = _call(
                        {
                            'stage': 'relational_without_instance',
                            'checked': checked_row,
                        }
                    )
                else:
                    response_id = str(bound['row']['id'])
                    displayed = str(instances[bound['source_row']])
                    different = bool(response_id != displayed)
                    item = _call(
                        {
                            'stage': 'attribute_relational_instance',
                            'checked': checked_row,
                            'instance': {
                                'ordinal': bound['row']['ordinal'],
                                'points': _points(displayed),
                                'different': different,
                            },
                        }
                    )
                accepted.append(item)
                at += 1
        data = {**data, 'response': 2, 'rows': 3}
        attributed_bytes = codec.attachment(accepted)
        _call(
            {
                'stage': 'check_relational_coverage',
                'data': data,
                'attribution': attribution,
                'rows': 4,
            },
            {2: response_bytes, 3: decoded_bytes, 4: attributed_bytes},
        )

        # Target metadata is read only after every entity is attributed and
        # all requested rows are covered. No pre-send task snapshot is used.
        task = (envelope or {}).get('task', {})
        words = vocabulary()
        native_kind = str(task.get('kind', ''))
        classification = words['classification'] in native_kind
        kind = _points(native_kind)
        meaning = _call(
            {
                'stage': 'relational_target_meaning',
                'kind': kind,
                'classification': classification,
            }
        )
        target = {
            'kind': kind,
            'classification': classification,
            'dtype_truth': None,
            'numeric_prefix': None,
            'dtype': None,
        }
        dtype = None
        if meaning['classification']:
            dtype = task.get('target', {}).get('dtype')
            target['dtype_truth'] = bool(dtype)
            if target['dtype_truth']:
                target['numeric_prefix'] = bool(
                    dtype.startswith(tuple(meaning['numeric_prefixes']))
                )
            if isinstance(dtype, str):
                target['dtype'] = _points(dtype)
        converted = []
        for row in rows:
            value = row['prediction']
            probabilities = row['probabilities']
            if target['numeric_prefix']:
                value = _encoded_native(as_declared(_native(value), dtype))
            native_probabilities = None
            if probabilities is not None:
                native_probabilities = []
                for key, probability in probabilities:
                    restored = (
                        as_declared(key, dtype)
                        if target['numeric_prefix']
                        else key
                    )
                    name = key if restored is key else str(restored)
                    native_probabilities.append(
                        {
                            'value': _encoded_native(restored),
                            'name': name,
                            'probability': probability,
                        }
                    )
            converted.append(
                {
                    'ordinal': row['ordinal'],
                    'prediction': value,
                    'probabilities': native_probabilities,
                }
            )
        part = {
            'data': data,
            'attribution': attribution,
            'rows': 4,
            'target': target,
            'target_rows': 5,
        }
        completed = _run(
            {'stage': 'complete_relational_response', 'part': part},
            {
                2: response_bytes,
                3: decoded_bytes,
                4: attributed_bytes,
                5: codec.attachment(converted),
            },
        )
        answer = results.expect(completed.result, results.Prepared)
        part = json.loads(completed.take(answer.value))
        buffers = {
            'response': completed.take(part['data']['response']),
            'decoded': completed.take(part['data']['rows']),
            'attributed': completed.take(part['rows']),
            'target': completed.take(part['target_rows']),
        }
    return CompletedResponse(part, buffers, entities)
