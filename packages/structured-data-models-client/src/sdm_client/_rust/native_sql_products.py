# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-pass useful SQL products and their explicit attachment handovers.

The closed product structure names every owned attachment.
Source/query meanings remain core data; this module evaluates no policy.
Request state attachment owners remain with their original Finished reader.
"""

from __future__ import annotations

import json
from typing import Any

from sdm_client._rust import results
from sdm_client._rust.codec import CodecError
from sdm_client._rust.native_products import (
    Bindings,
    Product,
    bound_groups,
    bound_label_input,
)
from sdm_client._rust.run import Finished

_FIELDS = {
    'source': ('source', 'identity', 'owner', 'target', 'tables'),
    'population': (
        'source',
        'identity',
        'owner',
        'rows',
        'batch_rows',
        'target',
    ),
    'label_inputs': (
        'population',
        'range',
        'owner',
        'target',
        'form',
        'condition_columns',
    ),
    'target_column': (
        'input',
        'owner',
        'derivation',
        'rows',
        'condition_values',
    ),
    'label_groups': (
        'population',
        'range',
        'owner',
        'native_target',
        'target_encoding',
    ),
    'label_batch': ('population', 'range', 'owner', 'complete'),
    'column_meaning': ('source', 'column', 'input', 'scalar'),
    'lookup': (
        'owner',
        'read',
        'source',
        'key',
        'range',
        'context_rows',
        'scored_rows',
    ),
    'lookup_rows': ('positions',),
}
_OUTPUTS = frozenset(_FIELDS) - {'lookup'}
_OWNED = frozenset(
    {
        'source',
        'population',
        'label_inputs',
        'label_groups',
        'label_batch',
        'lookup',
    }
)


def _object(
    value: Any, required: tuple[str, ...], optional: tuple[str, ...] = ()
) -> dict[str, Any]:
    if not isinstance(value, dict) or not set(required) <= set(value) <= set(
        required + optional
    ):
        raise CodecError('a native SQL product does not hold its closed fields')
    return value


def _attachment(value: Any) -> int:
    if type(value) is not int or not 0 <= value < 2**32:
        raise CodecError('a native SQL product names no attachment id')
    return value


def _ids(kind: str, data: dict[str, Any]) -> list[int]:
    fields = _FIELDS.get(kind)
    if fields is None:
        raise CodecError('unknown retained native SQL value')
    value = _object(
        data,
        fields,
        ('source_read', 'source_role', 'eligibility', 'time')
        if kind == 'label_inputs'
        else ('restored', 'what_if')
        if kind == 'label_groups'
        else (),
    )
    if kind in ('source', 'population'):
        _object(value['target'], ('source', 'column'))
    if kind == 'source':
        if not isinstance(value['tables'], list):
            raise CodecError('native SQL source tables are not an ordered list')
        for table in value['tables']:
            _object(table, ('source', 'read', 'columns'), ('time_column',))
    if kind in ('label_inputs', 'label_groups', 'label_batch', 'lookup'):
        _object(value['range'], ('start', 'stop'))
    if kind == 'label_groups' and value['native_target'] is not None:
        _object(value['native_target'], ('source', 'column'))
    if kind == 'column_meaning':
        _object(value['column'], ('source', 'column'))
        if not isinstance(value['input'], dict) or not isinstance(
            value['scalar'], dict
        ):
            raise CodecError(
                'native SQL column meaning has no closed input data'
            )
    if kind == 'label_inputs':
        _object(value['target'], ('source', 'column'))
        if value.get('time') is not None:
            _object(value['time'], ('source', 'column'))
        columns = value['condition_columns']
        if not isinstance(columns, list):
            raise CodecError(
                'native SQL condition columns are not an ordered list'
            )
        for column in columns:
            _object(column, ('source', 'column'))
    if kind in _OWNED:
        ids = [_attachment(value['owner'])]
        history = (
            value.get('eligibility')
            if kind == 'label_inputs'
            else value.get('what_if')
            if kind == 'label_groups'
            else None
        )
        if history is not None:
            history = _object(history, ('columns', 'values'))
            if not isinstance(history['columns'], list) or not isinstance(
                history['values'], list
            ):
                raise CodecError(
                    'native condition history has no ordered columns and values'
                )
            for column in history['columns']:
                ids.extend(_ids('label_inputs', column))
            for completed in history['values']:
                completed = _object(completed, ('column', 'owner'))
                _object(completed['column'], ('source', 'column'))
                ids.append(_attachment(completed['owner']))
        if kind == 'label_groups' and value.get('restored') is not None:
            restored = _object(value['restored'], ('rows', 'encoding'))
            ids.append(_attachment(restored['rows']))
        return ids
    if kind == 'column_meaning':
        return []
    if kind == 'lookup_rows':
        return [_attachment(value['positions'])]
    # All fields and attachment slots are validated before any owner is taken.
    ids = _ids('label_inputs', value['input']) + [_attachment(value['owner'])]
    conditions = value['condition_values']
    if not isinstance(conditions, list):
        raise CodecError(
            'native SQL completed conditions are not an ordered list'
        )
    for condition in conditions:
        condition = _object(condition, ('column', 'converted'))
        _object(condition['column'], ('source', 'column'))
        converted = _object(
            condition['converted'], ('rows', 'presence', 'encoding'), ('times',)
        )
        ids.append(_attachment(converted['rows']))
        if converted['presence'] is not None:
            ids.append(_attachment(converted['presence']))
        if converted.get('times') is not None:
            ids.append(_attachment(converted['times']))
    return ids


def owner_ids(kind: str, data: dict[str, Any]) -> list[int]:
    """Only typed opaque resources, never Arrow, ordinals or ordinary numbers."""
    _ids(kind, data)
    if kind == 'target_column':
        return owner_ids('label_inputs', data['input']) + [data['owner']]
    if kind not in _OWNED:
        return []
    out = [data['owner']]
    history = (
        data.get('eligibility')
        if kind == 'label_inputs'
        else data.get('what_if')
    )
    if history is not None:
        for column in history['columns']:
            out.extend(owner_ids('label_inputs', column))
        out.extend(column['owner'] for column in history['values'])
    return out


def _read(
    finished: Finished, expected: str | None
) -> Product | results.Gathered:
    answer = results.expect(finished.result, results.MaterializePrepared)
    _attachment(answer.value)
    raw = finished.take(answer.value)
    if not isinstance(raw, bytes):
        raise CodecError('a native SQL product has no JSON bytes')
    try:
        value = json.loads(raw)
    except ValueError as error:
        raise CodecError('a native SQL product has invalid JSON') from error
    value = _object(value, ('value', 'data'))
    kind = value['value']
    if not isinstance(kind, str) or kind not in _OUTPUTS | {'request'}:
        raise CodecError('unknown native SQL output value')
    if expected is not None and kind != expected:
        raise CodecError('native SQL operation returned another value')
    data = value['data']
    if kind == 'request':
        return results.Gathered.from_contract(data)
    ids = list(dict.fromkeys(_ids(kind, data)))
    if answer.value in ids:
        raise CodecError('a native SQL owner aliases its JSON value attachment')
    return Product(kind, data, {key: finished.take(key) for key in ids})


def read(finished: Finished) -> Product | results.Gathered:
    """Read the actual Source/Request alternative once, including Task results."""
    return _read(finished, None)


def adopt(finished: Finished, expected: str) -> Product:
    """Adopt one known useful result, rejecting another tag before owner takes."""
    if expected not in _OUTPUTS:
        raise CodecError('unknown native SQL product to adopt')
    value = _read(finished, expected)
    if not isinstance(value, Product):
        raise CodecError(
            'native SQL operation returned a request instead of data'
        )
    return value


def request(finished: Finished) -> results.Gathered:
    """Read the actual nested Materialized without taking payload/query owners."""
    value = _read(finished, 'request')
    if not isinstance(value, results.Gathered):
        raise CodecError(
            'native SQL operation returned data instead of a request'
        )
    return value


def bind(product: Product, bindings: Bindings) -> dict[str, Any]:
    """Rebind only typed slots, with one local binding per original key."""
    ids = list(dict.fromkeys(_ids(product.kind, product.data)))
    for key in product.owners:
        _attachment(key)
    if set(ids) != set(product.owners):
        raise CodecError(
            'native SQL product owners differ from its named slots'
        )
    bound = {key: bindings.add(product.owners[key]) for key in ids}
    data = product.data
    if product.kind == 'label_inputs':
        return bound_label_input(data, bound)
    if product.kind == 'label_groups':
        return bound_groups(data, bound)
    if product.kind in _OWNED:
        return {**data, 'owner': bound[data['owner']]}
    if product.kind == 'column_meaning':
        return dict(data)
    if product.kind == 'lookup_rows':
        return {**data, 'positions': bound[data['positions']]}
    return {
        **data,
        'input': bound_label_input(data['input'], bound),
        'owner': bound[data['owner']],
        'condition_values': [
            {
                **condition,
                'converted': {
                    **condition['converted'],
                    'rows': bound[condition['converted']['rows']],
                    'presence': None
                    if condition['converted']['presence'] is None
                    else bound[condition['converted']['presence']],
                    **(
                        {}
                        if condition['converted'].get('times') is None
                        else {'times': bound[condition['converted']['times']]}
                    ),
                },
            }
            for condition in data['condition_values']
        ],
    }
