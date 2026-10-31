# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reached native response reads and retained presentation owners."""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from sdm_api.generated.tfm_api import PredictionItem, ResponseContractError

from sdm_client.errors import StructuredDataError


@dataclass
class NativeResponse:
    request_id: str
    model: str
    items: tuple[PredictionItem, ...]
    metadata: dict[str, Any]


class Reader:
    def __init__(self, call: Callable[..., Any]) -> None:
        self.call = call

    def check(
        self, field: str, value: Any, kind: str, **facts: Any
    ) -> dict[str, Any]:
        checked = self.call(
            'check_direct_response_field',
            {
                'field': field,
                'observation': {'kind': kind, **facts},
            },
        )
        if checked['field'] != field:
            raise RuntimeError(
                'native response validation returned a different field'
            )
        if checked['fault'] is not None:
            self.refused(field, value, checked['fault'])
        return checked

    @staticmethod
    def refused(field: str, value: Any, fault: str) -> None:
        # Presentation only: the selected fault comes from the core before
        # any native repr is reached or the next field is consumed.
        name = {'response_id': 'id', 'row_id': 'id'}.get(field, field)
        model = (
            'PredictionResponse'
            if field in ('response_id', 'model', 'predictions', 'metadata')
            else 'PredictionItem'
            if field == 'row_index'
            else 'prediction'
        )
        if fault == 'missing':
            text = f'{model} response is missing required field {name!r}'
        elif fault == 'null':
            text = f'{model} response has a null {name!r}, which the contract requires'
        elif fault == 'not_text':
            text = f'{model} response has {name!r} = {value!r}, where the contract requires a string'
        elif fault == 'not_whole_kind':
            text = f'{model} response has {name!r} = {value!r}, where the contract requires a whole number'
        elif fault == 'not_whole_value':
            text = f'{model} response has {name!r} = {value!r}, which is not a whole number'
        elif fault == 'not_object':
            if field in ('response', 'row'):
                owner = (
                    'PredictionResponse'
                    if field == 'response'
                    else 'PredictionItem'
                )
                text = f'response carries {value!r} where the contract requires a {owner} object'
            elif field == 'metadata':
                text = f'{model} response field {name!r} is not an object'
            else:
                text = f'response carries {value!r} where the contract requires an object'
        elif fault == 'not_sequence':
            text = (
                f'{model} response field {name!r} is not a sequence'
                if field == 'predictions'
                else f'response carries {value!r} where the contract requires an array of numbers'
            )
        elif fault == 'not_number':
            text = f'response carries {value!r} where the contract requires a number'
        elif fault == 'not_finite':
            text = f'response carries a non-finite number {value!r}, which the contract does not permit'
        else:
            raise RuntimeError('unknown native response fault')
        raise ResponseContractError(text) from None

    def required(self, owner: Any, name: str, field: str) -> Any:
        self.check(field, None, 'presence', present=name in owner)
        return owner[name]

    def optional(self, field: str, value: Any) -> bool:
        return bool(
            self.check(field, value, 'optional_null', null=value is None)[
                'is_null'
            ]
        )

    def text(self, field: str, value: Any) -> Any:
        self.check(field, value, 'text', text=isinstance(value, str))
        return value

    def whole(self, field: str, value: Any) -> int | None:
        null = value is None
        boolean = not null and isinstance(value, bool)
        number_family = not (null or boolean) and isinstance(
            value, (int, float)
        )
        checked = self.check(
            field,
            value,
            'whole_kind',
            boolean=boolean,
            number_family=number_family,
            null=null,
        )
        if checked['is_null']:
            return None
        if isinstance(value, float):
            finite = math.isfinite(value)
            # int is reached again below after successful validation.
            integral = not bool(value != int(value)) if finite else None
            self.check(
                field, value, 'whole_float', finite=finite, integral=integral
            )
        return int(value)

    def number(self, field: str, value: Any) -> float:
        self.check(field, value, 'number_kind', boolean=isinstance(value, bool))
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            self.check(field, value, 'number_conversion', bits=None)
            raise AssertionError(
                'core accepted failed native float conversion'
            ) from None
        self.check(
            field,
            value,
            'number_conversion',
            bits=list(struct.pack('>d', number)),
        )
        return number

    def numbers(self, field: str, member: str, value: Any) -> Any:
        if self.optional(field, value):
            return None
        self.check(field, value, 'object', mapping=isinstance(value, Mapping))
        return {
            str(key): self.number(member, item) for key, item in value.items()
        }

    def array(self, field: str, member: str, value: Any) -> Any:
        if self.optional(field, value):
            return None
        string = isinstance(value, str)
        byte_string = not string and isinstance(value, bytes)
        mapping = not (string or byte_string) and isinstance(value, Mapping)
        iterable = not (string or byte_string or mapping) and isinstance(
            value, Iterable
        )
        self.check(
            field,
            value,
            'sequence',
            string=string,
            bytes=byte_string,
            mapping=mapping,
            iterable=iterable,
        )
        return tuple(self.number(member, item) for item in value)

    def copied(self, field: str, value: Any) -> Any:
        if self.optional(field, value):
            return None
        return (
            tuple(dict(item) for item in value)
            if field == 'rankings'
            else dict(value)
        )

    def item(self, value: Any) -> PredictionItem:
        self.check('row', value, 'object', mapping=isinstance(value, Mapping))
        identifier = value.get('id')
        identifier = (
            None
            if self.optional('row_id', identifier)
            else self.text('row_id', identifier)
        )
        row_index = self.whole(
            'row_index', self.required(value, 'row_index', 'row_index')
        )
        forecast_step = self.whole('forecast_step', value.get('forecast_step'))
        prediction = value.get('prediction')
        probabilities = self.numbers(
            'probabilities', 'probability', value.get('probabilities')
        )
        scores = self.array('scores', 'score', value.get('scores'))
        rankings = self.copied('rankings', value.get('rankings'))
        embeddings = self.array(
            'embeddings', 'embedding', value.get('embeddings')
        )
        quantiles = self.numbers(
            'quantiles', 'quantile', value.get('quantiles')
        )
        explanation = self.copied('explanation', value.get('explanation'))
        metadata = self.copied('row_metadata', value.get('metadata'))
        return PredictionItem(
            row_index=cast(int, row_index),
            id=identifier,
            forecast_step=forecast_step,
            prediction=prediction,
            probabilities=probabilities,
            scores=scores,
            rankings=rankings,
            embeddings=embeddings,
            quantiles=quantiles,
            explanation=explanation,
            metadata=metadata,
        )

    def response(self, value: Any, model_id: str) -> NativeResponse:
        try:
            self.check(
                'response', value, 'object', mapping=isinstance(value, Mapping)
            )
            request_id = self.text(
                'response_id', self.required(value, 'id', 'response_id')
            )
            model = self.text('model', self.required(value, 'model', 'model'))
            items = self.required(value, 'predictions', 'predictions')
            string = isinstance(items, str)
            byte_string = not string and isinstance(items, bytes)
            iterable = not (string or byte_string) and hasattr(
                items, '__iter__'
            )
            self.check(
                'predictions',
                items,
                'sequence',
                string=string,
                bytes=byte_string,
                mapping=False,
                iterable=iterable,
            )
            parsed = tuple(self.item(item) for item in items)
            metadata = self.required(value, 'metadata', 'metadata')
            self.check(
                'metadata',
                metadata,
                'object',
                mapping=isinstance(metadata, Mapping),
            )
            return NativeResponse(request_id, model, parsed, dict(metadata))
        except (ResponseContractError, TypeError, ValueError) as error:
            raise StructuredDataError(
                f'{model_id} returned a response that does not match the contract: {error}',
                code='INVALID_RESPONSE',
            ) from error


def finish(
    finished: Any,
    client: Any,
    model_id: str,
    options: Any,
    call: Callable[..., Any],
) -> Any:
    import json

    from sdm_client._rust import results
    from sdm_client._rust.predict import _adapter

    prepared = results.expect(finished.result, results.PreparedDirectCall)
    description = json.loads(finished.take(prepared.value))
    payload = json.loads(finished.take(description['payload']))
    if 'inference' in payload:
        payload['inference'] = options
    # Transport may mutate this payload. The reference consumes that same
    # object only after parsing the complete native response.
    response = Reader(call).response(
        client._send(_adapter(client, model_id), payload), model_id
    )
    return present(response, payload, description['bound'], call)


def present(
    response: NativeResponse,
    payload: dict[str, Any],
    bound: dict[str, Any],
    call: Callable[..., Any],
) -> Any:
    """Share Rust attribution and native presentation across model-call kinds."""
    from sdm_client.result import PredictionResult

    items = attribute(response.items, payload, call)
    task = (payload or {}).get('task', {})
    kind = str(task.get('kind', ''))
    classified = call(
        'layout_native_direct',
        {'kind': 'task_kind', 'points': [ord(char) for char in kind]},
    )['selected']
    dtype = None
    if classified:
        candidate = task.get('target', {}).get('dtype')
        numeric = bool(candidate) and candidate.startswith(
            ('int', 'float', 'bool')
        )
        if call(
            'layout_native_direct',
            {'kind': 'target_dtype', 'numeric': bool(numeric)},
        )['selected']:
            dtype = candidate
    restored = restore(items, dtype, call)
    result = PredictionResult(
        model=response.model,
        request_id=response.request_id,
        items=restored,
        metadata=response.metadata,
        outputs=bound['outputs'],
    )
    query = {
        'bound': bound,
        'rows': len(restored),
        'present': None,
    }
    names = call('observe_direct_outputs', query)
    present = [
        (
            name,
            result._explanation is not None
            if name == 'explanation'
            else any(
                getattr(item, name, None) not in (None, {}, [], ())
                for item in restored
            ),
        )
        for name in names
    ]
    call('observe_direct_outputs', {**query, 'present': present})
    return result


def named_key(answered: str, call: Callable[..., Any]) -> Any:
    from sdm_client._rust.direct import (
        _native_json,
        _native_literal,
        _native_value_call,
        _parsed_value,
    )

    if not call(
        'layout_native_direct', {'kind': 'id_length', 'length': len(answered)}
    )['selected']:
        return None
    native, failure = _native_json(answered)
    decision = _native_value_call(
        {
            'op': 'layout_native_direct',
            'kind': 'parsed',
            'json': True,
            'value': _parsed_value(native, failure),
        },
        failure,
    )
    if decision['literal']:
        assert failure is not None
        native, failure = _native_literal(answered, failure)
        decision = _native_value_call(
            {
                'op': 'layout_native_direct',
                'kind': 'parsed',
                'json': False,
                'value': _parsed_value(native, failure),
            },
            failure,
        )
    return str(native[0]) if decision['stringify'] else None


def attribute(
    items: tuple[PredictionItem, ...],
    payload: dict[str, Any],
    call: Callable[..., Any],
) -> list[PredictionItem]:
    import dataclasses

    from sdm_client._rust import errors
    from sdm_client.wire.dtypes import as_declared

    table = payload.get('predict', {}).get('instance_table', {})
    rows = table.get('rows')
    if not call(
        'layout_native_direct',
        {'kind': 'rows', 'is_list': isinstance(rows, list)},
    )['selected']:
        return list(items)
    columns = table.get('columns') or []
    key = (
        payload.get('schema', {}).get('instance_table', {}).get('primary_key')
        or []
    )
    single = call(
        'layout_native_direct', {'kind': 'key_count', 'count': len(key)}
    )['selected']
    key_name = key[0] if single else None
    key_dtype = (
        payload.get('schema', {})
        .get('instance_table', {})
        .get('columns', {})
        .get(key_name, {})
        .get('dtype')
        if key_name
        else None
    )
    keys = (
        [as_declared(row[columns.index(key_name)], key_dtype) for row in rows]
        if key_name and key_name in columns
        else None
    )
    # Retain the actual native text set: equality/hash overrides are observable.
    keys_in_batch = {str(value) for value in keys or ()}
    indices = [str(item.row_index) for item in items]
    prior = call(
        'layout_native_direct', {'kind': 'positions', 'indices': indices}
    )['prior']
    attributed = []
    for ordinal, item in enumerate(items):
        index = call(
            'layout_native_direct',
            {
                'kind': 'position',
                'index': indices[ordinal],
                'rows': len(rows),
                'prior': prior[ordinal],
            },
        )['index']
        if keys is not None and item.id is not None:
            answered = str(item.id)
            name = named_key(answered, call)
            if name is None and answered in keys_in_batch:
                name = answered
            different = None if name is None else bool(name != str(keys[index]))

            def response_id(fields: Any, held: PredictionItem = item) -> Any:
                return held.id

            def response_key(fields: Any, held_index: int = index) -> Any:
                return keys[held_index]

            with errors.holding(
                direct_response_id=response_id,
                direct_response_key=response_key,
            ):
                call(
                    'layout_native_direct',
                    {'kind': 'match', 'index': index, 'different': different},
                )
        value = None if keys is None else keys[index]
        replace = call(
            'layout_native_direct',
            {'kind': 'replacement', 'key_is_null': value is None},
        )['selected']
        attributed.append(
            dataclasses.replace(item, id=value) if replace else item
        )
    call(
        'layout_native_direct',
        {'kind': 'complete', 'indices': indices, 'rows': len(rows)},
    )
    return attributed


def restore(
    items: list[PredictionItem], dtype: str | None, call: Callable[..., Any]
) -> list[PredictionItem]:
    import dataclasses

    from sdm_client.wire.dtypes import as_declared

    if dtype is None:
        return items
    restored_items = []
    for item in items:
        changed: dict[str, Any] = {}
        value = item.prediction
        # The core selected this cast; the host retains native conversion effects.
        converted = as_declared(value, dtype)
        if call(
            'restore_direct_response',
            {'kind': 'prediction', 'same': converted is value},
        ):
            changed['prediction'] = converted
        given = item.probabilities
        if given:
            renamed = {}
            for key, probability in given.items():
                converted_key = as_declared(key, dtype)
                renamed[key if converted_key is key else str(converted_key)] = (
                    probability
                )
            keep = call(
                'restore_direct_response',
                {
                    'kind': 'classes',
                    'original_count': len(given),
                    'renamed_count': len(renamed),
                },
            )
            selected = renamed if keep else given
            different = selected != given
            if call(
                'restore_direct_response',
                {'kind': 'probabilities', 'different': bool(different)},
            ):
                changed['probabilities'] = selected
        restored_items.append(
            dataclasses.replace(item, **changed) if changed else item
        )
    return restored_items
