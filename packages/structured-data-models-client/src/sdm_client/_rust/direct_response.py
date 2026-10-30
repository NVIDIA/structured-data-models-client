# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reached native response reads and retained presentation owners."""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

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

    def check(self, field: str, value: Any, kind: str, **facts: Any) -> dict[str, Any]:
        checked = self.call('check_direct_response_field', {
            'field': field, 'observation': {'kind': kind, **facts},
        })
        if checked['field'] != field:
            raise RuntimeError('native response validation returned a different field')
        if checked['fault'] is not None:
            self.refused(field, value, checked['fault'])
        return checked

    @staticmethod
    def refused(field: str, value: Any, fault: str) -> None:
        # Presentation only: the selected fault comes from the core before
        # any native repr is reached or the next field is consumed.
        name = {'response_id': 'id', 'row_id': 'id'}.get(field, field)
        model = 'PredictionResponse' if field in ('response_id', 'model', 'predictions', 'metadata') else 'PredictionItem' if field == 'row_index' else 'prediction'
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
                owner = 'PredictionResponse' if field == 'response' else 'PredictionItem'
                text = f'response carries {value!r} where the contract requires a {owner} object'
            elif field == 'metadata':
                text = f'{model} response field {name!r} is not an object'
            else:
                text = f'response carries {value!r} where the contract requires an object'
        elif fault == 'not_sequence':
            text = f'{model} response field {name!r} is not a sequence' if field == 'predictions' else f'response carries {value!r} where the contract requires an array of numbers'
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
        return bool(self.check(field, value, 'optional_null', null=value is None)['is_null'])

    def text(self, field: str, value: Any) -> Any:
        self.check(field, value, 'text', text=isinstance(value, str))
        return value

    def whole(self, field: str, value: Any) -> int | None:
        null = value is None
        boolean = not null and isinstance(value, bool)
        number_family = not (null or boolean) and isinstance(value, (int, float))
        checked = self.check(field, value, 'whole_kind', boolean=boolean, number_family=number_family, null=null)
        if checked['is_null']:
            return None
        if isinstance(value, float):
            finite = math.isfinite(value)
            # int is reached again below after successful validation.
            integral = not bool(value != int(value)) if finite else None
            self.check(field, value, 'whole_float', finite=finite, integral=integral)
        return int(value)

    def number(self, field: str, value: Any) -> float:
        self.check(field, value, 'number_kind', boolean=isinstance(value, bool))
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            self.check(field, value, 'number_conversion', bits=None)
            raise AssertionError('core accepted failed native float conversion') from None
        self.check(field, value, 'number_conversion', bits=list(struct.pack('>d', number)))
        return number

    def numbers(self, field: str, member: str, value: Any) -> Any:
        if self.optional(field, value):
            return None
        self.check(field, value, 'object', mapping=isinstance(value, Mapping))
        return {str(key): self.number(member, item) for key, item in value.items()}

    def array(self, field: str, member: str, value: Any) -> Any:
        if self.optional(field, value):
            return None
        string = isinstance(value, str)
        byte_string = not string and isinstance(value, bytes)
        mapping = not (string or byte_string) and isinstance(value, Mapping)
        iterable = not (string or byte_string or mapping) and isinstance(value, Iterable)
        self.check(field, value, 'sequence', string=string, bytes=byte_string, mapping=mapping, iterable=iterable)
        return tuple(self.number(member, item) for item in value)

    def copied(self, field: str, value: Any) -> Any:
        if self.optional(field, value):
            return None
        return tuple(dict(item) for item in value) if field == 'rankings' else dict(value)

    def item(self, value: Any) -> PredictionItem:
        self.check('row', value, 'object', mapping=isinstance(value, Mapping))
        identifier = value.get('id')
        identifier = None if self.optional('row_id', identifier) else self.text('row_id', identifier)
        row_index = self.whole('row_index', self.required(value, 'row_index', 'row_index'))
        forecast_step = self.whole('forecast_step', value.get('forecast_step'))
        prediction = value.get('prediction')
        probabilities = self.numbers('probabilities', 'probability', value.get('probabilities'))
        scores = self.array('scores', 'score', value.get('scores'))
        rankings = self.copied('rankings', value.get('rankings'))
        embeddings = self.array('embeddings', 'embedding', value.get('embeddings'))
        quantiles = self.numbers('quantiles', 'quantile', value.get('quantiles'))
        explanation = self.copied('explanation', value.get('explanation'))
        metadata = self.copied('row_metadata', value.get('metadata'))
        return PredictionItem(row_index=row_index, id=identifier, forecast_step=forecast_step, prediction=prediction, probabilities=probabilities, scores=scores, rankings=rankings, embeddings=embeddings, quantiles=quantiles, explanation=explanation, metadata=metadata)

    def response(self, value: Any, model_id: str) -> NativeResponse:
        try:
            self.check('response', value, 'object', mapping=isinstance(value, Mapping))
            request_id = self.text('response_id', self.required(value, 'id', 'response_id'))
            model = self.text('model', self.required(value, 'model', 'model'))
            items = self.required(value, 'predictions', 'predictions')
            string = isinstance(items, str)
            byte_string = not string and isinstance(items, bytes)
            iterable = not (string or byte_string) and hasattr(items, '__iter__')
            self.check('predictions', items, 'sequence', string=string, bytes=byte_string, mapping=False, iterable=iterable)
            parsed = tuple(self.item(item) for item in items)
            metadata = self.required(value, 'metadata', 'metadata')
            self.check('metadata', metadata, 'object', mapping=isinstance(metadata, Mapping))
            return NativeResponse(request_id, model, parsed, dict(metadata))
        except (ResponseContractError, TypeError, ValueError) as error:
            raise StructuredDataError(
                f'{model_id} returned a response that does not match the contract: {error}',
                code='INVALID_RESPONSE',
            ) from error


def finish(
    finished: Any, client: Any, model_id: str, options: Any,
    call: Callable[..., Any], run_operation: Callable[..., Any],
    attribute: Callable[..., Any],
) -> Any:
    import dataclasses
    import json

    from sdm_client._rust import codec, results
    from sdm_client._rust.predict import _adapter
    from sdm_client.result import PredictionResult

    prepared = results.expect(finished.result, results.PreparedDirectCall)
    description = json.loads(finished.take(prepared.value))
    payload_bytes = finished.take(description['payload'])
    payload = json.loads(payload_bytes)
    payload['inference'] = options
    # The native transport owns open checks, serialization and endpoint wrapping.
    response = Reader(call).response(client._send(_adapter(client, model_id), payload), model_id)
    description['payload'] = 2
    identity = {
        'model': response.model,
        'request_id': response.request_id,
        'rows': [
            {'row_index': str(item.row_index), 'id': item.id}
            for item in response.items
        ],
    }

    def layout(attributed: Any) -> Any:
        answer = run_operation(
            'layout_native_direct',
            {'call': description, 'response': identity, 'attributed': attributed},
            {2: payload_bytes}, payload=2,
        )
        value = results.expect(answer.result, results.DirectValue)
        return json.loads(answer.take(value.value))

    decoded = layout(None)
    if any(item.id is not None and type(item.id) is not str for item in response.items):
        attributed = []
        for ordinal, item in enumerate(response.items):
            row = decoded['rows'][ordinal]
            required = call('check_native_direct_rows', {'scope': decoded['scope'], 'rows': [row]})[0]
            if required and type(item.id) is not str:
                identity['rows'][ordinal]['id'] = str(item.id)
                # Only effectful identifier subclasses need a refreshed view.
                decoded = layout(None)
                row = decoded['rows'][ordinal]
            attributed.extend(attribute(decoded['scope'], [row]))
    else:
        attributed = attribute(decoded['scope'], decoded['rows'])
    completed = layout(attributed)
    items = [
        response.items[ordinal] if replacement is None else dataclasses.replace(
            response.items[ordinal], id=codec.decode_scalar(replacement)
        )
        for ordinal, replacement in completed['rows']
    ]
    restored = restore(items, completed['restore_dtype'], call)
    result = PredictionResult(
        model=response.model, request_id=response.request_id, items=restored,
        metadata=response.metadata, outputs=description['bound']['outputs'],
    )
    query = {'bound': description['bound'], 'rows': len(restored), 'present': None}
    names = call('observe_direct_outputs', query)
    present = [
        (name, result._explanation is not None if name == 'explanation' else any(
            getattr(item, name, None) not in (None, {}, [], ()) for item in restored
        ))
        for name in names
    ]
    call('observe_direct_outputs', {**query, 'present': present})
    return result


def restore(items: list[PredictionItem], dtype: str | None, call: Callable[..., Any]) -> list[PredictionItem]:
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
        if call('restore_direct_response', {'kind': 'prediction', 'same': converted is value}):
            changed['prediction'] = converted
        given = item.probabilities
        if given:
            renamed = {}
            for key, probability in given.items():
                converted_key = as_declared(key, dtype)
                renamed[key if converted_key is key else str(converted_key)] = probability
            keep = call('restore_direct_response', {
                'kind': 'classes', 'original_count': len(given), 'renamed_count': len(renamed),
            })
            selected = renamed if keep else given
            different = selected != given
            if call('restore_direct_response', {'kind': 'probabilities', 'different': bool(different)}):
                changed['probabilities'] = selected
        restored_items.append(dataclasses.replace(item, **changed) if changed else item)
    return restored_items
