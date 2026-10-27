# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reached native observations for gathered request prediction.

Native containers remain owned locally.
Rust interprets task meaning, equality populations and model policy.
No hash, equality or row traversal is replayed to encode the observations.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping
from typing import Any

from sdm_client._rust import errors
from sdm_client._rust.prepare import _call


def _points(value: str) -> list[int]:
    # Bypass subclass iteration/encode overrides while reading the exact
    # native Unicode value already produced at its reached observation.
    encoded = str.encode(value, 'utf-32-le', 'surrogatepass')
    return [
        int.from_bytes(encoded[offset : offset + 4], 'little')
        for offset in range(0, len(encoded), 4)
    ]


def _prepared(value: dict[str, Any]) -> dict[str, Any]:
    return _call({'stage': 'relational', 'input': value})


def prepare(request: Any, model: Any, outputs: Any) -> dict[str, Any]:
    """Read native public properties at the existing model-check boundaries."""
    from sdm_client.models import spec_for

    model_id = model if isinstance(model, str) else spec_for(model).model_id
    call = _call({'stage': 'model', 'model': model_id})
    if outputs is None:
        call = _call(
            {'stage': 'outputs', 'call': call, 'kind': request.task.kind}
        )
    else:
        call = _call(
            {'stage': 'outputs', 'call': call, 'outputs': list(tuple(outputs))}
        )
    kind = request.task.kind
    anchored = bool(request.task.options.get('anchor_time_column'))
    relationships = bool(request.schema.relationships)
    tables = len(request.schema)
    call = _call(
        {
            'stage': 'capabilities',
            'call': call,
            'kind': kind,
            'anchored': anchored,
            'relationships': relationships,
            'tables': tables,
        }
    )
    call = _call({'stage': 'task', 'call': call, 'kind': request.task.kind})
    return _call(
        {
            'stage': 'capacity',
            'call': call,
            'rows': request._context_rows_per_call(),
        }
    )


def row_count(payload: Any, half: str) -> int | None:
    """The reached native table category and length, without row traversal."""
    if not isinstance(payload, Mapping):
        return None
    split = payload.get(half)
    if not isinstance(split, Mapping):
        return None
    table = split.get('instance_table')
    if not isinstance(table, Mapping):
        return None
    rows = table.get('rows')
    return len(rows) if isinstance(rows, (list, tuple)) else None


def nothing_to_score(payloads: Any) -> bool:
    return _prepared(
        {
            'data': 'scoring_rows',
            'rows': [row_count(payload, 'predict') for payload in payloads],
        }
    )['value']


def _requested_keys(requested: Any) -> set[Any]:
    """Native mapping/dataclass observations, preserving actual comparisons."""
    if isinstance(requested, Mapping):
        return set(requested)
    if dataclasses.is_dataclass(requested) and not isinstance(requested, type):
        keys = set()
        for field in dataclasses.fields(requested):
            observed = getattr(requested, field.name)
            default = (
                field.default_factory()
                if field.default_factory is not dataclasses.MISSING
                else field.default
            )
            if observed != default:
                keys.add(field.name)
        return keys
    return set()


def _inference(
    call: dict[str, Any], fields: dict[str, Any], inference: Any, request: Any
) -> Any:
    if not isinstance(inference, dict):
        return None
    config = inference.get('inference_config')
    if not isinstance(config, dict):
        return dict(inference)
    # Sets, sorting and comparisons execute once on the actual native keys.
    # Only the policy's accepted names came from Rust.
    unsupported = sorted(set(config) - frozenset(fields['inference_options']))
    asked = _requested_keys(request.requested_inference.get('inference_config'))
    refused = sorted(set(unsupported) & asked)
    with errors.holding(relational_unsupported=lambda observed: refused):
        _prepared(
            {
                'data': 'inference',
                'call': call,
                'refused': list(range(len(refused))),
                'statistic_option': 'output_type' in refused
                if refused
                else None,
            }
        )
    trimmed = dict(inference)
    sent = {key: value for key, value in config.items() if key in asked}
    if sent:
        trimmed['inference_config'] = sent
    else:
        del trimmed['inference_config']
    return trimmed


def envelopes(
    call: dict[str, Any], request: Any, payloads: Any
) -> list[dict[str, Any]]:
    """Fixed native shallow copies of the core-selected envelope fields."""
    from sdm_client.adapters.relational import _request_id

    planned = []
    for payload in payloads:
        envelope = dict(payload)
        meaning = task(payload)
        chosen = None
        truth = None
        if meaning['summarized']:
            chosen = (payload.get('output') or {}).get('prediction_statistic')
            truth = bool(chosen)
        fields = _prepared(
            {
                'data': 'call_fields',
                'call': call,
                'meaning': meaning,
                'statistic_chosen': truth,
            }
        )['value']
        envelope['model'] = fields['model']
        output = {'fields': fields['outputs']}
        statistic = fields['statistic']
        if statistic['source'] == 'retained':
            output['prediction_statistic'] = chosen
        elif statistic['source'] == 'default':
            output['prediction_statistic'] = statistic['value']
        envelope['output'] = output
        envelope['metadata'] = {
            **(payload.get('metadata') or {}),
            'request_id': _request_id(),
        }
        inference = _inference(call, fields, envelope.get('inference'), request)
        if inference is None:
            envelope.pop('inference', None)
        else:
            envelope['inference'] = inference
        planned.append(envelope)
    return planned


def count_calls(planned: Any) -> list[int | None]:
    counted = [row_count(envelope, 'predict') for envelope in planned]
    _prepared({'data': 'call_counts', 'rows': counted})
    return counted


def encoded(transport: Any, envelope: dict[str, Any]) -> bytes:
    """Use the transport's actual native encoding, once per reached call."""
    encode = getattr(transport, 'encode_prediction', None)
    if encode is not None:
        return encode(envelope)
    # A scripted transport owns its encoding kernel. No HTTP depth walk is
    # inserted into that route; the strict JSON value is admitted by Rust.
    return json.dumps(envelope, allow_nan=False).encode()


def context(call: dict[str, Any], payload: Any) -> None:
    # This native read has the same short-circuit boundary as the original
    # empty-context check. It never iterates rows or an ignored metadata value.
    table = payload.get('context', {}).get('instance_table')
    mapping = isinstance(table, dict)
    _prepared(
        {
            'data': 'context',
            'call': call,
            'facts': {
                'mapping': mapping,
                'rows_present': bool(table.get('rows')) if mapping else False,
            },
        }
    )


def task(payload: Any) -> dict[str, Any]:
    # str() is an actual native observation at each original guard, rather
    # than a cached spelling captured by an earlier whole-payload encoder.
    kind = str(payload.get('task', {}).get('kind', ''))
    result = _prepared({'data': 'task_meaning', 'kind': _points(kind)})
    return result['value']


def _equality(payload: Any, column: Any, *, omit_none: bool) -> Any:
    table = payload.get('context', {}).get('instance_table', {})
    columns = table.get('columns') or []
    rows = table.get('rows') or []
    if not columns or not rows or column not in columns:
        return None
    position = columns.index(column)
    members = set()
    eligible = []
    origins = {}
    observed = 0
    for row in rows:
        ordinal = observed
        observed += 1
        if not (len(row) > position):
            continue
        # Variation's original comprehension reads the selected cell twice
        # when it is present. A custom row can observe those separate reads.
        if omit_none and row[position] is None:
            continue
        value = row[position]
        eligible.append(ordinal)
        # Keep no extra native value alive after the original set insertion.
        # Updating the most recent genuine source ordinal also tolerates local
        # id reuse after a discarded equal temporary has been destroyed.
        origins[id(value)] = ordinal
        try:
            members.add(value)
        finally:
            del value
    # The exact set insertion already performed native hash/equality once.
    # Set iteration and local integer-id lookup introduce no such calls.
    distinct = [origins[id(member)] for member in members]
    return {
        'observed_rows': observed,
        'eligible_rows': eligible,
        'distinct_sources': distinct,
    }


def targets(
    call: dict[str, Any], payload: Any, meaning: dict[str, Any], purpose: str
) -> tuple[dict[str, Any], Any]:
    # The role is concrete task meaning resolved by Rust, not a second
    # host table of task kinds or classifier names.
    relevant = (
        meaning['role'] == 'classification'
        if purpose == 'class_limit'
        else meaning['role'] != 'other'
    )
    column = None
    observed = None
    if relevant:
        column = payload.get('task', {}).get('target', {}).get('column_name')
        observed = _equality(payload, column, omit_none=purpose == 'variation')
    result = _prepared(
        {
            'data': 'targets',
            'call': call,
            'meaning': meaning,
            'purpose': purpose,
            'equality': observed,
        }
    )
    return result['value'], column


def constant(
    call: dict[str, Any], population: dict[str, Any], column: Any, query: Any
) -> None:
    wire_name = ''.join(map(chr, population['wire_target_name']))
    # Preserve the original native inequality before any str(column).
    wire_target = not bool(column != wire_name)
    asked = (query or '') if wire_target else ''
    if not isinstance(asked, str):
        # Native regex input admission has its own exact TypeError for bytes
        # and other objects. Matching the target grammar remains in Rust.
        re.match('', asked)
    name = _prepared(
        {
            'data': 'target_name',
            'population': population,
            'wire_target': wire_target,
            'query': _points(asked),
        }
    )['value']
    if name['naming']['kind'] == 'question':
        name = _prepared(
            {'data': 'question_meaning', 'target': name, 'truth': bool(query)}
        )['value']
    kind = name['naming']['kind']
    if kind == 'column':
        display = _points(format(str(column), ''))
    elif kind == 'question':
        # This is the native !r conversion at the reached wording boundary,
        # before the core assembles the model's constant-context sentence.
        display = _points(format(repr(query), ''))
    else:
        display = None
    with errors.holding(relational_target=lambda fields: column):
        _prepared(
            {
                'data': 'constant',
                'call': call,
                'target': name,
                'display': display,
            }
        )


def preflight(
    call: dict[str, Any], payloads: tuple[Any, ...], query: Any
) -> None:
    """Observe each useful native domain in the original adapter order."""
    for payload in payloads:
        context(call, payload)
    for payload in payloads:
        meaning = task(payload)
        targets(call, payload, meaning, 'class_limit')
    for payload in payloads:
        meaning = task(payload)
        population, column = targets(call, payload, meaning, 'variation')
        if population['classes'] == 1:
            constant(call, population, column, query)
            break
    if 'explanation' not in call['outputs']:
        return
    counts = [
        len(
            payload.get('predict', {}).get('instance_table', {}).get('rows', [])
        )
        for payload in payloads
    ]
    _prepared({'data': 'explanation_size', 'call': call, 'rows': counts})
    modes = {
        str(payload['inference']['run_mode'])
        for payload in payloads
        if isinstance(payload.get('inference'), Mapping)
        and payload['inference'].get('run_mode') is not None
    }
    if modes:
        # Preserve the native set comparison, including custom str-subclass
        # hash/equality effects. Rust consumes that observed equality and the
        # complete resulting native values to apply explanation policy.
        fast_equal = not (modes != {'fast'})
        counts = []
    else:
        fast_equal = None
        counts = [
            len(
                payload.get('context', {})
                .get('instance_table', {})
                .get('rows', [])
            )
            for payload in payloads
        ]
    _prepared(
        {
            'data': 'explanation_mode',
            'call': call,
            'modes': [_points(mode) for mode in modes],
            'fast_equal': fast_equal,
            'context_rows': counts,
        }
    )
