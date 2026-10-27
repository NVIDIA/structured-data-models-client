# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reached native observations for gathered request prediction.

Native containers remain owned locally.
Rust interprets task meaning, equality populations and model policy.
No hash, equality or row traversal is replayed to encode the observations.
"""

from __future__ import annotations

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
    result = _prepared(
        {'data': 'task_meaning', 'kind': _points(kind)}
    )
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
    asked = query or '' if wire_target else ''
    name = _prepared(
        {
            'data': 'target_name',
            'population': population,
            'wire_target': wire_target,
            'query': _points(asked),
        }
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
        _prepared({'data': 'constant', 'call': call, 'target': name, 'display': display})


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
        len(payload.get('predict', {}).get('instance_table', {}).get('rows', []))
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
            len(payload.get('context', {}).get('instance_table', {}).get('rows', []))
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
