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
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypedDict

from sdm_client._rust import errors
from sdm_client._rust.prepare import _call


class _TaskFacts(TypedDict):
    purpose: str
    classification: bool | None
    regression_not_equal: bool | None
    summarized: bool | None


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


@dataclass(frozen=True)
class PreparedCall:
    """Core call facts beside this call's actual selected native output owners."""

    bound: dict[str, Any]
    outputs: tuple[Any, ...]


def model_call(model: Any) -> dict[str, Any]:
    """Actual native catalogue/handle observations and a core-resolved model."""
    from sdm_client.models import MODELS, spec_for

    catalogue = _call({'stage': 'model_catalogue'})
    if isinstance(model, str):
        selected_model = MODELS.get(model)
        position = next(
            (
                at
                for at, name in enumerate(catalogue)
                if selected_model is not None
                and selected_model is MODELS.get(sys.intern(name))
            ),
            None,
        )

        def refused_model(_fields: Any) -> tuple[str, Any]:
            return repr(model), model

        with errors.holding(relational_model=refused_model):
            call = _call({'stage': 'model_lookup', 'position': position})
    else:
        # Existing native handle/spec object-identity and form observations.
        # The canonical id is still admitted by the shared closed model policy.
        call = _call({'stage': 'model', 'model': spec_for(model).model_id})
    return call


def prepare(request: Any, model: Any, outputs: Any) -> PreparedCall:
    """Read native public properties at the existing model-check boundaries."""
    from sdm_client.client import _require_output_names

    # The public call and the old _prepare each reach this native Sequence
    # observation before model lookup. Do not collapse those traversals.
    _require_output_names(outputs)
    call = model_call(model)
    words = _call({'stage': 'call_vocabulary', 'call': call})
    if outputs is None:
        # Only this reached dictionary lookup observes the actual public key.
        # Its values are core vocabulary positions, not native model policy.
        defaults = {
            sys.intern(kind): position
            for position, kind in enumerate(words['default_kinds'])
        }
        position = defaults.get(request.task.kind)
        call = _call(
            {
                'stage': 'outputs',
                'call': call,
                'native_default': {'position': position},
            }
        )
        selected_outputs = tuple(sys.intern(name) for name in call['outputs'])
    else:
        selected_outputs = tuple(outputs)
        call = _call(
            {
                'stage': 'outputs',
                'call': call,
                'outputs': list(selected_outputs),
            }
        )
    kind = request.task.kind
    anchored = bool(request.task.options.get('anchor_time_column'))
    temporal = anchored or kind in frozenset(
        sys.intern(word) for word in words['temporal_tasks']
    )
    relationships = bool(request.schema.relationships)
    tables = len(request.schema)
    linked = kind == sys.intern(words['link_prediction'])
    forecasting = kind == sys.intern(words['forecasting'])
    # The original capability object retains both equality results before
    # model validation asks for their truth, in this order.
    native = {
        'temporal': temporal,
        'link_prediction': bool(linked),
        'forecasting': bool(forecasting),
    }
    call = _call(
        {
            'stage': 'capabilities',
            'call': call,
            'kind': kind,
            'anchored': anchored,
            'relationships': relationships,
            'tables': tables,
            'native': native,
        }
    )
    kind = request.task.kind
    admitted = kind in frozenset(sys.intern(word) for word in words['tasks'])

    def refused_task(_fields: Any) -> tuple[str, Any]:
        # Refusal presentation rereads the public owner: repr may itself
        # mutate it before the old exception's details take their fresh read.
        shown = repr(request.task.kind)
        return shown, request.task.kind

    with errors.holding(relational_model_task=refused_task):
        call = _call(
            {
                'stage': 'task',
                'call': call,
                'kind': kind,
                'admitted': admitted,
                'defer_outputs': True,
            }
        )
    supported_outputs = frozenset(sys.intern(name) for name in words['outputs'])
    unsupported = [
        at
        for at, name in enumerate(selected_outputs)
        if name not in supported_outputs
    ]

    def refused_outputs(_fields: Any) -> tuple[str, list[Any]]:
        unknown = [selected_outputs[at] for at in unsupported]
        shown = repr(sorted(unknown))
        return shown, sorted(unknown)

    with errors.holding(relational_model_outputs=refused_outputs):
        call = _call(
            {
                'stage': 'output_admission',
                'call': call,
                'unsupported': unsupported,
            }
        )
    bound = _call(
        {
            'stage': 'capacity',
            'call': call,
            'rows': request._context_rows_per_call(),
        }
    )
    return PreparedCall(bound, selected_outputs)


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
                'statistic_option': fields['inference_statistic_key'] in refused
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
    call: dict[str, Any],
    request: Any,
    payloads: Any,
    outputs: tuple[Any, ...] | None = None,
) -> list[dict[str, Any]]:
    """Fixed native shallow copies of the core-selected envelope fields."""
    from sdm_client.adapters.relational import _request_id

    planned = []
    for payload in payloads:
        envelope = dict(payload)
        meaning, _, _ = task(payload, 'output')
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
        output = {
            'fields': list(outputs)
            if outputs is not None
            else fields['outputs']
        }
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


def vocabulary() -> dict[str, Any]:
    return _prepared({'data': 'task_vocabulary'})['value']


def task(
    payload: Any, purpose: str
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    # Native substring/set comparisons can have observable subclass overrides.
    # Their literal vocabulary is core-owned; the host executes only these
    # actual comparisons at the original reached guard.
    words = vocabulary()
    kind = str(payload.get('task', {}).get('kind', ''))
    facts: _TaskFacts = {
        'purpose': purpose,
        'classification': None,
        'regression_not_equal': None,
        'summarized': None,
    }
    if purpose == 'output':
        facts['summarized'] = kind in frozenset(set(words['summarized']))
    else:
        facts['classification'] = words['classification'] in kind
        if purpose == 'variation' and not facts['classification']:
            facts['regression_not_equal'] = bool(kind != words['regression'])
    result = _prepared(
        {'data': 'task_meaning', 'kind': _points(kind), 'facts': facts}
    )
    return result['value'], kind, words


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
    call: dict[str, Any],
    population: dict[str, Any],
    column: Any,
    query: Any,
    kind: Any,
    words: dict[str, Any],
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
    naming_kind = name['naming']['kind']
    if naming_kind == 'column':
        display = _points(format(str(column), ''))
    elif naming_kind == 'question':
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
                'classification': words['classification'] in kind,
            }
        )


def preflight(
    call: dict[str, Any],
    payloads: tuple[Any, ...],
    query: Any,
    outputs: tuple[Any, ...] | None = None,
) -> None:
    """Observe each useful native domain in the original adapter order."""
    for payload in payloads:
        context(call, payload)
    _class_limits(call, payloads)
    _variation(call, payloads, query)
    if 'explanation' not in (
        outputs if outputs is not None else call['outputs']
    ):
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


def _class_limits(call: dict[str, Any], payloads: Any) -> None:
    for payload in payloads:
        meaning, _kind, _words = task(payload, 'class_limit')
        targets(call, payload, meaning, 'class_limit')


def _variation(call: dict[str, Any], payloads: Any, query: Any) -> None:
    for payload in payloads:
        meaning, kind, words = task(payload, 'variation')
        population, column = targets(call, payload, meaning, 'variation')
        if population['classes'] == 1:
            constant(call, population, column, query, kind, words)
            return


def absent(call: dict[str, Any], answer: Any, requested: Any) -> None:
    """Observe the reached native output-name and result-attribute populations."""
    if not requested or not answer.items:
        return
    words = _prepared({'data': 'output_vocabulary'})['value']
    names = list(dict.fromkeys(requested))
    item_fields = tuple(sys.intern(name) for name in words['item_fields'])
    factories: dict[str, Callable[[], object]] = {
        'null': lambda: None,
        'mapping': dict,
        'list': list,
        'tuple': tuple,
    }
    columns = []
    for name in names:
        meaning = _prepared(
            {
                'data': 'output_meaning',
                'explanation': bool(name == sys.intern(words['explanation'])),
                'item': None,
            }
        )['value']
        if meaning['role'] == 'candidate':
            meaning = _prepared(
                {
                    'data': 'output_meaning',
                    'explanation': meaning['explanation'],
                    'item': name in item_fields,
                }
            )['value']
        if meaning['role'] == 'explanation':
            nonempty = answer._explanation is not None
        elif meaning['role'] == 'item':
            nonempty = any(
                getattr(item, name, None)
                not in tuple(
                    factories[kind]() for kind in words['empty_values']
                )
                for item in answer.items
            )
        else:
            nonempty = None
        columns.append({'meaning': meaning, 'nonempty': nonempty})
    population = _prepared({'data': 'output_absence', 'columns': columns})[
        'value'
    ]
    if population['absent']:
        shown = repr([names[position] for position in population['absent']])
        _prepared(
            {
                'data': 'output_notice',
                'call': call,
                'population': population,
                'display': _points(shown),
            }
        )


def output_shapes(
    outputs: tuple[Any, ...], count: int
) -> tuple[set[Any], dict[str, Any]]:
    """One actual native set insertion per completed call, without replay."""
    shapes: set[tuple[Any, ...]] = set()
    origins: dict[int, int] = {}
    for position in range(count):
        origins.setdefault(id(outputs), position)
        shapes.add(outputs)
    return shapes, {
        'parts': count,
        'distinct_sources': [origins[id(shape)] for shape in shapes],
    }


def ranking_outputs(outputs: Any, total: int) -> dict[str, Any]:
    """One reached native ranking output guard and membership population."""
    words = _call(
        {
            'stage': 'ranking_output_vocabulary',
            'named': outputs is not None,
            'nonempty': bool(outputs) if outputs is not None else None,
        }
    )
    accepted = tuple(sys.intern(name) for name in words['outputs'])
    observed: list[Any] = []
    unsupported = []
    for name in outputs or ():
        position = len(observed)
        observed.append(name)
        if name not in accepted:
            unsupported.append(position)

    def refused(_fields: Any) -> list[str]:
        return [repr(observed[position]) for position in unsupported]

    with errors.holding(relational_ranking_outputs=refused):
        return _call(
            {
                'stage': 'validate_relational_ranking',
                'outputs': None if outputs is None else observed,
                'native': {'unsupported': unsupported},
                'total_parts': total,
                'cold': None,
            }
        )


def ranking_fields(outputs: tuple[Any, ...]) -> list[str]:
    words = _call(
        {'stage': 'ranking_output_vocabulary', 'named': False, 'nonempty': None}
    )
    prediction = sys.intern(words['prediction']) in outputs
    rankings = sys.intern(words['rankings']) in outputs
    return _prepared(
        {
            'data': 'ranking_fields',
            'prediction': prediction,
            'rankings': rankings,
        }
    )['value']


def forecast_explanations(
    model: Any, requested: Any, children: Any, answered: Any
) -> None:
    # Original call arguments resolve the model before entering the warning's
    # requested truth/containment guard, even when no warning will be emitted.
    resolved = model_call(model)
    words = _prepared({'data': 'output_vocabulary'})['value']
    asked = bool(requested) and sys.intern(words['explanation']) in requested
    steps = [about for _, about in answered] if asked else []
    explained = (
        [child.presented.explanation is not None for child in children]
        if asked
        else []
    )
    _prepared(
        {
            'data': 'forecast_explanations',
            'model': resolved['model'],
            'requested': asked,
            'steps': steps,
            'explained': explained,
        }
    )
