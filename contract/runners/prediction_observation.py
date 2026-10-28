# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate actual prediction evidence before projecting common host meaning.

Actual operations, typed inputs and observations stay in the artifact.
Only their checked common role/model is projected for comparison or replay.
This module reads JSON data and never evaluates SDK or user-owned objects.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any


class ObservationError(ValueError):
    """Missing or contradictory prediction evidence, without user values."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ObservationError(message)


def _ordinal(value: Any) -> bool:
    return type(value) is int and value >= 0


def _ordered_json(value: bytes | str) -> Any:
    def nonfinite(_token: str) -> Any:
        raise ObservationError('prediction input has a nonfinite JSON value')

    return json.loads(
        value,
        parse_int=lambda token: ('number', token),
        parse_float=lambda token: ('number', token),
        object_pairs_hook=lambda members: ('object', members),
        parse_constant=nonfinite,
    )


def _same_snapshot(left: Any, right: Any) -> bool:
    pending = [(left, right)]
    while pending:
        left, right = pending.pop()
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            if list(left) != list(right):
                return False
            pending.extend((left[key], right[key]) for key in left)
        elif isinstance(left, list):
            if len(left) != len(right):
                return False
            pending.extend(zip(left, right, strict=True))
        elif left != right:
            return False
    return True


def observed_call_body(call: dict[str, Any], encoded: dict[str, Any]) -> None:
    """Bind independently observed input bytes and the actual posted JSON.

    The standard decoder's internal tuple tags distinguish number tokens
    from all JSON objects, including duplicate and ordered object members.
    No serialized request is reconstructed from this comparison product.
    """
    original = base64.b64decode(call['input_body_base64'], validate=True)
    require(
        hashlib.sha256(original).hexdigest() == call['input_body_sha256'],
        'prediction input bytes differ from their observed hash',
    )
    body = call['body']
    if body is None:
        return
    require(
        isinstance(body, dict)
        and type(call['body_sha256']) is str
        and len(call['body_sha256']) == 64,
        'prediction posted body/hash is unobserved',
    )
    endpoint = encoded.get('model_endpoint')
    if endpoint is None:
        require(
            call['body_sha256'] == call['input_body_sha256']
            and _same_snapshot(
                json.loads(
                    original,
                    parse_int=lambda token: {'number': token},
                    parse_float=lambda token: {'number': token},
                ),
                body,
            ),
            'prediction direct body differs from its actual input bytes',
        )
        payload = body
    else:
        require(
            isinstance(endpoint, dict)
            and set(endpoint) == {'endpoint', 'name', 'max_request_bytes'}
            and endpoint['endpoint'] == 'serving_endpoint'
            and type(endpoint['name']) is str
            and _ordinal(endpoint['max_request_bytes']),
            'prediction has an unknown model endpoint observation',
        )
        rows = body.get('dataframe_records')
        require(
            set(body) == {'dataframe_records'}
            and isinstance(rows, list)
            and len(rows) == 1
            and isinstance(rows[0], dict)
            and set(rows[0]) == {'request_json'}
            and type(rows[0]['request_json']) is str,
            'prediction serving body has no single request_json row',
        )
        inner = rows[0]['request_json']
        require(
            _ordered_json(original) == _ordered_json(inner),
            'prediction serving request differs from its actual input JSON',
        )
        payload = json.loads(inner)
    metadata = payload.get('metadata') if isinstance(payload, dict) else None
    require(
        isinstance(metadata, dict)
        and metadata.get('request_id') == call['request_id'],
        'prediction posted request ID differs from its actual typed input',
    )


def _model(value: Any) -> str:
    require(
        isinstance(value, dict)
        and value.get('phase')
        in {
            'model_resolved',
            'outputs_resolved',
            'shape_accepted',
            'task_accepted',
            'task_admitted',
            'bound',
        },
        'prediction has no actual typed model preparation',
    )
    model = value.get('model')
    require(
        type(model) is str and bool(model), 'prediction model is unobserved'
    )
    return model


def _parts(prepared: dict[str, Any]) -> list[dict[str, Any]]:
    aggregation = prepared['input']['aggregation']
    kind = aggregation['kind']
    if kind == 'calls':
        groups = [aggregation]
    else:
        require(
            kind in ('steps', 'ranking'), 'unknown final prediction aggregation'
        )
        groups = aggregation['children']
    parts = []
    for group in groups:
        require(
            isinstance(group['parts'], list),
            'prediction parts are not observed',
        )
        if group.get('known_empty', False):
            require(
                not group['parts'] and not group['counted'],
                'known-empty prediction carries model calls',
            )
        else:
            require(
                len(group['parts']) == len(group['counted'])
                and bool(group['parts']),
                'prediction response counts are incomplete',
            )
        parts.extend(group['parts'])
    return parts


def _selected(
    final: dict[str, Any], calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {call['request_id']: call for call in calls}
    selected = []
    for part in _parts(final['input']):
        scope = part['data']['scope']
        call = by_id.get(scope['call_request_id'])
        require(
            call is not None
            and call['scope'] == scope
            and call['body'] is not None,
            'final prediction references an unobserved response scope',
        )
        require(
            part['attribution']['scope'] == scope,
            'final prediction attribution changed its response scope',
        )
        selected.append(call)
    runs = [call['run'] for call in selected]
    require(
        runs == sorted(set(runs)),
        'final prediction reorders or repeats actual calls',
    )
    return selected


def _final(
    final: dict[str, Any],
    records: dict[int, dict[str, Any]],
    calls: list[dict[str, Any]],
    *,
    complete: bool,
) -> dict[str, Any]:
    record = records.get(final['run'])
    require(
        record is not None
        and record['input']['stage'] == 'finish_relational_response'
        and record['input'] == final['input']
        and record['result'] == final['data']
        and record['error'] is None,
        'final prediction differs from its actual Run result',
    )
    selected = _selected(final, calls)
    scopes = {call['request_id']: call['scope'] for call in selected}
    lineage = final['data']['lineage']
    require(isinstance(lineage, list), 'final prediction lineage is unobserved')
    for source in lineage:
        scope = scopes.get(source['call_request_id'])
        require(
            scope is not None
            and _ordinal(source['answer_ordinal'])
            and source['answer_ordinal'] < scope['response_rows'],
            'final prediction lineage names an unobserved call row',
        )
        require(
            source['source_row'] is None or _ordinal(source['source_row']),
            'final prediction native source position is malformed',
        )
    require(
        final['metadata'] is None or isinstance(final['metadata'], dict),
        'final prediction metadata observation is malformed',
    )
    table = final['result']
    if complete:
        require(
            isinstance(final['metadata'], dict) and isinstance(table, dict),
            'final prediction metadata or Arrow result is unobserved',
        )
    if table is not None:
        require(
            isinstance(table, dict),
            'final prediction Arrow observation is malformed',
        )
        names = [column['name'] for column in table['columns']]
        require(
            len(names) == len(set(names)) and 'answer_index' in names,
            'final prediction Arrow columns are incomplete',
        )
        rows = table['rows']
        require(
            len(rows) == len(lineage)
            and all(len(row) == len(names) for row in rows),
            'final prediction Arrow rows disagree with lineage',
        )
        at = names.index('answer_index')
        require(
            sorted(int(row[at]['i64']) for row in rows)
            == list(range(len(rows))),
            'final prediction Arrow indices disagree with lineage',
        )
    predicted = final['data']['predicted']
    bodies = [call['body'] for call in selected]
    return {
        'body': bodies[-1] if bodies else None,
        **({'calls': bodies} if len(bodies) > 1 else {}),
        'model': predicted['model'],
        'request_id': predicted['request_id'],
        'metadata': final['metadata'],
        'result': table,
    }


def _below(child: Any, parent: int, entries: list[dict[str, Any]]) -> bool:
    seen = set()
    while child is not None:
        require(
            _ordinal(child) and child < len(entries) and child not in seen,
            'prediction parent chain is malformed',
        )
        if child == parent:
            return True
        seen.add(child)
        child = entries[child]['parent']
    return False


def _entry(
    entry: dict[str, Any],
    ordinal: int,
    records: dict[int, dict[str, Any]],
    operations: list[str],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    require(
        entry['ordinal'] == ordinal,
        'prediction entries are not in invocation order',
    )
    parent = entry['parent']
    require(
        parent is None or (_ordinal(parent) and parent < ordinal),
        'prediction parent is not an earlier actual entry',
    )
    argument = entry['argument']
    require(
        argument == {'form': 'unobserved'}
        or (
            set(argument) == {'form', 'value'}
            and argument['form'] == 'exact_string'
            and type(argument['value']) is str
        ),
        'prediction model argument observation is malformed',
    )
    require(
        entry['completion'] in ('returned', 'raised'),
        'prediction entry remains active',
    )
    runs = entry['runs']
    require(
        isinstance(runs, list)
        and all(_ordinal(run) and run < len(operations) for run in runs)
        and runs == sorted(set(runs)),
        'prediction actual Run membership is malformed',
    )
    owned = {
        run: record
        for run, record in records.items()
        if record['entry'] == ordinal
    }
    require(
        set(owned)
        == {run for run in runs if operations[run] == 'prepare_request'},
        'prediction entry and actual preparation owners disagree',
    )
    models = set()
    calls = entry['calls']
    require(isinstance(calls, list), 'prediction actual calls are unobserved')
    call_runs = [call['run'] for call in calls]
    require(
        call_runs == sorted(set(call_runs)),
        'prediction calls repeat or change actual order',
    )
    require(
        set(call_runs)
        == {
            run
            for run, record in owned.items()
            if record['input']['stage'] == 'call_relational'
        },
        'prediction call journal differs from actual typed call starts',
    )
    ids = [call['request_id'] for call in calls]
    require(
        all(type(value) is str and bool(value) for value in ids)
        and len(ids) == len(set(ids)),
        'prediction call request IDs are malformed or repeated',
    )
    for record in owned.values():
        value = record['result']
        if isinstance(value, dict) and 'phase' in value:
            models.add(_model(value))
    for call in calls:
        record = owned[call['run']]
        encoded = record['input']['input']
        require(
            encoded['request_id'] == call['request_id'],
            'prediction input request ID differs from its actual call',
        )
        body, scope = call['body'], call['scope']
        observed_call_body(call, encoded)
        if body is None:
            require(
                entry['completion'] == 'raised'
                and scope is None
                and call['body_sha256'] is None
                and record['result'] is None,
                'prediction call body is unobserved',
            )
            continue
        models.add(_model(encoded['call']))
        if scope is None:
            require(
                entry['completion'] == 'raised' and record['result'] is None,
                'prediction completed call has no actual response scope',
            )
        else:
            require(
                isinstance(record['result'], dict)
                and record['result']['scope'] == scope
                and scope['call_request_id'] == call['request_id']
                and _ordinal(scope['response_rows']),
                'prediction response scope differs from its actual call result',
            )
    results = entry['results']
    result_runs = [final['run'] for final in results]
    require(
        result_runs == sorted(set(result_runs)),
        'prediction result Runs repeat or change order',
    )
    require(
        set(result_runs)
        == {
            run
            for run, record in owned.items()
            if record['input']['stage'] == 'finish_relational_response'
            and record['result'] is not None
        },
        'prediction final result journal differs from actual typed completions',
    )
    rendered = []
    for final in results:
        rendered.append(
            _final(
                final, owned, calls, complete=entry['completion'] == 'returned'
            )
        )
        models.add(_model(final['input']['input']['bound']))
    require(
        len(models) <= 1,
        'prediction canonical model preparations contradict each other',
    )
    failure = entry.get('failure')
    if failure is not None:
        require(
            entry['completion'] == 'raised'
            and failure['run'] in records
            and _below(records[failure['run']]['entry'], ordinal, entries)
            and failure['relation'] in ('selected', 'cause', 'refused')
            and records[failure['run']]['error'] is not None,
            'prediction terminal error has no actual native presentation association',
        )
    native = bool(models) or (failure is not None and failure['run'] in owned)
    final = entry['final']
    if entry['completion'] == 'returned' and native:
        require(
            _ordinal(final) and final == len(results) - 1,
            'prediction returned without its final typed result',
        )
        require(
            [call['request_id'] for call in _selected(results[final], calls)]
            == ids,
            'prediction final result omits actual paid calls',
        )
    else:
        require(
            final is None,
            'a failed or unprepared entry claims a final prediction',
        )
    return {
        'ordinal': ordinal,
        'parent': parent,
        'model': next(iter(models)) if models else None,
        'native': native,
        'completion': entry['completion'],
        'failure': failure,
        '_rendered': rendered[final] if final is not None else None,
    }


def _view(core: dict[str, Any], operations: list[str]) -> dict[str, Any]:
    inputs = core['inputs']
    records = {}
    for record in inputs:
        if record['op'] != 'prepare_request':
            continue
        run = record['run']
        require(
            _ordinal(run)
            and run < len(operations)
            and operations[run] == 'prepare_request'
            and run not in records,
            'actual preparation Run is missing or duplicated',
        )
        require(
            not (record['result'] is not None and record['error'] is not None),
            'actual preparation has both a result and a native error',
        )
        records[run] = record
    require(
        set(records)
        == {
            at
            for at, operation in enumerate(operations)
            if operation == 'prepare_request'
        },
        'actual preparation operations have incomplete input evidence',
    )
    entries = core.get('predictions', [])
    require(
        isinstance(entries, list),
        'prediction entries are not observed as a list',
    )
    projected = [
        _entry(entry, at, records, operations, entries)
        for at, entry in enumerate(entries)
    ]
    owners = {}
    for entry in entries:
        for run in entry['runs']:
            require(
                run not in owners,
                'an actual Run belongs to more than one prediction entry',
            )
            owners[run] = entry['ordinal']
    for run, record in records.items():
        require(
            record['entry'] == owners.get(run),
            'actual preparation has a missing prediction owner',
        )
    # Only proved native preparation earns a common prediction role. Actual
    # nonprediction operations and unrelated preparation remain fully visible.
    native = {entry['ordinal']: entry for entry in projected if entry['native']}
    emitted = set()
    logical_operations = []
    for run, operation in enumerate(operations):
        entry = native.get(owners.get(run))
        if operation == 'prepare_request' and entry is not None:
            if entry['ordinal'] not in emitted:
                logical_operations.append('predict')
                emitted.add(entry['ordinal'])
        else:
            logical_operations.append(operation)
    logical_inputs = []
    emitted = set()
    for record in inputs:
        if record['op'] == 'predict':
            state = record['state']
            if core.get('materialized') is not None:
                require(
                    state == core['materialized'],
                    'legacy prediction state differs from its own materialized answer',
                )
            logical_inputs.append({'op': 'predict', 'model': record['model']})
        elif record['op'] == 'prepare_request' and record['entry'] in native:
            entry = native[record['entry']]
            if entry['ordinal'] not in emitted:
                logical_inputs.append(
                    {'op': 'predict', 'model': entry['model']}
                )
                emitted.add(entry['ordinal'])
        else:
            logical_inputs.append(record)
    last = next(
        (entry for entry in reversed(projected) if entry['parent'] is None),
        None,
    )
    if last is not None:
        require(
            core.get('predicted') == last['_rendered'],
            'common prediction differs from its actual final Arrow/metadata/result observations',
        )
        error = None
        if last['failure'] is not None:
            error = records[last['failure']['run']]['error']
        require(
            core.get('error') == error,
            'common prediction error lacks an actual selected native owner',
        )
    for entry in projected:
        del entry['_rendered']
    return {
        'operations': logical_operations,
        'inputs': logical_inputs,
        'predictions': projected,
    }


def view(core: dict[str, Any], operations: list[str]) -> dict[str, Any]:
    """Read only actual evidence; malformed evidence never receives a role."""
    try:
        return _view(core, operations)
    except ObservationError:
        raise
    except (
        KeyError,
        TypeError,
        ValueError,
        IndexError,
        AttributeError,
    ) as error:
        raise ObservationError(
            'prediction evidence has an invalid closed shape'
        ) from error
