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


def _sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in '0123456789abcdef' for character in value)
    )


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


def _reply_obligation(reply: Any) -> int:
    if reply is None:
        return 0
    require(isinstance(reply, dict), 'model reply observation is malformed')
    outcome = reply.get('outcome')
    if outcome in ('response', 'failed'):
        require(
            set(reply) == {'outcome', 'status'}
            and _ordinal(reply['status'])
            and reply['status'] <= 65535,
            'reported model response has no actual status',
        )
        return int(outcome == 'response')
    require(
        reply == {'outcome': 'transport_error'},
        'unknown model reply observation',
    )
    return 0


def check_receipts(
    attempts: list[dict[str, Any]], receipts: list[dict[str, Any]]
) -> None:
    """Require a monotone receipt partition, without inventing retry counts."""
    bodies: dict[str, Any] = {}
    required = []
    for attempt in attempts:
        digest = attempt['body_sha256']
        require(_sha256(digest), 'attempted body hash is unobserved')
        if digest in bodies:
            require(
                _same_snapshot(bodies[digest], attempt['body']),
                'one posted hash carries contradictory body observations',
            )
        else:
            bodies[digest] = attempt['body']
        required.append(_reply_obligation(attempt['reply']))
    hashes = []
    for receipt in receipts:
        digest = receipt.get('sha256')
        require(
            _sha256(digest)
            and digest in bodies
            and _same_snapshot(bodies[digest], receipt['parsed']),
            'endpoint receipt differs from every actual attempted body/hash',
        )
        hashes.append(digest)
    # reachable[j] means earlier attempts can account for exactly the first
    # j receipts. A response needs one; other observed outcomes may have none.
    reachable = [True] + [False] * len(hashes)
    for attempt, minimum in zip(attempts, required, strict=True):
        following = [minimum == 0 and reachable[0]]
        for at, digest in enumerate(hashes, start=1):
            following.append(
                (minimum == 0 and reachable[at])
                or (
                    digest == attempt['body_sha256']
                    and (reachable[at - 1] or following[at - 1])
                )
            )
        reachable = following
    require(
        reachable[-1],
        'endpoint receipts have no ordered allocation satisfying model responses',
    )


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
        isinstance(body, dict) and _sha256(call['body_sha256']),
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
        entry['completion'] in ('returned', 'raised') and 'carried' in entry,
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
        '_calls': calls,
        '_results': rendered,
    }


def _partial_entry(
    entry: dict[str, Any],
    rendered: list[dict[str, Any]],
    carried: tuple[dict[str, Any], dict[str, Any]] | None,
) -> dict[str, Any]:
    results = []
    positions = {
        call['request_id']: at for at, call in enumerate(entry['calls'])
    }
    for raw, value in zip(entry['results'], rendered, strict=True):
        selected = _selected(raw, entry['calls'])
        results.append(
            {
                **value,
                'aggregation': raw['input']['input']['aggregation']['kind'],
                'outputs': raw['data']['predicted']['outputs'],
                'call_indices': [
                    positions[call['request_id']] for call in selected
                ],
                'scopes': [
                    {
                        'call_index': positions[call['request_id']],
                        **{
                            key: call['scope'][key]
                            for key in ('model', 'request_id', 'response_rows')
                        },
                    }
                    for call in selected
                ],
                'lineage': [
                    {
                        'call_index': positions[source['call_request_id']],
                        'answer_ordinal': source['answer_ordinal'],
                        'source_row': source['source_row'],
                    }
                    for source in raw['data']['lineage']
                ],
            }
        )
    answered = None
    if carried is not None:
        raw, value = carried
        aggregation = raw['input']['input']['aggregation']
        groups = (
            [aggregation]
            if aggregation['kind'] == 'calls'
            else aggregation['children']
        )
        rows = 0
        for group in groups:
            require(
                all(
                    item is None or _ordinal(item) for item in group['counted']
                ),
                'carried prediction has malformed actual row counts',
            )
            rows += sum(item for item in group['counted'] if item is not None)
        # This describes the proved successful SDK handover, not a second
        # readback of arbitrary public error.details or the public result.
        answered = {
            'rows': rows,
            **{
                key: value[key]
                for key in ('result', 'model', 'request_id', 'metadata')
            },
            'outputs': raw['data']['predicted']['outputs'],
        }
    return {'observed': True, 'answered': answered, 'results': results}


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
    products = {}
    for entry, projected_entry in zip(entries, projected, strict=True):
        for raw, value in zip(
            entry['results'], projected_entry['_results'], strict=True
        ):
            require(
                raw['run'] not in products,
                'a native result belongs to multiple entries',
            )
            products[raw['run']] = (raw, value)
    carries = {}
    for entry in entries:
        carry = entry['carried']
        if carry is None:
            carries[entry['ordinal']] = None
            continue
        require(
            entry['completion'] == 'raised'
            and isinstance(carry, dict)
            and set(carry) == {'run', 'relation'}
            and _ordinal(carry['run'])
            and carry['run'] in products
            and carry['relation'] in ('selected', 'cause')
            and _below(
                records[carry['run']]['entry'], entry['ordinal'], entries
            ),
            'carried prediction has no owned successful native result association',
        )
        raw, value = products[carry['run']]
        require(
            raw['metadata'] is not None and raw['result'] is not None,
            'carried prediction result was not completely observed',
        )
        carries[entry['ordinal']] = (raw, value)
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
    legacy_calls = core.get('model_calls')
    require(
        legacy_calls is None or isinstance(legacy_calls, list),
        'legacy actual model calls are malformed',
    )
    taken = []
    for call in legacy_calls or []:
        require(
            set(call) == {'run', 'body', 'body_sha256'}
            and _ordinal(call['run'])
            and call['run'] < len(operations)
            and operations[call['run']] == 'predict'
            and isinstance(call['body'], dict)
            and _sha256(call['body_sha256']),
            'legacy model call lacks an actual prediction Run/body/hash',
        )
        taken.append((call['run'], call['body'], call['body_sha256']))
    require(
        [run for run, _body, _hash in taken]
        == sorted(run for run, _body, _hash in taken),
        'legacy model call observations change actual Run order',
    )
    for entry in projected:
        taken.extend(
            (call['run'], call['body'], call['body_sha256'])
            for call in entry['_calls']
            if call['body'] is not None
        )
    taken.sort(key=lambda item: item[0])
    calls = [body for _run, body, _hash in taken]
    hashes = [hash_ for _run, _body, hash_ in taken]
    if not entries and legacy_calls is None:
        predicted = core.get('predicted')
        calls = (
            predicted.get('calls', [predicted['body']])
            if predicted is not None and predicted.get('body') is not None
            else None
        )
        hashes = None
    partial = {
        'observed': 'answered' in core and core.get('error') is not None,
        'answered': core.get('answered'),
        'results': None,
    }
    if partial['answered'] is not None:
        answered = partial['answered']
        require(
            isinstance(answered, dict)
            and set(answered)
            == {'rows', 'result', 'model', 'request_id', 'metadata', 'outputs'}
            and _ordinal(answered['rows'])
            and isinstance(answered['result'], dict)
            and isinstance(answered['result']['columns'], list)
            and isinstance(answered['result']['rows'], list)
            and type(answered['model']) is str
            and type(answered['request_id']) is str
            and isinstance(answered['metadata'], dict)
            and isinstance(answered['outputs'], list)
            and all(type(value) is str for value in answered['outputs']),
            'legacy carried partial result is malformed',
        )
    if last is not None and last['completion'] == 'raised':
        partial = _partial_entry(
            entries[last['ordinal']], last['_results'], carries[last['ordinal']]
        )
    for entry in projected:
        del entry['_rendered']
        del entry['_calls']
        del entry['_results']
    return {
        'operations': logical_operations,
        'inputs': logical_inputs,
        'predictions': projected,
        'calls': calls,
        'call_hashes': hashes,
        'partial': partial,
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
