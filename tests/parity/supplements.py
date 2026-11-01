# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observe real transport and public result details for bounded output journeys."""

import contextlib
import dataclasses
import json
import re
from unittest.mock import patch
from urllib.parse import unquote, urlsplit


@contextlib.contextmanager
def observe_transport(url, calls):
    import requests

    origin = urlsplit(url).netloc
    original = requests.Session.send

    def send(session, request, **kwargs):
        parsed = urlsplit(request.url)
        operation = None
        if parsed.netloc == origin:
            if request.method == 'POST' and parsed.path.endswith(
                '/v1/sessions'
            ):
                operation = 'fit'
            elif request.method == 'POST' and '/v1/sessions/' in parsed.path:
                operation = 'score'
            elif request.method == 'DELETE' and '/v1/sessions/' in parsed.path:
                operation = 'close'
            elif request.method == 'POST' and parsed.path.endswith(
                '/v1/predictions'
            ):
                operation = 'predict'
        call = None
        if operation:
            call = {
                'operation': operation,
                'payload': json.loads(request.body) if request.body else None,
            }
            if operation in {'score', 'close'}:
                call['session_id'] = unquote(
                    parsed.path.split('/v1/sessions/', 1)[1].split('/', 1)[0]
                )
            calls.append(call)
        response = original(session, request, **kwargs)
        if (
            response.ok
            and call is not None
            and operation in {'predict', 'score'}
        ):
            call['response_id'] = response.json().get('id')
        elif response.ok and call is not None and operation == 'fit':
            call['session_id'] = response.json().get('session_id')
        return response

    with patch.object(requests.Session, 'send', send):
        yield


def result_details(result, scalar):
    return {
        'model': result.model,
        'request_id': result.request_id,
        'outputs': list(result.outputs),
        'items': [
            scalar(
                {
                    field.name: getattr(item, field.name)
                    for field in dataclasses.fields(item)
                }
            )
            for item in result.items
        ],
        'metadata': scalar(result.metadata),
        'explanation': scalar(result.explanation),
        'unscored_candidates': scalar(result.unscored_candidates),
    }


def holdout_key(request, entity='users'):
    if request.payloads:
        payload = request.payloads[0]
        links = [
            link
            for link in payload['schema']['relationships']
            if not link.get('source_table') and link['target_table'] == entity
        ]
        assert len(links) == 1 and len(links[0]['source_columns']) == 1
        column = links[0]['source_columns'][0]
        table = payload['context']['instance_table']
        return table['rows'][0][table['columns'].index(column)]
    table = request.context.instance_table
    key = request.schema[entity].primary_key
    return table[key].iloc[0] if key in table else table.index[0]


def assert_holdout_payloads(payloads, context_rows, held, calls):
    """Check the actual uploaded context and related rows, not mock plans."""
    fits = [call['payload'] for call in calls if call['operation'] == 'fit']
    assert len(fits) == 1
    fitted = fits[0]
    table = fitted['context']['instance_table']
    if payloads:
        original = payloads[0]
        link = next(
            link
            for link in original['schema']['relationships']
            if not link.get('source_table') and link['target_table'] == 'users'
        )
        entity = link['source_columns'][0]
        prior = original['context']['instance_table']
        entity_at = prior['columns'].index(entity)
        instance_at = prior['columns'].index('instance_id')
        removed = {
            row[instance_at] for row in prior['rows'] if row[entity_at] in held
        }
        assert removed
        assert all(
            row[table['columns'].index(entity)] not in held
            for row in table['rows']
        )
        assert len(prior['rows']) - len(table['rows']) == len(removed)
        for related in fitted['context'].get('related_tables', {}).values():
            at = related['columns'].index('instance_id')
            assert all(row[at] not in removed for row in related['rows'])
    else:
        # Tabular's wire key is the index, independently of its feature columns.
        primary = fitted['schema']['instance_table']['primary_key']
        if isinstance(primary, list):
            assert len(primary) == 1
            primary = primary[0]
        assert isinstance(primary, str)
        at = table['columns'].index(primary)
        remaining = {str(row[at]) for row in table['rows']}
        assert all(str(key) not in remaining for key in held)
        assert context_rows - len(table['rows']) == len(held)
    assert [call['operation'] for call in calls] == [
        'fit',
        'score',
        'score',
        'close',
    ]


def assert_holdout(request, held, calls):
    assert_holdout_payloads(request.payloads, request.context_rows, held, calls)


def response_groups(semantic, case):
    """Validate actual response correlations before naming volatile IDs."""
    calls = [
        call
        for call in semantic['transport']
        if call['operation'] in {'predict', 'score'}
    ]
    ids = [call.get('response_id') for call in calls]
    if not all(isinstance(value, str) and value for value in ids) or len(
        set(ids)
    ) != len(ids):
        raise ValueError('missing or repeated observed response IDs')
    groups = (
        [ids]
        if case['supplement']['kind'] == 'batch'
        else [[value] for value in ids]
    )
    details = semantic['result_details']
    if len(details) != len(groups):
        raise ValueError('result/call groups differ')
    for result, group in zip(details, groups, strict=True):
        if result['request_id'] != group[0]:
            raise ValueError('result does not name its first actual response')
        if len(group) > 1 and (
            result['metadata'].get('batch_request_ids') != group
            or result['metadata'].get('batches') != len(group)
        ):
            raise ValueError(
                'batch metadata does not name its ordered responses'
            )
    return ids, groups


def request_ids(semantic, case):
    """Bind each generated request nonce to its actual body and source metadata."""
    asked = semantic['request']
    if isinstance(asked, list) and any(
        not isinstance(item, dict) for item in asked
    ):
        raise ValueError('request metadata lacks a payload object')
    originals = (
        [item.get('metadata', {}) for item in asked]
        if isinstance(asked, list)
        else [{}]
    )
    if any('request_id' in metadata for metadata in originals):
        raise ValueError('supplement request already carries a generated nonce')
    calls = [
        call
        for call in semantic['transport']
        if call['operation'] in {'predict', 'fit', 'score'}
    ]
    if case['supplement']['kind'] == 'batch':
        if len(originals) != len(calls):
            raise ValueError('batch request metadata groups differ')
        expected = originals
    else:
        if len(originals) != 1:
            raise ValueError('non-batched request metadata groups differ')
        expected = originals * len(calls)
    ids = []
    for call, source in zip(calls, expected, strict=True):
        metadata = call['payload'].get('metadata', {})
        nonce = metadata.get('request_id')
        if (
            not isinstance(nonce, str)
            or re.fullmatch(r'sdm-[0-9a-f]{12}', nonce) is None
        ):
            raise ValueError('missing or invalid generated request ID')
        if {
            key: value for key, value in metadata.items() if key != 'request_id'
        } != source:
            raise ValueError(
                'outgoing request metadata does not match its source request'
            )
        ids.append(nonce)
    if len(set(ids)) != len(ids):
        raise ValueError('generated request ID reused across calls')
    return calls


def session_proof(semantic, case):
    if case['supplement']['kind'] != 'holdout':
        return
    calls = semantic['transport']
    if [call['operation'] for call in calls] != [
        'fit',
        'score',
        'score',
        'close',
    ]:
        raise ValueError('fitted session lifecycle is incomplete')
    handle = calls[0].get('session_id')
    if (
        not isinstance(handle, str)
        or not handle
        or any(call.get('session_id') != handle for call in calls)
    ):
        raise ValueError('scoring/close do not name the actual fitted session')
    requested = semantic['request']
    if isinstance(requested, list):
        if len(requested) != 1 or calls[0]['payload'].get(
            'schema'
        ) != requested[0].get('schema'):
            raise ValueError(
                'actual pinned schema differs from the source request'
            )
    if any(
        'schema' in call['payload']
        for call in calls
        if call['operation'] == 'score'
    ):
        raise ValueError(
            'scoring request unexpectedly redeclares pinned schema'
        )


def normalize_response_ids(semantic, case):
    ids, groups = response_groups(semantic, case)
    generated = request_ids(semantic, case)
    session_proof(semantic, case)
    names = {value: f'call:{index}' for index, value in enumerate(ids)}
    if case['supplement']['kind'] == 'holdout':
        for call in semantic['transport']:
            call['session_id'] = 'session:0'
    for index, call in enumerate(generated):
        call['payload']['metadata']['request_id'] = f'request:{index}'
    for call in semantic['transport']:
        if call['operation'] in {'predict', 'score'}:
            call['response_id'] = names[call['response_id']]
    for result, group in zip(semantic['result_details'], groups, strict=True):
        result['request_id'] = names[result['request_id']]
        if len(group) > 1:
            result['metadata']['batch_request_ids'] = [
                names[value] for value in group
            ]


def wire_inputs(record, case):
    """Validate raw-reader model calls and return detached, nonce-normalized inputs."""
    import copy

    semantic = record.get('semantic') or {}
    calls = semantic.get('wire_calls', [])
    error_stage = (record.get('error') or {}).get('stage')
    if record.get('status') != 'passed' and error_stage in {
        'source',
        'schema',
        'materialize',
    }:
        if calls:
            raise ValueError('model called before the recorded input refusal')
        return []
    fitted = case['journey'] == 'fit-predict-close'
    expected = ['fit', 'score', 'score', 'close'] if fitted else ['predict']
    if record.get('status') != 'passed' and fitted:
        if error_stage != 'fit':
            raise ValueError('unverified partial fitted-call journey')
        expected = ['fit']
    if (
        not calls
        or any(not isinstance(c, dict) for c in calls)
        or [c.get('operation') for c in calls] != expected
    ):
        raise ValueError('actual model call sequence is incomplete')
    if any(
        not isinstance(c.get('payload'), dict)
        for c in calls
        if c['operation'] != 'close'
    ):
        raise ValueError('model input payload is not an object')
    proof_case = {'supplement': {'kind': 'holdout' if fitted else 'outputs'}}
    observed = copy.deepcopy(semantic)
    observed['transport'] = observed['wire_calls']
    generated = request_ids(observed, proof_case)
    if fitted and record.get('status') == 'passed':
        session_proof(observed, proof_case)
    for index, call in enumerate(generated):
        call['payload']['metadata']['request_id'] = f'request:{index}'
    # Responses are not input fields. Session IDs have been correlated above.
    return [
        {'operation': call['operation'], 'payload': call['payload']}
        for call in observed['transport']
    ]
