# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observe real transport and public result details for bounded output journeys."""

import contextlib
import dataclasses
import json
from unittest.mock import patch
from urllib.parse import urlsplit


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
            calls.append(call)
        response = original(session, request, **kwargs)
        if call is not None and operation in {'predict', 'score'}:
            call['response_id'] = response.json().get('id')
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


def normalize_response_ids(semantic, case):
    ids, groups = response_groups(semantic, case)
    names = {value: f'call:{index}' for index, value in enumerate(ids)}
    for call in semantic['transport']:
        if call['operation'] in {'predict', 'score'}:
            call['response_id'] = names[call['response_id']]
    for result, group in zip(semantic['result_details'], groups, strict=True):
        result['request_id'] = names[result['request_id']]
        if len(group) > 1:
            result['metadata']['batch_request_ids'] = [
                names[value] for value in group
            ]
