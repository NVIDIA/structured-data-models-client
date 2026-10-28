# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse Java Trace facts remain observed without invented Python replies."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_t136_forecast_capture import _module
from test_t44_prediction_observation import reader

ROOT = Path(__file__).parents[2]
JAVA = _module('t44_java_capture_facts', ROOT / 'contract/runners/run_java.py')


def as_base64(value):
    return base64.b64encode(value).decode('ascii')


def sparse_trace(count):
    payload = {'predict': {'instance_table': {'rows': [[1]]}}}
    requests = [f'java-call-{at}' for at in range(count)]
    operation = {
        'op': 'predict',
        'model': 'kumo-relational',
        'request_id': requests[0],
        'state': {
            'version': 2,
            'query': 1,
            'payload': 2,
            'batches': [
                {'request_id': request_id, 'payload': 3 + at, 'rows': 1}
                for at, request_id in enumerate(requests[1:])
            ],
        },
    }
    attachments = {'1': {'bytes': as_base64(b'"a query"')}}
    for at in range(count):
        attachments[str(2 + at)] = {
            'bytes': as_base64(json.dumps(payload).encode())
        }
    events = [
        {
            'event': 'started',
            'message': as_base64(json.dumps({'operation': operation}).encode()),
            'attachments': attachments,
        }
    ]
    bodies = []
    for at, request_id in enumerate(requests):
        encoded = json.dumps(
            {'metadata': {'request_id': request_id}, 'value': at}
        ).encode()
        bodies.append(encoded)
        events.extend(
            [
                {
                    'event': 'stepped',
                    'step': {
                        'step': 'call_model',
                        'id': 7 + at,
                        'body': 10 + at,
                    },
                },
                {'event': 'taken', 'id': 10 + at, 'bytes': as_base64(encoded)},
            ]
        )
    return events, bodies


@pytest.mark.parametrize('count', [1, 2])
def test_sparse_trace_keeps_taken_calls_without_guessing_their_replies(count):
    events, bodies = sparse_trace(count)
    original = copy.deepcopy(events)
    events.append(
        {
            'event': 'failed',
            'kind': 'invalid_response',
            'code': 'INVALID_RESPONSE',
        }
    )
    captured = JAVA.captured(events)
    core = captured.core()
    assert captured.operations == ['predict']
    assert [call['run'] for call in core['model_calls']] == [0] * count
    assert [call['body_sha256'] for call in core['model_calls']] == [
        hashlib.sha256(body).hexdigest() for body in bodies
    ]
    assert [
        call['body']['metadata']['request_id'] for call in core['model_calls']
    ] == [f'java-call-{at}' for at in range(count)]
    assert all(call['reply'] is None for call in core['model_calls'])
    assert core['error'] == {
        'kind': 'invalid_response',
        'code': 'INVALID_RESPONSE',
    }
    assert core['predicted'] is None
    assert core.get('answered') is None
    [terminal] = core['legacy_predictions']
    assert terminal['run'] == 0
    assert terminal['completion'] == 'error'
    assert terminal['error'] == core['error']
    assert terminal['result'] is None
    viewed = reader().view(core, captured.operations)
    assert viewed['partial']['observed'] is False
    [event] = viewed['events']
    assert event['partial']['observed'] is False
    assert event['partial']['answered'] is None
    assert events[:-1] == original


@pytest.mark.parametrize('count', [1, 2])
def test_sparse_success_preserves_only_actually_taken_metadata(count):
    events, bodies = sparse_trace(count)
    result = {
        'op': 'predict',
        'result': 20,
        'metadata': 21,
        'model': 'actual-endpoint-model',
        'request_id': 'actual-endpoint-id',
        'outputs': ['prediction'],
    }
    metadata = {'native': 'actually taken Java bytes'}
    events.extend(
        [
            {'event': 'done', 'result': result},
            {
                'event': 'taken',
                'id': 21,
                'bytes': as_base64(json.dumps(metadata).encode()),
            },
        ]
    )
    original = copy.deepcopy(events)
    captured = JAVA.captured(events)
    core = captured.core()
    assert core['error'] is None
    assert core['predicted']['metadata'] == metadata
    assert core['predicted']['result'] is None
    assert core['predicted']['model'] == 'actual-endpoint-model'
    assert core['predicted']['request_id'] == 'actual-endpoint-id'
    assert len(core['model_calls']) == len(bodies) == count
    assert all(call['reply'] is None for call in core['model_calls'])
    [terminal] = core['legacy_predictions']
    assert terminal['completion'] == 'done'
    assert terminal['result']['metadata'] == metadata
    assert terminal['result']['result'] is None
    viewed = reader().view(core, captured.operations)
    assert len(viewed['calls']) == count
    assert events == original


def test_sparse_capture_declares_the_unavailable_interceptions():
    events, _ = sparse_trace(2)
    captured = JAVA.captured(events)
    core = captured.core()
    assert core['capture_capabilities'] == {
        'model_replies': False,
        'native_error_control': False,
    }
    assert 'answered' not in core
    assert 'answered' not in core['legacy_predictions'][0]
    assert all(call['reply'] is None for call in core['model_calls'])


@pytest.mark.parametrize('changed', ['top-answered', 'run-answered', 'reply'])
def test_sparse_unavailable_facts_cannot_be_presented_as_observed(changed):
    events, _ = sparse_trace(2)
    events.append(
        {
            'event': 'failed',
            'kind': 'invalid_response',
            'code': 'INVALID_RESPONSE',
        }
    )
    captured = JAVA.captured(events)
    core = captured.core()
    if changed == 'top-answered':
        core['answered'] = None
    elif changed == 'run-answered':
        core['legacy_predictions'][0]['answered'] = None
    else:
        core['model_calls'][0]['reply'] = {'outcome': 'response', 'status': 200}
    with pytest.raises(reader().ObservationError):
        reader().view(core, captured.operations)


@pytest.mark.parametrize(
    'changed', ['top-answered', 'run-answered', 'pending-reply']
)
def test_full_observer_cannot_lose_mandatory_available_facts(changed):
    from test_t44_prediction_delivery import capture_attempt

    captured = capture_attempt()
    core = captured.core()
    assert core['capture_capabilities'] == {
        'model_replies': True,
        'native_error_control': True,
    }
    if changed == 'top-answered':
        core.pop('answered')
    elif changed == 'run-answered':
        core['legacy_predictions'][0].pop('answered')
    else:
        core['model_calls'].append(copy.deepcopy(core['model_calls'][0]))
    with pytest.raises(reader().ObservationError):
        reader().view(core, captured.operations)
