# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A native call attempt, reported reply and endpoint receipt are distinct."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from test_t136_forecast_capture import RUNNER
from test_t44_prediction_observation import reader


def attempt(name, outcome='response', status=200):
    raw = json.dumps({'metadata': {'request_id': name}, 'value': name}).encode()
    return {
        'body': json.loads(raw),
        'body_sha256': hashlib.sha256(raw).hexdigest(),
        'reply': None
        if outcome is None
        else {
            'outcome': outcome,
            **({'status': status} if outcome in ('response', 'failed') else {}),
        },
        'model_endpoint': None,
    }


def receipt(call):
    return {
        'parsed': copy.deepcopy(call['body']),
        'sha256': call['body_sha256'],
    }


@pytest.mark.parametrize('outcome', ['failed', 'transport_error', None])
def test_zero_receipt_is_possible_without_a_reported_response(outcome):
    reader().check_receipts([attempt('A', outcome)], [])
    reader().check_receipts([attempt('A', outcome)], [receipt(attempt('A'))])


@pytest.mark.parametrize('status', [200, 204, 400, 503])
def test_every_reported_response_requires_its_own_receipt(status):
    call = attempt('A', status=status)
    reader().check_receipts([call], [receipt(call)])
    with pytest.raises(reader().ObservationError):
        reader().check_receipts([call], [])


def test_retries_and_adjacent_identical_attempts_have_a_valid_non_greedy_partition():
    call = attempt('A')
    reader().check_receipts([call], [receipt(call)] * 3)
    reader().check_receipts([call, copy.deepcopy(call)], [receipt(call)] * 2)
    with pytest.raises(reader().ObservationError):
        reader().check_receipts([call, copy.deepcopy(call)], [receipt(call)])


def test_identical_attempts_can_be_separated_by_a_zero_receipt_failure():
    call = attempt('A')
    attempts = [call, attempt('B', 'transport_error'), copy.deepcopy(call)]
    reader().check_receipts(attempts, [receipt(call), receipt(call)])
    with pytest.raises(reader().ObservationError):
        reader().check_receipts(attempts, [receipt(call)])


@pytest.mark.parametrize(
    'change', ['order', 'body', 'hash', 'extra', 'missing']
)
def test_no_independent_receipt_is_dropped_or_reassigned(change):
    calls = [attempt('A'), attempt('B')]
    received = [receipt(call) for call in calls]
    if change == 'order':
        received.reverse()
    elif change == 'body':
        received[0]['parsed']['value'] = 'not-the-taken-body'
    elif change == 'hash':
        received[0]['sha256'] = '0' * 64
    elif change == 'extra':
        received.append(receipt(attempt('C')))
    else:
        received.pop()
    with pytest.raises(reader().ObservationError):
        reader().check_receipts(calls, received)


def capture_attempt():
    captured = RUNNER._Capture()
    captured.started(
        json.dumps(
            {
                'operation': {
                    'op': 'predict',
                    'model': 'kumo-relational',
                    'request_id': 'A',
                    'state': {'version': 35, 'query': 1, 'payload': 2},
                }
            }
        ).encode(),
        {
            1: b'"a native query"',
            2: b'{"predict":{"instance_table":{"rows":[[1]]}}}',
        },
    )
    captured.stepped({'step': 'call_model', 'id': 7, 'body': 10})
    captured.taken(10, json.dumps(attempt('A')['body']).encode())
    return captured


@pytest.mark.parametrize('record_fetches', [False, True])
@pytest.mark.parametrize('outcome', ['response', 'failed', 'transport_error'])
def test_actual_model_reply_is_recorded_independently_of_fetch_recording(
    record_fetches, outcome
):
    captured = capture_attempt()
    captured.recording = record_fetches
    value = {
        'response': {
            'outcome': 'response',
            'status': 503,
            'content_type': 'application/json',
            'body': 1,
        },
        'failed': {
            'outcome': 'failed',
            'answer': {'status': 401, 'detail': 1, 'provider_error': 'private'},
        },
        'transport_error': {
            'outcome': 'transport_error',
            'kind': 'model_endpoint',
            'retryable': False,
        },
    }[outcome]
    captured.replied(
        json.dumps(
            {'message': 'model_reply', 'id': 7, 'outcome': value}
        ).encode(),
        {1: b'private response/driver bytes'},
    )
    [call] = captured.core()['model_calls']
    assert call['reply'] == {
        'outcome': outcome,
        **(
            {'status': 503 if outcome == 'response' else 401}
            if outcome != 'transport_error'
            else {}
        ),
    }
    assert 'private' not in json.dumps(call)
    assert captured.answered == []


def test_taken_attempt_without_reply_stays_explicitly_unobserved():
    [call] = capture_attempt().core()['model_calls']
    assert 'reply' in call
    assert call['reply'] is None


@pytest.mark.parametrize('bad', ['wrong-id', 'duplicate', 'before-take'])
def test_reply_cannot_attach_to_a_different_or_unreached_attempt(bad):
    captured = capture_attempt()
    reply = {
        'message': 'model_reply',
        'id': 8 if bad == 'wrong-id' else 7,
        'outcome': {
            'outcome': 'transport_error',
            'kind': 'model_endpoint',
            'retryable': False,
        },
    }
    if bad == 'before-take':
        captured.stepped({'step': 'call_model', 'id': 7, 'body': 11})
    if bad == 'duplicate':
        captured.replied(json.dumps(reply).encode(), {})
    with pytest.raises(RUNNER.RunnerFailure):
        captured.replied(json.dumps(reply).encode(), {})


def test_valid_reply_fact_is_kept_before_native_forwarding_rejects_it():
    class NativeFailure(Exception):
        kind = 'protocol'
        code = 'INVALID_MESSAGE'
        control = b'{"error":{}}'

    retained = NativeFailure('native reply rejected')
    captured = capture_attempt()
    owner = captured._run(None)
    made = []

    class NativeHandle:
        def reply(self, message, attachments):
            made.append((message, attachments))
            assert captured.core()['model_calls'][0]['reply'] == {
                'outcome': 'response',
                'status': 200,
            }
            raise retained

    handle = RUNNER._Handle(
        NativeHandle(),
        SimpleNamespace(CoreError=NativeFailure),
        captured,
        owner,
    )
    message = json.dumps(
        {
            'message': 'model_reply',
            'id': 7,
            'outcome': {
                'outcome': 'response',
                'status': 200,
                'content_type': 'application/json',
                'body': 1,
            },
        }
    ).encode()
    attached = {1: b'malformed native response'}
    with pytest.raises(NativeFailure) as raised:
        handle.reply(message, attached)
    assert raised.value is retained
    assert made == [(message, attached)]
    assert captured.core()['model_calls'][0]['reply'] == {
        'outcome': 'response',
        'status': 200,
    }
