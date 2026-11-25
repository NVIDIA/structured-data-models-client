# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rejected session capability probes never consume prediction witnesses."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runners'))
import run_java  # noqa: E402


def events_and_received(prediction_id='prediction'):
    bodies = [
        json.dumps({'metadata': {'request_id': identity}}).encode()
        for identity in ('session-probe', prediction_id)
    ]
    events = []
    for index, (endpoint, body) in enumerate(
        zip(('create_session', 'predict'), bodies, strict=True), 1
    ):
        events.extend(
            [
                {
                    'event': 'stepped',
                    'step': {
                        'step': 'call_model',
                        'endpoint': endpoint,
                        'body': index,
                    },
                },
                {
                    'event': 'taken',
                    'id': index,
                    'bytes': base64.b64encode(body).decode(),
                },
            ]
        )
    received = SimpleNamespace(
        prediction_only=True,
        requests=[
            ('POST', '/v1/sessions', bodies[0]),
            ('POST', '/v1/predictions', bodies[1]),
        ],
        responses=[404, 200],
    )
    return events, received


def capture(monkeypatch):
    actual = run_java.run_python._Capture()
    actual._request_ids = ['prediction']
    monkeypatch.setattr(run_java.run_python, '_Capture', lambda: actual)
    return actual


def test_rejected_probe_keeps_prediction_identity_and_separate_trace(
    monkeypatch,
):
    actual = capture(monkeypatch)
    events, received = events_and_received()
    run_java.check_sent({'id': 'probe'}, events, received)
    assert run_java.captured(events, received) is actual
    assert actual._request_ids == []
    assert actual._bodies == [{'metadata': {'request_id': 'prediction'}}]
    assert actual.session_probes == [
        {
            'method': 'POST',
            'path': '/v1/sessions',
            'status': 404,
            'body': received.requests[0][2],
        }
    ]


@pytest.mark.parametrize('status', [200, 201, 400, 500])
def test_only_actual_404_probe_is_excluded(status):
    events, received = events_and_received()
    received.responses[0] = status
    with pytest.raises(run_java.RunnerFailure, match='did not reject'):
        run_java.captured(events, received)


def test_prediction_request_id_remains_strict(monkeypatch):
    capture(monkeypatch)
    events, received = events_and_received('wrong')
    with pytest.raises(run_java.run_python.RunnerFailure, match='request id'):
        run_java.captured(events, received)


def test_probe_transport_bytes_and_operation_remain_strict():
    events, received = events_and_received()
    received.requests[0] = ('POST', '/v1/sessions', b'altered')
    with pytest.raises(run_java.RunnerFailure, match='other bytes'):
        run_java.check_sent({'id': 'probe'}, events, received)
    events, received = events_and_received()
    received.requests[0] = ('POST', '/other', received.requests[0][2])
    with pytest.raises(run_java.RunnerFailure, match='did not reject'):
        run_java.captured(events, received)
