# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The client holds a call to no size of its own and recovers from what the
endpoint says: a call refused as too large, by a gateway, or timed out is
asked again in halves, each with the whole context; a 422 is not."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import requests
from test_limits_and_absent_outputs import gathered
from test_tabular_direct_core import client

from sdm_client import SDMClient
from sdm_client.core.transport import _to_nim_error
from sdm_client.errors import StructuredDataError


class Scripted:
    r"""An endpoint that fails the first call as ``failure`` says, then
    answers every call with a prediction for each row it scores."""

    url = 'http://localhost'

    def __init__(self, failure: Exception) -> None:
        self.failure: Exception | None = failure
        self.calls: list[dict] = []

    def predict_body(self, body: bytes) -> bytes:
        payload = json.loads(body)
        self.calls.append(payload)
        if self.failure is not None:
            failure, self.failure = self.failure, None
            raise failure
        table = payload['predict']['instance_table']
        key = table['columns'].index('instance_id')
        return json.dumps(
            {
                'id': 'answer',
                'model': 'kumo-relational',
                'metadata': {},
                'predictions': [
                    {'id': str(row[key]), 'row_index': at, 'prediction': 1.0}
                    for at, row in enumerate(table['rows'])
                ],
            }
        ).encode()

    def predict(self, payload: dict) -> dict:
        return json.loads(self.predict_body(json.dumps(payload).encode()))

    def close(self) -> None:
        pass


def _timed_out() -> StructuredDataError:
    r"""The transport's account of an answer slower than its timeout."""
    error = StructuredDataError(
        'Request to http://localhost/v1/predictions timed out after 1 seconds',
        code='TRANSPORT_ERROR',
        details={'timeout': 1.0},
    )
    error.__cause__ = requests.ReadTimeout()
    return error


@pytest.mark.parametrize(
    'failure',
    [
        _to_nim_error(413, b'{"detail": "too large"}'),
        _to_nim_error(502, b'{}'),
        _to_nim_error(504, b'{}'),
        _timed_out(),
    ],
    ids=['413', '502', '504', 'timeout'],
)
def test_a_call_too_large_or_too_slow_is_asked_again_in_halves(failure):
    request = gathered(context_rows=30, run_mode='fast', predict_rows=4)
    endpoint = Scripted(failure)
    result = client(endpoint).predict(request, model='kumo-relational')
    assert list(result.to_pandas()['row_index']) == [0, 1, 2, 3]
    whole, *halves = endpoint.calls
    assert [len(c['predict']['instance_table']['rows']) for c in halves] == [
        2,
        2,
    ]
    assert all(c['context'] == whole['context'] for c in halves)


def test_a_422_is_not_asked_again_and_says_what_the_endpoint_said():
    request = gathered(context_rows=30, run_mode='fast', predict_rows=4)
    endpoint = Scripted(_to_nim_error(422, b'{"detail": "N_test 4 exceeds"}'))
    with pytest.raises(StructuredDataError) as refused:
        client(endpoint).predict(request, model='kumo-relational')
    assert len(endpoint.calls) == 1
    assert 'N_test 4 exceeds' in str(refused.value)


def test_a_call_of_one_row_refused_as_too_large_is_the_endpoint_s_refusal():
    request = gathered(context_rows=30, run_mode='fast', predict_rows=1)
    endpoint = Scripted(_to_nim_error(413, b'{"detail": "context too large"}'))
    with pytest.raises(StructuredDataError) as refused:
        client(endpoint).predict(request, model='kumo-relational')
    assert len(endpoint.calls) == 1
    assert 'context too large' in str(refused.value)


@contextmanager
def _nim(*failures: int) -> Iterator[tuple[SDMClient, list[dict]]]:
    r"""A NIM on a real socket, reached over the client's own HTTP transport:
    each call answered row by row, the first ones failed with ``failures``
    in turn, a 0 among them an answer too large to read."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from test_databricks_splitting import answer

    calls: list[dict] = []
    script = list(failures)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def _send(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self._send(200, b'{"data": [{"id": "kumo-tabular"}]}')

        def do_POST(self) -> None:
            length = int(self.headers.get('Content-Length') or 0)
            payload = json.loads(self.rfile.read(length))
            calls.append(payload)
            if script:
                status = script.pop(0)
                if status == 0:
                    self._send(200, b' ' * (64 * 1024 * 1024 + 1))
                else:
                    self._send(status, b'{"detail": "refused"}')
                return
            self._send(200, json.dumps(answer(payload)).encode())

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    try:
        url = f'http://127.0.0.1:{server.server_address[1]}'
        with SDMClient(url, max_retries=0) as nim:
            yield nim, calls
    finally:
        server.shutdown()
        server.server_close()


def _tabular(rows: int):
    import numpy as np
    import pandas as pd

    from sdm_client import Request

    context = pd.DataFrame(
        {
            'row_id': np.arange(40),
            'feature': np.arange(40) % 7,
            'target': np.arange(40) % 2,
        }
    )
    scoring = pd.DataFrame(
        {'row_id': np.arange(100, 100 + rows), 'feature': np.arange(rows) % 7}
    )
    return Request(
        context,
        scoring,
        target='target',
        task='classification',
    )


def test_a_tabular_request_is_scored_in_calls_of_10_000_rows():
    with _nim() as (nim, calls):
        result = nim.predict(_tabular(10_001), model='kumo-tabular')
    assert [len(c['predict']['instance_table']['rows']) for c in calls] == [
        10_000,
        1,
    ]
    assert all(c['context'] == calls[0]['context'] for c in calls)
    assert list(result.to_pandas()['row_index']) == list(range(10_001))


@pytest.mark.parametrize(
    'failure', [413, 502, 504, 0], ids=['413', '502', '504', 'answer-too-large']
)
def test_a_tabular_call_too_large_is_asked_again_in_halves(failure, caplog):
    caplog.set_level('INFO', logger='sdm_core')
    with _nim(failure) as (nim, calls):
        result = nim.predict(_tabular(4), model='kumo-tabular')
    assert [len(c['predict']['instance_table']['rows']) for c in calls] == [
        4,
        2,
        2,
    ]
    assert list(result.to_pandas()['row_index']) == [0, 1, 2, 3]
    assert any('smaller calls' in r.getMessage() for r in caplog.records)
