# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The socket timeout the client configures must reach the wire.

``requests_mock`` mounts its own adapter and never opens a socket, so these
tests talk to a local ``http.server`` that stalls before replying.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
import sdm_relational_engine
import sdm_relational_engine.rfm as rfm_engine
from sdm_relational_engine.client import NimClient

_READY_BODY = json.dumps({'status': 'healthy', 'check': 'ready'})
_MODELS_BODY = json.dumps(
    {'object': 'list', 'data': [{'id': 'kumo-relational'}]}
)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_GET(self) -> None:
        self._reply()

    def do_POST(self) -> None:
        self._reply()

    def log_message(self, *args: object) -> None:
        pass

    def _reply(self) -> None:
        length = int(self.headers.get('Content-Length') or 0)
        if length:
            self.rfile.read(length)
        time.sleep(self.server.delay)
        body = self.server.bodies.get(self.path, '{}').encode()
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            pass


@contextmanager
def _serve(delay: float = 0.0) -> Iterator[str]:
    server = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
    server.delay = delay
    server.bodies = {
        '/v1/health/ready': _READY_BODY,
        '/v1/models': _MODELS_BODY,
    }
    thread = threading.Thread(
        # `shutdown()` blocks until the serving loop notices, and the
        # default poll interval is half a second. Every server in this
        # file is torn down per test.
        target=lambda: server.serve_forever(poll_interval=0.01),
        daemon=True,
    )
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()


def test_post_honours_the_configured_timeout():
    with _serve(delay=5.0) as url:
        client = NimClient(url, timeout=0.5)
        started = time.monotonic()
        with pytest.raises(requests.Timeout):
            client._post('/v1/predictions', json={})
        elapsed = time.monotonic() - started
    assert elapsed < 4.0


def test_per_call_timeout_overrides_the_client_default():
    with _serve(delay=5.0) as url:
        client = NimClient(url, timeout=30.0)
        with pytest.raises(requests.Timeout):
            client._post('/v1/predictions', json={}, timeout=0.5)


def test_init_forwards_the_timeout_to_the_request_client():
    try:
        with _serve() as url:
            rfm_engine.init(
                url=url, timeout=7.0, _token=rfm_engine._CLIENT_TOKEN
            )
            assert sdm_relational_engine.global_state.client._timeout == 7.0
    finally:
        rfm_engine.global_state.reset()
        sdm_relational_engine.global_state.clear()


def test_max_retries_reaches_the_transport_policy():
    """/
    client-max-retries-never-reaches-the-sdm_relational_engine-path.md

    `NimClient(max_retries=...)` is documented without a model qualifier, next
    to `timeout`, which was made to reach both paths. This client used to
    hardcode `total=10, connect=3, read=3, status=5` regardless, so `0` still
    retried and a raised value changed nothing, and its prediction `POST` was
    not retried on a 5xx at all, because urllib3's default `allowed_methods`
    excludes `POST`.
    """
    for max_retries in (0, 3, 7):
        client = NimClient('https://tenant.example', max_retries=max_retries)
        policy = client._session.get_adapter(
            'https://tenant.example/v1/predictions'
        ).max_retries
        assert policy.total == max_retries
        assert policy.connect == max_retries
        assert policy.status == max_retries
        assert policy._is_method_retryable('POST')
        assert 503 in policy.status_forcelist
        assert 429 in policy.status_forcelist


def test_a_server_chosen_retry_after_is_capped():
    r"""Urllib3's own ceiling is six hours, which a `Retry-After` header on a
    retried request could park the caller for.
    """
    policy = (
        NimClient('https://tenant.example')
        ._session.get_adapter('https://tenant.example/v1/predictions')
        .max_retries
    )
    cap = getattr(policy, 'retry_after_max', None)
    if cap is not None:  # urllib3 >= 2.3 only
        assert cap == 60


def test_session_create_is_held_out_of_the_post_retries():
    """Creating a session pins the context before the response is written, so a
    re-sent create strands one pinned context per attempt. The routes below it
    are replayable and keep the full policy.
    """
    client = NimClient('https://tenant.example', max_retries=3)
    create = client._session.get_adapter(
        'https://tenant.example/v1/sessions'
    ).max_retries
    assert not create._is_method_retryable('POST')
    assert create.connect == 3

    for path in (
        '/v1/predictions',
        '/v1/sessions/s1/predictions',
        '/v1/sessions/s1',
    ):
        policy = client._session.get_adapter(
            f'https://tenant.example{path}'
        ).max_retries
        assert policy._is_method_retryable('POST'), path


def test_a_read_timeout_is_not_retried_and_keeps_its_type():
    r"""A read timeout on a prediction means the NIM may already be running it,
    so resending duplicates the most expensive work in the system. It also has
    to stay a `requests.Timeout`: that is the one failure the caller can fix
    from their side, and the driver's message says so.
    """
    with _serve(delay=5.0) as url:
        client = NimClient(url, timeout=0.5, max_retries=3)
        started = time.monotonic()
        with pytest.raises(requests.Timeout):
            client._post('/v1/predictions', json={})
        elapsed = time.monotonic() - started
    assert elapsed < 4.0


def test_init_forwards_max_retries_to_the_request_client():
    try:
        with _serve() as url:
            rfm_engine.init(
                url=url, max_retries=5, _token=rfm_engine._CLIENT_TOKEN
            )
            assert sdm_relational_engine.global_state.client._max_retries == 5
    finally:
        rfm_engine.global_state.reset()
        sdm_relational_engine.global_state.clear()
