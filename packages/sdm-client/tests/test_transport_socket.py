# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Transport behaviour that only a real socket can show.

``requests_mock`` mounts its own adapter, which bypasses the urllib3 ``Retry``
policy and the socket timeout entirely, so retry and timeout tests here talk to
a local ``http.server`` instead.
"""

from __future__ import annotations

import gzip
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import urllib3.util.retry

from sdm_client.core.transport import (
    _MAX_RESPONSE_BYTES,
    _RETRY_AFTER_MAX_SECONDS,
    Transport,
    _build_retry,
)
from sdm_client.errors import NimRequestError, StructuredDataError

# These bind a local socket, which a sandboxed or locked-down runner may
# refuse. Marked so such a runner can deselect them rather than report a
# failure it cannot act on.
pytestmark = pytest.mark.socket

_REFUSED_URL = 'http://127.0.0.1:1'
_RETRY_AFTER_SECONDS = 3600
_RETRY_AFTER_CAP_APPLIES = (
    getattr(_build_retry(1, 0.0), 'retry_after_max', None)
    == _RETRY_AFTER_MAX_SECONDS
)


def _predictions_body(size: int) -> bytes:
    r"""A syntactically valid prediction response of roughly ``size`` bytes."""
    filler = b'0,' * (max(size - 18, 0) // 2)
    return b'{"predictions":[' + filler + b'0]}'


@dataclass
class _Reply:
    status: int = 200
    body: str = '{"predictions": []}'
    raw: bytes | None = None
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0.0


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
        script = self.server.script
        reply = script.pop(0) if len(script) > 1 else script[0]
        self.server.received.append(self.path)
        if reply.delay:
            time.sleep(reply.delay)
        body = reply.raw if reply.raw is not None else reply.body.encode()
        try:
            self.send_response(reply.status)
            for name, value in reply.headers.items():
                self.send_header(name, value)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            pass


@contextmanager
def _serve(*script: _Reply) -> Iterator[tuple[ThreadingHTTPServer, str]]:
    server = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
    server.script = list(script)
    server.received = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()


def test_retry_policy_retries_transient_status_on_a_real_socket():
    with _serve(_Reply(status=503), _Reply(status=503), _Reply()) as (
        server,
        url,
    ):
        transport = Transport(url, max_retries=3, backoff_factor=0.0)
        assert transport.predict({'model': 'kumo-tabular'}) == {
            'predictions': []
        }
    assert server.received == ['/v1/predictions'] * 3


def test_retry_after_header_is_capped(monkeypatch):
    r"""A server-chosen ``Retry-After`` must not park the caller for an hour.

    ``Transport._build_retry`` can only cap it on urllib3 >= 2.3; older
    releases have no ``retry_after_max`` and honour the header in full, so the
    expected sleep follows the policy the transport actually built.
    """
    slept: list[float] = []
    monkeypatch.setattr(urllib3.util.retry.time, 'sleep', slept.append)

    with _serve(
        _Reply(status=503, headers={'Retry-After': str(_RETRY_AFTER_SECONDS)}),
        _Reply(),
    ) as (_, url):
        Transport(url, max_retries=1, backoff_factor=0.0).predict(
            {'model': 'kumo-tabular'}
        )

    assert slept == [
        _RETRY_AFTER_MAX_SECONDS
        if _RETRY_AFTER_CAP_APPLIES
        else _RETRY_AFTER_SECONDS
    ]


def test_timeout_bounds_a_slow_response():
    with _serve(_Reply(delay=5.0)) as (_, url):
        transport = Transport(url, timeout=0.5, max_retries=0)
        started = time.monotonic()
        with pytest.raises(StructuredDataError) as excinfo:
            transport.predict({'model': 'kumo-tabular'})
        elapsed = time.monotonic() - started
    assert excinfo.value.code == 'TRANSPORT_ERROR'
    assert elapsed < 4.0


def test_health_ready_wraps_unreachable_host():
    transport = Transport(_REFUSED_URL, timeout=2.0, max_retries=0)
    with pytest.raises(StructuredDataError) as excinfo:
        transport.health_ready()
    assert excinfo.value.code == 'TRANSPORT_ERROR'


def test_gzip_bomb_is_refused_instead_of_inflated():
    r"""A small compressed body that inflates past the cap must not be parsed.

    Without the cap ``requests`` inflates this to ``_MAX_RESPONSE_BYTES + 1``
    bytes of valid JSON and ``predict`` returns it.
    """
    bomb = gzip.compress(
        _predictions_body(_MAX_RESPONSE_BYTES + 1), compresslevel=1
    )
    assert len(bomb) < 1024 * 1024

    with _serve(_Reply(raw=bomb, headers={'Content-Encoding': 'gzip'})) as (
        _,
        url,
    ):
        transport = Transport(url, max_retries=0)
        with pytest.raises(StructuredDataError) as excinfo:
            transport.predict({'model': 'kumo-tabular'})

    assert excinfo.value.code == 'TRANSPORT_ERROR'
    assert 'exceeds' in str(excinfo.value)


@pytest.mark.parametrize('encode', [False, True])
def test_large_legitimate_response_is_returned_intact(encode):
    r"""The cap must be generous: an 8 MB prediction body still parses."""
    body = _predictions_body(8 * 1024 * 1024)
    headers = {'Content-Encoding': 'gzip'} if encode else {}
    raw = gzip.compress(body, compresslevel=1) if encode else body

    with _serve(_Reply(raw=raw, headers=headers)) as (_, url):
        transport = Transport(url, max_retries=0)
        result = transport.predict({'model': 'kumo-tabular'})

    assert len(body) > 8 * 1024 * 1024
    assert len(result['predictions']) == body.count(b',') + 1


def test_oversized_error_body_is_refused_before_it_becomes_a_message():
    bomb = gzip.compress(b'x' * (_MAX_RESPONSE_BYTES + 1), compresslevel=1)

    with _serve(
        _Reply(status=500, raw=bomb, headers={'Content-Encoding': 'gzip'})
    ) as (_, url):
        transport = Transport(url, max_retries=0)
        with pytest.raises(StructuredDataError) as excinfo:
            transport.predict({'model': 'kumo-tabular'})

    assert not isinstance(excinfo.value, NimRequestError)
    assert excinfo.value.code == 'TRANSPORT_ERROR'


def test_deeply_nested_json_stays_inside_the_error_contract():
    r"""``json`` raises ``RecursionError`` -- a ``RuntimeError``, not a
    ``ValueError`` -- so ``except StructuredDataError`` used to miss it entirely.
    """
    depth = 200_000
    nested = b'[' * depth + b']' * depth

    with _serve(_Reply(raw=nested)) as (_, url):
        transport = Transport(url, max_retries=0)
        with pytest.raises(StructuredDataError) as excinfo:
            transport.predict({'model': 'kumo-tabular'})

    assert excinfo.value.code == 'TRANSPORT_ERROR'


def test_closed_transport_stops_issuing_requests():
    with _serve(_Reply()) as (server, url):
        transport = Transport(url, max_retries=0)
        transport.predict({'model': 'kumo-tabular'})
        transport.close()
        with pytest.raises(StructuredDataError) as excinfo:
            transport.predict({'model': 'kumo-tabular'})
        with pytest.raises(StructuredDataError):
            transport.health_ready()
    assert excinfo.value.code == 'INVALID_CONFIGURATION'
    assert server.received == ['/v1/predictions']


def test_a_slow_session_create_is_not_retried():
    """Creating a session fits and pins the context before the response is
    written, so a read timeout that hides the answer from the client leaves the
    session behind on the NIM. Retrying makes one such orphan per attempt, and
    the client keeps at most the last id, so the earlier ones can never be
    released. The server-side request log is the ground truth here: the client
    genuinely cannot see the sessions it stranded.
    """
    with _serve(
        _Reply(delay=2.0),
        _Reply(delay=2.0),
        _Reply(body='{"session_id": "s2"}'),
    ) as (server, url):
        transport = Transport(
            url, timeout=0.5, max_retries=2, backoff_factor=0.0
        )
        with pytest.raises(StructuredDataError) as excinfo:
            transport.create_session({'model': 'kumo-tabular'})

    assert excinfo.value.code == 'TRANSPORT_ERROR'
    assert server.received == ['/v1/sessions']


def test_the_replayable_session_routes_keep_the_retry_policy():
    r"""Only the create is held out. Scoring against a pinned context is
    replayable, and must not lose its resilience to the narrower policy mounted
    on the prefix above it -- nor may the plain prediction route, which shares
    no prefix with either.
    """
    with _serve(_Reply(status=503), _Reply()) as (server, url):
        transport = Transport(url, max_retries=2, backoff_factor=0.0)
        transport.session_predict('s1', {'predict': {}})
    assert server.received == ['/v1/sessions/s1/predictions'] * 2

    with _serve(_Reply(status=503), _Reply()) as (server, url):
        transport = Transport(url, max_retries=2, backoff_factor=0.0)
        transport.predict({'model': 'kumo-tabular'})
    assert server.received == ['/v1/predictions'] * 2


def test_session_create_still_retries_a_connection_that_never_established():
    r"""Holding ``POST`` out of the retryable methods switches off the read and
    status retries but not the connect ones, which is the intent: a request
    that never reached the server cannot have pinned anything.
    """
    policy = _build_retry(3, 0.0, retry_post=False)
    assert policy.connect == 3
    assert not policy._is_method_retryable('POST')
    assert policy._is_method_retryable('GET')


_EMPTY_CONTEXT = (
    '{"code": "INTERNAL_ERROR", "detail": "The kumo-relational prediction at '
    'http://127.0.0.1:8000 failed unexpectedly with RuntimeError: Failed to '
    'collect any context examples. Is your predictive query too restrictive?"}'
)


def test_a_500_naming_the_callers_mistake_is_reported_as_theirs():
    """The status says server, the code says internal, and only the sentence
    says otherwise. An orchestrator branching on the code retries a request
    that can only fail the same way."""
    with _serve(_Reply(status=500, body=_EMPTY_CONTEXT)) as (_, url):
        transport = Transport(url, max_retries=0, backoff_factor=0.0)
        with pytest.raises(NimRequestError) as raised:
            transport.predict({'model': 'kumo-relational'})
    assert raised.value.code == 'INVALID_REQUEST'


def test_the_python_exception_the_server_leaked_is_not_shown():
    """``RuntimeError`` is a type from inside the endpoint. It names nothing
    the caller can act on and reads as a crash, which is the impression this
    whole path is being corrected for."""
    with _serve(_Reply(status=500, body=_EMPTY_CONTEXT)) as (_, url):
        transport = Transport(url, max_retries=0, backoff_factor=0.0)
        with pytest.raises(NimRequestError) as raised:
            transport.predict({'model': 'kumo-relational'})
    message = raised.value.message
    assert 'RuntimeError' not in message
    assert 'failed unexpectedly' not in message
    assert message.startswith('Failed to collect any context examples')
    assert 'too restrictive' in message


def test_what_the_endpoint_called_it_is_kept_beside_the_correction():
    with _serve(_Reply(status=500, body=_EMPTY_CONTEXT)) as (_, url):
        transport = Transport(url, max_retries=0, backoff_factor=0.0)
        with pytest.raises(NimRequestError) as raised:
            transport.predict({'model': 'kumo-relational'})
    error = raised.value
    assert error.status_code == 500
    assert error.details['endpoint_code'] == 'INTERNAL_ERROR'
    assert error.details['diagnosis'] == 'EMPTY_CONTEXT'


def test_a_500_naming_nothing_is_reported_as_the_endpoints():
    """The correction must not reach past the case it is for."""
    body = '{"code": "INTERNAL_ERROR", "detail": "Unexpected server error."}'
    with _serve(_Reply(status=500, body=body)) as (_, url):
        transport = Transport(url, max_retries=0, backoff_factor=0.0)
        with pytest.raises(NimRequestError) as raised:
            transport.predict({'model': 'kumo-relational'})
    assert raised.value.code == 'INTERNAL_ERROR'


def test_a_4xx_naming_the_same_thing_is_left_alone():
    """The correction is for a server reporting a caller's mistake as its own.
    A 4xx already says whose it is."""
    with _serve(_Reply(status=422, body=_EMPTY_CONTEXT)) as (_, url):
        transport = Transport(url, max_retries=0, backoff_factor=0.0)
        with pytest.raises(NimRequestError) as raised:
            transport.predict({'model': 'kumo-relational'})
    assert raised.value.code == 'INTERNAL_ERROR'


def test_a_detail_that_is_not_text_names_no_diagnosis():
    """An object under `detail` is not the shape the contract describes, and
    rendering one to text could match this on a debug field the endpoint never
    meant as its diagnosis."""
    body = (
        '{"code": "INTERNAL_ERROR", "detail": {"debug": "failed to collect '
        'any context examples"}}'
    )
    with _serve(_Reply(status=500, body=body)) as (_, url):
        transport = Transport(url, max_retries=0, backoff_factor=0.0)
        with pytest.raises(NimRequestError) as raised:
            transport.predict({'model': 'kumo-relational'})
    assert raised.value.code == 'INTERNAL_ERROR'


def test_the_retry_budget_is_what_it_was():
    """This fix reads the body and changes nothing about how often a request
    is sent, which is the one thing a correction to an error's classification
    must not quietly alter."""
    for status, sends in ((500, 4), (503, 4), (422, 1)):
        with _serve(_Reply(status=status, body=_EMPTY_CONTEXT)) as (
            server,
            url,
        ):
            transport = Transport(url, max_retries=3, backoff_factor=0.0)
            with pytest.raises(NimRequestError):
                transport.predict({'model': 'kumo-relational'})
        assert server.received == ['/v1/predictions'] * sends, status
