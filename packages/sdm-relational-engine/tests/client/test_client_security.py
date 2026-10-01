# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1
r"""``NimClient`` carries every Kumo Relational prediction, so it must enforce the
credential guards the other client documents, and importing the package must
not reach the network.
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
from sdm_relational_engine.client.client import (
    _MAX_RESPONSE_BYTES,
    NimClient,
    _raise_init_error,
)
from sdm_relational_engine.exceptions import InvalidResponseError

_READY = b'{"status": "ready"}'
_MODELS = b'{"object": "list", "data": [{"id": "kumo-relational"}]}'


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_GET(self) -> None:
        self.server.received.append(self.path)
        body = _MODELS if self.path.endswith('/v1/models') else _READY
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@contextmanager
def _serve() -> Iterator[tuple[ThreadingHTTPServer, str]]:
    server = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
    server.received = []
    thread = threading.Thread(
        # `shutdown()` blocks until the serving loop notices, and the
        # default poll interval is half a second. Every server in this
        # file is torn down per test.
        target=lambda: server.serve_forever(poll_interval=0.01),
        daemon=True,
    )
    thread.start()
    try:
        yield server, f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()


def test_plaintext_url_with_an_api_key_is_refused() -> None:
    with pytest.raises(ValueError, match='plaintext HTTP'):
        NimClient('http://nim.test:8000', api_key='SECRET')


@pytest.mark.parametrize(
    'url',
    [
        'https://nim.test:8000',
        'http://localhost:8000',
        'http://127.0.0.1:8000',
    ],
)
def test_supported_endpoints_still_accept_an_api_key(url: str) -> None:
    assert NimClient(url, api_key='SECRET')._url == url


def test_plaintext_url_without_an_api_key_is_allowed() -> None:
    assert NimClient('http://nim.test:8000')._url == 'http://nim.test:8000'


@pytest.mark.parametrize('url', ['nim.test:8000', 'ftp://nim.test', 'http://'])
def test_unusable_urls_are_refused(url: str) -> None:
    with pytest.raises(ValueError):
        NimClient(url)


def test_non_string_url_is_refused() -> None:
    with pytest.raises(ValueError, match='url must be a string'):
        NimClient(123)  # type: ignore[arg-type]


def test_init_error_body_is_truncated() -> None:
    response = requests.Response()
    response.status_code = 500
    response._content = b'x' * 8_000_000
    error = requests.exceptions.HTTPError(response=response)

    with pytest.raises(ValueError) as excinfo:
        _raise_init_error('https://nim.test', error)

    assert len(str(excinfo.value)) < 2_000
    assert 'truncated' in str(excinfo.value)


def test_importing_the_package_does_not_connect() -> None:
    with _serve() as (server, url):
        env = dict(os.environ)
        env['KUMO_RELATIONAL_API_ENDPOINT'] = url
        env['PYTHONPATH'] = os.pathsep.join(sys.path)
        result = subprocess.run(
            [sys.executable, '-c', 'import sdm_relational_engine'],
            env=env,
            capture_output=True,
            timeout=120,
        )

    assert result.returncode == 0, result.stderr.decode()
    assert server.received == []


def test_endpoint_env_var_still_initializes_on_first_use() -> None:
    r"""Dropping the import-time ``init()`` must not drop the env-var contract.

    Callers who set ``KUMO_RELATIONAL_API_ENDPOINT`` and never call ``init()`` explicitly
    kept working before; the connection just moves from import time to first
    use, where a network call is expected.
    """
    with _serve() as (server, url):
        env = dict(os.environ)
        env['KUMO_RELATIONAL_API_ENDPOINT'] = url
        env['PYTHONPATH'] = os.pathsep.join(sys.path)
        result = subprocess.run(
            [
                sys.executable,
                '-c',
                'import sdm_relational_engine; print(sdm_relational_engine.global_state.client._url)',
            ],
            env=env,
            capture_output=True,
            timeout=120,
        )
        received = list(server.received)

    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.decode().strip() == url
    assert received, 'first use should have authenticated'


def test_blank_endpoint_env_var_does_not_start_a_connection() -> None:
    r"""An empty ``KUMO_RELATIONAL_API_ENDPOINT`` means unconfigured, not a URL.

    A key-presence check treats ``KUMO_RELATIONAL_API_ENDPOINT=""`` as configured and
    routes the caller into a URL-parse error instead of saying no endpoint was
    given. A non-blank value that is not a URL still reaches URL validation,
    which names the offending value.
    """
    env = dict(os.environ)
    env['KUMO_RELATIONAL_API_ENDPOINT'] = ''
    env['PYTHONPATH'] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sdm_relational_engine; sdm_relational_engine.global_state.client',
        ],
        env=env,
        capture_output=True,
        timeout=120,
    )

    stderr = result.stderr.decode()
    assert result.returncode != 0
    assert 'must start with http' not in stderr, stderr
    assert 'Client creation or authentication failed' in stderr, stderr


def test_explicit_init_with_a_blank_endpoint_names_the_env_var() -> None:
    r"""``init()`` resolves the variable itself, so the same blank value has
    to reach its own missing-endpoint message rather than URL validation.
    """
    env = dict(os.environ)
    env['KUMO_RELATIONAL_API_ENDPOINT'] = ''
    env['PYTHONPATH'] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sdm_relational_engine; sdm_relational_engine.init()',
        ],
        env=env,
        capture_output=True,
        timeout=120,
    )

    stderr = result.stderr.decode()
    assert result.returncode != 0
    assert 'no endpoint URL was provided' in stderr, stderr


# The API key must not follow a redirect to another origin. ``requests_mock``
# short-circuits the adapter that resolves redirects, so these run over real
# sockets against two independent servers.


def _make_redirect_handler(state: dict) -> type[BaseHTTPRequestHandler]:
    class _RedirectHandler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def _respond(self) -> None:
            state['headers'].append(dict(self.headers))
            self.rfile.read(int(self.headers.get('Content-Length') or 0))
            location = state.get('redirect_to')
            if location and not state['redirected']:
                state['redirected'] = True
                self.send_response(307)
                self.send_header('Location', location)
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            body = _MODELS if self.path.endswith('/v1/models') else _READY
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = _respond
        do_POST = _respond

        def log_message(self, *args: object) -> None:
            pass

    return _RedirectHandler


@pytest.fixture
def redirect_servers() -> Iterator[tuple[dict, dict]]:
    states: list[dict] = []
    running: list[ThreadingHTTPServer] = []
    for _ in range(2):
        state: dict = {'headers': [], 'redirected': False}
        server = ThreadingHTTPServer(
            ('127.0.0.1', 0), _make_redirect_handler(state)
        )
        state['url'] = f'http://localhost:{server.server_address[1]}'
        thread = threading.Thread(
            # `shutdown()` blocks until the serving loop notices, and the
            # default poll interval is half a second. Every server in this
            # file is torn down per test.
            target=lambda: server.serve_forever(poll_interval=0.01),
            daemon=True,
        )
        thread.start()
        states.append(state)
        running.append(server)
    try:
        yield states[0], states[1]
    finally:
        for server in running:
            server.shutdown()
            server.server_close()


def test_api_key_is_not_forwarded_across_a_cross_origin_redirect(
    redirect_servers: tuple[dict, dict],
) -> None:
    origin, target = redirect_servers
    origin['redirect_to'] = target['url'] + '/v1/health/ready'

    NimClient(origin['url'], api_key='SECRET-TEST-KEY').authenticate()

    assert origin['headers'][0]['X-API-Key'] == 'SECRET-TEST-KEY'
    assert target['headers']
    assert all(
        headers.get('X-API-Key') is None for headers in target['headers']
    )


def test_api_key_is_kept_on_a_same_origin_redirect(
    redirect_servers: tuple[dict, dict],
) -> None:
    origin, _ = redirect_servers
    origin['redirect_to'] = origin['url'] + '/v1/health/ready/'

    NimClient(origin['url'], api_key='SECRET-TEST-KEY').authenticate()

    assert len(origin['headers']) >= 2
    assert all(
        headers['X-API-Key'] == 'SECRET-TEST-KEY'
        for headers in origin['headers']
    )


def test_redirects_are_still_followed(
    redirect_servers: tuple[dict, dict],
) -> None:
    r"""``allow_redirects=False`` would be the wrong fix: a legitimate 307/308
    still has to be followed, just without the credential. Readiness is proven
    from the redirect target's body, so a dropped chain would raise here.
    """
    origin, target = redirect_servers
    origin['redirect_to'] = target['url'] + '/v1/health/ready'

    NimClient(origin['url']).authenticate()

    assert len(target['headers']) == 1
    assert len(origin['headers']) == 2


# A hostile server must not be able to inflate a small compressed body into
# unbounded client memory. `requests_mock` never reaches the decompression
# path, so this runs over a real socket.


def _make_body_handler(state: dict) -> type[BaseHTTPRequestHandler]:
    class _BodyHandler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def _respond(self) -> None:
            self.rfile.read(int(self.headers.get('Content-Length') or 0))
            body = state['body']
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            if state.get('gzip'):
                self.send_header('Content-Encoding', 'gzip')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        do_GET = _respond
        do_POST = _respond

        def log_message(self, *args: object) -> None:
            pass

    return _BodyHandler


@contextmanager
def _serve_body(body: bytes, gzip_encoded: bool = False) -> Iterator[str]:
    state = {'body': body, 'gzip': gzip_encoded}
    server = ThreadingHTTPServer(('127.0.0.1', 0), _make_body_handler(state))
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


def test_a_compressed_body_cannot_inflate_past_the_cap() -> None:
    inflated = b'{"x": "' + b'0' * (_MAX_RESPONSE_BYTES + 4096) + b'"}'
    compressed = gzip.compress(inflated, compresslevel=1)
    assert len(compressed) < _MAX_RESPONSE_BYTES // 100

    with _serve_body(compressed, gzip_encoded=True) as url:
        client = NimClient(url)
        with pytest.raises(InvalidResponseError, match='byte limit'):
            client._post('/v1/predictions', json={'model': 'kumo-relational'})


def test_a_large_legal_body_is_still_delivered_in_full() -> None:
    r"""The cap must not truncate or reject a real prediction response, and the
    capped read must leave a response the existing callers can still use.
    """
    payload = {
        'predictions': [{'id': str(i), 'row_index': i} for i in range(300_000)]
    }
    body = json.dumps(payload).encode()
    assert 8_000_000 < len(body) < _MAX_RESPONSE_BYTES

    with _serve_body(body) as url:
        client = NimClient(url)
        response = client._post(
            '/v1/predictions', json={'model': 'kumo-relational'}
        )

    assert response.status_code == 200
    assert len(response.content) == len(body)
    assert response.json() == payload
    assert response.text.startswith('{"predictions"')
    assert b''.join(response.iter_content(1 << 20)) == body


def test_ambient_endpoint_does_not_connect_under_pytest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""A test run must never be connected to a real endpoint by accident.

    The removed import-time ``init()`` skipped itself under pytest; the lazy
    path has to keep that guard, or an exported ``KUMO_RELATIONAL_API_ENDPOINT`` would
    silently point somebody's test suite at a live NIM.
    """
    import sdm_relational_engine

    with _serve() as (server, url):
        monkeypatch.setenv('KUMO_RELATIONAL_API_ENDPOINT', url)
        monkeypatch.setattr(
            sdm_relational_engine.global_state, '_url', None, raising=False
        )
        with pytest.raises(ValueError, match='Client creation'):
            sdm_relational_engine.global_state.client
        assert server.received == []
