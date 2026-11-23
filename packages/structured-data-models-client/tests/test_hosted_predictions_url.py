# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An endpoint that serves predictions at one URL, called as it is.

``exact_url`` posts predictions to the URL given rather than to
``/v1/predictions`` below it, and ``bearer_token`` sends the key as
``Authorization: Bearer`` rather than ``X-API-Key``: what a hosted model in
the NVIDIA API catalog needs. No URL is recognised: without the options,
every URL is a base, its key sent as ``X-API-Key``.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sdm_client import SDMClient
from sdm_client.core.transport import Transport
from sdm_client.errors import StructuredDataError

_CATALOG = (
    'https://ai.api.nvidia.com/v1/structured-data/nvidia/kumo-relational/'
    'predictions'
)


def test_no_url_is_taken_as_exact_without_the_option() -> None:
    assert Transport(_CATALOG, 'key').serves_sessions is True
    assert Transport(_CATALOG, 'key', exact_url=True).serves_sessions is False


def test_the_client_hands_its_options_to_its_transport() -> None:
    client = SDMClient(_CATALOG, 'key', exact_url=True, bearer_token=True)
    try:
        assert client._transport.serves_sessions is False
        assert client._transport._session.headers['Authorization'] == (
            'Bearer key'
        )
        assert 'X-API-Key' not in client._transport._session.headers
    finally:
        client.close()


@pytest.fixture
def endpoint() -> Iterator[
    tuple[str, list[tuple[str, str, str | None, str | None]]]
]:
    seen: list[tuple[str, str, str | None, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - the stdlib's name
            self.rfile.read(int(self.headers.get('Content-Length', 0)))
            seen.append(
                (
                    'POST',
                    self.path,
                    self.headers.get('Authorization'),
                    self.headers.get('X-API-Key'),
                )
            )
            body = b'{"predictions": []}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}', seen
    finally:
        server.shutdown()
        server.server_close()


def test_an_exact_url_is_posted_to_as_it_is_with_a_bearer_token(
    endpoint: tuple[str, list],
) -> None:
    base, seen = endpoint
    transport = Transport(
        base + '/hosted/predictions/',
        'secret',
        max_retries=0,
        exact_url=True,
        bearer_token=True,
    )
    try:
        assert transport.predict_body(b'{}') == b'{"predictions": []}'
        assert seen == [('POST', '/hosted/predictions', 'Bearer secret', None)]
        assert transport.serves_sessions is False
        for call, code in (
            (lambda: transport.create_session({}), 'SESSIONS_UNSUPPORTED'),
            (lambda: transport.delete_session('s'), 'SESSIONS_UNSUPPORTED'),
            (transport.list_models, 'UNSUPPORTED_REQUEST'),
            (transport.health_ready, 'UNSUPPORTED_REQUEST'),
        ):
            with pytest.raises(StructuredDataError) as raised:
                call()
            assert raised.value.code == code
        assert len(seen) == 1
    finally:
        transport.close()


def test_a_base_url_keeps_its_routes_and_header(
    endpoint: tuple[str, list],
) -> None:
    base, seen = endpoint
    transport = Transport(base + '/', 'secret', max_retries=0)
    try:
        transport.predict_body(b'{}')
        assert seen == [('POST', '/v1/predictions', None, 'secret')]
        assert transport.serves_sessions is True
    finally:
        transport.close()


def test_each_option_stands_alone(endpoint: tuple[str, list]) -> None:
    base, seen = endpoint
    exact = Transport(base + '/p', 'secret', max_retries=0, exact_url=True)
    bearer = Transport(base, 'secret', max_retries=0, bearer_token=True)
    try:
        exact.predict_body(b'{}')
        bearer.predict_body(b'{}')
        assert seen == [
            ('POST', '/p', None, 'secret'),
            ('POST', '/v1/predictions', 'Bearer secret', None),
        ]
        assert bearer.serves_sessions is True
    finally:
        exact.close()
        bearer.close()
