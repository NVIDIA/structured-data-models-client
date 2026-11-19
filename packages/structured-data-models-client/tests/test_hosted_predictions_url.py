# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The API catalog's hosted relational model is called at its exact URL.

The catalog serves the model at one predictions URL and reads the key as a
bearer token; appending ``/v1/predictions`` to it finds no route. Every other
URL stays a base, its key sent as ``X-API-Key``. The catalog cannot be stubbed,
so the recognition is checked on its own and the exact-URL mode against a
local endpoint.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sdm_client.core import transport as transport_module
from sdm_client.core.transport import Transport
from sdm_client.errors import StructuredDataError

_CATALOG = (
    'https://ai.api.nvidia.com/v1/structured-data/nvidia/kumo-relational/'
    'predictions'
)


@pytest.mark.parametrize(
    ('url', 'hosted'),
    [
        (_CATALOG, True),
        (_CATALOG + '/', True),
        (_CATALOG.replace('ai.api', 'AI.API'), True),
        (_CATALOG.replace('https', 'http'), False),
        (_CATALOG.replace('.com/', '.com:8443/'), False),
        (_CATALOG + '?x=1', False),
        (_CATALOG + '/extra', False),
        ('https://ai.api.nvidia.com', False),
        ('https://ai.api.nvidia.com/v1/predictions', False),
        ('http://localhost:8000', False),
    ],
)
def test_only_the_catalogs_predictions_url_is_taken_as_it_is(
    url: str, hosted: bool
) -> None:
    assert transport_module._is_hosted_predictions(url) is hosted


def test_the_catalog_serves_no_sessions() -> None:
    assert Transport(_CATALOG, 'key').serves_sessions is False
    assert Transport('https://nim.example', 'key').serves_sessions is True


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
    endpoint: tuple[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    base, seen = endpoint
    monkeypatch.setattr(
        transport_module, '_is_hosted_predictions', lambda url: True
    )
    transport = Transport(
        base + '/hosted/predictions/', 'secret', max_retries=0
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
