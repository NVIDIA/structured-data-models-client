# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``SDMClient.predict`` through a Databricks Model Serving endpoint on the
core (T3.13).

On the core route the core writes the endpoint's envelope, weighs it
against the caller's limit and reads every answer; the serving transport
posts the core's bytes as they are through the workspace's SDK client
(``api_client.do``, the real one here, against a local workspace) and hands
back the bytes it was answered with, or the SDK's failure. Each outcome is
the old route's for the same endpoint answering the same way: the result,
the refusal's class, status, code and message, a failure the served model
answered with, the limit, and a call the SDK retried, whose byte body is
sent again whole.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import threading
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Request,
    Schema,
    SDMClient,
    StructuredDataError,
    _routing,
)
from sdm_client.core.databricks_serving import DatabricksServingTransport
from sdm_client.errors import NimRequestError

sdk = pytest.importorskip('databricks.sdk')

ROOT = Path(__file__).resolve().parents[3]
ENDPOINT = 'kumo-rfm-serving'
INVOCATIONS = f'/serving-endpoints/{ENDPOINT}/invocations'
SECRET = 'Sup3rSecret-CustomerValue-9f2a'
QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
MATERIALIZE = ','.join(
    _routing.FROM_SOURCE_DUCKDB + _routing.MATERIALIZE_DUCKDB
)
CORE = ','.join(
    _routing.FROM_SOURCE_DUCKDB
    + _routing.MATERIALIZE_DUCKDB
    + _routing.PREDICT_DUCKDB
)


def _scripted() -> Any:
    name = 'scripted_endpoint'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / 'contract/runners/scripted_endpoint.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _old(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    # A proxied runner would otherwise send these local requests to its
    # proxy.
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('NO_PROXY', '127.0.0.1')
    monkeypatch.setenv('no_proxy', '127.0.0.1')


def _shop() -> list[str]:
    lines = (
        (ROOT / 'contract/bases.yaml').read_text(encoding='utf-8').splitlines()
    )
    start = lines.index('  shop:') + 1
    statements = []
    for line in lines[start:]:
        if not line.startswith('  - '):
            break
        statements.append(line[len('  - ') :])
    return statements


def _materialized(monkeypatch: pytest.MonkeyPatch) -> Request:
    r"""base.regression's request, materialized on the core."""
    monkeypatch.setenv(_routing.ENVIRONMENT, MATERIALIZE)
    connection = duckdb.connect()
    for statement in _shop():
        connection.execute(statement)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(connection))
        request = PredictiveQuery(QUERY, schema).materialize(
            indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01')
        )
    monkeypatch.delenv(_routing.ENVIRONMENT)
    return request


def _served(body: bytes) -> tuple[int, dict[str, Any], dict[str, str]]:
    r"""The served model's answer to one envelope, by the scripted rule,
    in the endpoint's one ``response_json`` row."""
    records = json.loads(body)['dataframe_records']
    assert len(records) == 1
    inner = records[0]['request_json'].encode()
    status, reply = _scripted().answer('POST', _scripted().PREDICTIONS, inner)
    assert status == 200, reply
    return 200, {'predictions': [{'response_json': json.dumps(reply)}]}, {}


#: A reply the workspace gives one call: its status, its body (an object
#: is sent as JSON) and its headers; or a function of the body received.
Reply = tuple[int, Any, dict[str, str]]


@contextmanager
def _workspace(
    *replies: Reply | Callable[[bytes], Reply],
) -> Iterator[tuple[Any, list[tuple[str, bytes, dict[str, str]]]]]:
    r"""A real ``WorkspaceClient`` against a local workspace answering
    each call with the next of ``replies`` (the scripted model once they
    run out), and what the workspace received: path, body and headers."""
    received: list[tuple[str, bytes, dict[str, str]]] = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            length = int(self.headers.get('Content-Length') or 0)
            body = self.rfile.read(length)
            with lock:
                at = len(received)
                received.append((self.path, body, dict(self.headers)))
            reply = replies[at] if at < len(replies) else _served
            status, answer, headers = reply(body) if callable(reply) else reply
            out = (
                answer.encode()
                if isinstance(answer, str)
                else json.dumps(answer).encode()
            )
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header('Content-Length', str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        workspace = sdk.WorkspaceClient(
            config=sdk.core.Config(
                host=f'http://127.0.0.1:{server.server_address[1]}',
                token='dapi-test',
                auth_type='pat',
                retry_timeout_seconds=20,
            )
        )
        yield workspace, received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _client(workspace: Any, max_request_bytes: int | None = None) -> SDMClient:
    if max_request_bytes is None:
        return SDMClient.for_databricks_serving(
            ENDPOINT, workspace_client=workspace
        )
    client = SDMClient.__new__(SDMClient)
    client._start(
        DatabricksServingTransport(
            ENDPOINT, workspace, max_request_bytes=max_request_bytes
        )
    )
    return client


def _predict(
    monkeypatch: pytest.MonkeyPatch,
    request: Request,
    routes: str | None,
    *replies: Reply | Callable[[bytes], Reply],
    max_request_bytes: int | None = None,
) -> tuple[Any, list[tuple[str, bytes, dict[str, str]]]]:
    r"""The call's outcome on ``routes`` (the old route with none), the
    result or the exception raised, and what the workspace received."""
    if routes is None:
        monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    else:
        monkeypatch.setenv(_routing.ENVIRONMENT, routes)
    try:
        with _workspace(*replies) as (workspace, received):
            with _client(workspace, max_request_bytes) as client:
                try:
                    outcome: Any = client.predict(
                        request, model=KumoRelational()
                    )
                except StructuredDataError as error:
                    outcome = error
    finally:
        monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    return outcome, received


def _both(
    monkeypatch: pytest.MonkeyPatch,
    request: Request,
    *replies: Reply | Callable[[bytes], Reply],
    max_request_bytes: int | None = None,
) -> tuple[Any, Any, list[Any], list[Any]]:
    core, on_core = _predict(
        monkeypatch,
        request,
        CORE,
        *replies,
        max_request_bytes=max_request_bytes,
    )
    old, on_old = _predict(
        monkeypatch,
        request,
        None,
        *replies,
        max_request_bytes=max_request_bytes,
    )
    return core, old, on_core, on_old


def _call(body: bytes) -> dict[str, Any]:
    r"""The call an envelope carries, its drawn id taken out."""
    call = json.loads(json.loads(body)['dataframe_records'][0]['request_json'])
    call['metadata'].pop('request_id')
    return call


def _same_refusal(core: Any, old: Any) -> None:
    assert isinstance(old, NimRequestError), old
    assert type(core) is type(old), core
    assert (core.status_code, core.code, core.message) == (
        old.status_code,
        old.code,
        old.message,
    )
    # Every detail the old route gives, the core route gives alike.
    for name, value in old.details.items():
        if value is not None:
            assert core.details[name] == value, name
    shown = str(core) + json.dumps(core.details, default=str)
    assert SECRET not in shown
    assert core.__cause__ is None


def test_the_core_s_envelope_is_posted_through_the_sdk_and_its_row_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _materialized(monkeypatch)
    core, old, on_core, on_old = _both(monkeypatch, request)
    assert not isinstance(core, Exception), core
    pd.testing.assert_frame_equal(core.to_pandas(), old.to_pandas())
    assert core.items == old.items
    assert (core.model, core.metadata, core.outputs) == (
        old.model,
        old.metadata,
        old.outputs,
    )
    [(path, body, headers)] = on_core
    [(old_path, old_body, old_headers)] = on_old
    assert path == old_path == INVOCATIONS
    # The same call, in the envelope the old route weighs; posted as the
    # core wrote it, compact, every character past ASCII escaped.
    assert _call(body) == _call(old_body)
    assert body.isascii()
    assert body.startswith(b'{"dataframe_records":[{"request_json":"{')
    assert headers['Content-Type'] == 'application/json'
    # The workspace's own client, naming this one, as on the old route.
    assert headers['User-Agent'] == old_headers['User-Agent']
    assert 'structured-data-models-client' in headers['User-Agent']
    assert headers['Authorization'] == 'Bearer dapi-test'


def test_a_call_the_sdk_retries_sends_the_same_bytes_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _materialized(monkeypatch)
    busy: Reply = (
        429,
        {'error_code': 'REQUEST_LIMIT_EXCEEDED', 'message': 'slow down'},
        {'Retry-After': '0'},
    )
    core, received = _predict(monkeypatch, request, CORE, busy)
    assert not isinstance(core, Exception), core
    assert len(core) == 2
    [(_, first, _), (_, again, _)] = received
    assert first == again
    assert len(first) > 0


@pytest.mark.parametrize(
    'reply',
    [
        (
            400,
            {
                'error_code': 'BAD_REQUEST',
                'message': f'INVALID_SCHEMA: could not parse row {SECRET}',
            },
            {},
        ),
        (
            400,
            {
                'error_code': 'BAD_REQUEST',
                'message': f'BAD_REQUEST: could not parse row {SECRET}: 42',
            },
            {},
        ),
        (
            500,
            {
                'error_code': 'INTERNAL_ERROR',
                'message': 'failed to collect any context examples for '
                f'{SECRET}',
            },
            {},
        ),
        (
            404,
            {
                'error_code': 'RESOURCE_DOES_NOT_EXIST',
                'message': f'no endpoint {SECRET}',
            },
            {},
        ),
        (403, f'PERMISSION_DENIED: {SECRET} may not query', {}),
        (500, f'upstream failure quoting {SECRET}', {}),
    ],
    ids=[
        'model-code',
        'provider-code',
        'callers-fault',
        'not-found',
        'string-form',
        'unparsed',
    ],
)
def test_an_endpoint_s_failure_is_the_old_route_s_refusal(
    monkeypatch: pytest.MonkeyPatch, reply: Reply
) -> None:
    request = _materialized(monkeypatch)
    core, old, on_core, on_old = _both(monkeypatch, request, reply)
    _same_refusal(core, old)
    assert len(on_core) == len(on_old) == 1


@pytest.mark.parametrize(
    'answer',
    [
        {'status': 422, 'code': 'UNSUPPORTED_TASK', 'detail': SECRET},
        {'status': 500, 'title': f'the model failed on {SECRET}'},
        {
            'status': 500,
            'code': 'INTERNAL_ERROR',
            'detail': f'failed to collect any context examples for {SECRET}',
        },
    ],
    ids=['coded', 'uncoded', 'callers-fault'],
)
def test_a_failure_the_model_answered_with_is_the_old_route_s_refusal(
    monkeypatch: pytest.MonkeyPatch, answer: dict[str, Any]
) -> None:
    request = _materialized(monkeypatch)
    reply: Reply = (
        200,
        {'predictions': [{'response_json': json.dumps(answer)}]},
        {},
    )
    core, old, _, _ = _both(monkeypatch, request, reply)
    _same_refusal(core, old)


@pytest.mark.parametrize(
    'body',
    [
        {'predictions': []},
        {'predictions': [{'other': 1}]},
        {'predictions': [{'response_json': '[1]'}]},
        {'predictions': [{'response_json': '{'}]},
    ],
    ids=['no-row', 'no-column', 'not-an-object', 'not-json'],
)
def test_an_answer_the_old_route_cannot_read_is_its_transport_error(
    monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]
) -> None:
    request = _materialized(monkeypatch)
    core, old, _, _ = _both(monkeypatch, request, (200, body, {}))
    assert isinstance(old, StructuredDataError), old
    assert isinstance(core, StructuredDataError), core
    assert (core.code, str(core)) == (old.code, str(old))


def test_a_call_past_the_caller_s_limit_is_refused_before_it_is_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _materialized(monkeypatch)
    _, received = _predict(monkeypatch, request, CORE)
    [(_, body, _)] = received
    # One byte short of the envelope's size: refused on both routes, the
    # same words, nothing sent.
    core, old, on_core, on_old = _both(
        monkeypatch, request, max_request_bytes=len(body) - 1
    )
    assert on_core == on_old == []
    assert isinstance(old, StructuredDataError), old
    assert (type(core), core.code, str(core)) == (
        type(old),
        old.code,
        str(old),
    )
    assert core.details['limit'] == len(body) - 1
    assert core.details['bytes'] == old.details['bytes'] == len(body)
    # The size itself is admitted.
    core, received = _predict(
        monkeypatch, request, CORE, max_request_bytes=len(body)
    )
    assert not isinstance(core, Exception), core
    [(_, sent, _)] = received
    assert len(sent) == len(body)
    assert _call(sent) == _call(body)


@pytest.mark.parametrize(
    'reply',
    [None, (500, f'upstream failure quoting {SECRET}', {})],
    ids=['answered', 'unparsed-failure'],
)
def test_the_sdk_s_debug_log_carries_no_rows_on_the_core_route(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    reply: Reply | None,
) -> None:
    request = _materialized(monkeypatch)
    caplog.set_level(logging.DEBUG)
    replies = () if reply is None else (reply,)
    outcome, received = _predict(monkeypatch, request, CORE, *replies)
    assert received
    if reply is not None:
        assert isinstance(outcome, NimRequestError), outcome
        assert outcome.status_code == 503
    for said in ('request_json', 'dataframe_records', 'predictions', SECRET):
        assert said not in caplog.text, said
