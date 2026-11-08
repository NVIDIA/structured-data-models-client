# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Databricks serving transport through its real SDK and local HTTP.
Check wire envelopes, entity attribution, retries, request limits,
sanitized failures and debug logs without retired runtime comparisons.
No live Databricks service or model endpoint is used.
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
)
from sdm_client.core.databricks_serving import DatabricksServingTransport
from sdm_client.errors import NimRequestError

sdk = pytest.importorskip('databricks.sdk')

ROOT = Path(__file__).resolve().parents[3]
ENDPOINT = 'kumo-rfm-serving'
INVOCATIONS = f'/serving-endpoints/{ENDPOINT}/invocations'
SECRET = 'Sup3rSecret-CustomerValue-9f2a'
QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'


def _scripted() -> Any:
    name = 'scripted_endpoint'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / 'tests/support/scripted_endpoint.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _local_network(monkeypatch: pytest.MonkeyPatch) -> None:
    # A proxied runner would otherwise send these local requests to its
    # proxy.
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('NO_PROXY', '127.0.0.1')
    monkeypatch.setenv('no_proxy', '127.0.0.1')


def _shop() -> list[str]:
    return json.loads((Path(__file__).parent / 'data/shop.json').read_text())[
        'statements'
    ]


def _materialized(*, unicode_feature: bool = False) -> Request:
    r"""base.regression's request, materialized on the core."""
    connection = duckdb.connect()
    for statement in _shop():
        connection.execute(statement)
    if unicode_feature:
        connection.execute("UPDATE users SET country = 'é😀' WHERE user_id = 0")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(connection))
        request = PredictiveQuery(QUERY, schema).materialize(
            indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01')
        )
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
    request: Request,
    *replies: Reply | Callable[[bytes], Reply],
    max_request_bytes: int | None = None,
) -> tuple[Any, list[tuple[str, bytes, dict[str, str]]]]:
    """Public prediction through the real SDK against a local HTTP server."""
    with _workspace(*replies) as (workspace, received):
        with _client(workspace, max_request_bytes) as client:
            try:
                outcome: Any = client.predict(request, model=KumoRelational())
            except StructuredDataError as error:
                outcome = error
    return outcome, received


def _call(body: bytes) -> dict[str, Any]:
    r"""The call an envelope carries, its drawn id taken out."""
    call = json.loads(json.loads(body)['dataframe_records'][0]['request_json'])
    call['metadata'].pop('request_id')
    return call


def _assert_refusal(error: Any, status: int, code: str | None) -> None:
    assert isinstance(error, NimRequestError), error
    assert (error.status_code, error.code) == (status, code)
    assert error.details['endpoint'] == ENDPOINT
    if code in (
        'INVALID_SCHEMA',
        'INVALID_REQUEST',
        'UNSUPPORTED_TASK',
    ):
        expected_message = (
            f'serving endpoint {ENDPOINT!r} refused the request ({code})'
        )
    else:
        suffix = f' ({code})' if code else ''
        expected_message = (
            f'serving endpoint {ENDPOINT!r} request failed{suffix}'
        )
    assert error.message == expected_message
    shown = str(error) + json.dumps(error.details, default=str)
    assert SECRET not in shown
    assert error.__cause__ is None


def test_envelope_passes_through_sdk_and_result_keeps_entity_ids() -> None:
    request = _materialized()
    result, received = _predict(request)
    assert not isinstance(result, Exception), result
    assert result.to_pandas()['id'].tolist() == [0, 1]
    assert result.to_pandas()['prediction'].tolist() == [1.0, 1.0]
    [(path, body, headers)] = received
    assert path == INVOCATIONS
    call = _call(body)
    for field in ('context', 'predict', 'schema', 'task'):
        assert call[field] == request.payloads[0][field], field
    assert call['model'] == result.model == 'kumo-relational'
    assert result.metadata == {}
    assert result.outputs == tuple(call['output']['fields'])
    assert [item.row_index for item in result.items] == [0, 1]
    assert body.startswith(b'{"dataframe_records":[{"request_json":"{')
    assert headers['Content-Type'] == 'application/json'
    assert 'structured-data-models-client' in headers['User-Agent']
    assert headers['Authorization'] == 'Bearer dapi-test'


def test_a_call_the_sdk_retries_sends_the_same_bytes_again() -> None:
    request = _materialized()
    busy: Reply = (
        429,
        {'error_code': 'REQUEST_LIMIT_EXCEEDED', 'message': 'slow down'},
        {'Retry-After': '0'},
    )
    core, received = _predict(request, busy)
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
def test_endpoint_failure_is_sanitized_and_classified(reply: Reply) -> None:
    error, received = _predict(_materialized(), reply)
    status, body, _ = reply
    if isinstance(body, dict):
        provider_code = body['error_code']
        code = {
            'BAD_REQUEST': 'INVALID_SCHEMA'
            if body['message'].startswith('INVALID_SCHEMA:')
            else 'BAD_REQUEST',
            'INTERNAL_ERROR': 'INVALID_REQUEST',
            'RESOURCE_DOES_NOT_EXIST': 'RESOURCE_DOES_NOT_EXIST',
        }[provider_code]
        _assert_refusal(error, 503 if status == 500 else status, code)
        assert error.details['provider_code'] == provider_code
        if provider_code == 'INTERNAL_ERROR':
            assert error.details['diagnosis'] == 'EMPTY_CONTEXT'
            assert error.details['endpoint_code'] == 'INTERNAL_ERROR'
    else:
        _assert_refusal(
            error,
            403 if status == 403 else 503,
            'PERMISSION_DENIED' if status == 403 else None,
        )
    assert len(received) == 1


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
def test_served_model_failure_is_sanitized_and_classified(
    answer: dict[str, Any],
) -> None:
    reply: Reply = (
        200,
        {'predictions': [{'response_json': json.dumps(answer)}]},
        {},
    )
    error, received = _predict(_materialized(), reply)
    code = (
        'INVALID_REQUEST'
        if answer.get('code') == 'INTERNAL_ERROR'
        else answer.get('code')
    )
    _assert_refusal(error, answer['status'], code)
    if code == 'INVALID_REQUEST':
        assert error.details['diagnosis'] == 'EMPTY_CONTEXT'
    assert len(received) == 1


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
def test_malformed_serving_response_is_a_transport_error(
    body: dict[str, Any],
) -> None:
    error, received = _predict(_materialized(), (200, body, {}))
    assert isinstance(error, StructuredDataError), error
    assert error.code == 'TRANSPORT_ERROR'
    if body == {'predictions': []}:
        expected = f'serving endpoint {ENDPOINT!r} did not return exactly one prediction row'
    elif body == {'predictions': [{'other': 1}]}:
        expected = f"serving endpoint {ENDPOINT!r} answered with a row that has no 'response_json' string column"
    elif body == {'predictions': [{'response_json': '[1]'}]}:
        assert error.details['got'] == 'array'
        expected = f'expected a JSON object from serving endpoint {ENDPOINT!r}, got array'
    else:
        expected = f'serving endpoint {ENDPOINT!r} answered with a response that is not JSON'
    assert error.message == expected
    assert len(received) == 1


def test_a_call_past_the_caller_s_limit_is_refused_before_it_is_sent() -> None:
    request = _materialized()
    _, received = _predict(request)
    [(_, body, _)] = received
    # The limit applies to the actual UTF-8 envelope posted to the SDK.
    core, on_core = _predict(request, max_request_bytes=len(body) - 1)
    assert on_core == []
    assert isinstance(core, StructuredDataError), core
    assert core.details['limit'] == len(body) - 1
    assert core.details['bytes'] == len(body)
    # The size itself is admitted.
    core, received = _predict(request, max_request_bytes=len(body))
    assert not isinstance(core, Exception), core
    [(_, sent, _)] = received
    assert len(sent) == len(body)
    assert _call(sent) == _call(body)


def test_utf8_serving_preserves_public_payload_and_measures_the_sent_body() -> (
    None
):
    request = _materialized(unicode_feature=True)
    result, received = _predict(request)
    assert not isinstance(result, Exception), result
    [(_, body, _)] = received
    assert 'é😀'.encode() in body
    call = _call(body)
    for field in ('context', 'predict', 'schema', 'task'):
        assert call[field] == request.payloads[0][field], field
    assert result.to_pandas()['id'].tolist() == [0, 1]
    assert result.to_pandas()['prediction'].tolist() == [1.0, 1.0]
    refused, sent = _predict(request, max_request_bytes=len(body) - 1)
    assert isinstance(refused, StructuredDataError)
    assert sent == []
    assert refused.details['bytes'] == len(body)
    assert refused.details['limit'] == len(body) - 1


@pytest.mark.parametrize(
    'reply',
    [None, (500, f'upstream failure quoting {SECRET}', {})],
    ids=['answered', 'unparsed-failure'],
)
def test_the_sdk_s_debug_log_carries_no_rows_on_the_core_route(
    caplog: pytest.LogCaptureFixture,
    reply: Reply | None,
) -> None:
    request = _materialized()
    caplog.set_level(logging.DEBUG)
    replies = () if reply is None else (reply,)
    outcome, received = _predict(request, *replies)
    assert received
    if reply is not None:
        assert isinstance(outcome, NimRequestError), outcome
        assert outcome.status_code == 503
    for said in ('request_json', 'dataframe_records', 'predictions', SECRET):
        assert said not in caplog.text, said
