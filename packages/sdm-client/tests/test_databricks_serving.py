# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A client opened against a Databricks Model Serving endpoint.

The model is served by name inside a workspace, so a request travels through
the Databricks SDK wrapped as one ``request_json`` row, and the answer comes
back as one ``response_json`` row. Everything here runs offline against a fake
workspace client; what is checked is what would be sent and how what comes
back is read.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from typing import Any

import pandas as pd
import pytest

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Relationship,
    Schema,
    SDMClient,
)
from sdm_client.core.databricks_serving import DatabricksServingTransport
from sdm_client.errors import (
    MissingExtraError,
    NimRequestError,
    StructuredDataError,
)

ENDPOINT = 'kumo-rfm-serving'
SECRET = 'Sup3rSecret-CustomerValue-9f2a'
QUERY = 'PREDICT users.churn FOR EACH users.user_id'


def schema() -> Schema:
    frame = pd.DataFrame(
        {
            'user_id': list(range(1, 21)),
            'churn': ['no' if i % 3 == 0 else 'yes' for i in range(20)],
            'tenure': [float(i) for i in range(20)],
        }
    )
    built = Schema.from_data({'users': frame}, relationships=[])
    built['users'].primary_key = 'user_id'
    return built


def tabular_answer(ids: list[str]) -> dict[str, Any]:
    return {
        'object': 'prediction.response',
        'id': 'pred_1',
        'model': 'kumo-tabular',
        'metadata': {},
        'predictions': [
            {'id': value, 'row_index': index, 'prediction': 'yes'}
            for index, value in enumerate(ids)
        ],
    }


class Endpoints:
    """The part of ``WorkspaceClient.serving_endpoints`` the transport uses."""

    def __init__(self, answer: Any = None, raiser: BaseException | None = None):
        self.calls: list[dict[str, Any]] = []
        self.answer = answer
        self.raiser = raiser
        self.ready = 'READY'

    def query(self, *, name: str, **kwargs: Any) -> Any:
        self.calls.append({'name': name, **kwargs})
        if self.raiser is not None:
            raise self.raiser
        answer = self.answer
        if callable(answer):
            sent = json.loads(kwargs['dataframe_records'][0]['request_json'])
            answer = answer(sent)
        return {'predictions': [{'response_json': json.dumps(answer)}]}

    def get(self, *, name: str) -> Any:
        self.calls.append({'name': name, 'get': True})
        ready = type('Ready', (), {'value': self.ready})()
        state = type('State', (), {'ready': ready})()
        return type('Endpoint', (), {'state': state})()


class Workspace:
    """Fails loudly on anything the transport should not reach for."""

    def __init__(self, **kwargs: Any) -> None:
        self.serving_endpoints = Endpoints(**kwargs)

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f'transport reached for WorkspaceClient.{name}')


def opened(**kwargs: Any) -> tuple[SDMClient, Endpoints]:
    workspace = Workspace(**kwargs)
    client = SDMClient.for_databricks_serving(
        ENDPOINT, workspace_client=workspace
    )
    return client, workspace.serving_endpoints


def asked() -> Any:
    return PredictiveQuery(QUERY, schema()).materialize(indices=[3, 7])


# --- what is sent, and how the answer is read -------------------------------


def test_a_prediction_travels_as_one_request_json_row():
    client, endpoints = opened(answer=tabular_answer(['[3]', '[7]']))
    result = client.predict(asked(), model='kumo-tabular')

    (call,) = endpoints.calls
    assert call['name'] == ENDPOINT
    (record,) = call['dataframe_records']
    assert list(record) == ['request_json']
    sent = json.loads(record['request_json'])
    assert sent['model'] == 'kumo-tabular'
    assert {'task', 'schema', 'context', 'predict'} <= set(sent)
    assert [item.id for item in result.items] == [3, 7]


def test_the_sdk_s_own_response_type_is_read():
    """The fake above answers with a hand-made dict. This pins that shape to
    the type the SDK actually returns."""
    serving = pytest.importorskip('databricks.sdk.service.serving')

    class Typed(Endpoints):
        def query(self, *, name: str, **kwargs: Any) -> Any:
            self.calls.append({'name': name, **kwargs})
            return serving.QueryEndpointResponse(
                predictions=[
                    {
                        'response_json': json.dumps(
                            tabular_answer(['[3]', '[7]'])
                        )
                    }
                ]
            )

    workspace = Workspace()
    workspace.serving_endpoints = Typed()
    client = SDMClient.for_databricks_serving(
        ENDPOINT, workspace_client=workspace
    )
    result = client.predict(asked(), model='kumo-tabular')
    assert [item.id for item in result.items] == [3, 7]


def test_a_reply_that_is_not_one_row_is_a_transport_error():
    class TwoRows(Endpoints):
        def query(self, *, name: str, **kwargs: Any) -> Any:
            return {'predictions': [{'response_json': '{}'}] * 2}

    workspace = Workspace()
    workspace.serving_endpoints = TwoRows()
    client = SDMClient.for_databricks_serving(
        ENDPOINT, workspace_client=workspace
    )
    with pytest.raises(StructuredDataError) as error:
        client.predict(asked(), model='kumo-tabular')
    assert error.value.code == 'TRANSPORT_ERROR'
    assert ENDPOINT in str(error.value)


def test_a_question_asked_in_parts_is_one_call_per_part():
    """A forecast has no task on the endpoint and is asked once per step;
    each step is its own serving call."""
    pytest.importorskip('sdm_relational_engine')
    request = forecast_request()
    assert request.parts is not None

    def answer(sent: dict[str, Any]) -> dict[str, Any]:
        rows = sent['predict']['instance_table']['rows']
        return {
            'object': 'prediction.response',
            'id': 'pred',
            'model': 'kumo-relational',
            'metadata': {},
            'predictions': [
                {'id': str(row[0]), 'row_index': i, 'prediction': 1.0}
                for i, row in enumerate(rows)
            ],
        }

    client, endpoints = opened(answer=answer)
    result = client.predict(request, model=KumoRelational())

    assert len(endpoints.calls) == len(request.parts)
    steps = {item.forecast_step for item in result.items}
    assert steps == set(range(1, len(request.parts) + 1))


def forecast_request() -> Any:
    import numpy as np

    rng = np.random.default_rng(7)
    users = pd.DataFrame({'uid': range(40), 'seg': rng.choice([*'xy'], 40)})
    buys = pd.DataFrame(
        {
            'bid': range(1500),
            'uid': rng.integers(0, 40, 1500),
            'ts': pd.to_datetime('2023-01-01')
            + pd.to_timedelta(rng.integers(0, 300, 1500), 'D'),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        shop = Schema.from_data(
            {'users': users, 'buys': buys},
            primary_keys={'users': 'uid', 'buys': 'bid'},
            time_columns={'buys': 'ts'},
            relationships=[Relationship('buys', 'uid', 'users', 'uid')],
        )
        return PredictiveQuery(
            'PREDICT COUNT(buys.*, 0, 30, days) FORECAST 3 TIMEFRAMES '
            'FOR EACH users.uid',
            shop,
        ).materialize(model=KumoRelational(), indices=[1, 2])


# --- what is refused, and how failures read ---------------------------------


def test_fit_is_refused_before_anything_is_prepared():
    client, endpoints = opened(answer=tabular_answer([]))
    with pytest.raises(StructuredDataError) as error:
        client.fit(asked(), model='kumo-tabular')
    assert error.value.code == 'SESSIONS_UNSUPPORTED'
    assert 'predict()' in str(error.value)
    assert endpoints.calls == []


def test_the_models_route_is_refused_rather_than_answered_from_a_guess():
    client, _ = opened()
    with pytest.raises(StructuredDataError) as error:
        client.models()
    assert error.value.code == 'UNSUPPORTED_REQUEST'


def test_readiness_is_the_endpoint_s_own_state():
    client, endpoints = opened()
    assert client.health_ready() is True
    endpoints.ready = 'NOT_READY'
    assert client.health_ready() is False


@pytest.mark.parametrize(
    'said, code',
    [
        (
            'Encountered an unexpected error while evaluating the model. '
            "Error 'INVALID_SCHEMA: the request schema is invalid'",
            'INVALID_SCHEMA',
        ),
        (
            'INVALID_SCHEMA: Failed to parse input: row 3 customer_name=Jane '
            'Doe ssn=123-45-6789',
            'INVALID_SCHEMA',
        ),
        ('BAD_REQUEST: could not parse row NVDA_SECRET: 42', 'BAD_REQUEST'),
    ],
    ids=['quoted', 'leading', 'value-shaped-like-a-code'],
)
def test_the_served_model_s_code_is_repeated_and_its_text_is_not(said, code):
    """A model's error text can quote the rows it was sent, and a value can
    look like a code; only one where the model's message begins is read."""
    errors = pytest.importorskip('databricks.sdk.errors')
    client, _ = opened(raiser=errors.BadRequest(said, error_code='BAD_REQUEST'))
    with pytest.raises(NimRequestError) as error:
        client.predict(asked(), model='kumo-tabular')
    assert error.value.status_code == 400
    assert error.value.code == code
    assert error.value.details['provider_code'] == 'BAD_REQUEST'
    shown = str(error.value) + json.dumps(error.value.details)
    assert ENDPOINT in shown
    for detail in ('schema is invalid', 'Jane', '123-45', 'NVDA_SECRET'):
        assert detail not in shown


def test_a_server_error_that_is_the_caller_s_to_fix_is_reported_as_such():
    """As from a NIM: retrying a request that can only fail the same way
    does not help."""
    errors = pytest.importorskip('databricks.sdk.errors')
    refusal = errors.InternalError(
        f'failed to collect any context examples for {SECRET}',
        error_code='INTERNAL_ERROR',
    )
    client, _ = opened(raiser=refusal)
    with pytest.raises(NimRequestError) as error:
        client.predict(asked(), model='kumo-tabular')
    assert error.value.status_code >= 500
    assert error.value.code == 'INVALID_REQUEST'
    assert error.value.details['diagnosis'] == 'EMPTY_CONTEXT'
    assert error.value.details['endpoint_code'] == 'INTERNAL_ERROR'
    assert SECRET not in str(error.value) + json.dumps(error.value.details)


def test_a_gateway_failure_names_the_endpoint_and_nothing_it_said():
    """Only the served model's diagnosis is repeated. The gateway's own text is
    not held to that model's rule of carrying no request content."""
    errors = pytest.importorskip('databricks.sdk.errors')
    refusal = errors.PermissionDenied(
        f'PERMISSION_DENIED: {SECRET} may not query this endpoint',
        error_code='PERMISSION_DENIED',
    )
    client, _ = opened(raiser=refusal)
    with pytest.raises(NimRequestError) as error:
        client.predict(asked(), model='kumo-tabular')
    assert error.value.status_code == 403
    assert error.value.code == 'PERMISSION_DENIED'
    assert ENDPOINT in str(error.value)
    assert SECRET not in str(error.value)
    assert SECRET not in json.dumps(error.value.details)
    assert error.value.__cause__ is None


def test_an_unclassified_failure_is_retryable_rather_than_final():
    client, _ = opened(raiser=ConnectionError('socket closed'))
    with pytest.raises(NimRequestError) as error:
        client.predict(asked(), model='kumo-tabular')
    assert error.value.status_code == 503


def test_a_bug_is_not_reported_as_the_endpoint_s_failure():
    bug = TypeError('unexpected keyword argument')
    client, _ = opened(raiser=bug)
    with pytest.raises(TypeError) as error:
        client.predict(asked(), model='kumo-tabular')
    assert error.value is bug


@pytest.mark.parametrize(
    'body, code, diagnosis',
    [
        (
            {
                'status': 422,
                'code': 'UNSUPPORTED_TASK',
                'detail': f'the requested task is not supported for {SECRET}',
            },
            'UNSUPPORTED_TASK',
            None,
        ),
        ({'status': 500, 'title': f'the model failed on {SECRET}'}, None, None),
        (
            {
                'status': 500,
                'code': 'INTERNAL_ERROR',
                'detail': f'INVALID_SCHEMA: could not parse row {SECRET}',
            },
            'INVALID_SCHEMA',
            None,
        ),
        (
            {
                'status': 500,
                'code': 'INTERNAL_ERROR',
                'detail': 'prediction failed: failed to collect any context '
                f'examples for {SECRET}',
            },
            'INVALID_REQUEST',
            'EMPTY_CONTEXT',
        ),
        (
            {'status': 500, 'detail': 'could not parse row NVDA_SECRET: 42'},
            None,
            None,
        ),
    ],
    ids=[
        'with-detail',
        'title-only',
        'model-diagnosis',
        'callers-fault',
        'value-shaped-like-a-code',
    ],
)
def test_an_error_body_in_the_response_is_an_error_named_by_its_code(
    body, code, diagnosis
):
    """The body's text can quote the rows it was sent."""
    client, _ = opened(answer=body)
    with pytest.raises(NimRequestError) as error:
        client.predict(asked(), model='kumo-tabular')
    assert error.value.status_code == body['status']
    assert error.value.code == code
    assert error.value.details.get('diagnosis') == diagnosis
    shown = str(error.value) + json.dumps(error.value.details)
    assert ENDPOINT in shown
    assert SECRET not in shown
    assert 'NVDA_SECRET' not in shown


def test_an_oversized_request_is_refused_before_it_is_sent():
    workspace = Workspace()
    transport = DatabricksServingTransport(
        ENDPOINT, workspace, max_request_bytes=200
    )
    with pytest.raises(StructuredDataError) as error:
        transport.predict({'rows': ['{"a":"b"}'] * 50})
    assert error.value.code == 'INVALID_REQUEST'
    assert error.value.details['limit'] == 200
    assert error.value.details['bytes'] > 200
    assert workspace.serving_endpoints.calls == []


def test_the_size_counted_is_the_wrapped_envelope_not_the_payload():
    """Every quote is escaped again once the payload is a string inside the
    envelope, so a payload under the limit can be over it on the wire."""
    payload = {'rows': ['{"a":"b","c":"d"}'] * 40}
    unwrapped = len(json.dumps(payload, separators=(',', ':')))
    workspace = Workspace(answer={'predictions': []})
    transport = DatabricksServingTransport(
        ENDPOINT, workspace, max_request_bytes=unwrapped + 10
    )
    with pytest.raises(StructuredDataError):
        transport.predict(payload)


# --- opening one ------------------------------------------------------------


@pytest.fixture
def workspace_host() -> Any:
    """A local stand-in for a workspace; set ``answer`` to choose its reply."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    answer = {
        'status': 200,
        'body': json.dumps(
            {'predictions': [{'response_json': json.dumps({'ok': True})}]}
        ),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get('Content-Length', 0)))
            self.send_response(answer['status'])
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(answer['body'].encode())

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{server.server_port}', answer
    server.shutdown()


@pytest.mark.parametrize(
    'status, body',
    [
        (200, None),
        (500, 'upstream failure'),
        (500, f'could not parse row {SECRET}'),
    ],
    ids=['answered', 'unparseable-error', 'error-quoting-a-row'],
)
def test_the_debug_log_carries_no_rows(workspace_host, caplog, status, body):
    sdk = pytest.importorskip('databricks.sdk')
    host, answer = workspace_host
    if body is not None:
        answer.update(status=status, body=body)
    loggers = ('databricks.sdk', 'databricks.sdk.core')
    levels = {name: logging.getLogger(name).level for name in loggers}
    caplog.set_level(logging.DEBUG)
    workspace = sdk.WorkspaceClient(
        config=sdk.core.Config(
            host=host,
            token='dapi-test',
            auth_type='pat',
            retry_timeout_seconds=5,
        )
    )

    transport = DatabricksServingTransport(ENDPOINT, workspace_client=workspace)
    if status >= 400:
        with pytest.raises(NimRequestError):
            transport.predict({'context': {'rows': [[SECRET]]}})
    else:
        transport.predict({'context': {'rows': [[SECRET]]}})

    assert SECRET not in caplog.text
    assert {name: logging.getLogger(name).level for name in loggers} == levels


@pytest.mark.parametrize(
    'named',
    [
        f'https://user:{SECRET}@dbc.example.com/serving-endpoints/x',
        f'x?token={SECRET}',
        f'user:{SECRET}@x',
    ],
)
def test_a_url_is_refused_without_being_repeated(named):
    with pytest.raises(StructuredDataError) as error:
        SDMClient.for_databricks_serving(named, workspace_client=Workspace())
    assert error.value.code == 'INVALID_CONFIGURATION'
    assert SECRET not in str(error.value)


def test_the_client_says_where_it_sends():
    client, _ = opened()
    assert client.url == f'databricks-serving:{ENDPOINT}'
    assert ENDPOINT in repr(client)


def test_a_closed_client_sends_nothing():
    client, endpoints = opened(answer=tabular_answer(['[3]', '[7]']))
    client.close()
    with pytest.raises(StructuredDataError) as error:
        client.predict(asked(), model='kumo-tabular')
    assert error.value.code == 'INVALID_CONFIGURATION'
    assert endpoints.calls == []


def test_without_the_sdk_the_extra_to_install_is_named(monkeypatch):
    monkeypatch.setitem(sys.modules, 'databricks.sdk', None)
    with pytest.raises(MissingExtraError) as error:
        SDMClient.for_databricks_serving(ENDPOINT)
    assert error.value.details['extra'] == 'databricks-serving'
    assert 'sdm-client[databricks-serving]' in str(error.value)


@pytest.mark.parametrize(
    'timeout, config',
    [
        (None, {'http_timeout_seconds': 300.0, 'retry_timeout_seconds': 300}),
        (420.5, {'http_timeout_seconds': 420.5, 'retry_timeout_seconds': 421}),
    ],
)
def test_the_ambient_identity_is_used_when_no_client_is_given(
    monkeypatch, timeout, config
):
    """The SDK keeps retrying for its own deadline whatever one attempt's
    timeout, so the whole call is bounded only if that deadline is set too."""
    sdk = pytest.importorskip('databricks.sdk')
    built: list[Any] = []

    class Recorded:
        def __init__(self, *, config: Any) -> None:
            built.append(config)
            self.serving_endpoints = Endpoints()

    monkeypatch.setattr(sdk, 'WorkspaceClient', Recorded)
    monkeypatch.setattr(
        sdk.core,
        'Config',
        lambda **kwargs: kwargs,
    )
    SDMClient.for_databricks_serving(ENDPOINT, timeout=timeout)
    assert built == [config]


def test_a_timeout_for_an_injected_workspace_client_is_refused():
    """That client keeps its own configuration, so the timeout would be
    silently ignored."""
    with pytest.raises(StructuredDataError) as error:
        SDMClient.for_databricks_serving(
            ENDPOINT, workspace_client=Workspace(), timeout=30
        )
    assert error.value.code == 'INVALID_CONFIGURATION'


def test_a_workspace_that_cannot_authenticate_says_how_to_fix_it(monkeypatch):
    sdk = pytest.importorskip('databricks.sdk')

    def refuse(**kwargs: Any) -> Any:
        raise ValueError(f'default auth: cannot configure {SECRET}')

    monkeypatch.setattr(sdk, 'WorkspaceClient', refuse)
    with pytest.raises(StructuredDataError) as error:
        SDMClient.for_databricks_serving(ENDPOINT)
    assert error.value.code == 'INVALID_CONFIGURATION'
    assert 'workspace_client=' in str(error.value)
    assert SECRET not in str(error.value)


@pytest.mark.parametrize('bad', [0, -1.0, float('inf'), True])
def test_an_unusable_timeout_is_refused_at_construction(bad):
    with pytest.raises(StructuredDataError) as error:
        SDMClient.for_databricks_serving(ENDPOINT, timeout=bad)
    assert error.value.code == 'INVALID_CONFIGURATION'


def test_the_envelope_is_the_one_the_engine_sends():
    """The client and the engine each wrap a request for the same deployed
    endpoints, so they have to agree on the envelope."""
    pytest.importorskip('sdm_relational_engine')
    from sdm_relational_engine.client import databricks_serving as engine

    from sdm_client.core import databricks_serving as client

    assert client._REQUEST_COLUMN == engine.REQUEST_COLUMN
    assert client._RESPONSE_COLUMN == engine.RESPONSE_COLUMN
    assert client._MAX_REQUEST_BYTES == engine.MAX_REQUEST_BYTES
