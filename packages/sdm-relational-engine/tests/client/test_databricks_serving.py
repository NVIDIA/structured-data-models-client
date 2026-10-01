# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""The Databricks Model Serving transport, against a fake WorkspaceClient.

No network, no Databricks SDK required. The point of these tests is that the
transport satisfies the same seam as the NIM client while sharing none of its
assumptions.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from sdm_api.generated.tfm_api import TFMOperations
from sdm_relational_engine.client.databricks_serving import (
    REQUEST_COLUMN,
    RESPONSE_COLUMN,
    DatabricksServingClient,
    ServingResponse,
    _status_of,
)
from sdm_relational_engine.exceptions import HTTPException

# Derived, never hardcoded: a codegen path bump must move the tests with the
# transport instead of leaving both agreeing on a stale literal.
PREDICTION = TFMOperations.run_prediction.endpoint
SESSIONS = TFMOperations.create_session.endpoint
CANONICAL_RESPONSE = {
    'id': 'r1',
    'model': 'kumo-relational',
    'predictions': [{'row_index': 0, 'prediction': 1}],
    'metadata': {'task_kind': 'binary_classification'},
}
SECRET = 'Sup3rSecret-CustomerValue-9f2a'


class _Endpoints:
    def __init__(self, reply: Any = None, raiser: BaseException | None = None):
        self.calls: list[dict[str, Any]] = []
        self._reply = reply
        self._raiser = raiser

    def query(self, *, name: str, **kwargs: Any) -> Any:
        self.calls.append({'name': name, **kwargs})
        if self._raiser is not None:
            raise self._raiser
        if self._reply is not None:
            return self._reply
        return {
            'predictions': [{RESPONSE_COLUMN: json.dumps(CANONICAL_RESPONSE)}]
        }


class _Workspace:
    """Records every attribute touched, so unexpected calls are visible."""

    def __init__(self, reply: Any = None, raiser: BaseException | None = None):
        self.serving_endpoints = _Endpoints(reply, raiser)
        self.touched: list[str] = []

    def __getattr__(self, name: str) -> Any:
        self.touched.append(name)
        raise AssertionError(f'transport reached for WorkspaceClient.{name}')


def _client(**kwargs: Any) -> tuple[DatabricksServingClient, _Workspace]:
    workspace = _Workspace(**kwargs)
    return DatabricksServingClient('kumo-relational', workspace), workspace


def _wire_size(request: Any) -> int:
    """The bytes that actually cross the gateway, computed independently.

    Deliberately not a call into the transport: if both used the same helper a
    wrong measurement would agree with itself.
    """
    payload = json.dumps(request, allow_nan=False, separators=(',', ':'))
    body = json.dumps(
        {'dataframe_records': [{REQUEST_COLUMN: payload}]},
        separators=(',', ':'),
    )
    return len(body.encode('utf-8'))


# Quote-dense, like a real `format: arrays` cohort: every quote is re-escaped
# once the payload is nested in request_json.
QUOTE_DENSE = {
    'model': 'kumo-relational',
    'rows': ['{"a":"b","c":"d"}'] * 400,
}


# -- the seam --------------------------------------------------------------


def test_exposes_what_the_rfm_path_calls() -> None:
    """``RFMAPI`` calls ``_request``; ``GlobalState.clear()`` calls ``close``.

    Renaming either on this transport breaks the serving path only, so it is
    checked here rather than left to the NIM client's tests.
    """
    client, _ = _client()
    assert callable(client._request)
    assert callable(client.close)


def test_returns_a_serving_response() -> None:
    client, _ = _client()
    response = client._request(PREDICTION, json={'model': 'kumo-relational'})
    assert isinstance(response, ServingResponse)
    assert response.ok and response.status_code == 200
    assert response.json()['model'] == 'kumo-relational'


def test_has_no_authenticate_method() -> None:
    """Authenticate() probes /v1/health/ready and /v1/models. A named serving
    endpoint advertises neither, so the method must not exist here at all.
    """
    client, _ = _client()
    assert not hasattr(client, 'authenticate')


# -- no raw-NIM behaviour --------------------------------------------------


def test_queries_by_endpoint_name_only() -> None:
    client, workspace = _client()
    client._request(PREDICTION, json={'model': 'kumo-relational'})
    call = workspace.serving_endpoints.calls[0]
    assert call['name'] == 'kumo-relational'
    assert workspace.touched == [], 'transport touched an unexpected attribute'


def test_never_probes_health_or_models() -> None:
    """The negative assertion the contract calls for, as a real test."""
    client, workspace = _client()
    client._request(PREDICTION, json={'model': 'kumo-relational'})
    sent = json.dumps(workspace.serving_endpoints.calls)
    for probe in ('/v1/health/ready', '/v1/models', 'health', 'ready'):
        assert probe not in sent


def test_no_path_is_appended_to_anything() -> None:
    client, workspace = _client()
    client._request(PREDICTION, json={'model': 'kumo-relational'})
    call = workspace.serving_endpoints.calls[0]
    assert '/v1/predictions' not in call['name']
    assert 'url' not in call and 'path' not in call


@pytest.mark.parametrize(
    'bad',
    [
        'https://workspace.cloud.databricks.com/serving-endpoints/relational-endpoint',
        'workspace/relational-endpoint',
        'relational-endpoint?x=1',
        'relational-endpoint#frag',
        ' relational-endpoint',
        '',
    ],
)
def test_rejects_a_url_shaped_endpoint(bad: str) -> None:
    """The most likely way raw-NIM habits leak back in."""
    with pytest.raises(ValueError):
        DatabricksServingClient(bad, _Workspace())


@pytest.mark.parametrize(
    'bad',
    [
        'https://user:PWSECRET@acme.cloud.databricks.com/x?token=dapiTOK',
        'https://user:PWSECRET@acme.cloud.databricks.com/x?token=dapiTOK\n',
        '  https://user:PWSECRET@acme.cloud.databricks.com/x?token=dapiTOK  ',
        'user:PWSECRET@acme.cloud.databricks.com',
    ],
)
def test_a_rejected_url_never_echoes_credentials(bad: str) -> None:
    """A pasted URL usually arrives with trailing whitespace. An earlier
    version checked whitespace *first* and quoted the input, leaking exactly
    the credentials the URL check exists to protect. The schemeless case
    carries userinfo with none of the other markers.
    """
    with pytest.raises(ValueError) as caught:
        DatabricksServingClient(bad, _Workspace())
    for leak in ('PWSECRET', 'dapiTOK', 'acme.cloud.databricks.com'):
        assert leak not in str(caught.value)


# -- the v1 boundary -------------------------------------------------------


def test_sends_one_request_json_row() -> None:
    client, workspace = _client()
    request = {
        'model': 'kumo-relational',
        'task': {'kind': 'binary_classification'},
    }
    client._request(PREDICTION, json=request)
    records = workspace.serving_endpoints.calls[0]['dataframe_records']
    assert len(records) == 1
    assert list(records[0]) == [REQUEST_COLUMN]
    assert json.loads(records[0][REQUEST_COLUMN]) == request


def test_reads_one_response_json_row() -> None:
    client, _ = _client()
    body = client._request(PREDICTION, json={'model': 'kumo-relational'}).json()
    assert body == CANONICAL_RESPONSE


def test_rejects_a_non_prediction_endpoint() -> None:
    """Sessions are excluded from v1; a caller must fail loudly rather than
    have the request quietly reinterpreted as a prediction.
    """
    client, workspace = _client()
    with pytest.raises(HTTPException) as caught:
        client._request(SESSIONS, json={})
    assert caught.value.status_code == 501
    assert workspace.serving_endpoints.calls == []


@pytest.mark.parametrize(
    'reply',
    [
        {'predictions': []},
        {'predictions': [{'x': 1}, {'x': 2}]},
        {'predictions': [{'wrong_column': '{}'}]},
        {'nothing': True},
    ],
)
def test_rejects_a_malformed_reply(reply: Any) -> None:
    client, _ = _client(reply=reply)
    with pytest.raises(HTTPException) as caught:
        client._request(PREDICTION, json={'model': 'kumo-relational'})
    assert caught.value.status_code == 502


def test_rejects_an_oversized_request_before_egress() -> None:
    workspace = _Workspace()
    client = DatabricksServingClient(
        'kumo-relational', workspace, max_request_bytes=256
    )
    with pytest.raises(HTTPException) as caught:
        client._request(
            PREDICTION,
            json={'model': 'kumo-relational', 'pad': 'x' * 512},
        )
    assert caught.value.status_code == 413
    assert workspace.serving_endpoints.calls == [], (
        'paid to send an oversized body'
    )


def test_the_cap_measures_the_escaped_wire_body_not_the_inner_payload() -> None:
    """A pad of ``"x" * N`` escapes 1:1, the one payload shape that cannot
    detect this bug. Quote-dense rows inflate ~1.15x once nested inside
    request_json, so a request measured on the inner payload alone passes every
    local check and is refused at the gateway as a bare "request failed".
    """
    payload = json.dumps(QUOTE_DENSE, allow_nan=False, separators=(',', ':'))
    wire = _wire_size(QUOTE_DENSE)
    assert wire > len(payload) * 1.1, 'fixture is not quote-dense enough'

    # A cap the inner payload clears but the wire body does not.
    workspace = _Workspace()
    client = DatabricksServingClient(
        'kumo-relational', workspace, max_request_bytes=len(payload) + 64
    )
    with pytest.raises(HTTPException) as caught:
        client._request(PREDICTION, json=QUOTE_DENSE)
    assert caught.value.status_code == 413
    assert str(wire) in caught.value.detail, 'the 413 must name the wire size'
    assert workspace.serving_endpoints.calls == []


def test_what_is_measured_is_what_is_sent() -> None:
    """The serialized dataframe_records body actually handed to the client is
    within the cap, not merely the payload the cap was computed from.
    """
    wire = _wire_size(QUOTE_DENSE)
    workspace = _Workspace()
    client = DatabricksServingClient(
        'kumo-relational', workspace, max_request_bytes=wire
    )
    client._request(PREDICTION, json=QUOTE_DENSE)
    records = workspace.serving_endpoints.calls[0]['dataframe_records']
    body = json.dumps({'dataframe_records': records}, separators=(',', ':'))
    assert len(body.encode('utf-8')) <= wire


@pytest.mark.parametrize(
    ('cap_delta', 'is_sent'), [(-1, False), (0, True), (1, True)]
)
def test_the_cap_boundary_is_exact(cap_delta: int, is_sent: bool) -> None:
    """The cap is the largest size the gateway accepts, so exactly-cap ships."""
    wire = _wire_size(QUOTE_DENSE)
    workspace = _Workspace()
    client = DatabricksServingClient(
        'kumo-relational', workspace, max_request_bytes=wire + cap_delta
    )
    if is_sent:
        client._request(PREDICTION, json=QUOTE_DENSE)
        assert len(workspace.serving_endpoints.calls) == 1
        return
    with pytest.raises(HTTPException) as caught:
        client._request(PREDICTION, json=QUOTE_DENSE)
    assert caught.value.status_code == 413
    assert str(wire) in caught.value.detail
    assert str(wire + cap_delta) in caught.value.detail
    assert workspace.serving_endpoints.calls == []


@pytest.mark.parametrize('bad', [0, -1, 1.5, '16000000', None, True])
def test_rejects_an_invalid_max_request_bytes(bad: Any) -> None:
    """Documented as a user override, so it is user input."""
    with pytest.raises(ValueError):
        DatabricksServingClient(
            'kumo-relational', _Workspace(), max_request_bytes=bad
        )


def test_the_default_cap_is_the_platform_limit_not_the_contract_cap() -> None:
    """These are different limits and the smaller one has to win.

    Databricks refuses a serving request above 16 MB at the gateway; the
    dispatcher's own cap is 30 MiB. Defaulting to 30 MiB would let a request in
    between pass every local check and be rejected by the platform, coming back
    as a generic failure naming neither the size nor the limit, observed on a
    real endpoint, and indistinguishable from the endpoint being broken.
    """
    from sdm_relational_engine.client.databricks_serving import (
        MAX_REQUEST_BYTES,
    )

    assert MAX_REQUEST_BYTES == 16 * 1000 * 1000
    assert MAX_REQUEST_BYTES < 30 * 1024 * 1024, (
        "the transport cap must sit below the dispatcher's, or it cannot "
        'convert a gateway rejection into a local error'
    )


def test_a_request_over_the_platform_limit_never_leaves_the_process() -> None:
    """The default, not an injected bound: a body just over 16 MB must fail
    locally rather than be paid for and refused.
    """
    workspace = _Workspace()
    client = DatabricksServingClient('kumo-relational', workspace)
    with pytest.raises(HTTPException) as caught:
        client._request(
            PREDICTION,
            json={
                'model': 'kumo-relational',
                'pad': 'x' * (16 * 1000 * 1000),
            },
        )
    assert caught.value.status_code == 413
    assert workspace.serving_endpoints.calls == []


def test_rejects_non_finite_numbers() -> None:
    """NaN and Infinity are not JSON; the server rejects them, so failing here
    saves a round trip and gives a clearer error.
    """
    client, _ = _client()
    with pytest.raises(ValueError):
        client._request(
            PREDICTION,
            json={'model': 'kumo-relational', 'x': float('inf')},
        )


# -- error translation -----------------------------------------------------


@pytest.mark.parametrize(
    ('exc_name', 'expected'),
    [
        ('PermissionDenied', 403),
        ('Unauthenticated', 401),
        ('NotFound', 404),
        ('TooManyRequests', 429),
        ('RequestLimitExceeded', 429),
        ('DeadlineExceeded', 504),
        ('OperationTimeout', 504),
        ('TimeoutError', 504),
        ('Timeout', 504),
        ('BadRequest', 400),
        ('SomethingElse', 503),
    ],
)
def test_translates_provider_errors(exc_name: str, expected: int) -> None:
    error = type(exc_name, (Exception,), {})(f'boom on {SECRET}')
    client, _ = _client(raiser=error)
    with pytest.raises(HTTPException) as caught:
        client._request(PREDICTION, json={'model': 'kumo-relational'})
    assert caught.value.status_code == expected


def test_provider_messages_are_never_echoed() -> None:
    """A provider error can carry host names, request fragments and token
    material. None of it may reach the caller.
    """
    error = type('PermissionDenied', (Exception,), {})(
        f'denied for token abcd1234 on host acme.cloud.databricks.com: {SECRET}'
    )
    client, _ = _client(raiser=error)
    with pytest.raises(HTTPException) as caught:
        client._request(PREDICTION, json={'model': 'kumo-relational'})
    rendered = f'{caught.value.status_code} {caught.value.detail}'
    for leak in (SECRET, 'abcd1234', 'acme.cloud.databricks.com'):
        assert leak not in rendered
    assert caught.value.__cause__ is None


def test_debug_logging_carries_no_request_or_provider_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    error = type('BadRequest', (Exception,), {})(f'bad: {SECRET}')
    client, _ = _client(raiser=error)
    with pytest.raises(HTTPException):
        client._request(
            PREDICTION, json={'model': 'kumo-relational', 'note': SECRET}
        )
    captured = '\n'.join(record.getMessage() for record in caplog.records)
    assert SECRET not in captured


def test_debug_logging_keeps_the_provider_error_for_diagnosis(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A contract mismatch is otherwise unrecoverable without a packet capture:
    the message must survive somewhere a DEBUG handler can reach it, while the
    log's own message text stays free of provider content.
    """
    caplog.set_level(logging.DEBUG, logger='sdm_relational_engine')
    error = type('BadRequest', (Exception,), {})(f'bad: {SECRET}')
    client, _ = _client(raiser=error)
    with pytest.raises(HTTPException):
        client._request(PREDICTION, json={'model': 'kumo-relational'})
    record = next(
        r for r in caplog.records if r.name == 'sdm_relational_engine'
    )
    assert record.exc_info is not None
    assert SECRET not in record.getMessage()


def test_the_provider_error_code_is_surfaced_but_the_message_is_not() -> None:
    """error_code is an enum token, not free text, so it is safe to name and it
    is the whole difference between diagnosable and not.
    """
    error = type('BadRequest', (Exception,), {})(f'bad: {SECRET}')
    error.error_code = 'INVALID_PARAMETER_VALUE'
    client, _ = _client(raiser=error)
    with pytest.raises(HTTPException) as caught:
        client._request(PREDICTION, json={'model': 'kumo-relational'})
    assert 'INVALID_PARAMETER_VALUE' in caught.value.detail
    assert SECRET not in caught.value.detail


def test_a_non_enum_error_code_is_withheld() -> None:
    """Nothing guarantees a provider keeps error_code enum-valued, and free text
    is exactly the channel the no-echo guarantee closes.
    """
    error = type('BadRequest', (Exception,), {})('bad')
    error.error_code = f'denied for token on acme.databricks.com: {SECRET}'
    client, _ = _client(raiser=error)
    with pytest.raises(HTTPException) as caught:
        client._request(PREDICTION, json={'model': 'kumo-relational'})
    assert SECRET not in caught.value.detail


def test_silences_the_sdk_body_logger() -> None:
    """Databricks-sdk logs full request and response bodies at DEBUG, and RFM
    payloads carry sampled customer rows.
    """
    sdk_logger = logging.getLogger('databricks.sdk.core')
    original = sdk_logger.level
    try:
        sdk_logger.setLevel(logging.DEBUG)
        _client()
        assert sdk_logger.level >= logging.INFO
    finally:
        sdk_logger.setLevel(original)


def test_the_sdk_logger_floor_is_raised_not_overwritten() -> None:
    """Setting an exact level would make a user who chose ERROR *noisier*, for
    a third-party logger this module does not own.
    """
    sdk_logger = logging.getLogger('databricks.sdk.core')
    original = sdk_logger.level
    try:
        sdk_logger.setLevel(logging.ERROR)
        _client()
        assert sdk_logger.level == logging.ERROR
    finally:
        sdk_logger.setLevel(original)


# -- timeouts --------------------------------------------------------------


def test_the_timeout_reaches_a_self_constructed_workspace_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serving endpoints scale to zero; the client's ~60s default turns a normal
    cold start into a timeout.
    """
    sdk = pytest.importorskip('databricks.sdk', reason='databricks-sdk absent')
    core = pytest.importorskip(
        'databricks.sdk.core', reason='databricks-sdk absent'
    )
    seen: dict[str, Any] = {}

    class _FakeConfig:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    def _fake_workspace_client(*, config: Any) -> Any:
        seen['config'] = config
        return _Workspace()

    monkeypatch.setattr(core, 'Config', _FakeConfig)
    monkeypatch.setattr(sdk, 'WorkspaceClient', _fake_workspace_client)
    DatabricksServingClient('kumo-relational', timeout=123.0)
    assert seen['http_timeout_seconds'] == 123.0
    assert isinstance(seen['config'], _FakeConfig)


def test_the_default_timeout_allows_for_a_cold_start() -> None:
    from sdm_relational_engine.client.databricks_serving import (
        DEFAULT_TIMEOUT_SECONDS,
    )

    assert DEFAULT_TIMEOUT_SECONDS > 60, (
        'the client default already is ~60s; a shorter floor cannot survive a '
        'scale-to-zero wake-up'
    )


def test_an_injected_workspace_client_is_never_reconfigured() -> None:
    """The timeout is ours to set only on a client we built."""
    workspace = _Workspace()
    before = set(workspace.__dict__)
    DatabricksServingClient('kumo-relational', workspace, timeout=1.0)
    assert set(workspace.__dict__) == before
    assert workspace.touched == []


@pytest.mark.parametrize('bad', [0, -1, '60', None, True])
def test_rejects_an_invalid_timeout(bad: Any) -> None:
    """True is the one that matters: bool is an int, so it would otherwise
    pass and become a one-second timeout, the cold start this guards.
    """
    with pytest.raises(ValueError):
        DatabricksServingClient('kumo-relational', _Workspace(), timeout=bad)


def test_missing_extra_reports_the_extra_to_install() -> None:
    """Constructing without a workspace client and without databricks-sdk."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith('databricks'):
            raise ImportError(f'No module named {name!r}')
        return real_import(name, *args, **kwargs)

    builtins.__import__ = _blocked
    try:
        with pytest.raises(ImportError) as caught:
            DatabricksServingClient('kumo-relational')
        assert 'databricks-serving' in str(caught.value)
    finally:
        builtins.__import__ = real_import


# -- grounded against the real client type ------------------------------------


def test_the_prediction_path_tracks_the_generated_spec() -> None:
    """Hardcoding the path in both the transport and the test would let a
    codegen bump to /v2/ 501 every serving request with the suite still green.
    """
    from sdm_relational_engine.client import databricks_serving

    assert (
        TFMOperations.run_prediction.endpoint.get_path()
        == databricks_serving._PREDICTION_PATH
    )


@pytest.mark.parametrize(
    ('class_name', 'expected'),
    [
        ('PermissionDenied', 403),
        ('Unauthenticated', 401),
        ('ResourceDoesNotExist', 404),
        ('TooManyRequests', 429),
        ('RequestLimitExceeded', 429),
        ('DeadlineExceeded', 504),
        ('OperationTimeout', 504),
        ('TemporarilyUnavailable', 503),
    ],
)
def test_the_error_mapping_is_grounded_in_the_real_sdk_names(
    class_name: str, expected: int
) -> None:
    """The mapping matches on class names, so a renamed class degrades to 503
    silently. These are the names as the installed client actually spells them.
    """
    errors = pytest.importorskip(
        'databricks.sdk.errors', reason='databricks-sdk not installed'
    )
    assert _status_of(getattr(errors, class_name)('boom')) == expected


def test_reads_a_real_query_endpoint_response() -> None:
    """The fake WorkspaceClient above returns a hand-made dict. If that shape
    were wrong the whole suite would be confidently wrong in both directions,
    so this pins it to the client's actual return type.

    Skipped when databricks-sdk is absent; that is the same condition under
    which the transport itself cannot run.
    """
    serving = pytest.importorskip(
        'databricks.sdk.service.serving',
        reason='databricks-sdk not installed',
    )
    reply = serving.QueryEndpointResponse(
        predictions=[{RESPONSE_COLUMN: json.dumps(CANONICAL_RESPONSE)}]
    )
    client, _ = _client(reply=reply)
    body = client._request(PREDICTION, json={'model': 'kumo-relational'}).json()
    assert body == CANONICAL_RESPONSE


def test_the_query_signature_matches_what_the_transport_sends() -> None:
    """A future client that renames dataframe_records would break every call; the
    fake would keep passing because it accepts **kwargs.
    """
    import inspect

    serving = pytest.importorskip(
        'databricks.sdk.service.serving',
        reason='databricks-sdk not installed',
    )
    params = inspect.signature(serving.ServingEndpointsAPI.query).parameters
    assert 'name' in params
    assert 'dataframe_records' in params


def test_a_real_response_without_predictions_is_rejected() -> None:
    """QueryEndpointResponse.predictions is Optional, so this is reachable."""
    serving = pytest.importorskip(
        'databricks.sdk.service.serving',
        reason='databricks-sdk not installed',
    )
    client, _ = _client(reply=serving.QueryEndpointResponse())
    with pytest.raises(HTTPException) as caught:
        client._request(PREDICTION, json={'model': 'kumo-relational'})
    assert caught.value.status_code == 502
