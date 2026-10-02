# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A client's requests, carried to a Databricks Model Serving endpoint.

The model is served by name inside a workspace rather than at a URL, and is
reached with the workspace's own identity, so a notebook needs no token. Each
request travels as the one ``request_json`` row the served model reads, and
the answer comes back as one ``response_json`` row: the frozen envelope every
deployed endpoint reads and writes.

What a NIM serves and this does not: sessions, and a list of models. A
serving endpoint hosts one model and answers one request at a time.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections.abc import Mapping
from typing import Any

from sdm_client.core.databricks_user_agent import sdm_client_user_agent
from sdm_client.core.transport import (
    _check_encodable,
    _diagnosis_in,
    _validate_limits,
)
from sdm_client.errors import (
    MissingExtraError,
    NimRequestError,
    StructuredDataError,
)

logger = logging.getLogger('sdm_client')

#: A serving endpoint scales to zero, and a cold start outlasts the minute a
#: NIM call is given.
DEFAULT_TIMEOUT_SECONDS = 300.0

# Renaming either column breaks every endpoint already deployed.
_REQUEST_COLUMN = 'request_json'
_RESPONSE_COLUMN = 'response_json'

# The gateway's own limit, in decimal MB: "correcting" it to MiB lets through
# 5% more, which then fails as a bare "request failed".
_MAX_REQUEST_BYTES = 16 * 1000 * 1000

_ERROR_CODE_RE = re.compile(r'^[A-Z][A-Z0-9_]{0,63}$')

# The SDK logs each round trip, bodies included, at DEBUG on this logger, and a
# serving call's body is the caller's rows.
_SDK_LOGGER = 'databricks.sdk'
_SERVING_CALL = re.compile(r'/serving-endpoints/[^/\s]+/invocations\b')


class _WithoutServingBodies(logging.Filter):
    r"""Drop the SDK's round-trip record of a serving call, and only that."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.funcName != '_record_request_log':
            return True
        # The first line is ``METHOD /path``; a host may add a path prefix.
        first_line = record.getMessage().split('\n', 1)[0]
        return _SERVING_CALL.search(first_line) is None


_BODY_FILTER = _WithoutServingBodies()
_BODY_FILTER_LOCK = threading.Lock()


def _keep_rows_out_of_the_sdk_log() -> None:
    sdk_logger = logging.getLogger(_SDK_LOGGER)
    with _BODY_FILTER_LOCK:
        if _BODY_FILTER not in sdk_logger.filters:
            sdk_logger.addFilter(_BODY_FILTER)


# The served model reports a failure as ``CODE: detail``. Only the code is
# repeated: the detail is free text, and can quote the rows it was sent.
_MODEL_DIAGNOSIS = re.compile(r'([A-Z][A-Z0-9_]{2,63}): \S')

# The gateway's fixed sentences, then the model's own exception quoted. Only
# this prefix is trusted: an ``Error '`` further in can be the caller's text.
_QUOTED_MODEL_ERROR = re.compile(
    r'Encountered an unexpected error while evaluating the model\.'
    r"(?: [^'.]*\.)* Error '"
)

_NO_SESSIONS = (
    'a Databricks Model Serving endpoint has no sessions: it answers one '
    'request at a time and cannot pin a context. Use predict(), which sends '
    'the context with every call'
)


class DatabricksServingTransport:
    r"""The wire layer to one Databricks Model Serving endpoint.

    Not part of the public API surface; :meth:`SDMClient.for_databricks_serving`
    builds one.

    Args:
        endpoint: The serving endpoint's **name**, not its URL.
        workspace_client: A ``databricks.sdk.WorkspaceClient`` to call through.
            When omitted one is built from the ambient configuration, which is
            how a notebook authenticates without handling a token. Either
            way, its requests name sdm-client in the User-Agent. A passed
            ``WorkspaceClient`` itself is copied for that, not changed; a
            subclass or test double is used as given, without the name.
        timeout: Seconds one attempt may take, and the longest the SDK keeps
            retrying. Set on the client built here, or on the copy of a
            passed ``WorkspaceClient``, never on the caller's own. When
            omitted, a client built here gets 300 and a passed one keeps its
            own. Refused with a subclass or double, which cannot be copied.
        max_request_bytes: The gateway's size limit, checked on the encoded
            envelope before anything is sent.
    """

    serves_sessions = False

    def __init__(
        self,
        endpoint: str,
        workspace_client: Any | None = None,
        *,
        timeout: float | None = None,
        max_request_bytes: int = _MAX_REQUEST_BYTES,
    ) -> None:
        self._endpoint = _validate_endpoint_name(endpoint)
        if timeout is not None:
            _validate_limits(timeout, 0)
        if (
            isinstance(max_request_bytes, bool)
            or not isinstance(max_request_bytes, int)
            or max_request_bytes <= 0
        ):
            raise StructuredDataError(
                f'max_request_bytes must be a positive integer, got '
                f'{max_request_bytes!r}',
                code='INVALID_CONFIGURATION',
            )
        self._max_request_bytes = max_request_bytes
        self._closed = False
        _keep_rows_out_of_the_sdk_log()
        user_agent = sdm_client_user_agent()
        self._workspace = (
            _named_copy(workspace_client, user_agent, timeout)
            if workspace_client is not None
            else _ambient_workspace(
                float(DEFAULT_TIMEOUT_SECONDS if timeout is None else timeout),
                user_agent,
            )
        )

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def url(self) -> str:
        r"""Where requests go: the endpoint's name, since it has no URL."""
        return f'databricks-serving:{self._endpoint}'

    def close(self) -> None:
        r"""Retire the transport. The workspace client owns its connections."""
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise StructuredDataError(
                'this client is closed; construct a new SDMClient',
                code='INVALID_CONFIGURATION',
            )

    def health_ready(self) -> bool:
        r"""Whether the endpoint reports itself ready to serve."""
        self._require_open()
        try:
            described = self._workspace.serving_endpoints.get(
                name=self._endpoint
            )
        except OSError as error:
            raise self._error_from(error) from None
        ready = getattr(getattr(described, 'state', None), 'ready', None)
        return getattr(ready, 'value', ready) == 'READY'

    def list_models(self) -> list[str]:
        raise StructuredDataError(
            'a Databricks Model Serving endpoint serves one model and does '
            'not list the models it serves; name the model on predict()',
            code='UNSUPPORTED_REQUEST',
        )

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_open()
        _check_encodable(payload)
        try:
            records, size = _envelope(payload)
        except (TypeError, ValueError) as error:
            raise StructuredDataError(
                f'this request carries a value that cannot be sent as JSON: '
                f'{error}',
                code='INVALID_REQUEST',
            ) from error
        if size > self._max_request_bytes:
            raise StructuredDataError(
                f'this request is {size:,} bytes once wrapped for serving '
                f'endpoint {self._endpoint!r}, above the '
                f'{self._max_request_bytes:,} byte limit it accepts. Send '
                f'fewer rows per call: a smaller num_context_samples, or '
                f'fewer entities to score',
                code='INVALID_REQUEST',
                details={'bytes': size, 'limit': self._max_request_bytes},
            )
        try:
            reply = self._workspace.serving_endpoints.query(
                name=self._endpoint, dataframe_records=records
            )
        # The SDK's errors and its retry deadline are all OSError, as are
        # requests' own; anything else is a bug, and is not dressed as a 503.
        # The SDK's ValueError for a failed OAuth refresh escapes too; it
        # carries no rows.
        except OSError as error:
            raise self._error_from(error) from None
        return self._decoded(_response_body(reply, self._endpoint))

    def _decoded(self, body: str) -> dict[str, Any]:
        try:
            # A deeply nested body raises RecursionError, which is a
            # RuntimeError and would otherwise escape the error contract. Not
            # chained: a JSONDecodeError holds the whole body, rows and all.
            answer = json.loads(body)
        except (ValueError, RecursionError):
            raise StructuredDataError(
                f'serving endpoint {self._endpoint!r} answered with a '
                f'response that is not JSON',
                code='TRANSPORT_ERROR',
            ) from None
        if not isinstance(answer, dict):
            raise StructuredDataError(
                f'expected a JSON object from serving endpoint '
                f'{self._endpoint!r}, got {type(answer).__name__}',
                code='TRANSPORT_ERROR',
            )
        status = answer.get('status')
        # A failure says so by its status alone; a body need not explain it.
        if (
            isinstance(status, int)
            and not isinstance(status, bool)
            and status >= 400
        ):
            raise self._refused_in_band(status, answer)
        return answer

    def _refused_in_band(
        self, status: int, answer: dict[str, Any]
    ) -> NimRequestError:
        r"""A failure the served model answered with, named by its code alone.

        The body's text is not repeated: it can quote the rows it was sent.
        """
        said = answer.get('code')
        code = (
            said
            if isinstance(said, str) and _ERROR_CODE_RE.match(said)
            else None
        )
        detail = answer.get('detail')
        if status >= 500:
            found = _diagnosis_in(detail or answer.get('title'))
            if found is not None:
                return self._callers_fault(status, found[1], code, None)
            # A 5xx can be the model naming what was wrong with the request.
            if isinstance(detail, str):
                code = _leading_code(detail) or code
        return self._refusal(status, code, None)

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise StructuredDataError(_NO_SESSIONS, code='SESSIONS_UNSUPPORTED')

    def session_predict(
        self, session_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        raise StructuredDataError(_NO_SESSIONS, code='SESSIONS_UNSUPPORTED')

    def delete_session(self, session_id: str) -> None:
        raise StructuredDataError(_NO_SESSIONS, code='SESSIONS_UNSUPPORTED')

    def _error_from(self, error: BaseException) -> NimRequestError:
        r"""The endpoint's failure, as the client reports any endpoint's.

        The status comes from the SDK's error class and the code from the
        served model's own diagnosis where it gave one. The SDK's message is
        neither repeated nor logged: it can hold the request body, which is the
        caller's rows.
        """
        status = _status_of(error)
        provider_code = _error_code_of(error)
        said = getattr(error, 'message', None)
        text = said if isinstance(said, str) else str(error)
        code = _model_diagnosis(text, provider_code)
        logger.debug(
            'serving endpoint %r failed with %s (status %s, code %s)',
            self._endpoint,
            type(error).__name__,
            status,
            code or provider_code,
        )
        found = _diagnosis_in(text) if status >= 500 else None
        if found is not None:
            return self._callers_fault(
                status,
                found[1],
                code or provider_code,
                provider_code,
                provider_error=type(error).__name__,
            )
        return self._refusal(
            status, code, provider_code, provider_error=type(error).__name__
        )

    def _callers_fault(
        self,
        status: int,
        diagnosis: str,
        endpoint_code: str | None,
        provider_code: str | None,
        **details: Any,
    ) -> NimRequestError:
        r"""A failure the endpoint called its own, reported as the caller's.

        As a NIM's is: retrying one that can only fail the same way does not
        help, and what the endpoint called it is kept in ``details``.
        """
        return self._refusal(
            status,
            'INVALID_REQUEST',
            provider_code,
            endpoint_code=endpoint_code,
            diagnosis=diagnosis,
            **details,
        )

    def _refusal(
        self,
        status: int,
        code: str | None,
        provider_code: str | None,
        **details: Any,
    ) -> NimRequestError:
        r"""The error for a failure known only by its status and codes."""
        details = {'endpoint': self._endpoint, **details}
        if provider_code is not None:
            details['provider_code'] = provider_code
        if code is not None:
            return NimRequestError(
                status,
                code=code,
                message=(
                    f'serving endpoint {self._endpoint!r} refused the '
                    f'request ({code})'
                ),
                details=details,
            )
        suffix = f' ({provider_code})' if provider_code else ''
        return NimRequestError(
            status,
            code=provider_code,
            message=f'serving endpoint {self._endpoint!r} request failed{suffix}',
            details=details,
        )


def _leading_code(text: str) -> str | None:
    r"""The code ``text`` opens with, as ``CODE: detail``, if it does."""
    found = _MODEL_DIAGNOSIS.match(text)
    return found.group(1) if found else None


def _model_diagnosis(text: str, provider_code: str | None) -> str | None:
    r"""The code the served model opened its message with, if the SDK
    carried one.

    Read only where the model's message begins: at the start, after the
    provider's own ``CODE: `` prefix, or in the ``Error '...'`` right after
    the gateway's fixed wrapper. An upper-case word anywhere else can be the
    caller's value.
    The provider's code is skipped: ``BAD_REQUEST: ...`` is Databricks
    describing the call, not the model describing the request.
    """
    prefix = f'{provider_code}: '
    if provider_code is not None and text.startswith(prefix):
        text = text[len(prefix) :]
    starts = [text]
    quoted = _QUOTED_MODEL_ERROR.match(text)
    if quoted is not None:
        starts.append(text[quoted.end() :])
    for start in starts:
        code = _leading_code(start)
        if code is not None and code != provider_code:
            return code
    return None


def _validate_endpoint_name(name: object) -> str:
    r"""Reject anything URL-shaped, without echoing it.

    A serving endpoint is named, not addressed, and a pasted workspace URL can
    carry userinfo credentials or a token query parameter. So the URL scan runs
    before any check that echoes the value, and the rejection never quotes it.
    ``@`` is in the reject set because a schemeless ``user:token@host`` carries
    userinfo without any of the other markers.

    Raises:
        StructuredDataError: ``INVALID_CONFIGURATION``, with a message that
            does not contain ``name``.
    """
    if not isinstance(name, str) or not name.strip():
        refusal = 'endpoint must be a non-empty serving endpoint name'
    elif any(token in name for token in ('://', '/', '?', '#', '@')):
        refusal = 'endpoint must be a serving endpoint name, not a URL'
    elif name != name.strip():
        refusal = 'endpoint name has surrounding whitespace'
    else:
        return name
    raise StructuredDataError(refusal, code='INVALID_CONFIGURATION')


def _envelope(request: Mapping[str, Any]) -> tuple[list[dict[str, str]], int]:
    r"""The ``dataframe_records`` to send, and their size on the wire.

    The size is of the body the gateway weighs, which re-escapes every quote in
    the payload (~1.15x), and the records returned are the ones measured.
    ``ensure_ascii`` (the default) makes ``len`` the byte count.

    Raises:
        ValueError: If ``request`` holds a value with no strict JSON form,
            such as ``NaN``.
    """
    payload = json.dumps(request, allow_nan=False, separators=(',', ':'))
    records = [{_REQUEST_COLUMN: payload}]
    body = json.dumps({'dataframe_records': records}, separators=(',', ':'))
    return records, len(body)


def _response_body(reply: Any, endpoint: str) -> str:
    r"""The single canonical response string the contract promises.

    ``reply`` is whatever the SDK returned: an object with ``predictions`` or a
    mapping holding it. Each row may be a mapping or an object.

    Raises:
        StructuredDataError: ``TRANSPORT_ERROR``, if there is not exactly one
            row, or it holds no ``response_json`` string.
    """
    predictions = getattr(reply, 'predictions', None)
    if predictions is None and isinstance(reply, Mapping):
        predictions = reply.get('predictions')
    if not isinstance(predictions, list) or len(predictions) != 1:
        raise StructuredDataError(
            f'serving endpoint {endpoint!r} did not return exactly one '
            f'prediction row',
            code='TRANSPORT_ERROR',
        )
    row = predictions[0]
    if isinstance(row, Mapping):
        body = row.get(_RESPONSE_COLUMN)
    else:
        body = getattr(row, _RESPONSE_COLUMN, None)
    if not isinstance(body, str):
        raise StructuredDataError(
            f'serving endpoint {endpoint!r} answered with a row that has no '
            f'{_RESPONSE_COLUMN!r} string column',
            code='TRANSPORT_ERROR',
        )
    return body


def _status_of(error: BaseException) -> int:
    r"""Map a Databricks SDK error onto an HTTP-equivalent status.

    Matched by class name so the SDK need not be importable to read this
    module, and a version bump that moves the classes cannot break the mapping
    silently. An unclassified failure is 503 rather than 500, because it is
    usually the endpoint being unavailable, and 503 is what callers retry.
    """
    names = {cls.__name__ for cls in type(error).__mro__}
    if names & {'PermissionDenied', 'Unauthenticated'}:
        return 403 if 'PermissionDenied' in names else 401
    if 'NotFound' in names:
        return 404
    if 'ResourceConflict' in names:
        return 409
    if names & {'TooManyRequests', 'RequestLimitExceeded'}:
        return 429
    if names & {
        'DeadlineExceeded',
        'OperationTimeout',
        'TimeoutError',
        'Timeout',
    }:
        return 504
    if 'BadRequest' in names:
        return 400
    status = getattr(error, 'status_code', None)
    return status if isinstance(status, int) else 503


def _error_code_of(error: BaseException) -> str | None:
    r"""The provider's error code, when enum-shaped enough to be non-secret."""
    code = getattr(error, 'error_code', None)
    if isinstance(code, str) and _ERROR_CODE_RE.match(code):
        return code
    return None


def _named_copy(
    workspace_client: Any, user_agent: tuple[str, str], timeout: float | None
) -> Any:
    r"""A client like ``workspace_client`` whose requests name this SDK, and
    are bounded by ``timeout`` when one is given.

    The caller's client is left as it was: its config is copied, which shares
    the credentials by reference. Only the SDK's own class is rebuilt, since a
    subclass or a test double would lose what it adds; it is used as given.

    Raises:
        StructuredDataError: ``INVALID_CONFIGURATION``, if ``timeout`` is
            given for a client that is used as given.
    """
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError:
        return _as_given(workspace_client, timeout)
    if type(workspace_client) is not WorkspaceClient:
        return _as_given(workspace_client, timeout)
    config = workspace_client.config.copy()
    if timeout is not None:
        # Private SDK state: a release that reshapes it must not have the
        # bound set on the caller's own config instead.
        if not isinstance(getattr(config, '_inner', None), dict):
            raise StructuredDataError(
                'timeout cannot be applied to a copy of this databricks-sdk '
                "version's WorkspaceClient; set http_timeout_seconds and "
                "retry_timeout_seconds on the client's own Config instead",
                code='INVALID_CONFIGURATION',
            )
        # copy() shares the attribute dict with the caller's config, so a
        # bound set without this would change their client too.
        config._inner = dict(config._inner)
        # The SDK retries for its own retry_timeout_seconds whatever one
        # attempt's timeout, so both are bounded by the caller's.
        config.http_timeout_seconds = float(timeout)
        config.retry_timeout_seconds = math.ceil(timeout)
    # Before the client is built: its ApiClient reads both timeouts and the
    # User-Agent once, then.
    return WorkspaceClient(config=config.with_user_agent_extra(*user_agent))


def _as_given(workspace_client: Any, timeout: float | None) -> Any:
    kind = type(workspace_client).__name__
    if timeout is not None:
        # Ignoring it would leave the caller believing a bound they asked
        # for is in force.
        raise StructuredDataError(
            f'timeout is set on a copy of the workspace client, and a {kind} '
            f'cannot be copied: only a databricks.sdk.WorkspaceClient itself '
            f'can. Set http_timeout_seconds and retry_timeout_seconds on its '
            f'Config instead',
            code='INVALID_CONFIGURATION',
        )
    logger.debug(
        'workspace_client is a %s, not a WorkspaceClient; its requests are '
        'sent as given, without the sdm-client User-Agent entry',
        kind,
    )
    return workspace_client


def _ambient_workspace(timeout: float, user_agent: tuple[str, str]) -> Any:
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.core import Config
    except ImportError:
        raise MissingExtraError(
            'databricks-serving', 'databricks-sdk'
        ) from None
    try:
        # The SDK retries for its own retry_timeout_seconds whatever one
        # attempt's timeout, so both are bounded by the caller's.
        config = Config(
            http_timeout_seconds=timeout,
            retry_timeout_seconds=math.ceil(timeout),
        )
        # Before the client is built: it reads the User-Agent once, then.
        return WorkspaceClient(config=config.with_user_agent_extra(*user_agent))
    except Exception as error:
        raise StructuredDataError(
            f'could not authenticate to Databricks from the ambient '
            f'configuration ({type(error).__name__}). Run this in a Databricks '
            f'notebook, set DATABRICKS_HOST and a credential, or pass '
            f'workspace_client=',
            code='INVALID_CONFIGURATION',
        ) from None
