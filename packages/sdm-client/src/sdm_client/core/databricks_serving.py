# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A client's requests, carried to a Databricks Model Serving endpoint.

The model is served by name inside a workspace rather than at a URL, and is
reached with the workspace's own identity, so a notebook needs no token. Each
request travels as the one ``request_json`` row the served model reads, and
the answer comes back as one ``response_json`` row; the envelope is
:mod:`sdm_api.databricks_serving`, shared with the relational engine.

What a NIM serves and this does not: sessions, and a list of models. A
serving endpoint hosts one model and answers one request at a time.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from sdm_api.databricks_serving import (
    MAX_REQUEST_BYTES,
    envelope,
    error_code_of,
    response_body,
    status_of,
    validate_endpoint_name,
)

from sdm_client.core.transport import (
    _check_encodable,
    _snippet,
    _to_nim_error,
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

# The SDK logs each round trip, bodies included, at DEBUG on this logger, and a
# serving call's body is the caller's rows.
_SDK_LOGGER = 'databricks.sdk'
_SERVING_CALL = re.compile(r'POST /serving-endpoints/[^/\s]+/invocations\b')


class _WithoutServingBodies(logging.Filter):
    r"""Drop the SDK's round-trip record of a serving call, and only that."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.funcName != '_record_request_log':
            return True
        return not _SERVING_CALL.match(record.getMessage())


def _keep_rows_out_of_the_sdk_log() -> None:
    sdk_logger = logging.getLogger(_SDK_LOGGER)
    if not any(
        isinstance(f, _WithoutServingBodies) for f in sdk_logger.filters
    ):
        sdk_logger.addFilter(_WithoutServingBodies())


# The served model reports a failure as ``CODE: detail``, both drawn from a
# closed vocabulary that carries no request content.
_MODEL_DIAGNOSIS = re.compile(r'\b([A-Z][A-Z0-9_]{2,63}): ([^\n\'"]{1,300})')

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
            how a notebook authenticates without handling a token.
        timeout: Seconds to wait for one call, applied only to a workspace
            client built here; an injected one keeps its own configuration.
        max_request_bytes: The gateway's size limit, checked on the encoded
            envelope before anything is sent.
    """

    serves_sessions = False

    def __init__(
        self,
        endpoint: str,
        workspace_client: Any | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_request_bytes: int = MAX_REQUEST_BYTES,
    ) -> None:
        try:
            self._endpoint = validate_endpoint_name(endpoint)
        except ValueError as error:
            raise StructuredDataError(
                str(error), code='INVALID_CONFIGURATION'
            ) from None
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
        self._workspace = (
            workspace_client
            if workspace_client is not None
            else _ambient_workspace(float(timeout))
        )

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def url(self) -> str:
        r"""Where requests go, said the way the engine says it."""
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
        except Exception as error:
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
            records, size = envelope(payload)
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
        except Exception as error:
            raise self._error_from(error) from None
        try:
            body = response_body(reply)
        except ValueError as error:
            raise StructuredDataError(
                f'{error} (serving endpoint {self._endpoint!r})',
                code='TRANSPORT_ERROR',
            ) from None
        return self._decoded(body)

    def _decoded(self, body: str) -> dict[str, Any]:
        try:
            # A deeply nested body raises RecursionError, which is a
            # RuntimeError and would otherwise escape the error contract.
            answer = json.loads(body)
        except (ValueError, RecursionError) as error:
            raise StructuredDataError(
                f'serving endpoint {self._endpoint!r} answered with a '
                f'response that is not JSON',
                code='TRANSPORT_ERROR',
            ) from error
        if not isinstance(answer, dict):
            raise StructuredDataError(
                f'expected a JSON object from serving endpoint '
                f'{self._endpoint!r}, got {type(answer).__name__}',
                code='TRANSPORT_ERROR',
            )
        status = answer.get('status')
        if isinstance(status, int) and status >= 400 and 'detail' in answer:
            raise _to_nim_error(status, body.encode('utf-8'))
        return answer

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
        served model's own diagnosis where it gave one. Nothing else of the
        SDK's message is repeated: it is logged at DEBUG, and what reaches the
        caller is the endpoint's name and what the model said was wrong.
        """
        logger.debug(
            'serving endpoint %r failed with %s',
            self._endpoint,
            type(error).__name__,
            exc_info=error,
        )
        status = status_of(error)
        provider_code = error_code_of(error)
        details: dict[str, Any] = {
            'endpoint': self._endpoint,
            'provider_error': type(error).__name__,
        }
        if provider_code is not None:
            details['provider_code'] = provider_code
        said = _model_diagnosis(error, provider_code)
        if said is not None:
            code, detail = said
            return NimRequestError(
                status,
                code=code,
                message=(
                    f'serving endpoint {self._endpoint!r} refused the '
                    f'request: {_snippet(detail)}'
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


def _model_diagnosis(
    error: BaseException, provider_code: str | None
) -> tuple[str, str] | None:
    r"""The code and detail the served model gave, if the SDK carried them.

    The provider's own code is skipped: a gateway message of the form
    ``BAD_REQUEST: ...`` is Databricks describing the call, not the model
    describing the request, and its text is not held to the model's rule of
    carrying no request content.
    """
    said = getattr(error, 'message', None)
    text = said if isinstance(said, str) else str(error)
    for match in _MODEL_DIAGNOSIS.finditer(text):
        code, detail = match.group(1), match.group(2).strip()
        if code != provider_code and detail:
            return code, detail
    return None


def _ambient_workspace(timeout: float) -> Any:
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.core import Config
    except ImportError:
        raise MissingExtraError(
            'databricks-serving', 'databricks-sdk'
        ) from None
    try:
        return WorkspaceClient(config=Config(http_timeout_seconds=timeout))
    except Exception as error:
        raise StructuredDataError(
            f'could not authenticate to Databricks from the ambient '
            f'configuration ({type(error).__name__}). Run this in a Databricks '
            f'notebook, set DATABRICKS_HOST and a credential, or pass '
            f'workspace_client=',
            code='INVALID_CONFIGURATION',
        ) from None
