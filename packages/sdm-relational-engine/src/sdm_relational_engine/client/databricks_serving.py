# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Invoke Kumo Relational through a Databricks Model Serving endpoint.

The RFM execution path is unchanged: it still calls ``_request`` and reads
``ok`` / ``status_code`` / ``text`` / ``json()``. What changes is everything
underneath.

The boundary is the frozen v1 contract: one ``request_json`` row in, one
``response_json`` row out.

Two constants below are load-bearing and look like mistakes:

``MAX_REQUEST_BYTES`` is decimal MB rather than MiB because that is the
gateway's own limit; "correcting" it to MiB raises the cap by 5% and the excess
returns as a bare "request failed". It is deliberately not the dispatcher's
30 MiB model-boundary cap, which a request between the two would pass locally
and fail remotely.

``DEFAULT_TIMEOUT_SECONDS`` is far above the client default because a serving
endpoint scales to zero and a cold start outlasts ~60s.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from typing import Any

from sdm_api.endpoints import Endpoint
from sdm_api.generated.tfm_api import TFMOperations

from sdm_relational_engine.client.transport import ServingResponse
from sdm_relational_engine.exceptions import HTTPException

__all__ = ['DatabricksServingClient', 'ServingResponse']

logger = logging.getLogger('sdm_relational_engine')

# The frozen v1 request/response column names. A deployed serving endpoint
# reads and writes exactly these two, so renaming either breaks every endpoint
# already in production.
REQUEST_COLUMN = 'request_json'
RESPONSE_COLUMN = 'response_json'

_PREDICTION_PATH = TFMOperations.run_prediction.endpoint.get_path()

# Decimal MB, not MiB: see the module docstring before changing either.
MAX_REQUEST_BYTES = 16 * 1000 * 1000
DEFAULT_TIMEOUT_SECONDS = 300.0

_SDK_BODY_LOGGER = 'databricks.sdk.core'

_ERROR_CODE_RE = re.compile(r'^[A-Z][A-Z0-9_]{0,63}$')


def _validate_endpoint_name(name: str) -> str:
    r"""Reject anything URL-shaped.

    A serving endpoint is named, not addressed, and a pasted workspace URL can
    carry userinfo credentials or a token query parameter. So the URL scan runs
    before any check that echoes the value, and the rejection never quotes it.
    ``@`` is in the reject set because a schemeless ``user:token@host`` carries
    userinfo without any of the other markers.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError('endpoint must be a non-empty serving endpoint name')
    for token in ('://', '/', '?', '#', '@'):
        if token in name:
            raise ValueError(
                'endpoint must be a serving endpoint name, not a URL'
            )
    if name != name.strip():
        raise ValueError('endpoint name has surrounding whitespace')
    return name


def _status_of(error: BaseException) -> int:
    r"""Map a Databricks SDK error onto an HTTP-equivalent status.

    Matched by class name so ``databricks.sdk`` need not be importable to read
    this module, and a version bump that moves the classes cannot break the
    mapping silently. An unclassified failure is 503 rather than 500, because it
    is usually the endpoint being unavailable and 503 is what RFM retries.
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


class DatabricksServingClient:
    r"""Address a Kumo Relational model served as a Databricks Model Serving endpoint.

    Args:
        endpoint: The serving endpoint **name**.
        workspace_client: An existing ``WorkspaceClient``. When omitted one is
            constructed from the ambient Databricks configuration, which is how
            a notebook authenticates without handling a token.
        max_request_bytes: Client-side cap, checked before the request leaves
            the process. Rejected if boolean: ``True`` is an ``int`` and would
            reject every request.
        timeout: HTTP timeout in seconds, applied only to a self-constructed
            workspace client; an injected client keeps its own configuration.
            Rejected if boolean: ``float(True)`` is one second, which is the
            cold start this timeout exists for.
    """

    def __init__(
        self,
        endpoint: str,
        workspace_client: Any | None = None,
        *,
        max_request_bytes: int = MAX_REQUEST_BYTES,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._endpoint = _validate_endpoint_name(endpoint)
        if (
            not isinstance(max_request_bytes, int)
            or isinstance(max_request_bytes, bool)
            or max_request_bytes <= 0
        ):
            raise ValueError('max_request_bytes must be a positive integer')
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or timeout <= 0
        ):
            raise ValueError('timeout must be a positive number of seconds')
        self._max_request_bytes = max_request_bytes
        self._timeout = float(timeout)
        sdk_logger = logging.getLogger(_SDK_BODY_LOGGER)
        if sdk_logger.getEffectiveLevel() < logging.INFO:
            sdk_logger.setLevel(logging.INFO)
        if workspace_client is not None:
            self._workspace = workspace_client
        else:
            self._workspace = self._default_workspace(self._timeout)

    @staticmethod
    def _default_workspace(timeout: float) -> Any:
        try:
            from databricks.sdk import WorkspaceClient
            from databricks.sdk.core import Config
        except ImportError as error:
            raise ImportError(
                'Databricks Model Serving support requires the '
                "'databricks-serving' extra: pip install "
                "'sdm_relational_engine[databricks-serving]'"
            ) from error
        try:
            return WorkspaceClient(config=Config(http_timeout_seconds=timeout))
        except Exception:
            raise HTTPException(
                503,
                'could not authenticate to Databricks from the ambient '
                'configuration',
            ) from None

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def close(self) -> None:
        r"""Nothing to release: the workspace client owns its own transport."""

    def _request(self, endpoint: Endpoint, **kwargs: Any) -> ServingResponse:
        path = endpoint.get_path()
        if path != _PREDICTION_PATH:
            raise HTTPException(
                501,
                'Databricks Model Serving supports one-shot prediction only; '
                f'{path!r} is not available in this mode',
            )
        request = kwargs.get('json')
        if not isinstance(request, Mapping):
            raise HTTPException(400, 'prediction request must be a JSON object')
        return self._query(request)

    def _query(self, request: Mapping[str, Any]) -> ServingResponse:
        r"""Send one canonical request and return the serving reply.

        The cap is measured on the serialized envelope rather than on the
        payload, because the gateway weighs what is sent and every quote in the
        payload is re-escaped inside it (~1.15x): measuring the payload alone
        ships 18.3 MB believing it is 15.9. ``records`` is reused below, so what
        is measured is what is sent, and ``ensure_ascii`` makes ``len`` the byte
        count. The comparison is ``>`` because the cap is the largest size the
        gateway accepts.
        """
        payload = json.dumps(request, allow_nan=False, separators=(',', ':'))
        records = [{REQUEST_COLUMN: payload}]
        body = json.dumps({'dataframe_records': records}, separators=(',', ':'))
        size = len(body)
        if size > self._max_request_bytes:
            raise HTTPException(
                413,
                f'request body is {size} bytes on the wire, above the '
                f'{self._max_request_bytes} byte limit',
            )

        try:
            reply = self._workspace.serving_endpoints.query(
                name=self._endpoint,
                dataframe_records=records,
            )
        except Exception as error:
            status = _status_of(error)
            code = _error_code_of(error)
            logger.debug(
                'serving endpoint %r failed with %s',
                self._endpoint,
                type(error).__name__,
                exc_info=True,
            )
            detail = f'serving endpoint {self._endpoint!r} request failed'
            if code is not None:
                detail = f'{detail} ({code})'
            raise HTTPException(status, detail) from None

        return ServingResponse(200, self._body_of(reply))

    def _body_of(self, reply: Any) -> str:
        r"""Extract the single canonical response the contract promises."""
        predictions = getattr(reply, 'predictions', None)
        if predictions is None and isinstance(reply, Mapping):
            predictions = reply.get('predictions')
        if not isinstance(predictions, list) or len(predictions) != 1:
            raise HTTPException(
                502,
                'serving endpoint did not return exactly one prediction row',
            )

        row = predictions[0]
        if isinstance(row, Mapping):
            body = row.get(RESPONSE_COLUMN)
        else:
            body = getattr(row, RESPONSE_COLUMN, None)
        if not isinstance(body, str):
            raise HTTPException(
                502,
                f'serving endpoint response row has no {RESPONSE_COLUMN!r} '
                'string column',
            )
        return body
