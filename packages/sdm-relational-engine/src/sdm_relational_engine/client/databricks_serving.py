# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Invoke Kumo Relational through a Databricks Model Serving endpoint.

The RFM execution path is unchanged: it still calls ``_request`` and reads
``ok`` / ``status_code`` / ``text`` / ``json()``. What changes is everything
underneath.

The boundary is the frozen v1 contract: one ``request_json`` row in, one
``response_json`` row out. The envelope itself -- the two columns, the size
cap, and what the gateway's failures mean -- is owned by
:mod:`sdm_api.databricks_serving`, which the client's serving transport reads
too, so the two cannot drift apart. What stays here is the engine's side of
it: its ``HTTPException`` and ``ServingResponse``.

``MAX_REQUEST_BYTES`` is deliberately not the dispatcher's 30 MiB
model-boundary cap, which a request between the two would pass locally and
fail remotely.

``DEFAULT_TIMEOUT_SECONDS`` is far above the client default because a serving
endpoint scales to zero and a cold start outlasts ~60s.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from sdm_api.databricks_serving import (
    MAX_REQUEST_BYTES,
    REQUEST_COLUMN,
    RESPONSE_COLUMN,
    envelope,
    error_code_of,
    response_body,
    status_of,
    validate_endpoint_name,
)
from sdm_api.endpoints import Endpoint
from sdm_api.generated.tfm_api import TFMOperations

from sdm_relational_engine.client.transport import ServingResponse
from sdm_relational_engine.exceptions import HTTPException

__all__ = [
    'MAX_REQUEST_BYTES',
    'REQUEST_COLUMN',
    'RESPONSE_COLUMN',
    'DatabricksServingClient',
    'ServingResponse',
]

logger = logging.getLogger('sdm_relational_engine')

_PREDICTION_PATH = TFMOperations.run_prediction.endpoint.get_path()

DEFAULT_TIMEOUT_SECONDS = 300.0

_SDK_BODY_LOGGER = 'databricks.sdk.core'

# Kept under the names this module has always used: its tests read them here.
_status_of = status_of
_error_code_of = error_code_of
_validate_endpoint_name = validate_endpoint_name


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
        records, size = envelope(request)
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
        try:
            return response_body(reply)
        except ValueError as error:
            raise HTTPException(502, str(error)) from None
