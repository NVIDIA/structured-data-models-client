# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Invoke Kumo Relational through a Snowpark Container Services model service.

This is the peer of
:mod:`sdm_relational_engine.client.databricks_serving`, with the same contract:
one ``request_json`` row in and one ``response_json`` row out. Only the service
access method differs.

A Snowflake model service is invoked as a SQL method on the service,
``SELECT <service>!PREDICT(?)``, over a Snowpark session or a Snowflake
connection. This deliberately avoids a REST call to the service's ingress.
Ingress must be explicitly enabled, is available only to users of the account
that created it, and has a 90-second connection timeout. The SQL path works
from a Snowflake notebook with its existing session and externally with
ordinary warehouse credentials.

The request travels as a bind parameter, never interpolated into the statement.
A canonical RFM request is megabytes of sampled customer rows; splicing it into
SQL would put customer data in the query text, and query text is logged, shown
in history and truncated at a statement-size limit far below the payload cap.
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

__all__ = ['SnowflakeServingClient']

logger = logging.getLogger('sdm_relational_engine')

# The frozen v1 boundary; see CONTRACT_V1.md in the serving repository.
REQUEST_COLUMN = 'request_json'
RESPONSE_COLUMN = 'response_json'

_PREDICTION_PATH = TFMOperations.run_prediction.endpoint.get_path()

# Snowflake caps a bound string at 16 MB, which is what the payload travels as.
MAX_REQUEST_BYTES = 16 * 1000 * 1000

DEFAULT_TIMEOUT_SECONDS = 300.0

_SERVICE_NAME_RE = re.compile(
    r'^[A-Za-z_][A-Za-z0-9_$]*'
    r'(\.[A-Za-z_][A-Za-z0-9_$]*){0,2}$'
)


def _validate_service_name(name: str) -> str:
    """Reject anything that is not a bare, optionally-qualified service name.

    The service name is the one part of the statement that cannot be bound, so
    it is validated against a whitelist rather than escaped. Each part is an
    ordinary unquoted identifier, so whitespace, quotes, semicolons and a URL
    are all rejected before reaching the statement. The rejection never echoes
    the value: the likeliest wrong input is a pasted connection string or
    account URL, which can carry credentials.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError('service must be a non-empty Snowflake service name')
    if not _SERVICE_NAME_RE.match(name):
        raise ValueError(
            'service must be a Snowflake service name, optionally qualified as '
            'DATABASE.SCHEMA.SERVICE'
        )
    return name


def _status_of(error: BaseException) -> int:
    """Map a Snowflake driver error onto an HTTP-equivalent status.

    Matched by class name and message so ``snowflake.connector`` need not be
    importable to read this module, and a driver version that moves its classes
    cannot break the mapping silently. An unclassified failure is 503 rather
    than 500, because it is usually the service suspended or still scaling, and
    503 is what RFM retries.
    """
    names = {cls.__name__ for cls in type(error).__mro__}
    if 'DatabaseError' in names and 'ProgrammingError' not in names:
        return 503
    message = str(error).lower()
    if 'does not exist' in message or 'not authorized' in message:
        return 404 if 'does not exist' in message else 403
    if 'timeout' in message or 'timed out' in message:
        return 504
    return 503


class SnowflakeServingClient:
    """Address a Kumo Relational model served on Snowpark Container Services.

    Args:
        service: The service name, optionally qualified as
            ``DATABASE.SCHEMA.SERVICE``. A name, not a URL.
        session: An existing Snowpark ``Session`` or a ``snowflake.connector``
            connection. When omitted, the active Snowpark session is used,
            which is how a Snowflake notebook connects without handling
            credentials. Tested against ``None`` rather than truthiness,
            because some Snowpark ``Session`` versions define ``__bool__`` and
            a falsy one would silently fall back to the ambient session.
        method: The service method to call. ``PREDICT`` is the inference method
            the model registry generates for a ``CustomModel``.
        max_request_bytes: Client-side cap, checked before the statement is
            issued. Rejected if boolean: ``True`` is an ``int`` and would
            reject every request.
        timeout: Statement timeout in seconds, applied per request. Rejected if
            boolean, for the same reason. A cold service has to start a
            container, so this is far above a warehouse default.
    """

    def __init__(
        self,
        service: str,
        session: Any | None = None,
        *,
        method: str = 'PREDICT',
        max_request_bytes: int = MAX_REQUEST_BYTES,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._service = _validate_service_name(service)
        self._method = _validate_service_name(method)
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
        if session is not None:
            self._session = session
        else:
            self._session = self._active_session()

    @staticmethod
    def _active_session() -> Any:
        """The ambient session, for code already running inside Snowflake.

        The failure is reported as 503 without the provider's own message,
        which can name the account and the local connection file.
        """
        try:
            from snowflake.snowpark.context import get_active_session
        except ImportError as error:
            raise ImportError(
                'Snowflake serving support requires the '
                "'snowflake-serving' extra: pip install "
                "'sdm_relational_engine[snowflake-serving]'"
            ) from error
        try:
            return get_active_session()
        except Exception:
            raise HTTPException(
                503,
                'no active Snowflake session; pass session= explicitly when '
                'not running inside Snowflake',
            ) from None

    @property
    def service(self) -> str:
        return self._service

    def close(self) -> None:
        """Nothing to release: the session outlives this client and may be the
        notebook's own.
        """

    def _request(self, endpoint: Endpoint, **kwargs: Any) -> ServingResponse:
        path = endpoint.get_path()
        if path != _PREDICTION_PATH:
            raise HTTPException(
                501,
                'Snowflake model serving supports one-shot prediction only; '
                f'{path!r} is not available in this mode',
            )
        request = kwargs.get('json')
        if not isinstance(request, Mapping):
            raise HTTPException(400, 'prediction request must be a JSON object')
        return self._query(request)

    def _query(self, request: Mapping[str, Any]) -> ServingResponse:
        """Send one canonical request and return the serving reply.

        The cap is measured on the payload itself, which is what is bound and
        therefore what is sent: unlike an HTTP envelope there is no re-escaping
        on the way out. The comparison is ``>`` because the cap is the largest
        size the binder accepts. ``allow_nan=False`` because NaN and Infinity
        are not JSON, and Snowflake would reject them after the round trip.
        """
        payload = json.dumps(request, allow_nan=False, separators=(',', ':'))
        size = len(payload.encode('utf-8'))
        if size > self._max_request_bytes:
            raise HTTPException(
                413,
                f'request payload is {size} bytes, above the '
                f'{self._max_request_bytes} byte limit',
            )

        try:
            rows = self._execute(payload)
        except Exception as error:
            status = _status_of(error)
            logger.debug(
                'snowflake service %r failed with %s',
                self._service,
                type(error).__name__,
                exc_info=True,
            )
            raise HTTPException(
                status,
                f'snowflake service {self._service!r} request failed',
            ) from None

        return ServingResponse(200, self._body_of(rows))

    def _statement(self, placeholder: str) -> str:
        return (
            f'SELECT {self._service}!{self._method}({placeholder}) '
            f'AS {RESPONSE_COLUMN}'
        )

    @staticmethod
    def _connector_placeholder() -> str:
        """The bind marker ``snowflake.connector`` is currently configured for.

        The two clients disagree. Snowpark binds with ``?``; the connector
        binds with whatever ``snowflake.connector.paramstyle`` says, and that
        defaults to ``pyformat``, meaning ``%s``. Passing ``?`` under
        ``pyformat`` does not raise a binding error: the connector falls
        through to string formatting and reports "not all arguments converted
        during string formatting", which names neither the placeholder nor the
        paramstyle.

        Read at call time rather than at import, because it is a module-level
        global an application is free to set after this module is imported.
        """
        try:
            from snowflake.connector import paramstyle
        except ImportError:
            return '%s'
        return '?' if paramstyle == 'qmark' else '%s'

    def _raise_timeout(self, run: Any) -> None:
        """Give the request longer than the warehouse default, if allowed.

        Best effort, because a stored procedure may not run ``ALTER SESSION``:
        Snowflake rejects it with "Unsupported statement type", which is a
        property of where this is running rather than anything about the
        request. Raising it is a convenience for a cold service that has to
        start a container; failing to raise it is not a reason to give up the
        prediction, and inside a procedure the caller's own timeout applies
        anyway.
        """
        try:
            run(
                'ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = '
                f'{int(self._timeout)}'
            )
        except Exception:
            logger.debug('could not raise the statement timeout', exc_info=True)

    def _execute(self, payload: str) -> Any:
        """Send the payload over a Snowpark session or a raw connection.

        Both are accepted because a notebook holds a Snowpark ``Session`` while
        a script more often holds a ``snowflake.connector`` connection, and
        requiring one to be converted to the other is friction with no payoff.
        """
        if hasattr(self._session, 'sql'):
            self._raise_timeout(lambda sql: self._session.sql(sql).collect())
            return self._session.sql(
                self._statement('?'), params=[payload]
            ).collect()
        with self._session.cursor() as cursor:
            self._raise_timeout(cursor.execute)
            cursor.execute(
                self._statement(self._connector_placeholder()), (payload,)
            )
            return cursor.fetchall()

    def _body_of(self, rows: Any) -> str:
        """Extract the single canonical response the contract promises."""
        if not isinstance(rows, list) or len(rows) != 1:
            raise HTTPException(
                502,
                'snowflake service did not return exactly one prediction row',
            )
        row = rows[0]
        body = None
        if isinstance(row, Mapping):
            body = row.get(RESPONSE_COLUMN) or row.get(RESPONSE_COLUMN.upper())
        elif hasattr(row, 'as_dict'):
            values = row.as_dict()
            body = values.get(RESPONSE_COLUMN) or values.get(
                RESPONSE_COLUMN.upper()
            )
        elif isinstance(row, (list, tuple)) and row:
            body = row[0]
        if not isinstance(body, str):
            raise HTTPException(
                502,
                f'snowflake service response row has no {RESPONSE_COLUMN!r} '
                'string column',
            )
        return self._unwrap(body)

    @staticmethod
    def _unwrap(body: str) -> str:
        """Return the canonical response, which arrives nested one level.

        A service method returns the model's output *row* as a JSON object
        keyed by column name, so a one-column frame comes back as
        ``{"response_json": "{...}"}`` rather than as the string itself. The
        SQL column and the model's column have the same name, so the nesting is
        invisible until something reads the result: it surfaces as the parser
        reporting a missing ``id``. This reads like a model contract violation,
        not an envelope around the response.

        Unwrapped only when the payload really is that envelope. A canonical
        response is also a JSON object, but it has no ``response_json`` member,
        so it is returned untouched.
        """
        if not body.lstrip().startswith('{'):
            return body
        try:
            parsed = json.loads(body)
        except ValueError:
            return body
        if isinstance(parsed, dict):
            nested = parsed.get(RESPONSE_COLUMN)
            if isinstance(nested, str):
                return nested
        return body
