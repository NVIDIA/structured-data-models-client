# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import http
from typing import Any


class KumoRelationalError(Exception):
    r"""Base class for the errors this package defines.

    ``except KumoRelationalError`` covers those, which is every failure
    that has a name of its own here: a connection that could not be made,
    a response that could not be read, a graph that could not be built.
    It is not a catch-all for the package. An argument that is simply
    wrong still raises the built-in that fits it, and a column this table
    does not carry still raises :class:`KeyError`.

    Each subclass also inherits the built-in that suits it
    (:class:`ValueError`, :class:`RuntimeError`), so either catch works.
    """


class ClientInitializationError(KumoRelationalError, ValueError):
    r"""The client could not be pointed at a usable NIM.

    Raised while establishing the connection rather than while serving a
    prediction, so the request itself was never sent. The subclasses separate
    the cases worth reacting to differently: bad credentials are not worth a
    retry, an unreachable or slow server usually is.
    """


class AuthenticationError(ClientInitializationError):
    r"""The NIM (or the gateway in front of it) rejected the credentials.

    NIMs are unauthenticated by contract, so this only happens behind an
    authenticating gateway, and means the ``api_key`` is missing or wrong.
    """


class NimUnreachableError(ClientInitializationError):
    r"""No connection to the NIM could be established at all."""


class NimTimeoutError(ClientInitializationError):
    r"""The NIM did not answer the connection attempt in time."""


class UnknownDatasetError(KumoRelationalError, ValueError):
    r"""A named dataset does not exist in the RelBench registry."""


class GraphConstructionError(KumoRelationalError):
    r"""A data source could not be read while building a graph.

    The driver's own exception is kept as ``__cause__``. Without this, reading
    a warehouse raised whatever that driver raises -- a
    ``databricks.sql.exc.ServerOperationError``, a
    ``snowflake.connector.errors.ProgrammingError`` -- none of which share a
    base with anything else this client raises, so building a graph could not be
    guarded by the same ``except`` as using one.
    """


class InvalidResponseError(KumoRelationalError, ValueError):
    r"""The NIM answered, but its response does not match the contract.

    Subclasses :class:`ValueError` so existing callers keep working, while
    letting the client report a malformed *server* response as such instead of
    blaming the caller's request. It also stops a body that will never parse
    from being retried as though the NIM were merely busy.
    """


class NimFailureError(KumoRelationalError, RuntimeError):
    r"""A failed KumoRelational NIM prediction, classified and ready to re-wrap.

    Subclasses :class:`RuntimeError` so existing callers keep working, and
    carries the structured facts a caller (or the ``sdm-client`` adapter) needs
    to map it onto its own error hierarchy without re-parsing the message:
    the HTTP status, the problem document's ``detail``, its RFC-9457
    ``invalid_params`` entries, and whether the failure looks transient.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        detail: str | None = None,
        invalid_params: list[dict[str, Any]] | None = None,
        transient: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail
        self.invalid_params = invalid_params or []
        self.transient = transient


class HTTPException(KumoRelationalError):
    r"""An HTTP exception, with detailed information and headers."""

    def __init__(
        self,
        status_code: int,
        detail: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        # Derived from starlette/blob/master/starlette/exceptions.py
        if detail is None:
            detail = http.HTTPStatus(status_code).phrase
        self.status_code = status_code
        self.detail = detail
        self.headers = headers

    def __str__(self) -> str:
        return f'{self.status_code}: {self.detail}'

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f'{class_name}(status_code={self.status_code!r}, '
            f'detail={self.detail!r})'
        )
