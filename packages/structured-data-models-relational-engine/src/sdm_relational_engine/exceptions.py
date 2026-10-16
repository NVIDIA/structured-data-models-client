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

    Where a built-in fits a subclass it inherits that too
    (:class:`ValueError`, :class:`RuntimeError`), so either catch works
    for those. Not every subclass has one.
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


class TableNotFoundError(KumoRelationalError, ValueError):
    r"""A table named for a graph is not in its source."""


class EmptyTableError(KumoRelationalError, RuntimeError):
    r"""A table named for a graph holds no rows.

    A :class:`RuntimeError` as well, which an empty table was raised as
    before.
    """


class MetricViewAsTableError(KumoRelationalError, ValueError):
    r"""A Databricks metric view was named where a table was expected.

    A metric view's measures are aggregations with no value per row, so
    Databricks refuses to read its rows as a table's. Raised before that
    read, from the catalog's own record of what the relation is.

    Attributes:
        name: The metric view, spelled as the caller named it.
    """

    def __init__(self, message: str, *, name: str) -> None:
        super().__init__(message)
        self.name = name


class ViewConversionError(KumoRelationalError, ValueError):
    r"""A metric or semantic view is refused as a graph, for what it is.

    Raised for the view itself, not for a failure reading it: a filter on a
    joined table, a query as its source, a metric view as its source, a
    definition that is no metric view's. The message says what to change in
    the view. A ``ValueError`` as well, which these refusals were raised as
    before.
    """


class NoEligibleEntityError(KumoRelationalError, ValueError):
    r"""No entity existed at the anchor, so nothing could be asked about.

    Separate from an empty sample, whose remedy is to widen the query. This
    one says the entity table's time column records something other than
    when each entity began.
    """


class NoTablesFoundError(KumoRelationalError, ValueError):
    r"""A source searched for its tables holds none."""


class GraphConstructionError(KumoRelationalError):
    r"""A data source could not be read while building a graph.

    The driver's own exception is kept as ``__cause__``. Without this, reading
    a warehouse raised whatever that driver raises, such as a
    ``databricks.sql.exc.ServerOperationError``, a
    ``snowflake.connector.errors.ProgrammingError``. These exceptions share no
    base with anything else this client raises, so building a graph could not
    be guarded by the same ``except`` as using one.
    """


class InvalidResponseError(KumoRelationalError, ValueError):
    r"""The NIM answered, but its response does not match the contract.

    Subclasses :class:`ValueError` so existing callers keep working, while
    letting the client report a malformed *server* response as such instead of
    blaming the caller's request. It also stops a body that will never parse
    from being retried as though the NIM were merely busy.
    """


class NimFailureError(KumoRelationalError, RuntimeError):
    r"""Represent a classified KumoRelational NIM prediction failure.

    Subclasses :class:`RuntimeError` so existing callers keep working, and
    carries the structured facts a caller (or the ``structured-data-models-client`` adapter) needs
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
