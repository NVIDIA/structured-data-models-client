# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import difflib
import importlib
import pathlib
import re
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from typing import Any

_TABLE_IDENTIFIER_RE = re.compile(
    r'^[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*)*$',
)


class ConnectorError(Exception):
    r"""Base class for every error this package raises.

    ``except ConnectorError`` is the one catch that covers the whole connector
    layer, including a missing optional driver. Each instance carries a stable
    ``code`` so a caller can branch without matching on message text.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class MissingBackendError(ConnectorError, ModuleNotFoundError):
    r"""Raised when a backend's driver package is not installed.

    Distinct from a plain ``ImportError``, which indicates an installed but
    broken driver (e.g. a native ABI mismatch) and must not be reported as an
    absent optional dependency. Subclasses :class:`ModuleNotFoundError` as well
    as :class:`ConnectorError` so callers written against either keep working.
    """

    def __init__(self, backend: str, driver: str) -> None:
        super().__init__(
            f'The {backend!r} backend requires {driver!r}. Install it via the '
            f'{backend!r} extra, e.g. `pip install '
            f"'sdm-connectors[{backend}]'`.",
            code='MISSING_BACKEND',
            details={'backend': backend, 'driver': driver},
        )
        self.backend = backend
        self.driver = driver


_REJECTED_KEYWORD_MESSAGES = (
    'unexpected keyword argument',
    'incompatible function arguments',
)


def connect_with(backend: str, opener: Any, *args: Any, **kwargs: Any) -> Any:
    r"""Open a connection, reporting a rejected keyword against the connector.

    A file-backed backend takes its path as ``database`` or ``uri`` and passes
    anything else through to the driver, so a caller reaching for ``path`` --
    the name :func:`read` uses for a local file -- got the driver's own
    ``Connection.__init__() got an unexpected keyword argument 'path'``, which
    names neither the connector nor the two arguments it does accept.

    Both phrasings are matched: a driver written in Python reports the rejected
    keyword the way CPython does, while a compiled one built with pybind11 --
    DuckDB's, for instance -- reports "incompatible function arguments" and
    dumps its whole overload set instead.
    """
    try:
        return opener(*args, **kwargs)
    except TypeError as error:
        if not any(text in str(error) for text in _REJECTED_KEYWORD_MESSAGES):
            raise
        raise ConnectorError(
            f'{backend} connector got an unsupported argument: {error}. It '
            f"takes the database as 'database' or 'uri'; other arguments are "
            f'passed to the driver. To read a CSV or Parquet file, use '
            f'read(path) instead.',
            code='INVALID_CONNECTOR_ARGS',
            details={'backend': backend, 'arguments': sorted(kwargs)},
        ) from error


def require_driver(
    backend: str,
    driver: str,
    module: str,
    absent: tuple[str, ...] | None = None,
) -> Any:
    r"""Import a backend's driver module, or raise MissingBackendError.

    ``absent`` lists the module names whose ModuleNotFoundError means the
    optional driver is not installed; any other missing module signals a
    broken installation and propagates unchanged.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        if error.name not in (absent or (module.partition('.')[0],)):
            raise
        raise MissingBackendError(backend, driver) from error


@contextmanager
def driver_guard(
    code: str,
    context: str,
    *,
    missing_as_not_found: bool = False,
    **details: Any,
) -> Iterator[None]:
    r"""Translate driver exceptions into ConnectorError with a stable code.

    ConnectorError and MissingBackendError pass through untouched so guarded
    regions never double-wrap, and ImportError propagates unchanged because a
    broken driver installation must not be reported as a connect/query/read
    failure. The original driver exception stays chained as ``__cause__``.
    TypeError maps to INVALID_CONNECTOR_ARGS (a deterministic caller error,
    never worth retrying); FileNotFoundError maps to NOT_FOUND only where the
    guarded region reads data by path (``missing_as_not_found=True``).
    """
    try:
        yield
    except (ConnectorError, ImportError):
        raise
    except Exception as error:
        if isinstance(error, TypeError):
            mapped = 'INVALID_CONNECTOR_ARGS'
        elif missing_as_not_found and isinstance(error, FileNotFoundError):
            mapped = 'NOT_FOUND'
        else:
            mapped = code
        raise ConnectorError(
            f'{context}: {str(error) or type(error).__name__}',
            code=mapped,
            details={'driver_error': type(error).__name__, **details},
        ) from error


def require_existing_database(backend: str, uri: str) -> None:
    r"""Reject a database path that does not exist yet.

    The file-backed drivers open a missing path in create-if-missing mode, so
    a mistyped path leaves a new empty database on disk and the failure then
    surfaces as a missing *table* -- sending the caller to debug the wrong
    problem, having silently written to their working tree. A read must not
    write. ``:memory:`` and URI forms that carry their own open mode are
    passed through untouched.

    Shared by every file-backed backend so the two cannot drift apart on what
    the same mistake does.

    Args:
        backend: The connector name, for the error message.
        uri: The database path supplied by the caller.
    """
    if uri == ':memory:' or uri.startswith(('file:', ':')):
        return
    if not pathlib.Path(uri).exists():
        raise ConnectorError(
            f'{backend} database file {uri!r} does not exist',
            code='NOT_FOUND',
            details={'database': uri},
        )


def check_connect_args(
    backend: str,
    kwargs: dict[str, Any],
    allowed: Collection[str],
) -> None:
    r"""Reject connection keywords the driver would silently ignore.

    The Snowflake and Databricks drivers accept ``**kwargs`` without complaint,
    so a misspelled parameter is otherwise dropped and the read runs against
    whatever catalog/schema the connection defaults to: a wrong answer rather
    than an error. ``allowed`` is read off the driver itself so it stays
    accurate across driver upgrades, and ``driver_options`` is the escape hatch
    for anything genuinely outside it.
    """
    unknown = sorted(set(kwargs) - set(allowed))
    if not unknown:
        return
    named = []
    for name in unknown:
        close = difflib.get_close_matches(name, allowed, n=1)
        named.append(
            f'{name!r} (did you mean {close[0]!r}?)' if close else repr(name)
        )
    raise ConnectorError(
        f'{backend} connector got unexpected arguments {", ".join(named)}; '
        "pass driver-specific options through 'driver_options'",
        code='INVALID_CONNECTOR_ARGS',
        details={'arguments': unknown},
    )


def merge_driver_options(
    backend: str,
    kwargs: dict[str, Any],
    driver_options: dict[str, Any] | None,
) -> dict[str, Any]:
    r"""Fold the ``driver_options`` escape hatch into the connection arguments.

    ``driver_options`` deliberately bypasses the allow-list, so it must not be
    able to quietly restate an argument that was already validated: naming a
    key both ways is rejected rather than letting one silently win, matching
    every other mutually-exclusive pair in this package. Merging before the
    backend inspects the result also means an option supplied through the
    escape hatch is never silently dropped from the connection decision.
    """
    options = dict(driver_options or {})
    if not options:
        return kwargs
    overlap = sorted(set(options) & set(kwargs))
    if overlap:
        raise ConnectorError(
            f'{backend} connector got {overlap} both as a connection argument '
            f"and in 'driver_options'; pass each one once",
            code='INVALID_CONNECTOR_ARGS',
            details={'arguments': overlap},
        )
    return {**kwargs, **options}


def quote_ident(ident: str, char: str = '"') -> str:
    r"""Quotes a SQL identifier, doubling any embedded quote character.

    Doubling is sufficient for an *identifier* -- no supported backend lets a
    backslash escape out of ``"``, ``` ` ``` or ``[]`` quoting. It is **not**
    sufficient for a string literal on a backend that also honours backslash
    escapes: Snowflake and Databricks both read ``\'`` as an escaped quote, so
    ``char="'"`` leaves a value containing ``\'`` able to close the literal and
    run the rest as SQL. Bind such a value as a query parameter instead. The
    literal mode remains only for backends whose single-quoted strings are
    standard-conforming (DuckDB, SQLite), where a backslash is an ordinary
    character.

    Args:
        ident: The identifier to quote. A single component -- a dotted name
            must be split and quoted per part.
        char: The quote character, e.g. ``'"'`` (default) or ``"'"`` for a
            string literal on a standard-conforming backend.

    Returns:
        The quoted identifier, e.g. ``'"my table"'``.
    """
    return char + ident.replace(char, char + char) + char


def resolve_sql(*, table: str | None, query: str | None) -> str:
    r"""Turns a ``table=`` or ``query=`` argument pair into one SQL statement.

    Exactly one must be given. ``query`` is returned verbatim -- the caller owns
    its safety. ``table`` is checked against a strict identifier pattern and
    interpolated unquoted, which also means it is subject to each backend's
    default case folding (Snowflake upper-cases, Databricks lower-cases). Names
    that need quoting -- spaces, non-ASCII characters, a leading digit -- are
    rejected; reach them with a hand-written ``query`` using
    :func:`quote_ident`.

    A reserved word such as ``select`` is a plain identifier by this pattern, so
    it is *not* rejected here; it reaches the driver unquoted and fails at
    execution as ``QUERY_FAILED``. Quoting is dialect-specific, so this
    layer -- which has no backend context -- does not attempt it.

    Args:
        table: A dot-separated identifier matching
            ``[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*)*``.
        query: A complete SQL statement.

    Returns:
        The SQL to execute.

    Raises:
        ConnectorError: ``INVALID_CONNECTOR_ARGS`` if both or neither argument
            is given, or if ``table`` is not a plain identifier.
    """
    if (table is None) == (query is None):
        raise ConnectorError(
            "exactly one of 'table' or 'query' must be provided",
            code='INVALID_CONNECTOR_ARGS',
        )
    if query is not None:
        return query
    if not _TABLE_IDENTIFIER_RE.fullmatch(table):
        raise ConnectorError(
            f'invalid table identifier {table!r}: only plain, unquoted, '
            f'dot-separated identifiers are accepted here. A name that needs '
            f'quoting (spaces, non-ASCII characters, a leading digit) has to '
            f'go through query=, e.g. '
            f'query=f"SELECT * FROM {{quote_ident(name)}}" -- quoting it '
            f'also changes how the backend folds its case.',
            code='INVALID_CONNECTOR_ARGS',
            details={'table': table},
        )
    return f'SELECT * FROM {table}'
