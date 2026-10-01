# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

from __future__ import annotations

import contextlib
import importlib
import weakref
from typing import Any

from sdm_connectors.sql import ConnectorError, driver_guard

_SQL_BACKENDS = ('sqlite', 'duckdb', 'postgres', 'snowflake', 'databricks')

OWNED_ATTR = '_sdm_connectors_owned'

_ownership: weakref.WeakKeyDictionary[Any, bool] = weakref.WeakKeyDictionary()


def connect(backend: str, *args: Any, **kwargs: Any) -> Any:
    r"""Opens a connection to one of the supported SQL backends.

    Args:
        backend: One of ``'sqlite'``, ``'duckdb'``, ``'postgres'``,
            ``'snowflake'`` or ``'databricks'``.
        *args: Positional arguments forwarded to the backend's driver, e.g. the
            database path for ``sqlite`` / ``duckdb``.
        **kwargs: Keyword arguments forwarded to the backend's driver.

    Returns:
        An open DB-API style connection owned by the caller.

    Raises:
        ConnectorError: ``UNKNOWN_CONNECTOR`` for an unsupported ``backend``,
            ``CONNECT_FAILED`` if the driver refuses the connection.
        MissingBackendError: If the backend's optional driver is not installed.
        ImportError: If the driver is installed but broken.
    """
    if backend not in _SQL_BACKENDS:
        raise ConnectorError(
            f'unknown backend {backend!r}; supported: {list(_SQL_BACKENDS)}',
            code='UNKNOWN_CONNECTOR',
        )
    module = importlib.import_module(f'{__name__}.{backend}')
    with driver_guard('CONNECT_FAILED', f'failed to connect to {backend!r}'):
        return module.connect(*args, **kwargs)


def mark_owned(connection: Any, owned: bool) -> None:
    try:
        _ownership[connection] = owned
        return
    except TypeError:
        pass
    with contextlib.suppress(AttributeError, TypeError):
        setattr(connection, OWNED_ATTR, owned)


def owns_connection(connection: Any) -> bool:
    r"""Whether we opened this connection, vs. borrowed an active session.

    Borrowed connections (e.g. an active Snowpark session) must not be closed
    by the reader, since the caller still owns them. Ownership is tracked
    out-of-band so a borrowed connection that rejects attribute writes is never
    misread as owned.
    """
    try:
        if connection in _ownership:
            return _ownership[connection]
    except TypeError:
        pass
    return getattr(connection, OWNED_ATTR, True)
