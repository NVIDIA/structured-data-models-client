# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

from sdm_connectors.sql import (
    ConnectorError,
    connect_with,
    require_driver,
    require_existing_database,
)

adbc = require_driver(
    'sqlite',
    'adbc-driver-sqlite',
    'adbc_driver_sqlite.dbapi',
)

Connection: TypeAlias = adbc.AdbcSqliteConnection


def connect(
    uri: str | Path | None = None,
    *,
    database: str | Path | None = None,
    **kwargs: Any,
) -> Connection:
    r"""Open a sqlite database. ``database`` is an alias for ``uri``.

    sqlite opens missing files in create-if-missing mode, so a mistyped path
    would leave a new empty database on disk and then report the *table* as
    missing. A read must not write, so a filesystem path that does not exist is
    reported as ``NOT_FOUND`` before the driver sees it. ``:memory:`` and
    ``file:`` URIs (which carry their own mode) are passed through untouched.
    """
    if uri is not None and database is not None:
        raise ConnectorError(
            "sqlite connector accepts exactly one of 'database' or 'uri', "
            'not both',
            code='INVALID_CONNECTOR_ARGS',
        )
    uri = uri if uri is not None else database
    if uri is not None:
        uri = str(uri)
        require_existing_database('sqlite', uri)
    return connect_with('sqlite', adbc.connect, uri, **kwargs)
