# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from sdm_connectors.sql import (
    ConnectorError,
    connect_with,
    require_driver,
    require_existing_database,
)

# A type checker reads the driver's own types; at runtime its absence is
# reported as MissingBackendError rather than an ImportError.
if TYPE_CHECKING:
    import duckdb
    import pyarrow as pa
else:
    duckdb = require_driver('duckdb', 'duckdb', 'duckdb')

Connection: TypeAlias = duckdb.DuckDBPyConnection


def connect(
    uri: str | Path | None = None,
    *,
    database: str | Path | None = None,
    **kwargs: Any,
) -> Connection:
    r"""Open a DuckDB database. ``database`` is an alias for ``uri``.

    DuckDB opens a missing path in create-if-missing mode, so a mistyped path
    is reported as ``NOT_FOUND`` before the driver can write an empty database
    into the caller's working tree. This matches the contract the sibling
    ``sqlite`` connector enforces for the same mistake and the one
    ``sdm_relational_engine.rfm.backend.duckdb.connect`` enforces for the
    separate ``adbc_driver_duckdb`` entry point the sampler needs.
    """
    if uri is not None and database is not None:
        raise ConnectorError(
            "duckdb connector accepts exactly one of 'database' or 'uri', "
            'not both',
            code='INVALID_CONNECTOR_ARGS',
        )
    uri = uri if uri is not None else database
    if uri is None:
        return connect_with('duckdb', duckdb.connect, ':memory:', **kwargs)
    uri = str(uri)
    require_existing_database('duckdb', uri)
    return connect_with('duckdb', duckdb.connect, uri, **kwargs)


def run(
    connection: Connection, sql: str, params: Sequence[Any] | None = None
) -> pa.Table:
    r"""Run one statement and return what it reads.

    ``?`` markers are bound positionally from ``params``. The statement runs
    on a cursor, DuckDB's own connection to the same database, so it sees the
    database's tables but not the caller's temporary objects or registered
    frames, and leaves the caller's connection as it was.
    """
    with connection.cursor() as cursor:
        cursor.execute(sql, None if params is None else list(params))
        return cursor.to_arrow_table()
