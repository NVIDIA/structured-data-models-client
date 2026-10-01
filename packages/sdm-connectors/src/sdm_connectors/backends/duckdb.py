# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

from sdm_connectors.sql import (
    ConnectorError,
    connect_with,
    require_driver,
    require_existing_database,
)

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
