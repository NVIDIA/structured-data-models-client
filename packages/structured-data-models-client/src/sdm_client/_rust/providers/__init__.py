# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The Python host's data providers: each answers the fetch steps of a core
run, one request at a time, with an object that exports Arrow through
``__arrow_c_stream__``.

A provider reads no pandas and converts nothing: the driver's Arrow, or,
from a driver that gives rows (psycopg), the Arrow pyarrow builds of its
values, is what crosses to the core, where admission and the schema checks
hold it (design, "Core, host and binding"). A test holds every module here
to that, by its imports and by running one with pandas unimportable.
"""

from __future__ import annotations

from typing import Any

from sdm_client._rust.providers.databricks import (
    DatabricksSparkProvider,
    DatabricksSqlProvider,
)
from sdm_client._rust.providers.duckdb import DuckDBProvider
from sdm_client._rust.providers.postgres import PostgresProvider
from sdm_client._rust.providers.sqlite import SqliteProvider

__all__ = [
    'DatabricksSparkProvider',
    'DatabricksSqlProvider',
    'DuckDBProvider',
    'PostgresProvider',
    'SqliteProvider',
    'for_source',
]

#: Each kind of source the core reads (``Source.kind``): the built-in SQL
#: dialect it renders for it, and the name of the provider class answering
#: on its connection, read when the run starts.
_BY_KIND: dict[str, tuple[str, str]] = {
    'duckdb': ('duckdb', 'DuckDBProvider'),
    'sqlite': ('sqlite', 'SqliteProvider'),
    'postgres': ('postgres', 'PostgresProvider'),
    'databricks-sql': ('databricks', 'DatabricksSqlProvider'),
    'databricks-spark': ('databricks', 'DatabricksSparkProvider'),
}


def for_source(source: Any) -> tuple[dict[str, Any], Any]:
    r"""The provider choice a run over ``source`` names, and the provider
    answering its fetches, by the source's kind (``Source.kind``, design fix
    F): its built-in SQL dialect, DuckDB's, SQLite's (T2.2), Postgres's
    (T2.3) or Databricks's (T3.4, a SQL warehouse or a Spark session), on
    its connection.

    The choice carries the source's namespace (contract v28,
    ``Source.namespace_parts()``, design fix C): the schema a
    ``PostgresSource`` was given, the catalog and schema of a
    ``DatabricksSource``, each ``None`` where the connection's current one
    is meant, under which the core reads each table and lists the source's
    tables, as the reference qualifies each read by them; left out where
    the source names none.
    """
    dialect, name = _BY_KIND[source.kind]
    made = globals()[name]
    choice: dict[str, Any] = {'provider': 'sql', 'dialect': dialect}
    parts = source.namespace_parts()
    if any(part is not None for part in parts):
        choice['namespace'] = parts
    return choice, made(source.connection)
