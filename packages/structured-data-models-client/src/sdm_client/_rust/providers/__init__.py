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

from sdm_client._rust.providers.duckdb import DuckDBProvider
from sdm_client._rust.providers.postgres import PostgresProvider
from sdm_client._rust.providers.sqlite import SqliteProvider


def for_source(source: Any) -> tuple[dict[str, Any], Any]:
    r"""The provider choice a run over ``source`` names, and the provider
    answering its fetches: its backend's built-in SQL dialect, DuckDB's,
    SQLite's (T2.2) or Postgres's (T2.3), on its connection.

    The choice carries the source's namespace (contract v28), the schema a
    ``PostgresSource`` was given, under which the core reads each table and
    lists the source's tables, as the reference qualifies each read by it;
    a source given none is read in the connection's current schema.
    """
    backend = source.backend
    made = {
        'duckdb': DuckDBProvider,
        'sqlite': SqliteProvider,
        'postgres': PostgresProvider,
    }[backend]
    choice: dict[str, Any] = {'provider': 'sql', 'dialect': backend}
    named = getattr(source, 'namespace', None) or {}
    schema = named.get('schema')
    if schema is not None:
        choice['namespace'] = [schema]
    return choice, made(source.connection)
