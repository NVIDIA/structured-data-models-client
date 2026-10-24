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
from sdm_client._rust.providers.snowflake import SnowflakeProvider
from sdm_client._rust.providers.sqlite import SqliteProvider

__all__ = [
    'DatabricksSparkProvider',
    'DatabricksSqlProvider',
    'DuckDBProvider',
    'PostgresProvider',
    'SnowflakeProvider',
    'SqliteProvider',
    'for_source',
    'view_choice',
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
    'snowflake': ('snowflake', 'SnowflakeProvider'),
}


def for_source(source: Any) -> tuple[dict[str, Any], Any]:
    r"""The provider choice a run over ``source`` names, and the provider
    answering its fetches, by the source's kind (``Source.kind``, design fix
    F): its built-in SQL dialect, DuckDB's, SQLite's (T2.2), Postgres's
    (T2.3) or Databricks's (T3.4, a SQL warehouse or a Spark session), on
    its connection, or Snowflake's (T3.6, a connector connection or a
    Snowpark session's own).

    The choice carries the source's namespace (contract v28,
    ``Source.namespace_parts()``, design fix C): the schema a
    ``PostgresSource`` was given, the catalog and schema of a
    ``DatabricksSource``, the database and schema of a ``SnowflakeSource``,
    each ``None`` where the connection's current one
    is meant, under which the core reads each table and lists the source's
    tables, as the reference qualifies each read by them; left out where
    the source names none.

    A semantic model (T3.5, design S6) is read through the source it is
    kept in, on that source's dialect and provider, and its choice names
    the model (``view``): its name as given and whether its expressions
    are trusted. The core reads the model afresh at every run, so the
    choice is all a schema built from it keeps of it.

    Raises:
        RouteError: A Snowflake semantic view, whose dialect and provider
            are T3.6's and not in this build.
    """
    model = _model_of(source)
    kept = source if model is None else model.source
    choice: dict[str, Any] = {'provider': 'sql'}
    if model is not None and kept.kind not in _BY_KIND:
        from sdm_client._routing import RouteError

        raise RouteError(
            f'{source!r} is kept in a {kept.kind} source, which the core '
            'reads from T3.6 on'
        )
    dialect, name = _BY_KIND[kept.kind]
    made = globals()[name]
    choice['dialect'] = dialect
    parts = kept.namespace_parts()
    if any(part is not None for part in parts):
        choice['namespace'] = parts
    if model is not None:
        choice['view'] = view_choice(model)
    return choice, made(kept.connection)


def view_choice(model: Any) -> dict[str, Any]:
    r"""The contract's ``ViewChoice`` of the semantic model ``model``: its
    name as the caller gave it, which the core resolves against the
    namespace, and whether its expressions are trusted."""
    return {
        'name': model.name,
        'trust_expressions': bool(model.trust_expressions),
    }


#: The kinds of a semantic model (``SemanticModel.kind``), each read
#: through the source it is kept in (``SemanticModel.source``).
_MODEL_KINDS = frozenset({'databricks-metric-view', 'snowflake-semantic-view'})


def _model_of(source: Any) -> Any:
    r"""``source`` if it is a semantic model, else ``None``."""
    return source if source.kind in _MODEL_KINDS else None
