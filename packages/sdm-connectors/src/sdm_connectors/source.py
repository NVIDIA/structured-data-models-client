# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Where a schema's tables are, and how to reach them.

A :class:`Source` pairs a connection with the namespace its tables live in. It
reads nothing itself: it is what a schema over tables that stay where they are
is built from, so that sampling can run as SQL against the tables in place.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sdm_connectors.sql import ConnectorError

# The names each backend qualifies a table with, in the order it nests them.
_NAMESPACES: Mapping[str, tuple[str, ...]] = {
    'databricks': ('catalog', 'schema'),
    'snowflake': ('database', 'schema'),
    'postgres': ('schema',),
    'duckdb': (),
    'sqlite': (),
}

# Recognised by the module that defines the connection's class, or any of its
# bases, so a driver is never imported just to be ruled out.
_DRIVER_MODULES: tuple[tuple[str, str], ...] = (
    ('databricks.sql', 'databricks'),
    ('sdm_connectors.backends.spark', 'databricks'),
    ('snowflake.connector', 'snowflake'),
    ('psycopg', 'postgres'),
    ('duckdb', 'duckdb'),
    ('_duckdb', 'duckdb'),
    ('adbc_driver_sqlite', 'sqlite'),
)

_ACCEPTED = (
    'a Databricks SQL connection or a Spark session, a Snowflake connection '
    'or Snowpark session, a psycopg connection, a DuckDB connection, or an '
    'ADBC SQLite connection -- what sdm_connectors.connect returns'
)


def _modules(value: Any) -> list[str]:
    return [cls.__module__ or '' for cls in type(value).__mro__]


def _is_spark_session(value: Any) -> bool:
    return any(
        cls.__name__ == 'SparkSession'
        and (cls.__module__ or '').startswith('pyspark.')
        for cls in type(value).__mro__
    )


def _is_snowpark_session(value: Any) -> bool:
    return any(
        cls.__name__ == 'Session'
        and (cls.__module__ or '').startswith('snowflake.snowpark')
        for cls in type(value).__mro__
    )


def _backend_of(connection: Any) -> str | None:
    for module in _modules(connection):
        for prefix, backend in _DRIVER_MODULES:
            if module == prefix or module.startswith(prefix + '.'):
                return backend
    return None


class Source:
    r"""A connection, and the namespace its tables live in.

    >>> source = Source(spark, catalog='main', schema='sales')
    >>> source = Source(connect('databricks', ...), catalog='main', schema='sales')
    >>> source = Source(connect('snowflake', ...), database='SALES', schema='PUBLIC')
    >>> source = Source(connect('postgres', ...), schema='public')
    >>> source = Source(connect('duckdb', 'sales.duckdb'))

    A Databricks notebook's ``spark`` is accepted as it is and sampled through
    ``spark.sql``, so no SQL warehouse is needed; a Snowpark ``session`` is
    reached through its own connector connection.

    The source never closes the connection. It belongs to whoever opened it.

    Args:
        connection: An open connection, or a notebook session.
        catalog: The Databricks catalog. Defaults to the connection's current
            one.
        database: The Snowflake database. Defaults to the connection's current
            one.
        schema: The schema the tables are in. Defaults to the connection's
            current one.

    Raises:
        ConnectorError: ``UNKNOWN_CONNECTOR`` for a connection this package
            cannot sample through, ``INVALID_CONNECTOR_ARGS`` for a namespace
            the backend does not have.
    """

    def __init__(
        self,
        connection: Any,
        *,
        catalog: str | None = None,
        database: str | None = None,
        schema: str | None = None,
    ) -> None:
        if _is_spark_session(connection):
            from sdm_connectors.backends.spark import SparkConnection

            connection = SparkConnection(connection)
        elif _is_snowpark_session(connection):
            connection = connection.connection
        backend = _backend_of(connection)
        if backend is None:
            raise ConnectorError(
                f'a Source needs {_ACCEPTED}; got {type(connection).__name__}',
                code='UNKNOWN_CONNECTOR',
            )
        given = {
            name: value
            for name, value in (
                ('catalog', catalog),
                ('database', database),
                ('schema', schema),
            )
            if value is not None
        }
        allowed = _NAMESPACES[backend]
        foreign = sorted(set(given) - set(allowed))
        if foreign:
            has = (
                f'is qualified by {" and ".join(allowed)}'
                if allowed
                else 'has no namespace to name'
            )
            raise ConnectorError(
                f'a {backend} source {has}, so {", ".join(foreign)} '
                f'cannot be given',
                code='INVALID_CONNECTOR_ARGS',
            )
        for name, value in given.items():
            if not isinstance(value, str) or not value:
                raise ConnectorError(
                    f'{name} must be a non-empty string, got {value!r}',
                    code='INVALID_CONNECTOR_ARGS',
                )
        self._backend = backend
        self._connection = connection
        self._namespace = given

    @property
    def backend(self) -> str:
        r"""Which engine the tables are sampled with: ``'databricks'``,
        ``'snowflake'``, ``'postgres'``, ``'duckdb'`` or ``'sqlite'``.
        """
        return self._backend

    @property
    def connection(self) -> Any:
        r"""The connection sampling runs through."""
        return self._connection

    @property
    def namespace(self) -> dict[str, str]:
        r"""The namespace given, by name: ``catalog``, ``database``,
        ``schema``. Anything left out is the connection's current one.
        """
        return dict(self._namespace)

    def __reduce__(self) -> Any:
        raise TypeError(
            'a Source holds an open connection, which cannot be pickled'
        )

    def __repr__(self) -> str:
        where = ', '.join(f'{k}={v!r}' for k, v in self._namespace.items())
        return f'Source({self._backend}{", " if where else ""}{where})'
