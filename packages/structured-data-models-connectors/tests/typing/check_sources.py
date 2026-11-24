# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What a type checker makes of a source over a connection these
connectors open, checked by mypy rather than run.

Named without ``test_`` so that pytest leaves it alone. Each ``assert_type`` is
an inference that must not change, and each ``type: ignore`` a call that must
stay refused: the repository warns on an unused ignore, so a refused call that
starts to type-check fails the check.
"""

from __future__ import annotations

import sqlite3
from typing import Any, assert_type

import adbc_driver_sqlite.dbapi
import databricks.sql.client
import duckdb
import psycopg
import pyspark.sql
import pyspark.sql.connect.session
import snowflake.connector
import snowflake.snowpark
from databricks.sdk import WorkspaceClient
from sdm_client import (
    DatabricksSource,
    DuckDBSource,
    PostgresSource,
    SnowflakeSource,
    Source,
    SQLiteSource,
)

from sdm_connectors.backends.spark import SparkConnection
from sdm_connectors.databricks import lakebase, warehouse


def each_class_takes_its_driver_and_keeps_its_type(
    warehouse: databricks.sql.client.Connection,
    spark: pyspark.sql.SparkSession,
    spark_connect: pyspark.sql.connect.session.SparkSession,
    snowflake_connection: snowflake.connector.SnowflakeConnection,
    snowpark: snowflake.snowpark.Session,
    postgres: psycopg.Connection[Any],
    duckdb_connection: duckdb.DuckDBPyConnection,
    sqlite: adbc_driver_sqlite.dbapi.AdbcSqliteConnection,
) -> None:
    databricks_connection = databricks.sql.client.Connection | SparkConnection
    assert_type(
        DatabricksSource(warehouse, catalog='main').connection,
        databricks_connection,
    )
    assert_type(DatabricksSource(spark).connection, databricks_connection)
    assert_type(
        DatabricksSource(spark_connect).connection, databricks_connection
    )
    assert_type(
        SnowflakeSource(snowflake_connection, schema='PUBLIC').connection,
        snowflake.connector.SnowflakeConnection,
    )
    assert_type(
        SnowflakeSource(snowpark).connection,
        snowflake.connector.SnowflakeConnection,
    )
    assert_type(
        PostgresSource(postgres, schema='public').connection,
        psycopg.Connection[Any],
    )
    assert_type(
        DuckDBSource(duckdb_connection).connection, duckdb.DuckDBPyConnection
    )
    assert_type(
        SQLiteSource(sqlite).connection,
        adbc_driver_sqlite.dbapi.AdbcSqliteConnection,
    )
    assert_type(
        DatabricksSource(spark).semantic_model('sales_metrics'),
        DatabricksSource,
    )
    assert_type(
        SnowflakeSource(snowpark).semantic_model('SALES'), SnowflakeSource
    )
    source: Source = DuckDBSource(duckdb_connection)
    assert_type(source, Source)


def the_databricks_helpers_return_the_drivers_own_connections(
    workspace: WorkspaceClient,
) -> None:
    assert_type(
        warehouse('abc123', workspace=workspace),
        databricks.sql.client.Connection,
    )
    assert_type(
        lakebase('projects/p/branches/b/endpoints/e', database='postgres'),
        psycopg.Connection[Any],
    )
    assert_type(
        DatabricksSource(warehouse('abc123'), catalog='main', schema='sales'),
        DatabricksSource,
    )
    assert_type(
        PostgresSource(
            lakebase('e', database='postgres', workspace=workspace),
            schema='public',
        ),
        PostgresSource,
    )


def the_wrong_connection_is_refused(
    postgres: psycopg.Connection[Any],
    duckdb_connection: duckdb.DuckDBPyConnection,
    sqlite: adbc_driver_sqlite.dbapi.AdbcSqliteConnection,
    sqlite3_connection: sqlite3.Connection,
    async_postgres: psycopg.AsyncConnection[Any],
) -> None:
    DatabricksSource(42)  # type: ignore[arg-type]
    DatabricksSource(postgres)  # type: ignore[arg-type]
    SnowflakeSource(duckdb_connection)  # type: ignore[arg-type]
    PostgresSource(duckdb_connection)  # type: ignore[arg-type]
    DuckDBSource(sqlite)  # type: ignore[arg-type]
    SQLiteSource(duckdb_connection)  # type: ignore[arg-type]
    # Each has a member of the driver it is not: the ADBC SQLite one's, and
    # psycopg's, without the synchronous commit the engine reads through.
    SQLiteSource(sqlite3_connection)  # type: ignore[arg-type]
    PostgresSource(async_postgres)  # type: ignore[arg-type]
    # A source is abstract: it is a built-in one, or the caller's subclass.
    Source()  # type: ignore[abstract]


def a_namespace_the_warehouse_lacks_is_refused(
    postgres: psycopg.Connection[Any],
    duckdb_connection: duckdb.DuckDBPyConnection,
) -> None:
    DuckDBSource(duckdb_connection, schema='main')  # type: ignore[call-arg]
    PostgresSource(postgres, database='sales')  # type: ignore[call-arg]
