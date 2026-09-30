# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What a type checker makes of a source, checked by mypy rather than run.

Named without ``test_`` so that pytest leaves it alone. Each ``assert_type`` is
an inference that must not change, and each ``type: ignore`` a call that must
stay refused: the repository warns on an unused ignore, so a refused call that
starts to type-check fails the check.
"""

from __future__ import annotations

from typing import Any, assert_type

import adbc_driver_sqlite.dbapi
import databricks.sql.client
import duckdb
import psycopg
import pyspark.sql
import pyspark.sql.connect.session
import snowflake.connector
import snowflake.snowpark

from sdm_connectors import (
    DatabricksSource,
    DuckDBSource,
    PostgresSource,
    SemanticModel,
    SnowflakeSource,
    Source,
    SQLiteSource,
)
from sdm_connectors.backends.spark import SparkConnection


def source_is_typed_as_the_class_it_returns(
    warehouse: databricks.sql.client.Connection,
    spark: pyspark.sql.SparkSession,
    spark_connect: pyspark.sql.connect.session.SparkSession,
    snowflake_connection: snowflake.connector.SnowflakeConnection,
    snowpark: snowflake.snowpark.Session,
    postgres: psycopg.Connection[Any],
    duckdb_connection: duckdb.DuckDBPyConnection,
    sqlite: adbc_driver_sqlite.dbapi.AdbcSqliteConnection,
) -> None:
    assert_type(
        Source(warehouse, catalog='main', schema='sales'), DatabricksSource
    )
    assert_type(Source(spark, catalog='main'), DatabricksSource)
    assert_type(Source(spark_connect), DatabricksSource)
    assert_type(
        Source(snowflake_connection, database='SALES', schema='PUBLIC'),
        SnowflakeSource,
    )
    assert_type(Source(snowpark), SnowflakeSource)
    assert_type(Source(postgres, schema='public'), PostgresSource)
    assert_type(Source(duckdb_connection), DuckDBSource)
    assert_type(Source(sqlite), SQLiteSource)
    assert_type(Source(duckdb_connection).connection, duckdb.DuckDBPyConnection)


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
        DatabricksSource(spark).semantic_model('sales_metrics'), SemanticModel
    )
    assert_type(
        SnowflakeSource(snowpark).semantic_model('SALES'), SemanticModel
    )


def the_wrong_connection_is_refused(
    postgres: psycopg.Connection[Any],
    duckdb_connection: duckdb.DuckDBPyConnection,
    sqlite: adbc_driver_sqlite.dbapi.AdbcSqliteConnection,
) -> None:
    DatabricksSource(42)  # type: ignore[arg-type]
    DatabricksSource(postgres)  # type: ignore[arg-type]
    SnowflakeSource(duckdb_connection)  # type: ignore[arg-type]
    PostgresSource(duckdb_connection)  # type: ignore[arg-type]
    DuckDBSource(sqlite)  # type: ignore[arg-type]
    SQLiteSource(duckdb_connection)  # type: ignore[arg-type]
    Source(42)  # type: ignore[call-overload]
    Source('postgresql://localhost/sales')  # type: ignore[call-overload]


def a_namespace_the_warehouse_lacks_is_refused(
    postgres: psycopg.Connection[Any],
    duckdb_connection: duckdb.DuckDBPyConnection,
) -> None:
    Source(duckdb_connection, schema='main')  # type: ignore[call-overload]
    Source(postgres, catalog='main')  # type: ignore[call-overload]
    DuckDBSource(duckdb_connection, schema='main')  # type: ignore[call-arg]
    PostgresSource(postgres, database='sales')  # type: ignore[call-arg]
