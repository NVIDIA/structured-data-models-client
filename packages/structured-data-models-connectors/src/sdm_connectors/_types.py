# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The sessions a source takes, and what ``Source(connection)`` tells
apart, as a type checker sees them.

Every driver is imported under ``TYPE_CHECKING`` only, so this package still
imports on a machine that has none of them. A driver's connection type is its
backend's ``Connection``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

if TYPE_CHECKING:
    import pyspark.sql
    import pyspark.sql.connect.session
    import snowflake.snowpark

    #: A notebook's ``spark``: classic, or Spark Connect as serverless runs it.
    SparkSession: TypeAlias = (
        pyspark.sql.SparkSession | pyspark.sql.connect.session.SparkSession
    )
    SnowparkSession: TypeAlias = snowflake.snowpark.Session

# What `Source(connection)` dispatches on: members that tell the drivers apart,
# not their classes, since a missing driver is Any and would claim any call.


class DatabricksLike(Protocol):
    def get_session_id_hex(self, *args: Any, **kwargs: Any) -> Any: ...


class SnowflakeLike(Protocol):
    def execute_string(self, *args: Any, **kwargs: Any) -> Any: ...


class PostgresLike(Protocol):
    @property
    def pgconn(self) -> Any: ...

    # A synchronous commit, which psycopg's AsyncConnection does not have.
    def commit(self) -> None: ...


class DuckDBLike(Protocol):
    def install_extension(self, *args: Any, **kwargs: Any) -> Any: ...


class SQLiteLike(Protocol):
    # ADBC's, so the standard library's sqlite3, which has the other, is not.
    def adbc_get_info(self, *args: Any, **kwargs: Any) -> Any: ...

    def enable_load_extension(self, *args: Any, **kwargs: Any) -> Any: ...


class SparkLike(Protocol):
    @property
    def readStream(self) -> Any: ...


class SnowparkLike(Protocol):
    def get_current_database(self, *args: Any, **kwargs: Any) -> Any: ...
