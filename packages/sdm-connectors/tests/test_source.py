# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from sdm_connectors import ConnectorError, Source, connect


def _class_in(module: str, name: str, **attributes: Any) -> type:
    return type(name, (), {'__module__': module, **attributes})


def test_a_duckdb_connection_is_a_source() -> None:
    pytest.importorskip('duckdb')
    connection = connect('duckdb')
    source = Source(connection)

    assert source.backend == 'duckdb'
    assert source.connection is connection
    assert source.namespace == {}


def test_a_sqlite_connection_is_a_source(tmp_path: Path) -> None:
    pytest.importorskip('adbc_driver_sqlite')
    path = tmp_path / 'db.sqlite'
    sqlite3.connect(path).close()

    assert Source(connect('sqlite', str(path))).backend == 'sqlite'


def test_the_namespace_is_the_backend_s_own() -> None:
    databricks = _class_in('databricks.sql.client', 'Connection')()
    snowflake = _class_in('snowflake.connector.connection', 'Conn')()
    postgres = _class_in('psycopg', 'Connection')()

    source = Source(databricks, catalog='main', schema='sales')
    assert (source.backend, source.namespace) == (
        'databricks',
        {'catalog': 'main', 'schema': 'sales'},
    )
    assert Source(snowflake, database='DB').namespace == {'database': 'DB'}
    assert Source(postgres, schema='public').backend == 'postgres'

    with pytest.raises(ConnectorError, match='catalog and schema') as error:
        Source(databricks, database='DB')
    assert error.value.code == 'INVALID_CONNECTOR_ARGS'
    with pytest.raises(ConnectorError, match='no namespace'):
        Source(connect('duckdb'), schema='main')
    with pytest.raises(ConnectorError, match='non-empty string'):
        Source(postgres, schema='')


def test_anything_else_is_refused_by_name() -> None:
    with pytest.raises(ConnectorError, match='got object') as error:
        Source(object())
    assert error.value.code == 'UNKNOWN_CONNECTOR'


def test_a_notebook_spark_session_is_sampled_through_spark_sql() -> None:
    session_type = _class_in(
        'pyspark.sql.session',
        'SparkSession',
        sql=lambda self, *args, **kwargs: None,
        version='4.0.0',
    )

    source = Source(session_type(), catalog='main', schema='sales')

    from sdm_connectors.backends.spark import SparkConnection

    assert source.backend == 'databricks'
    assert isinstance(source.connection, SparkConnection)


def test_a_snowpark_session_is_reached_through_its_connection() -> None:
    connection = _class_in('snowflake.connector.connection', 'Conn')()
    session = _class_in('snowflake.snowpark.session', 'Session')()
    session.connection = connection

    source = Source(session, database='DB', schema='PUBLIC')

    assert source.backend == 'snowflake'
    assert source.connection is connection


def test_a_source_does_not_pickle_its_connection() -> None:
    pytest.importorskip('duckdb')
    with pytest.raises(TypeError, match='cannot be pickled'):
        pickle.dumps(Source(connect('duckdb')))


def test_the_repr_names_where_not_how() -> None:
    databricks = _class_in('databricks.sql.client', 'Connection')()
    assert (
        repr(Source(databricks, catalog='main', schema='sales'))
        == "Source(databricks, catalog='main', schema='sales')"
    )
