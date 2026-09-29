# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from sdm_connectors import (
    ConnectorError,
    DatabricksSource,
    DuckDBSource,
    PostgresSource,
    SemanticModel,
    SnowflakeSource,
    Source,
    SQLiteSource,
    connect,
)


def _class_in(module: str, name: str, **attributes: Any) -> type:
    return type(name, (), {'__module__': module, **attributes})


def _databricks() -> Any:
    return _class_in('databricks.sql.client', 'Connection')()


def _snowflake() -> Any:
    return _class_in('snowflake.connector.connection', 'Conn')()


def _postgres() -> Any:
    return _class_in('psycopg', 'Connection')()


def test_a_duckdb_connection_is_a_duckdb_source() -> None:
    pytest.importorskip('duckdb')
    connection = connect('duckdb')
    source = DuckDBSource(connection)

    assert source.backend == 'duckdb'
    assert source.connection is connection
    assert source.namespace == {}


def test_a_sqlite_connection_is_a_sqlite_source(tmp_path: Path) -> None:
    pytest.importorskip('adbc_driver_sqlite')
    path = tmp_path / 'db.sqlite'
    sqlite3.connect(path).close()

    assert SQLiteSource(connect('sqlite', str(path))).backend == 'sqlite'


def test_each_class_names_its_own_namespace() -> None:
    source = DatabricksSource(_databricks(), catalog='main', schema='sales')
    assert (source.backend, source.namespace) == (
        'databricks',
        {'catalog': 'main', 'schema': 'sales'},
    )
    snowflake = SnowflakeSource(_snowflake(), database='DB')
    assert snowflake.namespace == {'database': 'DB'}
    assert PostgresSource(_postgres(), schema='public').backend == 'postgres'

    with pytest.raises(ConnectorError, match='non-empty string'):
        PostgresSource(_postgres(), schema='')


def test_a_class_refuses_another_warehouse_s_connection() -> None:
    with pytest.raises(ConnectorError, match='DatabricksSource needs') as error:
        DatabricksSource(_postgres(), catalog='main')
    assert error.value.code == 'UNKNOWN_CONNECTOR'
    with pytest.raises(ConnectorError, match='PostgresSource needs'):
        PostgresSource(_databricks())


def test_source_returns_the_class_that_fits_the_connection() -> None:
    assert type(Source(_databricks(), catalog='main')) is DatabricksSource
    assert type(Source(_snowflake(), database='DB')) is SnowflakeSource
    assert type(Source(_postgres(), schema='public')) is PostgresSource

    with pytest.raises(ConnectorError, match='catalog and schema') as error:
        Source(_databricks(), database='DB')
    assert error.value.code == 'INVALID_CONNECTOR_ARGS'


def test_duckdb_has_no_namespace_to_name() -> None:
    pytest.importorskip('duckdb')
    with pytest.raises(ConnectorError, match='no namespace'):
        Source(connect('duckdb'), schema='main')


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

    from sdm_connectors.backends.spark import SparkConnection

    for source in (
        DatabricksSource(session_type(), catalog='main', schema='sales'),
        Source(session_type(), catalog='main', schema='sales'),
    ):
        assert isinstance(source, DatabricksSource)
        assert isinstance(source.connection, SparkConnection)


def test_a_snowpark_session_is_reached_through_its_connection() -> None:
    connection = _snowflake()
    session = _class_in('snowflake.snowpark.session', 'Session')()
    session.connection = connection

    for source in (
        SnowflakeSource(session, database='DB', schema='PUBLIC'),
        Source(session, database='DB', schema='PUBLIC'),
    ):
        assert isinstance(source, SnowflakeSource)
        assert source.connection is connection


def test_a_semantic_model_is_a_source_kept_in_databricks_or_snowflake() -> None:
    databricks = DatabricksSource(_databricks(), catalog='main', schema='s')
    model = databricks.semantic_model('sales_metrics')

    assert isinstance(model, SemanticModel)
    assert isinstance(model, Source)
    assert model.source is databricks
    assert model.name == 'sales_metrics'
    assert model.backend == 'databricks'
    assert model.namespace == {'catalog': 'main', 'schema': 's'}
    assert (
        SnowflakeSource(_snowflake()).semantic_model('SV').backend
        == 'snowflake'
    )


def test_only_warehouses_that_keep_semantic_models_offer_them() -> None:
    for cls in (PostgresSource, DuckDBSource, SQLiteSource):
        assert not hasattr(cls, 'semantic_model')
    with pytest.raises(ConnectorError, match='kept in Databricks or'):
        SemanticModel(PostgresSource(_postgres()), 'x')  # type: ignore[arg-type]


def test_a_source_does_not_pickle_its_connection() -> None:
    pytest.importorskip('duckdb')
    with pytest.raises(TypeError, match='cannot be pickled'):
        pickle.dumps(DuckDBSource(connect('duckdb')))


def test_the_repr_names_where_not_how() -> None:
    source = DatabricksSource(_databricks(), catalog='main', schema='sales')
    assert repr(source) == "DatabricksSource(catalog='main', schema='sales')"
    assert repr(source.semantic_model('m')) == (
        "SemanticModel('m' in DatabricksSource(catalog='main', schema='sales'))"
    )
