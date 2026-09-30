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


def _duckdb(tmp_path: Path) -> Any:
    pytest.importorskip('duckdb')
    return connect('duckdb')


def _sqlite(tmp_path: Path) -> Any:
    pytest.importorskip('adbc_driver_sqlite')
    path = tmp_path / 'db.sqlite'
    sqlite3.connect(path).close()
    return connect('sqlite', str(path))


@pytest.mark.parametrize(
    'cls, connection, backend, namespace',
    [
        (DuckDBSource, _duckdb, 'duckdb', {}),
        (SQLiteSource, _sqlite, 'sqlite', {}),
        (
            DatabricksSource,
            lambda _: _databricks(),
            'databricks',
            {'catalog': 'main', 'schema': 'sales'},
        ),
        (
            SnowflakeSource,
            lambda _: _snowflake(),
            'snowflake',
            {'database': 'DB'},
        ),
        (
            PostgresSource,
            lambda _: _postgres(),
            'postgres',
            {'schema': 'public'},
        ),
    ],
    ids=['duckdb', 'sqlite', 'databricks', 'snowflake', 'postgres'],
)
def test_a_source_is_the_class_that_fits_its_connection(
    cls: type, connection: Any, backend: str, namespace: dict, tmp_path: Path
) -> None:
    opened = connection(tmp_path)

    for source in (cls(opened, **namespace), Source(opened, **namespace)):
        assert type(source) is cls
        assert source.connection is opened
        assert (source.backend, source.namespace) == (backend, namespace)


@pytest.mark.parametrize(
    'call, match, code',
    [
        (
            lambda: PostgresSource(_postgres(), schema=''),
            'non-empty string',
            'INVALID_CONNECTOR_ARGS',
        ),
        (
            lambda: DatabricksSource(_postgres(), catalog='main'),
            'DatabricksSource needs',
            'UNKNOWN_CONNECTOR',
        ),
        (
            lambda: PostgresSource(_databricks()),
            'PostgresSource needs',
            'UNKNOWN_CONNECTOR',
        ),
        (
            lambda: Source(_databricks(), database='DB'),
            'catalog and schema',
            'INVALID_CONNECTOR_ARGS',
        ),
        (
            lambda: Source(_duckdb(Path()), schema='main'),
            'no namespace',
            'INVALID_CONNECTOR_ARGS',
        ),
        (lambda: Source(object()), 'got object', 'UNKNOWN_CONNECTOR'),
        (
            lambda: SemanticModel(PostgresSource(_postgres()), 'x'),  # type: ignore[arg-type]
            'kept in Databricks or',
            'INVALID_CONNECTOR_ARGS',
        ),
        (
            lambda: DatabricksSource(_databricks(), catalog='c').semantic_model(
                ''
            ),
            'needs its name',
            'INVALID_CONNECTOR_ARGS',
        ),
    ],
)
def test_what_a_source_cannot_be_is_refused(
    call: Any, match: str, code: str
) -> None:
    with pytest.raises(ConnectorError, match=match) as error:
        call()
    assert error.value.code == code


def _spark_session() -> Any:
    return _class_in(
        'pyspark.sql.session',
        'SparkSession',
        sql=lambda self, *args, **kwargs: None,
        version='4.0.0',
    )()


def _snowpark_session() -> Any:
    session = _class_in('snowflake.snowpark.session', 'Session')()
    session.connection = _snowflake()
    return session


@pytest.mark.parametrize(
    'session, cls, namespace, reaches',
    [
        (
            _spark_session,
            DatabricksSource,
            {'catalog': 'main', 'schema': 'sales'},
            lambda source, session: source.connection.session is session,
        ),
        (
            _snowpark_session,
            SnowflakeSource,
            {'database': 'DB', 'schema': 'PUBLIC'},
            lambda source, session: source.connection is session.connection,
        ),
    ],
    ids=['spark', 'snowpark'],
)
def test_a_notebook_session_is_a_source(
    session: Any, cls: type, namespace: dict, reaches: Any
) -> None:
    opened = session()

    for source in (cls(opened, **namespace), Source(opened, **namespace)):
        assert type(source) is cls
        assert reaches(source, opened)


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
