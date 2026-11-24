# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pytest
from sdm_connectors import ConnectorError

from sdm_client import (
    DatabricksSource,
    DuckDBSource,
    PostgresSource,
    SnowflakeSource,
    Source,
    SQLiteSource,
    StructuredDataError,
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
    duckdb = pytest.importorskip('duckdb')
    return duckdb.connect()


def _sqlite(tmp_path: Path) -> Any:
    dbapi = pytest.importorskip('adbc_driver_sqlite.dbapi')
    return dbapi.connect(str(tmp_path / 'db.sqlite'))


@pytest.mark.parametrize(
    'cls, connection, namespace, choice',
    [
        (DuckDBSource, _duckdb, {}, {'dialect': 'duckdb'}),
        (SQLiteSource, _sqlite, {}, {'dialect': 'sqlite'}),
        (
            DatabricksSource,
            lambda _: _databricks(),
            {'catalog': 'main', 'schema': 'sales'},
            {'dialect': 'databricks', 'namespace': ['main', 'sales']},
        ),
        (
            SnowflakeSource,
            lambda _: _snowflake(),
            {'database': 'DB'},
            {'dialect': 'snowflake', 'namespace': ['DB', None]},
        ),
        (
            PostgresSource,
            lambda _: _postgres(),
            {'schema': 'public'},
            {'dialect': 'postgres', 'namespace': ['public']},
        ),
    ],
    ids=['duckdb', 'sqlite', 'databricks', 'snowflake', 'postgres'],
)
def test_a_built_in_source_is_a_source_over_its_connection(
    cls: type, connection: Any, namespace: dict, choice: dict, tmp_path: Path
) -> None:
    opened = connection(tmp_path)

    source = cls(opened, **namespace)
    assert isinstance(source, Source)
    assert source.connection is opened
    assert source.namespace == namespace
    # The namespace outermost first, by the class's order; a part left to
    # the connection is None, and none given is left out.
    assert source._sql_choice() == {'provider': 'sql', **choice}


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
            lambda: DuckDBSource(object()),  # type: ignore[arg-type]
            'got object',
            'UNKNOWN_CONNECTOR',
        ),
        (
            lambda: DatabricksSource(_databricks(), catalog='c').semantic_model(
                ''
            ),
            'needs its name',
            'INVALID_CONNECTOR_ARGS',
        ),
        (
            lambda: DatabricksSource(_databricks(), catalog='c').semantic_model(
                'm',
                trust_expressions='yes',  # type: ignore[arg-type]
            ),
            'trust_expressions is True or False',
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

    source = cls(opened, **namespace)
    assert type(source) is cls
    assert reaches(source, opened)


def test_both_databricks_connections_are_read_by_their_own_driver() -> None:
    warehouse = DatabricksSource(_databricks(), catalog='main', schema='s')
    spark = DatabricksSource(_spark_session(), schema='s')
    assert warehouse._driver == 'DatabricksSqlProvider'
    assert spark._driver == 'DatabricksSparkProvider'
    assert spark.semantic_model('m')._driver == 'DatabricksSparkProvider'
    assert spark._sql_choice()['namespace'] == [None, 's']


def test_a_semantic_model_is_a_source_of_the_same_class() -> None:
    databricks = DatabricksSource(_databricks(), catalog='main', schema='s')
    model = databricks.semantic_model('sales_metrics')

    assert type(model) is DatabricksSource
    assert model.connection is databricks.connection
    assert model.namespace == {'catalog': 'main', 'schema': 's'}
    # Read in its source's namespace, which the source keeps unchanged.
    assert model._sql_choice() == {
        'provider': 'sql',
        'dialect': 'databricks',
        'namespace': ['main', 's'],
        'view': {'name': 'sales_metrics', 'trust_expressions': False},
    }
    assert 'view' not in databricks._sql_choice()
    snowflake = SnowflakeSource(_snowflake()).semantic_model(
        'SV', trust_expressions=True
    )
    assert type(snowflake) is SnowflakeSource
    assert snowflake._sql_choice()['view'] == {
        'name': 'SV',
        'trust_expressions': True,
    }


def test_a_source_does_not_pickle_its_connection() -> None:
    duckdb = pytest.importorskip('duckdb')
    with pytest.raises(TypeError, match='cannot be pickled'):
        pickle.dumps(DuckDBSource(duckdb.connect()))


def test_the_repr_names_where_not_how() -> None:
    source = DatabricksSource(_databricks(), catalog='main', schema='sales')
    assert repr(source) == "DatabricksSource(catalog='main', schema='sales')"
    assert repr(source.semantic_model('m')) == (
        "SemanticModel('m' in DatabricksSource(catalog='main', schema='sales'))"
    )
    assert repr(source.semantic_model('m', trust_expressions=True)) == (
        "SemanticModel('m' in DatabricksSource(catalog='main', schema='sales'), "
        'trust_expressions=True)'
    )


def _shop(connection: Any) -> None:
    for statement in (
        'CREATE TABLE users (user_id BIGINT PRIMARY KEY, age BIGINT, '
        'country VARCHAR)',
        "INSERT INTO users SELECT i, 20 + i % 50, ['US', 'DE', 'IN'][1 + i % 3] "
        'FROM range(40) t(i)',
        'CREATE TABLE orders (order_id BIGINT PRIMARY KEY, user_id BIGINT '
        'REFERENCES users (user_id), price DOUBLE, ts TIMESTAMP)',
        'INSERT INTO orders SELECT i, i % 40, 10 + i * 1.5, TIMESTAMP '
        "'2023-06-01' + to_days(CAST(i AS INTEGER)) FROM range(400) t(i)",
    ):
        connection.execute(statement)


def test_a_built_in_source_answers_directly_what_a_run_receives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A built-in source is asked through the same ``capabilities`` and
    ``answer`` as any source: asked directly, it answers each request of a
    run exactly what the core received from it in the run."""
    import pandas as pd
    import pyarrow as pa

    from sdm_client import PredictiveQuery, Schema
    from sdm_client._rust import host_providers as host

    connection = _duckdb(tmp_path)
    _shop(connection)
    source = DuckDBSource(connection)
    received: list[tuple[dict[str, Any], dict[int, pa.Table], pa.Table]] = []
    asked = host.HostProvider.__call__

    def recorded(provider: Any, step: Any, keys: Any = None) -> Any:
        # Each key set and the answer, read whole, so each can be read again.
        tables = {
            at: host._reader(given).read_all()
            for at, given in (keys or {}).items()
        }
        answer = pa.table(asked(provider, step, tables))
        received.append((step, tables, answer))
        return answer

    monkeypatch.setattr(host.HostProvider, '__call__', recorded)
    schema = Schema.from_source(source, {'users': 'users', 'orders': 'orders'})
    PredictiveQuery(
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id', schema
    ).materialize(indices=[0, 1, 2], anchor_time=pd.Timestamp('2024-03-01'))
    monkeypatch.undo()

    kinds = {step['request']['request'] for step, _, _ in received}
    assert {'describe', 'time_bounds', 'neighbors'} <= kinds, kinds
    # The same requests, asked directly, in a fresh operation.
    assert source.capabilities() == host.read_capabilities(
        host.chosen(source)[0]['capabilities']
    )
    for step, tables, answer in received:
        request = host.read_request(step['request'], tables)
        expected = host.expected_schema(step['plan'].get('expect'))
        direct = pa.table(source.answer(request, expected))
        # The same rows of the same schema; a request of no order, in any.
        assert direct.schema == answer.schema, step['request']
        assert sorted(map(repr, direct.to_pylist())) == sorted(
            map(repr, answer.to_pylist())
        ), step['request']


def test_a_built_in_source_refuses_by_name_what_its_sql_cannot_answer(
    tmp_path: Path,
) -> None:
    from sdm_client import provider as p

    source = DuckDBSource(_duckdb(tmp_path))
    # A scan of no column is no statement DuckDB's SQL writes.
    request = p.Scan(p.Relation(('t',)), (), (), None)
    with pytest.raises(StructuredDataError) as refused:
        source.answer(request, None)
    assert refused.value.code == 'INVALID_REQUEST'
