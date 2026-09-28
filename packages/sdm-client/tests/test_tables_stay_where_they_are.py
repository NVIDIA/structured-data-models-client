# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A schema over tables in their source, sampled there with SQL.

DuckDB and SQLite stand in for a warehouse: the engine samples all of them
through the same SQL sampler, and these two need no credentials. What is
checked is what a caller would notice being wrong -- that nothing is read
whole, that the source's own sampler runs, and that every label the sampler
computes is the one the rows say.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('sdm_relational_engine.relationallib')
duckdb = pytest.importorskip('duckdb')
pytest.importorskip('adbc_driver_sqlite')

from sdm_connectors import Source, connect  # noqa: E402

from sdm_client import (  # noqa: E402
    KumoTabular,
    PredictiveQuery,
    Relationship,
    Schema,
    SourceTable,
    StructuredDataError,
    Table,
)
from sdm_client.core.warnings import SDMWarning  # noqa: E402
from sdm_client.materializers import RelationalMaterializer  # noqa: E402

QUERY = 'PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.user_id'
TABLES = {'users': 'app_users', 'orders': 'app_orders'}


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    users = pd.DataFrame(
        {
            'user_id': range(1, 51),
            'age': rng.integers(18, 70, 50),
            'country': rng.choice(['US', 'DE', 'IN'], 50),
        }
    )
    # Hourly timestamps over a year, so windows of whole days land exactly on
    # stored rows: the edges are where a text comparison goes wrong.
    orders = pd.DataFrame(
        {
            'order_id': range(1, 1001),
            'user_id': rng.integers(1, 51, 1000),
            'amount': rng.gamma(2, 20, 1000).round(2),
            'ts': pd.Timestamp('2024-01-01')
            + pd.to_timedelta(rng.integers(0, 365 * 24, 1000), unit='h'),
        }
    )
    return users, orders


USERS, ORDERS = _frames()


@pytest.fixture
def duckdb_source() -> Any:
    connection = connect('duckdb')
    connection.register('users_frame', USERS)
    connection.register('orders_frame', ORDERS)
    connection.execute('CREATE TABLE app_users AS SELECT * FROM users_frame')
    connection.execute('CREATE TABLE app_orders AS SELECT * FROM orders_frame')
    yield Source(connection)
    connection.close()


@pytest.fixture
def sqlite_source(tmp_path: Path) -> Any:
    path = tmp_path / 'sales.sqlite'
    sqlite3.connect(path).close()
    connection = connect('sqlite', str(path))
    # Ingested through ADBC, which stores a timestamp as ISO text with
    # fractional seconds -- the spelling whose window edges went wrong.
    with connection.cursor() as cursor:
        cursor.adbc_ingest('app_users', USERS, mode='create')
        cursor.adbc_ingest('app_orders', ORDERS, mode='create')
    connection.commit()
    yield Source(connection)
    connection.close()


@pytest.fixture(params=['duckdb', 'sqlite'])
def source(request: pytest.FixtureRequest) -> Any:
    return request.getfixturevalue(f'{request.param}_source')


def _schema(source: Any, **overrides: Any) -> Schema:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return Schema.from_source(source, TABLES, **overrides)


def _payload(schema: Any, seed: int = 7) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        request = PredictiveQuery(QUERY, schema).materialize(
            indices=[1, 2, 3], random_seed=seed
        )
    (payload,) = request._payloads
    return json.loads(json.dumps(payload, default=str))


def test_the_tables_are_described_without_being_read(source: Any) -> None:
    schema = _schema(source)

    assert schema.source is source
    assert all(isinstance(t, SourceTable) for t in schema.tables.values())
    assert not any(hasattr(t, 'data') for t in schema.tables.values())
    assert schema['users'].source_name == 'app_users'
    assert schema['users'].primary_key == 'user_id'
    assert schema['orders'].primary_key == 'order_id'
    assert schema['orders'].time_column == 'ts'
    assert schema['orders']['amount'].stype == 'numerical'
    assert schema['users']['country'].stype == 'categorical'
    assert schema.relationships == [
        Relationship('orders', 'user_id', 'users', 'user_id')
    ]


def test_sampling_runs_in_the_source(source: Any) -> None:
    from sdm_relational_engine.rfm.base import DataBackend

    graph = RelationalMaterializer(_schema(source))._build_graph()

    assert graph.backend == DataBackend(source.backend)


@pytest.mark.parametrize('seed', [7, 11, 23])
def test_every_sampled_label_is_the_sum_the_rows_give(
    source: Any, seed: int
) -> None:
    r"""Each context row's label is recomputed from the frames and compared.

    SQLite once lost one label in six hundred here: its window bound was
    compared as text and lacked the fractional seconds the column stores, so
    an order exactly on the end of a window fell outside it.
    """
    rows = _payload(_schema(source), seed)['context']['instance_table']['rows']

    wrong = []
    for _, user, anchor, label in rows:
        at = pd.Timestamp(anchor).tz_localize(None)
        inside = (
            (ORDERS.user_id == user)
            & (ORDERS.ts > at)
            & (ORDERS.ts <= at + pd.Timedelta(days=30))
        )
        expected = ORDERS.amount[inside].sum()
        if not np.isclose(label, expected, rtol=1e-5, atol=1e-3):
            wrong.append((user, anchor, label, expected))
    assert rows
    assert not wrong


def test_a_scored_entity_sees_the_history_it_would_in_memory(
    source: Any,
) -> None:
    def history(payload: dict[str, Any]) -> list[tuple[Any, ...]]:
        table = payload['predict']['related_tables']['orders']
        at = {name: i for i, name in enumerate(table['columns'])}
        return sorted(
            (row[at['user_id']], row[at['ts']], row[at['amount']])
            for row in table['rows']
        )

    in_place = _payload(_schema(source))
    in_memory = _payload({'users': USERS, 'orders': ORDERS})

    assert history(in_place) == history(in_memory)
    assert (
        in_place['predict']['instance_table']
        == in_memory['predict']['instance_table']
    )


def test_a_declared_foreign_key_links_what_names_would_not() -> None:
    connection = duckdb.connect()
    connection.execute('CREATE TABLE u (user_id INTEGER PRIMARY KEY, age INT)')
    connection.execute(
        'CREATE TABLE o (order_id INTEGER PRIMARY KEY, '
        'buyer INTEGER REFERENCES u(user_id), amount DOUBLE, ts TIMESTAMP)'
    )
    connection.execute('INSERT INTO u VALUES (1, 30)')
    connection.execute("INSERT INTO o VALUES (1, 1, 9.5, '2024-01-01')")

    schema = Schema.from_source(
        Source(connection), {'users': 'u', 'orders': 'o'}
    )

    assert schema.relationships == [
        Relationship('orders', 'buyer', 'users', 'user_id')
    ]


def test_overrides_mean_what_they_mean_on_from_data(
    duckdb_source: Any,
) -> None:
    schema = _schema(
        duckdb_source,
        primary_keys={'orders': 'order_id'},
        time_columns={'orders': 'ts'},
        relationships=[],
    )
    assert schema.relationships == []

    with pytest.raises(StructuredDataError, match="no table 'order'"):
        _schema(duckdb_source, time_columns={'order': 'ts'})


def test_a_column_with_no_scalar_form_is_left_out_and_named() -> None:
    connection = duckdb.connect()
    connection.execute(
        'CREATE TABLE t (id INTEGER PRIMARY KEY, tags INTEGER[], x DOUBLE)'
    )
    connection.execute('INSERT INTO t VALUES (1, [1, 2], 0.5)')

    with pytest.warns(SDMWarning, match=r'leaves out tags'):
        schema = Schema.from_source(Source(connection), {'t': 't'})

    assert list(schema['t'].columns) == ['id', 'x']


def test_a_column_left_out_is_neither_read_nor_sent(duckdb_source: Any) -> None:
    schema = _schema(duckdb_source)
    del schema['users']['country']

    assert 'country' not in schema['users']
    graph = RelationalMaterializer(schema)._build_graph()
    assert 'country' not in graph['users']
    payload = _payload(schema)
    assert (
        'country'
        not in payload['context']['related_tables']['users']['columns']
    )

    with pytest.raises(StructuredDataError, match='primary key; set that'):
        del schema['users']['user_id']
    with pytest.raises(StructuredDataError, match='time column; set that'):
        del schema['orders']['ts']
    with pytest.raises(StructuredDataError, match="has no column 'nope'"):
        del schema['users']['nope']


def test_the_tabular_model_is_refused_rather_than_fed_nothing(
    duckdb_source: Any,
) -> None:
    schema = Schema.from_source(
        duckdb_source, {'users': 'app_users'}, relationships=[]
    )
    query = PredictiveQuery('PREDICT users.age FOR EACH users.user_id', schema)

    with pytest.raises(StructuredDataError, match='only the relational'):
        query.materialize(indices=[1], model=KumoTabular())
    with pytest.raises(StructuredDataError, match='only the relational'):
        query.materialize(predict=USERS.head(1))


def test_a_one_table_source_schema_is_gathered_for_the_relational_model(
    duckdb_source: Any,
) -> None:
    schema = Schema.from_source(
        duckdb_source, {'users': 'app_users'}, relationships=[]
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        request = PredictiveQuery(
            'PREDICT users.age FOR EACH users.user_id', schema
        ).materialize(indices=[1, 2])

    assert request.predict_rows == 2
    assert request.schema.source is duckdb_source


def test_a_schema_is_frames_or_tables_in_one_source(duckdb_source: Any) -> None:
    in_place = _schema(duckdb_source)
    frame = Table('users', USERS, {})

    with pytest.raises(StructuredDataError, match='samples one or the other'):
        Schema(
            {'users': frame, 'orders': in_place['orders']},
            source=duckdb_source,
        )
    with pytest.raises(StructuredDataError, match='needs that source'):
        Schema({'orders': in_place['orders']})
    with pytest.raises(StructuredDataError, match='which a source does not'):
        Schema({'users': frame}, source=duckdb_source)


def test_a_connection_is_not_a_source(duckdb_source: Any) -> None:
    with pytest.raises(StructuredDataError, match=r'needs an sdm_connectors'):
        Schema.from_source(duckdb_source.connection, TABLES)


def test_a_snapshot_keeps_the_source(duckdb_source: Any) -> None:
    schema = _schema(duckdb_source)
    copied = schema.snapshot()

    assert copied.source is duckdb_source
    assert isinstance(copied['orders'], SourceTable)
    assert copied['orders'].source_name == 'app_orders'
    copied['orders']['amount'].stype = 'categorical'
    assert schema['orders']['amount'].stype == 'numerical'


def test_a_source_schema_does_not_pickle(duckdb_source: Any) -> None:
    with pytest.raises(TypeError, match='cannot be pickled'):
        pickle.dumps(_schema(duckdb_source))
