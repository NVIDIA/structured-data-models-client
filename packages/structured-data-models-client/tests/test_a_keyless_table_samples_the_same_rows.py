# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A table with no key is sampled the same however its rows are stored.

Neighbours are kept in time order, newest first, and a keyless table's rows
that share a timestamp are ordered by their other columns, so Databricks
serverless Spark and a SQL warehouse build one request for one question. Two
copies of one table, stored in opposite row orders, stand in for the two
engines here.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

pytest.importorskip('sdm_relational_engine.relationallib')
duckdb = pytest.importorskip('duckdb')
dbapi = pytest.importorskip('adbc_driver_sqlite.dbapi')

from sdm_connectors import DuckDBSource, SQLiteSource  # noqa: E402

from sdm_client import PredictiveQuery, Relationship, Schema  # noqa: E402

USERS = pd.DataFrame({'user_id': range(1, 21), 'age': range(20, 40)})
# Every user has five visits on each of three days, so a day's visits tie on
# the time column and only their other columns tell them apart.
VISITS = pd.DataFrame(
    [
        {
            'user_id': user,
            'day': pd.Timestamp('2024-01-01') + pd.Timedelta(days=7 * week),
            'page': f'p{(user * 7 + week * 3 + visit) % 11}',
            'seconds': (user * 13 + week * 5 + visit * 17) % 97,
        }
        for user in range(1, 21)
        for week in range(3)
        for visit in range(5)
    ]
)
QUERY = 'PREDICT COUNT(visits.*, 0, 7, days) FOR EACH users.user_id'


def _duckdb(visits: pd.DataFrame, tmp_path: Path) -> Any:
    connection = duckdb.connect()
    # Registered as Arrow: DuckDB cannot read pandas 3's string column in place.
    connection.register('users_frame', pa.Table.from_pandas(USERS))
    connection.register('visits_frame', pa.Table.from_pandas(visits))
    connection.execute('CREATE TABLE users AS SELECT * FROM users_frame')
    connection.execute('CREATE TABLE visits AS SELECT * FROM visits_frame')
    return DuckDBSource(connection)


def _sqlite(visits: pd.DataFrame, tmp_path: Path) -> Any:
    connection = dbapi.connect(str(tmp_path / f'db-{id(visits)}.sqlite'))
    with connection.cursor() as cursor:
        cursor.adbc_ingest('users', USERS, mode='create')
        cursor.adbc_ingest('visits', visits, mode='create')
    connection.commit()
    return SQLiteSource(connection)


def _predict_part(source: Any) -> str:
    schema = Schema.from_source(
        source,
        {'users': 'users', 'visits': 'visits'},
        primary_keys={'users': 'user_id'},
        time_columns={'visits': 'day'},
        relationships=[Relationship('visits', 'user_id', 'users', 'user_id')],
    )
    assert schema['visits'].primary_key is None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        request = PredictiveQuery(QUERY, schema).materialize(
            indices=[1, 2, 3], num_neighbors=[4, 4], random_seed=7
        )
    # The scored entities and their neighbourhood. The context's entities are
    # drawn at random, which SQLite cannot seed, so they are not compared.
    return json.dumps(
        [payload['predict'] for payload in request._payloads],
        sort_keys=True,
        default=str,
    )


@pytest.mark.parametrize('build', [_duckdb, _sqlite], ids=['duckdb', 'sqlite'])
def test_rows_tied_in_time_are_kept_and_sent_in_one_order(
    build: Any, tmp_path: Path
) -> None:
    stored = build(VISITS, tmp_path)
    reversed_ = build(VISITS.iloc[::-1].reset_index(drop=True), tmp_path)

    assert _predict_part(stored) == _predict_part(reversed_)
