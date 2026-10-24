# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A floating PostgreSQL primary key finds itself through the public SDK."""

from __future__ import annotations

import os
import uuid
import warnings

import pytest

psycopg = pytest.importorskip('psycopg')
from psycopg import sql  # noqa: E402

from sdm_client import PredictiveQuery, Schema, _routing  # noqa: E402
from sdm_connectors import PostgresSource  # noqa: E402


@pytest.mark.parametrize('source_type', ['REAL', 'DOUBLE PRECISION'])
def test_fractional_floating_entity_key_materializes_its_own_row(
    monkeypatch, source_type
):
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        pytest.skip('needs the local Postgres test service')
    monkeypatch.setenv(
        _routing.ENVIRONMENT,
        ','.join(_routing.FROM_SOURCE_POSTGRES + _routing.MATERIALIZE_POSTGRES),
    )
    namespace = 't39_float_key_' + uuid.uuid4().hex[:12]
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(
            sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(namespace))
        )
        try:
            connection.execute(
                sql.SQL(
                    'CREATE TABLE {}.users(user_id {} PRIMARY KEY, score DOUBLE PRECISION, label TEXT)'
                ).format(sql.Identifier(namespace), sql.SQL(source_type))
            )
            connection.execute(
                sql.SQL(
                    "INSERT INTO {}.users SELECT i::double precision/10, i*0.5, 'row-'||i::text FROM generate_series(1,8) AS i"
                ).format(sql.Identifier(namespace))
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                schema = Schema.from_source(
                    PostgresSource(connection, schema=namespace)
                )
                request = PredictiveQuery(
                    'PREDICT users.score FOR EACH users.user_id', schema
                ).materialize(indices=[0.1])
            entities = request.payloads[0]['predict']['related_tables']['users']
            label = entities['columns'].index('label')
            key = entities['columns'].index('user_id')
            assert len(entities['rows']) == 1
            assert entities['rows'][0][label] == 'row-1'
            assert entities['rows'][0][key] == 0.1
        finally:
            connection.execute(
                sql.SQL('DROP SCHEMA {} CASCADE').format(
                    sql.Identifier(namespace)
                )
            )
