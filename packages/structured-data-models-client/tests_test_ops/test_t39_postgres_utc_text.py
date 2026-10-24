# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zoned Postgres text is UTC without changing the caller's session."""

from __future__ import annotations

import os
import uuid
import warnings
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import psycopg
import pytest
from psycopg import sql
from psycopg.types.multirange import Multirange
from psycopg.types.range import Range
from sdm_connectors import PostgresSource

from sdm_client import PredictiveQuery, Schema, _routing
from sdm_client._rust.providers.postgres_arrow import rows_arrow


def _column(name, oid, values, *, array_oid=0, element_oid=None):
    info = SimpleNamespace(
        name=name, oid=element_oid or oid, array_oid=array_oid
    )
    connection = SimpleNamespace(
        adapters=SimpleNamespace(types=SimpleNamespace(get=lambda _: info))
    )
    column = SimpleNamespace(
        name='v', type_code=oid, precision=None, scale=None
    )
    table = rows_arrow([column], [(value,) for value in values], connection)
    return table.column('v').to_pylist()


def test_zoned_range_and_array_text_is_utc_at_a_dst_transition():
    zone = ZoneInfo('America/Los_Angeles')
    lower = datetime(2024, 3, 10, 1, 30, 0, 123456, tzinfo=zone)
    upper = datetime(2024, 3, 10, 3, 30, 0, 654321, tzinfo=zone)
    span = Range(lower, upper, '[)')
    expected = (
        '[2024-03-10 09:30:00.123456+00:00, 2024-03-10 10:30:00.654321+00:00)'
    )
    assert _column('tstzrange', 3910, [span, None, Range(empty=True)]) == [
        expected,
        None,
        'empty',
    ]
    assert _column(
        'tstzmultirange', 4534, [Multirange([span]), Multirange()]
    ) == ['{' + expected + '}', '{}']
    assert _column(
        'timestamptz',
        1185,
        [[lower, None, upper], None],
        array_oid=1185,
        element_oid=1184,
    ) == [
        [
            '2024-03-10 09:30:00.123456+00:00',
            None,
            '2024-03-10 10:30:00.654321+00:00',
        ],
        None,
    ]


def test_open_ranges_and_utc_year_edges_keep_their_values():
    first = datetime(1, 1, 1, 5, tzinfo=timezone(timedelta(hours=14)))
    last = datetime(9999, 12, 31, 23, tzinfo=timezone(timedelta(hours=-3)))
    assert _column(
        'timestamptz', 1185, [[first, last]], array_oid=1185, element_oid=1184
    ) == [['0000-12-31 15:00:00+00:00', '10000-01-01 02:00:00+00:00']]
    assert _column(
        'tstzrange', 3910, [Range(first, None), Range(None, last)]
    ) == [
        '[0000-12-31 15:00:00+00:00, None)',
        '(None, 10000-01-01 02:00:00+00:00)',
    ]


def test_timezone_free_and_non_temporal_ranges_keep_native_text():
    naive = Range(datetime(2024, 3, 10, 2, 30), datetime(2024, 3, 10, 3))
    numeric = Range(Decimal('1.50'), Decimal('2.00'))
    assert _column('tsrange', 3908, [naive]) == [str(naive)]
    assert _column('numrange', 3906, [numeric]) == [str(numeric)]
    assert _column(
        'timestamp', 1115, [[naive.lower]], array_oid=1115, element_oid=1114
    ) == [[naive.lower]]


def test_materialized_range_feature_does_not_follow_connection_timezone(
    monkeypatch,
):
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        pytest.fail('needs the local Postgres test service')
    monkeypatch.setenv(
        _routing.ENVIRONMENT,
        ','.join(_routing.FROM_SOURCE_POSTGRES + _routing.MATERIALIZE_POSTGRES),
    )
    namespace = 't39_utc_' + uuid.uuid4().hex[:12]
    with psycopg.connect(dsn, autocommit=True) as owner:
        owner.execute(
            sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(namespace))
        )
        try:
            owner.execute(
                sql.SQL(
                    'CREATE TABLE {}.users(user_id INTEGER PRIMARY KEY, score DOUBLE PRECISION, span TSTZRANGE)'
                ).format(sql.Identifier(namespace))
            )
            owner.execute(
                sql.SQL(
                    "INSERT INTO {}.users SELECT i,i*0.5,tstzrange('2024-03-10 09:30:00.123456+00','2024-03-10 10:30:00.654321+00','[)') FROM generate_series(0,7) AS i"
                ).format(sql.Identifier(namespace))
            )
            features = []
            for zone in ('UTC', 'America/Los_Angeles', 'Europe/Berlin'):
                with psycopg.connect(dsn, autocommit=True) as connection:
                    connection.execute(
                        "SELECT set_config('TimeZone', %s, false)", (zone,)
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        schema = Schema.from_source(
                            PostgresSource(connection, schema=namespace)
                        )
                        request = PredictiveQuery(
                            'PREDICT users.score FOR EACH users.user_id', schema
                        ).materialize(indices=[0, 1])
                    assert (
                        connection.execute('SHOW TimeZone').fetchone()[0]
                        == zone
                    )
                    table = request.payloads[0]['predict']['related_tables'][
                        'users'
                    ]
                    at = table['columns'].index('span')
                    features.append([row[at] for row in table['rows']])
            assert (
                features
                == [
                    [
                        '[2024-03-10 09:30:00.123456+00:00, 2024-03-10 10:30:00.654321+00:00)'
                    ]
                    * 2
                ]
                * 3
            )
        finally:
            owner.execute(
                sql.SQL('DROP SCHEMA {} CASCADE').format(
                    sql.Identifier(namespace)
                )
            )
