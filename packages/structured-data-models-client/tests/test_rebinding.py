# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pandas as pd
import pytest

from sdm_client import KumoRelational, PredictiveQuery, Schema
from sdm_client.errors import StructuredDataError


@pytest.fixture(params=['duckdb', 'sqlite'])
def sources(request, tmp_path):
    from sdm_connectors import DuckDBSource, SQLiteSource

    connections = []
    result = []
    for offset in [0, 100]:
        if request.param == 'duckdb':
            import duckdb

            connection = duckdb.connect()
            source = DuckDBSource(connection)
        else:
            import adbc_driver_sqlite.dbapi

            connection = adbc_driver_sqlite.dbapi.connect(
                str(tmp_path / f'source-{offset}.db')
            )
            source = SQLiteSource(connection)
        connections.append(connection)
        cursor = connection.cursor()
        cursor.execute(
            'CREATE TABLE items(id INTEGER PRIMARY KEY, value DOUBLE)'
        )
        cursor.executemany(
            'INSERT INTO items VALUES (?, ?)',
            [(i, float(i + offset)) for i in range(30)],
        )
        if request.param == 'sqlite':
            connection.commit()
        cursor.close()
        result.append(source)
    yield result
    for connection in connections:
        connection.close()


def labels(schema):
    frame = PredictiveQuery(
        'PREDICT items.doubled FOR EACH items.id', schema
    ).labels(100)
    return dict(zip(frame.ENTITY, frame.TARGET, strict=True))


def test_rebind_uses_new_connection_preserves_declarations_and_old_requests(
    sources,
):
    original = Schema.from_source(
        sources[0],
        {'items': 'items'},
        primary_keys={'items': 'id'},
        relationships=[],
    )
    original = original.with_sql_column('items', 'doubled', 'value * 2')
    query = PredictiveQuery('PREDICT items.doubled FOR EACH items.id', original)
    request = query.materialize(
        indices=[0, 1], model=KumoRelational(), random_seed=7
    )
    prepared_payloads = copy.deepcopy(request.payloads)
    rebound = original.with_source(sources[1])
    assert request.payloads == prepared_payloads
    assert rebound.source is sources[1]
    assert original.source is sources[0]
    assert rebound['items'].primary_key == 'id'
    assert (
        rebound['items']['doubled'].dtype == original['items']['doubled'].dtype
    )
    assert rebound.relationships == original.relationships
    assert rebound._inference == original._inference
    assert labels(rebound) == {i: float((i + 100) * 2) for i in range(30)}
    assert labels(original) == {i: float(i * 2) for i in range(30)}
    assert request.predict_rows == 2
    rebound['items']['value'].stype = 'categorical'
    assert original['items']['value'].stype != 'categorical'
    assert (
        query.materialize(indices=[0], model=KumoRelational()).predict_rows == 1
    )


def test_rebinding_refuses_memory_and_missing_columns_atomically(sources):
    with pytest.raises(StructuredDataError, match='source-backed'):
        Schema.from_data({'items': pd.DataFrame({'id': [1, 2]})}).with_source(
            sources[0]
        )
    original = Schema.from_source(sources[0], {'items': 'items'})
    cursor = sources[1].connection.cursor()
    cursor.execute('ALTER TABLE items DROP COLUMN value')
    cursor.close()
    with pytest.raises(StructuredDataError):
        original.with_source(sources[1])
    assert original.source is sources[0]
    assert 'value' in original['items'].columns


def test_rebinding_rejects_backend_changes(sources):
    import duckdb
    from sdm_connectors import DuckDBSource

    original = Schema.from_source(sources[0], {'items': 'items'})
    with duckdb.connect() as connection:
        other = DuckDBSource(connection)
        if sources[0].backend != 'duckdb':
            with pytest.raises(StructuredDataError, match='same backend'):
                original.with_source(other)


def test_rebinding_retains_time_end_relationship_and_explicit_types(sources):
    from sdm_client import Relationship

    for source in sources:
        cursor = source.connection.cursor()
        cursor.execute(
            'CREATE TABLE events(event_id INTEGER PRIMARY KEY, item_id INTEGER, opened TIMESTAMP, closed TIMESTAMP)'
        )
        cursor.executemany(
            'INSERT INTO events VALUES (?, ?, ?, ?)',
            [(i, i, '2024-01-01', '2024-02-01') for i in range(30)],
        )
        source.connection.commit()
        cursor.close()
    original = Schema.from_source(
        sources[0],
        {'items': 'items', 'events': 'events'},
        primary_keys={'items': 'id', 'events': 'event_id'},
        time_columns={'events': 'opened'},
        relationships=[Relationship('events', 'item_id', 'items', 'id')],
    )
    original['events'].end_time_column = 'closed'
    original['items']['value'].stype = 'categorical'
    rebound = original.with_source(sources[1])
    assert rebound['events'].time_column == 'opened'
    assert rebound['events'].end_time_column == 'closed'
    assert rebound['items']['value'].stype == 'categorical'
    assert rebound.relationships == original.relationships


def test_rebind_after_old_connection_closed(sources):
    schema = Schema.from_source(
        sources[0], {'items': 'items'}, primary_keys={'items': 'id'}
    )
    schema = schema.with_sql_column('items', 'doubled', 'value * 2')
    sources[0].connection.close()
    rebound = schema.with_source(sources[1])
    assert labels(rebound) == {i: float((i + 100) * 2) for i in range(30)}
