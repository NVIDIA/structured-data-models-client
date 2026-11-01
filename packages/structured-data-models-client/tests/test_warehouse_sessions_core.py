# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warehouse-prepared requests use shared Rust session rules over real HTTP APIs."""

import copy
from dataclasses import dataclass
from datetime import datetime, timedelta

import duckdb
import pytest
from adbc_driver_sqlite import dbapi as sqlite
from sdm_connectors import DuckDBSource, SQLiteSource

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Schema,
    SDMClient,
    _routing,
)
from sdm_client.errors import StructuredDataError
from sdm_client.wire.keys import _named_by, keys_in

URL = 'http://localhost'
SESSIONS = URL + '/v1/sessions'


@pytest.fixture(params=['duckdb', 'sqlite'])
def warehouse(request, monkeypatch):
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    connection = (
        duckdb.connect() if request.param == 'duckdb' else sqlite.connect()
    )
    cursor = connection.cursor()
    cursor.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, value DOUBLE)')
    cursor.executemany(
        'INSERT INTO users VALUES (?, ?)', [(i, i * 1.25) for i in range(20)]
    )
    cursor.execute(
        'CREATE TABLE events (event INTEGER PRIMARY KEY, id INTEGER, '
        'feature DOUBLE, event_time TIMESTAMP, FOREIGN KEY (id) REFERENCES users(id))'
    )
    cursor.executemany(
        'INSERT INTO events VALUES (?, ?, ?, ?)',
        [
            (
                i,
                i % 20,
                i * 2.5,
                (datetime(2024, 1, 1) + timedelta(days=i)).isoformat(),
            )
            for i in range(40)
        ],
    )
    connection.commit()
    cursor.close()
    source = (
        DuckDBSource(connection)
        if request.param == 'duckdb'
        else SQLiteSource(connection)
    )
    schema = Schema.from_source(source)
    schema['users']['value'].stype = 'numerical'
    schema['events'].time_column = 'event_time'
    try:
        yield PredictiveQuery('PREDICT users.value FOR EACH users.id', schema)
    finally:
        connection.close()


def gather(query, ids):
    return query.materialize(
        indices=ids,
        model=KumoRelational(),
        num_context_samples=20,
        random_seed=7,
    )


@pytest.fixture
def endpoint(requests_mock):
    created, scored = [], []

    def create(request, context):
        created.append(request.json())
        return {'session_id': 'warehouse-1'}

    def score(request, context):
        payload = request.json()
        scored.append(payload)
        table = payload['predict']['instance_table']
        identity = table['columns'].index('instance_id')
        return {
            'model': 'kumo-relational',
            'id': 'answer',
            'metadata': {},
            'predictions': [
                {
                    'id': str(row[identity]),
                    'row_index': i,
                    'prediction': 5.0 + i,
                }
                for i, row in reversed(list(enumerate(table['rows'])))
            ],
        }

    requests_mock.post(SESSIONS, json=create)
    requests_mock.post(SESSIONS + '/warehouse-1/predictions', json=score)
    deleted = requests_mock.delete(SESSIONS + '/warehouse-1', status_code=204)
    return created, scored, deleted


def test_warehouse_public_lifecycle_preserves_rows_and_related_holdout(
    warehouse, endpoint
):
    created, scored, deleted = endpoint
    opening = gather(warehouse, [1, 2])
    before = copy.deepcopy(opening.payloads)
    with SDMClient(URL, max_retries=0) as client:
        fitted = client.fit(opening, model=KumoRelational(), holdout=[3, 4])
        assert fitted._relational_session is not None
        assert opening.payloads == before
        uploaded = created[0]
        assert set(uploaded) == {
            'model',
            'task',
            'schema',
            'context',
            'metadata',
        }
        assert not keys_in(uploaded, 'context') & {
            _named_by([3]),
            _named_by([4]),
        }
        table = uploaded['context']['instance_table']
        at = table['columns'].index('instance_id')
        kept = {row[at] for row in table['rows']}
        for table in uploaded['context']['related_tables'].values():
            at = table['columns'].index('instance_id')
            assert all(row[at] in kept for row in table['rows'])
        for ids in ([3, 4], [1, 2]):
            result = fitted.predict(gather(warehouse, ids))
            assert result.to_pandas()['prediction'].tolist() == [5.0, 6.0]
        assert all(
            'context' not in payload and 'schema' not in payload
            for payload in scored
        )
        assert 'context' not in fitted._relational_session_pinned.decode()
        fitted.close()
        fitted.close()
    assert deleted.call_count == 1


def test_warehouse_changed_question_and_failed_delete(
    warehouse, endpoint, requests_mock
):
    _, scored, _ = endpoint
    deleted = requests_mock.delete(
        SESSIONS + '/warehouse-1',
        [
            {'status_code': 500, 'json': {'detail': 'retry deletion'}},
            {'status_code': 204},
        ],
    )
    with SDMClient(URL, max_retries=0) as client:
        fitted = client.fit(gather(warehouse, [1]), model=KumoRelational())
        changed = PredictiveQuery(
            'PREDICT users.value > 3 FOR EACH users.id', warehouse.schema
        )
        with pytest.raises(StructuredDataError, match='different question'):
            fitted.predict(gather(changed, [2]))
        assert not scored
        with pytest.raises(StructuredDataError):
            fitted.close()
        assert not fitted._closed
        fitted.close()
        assert fitted._closed
    assert deleted.call_count == 2


def test_dataclass_inference_is_observed_without_iterating_it(
    warehouse, endpoint
):
    @dataclass
    class Config:
        num_estimators: int = 1

    request = gather(warehouse, [1])
    request.requested_inference['inference_config'] = Config()
    with SDMClient(URL, max_retries=0) as client:
        fitted = client.fit(request, model=KumoRelational())
        assert fitted.predict(request).to_pandas()['prediction'].tolist() == [
            5.0
        ]


def test_temporal_session_keeps_anchor_and_holdout_identity(
    warehouse, endpoint
):
    created, scored, _ = endpoint
    query = PredictiveQuery(
        'PREDICT COUNT(events.*, 0, 7, days) FOR EACH users.id',
        warehouse.schema,
    )
    opening = gather(query, [1, 2])
    with SDMClient(URL, max_retries=0) as client:
        fitted = client.fit(opening, model=KumoRelational(), holdout=[3])
        assert fitted._relational_session['windowed']
        assert not keys_in(created[0], 'context') & {_named_by([3])}
        result = fitted.predict(gather(query, [3]))
        assert result.to_pandas()['prediction'].tolist() == [5.0]
        anchor = created[0]['task']['anchor_time_column']
        assert anchor in scored[0]['predict']['instance_table']['columns']


def test_multiple_contexts_refused_before_upload(warehouse, endpoint):
    created, _, _ = endpoint
    opening = warehouse.materialize(
        indices=[1, 2], model=KumoRelational(), rows_per_call=1, random_seed=7
    )
    assert len(opening.payloads) == 2
    with SDMClient(URL, max_retries=0) as client:
        with pytest.raises(StructuredDataError, match='one context'):
            client.fit(opening, model=KumoRelational())
    assert created == []


def test_fit_refusal_retains_column_advice(warehouse, requests_mock):
    request = gather(warehouse, [1])
    # Represent a supported caller edit to the materialized schema.
    request.payloads[0]['schema']['related_tables']['events']['columns'][
        'feature'
    ]['stype'] = 'text'
    requests_mock.post(
        SESSIONS,
        status_code=422,
        json={
            'code': 'UNSUPPORTED_DTYPE',
            'detail': "semantic type 'text' is unsupported",
        },
    )
    with SDMClient(URL, max_retries=0) as client:
        with pytest.raises(StructuredDataError) as raised:
            client.fit(request, model=KumoRelational())
    assert 'events.feature' in str(raised.value)
    assert "stype = 'categorical'" in str(raised.value)


@pytest.mark.parametrize('backend', list(_routing.WAREHOUSE_SESSIONS))
def test_session_route_can_be_selected_independently(backend, monkeypatch):
    from types import SimpleNamespace

    source = SimpleNamespace(kind=backend)
    monkeypatch.setenv(_routing.ENVIRONMENT, '')
    assert not _routing.on_the_core(source, _routing.WAREHOUSE_SESSIONS)
    monkeypatch.setenv(
        _routing.ENVIRONMENT, ','.join(_routing.WAREHOUSE_SESSIONS[backend])
    )
    assert _routing.on_the_core(source, _routing.WAREHOUSE_SESSIONS)
