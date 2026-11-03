# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public expiry and future-information boundaries on the same data in each host."""

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

from sdm_client import (
    Column,
    KumoRelational,
    PredictiveQuery,
    Schema,
    SourceTable,
)
from sdm_client.errors import StructuredDataError

FIXTURE = json.loads(
    (
        Path(__file__).resolve().parents[3]
        / 'tests/fixtures/end-time/accounts.json'
    ).read_text()
)
ANCHOR = pd.Timestamp(FIXTURE['anchor'])


def configure(schema):
    schema['accounts'].primary_key = 'id'
    schema['accounts'].time_column = 'opened'
    schema['accounts']['value'].stype = 'numerical'
    schema['accounts'].end_time_column = 'closed'
    return schema


@pytest.fixture(params=['memory', 'sqlite', 'duckdb'])
def accounts(request):
    frame = pd.DataFrame(
        FIXTURE['rows'], columns=['id', 'value', 'opened', 'closed']
    )
    if request.param == 'memory':
        for name in ('opened', 'closed'):
            frame[name] = pd.to_datetime(frame[name])
        yield configure(Schema.from_data({'accounts': frame}))
        return
    import duckdb
    from adbc_driver_sqlite import dbapi as sqlite
    from sdm_connectors import DuckDBSource, SQLiteSource

    connection = (
        duckdb.connect() if request.param == 'duckdb' else sqlite.connect()
    )
    try:
        cursor = connection.cursor()
        dtype = 'TIMESTAMP' if request.param == 'duckdb' else 'TEXT'
        cursor.execute(
            f'CREATE TABLE accounts(id BIGINT PRIMARY KEY, value DOUBLE, opened {dtype}, closed {dtype})'
        )
        cursor.executemany(
            'INSERT INTO accounts VALUES (?, ?, ?, ?)', FIXTURE['rows']
        )
        connection.commit()
        cursor.close()
        source = (
            DuckDBSource(connection)
            if request.param == 'duckdb'
            else SQLiteSource(connection)
        )
        yield configure(Schema.from_source(source))
    finally:
        connection.close()


def query(schema):
    return PredictiveQuery(
        'PREDICT accounts.value FOR EACH accounts.id', schema
    )


def rows(table):
    return [
        dict(zip(table['columns'], row, strict=True)) for row in table['rows']
    ]


def test_labels_do_not_sample_expired_accounts(accounts):
    output = query(accounts).labels(100, anchor_time=ANCHOR)
    assert set(output.ENTITY) == set(FIXTURE['eligible'])
    assert dict(zip(output.ENTITY, output.TARGET, strict=True)) == {
        3: 3.0,
        4: 4.0,
        5: 5.0,
    }
    # Entity-time anchors intentionally retain the old expiry-bypass rule.
    assert set(query(accounts).labels(100, anchor_time='entity').ENTITY) == set(
        range(6)
    )


def test_explicit_expired_ids_remain_and_future_values_are_hidden(accounts):
    request = query(accounts).materialize(
        indices=[0, 3],
        anchor_time=ANCHOR,
        num_context_samples=3,
        model=KumoRelational(),
    )
    payload = request.payloads[0]
    context = rows(payload['context']['related_tables']['accounts'])
    prediction = rows(payload['predict']['related_tables']['accounts'])
    assert {row['id'] for row in context} == {4, 5}
    assert [row['id'] for row in prediction] == [0, 3]
    assert all(row['closed'] is None for row in context)
    assert str(prediction[0]['closed']).startswith('2024-01-02')
    assert prediction[1]['closed'] is None
    assert accounts['accounts'].end_time_column == 'closed'
    assert accounts['accounts'].metadata_report().end_time_column == 'closed'
    if accounts.source is None:
        assert accounts['accounts'].data.loc[3, 'closed'] == pd.Timestamp(
            '2024-01-05'
        )
        restored = pickle.loads(pickle.dumps(accounts))
        assert restored['accounts'].end_time_column == 'closed'
        assert set(query(restored).labels(100, anchor_time=ANCHOR).ENTITY) == {
            3,
            4,
            5,
        }
        restored_request = pickle.loads(pickle.dumps(request))
        assert restored_request.payloads == request.payloads


def test_clear_role_and_end_only_tables(accounts):
    accounts['accounts'].time_column = None
    assert set(query(accounts).labels(100, anchor_time=ANCHOR).ENTITY) == {
        3,
        4,
        5,
    }
    accounts['accounts'].end_time_column = None
    assert set(query(accounts).labels(100, anchor_time=ANCHOR).ENTITY) == set(
        range(6)
    )
    # A cleared start must not be re-adopted when assigning an end role.
    accounts['accounts'].end_time_column = 'closed'
    assert accounts['accounts'].time_column is None
    assert accounts.infer_metadata()['accounts'].time_column != 'closed'


def test_invalid_role_edits_are_atomic(accounts):
    table = accounts['accounts']
    for value in ('id', 'opened', 'absent', 'value', 42):
        with pytest.raises(StructuredDataError):
            table.end_time_column = value
        assert table.end_time_column == 'closed'
    with pytest.raises(StructuredDataError):
        table.time_column = 'closed'
    with pytest.raises(StructuredDataError):
        table.primary_key = 'closed'
    with pytest.raises(StructuredDataError):
        table['closed'].stype = 'categorical'
    assert table.time_column == 'opened'
    assert table.primary_key == 'id'


def test_manual_source_table_end_role_is_checked():
    columns = {
        'id': Column('id', 'int64', 'ID'),
        'closed': Column('closed', 'timestamp[ns]', 'timestamp'),
    }
    with pytest.raises(StructuredDataError):
        SourceTable(
            'accounts',
            'accounts',
            columns,
            primary_key='id',
            end_time_column='absent',
        )


def test_timezone_aware_end_values_use_the_same_instant():
    frame = pd.DataFrame(
        FIXTURE['rows'], columns=['id', 'value', 'opened', 'closed']
    )
    for name in ('opened', 'closed'):
        frame[name] = pd.to_datetime(frame[name], utc=True).dt.tz_convert(
            'Europe/Berlin'
        )
    schema = configure(Schema.from_data({'accounts': frame}))
    output = query(schema).labels(
        100, anchor_time=pd.Timestamp('2024-01-04T01:00:00+01:00')
    )
    assert set(output.ENTITY) == {3, 4, 5}
    request = query(schema).materialize(
        indices=[3], anchor_time=ANCHOR, model=KumoRelational()
    )
    assert (
        rows(request.payloads[0]['predict']['related_tables']['accounts'])[0][
            'closed'
        ]
        is None
    )


def forecast_schema():
    frame = pd.DataFrame(
        FIXTURE['rows'], columns=['id', 'value', 'opened', 'closed']
    )
    for name in ('opened', 'closed'):
        frame[name] = pd.to_datetime(frame[name])
    orders = pd.DataFrame(
        {
            'event': range(48),
            'id': [i % 6 for i in range(48)],
            'at': [
                pd.Timestamp('2024-01-01') + pd.Timedelta(days=i // 6)
                for i in range(48)
            ],
        }
    )
    from sdm_client import Relationship

    schema = Schema.from_data(
        {'accounts': frame, 'orders': orders},
        primary_keys={'accounts': 'id', 'orders': 'event'},
        time_columns={'accounts': 'opened', 'orders': 'at'},
        relationships=[Relationship('orders', 'id', 'accounts', 'id')],
    )
    return configure(schema)


def test_forecast_parts_keep_future_end_hidden_at_shared_anchor():
    schema = forecast_schema()
    text = 'PREDICT COUNT(orders.*, 0, 1, days) FORECAST 3 TIMEFRAMES FOR accounts.id = 3'
    request = PredictiveQuery(text, schema).materialize(
        indices=[3],
        anchor_time=ANCHOR,
        num_context_samples=3,
        model=KumoRelational(),
    )
    seen = []
    for payload in request.payloads:
        prediction = rows(payload['predict']['related_tables']['accounts'])
        assert {row['id'] for row in prediction} == {3}
        seen.append(prediction[0]['closed'])
    assert len(seen) == 3
    assert seen == [None, None, None]
    for payload in request.payloads:
        assert all(
            str(row['__kumo_anchor_time']).startswith('2024-01-04')
            for row in rows(payload['predict']['instance_table'])
        )
    # A later explicit request may expose the end once that date is known.
    later = query(schema).materialize(
        indices=[3],
        anchor_time=pd.Timestamp('2024-01-05'),
        model=KumoRelational(),
    )
    assert str(
        rows(later.payloads[0]['predict']['related_tables']['accounts'])[0][
            'closed'
        ]
    ).startswith('2024-01-05')
    # Explicit forecast IDs bypass initial expiry, unlike auto-selected entities.
    expired = PredictiveQuery(text.replace('= 3', '= 0'), schema).labels(
        3, anchor_time=ANCHOR
    )
    assert expired.ENTITY.tolist() == [0, 0, 0]
    assert expired.TARGET.tolist() == [1.0, 1.0, 1.0]


def test_fit_preparation_uses_masked_context_and_keeps_null_ends():
    from sdm_client import SDMClient

    class Endpoint:
        serves_sessions = True
        url = 'http://localhost'

        def create_session(self, payload):
            self.payload = payload
            return {'session_id': 'end-times'}

        def delete_session(self, session_id):
            assert session_id == 'end-times'

        def close(self):
            pass

    schema = forecast_schema()
    endpoint = Endpoint()
    client = SDMClient.__new__(SDMClient)
    client._start(endpoint)
    try:
        request = query(schema).materialize(
            indices=[0],
            anchor_time=ANCHOR,
            num_context_samples=3,
            model=KumoRelational(),
        )
        fitted = client.fit(request, model=KumoRelational())
        context = rows(
            endpoint.payload['context']['related_tables']['accounts']
        )
        assert {row['id'] for row in context} == {3, 4, 5}
        assert all(row['closed'] is None for row in context)
        fitted.close()
    finally:
        client.close()
