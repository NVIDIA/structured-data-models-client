# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Current source selection and nested-column behavior through public APIs."""

from pathlib import Path

import duckdb
import pandas as pd
import pytest
import yaml
from sdm_client import PredictiveQuery, Schema, StructuredDataError
from sdm_connectors import DuckDBSource

ROOT = Path(__file__).resolve().parents[2]
QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'


@pytest.fixture
def connection():
    with duckdb.connect() as con:
        for statement in yaml.safe_load(
            (ROOT / 'contract/bases.yaml').read_text()
        )['bases']['shop']:
            con.execute(statement)
        yield con


def materialize(schema):
    return PredictiveQuery(QUERY, schema).materialize(
        indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01')
    )


@pytest.mark.parametrize('table', ['users', 'orders'])
def test_list_features_are_omitted_from_sampled_payload(connection, table):
    connection.execute(
        f'CREATE OR REPLACE TABLE {table} AS SELECT *, [1,2]::INTEGER[] AS f FROM {table}'
    )
    assert connection.execute(f'SELECT f FROM {table} LIMIT 1').fetchone()[
        0
    ] == [1, 2]
    schema = Schema.from_source(DuckDBSource(connection))
    request = materialize(schema)
    assert request.payloads
    for body in request.payloads:
        assert 'f' not in body['schema']['related_tables'][table]['columns']
        for part in ('context', 'predict'):
            sampled = body[part]['related_tables'][table]
            assert sampled['rows']
            assert 'f' not in sampled['columns']


@pytest.mark.parametrize('table', ['users', 'orders'])
def test_struct_feature_is_refused_at_source_admission(connection, table):
    connection.execute(
        f"CREATE OR REPLACE TABLE {table} AS SELECT *, {{'a':1,'b':'text'}} AS f FROM {table}"
    )
    with pytest.raises(StructuredDataError) as raised:
        Schema.from_source(DuckDBSource(connection))
    assert raised.value.code == 'UNSUPPORTED_REQUEST'
    assert raised.value.details['rule'] == 'column_type'


@pytest.mark.parametrize('include_empty', [False, True])
def test_only_selected_empty_tables_block_materialization(
    connection, include_empty
):
    connection.execute('CREATE TABLE extra (extra_id BIGINT, v INTEGER)')
    tables = {'users': 'users', 'orders': 'orders'}
    if include_empty:
        tables['extra'] = 'extra'
        with pytest.raises(StructuredDataError) as raised:
            Schema.from_source(DuckDBSource(connection), tables=tables)
        assert raised.value.code == 'INVALID_REQUEST'
    else:
        request = materialize(
            Schema.from_source(DuckDBSource(connection), tables=tables)
        )
        assert request.payloads
        for body in request.payloads:
            assert 'extra' not in body['schema']['related_tables']
            assert body['predict']['instance_table']['rows']


@pytest.mark.parametrize(
    ('key_type', 'amounts', 'dropped'),
    [('VARCHAR', [1.0, 2.0], True), ('DECIMAL(18, 0)', [0.0, 1.0, 2.0], False)],
)
def test_sampled_null_key_rows_and_warning(key_type, amounts, dropped):
    import warnings

    with duckdb.connect() as con, warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        con.execute(
            'CREATE TABLE accounts AS SELECT i AS account_id, i::DOUBLE AS amount FROM range(200) t(i)'
        )
        key = (
            "'k'||i::VARCHAR" if key_type == 'VARCHAR' else 'i::DECIMAL(18, 0)'
        )
        con.execute(
            f'CREATE TABLE payments AS SELECT CASE WHEN i=0 THEN NULL ELSE {key} END AS payment_id, CASE WHEN i<3 THEN 3 ELSE 150 END AS account_id, i::DOUBLE AS amount FROM range(200) t(i)'
        )
        request = PredictiveQuery(
            'PREDICT accounts.amount FOR EACH accounts.account_id',
            Schema.from_source(DuckDBSource(con)),
        ).materialize(indices=[3], num_context_samples=1, random_seed=0)
        assert len(request.payloads) == 1
        table = request.payloads[0]['predict']['related_tables']['payments']
        at = table['columns'].index('amount')
        assert sorted(row[at] for row in table['rows']) == amounts
    dropped_warnings = [
        str(w.message) for w in seen if '[NULL_KEYS_LEFT_OUT]' in str(w.message)
    ]
    assert len(dropped_warnings) == int(dropped)
    if dropped:
        assert 'payments.payment_id' in dropped_warnings[0]
        assert '1 of the rows sampled' in dropped_warnings[0]


@pytest.mark.parametrize(
    'text', ['https://example.test/a;b', '12:34:56', '3:2']
)
def test_separator_text_remains_a_string_feature(text):
    with duckdb.connect() as con:
        con.execute(
            'CREATE TABLE people AS SELECT i AS id, i::DOUBLE AS y, ?::VARCHAR AS note FROM range(10) t(i)',
            [text],
        )
        request = PredictiveQuery(
            'PREDICT people.y FOR EACH people.id',
            Schema.from_source(DuckDBSource(con)),
        ).materialize(indices=[0, 1], num_context_samples=3, random_seed=0)
        assert request.payloads
        for body in request.payloads:
            assert body['schema']['related_tables']['people']['columns'][
                'note'
            ] == {'dtype': 'string', 'stype': 'categorical'}
            for part in ('context', 'predict'):
                table = body[part]['related_tables']['people']
                at = table['columns'].index('note')
                assert table['rows']
                assert all(row[at] == text for row in table['rows'])


def test_clearing_schema_attributes_preserves_the_other_attribute(connection):
    schema = Schema.from_source(
        DuckDBSource(connection),
        primary_keys={'orders': 'order_id'},
        time_columns={'orders': 'ts'},
    )
    table = schema['orders']
    assert table.primary_key == 'order_id'
    assert table.time_column == 'ts'
    table.time_column = None
    assert table.time_column is None
    assert table.primary_key == 'order_id'
    table.time_column = 'ts'
    table.primary_key = None
    assert table.primary_key is None
    assert table.time_column == 'ts'


@pytest.mark.parametrize(
    ('instant', 'valid'),
    [
        ('9999-12-31 14:59:59.999999+00', True),
        ('9999-12-31 15:00:00+00', False),
    ],
)
def test_zoned_source_values_respect_the_supported_year_boundary(
    connection, instant, valid
):
    connection.execute("SET GLOBAL TimeZone = 'Asia/Tokyo'")
    connection.execute('ALTER TABLE users ADD COLUMN f TIMESTAMPTZ')
    connection.execute(
        "UPDATE users SET f = TIMESTAMPTZ '2024-01-01 00:00:00+00'"
    )
    connection.execute(
        'UPDATE users SET f = CAST(? AS TIMESTAMPTZ) WHERE user_id = 0',
        [instant],
    )
    if valid:
        schema = Schema.from_source(DuckDBSource(connection))
        assert 'f' in schema['users'].columns
    else:
        with pytest.raises(StructuredDataError) as raised:
            Schema.from_source(DuckDBSource(connection))
        assert raised.value.code not in (None, 'INTERNAL_ERROR')
        assert raised.value.details['table'] == 'users'
        assert raised.value.details['column'] == 'f'


@pytest.mark.parametrize('aggregation', ['AVG', 'MIN', 'MAX'])
@pytest.mark.parametrize(
    ('operator', 'expected'), [('= NULL', False), ('!= NULL', True)]
)
def test_aggregate_null_comparisons_label_populated_windows(
    connection, aggregation, operator, expected
):
    connection.execute('DROP TABLE orders')
    connection.execute(
        "CREATE TABLE orders AS SELECT i::BIGINT AS order_id, (i % 8)::BIGINT AS user_id, (10+i*1.5)::DOUBLE AS price, TIMESTAMP '2023-01-01'+to_days((i // 8)::INTEGER) AS ts FROM range(4800) t(i)"
    )
    schema = Schema.from_source(DuckDBSource(connection))
    query = f'PREDICT {aggregation}(orders.price, 0, 30, days) {operator} FOR EACH users.user_id'
    request = PredictiveQuery(query, schema).materialize(
        indices=[0, 1],
        anchor_time=pd.Timestamp('2024-03-01'),
        num_context_samples=3,
        random_seed=42,
    )
    assert request.payloads
    labels = []
    for payload in request.payloads:
        table = payload['context']['instance_table']
        at = table['columns'].index('TARGET')
        labels.extend(row[at] for row in table['rows'])
    assert labels
    assert all(label == expected for label in labels), labels


def test_average_null_comparison_refuses_windows_without_values(connection):
    schema = Schema.from_source(DuckDBSource(connection))
    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery(
            'PREDICT AVG(orders.price, 0, 30, days) = NULL FOR EACH users.user_id',
            schema,
        ).materialize(
            indices=[0, 1],
            anchor_time=pd.Timestamp('2024-03-01'),
            num_context_samples=3,
            random_seed=42,
        )
    assert raised.value.code == 'UNSUPPORTED_REQUEST'
    assert raised.value.details['rule'] == 'average_null'
