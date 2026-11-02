# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warehouse schemas validate current source facts on every materialization."""

import warnings

import duckdb
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import (
    Column,
    PredictiveQuery,
    Relationship,
    Schema,
    StructuredDataError,
)

SHOP = (
    'CREATE TABLE users (user_id BIGINT PRIMARY KEY, age BIGINT)',
    'INSERT INTO users SELECT i, 20 + i % 50 FROM range(40) t(i)',
    'CREATE TABLE orders (order_id BIGINT PRIMARY KEY, user_id BIGINT, '
    'price DOUBLE, ts TIMESTAMP)',
    'INSERT INTO orders SELECT i, i % 40, 10 + i * 1.5, TIMESTAMP '
    "'2023-06-01' + to_days(CAST(i AS INTEGER)) FROM range(400) t(i)",
)


@pytest.fixture
def connection():
    with duckdb.connect() as conn:
        yield conn


def describe(connection, *, key_type='BIGINT', rebuilt=False):
    for statement in SHOP:
        connection.execute(
            statement.replace('user_id BIGINT,', f'user_id {key_type},')
        )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(
            DuckDBSource(connection), {'users': 'users', 'orders': 'orders'}
        )
    if rebuilt:
        return Schema(schema.tables, schema.relationships, source=schema.source)
    return schema


def materialize(schema):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return PredictiveQuery(
            'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
            schema,
        ).materialize(indices=[0, 1, 2], anchor_time=pd.Timestamp('2024-03-01'))


def assert_prediction_rows(request):
    assert request.predict.instance_table['__kumo_entity_ref_0'].tolist() == [
        0,
        1,
        2,
    ]
    assert len(request.context.instance_table) > 0


@pytest.mark.parametrize('meaning', ['text', 'timestamp', 'numerical'])
def test_schema_edits_validate_the_current_declared_dtype(connection, meaning):
    schema = describe(connection)
    column = schema['users']['age']
    column.dtype = 'string'
    if meaning == 'numerical':
        with pytest.raises(StructuredDataError) as error:
            column.stype = meaning
        assert error.value.code == 'INVALID_REQUEST'
        return
    column.stype = meaning
    assert (column.dtype, column.stype, column.stype_declared) == (
        'string',
        meaning,
        True,
    )


@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
def test_added_time_column_accepts_every_timestamp_unit(connection, unit):
    schema = describe(connection)
    schema['users'].columns['seen'] = Column(
        'seen', f'timestamp[{unit}]', 'timestamp'
    )
    schema['users'].time_column = 'seen'
    assert schema['users'].time_column == 'seen'
    assert schema['users']['seen'].stype == 'timestamp'


@pytest.mark.parametrize('fresh', [False, True])
def test_foreign_key_declaration_cannot_hide_source_type(connection, fresh):
    schema = describe(connection, key_type='VARCHAR')
    schema.relationships = []
    schema['orders']['user_id'].dtype = 'int64'
    schema.relationships = [
        Relationship('orders', 'user_id', 'users', 'user_id')
    ]
    if fresh:
        schema['users'].columns['age'] = Column('age', 'int64', 'numerical')
    with pytest.raises(StructuredDataError) as error:
        materialize(schema)
    assert error.value.code == 'INVALID_REQUEST'
    assert 'orders.user_id' in str(error.value)
    assert 'the source holds it as' in str(error.value)


@pytest.mark.parametrize('previous_run', [False, True])
def test_dropped_column_is_detected_on_next_materialization(
    connection, previous_run
):
    schema = describe(connection, rebuilt=True)
    if previous_run:
        assert_prediction_rows(materialize(schema))
    connection.execute('ALTER TABLE users DROP COLUMN age')
    with pytest.raises(StructuredDataError) as error:
        materialize(schema)
    assert error.value.code == 'INVALID_REQUEST'
    assert 'users.age' in str(error.value)
    assert 'the source has no such column' in str(error.value)


@pytest.mark.parametrize('change', ['added', 'retyped', 'retyped_unmatched'])
def test_materialization_reads_changed_source_columns(connection, change):
    schema = describe(connection)
    column = 'plan' if change == 'added' else 'age'
    if change == 'added':
        connection.execute(
            "ALTER TABLE users ADD COLUMN plan VARCHAR DEFAULT 'NEW_FEATURE'"
        )
    else:
        connection.execute('ALTER TABLE users ALTER COLUMN age TYPE VARCHAR')
        connection.execute("UPDATE users SET age = 'NEW_FEATURE'")
    if change == 'retyped_unmatched':
        with pytest.raises(StructuredDataError) as error:
            materialize(schema)
        assert error.value.code == 'INVALID_REQUEST'
        assert 'users.age' in str(error.value)
        assert 'the source holds it as' in str(error.value)
        return
    schema['users'].columns[column] = Column(column, 'string', 'categorical')
    request = materialize(schema)
    assert_prediction_rows(request)
    assert set(request.context.related_tables['users'][column]) == {
        'NEW_FEATURE'
    }
    assert set(request.predict.related_tables['users'][column]) == {
        'NEW_FEATURE'
    }


def test_rebuilt_schema_reads_new_type_and_keeps_declared_meaning(connection):
    schema = describe(connection, rebuilt=True)
    connection.execute('ALTER TABLE users ALTER COLUMN age TYPE VARCHAR')
    connection.execute("UPDATE users SET age = 'CHANGED'")
    schema['users']['age'].dtype = 'string'
    schema['users']['age'].stype = 'text'
    request = materialize(schema)
    assert_prediction_rows(request)
    assert schema['users']['age'].stype == 'text'
    assert set(map(tuple, request.context.related_tables['users']['age'])) == {
        ('changed',)
    }
    assert set(map(tuple, request.predict.related_tables['users']['age'])) == {
        ('changed',)
    }


# Accepted edits for historical dtype spellings, independently specified from
# the public semantic-type and join-family rules. Other combinations refuse.
EDITABLE_TYPES = {
    'int64': ('key', 'meaning', 'link'),
    'int32': ('key', 'meaning', 'link'),
    'int': ('key', 'meaning', 'link'),
    'int16': ('key', 'meaning', 'link'),
    'byte': ('key', 'meaning'),
    'float64': ('key', 'meaning', 'link'),
    'float32': ('key', 'meaning', 'link'),
    'float': ('key', 'meaning', 'link'),
    'timestamp[s]': ('time',),
    'timestamp[ms]': ('time',),
    'timestamp[us]': ('time',),
    'timestamp[ns]': ('time',),
    'date': ('time',),
    'time': ('time',),
    'string': ('key', 'time', 'meaning'),
    'bool': (),  # Cannot construct a Boolean ID column.
    'decimal(9, 2)': ('key', 'time', 'meaning'),
}


@pytest.mark.parametrize('dtype', EDITABLE_TYPES)
@pytest.mark.parametrize('edit', ['key', 'time', 'meaning', 'link'])
def test_added_column_edits_follow_declared_dtype_rules(
    connection, dtype, edit
):
    schema = describe(connection)
    initial = 'timestamp' if dtype.startswith(('time', 'date')) else 'ID'

    def apply():
        schema['orders'].columns['extra'] = Column('extra', dtype, initial)
        if edit == 'key':
            schema['orders'].primary_key = 'extra'
        elif edit == 'time':
            schema['orders'].time_column = 'extra'
        elif edit == 'meaning':
            schema['orders']['extra'].stype = 'categorical'
        else:
            schema.relationships = [
                *schema.relationships,
                Relationship('orders', 'extra', 'users', 'user_id'),
            ]

    if edit not in EDITABLE_TYPES[dtype]:
        with pytest.raises(StructuredDataError) as error:
            apply()
        assert error.value.code == 'INVALID_REQUEST'
        return
    apply()
    table = schema['orders']
    assert (
        table.primary_key,
        table.time_column,
        table['extra'].stype,
        len(schema.relationships),
    ) == {
        'key': ('extra', 'ts', 'ID', 1),
        'time': ('order_id', 'extra', 'timestamp', 1),
        'meaning': ('order_id', 'ts', 'categorical', 1),
        'link': ('order_id', 'ts', 'ID', 2),
    }[edit]


@pytest.mark.parametrize(
    ('table', 'column', 'dtype', 'meaning', 'outcome'),
    [
        ('users', 'age', 'string', 'text', 'source_refusal'),
        ('users', 'age', 'string', 'numerical', 'edit_refusal'),
        ('orders', 'price', 'string', 'categorical', 'query_refusal'),
        ('users', 'user_id', 'float64', 'ID', 'accepted'),
        ('orders', 'ts', 'string', 'text', 'source_refusal'),
    ],
)
def test_retyped_columns_validate_meaning_then_source(
    connection, table, column, dtype, meaning, outcome
):
    schema = describe(connection)
    schema[table][column].dtype = dtype
    if outcome == 'edit_refusal':
        with pytest.raises(StructuredDataError) as error:
            schema[table][column].stype = meaning
        assert error.value.code == 'INVALID_REQUEST'
        return
    schema[table][column].stype = meaning
    if outcome == 'accepted':
        assert_prediction_rows(materialize(schema))
        return
    with pytest.raises(StructuredDataError) as error:
        materialize(schema)
    assert error.value.code == 'INVALID_REQUEST'
    if outcome == 'query_refusal':
        assert 'Aggregation SUM can only operate on integers and floats' in str(
            error.value
        )
        assert 'orders.price has data type string' in str(error.value)
        return
    assert f'{table}.{column} is given as' in str(error.value)
    assert 'the source holds it as' in str(error.value)


@pytest.mark.parametrize(
    'dtype',
    ['int8', 'int128', 'float16', 'float8', 'timestamp[us, UTC]', 'uint64'],
)
@pytest.mark.parametrize('replace_column', [False, True])
def test_unlisted_dtype_names_keep_join_family_rules(
    connection, dtype, replace_column
):
    schema = describe(connection)
    schema.relationships = []
    if replace_column:
        del schema['orders']['user_id']
        schema['orders'].columns['user_id'] = Column('user_id', dtype, 'ID')
    else:
        schema['orders']['user_id'].dtype = dtype
    link = Relationship('orders', 'user_id', 'users', 'user_id')
    if dtype in ('timestamp[us, UTC]', 'uint64'):
        with pytest.raises(StructuredDataError) as error:
            schema.relationships = [link]
        assert error.value.code == 'INVALID_REQUEST'
        assert 'user_id' in str(error.value)
        return
    schema.relationships = [link]
    assert schema.relationships == [link]
    assert_prediction_rows(materialize(schema))


@pytest.mark.parametrize('meaning', ['categorical', 'ID', 'numerical'])
def test_known_unsupported_dtype_rejects_supported_meanings(
    connection, meaning
):
    schema = describe(connection)
    schema['users'].columns['odd'] = Column('odd', 'string', 'text')
    schema['users']['odd'].dtype = 'unsupported'
    with pytest.raises(StructuredDataError) as error:
        schema['users']['odd'].stype = meaning
    assert error.value.code == 'INVALID_REQUEST'
    assert schema['users']['odd'].stype == 'text'


@pytest.mark.parametrize(
    ('dtype', 'edit'),
    [
        ('unsupported', 'key'),
        ('unsupported', 'time'),
        ('byte', 'time'),
        ('uuid', 'key'),
        ('json', 'link'),
    ],
)
def test_unknown_types_and_known_unsupported_have_distinct_edit_rules(
    connection, dtype, edit
):
    schema = describe(connection)
    schema['orders'].columns['extra'] = Column(
        'extra', 'string' if dtype == 'unsupported' else dtype, 'ID'
    )

    schema['orders']['extra'].dtype = dtype

    def apply():
        if edit == 'key':
            schema['orders'].primary_key = 'extra'
        elif edit == 'time':
            schema['orders'].time_column = 'extra'
        else:
            schema['users']['user_id'].dtype = 'decimal(9, 2)'
            schema.relationships = [
                Relationship('orders', 'extra', 'users', 'user_id')
            ]

    if dtype == 'uuid':
        apply()
        assert schema['orders'].primary_key == 'extra'
        assert schema['orders'].time_column == 'ts'
        assert len(schema.relationships) == 1
        return
    with pytest.raises(StructuredDataError) as error:
        apply()
    assert error.value.code == 'INVALID_REQUEST'
    assert 'extra' in str(error.value)
    assert dtype in str(error.value)


def test_materialization_rechecks_mutated_declared_meaning(connection):
    schema = describe(connection)
    schema['users']['age'].stype = 'numerical'
    schema['users']['age'].dtype = 'string'
    with pytest.raises(StructuredDataError) as error:
        materialize(schema)
    assert error.value.code == 'INVALID_REQUEST'
    assert 'age' in str(error.value)
    assert 'string' in str(error.value)
    assert 'numerical' in str(error.value)


def test_materialization_rechecks_mutated_foreign_key_type(connection):
    schema = describe(connection)
    schema['orders']['user_id'].dtype = 'string'
    with pytest.raises(StructuredDataError) as error:
        materialize(schema)
    assert error.value.code == 'INVALID_REQUEST'
    assert 'cannot match' in str(error.value)
    assert 'user_id' in str(error.value)


def test_unsupported_is_not_a_user_assignable_semantic_type():
    with pytest.raises(StructuredDataError, match='not a semantic type'):
        Column('odd', 'unsupported', 'unsupported')
