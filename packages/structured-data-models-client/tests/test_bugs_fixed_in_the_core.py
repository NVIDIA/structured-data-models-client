# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Bugs of the Python implementation, fixed once in the core: each answers
correctly now, or is refused as an ``sdm_client`` error that says what it
refuses."""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
import pytest

duckdb = pytest.importorskip('duckdb')

from sdm_connectors import DuckDBSource  # noqa: E402

from sdm_client import PredictiveQuery, Schema  # noqa: E402
from sdm_client.errors import StructuredDataError  # noqa: E402

QUESTION = 'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id'


@pytest.fixture(autouse=True)
def _quiet() -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


def _shop() -> tuple[pd.DataFrame, pd.DataFrame]:
    users = pd.DataFrame({'user_id': range(1, 21), 'age': range(20, 40)})
    orders = pd.DataFrame(
        {
            'order_id': range(1, 201),
            'user_id': [1 + i % 20 for i in range(200)],
            'quantity': [1 + i % 5 for i in range(200)],
            'ordered_at': pd.date_range('2024-01-01', periods=200, freq='D'),
        }
    )
    return users, orders


def _connection(frames: dict[str, pd.DataFrame]) -> Any:
    connection = duckdb.connect()
    for name, frame in frames.items():
        connection.register('frame', frame)
        connection.execute(f'CREATE TABLE {name} AS SELECT * FROM frame')
        connection.unregister('frame')
    return connection


def _refused(error: pytest.ExceptionInfo[StructuredDataError], *names: str):
    message = str(error.value)
    assert error.value.code != 'INTERNAL_ERROR', message
    assert 'failed while building' not in message, message
    for name in names:
        assert name in message, message


def test_an_empty_table_in_a_source_is_refused_by_name():
    connection = duckdb.connect()
    connection.execute('CREATE TABLE users (user_id INTEGER PRIMARY KEY)')
    connection.execute(
        'CREATE TABLE orders (order_id INTEGER, user_id INTEGER)'
    )
    connection.execute('INSERT INTO users VALUES (1), (2)')
    with pytest.raises(StructuredDataError) as error:
        Schema.from_source(DuckDBSource(connection))
    _refused(error, "'orders'", 'empty')


@pytest.mark.parametrize(
    ('column', 'values', 'dtype'),
    [
        ('signal', [complex(i, 1) for i in range(20)], 'complex128'),
        (
            'joined',
            pd.period_range('2020-01', periods=20, freq='M'),
            'period[M]',
        ),
        ('band', pd.interval_range(0, 20), 'interval[int64, right]'),
    ],
)
def test_a_column_of_a_type_no_model_reads_is_refused_by_name(
    column, values, dtype
):
    users, orders = _shop()
    users[column] = values
    schema = Schema.from_data({'users': users, 'orders': orders})
    with pytest.raises(StructuredDataError) as error:
        PredictiveQuery(QUESTION, schema).materialize(
            indices=[1, 2], random_seed=7
        )
    assert error.value.code == 'UNSUPPORTED_REQUEST', str(error.value)
    _refused(error, f'users.{column}', dtype)


def test_a_table_emptied_after_the_schema_is_refused_by_name():
    users, orders = _shop()
    connection = _connection({'users': users, 'orders': orders})
    schema = Schema.from_source(DuckDBSource(connection))
    connection.execute('DELETE FROM orders')
    with pytest.raises(StructuredDataError) as error:
        PredictiveQuery(QUESTION, schema).materialize(
            indices=[1, 2], random_seed=7
        )
    _refused(error, "'orders'")
    assert 'Cannot cast' not in str(error.value)


@pytest.mark.parametrize('name', ['instance_id', '__kumo_anchor_time'])
def test_a_reserved_name_added_after_the_schema_is_refused(name):
    users, orders = _shop()
    schema = Schema.from_data({'users': users, 'orders': orders})
    users[name] = 1
    with pytest.raises(StructuredDataError) as error:
        PredictiveQuery(QUESTION, schema).materialize(
            indices=[1, 2], random_seed=7
        )
    _refused(error, f'users.{name}')
    assert 'cannot insert' not in str(error.value)


@pytest.mark.parametrize('column', ['churn', 'user_id'])
def test_a_column_dropped_after_the_schema_is_refused_by_name(column):
    people = pd.DataFrame(
        {
            'user_id': range(12),
            'age': [20.0 + i for i in range(12)],
            'segment': ['south', 'north', 'north'] * 4,
            'churn': [0, 1] * 6,
        }
    )
    schema = Schema.from_data({'people': people})
    scored = people.head(3).drop(columns=['churn'])
    people.drop(columns=[column], inplace=True)
    with pytest.raises(StructuredDataError) as error:
        PredictiveQuery(
            'PREDICT people.churn FOR EACH people.user_id', schema
        ).materialize(predict=scored)
    _refused(error, f'people.{column}')


def _shop_with_opening_times(backend: str) -> Any:
    r"""A source over the shop whose users have a ``TIME`` of opening."""
    users, orders = _shop()
    if backend == 'duckdb':
        connection = _connection({'users': users, 'orders': orders})
        connection.execute('ALTER TABLE users ADD COLUMN opens TIME')
        connection.execute(
            "UPDATE users SET opens = TIME '10:00:00' + "
            'INTERVAL (user_id % 3) MINUTE'
        )
        return DuckDBSource(connection)
    import os

    psycopg = pytest.importorskip('psycopg')
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        pytest.skip('no PostgreSQL to test against')
    from sdm_connectors import PostgresSource

    connection = psycopg.connect(dsn, autocommit=True)
    connection.execute('DROP SCHEMA IF EXISTS fixed_time_of_day CASCADE')
    connection.execute('CREATE SCHEMA fixed_time_of_day')
    connection.execute(
        'CREATE TABLE fixed_time_of_day.users (user_id BIGINT PRIMARY KEY, '
        'age BIGINT, opens TIME)'
    )
    connection.execute(
        'CREATE TABLE fixed_time_of_day.orders (order_id BIGINT PRIMARY KEY, '
        'user_id BIGINT, quantity BIGINT, ordered_at TIMESTAMP)'
    )
    with connection.cursor() as cursor:
        cursor.executemany(
            'INSERT INTO fixed_time_of_day.users VALUES (%s, %s, %s)',
            [
                (int(r.user_id), int(r.age), f'10:0{int(r.user_id) % 3}:00')
                for r in users.itertuples()
            ],
        )
        cursor.executemany(
            'INSERT INTO fixed_time_of_day.orders VALUES (%s, %s, %s, %s)',
            [
                (
                    int(r.order_id),
                    int(r.user_id),
                    int(r.quantity),
                    r.ordered_at.to_pydatetime(),
                )
                for r in orders.itertuples()
            ],
        )
    return PostgresSource(connection, schema='fixed_time_of_day')


@pytest.mark.parametrize('backend', ['duckdb', 'postgres'])
def test_a_time_of_day_is_text_and_not_the_time_column(backend):
    r"""A ``TIME`` holds a time of day, which no date goes with: today's took
    it for the table's time column and read every value of it as ``NaT``."""
    import json

    schema = Schema.from_source(
        _shop_with_opening_times(backend),
        primary_keys={'users': 'user_id', 'orders': 'order_id'},
    )
    assert schema['users'].time_column is None
    assert schema['users']['opens'].dtype == 'string'
    request = PredictiveQuery(QUESTION, schema).materialize(
        indices=[1, 2], num_context_samples=5, random_seed=7
    )
    sent = json.loads(json.dumps(request.payloads[0], default=str))
    users_sent = sent['predict']['related_tables']['users']
    at = users_sent['columns'].index('opens')
    assert {row[at] for row in users_sent['rows']} <= {
        '10:00:00',
        '10:01:00',
        '10:02:00',
    }, users_sent


def test_an_index_past_uint64_is_refused_by_value():
    users, orders = _shop()
    schema = Schema.from_data({'users': users, 'orders': orders})
    with pytest.raises(StructuredDataError) as error:
        PredictiveQuery(QUESTION, schema).materialize(
            indices=[2**64], random_seed=7
        )
    _refused(error, str(2**64), "'users'")


def _stocked() -> dict[str, pd.DataFrame]:
    users, orders = _shop()
    stores = pd.DataFrame({'store_code': ['n', 's'], 'size': [3, 5]})
    orders['store_code'] = ['n', 's'] * 100
    stock = pd.DataFrame(
        {
            'store_code': ['n', 'n', 's', 's'],
            'item': [1, 2, 1, 2],
            'units': [5, 0, 2, 7],
        }
    )
    return {'users': users, 'orders': orders, 'stores': stores, 'stock': stock}


def test_a_composite_key_on_a_sql_source_materializes():
    r"""The key's derived column is the client's own, so the SQL backends'
    reserved prefix does not refuse it (today's did, on every SQL source)."""
    connection = _connection(_stocked())
    schema = Schema.from_source(
        DuckDBSource(connection),
        primary_keys={
            'users': 'user_id',
            'orders': 'order_id',
            'stores': 'store_code',
            'stock': ('store_code', 'item'),
        },
    )
    request = PredictiveQuery(QUESTION, schema).materialize(
        indices=[1, 2], random_seed=7
    )
    assert request.payloads


def test_an_unreadable_earliest_time_on_sqlite_is_dropped_as_the_others_are():
    r"""An unreadable text time is dropped wherever it is; as the earliest
    or the latest, today's failed the whole request instead."""
    import sqlite3

    from sdm_connectors import SQLiteSource

    dbapi = pytest.importorskip('adbc_driver_sqlite.dbapi')
    users, orders = _shop()
    orders['ordered_at'] = orders['ordered_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
    orders.loc[0, 'ordered_at'] = ''
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'shop.db'
        with sqlite3.connect(path) as written:
            users.to_sql('users', written, index=False)
            orders.to_sql('orders', written, index=False)
        connection = dbapi.connect(str(path))
        try:
            schema = Schema.from_source(
                SQLiteSource(connection),
                primary_keys={'users': 'user_id', 'orders': 'order_id'},
                time_columns={'orders': 'ordered_at'},
            )
            request = PredictiveQuery(QUESTION, schema).materialize(
                indices=[1, 2], random_seed=7
            )
        finally:
            connection.close()
    assert request.payloads


def test_a_null_foreign_key_on_a_sql_source_joins_no_row():
    r"""An order with no user has no user in its subgraph; today's read the
    null's code, -1, as the last user looked up, and joined that one."""
    import json

    users, orders = _shop()
    orders['user_id'] = pd.array(
        [None if i % 4 == 0 else 1 + i % 20 for i in range(200)],
        dtype='Int64',
    )
    connection = _connection({'users': users, 'orders': orders})
    schema = Schema.from_source(
        DuckDBSource(connection),
        primary_keys={'users': 'user_id', 'orders': 'order_id'},
    )
    request = PredictiveQuery(
        'PREDICT orders.quantity FOR EACH orders.order_id', schema
    ).materialize(indices=[1, 2, 5], random_seed=7, num_context_samples=8)
    payload = json.loads(json.dumps(request.payloads[0], default=str))
    checked = 0
    for part in ('context', 'predict'):
        tables = payload[part]
        entity = {row[0]: row[1] for row in tables['instance_table']['rows']}
        orders_rows = tables['related_tables']['orders']
        at = orders_rows['columns'].index('user_id')
        keyless = {
            row[0]
            for row in orders_rows['rows']
            if entity.get(row[0]) == row[1] and row[at] is None
        }
        with_a_user = {
            row[0] for row in tables['related_tables']['users']['rows']
        }
        assert not keyless & with_a_user, (part, keyless & with_a_user)
        checked += len(keyless)
    assert checked, 'no example has an order without a user'


def test_a_list_of_a_key_linked_to_two_tables_is_refused_by_name():
    r"""``LIST_DISTINCT`` of a key ranks the rows of the one table it links
    to; linked to two, today's ended in the engine's NotImplementedError,
    as INTERNAL_ERROR."""
    linked = ('products', 'catalog')
    from sdm_client import Relationship

    users, orders = _shop()
    orders['product_id'] = [1 + i % 6 for i in range(200)]
    products = pd.DataFrame({'product_id': range(1, 7), 'price': range(6)})
    catalog = pd.DataFrame({'sku': range(1, 7), 'aisle': list('abcabc')})
    schema = Schema.from_data(
        {
            'users': users,
            'orders': orders,
            'products': products,
            'catalog': catalog,
        },
        primary_keys={'catalog': 'sku'},
        relationships=[
            Relationship('orders', 'user_id', 'users', 'user_id'),
            *(
                Relationship(
                    'orders',
                    'product_id',
                    table,
                    'sku' if table == 'catalog' else 'product_id',
                )
                for table in linked
            ),
        ],
    )
    with pytest.raises(StructuredDataError) as error:
        PredictiveQuery(
            'PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) '
            'FOR EACH users.user_id',
            schema,
        ).materialize(indices=[1, 2, 3], random_seed=7)
    assert error.value.code == 'UNSUPPORTED_REQUEST', str(error.value)
    _refused(error, 'orders.product_id', *linked)


def test_decimal_entity_keys_on_a_sql_source_materialize():
    r"""A whole ``DECIMAL`` key is an integer key; as the entity of a
    question, today's engine refused its task table's data type, as
    INTERNAL_ERROR."""
    users, orders = _shop()
    connection = _connection({'users': users, 'orders': orders})
    for table, column in (('users', 'user_id'), ('orders', 'user_id')):
        connection.execute(
            f'ALTER TABLE {table} ALTER {column} TYPE DECIMAL(10, 0)'
        )
    schema = Schema.from_source(
        DuckDBSource(connection),
        primary_keys={'users': 'user_id', 'orders': 'order_id'},
    )
    request = PredictiveQuery(QUESTION, schema).materialize(
        indices=[1, 2, 3], num_context_samples=10, random_seed=7
    )
    assert request.payloads


def test_a_frame_column_named_like_a_derived_key_is_refused_by_name():
    r"""A composite key is folded into a column named ``__kumo_key_...``;
    a frame column of that name made today's gather fail with "Primary key
    ... must have 'ID' semantic type", of a key the caller never named."""
    frames = _stocked()
    derived = '__kumo_key_' + '_'.join(
        name.encode().hex() for name in ('store_code', 'item')
    )
    frames['stock'][derived] = range(len(frames['stock']))
    schema = Schema.from_data(
        frames,
        primary_keys={
            'users': 'user_id',
            'orders': 'order_id',
            'stores': 'store_code',
            'stock': ('store_code', 'item'),
        },
    )
    with pytest.raises(StructuredDataError) as error:
        PredictiveQuery(QUESTION, schema).materialize(
            indices=[1, 2], random_seed=7
        )
    _refused(error, f'stock.{derived}')
    assert 'semantic type' not in str(error.value)


def test_uint64_keys_past_int64_are_sent_as_the_text_the_schema_says():
    r"""``Schema.from_data`` calls a ``uint64`` key holding 2**63 or more
    ``string``, since no wire integer holds it; today's then sent the key as
    a number and failed ("outside the range int64 represents")."""
    import json

    import numpy as np

    base = 2**63
    users, orders = _shop()
    users['user_id'] = np.array([base + i for i in range(20)], dtype='uint64')
    orders['user_id'] = np.array(
        [base + i % 20 for i in range(200)], dtype='uint64'
    )
    schema = Schema.from_data({'users': users, 'orders': orders})
    assert schema['users']['user_id'].dtype == 'string'
    request = PredictiveQuery(
        'PREDICT SUM(orders.quantity, 0, 30, days) FOR EACH users.user_id',
        schema,
    ).materialize(
        indices=[base + 1, base + 2], num_context_samples=5, random_seed=7
    )
    predict = json.loads(json.dumps(request.payloads[0], default=str))[
        'predict'
    ]
    assert [row[1] for row in predict['instance_table']['rows']] == [
        str(base + 1),
        str(base + 2),
    ]


# Phase 8.5's notes for Phase 9.


def test_a_key_error_the_core_raises_is_rebuilt_from_its_repr():
    r"""The core says a ``KeyError`` as today's ``str(error)``, the repr of
    its key; the wrapper raises it with the key itself, not its repr."""
    from sdm_client import _core

    class Raised(Exception):
        error_class = 'builtins.KeyError'
        code = None
        message = "'orders'"
        details: dict[str, Any] = {}
        status_code = None
        warnings: list[Any] = []
        chain: list[Any] = []

    made = _core.error_of(Raised())
    assert type(made) is KeyError
    assert made.args == ('orders',)
    assert str(made) == "'orders'"


def test_schema_tables_stays_the_live_dict_it_was():
    r"""Phase 9 keeps ``Schema.tables`` the dict the schema reads at each
    call, as today's: a table taken out of it is gone from the next gather,
    and the relationship that named it is refused by name."""
    users, orders = _shop()
    extra = pd.DataFrame({'extra_id': range(5), 'user_id': [1, 2, 3, 4, 5]})
    schema = Schema.from_data(
        {'users': users, 'orders': orders, 'extra': extra}
    )
    del schema.tables['extra']
    with pytest.raises(StructuredDataError, match="no table 'extra'"):
        PredictiveQuery(QUESTION, schema).materialize(
            indices=[1, 2], random_seed=7
        )
    schema.relationships = [
        r for r in schema.relationships if r.source_table != 'extra'
    ]
    request = PredictiveQuery(QUESTION, schema).materialize(
        indices=[1, 2], random_seed=7
    )
    assert 'extra' not in request.payloads[0]['context']['related_tables']


@pytest.mark.skipif(
    __import__('numpy').dtype('longdouble').itemsize <= 8,
    reason='a long double is a float64 on this platform',
)
def test_a_long_double_column_is_sent_as_float64():
    r"""On Linux x86-64 a long double is 80 bits, which Arrow has no type
    for; it is read as float64, as ``Schema.from_data`` says it is."""
    import numpy as np

    users, orders = _shop()
    users['score'] = np.arange(20, dtype=np.longdouble) / 4
    schema = Schema.from_data({'users': users, 'orders': orders})
    assert schema['users']['score'].dtype == 'float64'
    request = PredictiveQuery(QUESTION, schema).materialize(
        indices=[1, 2], random_seed=7
    )
    users_sent = request.payloads[0]['context']['related_tables']['users']
    at = users_sent['columns'].index('score')
    assert {type(row[at]) for row in users_sent['rows']} == {float}


@pytest.mark.parametrize('kind', ['bool8', 'json_', 'uuid'])
def test_extension_descriptor_keeps_the_actual_inference_primitive(kind):
    import json

    pa = pytest.importorskip('pyarrow')
    if not hasattr(pa, kind):
        pytest.skip('this Arrow release has no requested extension')
    values = {
        'bool8': [1, None, 0],
        'json_': ['{"value":1}', None, '{"value":2}'],
        'uuid': [bytes(16), None, bytes([1]) * 16],
    }[kind]
    arrow_type = getattr(pa, kind)()
    series = pd.Series(
        pa.array(values, type=arrow_type), dtype=pd.ArrowDtype(arrow_type)
    )
    from sdm_core.describe import describe_frames

    tables, encoded = describe_frames(
        {'items': pd.DataFrame({'item_id': series})}
    )
    tag = json.loads(encoded)['items']['item_id']
    assert tables['items'].num_rows == len(series)
    assert tag['dtype'] == str(series.dtype)
    try:
        content = pd.api.types.infer_dtype(series, skipna=True)
    except NotImplementedError as error:
        assert tag['infer_dtype_error'] == str(error)
        assert 'content' not in tag
        expected = NotImplementedError
        message = str(error)
    else:
        assert tag['content'] == content
        assert 'infer_dtype_error' not in tag
        expected = pa.ArrowNotImplementedError
        message = f"Function 'unique' has no kernel matching input types ({arrow_type})"
    with pytest.raises(expected) as raised:
        Schema.from_data({'items': pd.DataFrame({'item_id': series})})
    assert str(raised.value) == message


@pytest.mark.parametrize(
    'kind',
    [
        'list',
        'large_list',
        'fixed_list',
        'nested_list',
        'map',
        'sorted_map',
        'struct',
        'halffloat',
        'bool8',
        'json_',
        'uuid',
    ],
)
def test_declared_arrow_foreign_key_uses_unique_before_gather(kind):
    pa = pytest.importorskip('pyarrow')
    if kind in ('bool8', 'json_', 'uuid') and not hasattr(pa, kind):
        pytest.skip('this Arrow release has no requested extension')
    constructors = {
        'list': lambda: (pa.list_(pa.int64()), [1, 2]),
        'large_list': lambda: (pa.large_list(pa.int64()), [1, 2]),
        'fixed_list': lambda: (pa.list_(pa.int64(), 2), [1, 2]),
        'nested_list': lambda: (pa.list_(pa.list_(pa.int64())), [[1, 2]]),
        'map': lambda: (pa.map_(pa.string(), pa.int64()), [('value', 1)]),
        'sorted_map': lambda: (
            pa.map_(pa.string(), pa.int64(), keys_sorted=True),
            [('value', 1)],
        ),
        'struct': lambda: (
            pa.struct([('number', pa.int64()), ('label', pa.string())]),
            {'number': 1, 'label': 'one'},
        ),
        'halffloat': lambda: (pa.float16(), 1.0),
        'bool8': lambda: (pa.bool8(), 1),
        'json_': lambda: (pa.json_(), '{"value":1}'),
        'uuid': lambda: (pa.uuid(), bytes(16)),
    }
    users, orders = _shop()
    schema = Schema.from_data({'users': users, 'orders': orders})
    arrow_type, value = constructors[kind]()
    orders = orders.copy()
    orders['user_id'] = pd.Series(
        pa.array([value] * len(orders), type=arrow_type),
        dtype=pd.ArrowDtype(arrow_type),
    )
    schema['orders'].data = orders
    with pytest.raises(pa.ArrowNotImplementedError) as raised:
        PredictiveQuery(QUESTION, schema).materialize(
            indices=[1, 2], random_seed=7
        )
    assert (
        str(raised.value)
        == f"Function 'unique' has no kernel matching input types ({arrow_type})"
    )
