# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public DuckDB schema inference, keys, relationships and source naming.
Fixed policy expectations cover columns, source names, keys, times, links
and semantic flags without installing the retired Python engine.
Historical frozen-client comparisons remain in the external archive.
"""

from __future__ import annotations

import ast
import json
import re
import warnings
from pathlib import Path
from typing import Any

import duckdb
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import Relationship, Schema, StructuredDataError
from sdm_client._rust import providers
from sdm_client.schema import _described_on_the_core

ROOT = Path(__file__).resolve().parents[3]
PROVIDERS = Path(providers.__file__).resolve().parent


def _shop() -> list[str]:
    return json.loads((Path(__file__).parent / 'data/shop.json').read_text())[
        'statements'
    ]


def _connection(*setup: str) -> Any:
    connection = duckdb.connect()
    for statement in [*_shop(), *setup]:
        connection.execute(statement)
    return connection


def _projection(schema: Schema) -> dict[str, Any]:
    """The public schema as the contract runner projects it."""
    return {
        'tables': [
            {
                'name': name,
                'columns': [
                    [column, c.dtype, c.stype]
                    for column, c in table.columns.items()
                ],
                'primary_key': table.primary_key,
                'time_column': table.time_column,
                'duplicate_keys': table.duplicate_keys,
            }
            for name, table in schema.tables.items()
        ],
        'links': [
            [
                link.source_table,
                link.source_column,
                link.target_table,
                link.target_column,
            ]
            for link in schema.relationships
        ],
    }


REFERENCE = {
    'tables': [
        {
            'name': 'orders',
            'columns': [
                ['order_id', 'int64', 'ID'],
                ['user_id', 'int64', 'ID'],
                ['price', 'float64', 'numerical'],
                ['ts', 'timestamp[us]', 'timestamp'],
            ],
            'primary_key': 'order_id',
            'time_column': 'ts',
            'duplicate_keys': 0,
        },
        {
            'name': 'users',
            'columns': [
                ['user_id', 'int64', 'ID'],
                ['age', 'int64', 'numerical'],
                ['country', 'string', 'categorical'],
            ],
            'primary_key': 'user_id',
            'time_column': None,
            'duplicate_keys': 0,
        },
    ],
    'links': [['orders', 'user_id', 'users', 'user_id']],
}


def test_named_table_keeps_its_source_name() -> None:
    schema = Schema.from_source(
        DuckDBSource(_connection()), {'people': 'users'}
    )
    assert list(schema.tables) == ['people']
    assert schema['people'].source_name == 'users'
    schema = Schema.from_source(
        DuckDBSource(_connection()),
        {'people': 'users'},
        primary_keys={'people': 'user_id'},
    )
    assert schema['people'].primary_key == 'user_id'
    assert [
        (c, col.dtype, col.stype) for c, col in schema['people'].columns.items()
    ] == [
        ('user_id', 'int64', 'ID'),
        ('age', 'int64', 'numerical'),
        ('country', 'string', 'categorical'),
    ]
    assert schema.relationships == []
    assert _described_on_the_core(schema)


def test_base_regression_s_schema_on_the_core_is_the_reference_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _connection()
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        schema = Schema.from_source(DuckDBSource(connection))
    assert _projection(schema) == REFERENCE
    assert [type(w.message).__name__ for w in shown] == []
    assert schema.source is not None
    unchecked = Schema.from_source(DuckDBSource(connection), check_keys=False)
    assert [t.duplicate_keys for t in unchecked.tables.values()] == [None, None]


def test_what_the_core_does_not_describe_yet_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refusals = [
        (
            lambda: _connection('CREATE TABLE z (z_id BIGINT)'),
            {},
            'empty_table',
            'INVALID_REQUEST',
        ),
        (
            lambda: _connection(
                "CREATE OR REPLACE TABLE users AS SELECT *, {'a': 1} AS s FROM users"
            ),
            {},
            'column_type',
            'UNSUPPORTED_REQUEST',
        ),
    ]
    for build, arguments, rule, code in refusals:
        with pytest.raises(StructuredDataError) as refused:
            Schema.from_source(DuckDBSource(build()), **arguments)
        assert refused.value.code == code
        assert refused.value.details['rule'] == rule
    schema = Schema.from_source(
        DuckDBSource(_connection()),
        primary_keys={'orders': ['order_id', 'user_id']},
    )
    assert tuple(schema['orders'].primary_key) == ('order_id', 'user_id')


def test_the_bridge_refuses_none_of_the_call_s_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = Schema.from_source(
        DuckDBSource(_connection()),
        relationships=[],
        time_columns={'orders': None},
    )
    assert schema.relationships == []
    assert schema['orders'].time_column is None
    assert schema['orders']['ts'].stype == 'timestamp'
    with pytest.raises(StructuredDataError) as refused:
        Schema.from_source(
            DuckDBSource(_connection()),
            relationships=[('orders', 'user_id', 'users', 'user_id')],
        )
    assert refused.value.code == 'INVALID_REQUEST'
    assert 'must be a Relationship' in str(refused.value)


SCHEMA_RULES = {
    'edit stamps': (
        {'TIME_COLUMNS_GUESSED'},
        (
            'CREATE TABLE logs (log_id BIGINT, user_id BIGINT, updated_at TIMESTAMP)',
            "INSERT INTO logs SELECT i, i % 3, TIMESTAMP '2024-01-01' + to_days(CAST(i AS INTEGER)) FROM range(9) t(i)",
            'CREATE TABLE plans (plan_id BIGINT, modified_at TIMESTAMP)',
            "INSERT INTO plans SELECT i, TIMESTAMP '2024-01-01' + to_days(CAST(i AS INTEGER)) FROM range(4) t(i)",
            'CREATE TABLE subscriptions (sub_id BIGINT, plan_id BIGINT, _loaded_at TIMESTAMP, birth_date DATE, started TIMESTAMP)',
            "INSERT INTO subscriptions SELECT i, i % 4, TIMESTAMP '2024-02-01' + to_days(CAST(i AS INTEGER)), DATE '1990-01-01' + CAST(i AS INTEGER), TIMESTAMP '2023-01-01' + to_days(CAST(i AS INTEGER)) FROM range(8) t(i)",
        ),
    ),
    'links': (
        {'LINK_CANNOT_JOIN', 'TWO_WAYS_TO_A_TABLE', 'TABLES_NAME_EACH_OTHER'},
        (
            'CREATE TABLE accounts (account_id BIGINT PRIMARY KEY, refund_id BIGINT)',
            'INSERT INTO accounts SELECT i, i FROM range(4) t(i)',
            'CREATE TABLE refunds (refund_id BIGINT PRIMARY KEY, account_id BIGINT, payer BIGINT REFERENCES accounts (account_id))',
            'INSERT INTO refunds SELECT i, i % 4, (i + 1) % 4 FROM range(6) t(i)',
            'CREATE TABLE notes (note_id BIGINT, account_id VARCHAR)',
            "INSERT INTO notes SELECT i, 'a' || chr(97 + CAST(i AS INTEGER)) FROM range(5) t(i)",
            'CREATE TABLE items (item_id BIGINT PRIMARY KEY, sku VARCHAR UNIQUE)',
            "INSERT INTO items SELECT i, 's' || chr(97 + CAST(i AS INTEGER)) FROM range(5) t(i)",
            'CREATE TABLE lines (line_id BIGINT, sku VARCHAR REFERENCES items (sku))',
            "INSERT INTO lines SELECT i, 's' || chr(97 + CAST(i % 5 AS INTEGER)) FROM range(9) t(i)",
        ),
    ),
    'names a remedy quotes': (
        {'TIME_COLUMNS_GUESSED'},
        (
            *(
                statement
                for name in ("odd'name", 'back\\slash', 'tab\there')
                for statement in (
                    f'CREATE TABLE "{name}" (id BIGINT, updated_at TIMESTAMP)',
                    f'''INSERT INTO "{name}" SELECT i, TIMESTAMP '2024-01-01' + to_days(CAST(i AS INTEGER)) FROM range(5) t(i)''',
                )
            ),
            'CREATE TABLE plans (plan_id BIGINT, "modified ""x""" TIMESTAMP, "changed \'a\' ""b""" TIMESTAMP)',
            "INSERT INTO plans SELECT i, TIMESTAMP '2024-01-01' + to_days(CAST(i AS INTEGER)), TIMESTAMP '2024-02-01' + to_days(CAST(i AS INTEGER)) FROM range(4) t(i)",
            'CREATE TABLE subs (sub_id BIGINT, plan_id BIGINT)',
            'INSERT INTO subs SELECT i, i % 4 FROM range(8) t(i)',
        ),
    ),
    'time values': (
        {'TIME_COLUMNS_GUESSED'},
        (
            'CREATE TABLE plans (plan_id BIGINT, stamped TIMESTAMP, ended DATE, gone TIMESTAMP)',
            "INSERT INTO plans SELECT i, TIMESTAMP '2024-01-01 10:30:00.25', CASE WHEN i % 4 = 0 THEN DATE '2024-03-01' + CAST(i AS INTEGER) END, NULL FROM range(9) t(i)",
            'CREATE TABLE subs (sub_id BIGINT, plan_id BIGINT)',
            'INSERT INTO subs SELECT i, i % 9 FROM range(12) t(i)',
        ),
    ),
    'several timestamps': (
        {'TIME_COLUMNS_GUESSED'},
        (
            'CREATE TABLE shipments (shipment_id BIGINT, user_id BIGINT, shipped TIMESTAMP, ordered DATE)',
            "INSERT INTO shipments SELECT i, i % 8, TIMESTAMP '2023-02-01' + to_days(CAST(i AS INTEGER)), DATE '2023-01-01' + CAST(i AS INTEGER) FROM range(9) t(i)",
            'CREATE TABLE visits (visit_id BIGINT, user_id BIGINT, seen TIMESTAMP, created_at TIMESTAMP)',
            "INSERT INTO visits SELECT i, i % 8, TIMESTAMP '2020-01-01' + to_days(CAST(i AS INTEGER)), TIMESTAMP '2024-01-01' + to_days(CAST(i AS INTEGER)) FROM range(9) t(i)",
        ),
    ),
    'a time of day': (
        {'TIME_VALUES_UNREAD', 'TIME_COLUMNS_GUESSED'},
        (
            'CREATE OR REPLACE TABLE users AS SELECT *, make_time(user_id::BIGINT, 30, 0) AS f FROM users',
        ),
    ),
    'a list left out': (
        {'COLUMNS_LEFT_OUT'},
        (
            "CREATE OR REPLACE TABLE users AS SELECT *, [user_id, 2]::INTEGER[] AS f, ['a', country] AS g FROM users",
        ),
    ),
    'more reasons': (
        {'TIME_COLUMNS_GUESSED', 'LINK_CANNOT_JOIN'},
        (
            'CREATE TABLE people (person_id BIGINT, birth_date TIMESTAMP, stamped TIMESTAMP)',
            "INSERT INTO people SELECT i, TIMESTAMP '1980-01-01' + to_days(CAST(i AS INTEGER)), CASE WHEN i % 3 = 0 THEN NULL ELSE TIMESTAMP '2024-01-01 08:00:00' END FROM range(9) t(i)",
            'CREATE TABLE visits (visit_id BIGINT, person_id BIGINT)',
            'INSERT INTO visits SELECT i, i % 9 FROM range(12) t(i)',
            'CREATE TABLE codes (u1 BIGINT UNIQUE, u2 BIGINT UNIQUE)',
            'INSERT INTO codes SELECT i, i + 10 FROM range(5) t(i)',
            'CREATE TABLE coded (coded_id BIGINT, c1 BIGINT REFERENCES codes (u1), c2 BIGINT REFERENCES codes (u2))',
            'INSERT INTO coded SELECT i, i % 5, i % 5 + 10 FROM range(7) t(i)',
        ),
    ),
    'a key by reference': (
        set(),
        (
            'CREATE TABLE codes (tag VARCHAR UNIQUE, label VARCHAR UNIQUE)',
            "INSERT INTO codes SELECT 't' || chr(97 + CAST(i AS INTEGER)), 'l' || chr(97 + CAST(i AS INTEGER)) FROM range(5) t(i)",
            'CREATE TABLE coded (coded_id BIGINT, tag VARCHAR REFERENCES codes (tag))',
            "INSERT INTO coded SELECT i, 't' || chr(97 + CAST(i % 5 AS INTEGER)) FROM range(9) t(i)",
        ),
    ),
}


def _said(shown: list[Any]) -> list[Any]:
    """The warnings a call said, without a ``ResourceWarning`` the
    collector raised meanwhile for another test's unclosed connection,
    which ``simplefilter('always')`` lets in at any point (T1.23 fix 1)."""
    return [w for w in shown if not issubclass(w.category, ResourceWarning)]


def _described(
    setup: tuple[str, ...],
) -> tuple[dict[str, Any], list[tuple[str, str]], Schema]:
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        schema = Schema.from_source(DuckDBSource(_connection(*setup)))
    told = _said(shown)
    said = [(type(w.message).__name__, str(w.message)) for w in told]
    return (_projection(schema), said, schema)


def _before_key(schema: Schema) -> dict[tuple[str, str], str]:
    return {
        (table, column): c._stype_before_key
        for table, shaped in schema.tables.items()
        for column, c in shaped.columns.items()
        if c._stype_before_key is not None
    }


def _declared(schema: Schema) -> dict[tuple[str, str], bool]:
    """Each column's ``stype_declared``: a time column, or one inference
    took as it before the links refused it, reads declared on both routes
    (subagent P3-2, T1.21d's review)."""
    return {
        (table, column): c.stype_declared
        for table, shaped in schema.tables.items()
        for column, c in shaped.columns.items()
    }


def _expected_table(key, time, *columns):
    return {'key': key, 'time': time, 'columns': list(columns)}


def _ids(*names):
    return [(name, 'int64', 'ID') for name in names]


def _times(*names):
    return [(name, 'timestamp[us]', 'timestamp') for name in names]


# Expectations describe the authored SQL fixtures and schema policy explicitly.
# Helpers only construct literal tuples; none inspect candidate output.
SCHEMA_EXPECTED = {
    'edit stamps': {
        'logs': _expected_table(
            'log_id',
            'updated_at',
            *_ids('log_id', 'user_id'),
            *_times('updated_at'),
        ),
        'plans': _expected_table(
            'plan_id', None, *_ids('plan_id'), *_times('modified_at')
        ),
        'subscriptions': _expected_table(
            None,
            'started',
            *_ids('sub_id', 'plan_id'),
            *_times('_loaded_at', 'birth_date', 'started'),
        ),
    },
    'links': {
        'accounts': _expected_table(
            'account_id', None, *_ids('account_id', 'refund_id')
        ),
        'refunds': _expected_table(
            'refund_id', None, *_ids('refund_id', 'account_id', 'payer')
        ),
        'notes': _expected_table(
            'note_id', None, *_ids('note_id'), ('account_id', 'string', 'ID')
        ),
        'items': _expected_table(
            'item_id', None, *_ids('item_id'), ('sku', 'string', 'categorical')
        ),
        'lines': _expected_table(
            'line_id', None, *_ids('line_id'), ('sku', 'string', 'categorical')
        ),
    },
    'names a remedy quotes': {
        **{
            name: _expected_table(
                'id', 'updated_at', *_ids('id'), *_times('updated_at')
            )
            for name in ("odd'name", 'back\\slash', 'tab\there')
        },
        'plans': _expected_table(
            'plan_id',
            None,
            *_ids('plan_id'),
            *_times('modified "x"', 'changed \'a\' "b"'),
        ),
        'subs': _expected_table('sub_id', None, *_ids('sub_id', 'plan_id')),
    },
    'time values': {
        'plans': _expected_table(
            'plan_id',
            None,
            *_ids('plan_id'),
            *_times('stamped', 'ended', 'gone'),
        ),
        'subs': _expected_table('sub_id', None, *_ids('sub_id', 'plan_id')),
    },
    'several timestamps': {
        'shipments': _expected_table(
            'shipment_id',
            'ordered',
            *_ids('shipment_id', 'user_id'),
            *_times('shipped', 'ordered'),
        ),
        'visits': _expected_table(
            'visit_id',
            'created_at',
            *_ids('visit_id', 'user_id'),
            *_times('seen', 'created_at'),
        ),
    },
    'a time of day': {},
    'a list left out': {},
    'more reasons': {
        'people': _expected_table(
            None, None, *_ids('person_id'), *_times('birth_date', 'stamped')
        ),
        'visits': _expected_table(
            'visit_id', None, *_ids('visit_id', 'person_id')
        ),
        'codes': _expected_table(
            None,
            None,
            ('u1', 'int64', 'categorical'),
            ('u2', 'int64', 'categorical'),
        ),
        'coded': _expected_table(
            'coded_id',
            None,
            *_ids('coded_id'),
            ('c1', 'int64', 'categorical'),
            ('c2', 'int64', 'categorical'),
        ),
    },
    'a key by reference': {
        'codes': _expected_table(
            'tag',
            None,
            ('tag', 'string', 'ID'),
            ('label', 'string', 'categorical'),
        ),
        'coded': _expected_table(
            'coded_id', None, *_ids('coded_id'), ('tag', 'string', 'ID')
        ),
    },
}

SCHEMA_LINKS = {
    'edit stamps': [
        ('logs', 'user_id', 'users', 'user_id'),
        ('orders', 'user_id', 'users', 'user_id'),
        ('subscriptions', 'plan_id', 'plans', 'plan_id'),
    ],
    'links': [
        ('refunds', 'payer', 'accounts', 'account_id'),
        ('accounts', 'refund_id', 'refunds', 'refund_id'),
        ('orders', 'user_id', 'users', 'user_id'),
        ('refunds', 'account_id', 'accounts', 'account_id'),
    ],
    'names a remedy quotes': [
        ('orders', 'user_id', 'users', 'user_id'),
        ('subs', 'plan_id', 'plans', 'plan_id'),
    ],
    'time values': [
        ('orders', 'user_id', 'users', 'user_id'),
        ('subs', 'plan_id', 'plans', 'plan_id'),
    ],
    'several timestamps': [
        ('orders', 'user_id', 'users', 'user_id'),
        ('shipments', 'user_id', 'users', 'user_id'),
        ('visits', 'user_id', 'users', 'user_id'),
    ],
    'a key by reference': [
        ('coded', 'tag', 'codes', 'tag'),
        ('orders', 'user_id', 'users', 'user_id'),
    ],
}


SCHEMA_DECLARED = {
    'edit stamps': {
        ('orders', 'ts'),
        ('logs', 'updated_at'),
        ('plans', 'modified_at'),
        ('subscriptions', 'started'),
    },
    'names a remedy quotes': {
        ('orders', 'ts'),
        *((n, 'updated_at') for n in ("odd'name", 'back\\slash', 'tab\there')),
    },
    'several timestamps': {
        ('orders', 'ts'),
        ('shipments', 'ordered'),
        ('visits', 'created_at'),
    },
}


def _assert_schema_fixture(name, schema):
    expected = {
        table['name']: _expected_table(
            table['primary_key'],
            table['time_column'],
            *(tuple(c) for c in table['columns']),
        )
        for table in REFERENCE['tables']
    }
    expected.update(SCHEMA_EXPECTED[name])
    if name == 'a time of day':
        # Time-of-day values normalize to null timestamps, not a selected time column.
        expected['users']['columns'].append(('f', 'timestamp[us]', 'timestamp'))
    actual = {
        name: _expected_table(
            t.primary_key,
            t.time_column,
            *((c, v.dtype, v.stype) for c, v in t.columns.items()),
        )
        for name, t in schema.tables.items()
    }
    assert actual == expected
    assert {
        k for k, v in _declared(schema).items() if v
    } == SCHEMA_DECLARED.get(name, {('orders', 'ts')})
    expected_before = {
        'links': {('refunds', 'payer'): 'categorical'},
        'a key by reference': {
            ('coded', 'tag'): 'categorical',
            ('codes', 'tag'): 'categorical',
        },
    }
    assert _before_key(schema) == expected_before.get(name, {})
    assert {n: t.source_name for n, t in schema.tables.items()} == {
        n: n for n in expected
    }
    assert [
        tuple(link) for link in _projection(schema)['links']
    ] == SCHEMA_LINKS.get(name, [('orders', 'user_id', 'users', 'user_id')])


@pytest.mark.parametrize('name', sorted(SCHEMA_RULES))
def test_the_core_assembles_the_schema_as_the_old_route_does(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    codes, setup = SCHEMA_RULES[name]
    core, said, schema = _described(setup)
    _assert_schema_fixture(name, schema)
    given = [text.split(']', 1)[0].lstrip('[') for _, text in said]
    assert set(given) == codes, given
    if name == 'edit stamps':
        assert schema['plans'].time_column is None
        assert schema['plans']['modified_at'].stype_declared is True
    if name == 'names a remedy quotes':
        [(_, text)] = said
        remedies = re.findall('time_columns=(\\{[^\\n]*?\\})\\.(?:\\n|$)', text)
        assert len(remedies) == 5, text
        named = {
            (table, column)
            for remedy in remedies
            for table, column in ast.literal_eval(remedy).items()
        }
        assert named == {
            ("odd'name", None),
            ('back\\slash', None),
            ('tab\there', None),
            ('plans', 'modified "x"'),
            ('plans', 'changed \'a\' "b"'),
        }, text
    if name == 'a key by reference':
        assert schema['codes'].primary_key == 'tag'
        assert _before_key(schema) == {
            ('codes', 'tag'): 'categorical',
            ('coded', 'tag'): 'categorical',
        }
        edited = schema.snapshot()
        edited.relationships = []
        edited['codes'].primary_key = 'label'
        assert edited['codes']['tag'].stype == 'categorical'
        assert edited['coded']['tag'].stype == 'categorical'


def _described_with(name: str) -> Any:
    """``Schema.from_source`` over a DuckDB source of one table, ``name``,
    on ``routes`` (the old route when empty)."""
    quoted = '"' + name.replace('"', '""') + '"'
    setup = (
        f'CREATE TABLE {quoted} (id INTEGER PRIMARY KEY, label VARCHAR)',
        f"INSERT INTO {quoted} VALUES (1, 'a'), (2, 'b'), (3, 'c')",
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_source(DuckDBSource(_connection(*setup)))


def test_a_table_named_with_a_double_quote_is_described_on_the_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = 'double"quote'
    schema = _described_with(name)
    assert name in list(schema.tables)
    assert schema[name].primary_key == 'id'
    assert list(schema[name].columns) == ['id', 'label']


def test_a_table_named_with_a_dot_is_refused_on_both_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(StructuredDataError) as core:
        _described_with('a.b')
    assert core.value.code == 'INVALID_REQUEST'
    assert core.value.details['field'] == 'tables'


NAMED = {
    'a key moved': ((), {'primary_keys': {'users': 'age'}}),
    'a time column among several': (
        (
            'CREATE OR REPLACE TABLE orders AS SELECT *, ts + INTERVAL 1 DAY AS shipped FROM orders',
        ),
        {'time_columns': {'orders': 'shipped'}},
    ),
    'text as a time column': (
        (
            "CREATE OR REPLACE TABLE users AS SELECT *, 'n' || chr(97 + user_id::INTEGER) AS note FROM users",
        ),
        {'time_columns': {'users': 'note'}},
    ),
    'both': (
        (),
        {
            'primary_keys': {'orders': 'order_id', 'users': 'user_id'},
            'time_columns': {'orders': 'ts'},
        },
    ),
    'a time column cleared': ((), {'time_columns': {'orders': None}}),
    'a time column cleared among several': (
        (
            'CREATE OR REPLACE TABLE orders AS SELECT *, ts + INTERVAL 1 DAY AS shipped FROM orders',
        ),
        {'time_columns': {'orders': None}, 'primary_keys': {'users': 'age'}},
    ),
    'no relationships': ((), {'relationships': []}),
    'a relationship given': (
        (),
        {
            'relationships': [
                Relationship('orders', 'user_id', 'users', 'user_id')
            ]
        },
    ),
    'tables named': (
        (),
        {'tables': {'buyers': 'users', 'purchases': 'orders'}},
    ),
    'tables named, with everything': (
        (
            'CREATE OR REPLACE TABLE orders AS SELECT *, ts + INTERVAL 1 DAY AS shipped FROM orders',
        ),
        {
            'tables': {'purchases': 'orders', 'buyers': 'users'},
            'primary_keys': {'buyers': 'user_id'},
            'time_columns': {'purchases': 'shipped', 'buyers': None},
            'relationships': [
                Relationship('purchases', 'user_id', 'buyers', 'user_id')
            ],
        },
    ),
    'tables named, a subset': ((), {'tables': {'users': 'users'}}),
}


def _from_source(
    setup: tuple[str, ...], arguments: dict[str, Any]
) -> tuple[dict[str, Any], list[tuple[str, str]], Schema]:
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        schema = Schema.from_source(
            DuckDBSource(_connection(*setup)), **arguments
        )
    said = [
        (
            type(w.message).__name__,
            str(w.message).split('] ', 1)[1]
            if str(w.message).startswith('[')
            else str(w.message),
        )
        for w in _said(shown)
    ]
    return (_projection(schema), said, schema)


NAMED_EXPECTED = {
    'a key moved': (
        {
            'orders': ('orders', 'order_id', 'ts'),
            'users': ('users', 'age', None),
        },
        [],
    ),
    'a time column among several': (
        {
            'orders': ('orders', 'order_id', 'shipped'),
            'users': ('users', 'user_id', None),
        },
        [('orders', 'user_id', 'users', 'user_id')],
    ),
    'text as a time column': (
        {
            'orders': ('orders', 'order_id', 'ts'),
            'users': ('users', 'user_id', 'note'),
        },
        [('orders', 'user_id', 'users', 'user_id')],
    ),
    'both': (
        {
            'orders': ('orders', 'order_id', 'ts'),
            'users': ('users', 'user_id', None),
        },
        [('orders', 'user_id', 'users', 'user_id')],
    ),
    'a time column cleared': (
        {
            'orders': ('orders', 'order_id', None),
            'users': ('users', 'user_id', None),
        },
        [('orders', 'user_id', 'users', 'user_id')],
    ),
    'a time column cleared among several': (
        {
            'orders': ('orders', 'order_id', None),
            'users': ('users', 'age', None),
        },
        [],
    ),
    'no relationships': (
        {
            'orders': ('orders', 'order_id', 'ts'),
            'users': ('users', 'user_id', None),
        },
        [],
    ),
    'a relationship given': (
        {
            'orders': ('orders', 'order_id', 'ts'),
            'users': ('users', 'user_id', None),
        },
        [('orders', 'user_id', 'users', 'user_id')],
    ),
    'tables named': (
        {'buyers': ('users', None, None), 'purchases': ('orders', None, 'ts')},
        [],
    ),
    'tables named, with everything': (
        {
            'buyers': ('users', 'user_id', None),
            'purchases': ('orders', None, 'shipped'),
        },
        [('purchases', 'user_id', 'buyers', 'user_id')],
    ),
    'tables named, a subset': ({'users': ('users', 'user_id', None)}, []),
}


@pytest.mark.parametrize('name', sorted(NAMED))
def test_keys_and_time_columns_named_are_applied_as_the_old_route_does(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    setup, arguments = NAMED[name]
    core, said, schema = _from_source(setup, arguments)
    assert _described_on_the_core(schema)
    expected_tables, expected_links = NAMED_EXPECTED[name]
    assert {
        n: (t.source_name, t.primary_key, t.time_column)
        for n, t in schema.tables.items()
    } == expected_tables
    assert [tuple(link) for link in core['links']] == expected_links
    columns = {
        table['name']: [list(c) for c in table['columns']]
        for table in REFERENCE['tables']
    }
    if name in (
        'a time column among several',
        'a time column cleared among several',
        'tables named, with everything',
    ):
        columns['orders'].append(['shipped', 'timestamp[us]', 'timestamp'])
    if name == 'text as a time column':
        columns['users'].append(['note', 'string', 'timestamp'])
    if name in ('a key moved', 'a time column cleared among several'):
        columns['users'][1][2] = 'ID'
    assert {
        n: [[c, v.dtype, v.stype] for c, v in t.columns.items()]
        for n, t in schema.tables.items()
    } == {n: columns[source] for n, (source, _, _) in expected_tables.items()}
    declared = {
        (n, 'ts')
        for n, (source, _, _) in expected_tables.items()
        if source == 'orders'
    }
    if name in ('a time column among several', 'tables named, with everything'):
        declared.add(
            ('purchases' if name.startswith('tables') else 'orders', 'shipped')
        )
    if name == 'text as a time column':
        declared.add(('users', 'note'))
    assert {k for k, v in _declared(schema).items() if v} == declared
    assert _before_key(schema) == (
        {('users', 'age'): 'numerical'}
        if name in ('a key moved', 'a time column cleared among several')
        else {}
    )

    for table, key in arguments.get('primary_keys', {}).items():
        assert schema[table].primary_key == (
            key if isinstance(key, str) else tuple(key)
        )
        for column in [key] if isinstance(key, str) else key:
            assert schema[table][column].stype == 'ID'
    for table, column in arguments.get('time_columns', {}).items():
        assert schema[table].time_column == column
        if column is not None:
            assert schema[table][column].stype == 'timestamp'


@pytest.mark.parametrize(
    'arguments',
    [
        {'primary_keys': {'nowhere': 'id'}},
        {'primary_keys': {'users': 'nothing'}},
        {'primary_keys': {'orders': 'ts'}},
        {'time_columns': {'nowhere': 'ts'}},
        {'time_columns': {'orders': 'nothing'}},
        {'time_columns': {'orders': 'order_id'}},
        {'time_columns': {'orders': 'price'}},
        {'time_columns': {'nowhere': None}},
        {'relationships': [Relationship('orders', 'gone', 'users', 'user_id')]},
        {'relationships': [Relationship('orders', 'user_id', 'users', 'age')]},
        {'relationships': [Relationship('orders', 'ts', 'users', 'user_id')]},
        {
            'tables': {'buyers': 'users'},
            'relationships': [
                Relationship('orders', 'user_id', 'users', 'user_id')
            ],
        },
    ],
)
def test_a_key_or_time_column_named_wrongly_is_refused_as_the_old_route_does(
    monkeypatch: pytest.MonkeyPatch, arguments: dict[str, Any]
) -> None:
    with pytest.raises(StructuredDataError) as core:
        _from_source((), arguments)
    assert core.value.code == 'INVALID_REQUEST'


def _failing(provider: Any, table: str | None) -> Any:
    """``provider``, but failing, as a driver fails, the reading of
    ``table``'s columns: a provider's error the core did not cause. The
    request names the table as a relation of one part (contract v26)."""

    def answer(step: dict[str, Any], keys: Any) -> Any:
        request = step['request']
        if (
            table is not None
            and request.get('part') == 'columns'
            and (request.get('tables') == [{'source': [table]}])
        ):
            raise duckdb.IOException('the source went away')
        return provider(step, keys)

    return answer


def _named_on_the_core(
    connection: Any, tables: dict[str, str], fail_columns_of: str | None = None
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """``Schema.from_source`` over ``tables`` named, on the core route
    (T1.24), its provider failing the reading of ``fail_columns_of``'s
    columns when named."""
    from unittest import mock

    from sdm_client._rust import providers as bridge

    provider = bridge.DuckDBProvider
    with (
        mock.patch.object(
            bridge,
            'DuckDBProvider',
            lambda held: _failing(provider(held), fail_columns_of),
        ),
        warnings.catch_warnings(record=True) as shown,
    ):
        warnings.simplefilter('always')
        schema = Schema.from_source(DuckDBSource(connection), tables)
        assert _described_on_the_core(schema)
    said = [
        (type(w.message).__name__, str(w.message).split('] ', 1)[1])
        for w in _said(shown)
    ]
    projected = _projection(schema)
    projected['sources'] = {n: t.source_name for n, t in schema.tables.items()}
    return projected, said


REPEATED = {
    'exact': (
        (
            'CREATE TABLE parent (id BIGINT PRIMARY KEY)',
            'INSERT INTO parent SELECT i FROM range(4) t(i)',
            'CREATE TABLE child (child_id BIGINT PRIMARY KEY, parent_id BIGINT REFERENCES parent (id))',
            'INSERT INTO child SELECT i, i % 4 FROM range(9) t(i)',
        ),
        {'first': 'parent', 'second': 'parent', 'children': 'child'},
    ),
    'folded': (
        (
            'CREATE TABLE "Parent" (id BIGINT PRIMARY KEY)',
            'INSERT INTO "Parent" SELECT i FROM range(4) t(i)',
            'CREATE TABLE child (child_id BIGINT PRIMARY KEY, parent_id BIGINT REFERENCES parent (id))',
            'INSERT INTO child SELECT i, i % 4 FROM range(9) t(i)',
        ),
        {'p1': 'Parent', 'kids': 'child', 'p2': 'Parent'},
    ),
}


@pytest.mark.parametrize('name', sorted(REPEATED))
def test_a_source_table_named_twice_is_joined_as_the_old_route_joins_it(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    setup, tables = REPEATED[name]
    connection = duckdb.connect()
    for statement in setup:
        connection.execute(statement)
    core, said = _named_on_the_core(connection, tables)
    last = [n for n, s in tables.items() if s.lower() == 'parent'][-1]
    child = next(n for n, source in tables.items() if source == 'child')
    assert core['links'] == [[child, 'parent_id', last, 'id']]
    assert core['sources'] == tables
    assert {
        t['name']: (t['columns'], t['primary_key'], t['time_column'])
        for t in core['tables']
    } == {
        alias: (
            [['child_id', 'int64', 'ID'], ['parent_id', 'int64', 'ID']],
            'child_id',
            None,
        )
        if source == 'child'
        else ([['id', 'int64', 'ID']], 'id', None)
        for alias, source in tables.items()
    }


@pytest.mark.parametrize('alias', ['a.b', 'a`b', 'a\nb', 'a\rb'])
def test_a_name_the_schema_may_not_give_a_table_is_refused_as_the_old_route_does(
    monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    with pytest.raises(StructuredDataError) as core:
        _named_on_the_core(_connection(), {alias: 'users'})
    assert core.value.code == 'INVALID_REQUEST'
    assert core.value.details['field'] == 'tables'


def test_a_name_over_a_quoted_source_table_is_taken_as_the_old_route_takes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _connection(
        'CREATE TABLE "odd`name" (thing_id BIGINT PRIMARY KEY, label VARCHAR)',
        "INSERT INTO \"odd`name\" VALUES (1, 'a'), (2, 'b'), (3, 'c')",
    )
    core, said = _named_on_the_core(connection, {'things': 'odd`name'})
    assert core == {
        'tables': [
            {
                'name': 'things',
                'columns': [
                    ['thing_id', 'int64', 'ID'],
                    ['label', 'string', 'categorical'],
                ],
                'primary_key': 'thing_id',
                'time_column': None,
                'duplicate_keys': 0,
            }
        ],
        'links': [],
        'sources': {'things': 'odd`name'},
    }


def test_a_source_table_that_is_not_there_is_refused_as_the_old_route_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(StructuredDataError) as core:
        _named_on_the_core(_connection(), {'x': 'nope'})
    assert core.value.code == 'INVALID_REQUEST'
    assert core.value.details['field'] == 'tables'
    core_schema, _ = _named_on_the_core(_connection(), {'x': 'USERS'})
    assert core_schema == {
        'tables': [
            {
                'name': 'x',
                'columns': [
                    ['user_id', 'int64', 'ID'],
                    ['age', 'int64', 'numerical'],
                    ['country', 'string', 'categorical'],
                ],
                'primary_key': None,
                'time_column': None,
                'duplicate_keys': 0,
            }
        ],
        'links': [],
        'sources': {'x': 'USERS'},
    }


HELD = (
    'CREATE TABLE parent (id BIGINT PRIMARY KEY, label VARCHAR)',
    "INSERT INTO parent VALUES (1, 'a'), (2, 'b'), (3, 'c')",
    'CREATE VIEW parent_view AS SELECT * FROM parent',
    'CREATE TABLE "Äpfel" (apfel_id BIGINT PRIMARY KEY)',
    'INSERT INTO "Äpfel" VALUES (1), (2)',
    'CREATE SCHEMA other',
    'CREATE TABLE other.elsewhere (id BIGINT PRIMARY KEY)',
    'INSERT INTO other.elsewhere VALUES (1), (2)',
    'CREATE VIEW other.elsewhere_view AS SELECT * FROM other.elsewhere',
    "ATTACH ':memory:' AS att",
    'CREATE TABLE att.attached (id BIGINT PRIMARY KEY)',
    'INSERT INTO att.attached VALUES (1), (2)',
)


def test_a_relation_the_source_holds_keeps_its_provider_s_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def held() -> Any:
        connection = duckdb.connect()
        for statement in HELD:
            connection.execute(statement)
        return connection

    core, _ = _named_on_the_core(held(), {'valid': 'parent_view'})
    assert [t['name'] for t in core['tables']] == ['valid']
    for source in ('parent_view', 'parent', 'PARENT'):
        with pytest.raises(StructuredDataError) as failed:
            _named_on_the_core(held(), {'x': source}, fail_columns_of=source)
        assert failed.value.code == 'PROVIDER_ERROR', source
    for source in ('nope', 'äpfel', 'elsewhere', 'elsewhere_view', 'attached'):
        with pytest.raises(StructuredDataError) as missing:
            _named_on_the_core(held(), {'x': source})
        assert missing.value.code == 'INVALID_REQUEST', source
        assert missing.value.details['field'] == 'tables'


def test_where_the_core_and_the_old_route_differ_on_names_both_are_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dotted = (
        'CREATE TABLE "a.b" (id BIGINT PRIMARY KEY)',
        'INSERT INTO "a.b" VALUES (1), (2)',
    )
    core, _ = _named_on_the_core(_connection(*dotted), {'ok': 'a.b'})
    assert [t['name'] for t in core['tables']] == ['ok']
    for source in ('main.users', '"users"'):
        with pytest.raises(StructuredDataError) as named:
            _named_on_the_core(_connection(), {'x': source})
        assert named.value.code == 'INVALID_REQUEST', source
    empty = ('CREATE TABLE "a`b" (id BIGINT)',)
    with pytest.raises(StructuredDataError) as both:
        Schema.from_source(DuckDBSource(_connection(*empty)))
    assert both.value.code == 'INVALID_REQUEST'


def test_a_listed_table_is_held_to_the_schema_s_rule_on_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(StructuredDataError) as core:
        Schema.from_source(
            DuckDBSource(
                _connection(
                    'CREATE TABLE "a`b" (id BIGINT PRIMARY KEY)',
                    'INSERT INTO "a`b" VALUES (1), (2)',
                )
            )
        )
    assert core.value.code == 'INVALID_REQUEST'
    assert core.value.details['field'] == 'tables'


SYSTEM_VIEWS = [
    ('one table', 'duckdb_tables', 'column_type', 'RuntimeError'),
    ('shop', 'duckdb_tables', 'column_type', 'RuntimeError'),
    ('one table', 'duckdb_views', 'empty_table', 'RuntimeError'),
    ('one table', 'duckdb_columns', None, None),
    ('one table', 'pg_class', None, None),
    ('shop', 'pg_class', None, None),
    ('shop', 'duckdb_columns', None, None),
]


def _system_view_source(base: str) -> Any:
    if base == 'shop':
        return _connection()
    connection = duckdb.connect()
    connection.execute('CREATE TABLE t (a INTEGER)')
    connection.execute('INSERT INTO t VALUES (1)')
    return connection


def _named_view(base: str, view: str) -> Schema:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_source(
            DuckDBSource(_system_view_source(base)), {'x': view}
        )


@pytest.mark.parametrize(('base', 'view', 'core_rule', 'old'), SYSTEM_VIEWS)
def test_a_system_view_named_as_a_table_is_pinned_on_both_routes(
    monkeypatch: pytest.MonkeyPatch,
    base: str,
    view: str,
    core_rule: str | None,
    old: str | None,
) -> None:
    if core_rule is None:
        assert list(_named_view(base, view).tables) == ['x']
    else:
        with pytest.raises(StructuredDataError) as refused:
            _named_view(base, view)
        assert type(refused.value).__name__ == (
            'StructuredDataError'
            if core_rule == 'empty_table'
            else 'UnsupportedRequestError'
        )
        assert refused.value.details['rule'] == core_rule
