# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact numeric keys survive catalog, supplied-key and edited-relationship paths."""

from __future__ import annotations

import functools
import importlib.util
import os
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]

RUNNER = ROOT / 'contract/runners/run_python.py'

CASES = ROOT / 'contract/cases/s1'


DATABASES = ('duckdb', 'sqlite', 'postgres')

FIRST = 2**53

GIVEN_KEY = 'key.past-2-53.given-key'

GIVEN_LINK = 'key.past-2-53.given-link'

EDITED_LINK = 'key.past-2-53.edited-link'

CATALOG = 'key.past-2-53.catalog'

CASE_IDS = (GIVEN_KEY, GIVEN_LINK, EDITED_LINK, CATALOG)

LINK = {
    'source_table': 'orders',
    'source_column': 'user_id',
    'target_table': 'users',
    'target_column': 'user_id',
}


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('test_exact_numeric_keys_run_python', RUNNER)


R = _module(
    'exact_key_scenarios', ROOT / 'contract/support/exact_key_scenarios.py'
)

GIVEN_VECTORS = tuple(R.CONTROL)


def _server() -> bool:
    r"""Whether a Postgres server is named; where none is, one must not be
    required."""
    if os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    pytest.skip('SDM_POSTGRES_TEST_DSN is not configured')


BY_ID = {vector['id']: vector for vector in R.VECTORS}


@functools.cache
def observed(vector_id: str) -> dict[str, Any]:
    _server()
    return R.observe(BY_ID[vector_id], os.environ['SDM_POSTGRES_TEST_DSN'])


def assert_exact_materialization(
    answer: dict[str, Any], vector_id: str
) -> None:
    import pandas as pd

    assert answer['outcome'] == {
        'result': 'success',
        'stage': 'materialize',
        'error': None,
        'code': None,
    }
    assert answer['postgres'] == {
        'users.user_id': 'numeric',
        'orders.user_id': 'numeric',
    }
    schema = answer['schema']
    assert schema['links'] == [['orders', 'user_id', 'users', 'user_id']]
    tables = {t['name']: t for t in schema['tables']}
    assert set(tables) == {'users', 'orders'}
    for name, table in tables.items():
        assert table['primary_key'] == (
            'user_id' if name == 'users' else 'order_id'
        )
        assert table['time_column'] == (None if name == 'users' else 'ts')
        assert table['duplicate_keys'] == 0
        columns = (
            [('user_id', 'string', 'ID'), ('age', 'float64', 'numerical')]
            if name == 'users'
            else [
                ('order_id', 'int64', 'ID'),
                ('user_id', 'string', 'ID'),
                ('price', 'float64', 'numerical'),
                ('ts', 'timestamp[us]', 'timestamp'),
            ]
        )
        assert table['columns'] == [
            dict(name=col, dtype=dtype, stype=stype)
            for col, dtype, stype in columns
        ]
    assert len(answer['raw_payloads']) == 1
    for payload in answer['raw_payloads']:
        assert payload['task'] == {
            'kind': 'regression',
            'target': {'column_name': 'TARGET', 'dtype': 'float32'},
            'entity_table_names': ['users'],
            'anchor_time_column': '__kumo_anchor_time',
        }
        wire_schema = payload['schema']
        assert wire_schema['instance_table']['columns'][
            '__kumo_entity_ref_0'
        ] == {'dtype': 'string', 'stype': 'ID'}
        for name in ('users', 'orders'):
            assert (
                wire_schema['related_tables'][name]['columns']['user_id'][
                    'dtype'
                ]
                == 'string'
            )
        assert wire_schema['relationships'] == [
            {
                'source_columns': ['__kumo_entity_ref_0'],
                'target_table': 'users',
                'target_columns': ['user_id'],
            },
            {
                'source_table': 'orders',
                'source_columns': ['user_id'],
                'target_table': 'users',
                'target_columns': ['user_id'],
            },
        ]
        target = payload['task']['target']['column_name']
        for half in ('context', 'predict'):
            table = payload[half]['instance_table']
            rows = [
                dict(zip(table['columns'], row, strict=True))
                for row in table['rows']
            ]
            assert len(rows) == (
                1
                if half == 'predict'
                else 19
                if vector_id.startswith('pk.')
                else 20
            )
            keys = [row['__kumo_entity_ref_0'] for row in rows]
            expected_keys = (
                {str(FIRST + 1)}
                if half == 'predict'
                else {
                    str(FIRST + i)
                    for i in range(20)
                    if not (vector_id.startswith('pk.') and i == 1)
                }
            )
            assert set(keys) == expected_keys and len(set(keys)) == len(keys)
            by_instance = {r['instance_id']: r for r in rows}
            assert len(by_instance) == len(rows)
            for row in rows:
                key = row['__kumo_entity_ref_0']
                assert type(key) is str and FIRST <= int(key) < FIRST + 20
                expected_day = (
                    31
                    if half == 'predict' or vector_id.startswith('pk.')
                    else 1
                )
                assert row['__kumo_anchor_time'] == (
                    '2024-02-01T00:00:00.000000Z'
                    if expected_day == 31
                    else '2024-01-02T00:00:00.000000Z'
                )
                if half == 'predict':
                    assert key == str(FIRST + 1)
                elif vector_id.startswith('pk.'):
                    assert type(row[target]) is float
                    assert row[target] == 20.0 + (int(key) - FIRST)
                else:
                    day = (
                        pd.Timestamp(row['__kumo_anchor_time'])
                        - pd.Timestamp('2024-01-01', tz='UTC')
                    ).days
                    expected = sum(
                        10 + order * 1.5
                        for order in range(60)
                        if FIRST + order % 20 == int(key)
                        and day < order <= day + 30
                    )
                    assert type(row[target]) is float
                    assert row[target] == float(expected)
            assert set(payload[half]['related_tables']) == {'users', 'orders'}
            for name, related in payload[half]['related_tables'].items():
                for values in related['rows']:
                    row = dict(zip(related['columns'], values, strict=True))
                    key = row['user_id']
                    assert type(key) is str and FIRST <= int(key) < FIRST + 20
                    assert (
                        key
                        == by_instance[row['instance_id']][
                            '__kumo_entity_ref_0'
                        ]
                    )
                    if name == 'users' and vector_id.startswith('pk.'):
                        assert (
                            'age' not in row
                        )  # Target values must not leak into features.
                    elif name == 'users':
                        assert row['age'] == 20.0 + (int(key) - FIRST)
                    else:
                        order = int((row['price'] - 10) / 1.5)
                        assert key == str(FIRST + order % 20)
                if name == 'users':
                    assert related['columns'] == ['instance_id', 'user_id'] + (
                        [] if vector_id.startswith('pk.') else ['age']
                    )
                    assert related['rows'] == [
                        [r['instance_id'], r['__kumo_entity_ref_0']]
                        + (
                            []
                            if vector_id.startswith('pk.')
                            else [
                                20.0 + (int(r['__kumo_entity_ref_0']) - FIRST)
                            ]
                        )
                        for r in rows
                    ]
                else:
                    assert related['columns'] == [
                        'instance_id',
                        '__node_id',
                        'price',
                        'ts',
                        'user_id',
                    ]
                    assert len({(r[0], r[1]) for r in related['rows']}) == len(
                        related['rows']
                    )
                    for instance, item in by_instance.items():
                        expected_orders = [
                            order
                            for order in reversed(range(60))
                            if FIRST + order % 20
                            == int(item['__kumo_entity_ref_0'])
                            and order <= expected_day
                        ]
                        history = [
                            dict(zip(related['columns'], r, strict=True))
                            for r in related['rows']
                            if r[0] == instance
                        ]
                        assert [r['price'] for r in history] == [
                            10 + order * 1.5 for order in expected_orders
                        ]
                        assert [r['ts'] for r in history] == [
                            (
                                pd.Timestamp('2024-01-01', tz='UTC')
                                + pd.Timedelta(days=order)
                            ).strftime('%Y-%m-%dT%H:%M:%S.000000Z')
                            for order in expected_orders
                        ]


@pytest.mark.parametrize('vector_id', sorted(set(R.CONTROL.values())))
def test_catalog_numeric_keys_materialize_without_rounding(
    vector_id: str,
) -> None:
    assert_exact_materialization(observed(vector_id), vector_id)


@pytest.mark.parametrize('vector_id', GIVEN_VECTORS)
def test_given_and_edited_keys_match_catalog_semantics(vector_id: str) -> None:
    got = observed(vector_id)
    assert_exact_materialization(got, vector_id)
    control = observed(R.CONTROL[vector_id])
    assert_exact_materialization(control, R.CONTROL[vector_id])
    for field in ('outcome', 'schema', 'payloads'):
        assert got[field] == control[field], field


@functools.cache
def cases() -> dict[str, tuple[dict[str, Any], str]]:
    r"""The cases by id, each with its digest, as the runner loads
    them."""
    loaded = runner().load_cases([CASES / f'{c}.json' for c in CASE_IDS])
    return {case['id']: (case, digest) for case, digest in loaded}


def _built(case: dict[str, Any]) -> Any:
    r"""The case's DuckDB database, its base and setup run, as the runner
    builds it."""
    import duckdb

    connection = duckdb.connect()
    for statement in [
        *runner().load_bases()[case['input']['base']],
        *case['input'].get('setup', []),
    ]:
        connection.execute(statement)
    return connection


@pytest.mark.parametrize('case_id', CASE_IDS)
def test_each_case_holds_an_exact_key_past_2_53(case_id: str) -> None:
    r"""Exit "cases in the corpus on all three databases": each case runs
    through predict on DuckDB, SQLite and Postgres over ``user_id`` a
    ``DECIMAL(38, 0)`` in both tables, every id past 2**53, two of them
    one double (so a key read as doubles merges them), asked for 2**53 + 1
    and 2**53 + 2; the key and link declared only in ``catalog``, given in
    ``given-key`` (the key) and ``given-link`` (the keys and link), and the
    link set by an edit in ``edited-link`` (given no link)."""
    case, _ = cases()[case_id]
    given = case['input']
    assert case['status'] == 'acceptance' and case['slice'] == 'S1'
    assert case['until'] == 'predict'
    assert runner().case_databases(case) == DATABASES
    assert given['pql'] == (
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
    )
    assert given['materialize']['indices'] == [FIRST + 1, FIRST + 2]
    connection = _built(case)
    try:
        types = dict(
            connection.execute(
                'SELECT table_name, data_type FROM duckdb_columns() '
                "WHERE column_name = 'user_id'"
            ).fetchall()
        )
        ids = [
            Decimal(row[0])
            for row in connection.execute(
                'SELECT user_id FROM users UNION SELECT user_id FROM orders'
            ).fetchall()
        ]
        constraints = sorted(
            row[0]
            for row in connection.execute(
                'SELECT constraint_type FROM duckdb_constraints() '
                "WHERE constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')"
            ).fetchall()
        )
    finally:
        connection.close()
    assert types == {'users': 'DECIMAL(38,0)', 'orders': 'DECIMAL(38,0)'}
    assert min(ids) >= FIRST
    assert len({float(i) for i in ids}) < len(set(ids))
    if case_id == CATALOG:
        assert 'from_source' not in given and 'edits' not in given
        assert constraints == ['FOREIGN KEY', 'PRIMARY KEY', 'PRIMARY KEY']
        return
    assert constraints == []
    keys = {'users': 'user_id', 'orders': 'order_id'}
    expected = {
        GIVEN_KEY: ({'primary_keys': {'users': 'user_id'}}, None),
        GIVEN_LINK: ({'primary_keys': keys, 'relationships': [LINK]}, None),
        EDITED_LINK: (
            {'primary_keys': keys, 'relationships': []},
            [{'op': 'set_relationships', 'relationships': [LINK]}],
        ),
    }[case_id]
    assert (given['from_source'], given.get('edits')) == expected
