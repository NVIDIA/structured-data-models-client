# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLite provider and exact-key joins."""

from __future__ import annotations

import functools
import importlib.util
import json
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

DATABASE = 'sqlite'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


M = _module(
    'sqlite_key_scenarios',
    ROOT / 'contract/support/sqlite_key_scenarios.py',
)

R = M.R


def test_the_core_reports_a_sqlite_provider() -> None:
    r"""Exit (the staging rule's marker): the core under test reports
    a SQLite provider (``sdm_client._native.PROVIDERS``), so a SQLite job
    and the gate run its core legs there."""

    def check() -> None:
        from sdm_client import _native

        assert DATABASE in tuple(_native.PROVIDERS)

    check()


def _by_id() -> dict[str, dict[str, Any]]:
    return {vector['id']: vector for vector in M.VECTORS}


def test_the_join_check_fails_on_a_wrong_join() -> None:
    r"""Control::func:`joins` passes the reference's own request on SQLite
    (``base.regression@sqlite``: the shop, its question and entities, as
    ``pair.bigint.bigint`` asks) and fails it with one order moved to
    another entity, and with one entity's age taken from another."""
    vector = _by_id()['pair.bigint.bigint']
    observation = _core_runs()['pair.bigint.bigint']
    [body] = [p['parsed'] for p in observation['public']['payloads']]
    assert M.joins(body, M.offset(vector)) == []
    orders = body['predict']['related_tables']['orders']
    at = orders['columns'].index('user_id')
    moved = json.loads(json.dumps(body))
    moved['predict']['related_tables']['orders']['rows'][0][at] = {
        'number': '5'
    }
    assert M.joins(moved, M.offset(vector)) != []
    users = body['context']['related_tables']['users']
    at = users['columns'].index('age')
    aged = json.loads(json.dumps(body))
    aged['context']['related_tables']['users']['rows'][0][at] = {'number': '99'}
    assert M.joins(aged, M.offset(vector)) != []


@functools.cache
def _core_runs() -> dict[str, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix='sqlite_sql_key_joins-') as tmp:
        result: dict[str, dict[str, Any]] = M.run(
            'python-core', M.VECTORS, Path(tmp)
        )
        return result


def _key_dtypes(name: str) -> tuple[str, str]:
    """The SQLite loader stores wide integers and decimals as exact text."""

    def dtype(tag: str) -> str:
        if tag == 'bigint':
            return 'int64'
        if tag in ('double', 'float'):
            return 'float64'
        return 'string'

    if name.startswith(('pair.', 'wide.')):
        _, primary, foreign = name.split('.')
        return dtype(primary), dtype(foreign)
    assert name.startswith('decimal.key.')
    if '.foreign-only.' in name:
        return 'int64', 'string'
    if '.primary-only.' in name:
        return 'string', 'int64'
    return 'string', 'string'


def _assert_shop_rows(body: dict[str, Any], base: Decimal) -> None:
    """Complete history and labels calculated from the independent shop formula."""
    origin = datetime(2023, 1, 1, tzinfo=timezone.utc)
    anchor = datetime(2024, 3, 1, tzinfo=timezone.utc)
    number = M._number

    def parse_time(value: str) -> datetime:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))

    seen: set[Decimal] = set()
    for part in ('context', 'predict'):
        section = body[part]
        instances = section['instance_table']
        rows = [
            dict(zip(instances['columns'], row, strict=True))
            for row in instances['rows']
        ]
        identities = [
            (
                number(row['__kumo_entity_ref_0']) - base,
                parse_time(row['__kumo_anchor_time']),
            )
            for row in rows
        ]
        expected = (
            [
                (Decimal(entity), anchor - timedelta(days=30 * step))
                for entity in range(8)
                for step in range(1, 15)
            ]
            if part == 'context'
            else [(Decimal(0), anchor), (Decimal(1), anchor)]
        )
        assert Counter(identities) == Counter(expected)
        ids = [number(row['instance_id']) for row in rows]
        assert len(ids) == len(set(ids))
        assert seen.isdisjoint(ids)
        seen.update(ids)
        related = section['related_tables']
        assert set(related) == {'orders', 'users'}
        orders = [
            dict(zip(related['orders']['columns'], row, strict=True))
            for row in related['orders']['rows']
        ]
        users = [
            dict(zip(related['users']['columns'], row, strict=True))
            for row in related['users']['rows']
        ]
        assert Counter(number(row['instance_id']) for row in users) == Counter(
            ids
        )
        assert {number(row['instance_id']) for row in orders} <= set(ids)
        assert len({number(row['__node_id']) for row in orders}) == len(orders)
        for user in users:
            entity = int(number(user['user_id']) - base)
            assert user['country'] == ['US', 'DE', 'IN'][entity % 3]

        for instance, (entity, at) in zip(rows, identities, strict=True):
            own = number(instance['instance_id'])
            source = [
                (
                    origin + timedelta(days=7 * order),
                    Decimal(10) + Decimal('1.5') * order,
                )
                for order in range(80)
                if order % 8 == entity
            ]
            history = [(time, price) for time, price in source if time <= at]
            actual = [
                (parse_time(row['ts']), number(row['price']))
                for row in orders
                if number(row['instance_id']) == own
            ]
            assert Counter(actual) == Counter(history)
            if part == 'context':
                label = sum(
                    (
                        price
                        for time, price in source
                        if at < time <= at + timedelta(days=30)
                    ),
                    Decimal(0),
                )
                assert number(instance['TARGET']) == label
        assert not M.joins(body, base)


def _assert_wire(body: dict[str, Any], primary: str) -> None:
    assert body['task'] == {
        'anchor_time_column': '__kumo_anchor_time',
        'entity_table_names': ['users'],
        'kind': 'regression',
        'target': {'column_name': 'TARGET', 'dtype': 'float32'},
    }
    assert body['metadata']['num_context_examples'] == {'number': '112'}
    assert body['metadata']['num_prediction_examples'] == {'number': '2'}
    schema = body['schema']
    instance = schema['instance_table']
    assert instance['primary_key'] == 'instance_id'
    assert {
        name: info['dtype'] for name, info in instance['columns'].items()
    } == {
        'instance_id': 'int64',
        '__kumo_entity_ref_0': primary,
        '__kumo_anchor_time': 'timestamp[us]',
        'TARGET': 'float32',
    }
    assert instance['columns']['TARGET']['stype'] == 'numerical'
    assert instance['columns']['__kumo_anchor_time']['stype'] == 'timestamp'
    assert instance['columns']['__kumo_entity_ref_0']['stype'] == 'ID'
    assert instance['columns']['instance_id']['nullable'] is False
    related = schema['related_tables']
    assert set(related) == {'users', 'orders'}
    assert related['users']['primary_key'] == ['instance_id', 'user_id']
    assert related['orders']['primary_key'] == ['instance_id', '__node_id']
    assert {
        name: info['dtype']
        for name, info in related['users']['columns'].items()
    } == {
        'instance_id': 'int64',
        'user_id': primary,
        'age': 'int64',
        'country': 'string',
    }
    assert {
        name: info['dtype']
        for name, info in related['orders']['columns'].items()
    } == {
        'instance_id': 'int64',
        '__node_id': 'int64',
        # Graph references use the target primary key's wire type.
        'user_id': primary,
        'price': 'float64',
        'ts': 'timestamp[us]',
    }
    assert related['orders']['columns']['ts']['stype'] == 'timestamp'
    assert related['orders']['columns']['price']['stype'] == 'numerical'
    assert schema['relationships'] == [
        {
            'source_columns': ['__kumo_entity_ref_0'],
            'target_columns': ['user_id'],
            'target_table': 'users',
        },
        {
            'source_columns': ['user_id'],
            'source_table': 'orders',
            'target_columns': ['user_id'],
            'target_table': 'users',
        },
    ]


def test_each_key_pair_preserves_exact_values_and_joins():
    observed = _core_runs()
    assert set(observed) == set(_by_id())
    for vector in M.VECTORS:
        key = vector['id']
        observation = observed[key]
        got = R.project(observation, vector)
        primary, foreign = _key_dtypes(key)
        compatible = (primary == 'string') == (foreign == 'string')
        assert observation['identity']['native_loaded'], key
        tables = {table['name']: table for table in got['schema']['tables']}
        assert set(tables) == {'users', 'orders'}, key
        for table_name, key_dtype in [('users', primary), ('orders', foreign)]:
            table = tables[table_name]
            assert table['primary_key'] == (
                'user_id' if table_name == 'users' else 'order_id'
            ), key
            assert table['time_column'] == (
                'ts' if table_name == 'orders' else None
            ), key
            assert table['duplicate_keys'] == 0, key
            cols = {
                column['name']: (column['dtype'], column['stype'])
                for column in table['columns']
            }
            expected = (
                {
                    'user_id': (key_dtype, 'ID'),
                    'age': ('int64', 'numerical'),
                    'country': ('string', 'categorical'),
                }
                if table_name == 'users'
                else {
                    'order_id': ('int64', 'ID'),
                    'user_id': (key_dtype, 'ID'),
                    'price': ('float64', 'numerical'),
                    'ts': ('string', 'timestamp'),
                }
            )
            assert cols == expected, key
        assert got['schema']['links'] == (
            [['orders', 'user_id', 'users', 'user_id']] if compatible else []
        ), key
        assert got['warnings'] == (
            ['UserWarning'] if compatible else ['SDMWarning']
        ), key
        if not compatible:
            assert got['outcome'] == {
                'result': 'refusal',
                'stage': 'materialize',
                'error': 'StructuredDataError',
                'code': 'INVALID_REQUEST',
            }, key
            assert not observation['public']['payloads'], key
            assert got['result'] is None, key
            continue
        assert got['outcome'] == {
            'result': 'success',
            'stage': 'predict',
            'error': None,
            'code': None,
        }, key
        assert got['result']['rows'] == 2, key
        assert got['result']['columns'] == [
            {'name': 'row_index', 'dtype': 'int64'},
            {'name': 'id', 'dtype': 'str' if primary == 'string' else primary},
            {'name': 'prediction', 'dtype': 'float64'},
        ], key
        result = observation['core']['predicted']['result']
        names = [column['name'] for column in result['columns']]
        assert names == (
            ['row_index', 'id']
            + (['id_as_written'] if primary == 'float64' else [])
            + ['prediction', 'answer_index']
        ), key
        assert len(result['rows']) == 2
        for position, (cells, entity) in enumerate(
            zip(result['rows'], M.indices(vector), strict=True)
        ):
            row = dict(zip(names, cells, strict=True))
            assert row['row_index'] == {'i64': str(position)}
            assert row['prediction'] == {
                'f64_bits': '3ff0000000000000'
            }  # Endpoint returns 1.0.
            assert row['answer_index'] == {'i64': str(position)}
            if primary == 'int64':
                assert row['id'] == {'i64': str(entity)}
            else:
                assert set(row['id']) == {'utf8'}
                assert Decimal(row['id']['utf8']) == Decimal(str(entity))
                if primary == 'float64':
                    assert row['id_as_written'] == row['id']
                    assert isinstance(
                        json.loads(row['id_as_written']['utf8']), float
                    )
        payloads = observation['public']['payloads']
        assert len(payloads) == 1, key
        _assert_wire(payloads[0]['parsed'], primary)
        _assert_shop_rows(payloads[0]['parsed'], M.offset(vector))
        assert M.joined(observation, vector) is True, key
