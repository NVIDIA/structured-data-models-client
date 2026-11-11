# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLite source contracts for scratch-table isolation, keys and repeated calls.

Expected values are calculated from the source rows.
The current public client is the only implementation executed.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


R = _module(
    'sqlite_scenarios',
    ROOT / 'contract/support/sqlite_scenarios.py',
)
SEEDS = ('seeds-table', 'seeds-table-numbered', 'seeds-table-upper')


def _on_the_core(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    locate_counts: list[int] | None = None,
) -> dict[str, Any]:
    r"""The scenario through the public calls on SQLite's core routes:
    ``from_source`` (tables named or not), ``materialize`` and
    validation."""
    if locate_counts is not None:
        from sdm_client import PredictiveQuery
        from sdm_client._rust.providers.sqlite import SqliteProvider

        provider = SqliteProvider.__call__
        materialize = PredictiveQuery.materialize
        located: list[int] = []

        def fetch(self: Any, step: dict[str, Any], keys: Any = None) -> Any:
            if step.get('purpose') != 'locate':
                return provider(self, step, keys)
            with R._runner().StatementLog('sqlite') as log:
                answer = provider(self, step, keys)
            located.append(sum(bool(text) for text in log.texts))
            return answer

        def call(self: Any, *args: Any, **kwargs: Any) -> Any:
            before = sum(located)
            try:
                return materialize(self, *args, **kwargs)
            finally:
                locate_counts.append(sum(located) - before)

        monkeypatch.setattr(SqliteProvider, '__call__', fetch)
        monkeypatch.setattr(PredictiveQuery, 'materialize', call)
    answer: dict[str, Any] = R.run(R.SCENARIOS[name])
    return answer


def _rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(zip(table['columns'], row, strict=True)) for row in table['rows']
    ]


def _success(answer: dict[str, Any]) -> dict[str, Any]:
    assert answer['outcome'] == {'result': 'success', 'stage': 'materialize'}
    assert len(answer['payloads']) == 1
    return answer['payloads'][0]


def _simple_request(
    answer: dict[str, Any], *, fractional: bool = False
) -> None:
    body = _success(answer)
    [table] = answer['schema']['tables']
    assert table['primary_key'] == 'id'
    assert table['time_column'] is None
    assert table['columns'] == [
        {'name': 'id', 'dtype': 'int64', 'stype': 'ID'},
        {'name': 'age', 'dtype': 'float64', 'stype': 'numerical'},
    ]
    assert answer['schema']['links'] == []
    shift = 0.5 if fractional else 0
    expected = [102.5, 106.5, 107.5] if fractional else [102, 106, 105]
    context = _rows(body['context']['instance_table'])
    predict = _rows(body['predict']['instance_table'])
    assert [row['__kumo_entity_ref_0'] for row in context] == expected
    assert [row['TARGET'] for row in context] == [
        (key - shift - 100) * 10 for key in expected
    ]
    assert [row['__kumo_entity_ref_0'] for row in predict] == [101 + shift]
    assert body['task']['kind'] == 'regression'
    assert body['task']['target'] == {
        'column_name': 'TARGET',
        'dtype': 'float32',
    }
    assert body['metadata']['num_context_examples'] == 3
    assert body['metadata']['num_prediction_examples'] == 1
    wire = body['schema']
    dtype = 'float64' if fractional else 'int64'
    assert (
        wire['instance_table']['columns']['__kumo_entity_ref_0']['dtype']
        == dtype
    )
    assert wire['related_tables']['users']['columns']['id']['dtype'] == dtype
    assert wire['related_tables']['users']['primary_key'] == [
        'instance_id',
        'id',
    ]
    assert set(wire['related_tables']['users']['columns']) == {
        'instance_id',
        'id',
    }
    seen: set[int] = set()
    for part, rows in [('context', context), ('predict', predict)]:
        ids = [row['instance_id'] for row in rows]
        assert len(ids) == len(set(ids)) and seen.isdisjoint(ids)
        seen.update(ids)
        assert set(body[part]['related_tables']) == {'users'}
        users = _rows(body[part]['related_tables']['users'])
        assert users == [
            {
                'instance_id': row['instance_id'],
                'id': row['__kumo_entity_ref_0'],
            }
            for row in rows
        ]
        assert all(row['id'] is not None for row in users)


def _relational_schema(
    answer: dict[str, Any], *, fractions: bool = False, composite: bool = False
) -> None:
    tables = {table['name']: table for table in answer['schema']['tables']}
    assert set(tables) == {'users', 'orders'}
    assert tables['users']['primary_key'] == 'user_id'
    assert tables['users']['time_column'] is None
    assert tables['orders']['primary_key'] == 'order_id'
    assert tables['orders']['time_column'] == 'ts'
    feature = (
        ('score', 'int64', 'numerical')
        if fractions
        else ('country', 'string', 'text' if composite else 'categorical')
    )
    assert tables['users']['columns'] == [
        {'name': 'user_id', 'dtype': 'int64', 'stype': 'ID'},
        dict(zip(('name', 'dtype', 'stype'), feature, strict=True)),
    ]
    assert tables['orders']['columns'] == [
        {'name': 'order_id', 'dtype': 'int64', 'stype': 'ID'},
        {'name': 'user_id', 'dtype': 'int64', 'stype': 'ID'},
        {'name': 'ts', 'dtype': 'string', 'stype': 'timestamp'},
        {'name': 'price', 'dtype': 'float64', 'stype': 'numerical'},
    ]
    assert answer['schema']['links'] == [
        ['orders', 'user_id', 'users', 'user_id']
    ]


def _relational_request(
    answer: dict[str, Any], *, fractions: bool = False
) -> None:
    _relational_schema(answer, fractions=fractions)
    body = _success(answer)
    anchor = datetime(2024, 3, 1, tzinfo=timezone.utc)
    assert body['task']['kind'] == 'regression'
    assert body['task']['target'] == {
        'column_name': 'TARGET',
        'dtype': 'float32',
    }
    feature = 'score' if fractions else 'country'
    assert body['metadata']['num_context_examples'] == 40
    assert body['metadata']['num_prediction_examples'] == 2
    wire = body['schema']
    assert wire['related_tables']['users']['columns'][feature]['dtype'] == (
        'float64' if fractions else 'string'
    )
    assert (
        wire['related_tables']['orders']['columns']['ts']['dtype']
        == 'timestamp[us]'
    )
    assert wire['instance_table']['primary_key'] == 'instance_id'
    assert {
        name: info['dtype']
        for name, info in wire['instance_table']['columns'].items()
    } == {
        'instance_id': 'int64',
        '__kumo_entity_ref_0': 'int64',
        '__kumo_anchor_time': 'timestamp[us]',
        'TARGET': 'float32',
    }
    assert wire['related_tables']['users']['primary_key'] == [
        'instance_id',
        'user_id',
    ]
    assert wire['related_tables']['orders']['primary_key'] == [
        'instance_id',
        '__node_id',
    ]
    assert {
        name: info['dtype']
        for name, info in wire['related_tables']['users']['columns'].items()
    } == {
        'instance_id': 'int64',
        'user_id': 'int64',
        feature: 'float64' if fractions else 'string',
    }
    assert {
        name: info['dtype']
        for name, info in wire['related_tables']['orders']['columns'].items()
    } == {
        'instance_id': 'int64',
        '__node_id': 'int64',
        'user_id': 'int64',
        'price': 'float64',
        'ts': 'timestamp[us]',
    }
    assert wire['relationships'] == [
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
    seen: set[int] = set()
    for part in ('context', 'predict'):
        section = body[part]
        rows = _rows(section['instance_table'])
        entities = [row['__kumo_entity_ref_0'] for row in rows]
        assert Counter(entities) == Counter(
            range(40) if part == 'context' else [0, 1]
        )
        at = anchor - timedelta(days=30) if part == 'context' else anchor
        assert all(
            datetime.fromisoformat(
                row['__kumo_anchor_time'].replace('Z', '+00:00')
            )
            == at
            for row in rows
        )
        ids = [row['instance_id'] for row in rows]
        assert len(ids) == len(set(ids)) and seen.isdisjoint(ids)
        seen.update(ids)
        assert set(section['related_tables']) == {'users', 'orders'}
        users = _rows(section['related_tables']['users'])
        orders = _rows(section['related_tables']['orders'])
        assert Counter(row['instance_id'] for row in users) == Counter(ids)
        assert {row['instance_id'] for row in orders} <= set(ids)
        assert len({row['__node_id'] for row in orders}) == len(orders)
        for instance in rows:
            own, entity = (
                instance['instance_id'],
                instance['__kumo_entity_ref_0'],
            )
            [user] = [row for row in users if row['instance_id'] == own]
            assert user['user_id'] == entity
            assert user[feature] == (
                entity + 0.5 if fractions else f'c{entity % 3}'
            )
            source = [
                (
                    datetime(
                        2024,
                        1 + order % 2,
                        10 + order % 15,
                        tzinfo=timezone.utc,
                    ),
                    order + 0.5,
                )
                for order in range(200)
                if order % 40 == entity
            ]
            actual = [row for row in orders if row['instance_id'] == own]
            assert all(row['user_id'] == entity for row in actual)
            assert Counter(
                (
                    datetime.fromisoformat(row['ts'].replace('Z', '+00:00')),
                    row['price'],
                )
                for row in actual
            ) == Counter((time, price) for time, price in source if time <= at)
            if part == 'context':
                assert instance['TARGET'] == sum(
                    price
                    for time, price in source
                    if at < time <= at + timedelta(days=30)
                )


@pytest.mark.parametrize('name', SEEDS)
def test_caller_temporary_table_survives_materialize(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    got = _on_the_core(monkeypatch, name)
    assert got['temp_after'] == {
        'columns': ['important'],
        'rows': [['keep me']],
    }
    _simple_request(got)


def test_declared_integer_key_preserves_fractional_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _simple_request(_on_the_core(monkeypatch, 'float-keys'), fractional=True)


def test_null_primary_key_is_filtered_without_losing_valid_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _simple_request(_on_the_core(monkeypatch, 'null-key'))


def test_declared_integer_feature_preserves_fractional_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _relational_request(
        _on_the_core(monkeypatch, 'integer-fractions'), fractions=True
    )


def test_composite_key_does_not_make_its_first_column_unique(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    got = _on_the_core(monkeypatch, 'composite-key')
    _relational_schema(got, composite=True)
    assert got['outcome'] == {
        'result': 'refusal',
        'stage': 'materialize',
        'error': 'StructuredDataError',
        'code': 'INVALID_REQUEST',
    }
    assert got['payloads'] is None


def test_partial_unique_index_refreshes_facts_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locate: list[int] = []
    got = _on_the_core(monkeypatch, 'partial-unique', locate)
    assert locate == [7]
    # Eleven description reads, then seven refresh and seven gather reads.
    assert got['statements'] == {'from_source': 11, 'materialize': 14}
    _relational_request(got)


def test_each_materialize_refreshes_facts_and_keeps_results_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locate: list[int] = []
    got = _on_the_core(monkeypatch, 'materialize-twice', locate)
    assert locate == [6, 6]
    assert got['statements'] == {'from_source': 10, 'materialize': 14}
    _relational_request(got)
    assert len(got['repeated']) == 1
    # Both calls refresh source facts; only the two index checks stay cached.
    assert got['repeated'][0]['statements'] == 12
    assert got['repeated'][0]['payloads'] == got['payloads']
