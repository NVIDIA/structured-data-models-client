# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Connector schema, key, time and materialization behavior with direct assertions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import schema_scenarios as R

KEYS = R.keys()


def _require_database(database):
    if database == 'postgres' and not os.environ.get('SDM_POSTGRES_TEST_DSN'):
        if os.environ.get('SDM_REQUIRE_POSTGRES'):
            pytest.fail('SDM_REQUIRE_POSTGRES requires SDM_POSTGRES_TEST_DSN')
        pytest.skip('requires SDM_POSTGRES_TEST_DSN')


@pytest.mark.parametrize('key', KEYS)
def test_connector_schema_decisions(key: str, tmp_path: Path) -> None:
    """Each connector case enforces its schema or materialization contract."""
    name, database = key.rsplit('@', 1)
    _require_database(database)
    found = R.SCENARIOS[name]
    actual = R.observe(found, database, tmp_path)
    if found.family == 'time':
        _assert_time(name, actual)
    elif found.family == 'keys':
        _assert_keys(found, actual)
    elif found.family == 'described':
        _assert_described(name, actual)
    elif found.family == 'refusals':
        _assert_refusal(name, actual)
    else:
        assert found.family == 'postgres'
        _assert_postgres(name, actual)


# Chosen from the input columns' documented semantics, not recorded schemas.
TIME_SELECTIONS = {
    'an-order-date-beside-an-edit-stamp': 'OrderDate',
    'a-lone-edit-stamp-on-a-table-nothing-links-to': 'updated_at',
    'two-edit-stamps-and-nothing-else': None,
    'a-lone-load-stamp': None,
    'a-lone-constant-stamp': None,
    'a-stamp-constant-only-in-its-first-rows': 'ordered',
    'a-creation-stamp-beside-another': 'created_at',
    'two-event-stamps': 'ordered_at',
    'an-order-header': 'OrderDate',
    'the-earliest-only-past-the-rows-read': 'shipped_at',
    'a-hire-date-beside-a-birth-date': 'HireDate',
    'a-lone-date-of-birth': None,
    'a-mostly-empty-stamp-beside-the-order-date': 'order_date',
    'a-lone-mostly-empty-stamp': None,
    'a-placeholder-filled-stamp-beside-the-order-date': 'order_date',
    'every-candidate-rejected': None,
    'two-zoned-stamps': 'ordered_at',
    'two-text-stamps': 'ordered_at',
    'two-dates': 'ordered_on',
    'a-date-beside-its-timestamp': 'ordered_at',
}


def _assert_time(name, actual):
    assert actual['outcome'] == 'schema'
    selected = {
        table: facts['time_column'] for table, facts in actual['tables'].items()
    }
    case = name.split('.')[1]
    if case in TIME_SELECTIONS:
        expected = {'orders': TIME_SELECTIONS[case], 'customers': None}
    elif 'a-log-and-a-linked-table' in case:
        expected = {
            'users': None,
            'orders': None,
            'order_status_updates': 'updated_at',
        }
    elif case in {'every-guess-in-one-notice'} or case.startswith(
        'the-override-for-'
    ):
        expected = {
            'customers': None,
            'orders': None,
            'shipments': 'packed_at',
            'order_status_updates': 'updated_at',
        }
        if case.startswith('the-override-for-'):
            table, column = case.removeprefix('the-override-for-').rsplit(
                '-', 1
            )
            expected[table] = None if column == 'none' else column
            assert f'- {table}.' not in _said(actual)
    elif case in {
        'a-link-the-caller-names-refuses-the-edit-stamp',
        'no-link-keeps-the-edit-stamp',
    }:
        expected = {
            'accounts': None if case.startswith('a-link') else 'updated_at',
            'logins': 'at',
        }
    elif case == 'a-one-row-table-keeps-its-timestamp':
        expected = {'orders': 'ordered', 'customers': None}
    elif case == 'the-caller-names-one-table':
        expected = {'orders': 'shipped_at', 'customers': None}
        assert '- orders.' not in _said(actual)
    else:
        assert case == 'the-caller-clears-one-table', case
        expected = {'orders': 'ordered_at', 'customers': None}
        assert '- customers.' not in _said(actual)
    assert selected == expected
    assert _said(actual).startswith('Some time columns were inferred')


def _assert_keys(scenario, actual):
    name = scenario.name
    _assert_key_outcome(name, actual)
    if name.startswith('keys.score.'):
        ids = name.rsplit('.', 1)[1].split('-')
        if 'u007' in ids:
            assert actual['code'] == 'INVALID_REQUEST'
        else:
            columns = actual['predict']['columns']
            entity = columns.index('__kumo_entity_ref_0')
            assert [
                json.loads(row)[entity] for row in actual['predict']['rows']
            ] == ids
            assert 'TARGET' not in columns
            assert actual['parts'] == 1
        return
    if name.endswith('declared-in-the-catalog'):
        assert actual['tables']['users']['primary_key'] == 'user_id'
        assert actual['tables']['users']['duplicate_keys'] == 0
        return
    frames = scenario.frames()
    users = frames['app_users']
    if 'composite-repeating-once' in name:
        key = ['user_id', 'line']
    elif 'composite' in name:
        key = ['country', 'age']
    else:
        key = ['user_id']
    duplicates = int(users.duplicated(key).sum())
    missing = bool(users[key].isna().any(axis=None))
    stated = '.stated.' in name
    unchecked = name.endswith('.unchecked')
    too_many = duplicates * 100 > len(users)
    if stated and (missing or too_many) and not unchecked:
        assert actual['outcome'] == 'refusal'
        assert actual['code'] == 'INVALID_REQUEST'
        assert 'users' in actual['message']
        assert all(column in actual['message'] for column in key)
        assert ('no value' if missing else 'not unique enough') in actual[
            'message'
        ]
    else:
        assert actual['outcome'] == 'schema'
        table = actual['tables']['users']
        expected_key = (
            None
            if too_many and not unchecked
            else (key[0] if len(key) == 1 else key)
        )
        assert table['primary_key'] == expected_key
        if unchecked:
            assert table['duplicate_keys'] is None
        elif expected_key is not None and not missing:
            assert table['duplicate_keys'] == duplicates
        assert set(table['columns']) == set(users.columns)


def _assert_payload(actual, static, ids):
    assert actual['outcome'] == 'payload'
    assert actual['parts'] == 1
    assert actual['task'] == {
        'kind': 'multiclass_classification' if static else 'regression',
        'target': {
            'column_name': 'TARGET',
            'dtype': 'string',
            'classes': ['DE', 'IN', 'US'],
        }
        if static
        else {'column_name': 'TARGET', 'dtype': 'float32'},
        'entity_table_names': ['users'],
        'anchor_time_column': '__kumo_anchor_time',
    }
    table = actual['predict']
    assert table['columns'] == [
        'instance_id',
        '__kumo_entity_ref_0',
        '__kumo_anchor_time',
    ]
    rows = [json.loads(row) for row in table['rows']]
    assert [row[1] for row in rows] == ids
    assert {row[2] for row in rows} == {'2024-12-01T00:00:00.000000Z'}
    assert len({row[0] for row in rows}) == len(ids)
    assert 'TARGET' in actual['context_columns']
    expected_users = (
        ['age', 'instance_id', 'user_id']
        if static
        else ['age', 'country', 'instance_id', 'user_id']
    )
    assert actual['context_related']['users'] == expected_users
    related = actual['predict_related']['users']
    assert sorted(related['columns']) == expected_users
    originals = R.USERS.set_index('user_id')
    observed = []
    for row in related['rows']:
        cells = dict(zip(related['columns'], json.loads(row), strict=True))
        observed.append(cells['user_id'])
        assert cells['age'] == int(originals.loc[cells['user_id'], 'age'])
        if not static:
            assert (
                cells['country'] == originals.loc[cells['user_id'], 'country']
            )
    assert sorted(observed) == sorted(ids)


def _assert_described(name, actual):
    if name.endswith(('.static', '.temporal')):
        _assert_payload(
            actual, name.endswith('.static'), ['u001', 'u002', 'u003']
        )
    else:
        assert actual['outcome'] == 'refusal'
        assert actual['code'] == 'INVALID_REQUEST'
        if name.endswith('a-missing-table'):
            assert 'nope' in actual['message'] and 'exist' in actual['message']
        else:
            assert name.endswith('a-repeated-id')
            assert "Cannot score 'u007'" in actual['message']


def _assert_refusal(name, actual):
    assert actual['outcome'] == 'refusal'
    assert actual['code'] == 'INVALID_REQUEST'
    message = actual['message'].lower()
    assert 'the request is not one the core accepts' not in message
    if name.endswith('an-empty-source'):
        assert 'no table' in message
    else:
        assert 'nope' in message
        if 'column' in name:
            assert 'column' in message
        else:
            assert 'table' in message


def _assert_postgres(name, actual):
    if 'described-and-asked' in name:
        _assert_payload(
            actual['materialize'],
            name.endswith('.static'),
            ['u001', 'u002', 'u003'],
        )
        described = actual['describe']
        assert set(described['tables']) == {'users', 'orders'}
        assert described['tables']['users']['primary_key'] == 'user_id'
        assert ['orders', 'user_id', 'users', 'user_id'] in described[
            'relationships'
        ]
        return
    if 'keys-a-keyless-table' in name:
        described = actual['describe']
        assert described['outcome'] == 'schema'
        assert described['tables']['region']['primary_key'] == 'code'
        assert described['tables']['store']['primary_key'] == 'store_id'
        assert described['relationships'] == [
            ['sale', 'region_code', 'region', 'code']
        ]
        assert actual['materialize'] == {
            'outcome': 'labels',
            'labels': ['["n", 15.0]', '["s", 15.0]'],
        }
        return
    assert actual['outcome'] == 'schema'
    if 'matches-exactly' in name:
        assert set(actual['tables']) == {'Parent', 'parent', 'child'}
        assert actual['relationships'] == [
            ['child', 'parent_id', 'Parent', 'parent_id']
        ]
        return
    all_tables = 'every-table' in name
    users, orders = (
        ('<users>', '<orders>') if all_tables else ('users', 'orders')
    )
    assert set(actual['tables']) == {users, orders}
    for table, columns, key, source in [
        (users, ['country', 'user_id'], 'user_id', '<users>'),
        (
            orders,
            ['amount', 'order_id', 'ts', 'user_id'],
            'order_id',
            '<orders>',
        ),
    ]:
        facts = actual['tables'][table]
        assert facts['columns'] == columns
        assert facts['primary_key'] == key
        assert facts['source_name'] == source
    assert actual['relationships'] == [[orders, 'user_id', users, 'user_id']]


def _said(answer: dict[str, Any]) -> str:
    return '\n'.join(text for _, text in answer.get('notices', []))


def _assert_key_outcome(name: str, first: Any) -> None:
    if name.startswith('keys.score.'):
        ids = name.rsplit('.', 1)[1].split('-')
        if 'u007' in ids:
            assert first['outcome'] == 'refusal'
            assert "Cannot score 'u007'" in first['message']
        else:
            assert first['outcome'] == 'payload'
            assert len(first['predict']['rows']) == len(ids)
    elif name.startswith('keys.'):
        users = first.get('tables', {}).get('users', {})
        if name.endswith('.unchecked'):
            assert users['duplicate_keys'] is None
            assert first['notices'] == []
        elif '.holds' in name or name.endswith('in-the-catalog'):
            assert first['outcome'] == 'schema'
            assert users['duplicate_keys'] == 0
            assert first['notices'] == []
        elif 'missing-a-value' in name:
            assert 'no value' in (first.get('message', '') + _said(first))
        else:
            assert 'repeat' in name
            assert (
                'not unique enough' in first.get('message', '')
                or 'already been used' in _said(first)
                or 'repeat a value' in _said(first)
            ), first


def test_the_core_s_warning_prefix_is_set_aside_and_nothing_else() -> None:
    r"""Control: :func:`bare` takes off one ``[CODE] `` and keeps the rest,
    so a notice the core words otherwise still differs."""
    assert R.bare('[KEY_REPEATS] users.user_id identifies') == (
        'users.user_id identifies'
    )
    assert R.bare('[INVALID_REQUEST] x') == 'x'
    assert R.bare('Some time columns [X] y') == 'Some time columns [X] y'
    assert R.bare('[not a code] y') == '[not a code] y'
