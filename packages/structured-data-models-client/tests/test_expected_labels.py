# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public label construction checked against source rows and exact arithmetic."""

from __future__ import annotations

import operator
import warnings
from fractions import Fraction
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import PredictiveQuery, Schema
from sdm_client.errors import StructuredDataError


def _float32_once(exact: Fraction) -> float:
    r"""The float32 nearest ``exact``, ties to even: one rounding, never
    through a float64 first."""
    import numpy as np

    first = np.float32(float(exact))
    candidates = {
        first,
        np.nextafter(first, np.float32(np.inf)),
        np.nextafter(first, np.float32(-np.inf)),
    }

    def key(c: Any) -> tuple[Fraction, int]:
        bits = int(np.array(c, dtype=np.float32).view(np.uint32))
        return (abs(Fraction(float(c)) - exact), bits & 1)

    return float(min(candidates, key=key))


def _database(sql_type='DOUBLE'):
    connection = duckdb.connect()
    connection.execute('CREATE TABLE users(user_id BIGINT PRIMARY KEY)')
    connection.execute('INSERT INTO users VALUES (0),(1),(2),(3),(4)')
    connection.execute(
        'CREATE TABLE orders(order_id BIGINT PRIMARY KEY, '
        'user_id BIGINT REFERENCES users(user_id), '
        f'value {sql_type}, ts TIMESTAMP)'
    )
    return connection


def _request(
    connection, target, context, end, *, indices=None, numerical_feature=False
):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(connection))
        if numerical_feature:
            schema['users']['feature'].stype = 'numerical'
        return (
            PredictiveQuery(f'PREDICT {target} FOR EACH users.user_id', schema)
            .materialize(
                indices=list(range(5)) if indices is None else indices,
                anchor_time=end + pd.Timedelta(days=1),
                context_anchor_time=context,
                num_context_samples=100,
            )
            .payloads[0]
        )


def _context(payload, at):
    table = payload['context']['instance_table']
    assert table['columns'] == [
        'instance_id',
        '__kumo_entity_ref_0',
        '__kumo_anchor_time',
        'TARGET',
    ]
    rows = table['rows']
    assert len({row[0] for row in rows}) == len(rows)
    assert len({row[1] for row in rows}) == len(rows)
    assert all(pd.Timestamp(row[2]) == at.tz_localize('UTC') for row in rows)
    return {row[1]: row[3] for row in rows}


def _boundary_rows(connection, start, end):
    connection.executemany(
        'INSERT INTO orders VALUES (?, ?, ?, ?)',
        [
            (0, 0, 1, start),
            (1, 0, 2, start + pd.Timedelta(microseconds=1)),
            (2, 0, 4, end),
            (3, 0, 8, end + pd.Timedelta(microseconds=1)),
            (4, 0, 16, None),
            (5, 1, 7, end),
            (6, 2, None, end),
        ],
    )


@pytest.mark.parametrize('sql_type', ['BIGINT', 'DOUBLE', 'DECIMAL(38,18)'])
@pytest.mark.parametrize('aggregate', ['SUM', 'COUNT', 'AVG', 'MIN', 'MAX'])
@pytest.mark.parametrize('year', [2023, 2024])
def test_month_labels_use_source_values_and_preserve_empty_groups(
    sql_type, aggregate, year
):
    context = pd.Timestamp(year=year, month=1, day=31)
    end = context + pd.DateOffset(months=1)
    with _database(sql_type) as connection:
        _boundary_rows(connection, context, end)
        payload = _request(
            connection, f'{aggregate}(orders.value, 0, 1, months)', context, end
        )
    assert payload['task']['kind'] == 'regression'
    assert payload['task']['target'] == {
        'column_name': 'TARGET',
        'dtype': 'float32',
    }
    expected = {
        'SUM': {0: 6.0, 1: 7.0, 2: 0.0, 3: 0.0, 4: 0.0},
        'COUNT': {0: 2.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0},
        'AVG': {0: 3.0, 1: 7.0},
        'MIN': {0: 2.0, 1: 7.0},
        'MAX': {0: 4.0, 1: 7.0},
    }[aggregate]
    assert _context(payload, context) == expected
    assert all(type(v) is float for v in _context(payload, context).values())


@pytest.mark.parametrize(
    'context,start_count,end_count,unit',
    [
        ('2024-01-31', 1, 2, 'months'),
        ('2024-02-29', 0, 12, 'months'),
        ('2024-01-31', 0, 1, 'months'),
        ('2024-01-01', 0, 2, 'weeks'),
    ],
)
def test_calendar_offsets_include_end_but_exclude_start(
    context, start_count, end_count, unit
):
    context = pd.Timestamp(context)
    start = context + pd.DateOffset(**{unit: start_count})
    end = context + pd.DateOffset(**{unit: end_count})
    with _database() as connection:
        _boundary_rows(connection, start, end)
        if start > context:
            connection.execute(
                'INSERT INTO orders VALUES (99, 4, 0, ?)', [context]
            )
        payload = _request(
            connection,
            f'SUM(orders.value, {start_count}, {end_count}, {unit})',
            context,
            end,
        )
    assert _context(payload, context) == {
        0: 6.0,
        1: 7.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
    }


@pytest.mark.parametrize(
    'symbol,compare',
    [
        ('>', operator.gt),
        ('>=', operator.ge),
        ('=', operator.eq),
        ('!=', operator.ne),
        ('<', operator.lt),
        ('<=', operator.le),
    ],
)
def test_all_comparisons_produce_boolean_labels(symbol, compare):
    context = pd.Timestamp('2024-01-31')
    end = context + pd.DateOffset(months=1)
    with _database() as connection:
        _boundary_rows(connection, context, end)
        payload = _request(
            connection,
            f'COUNT(orders.*, 0, 1, months) {symbol} 1',
            context,
            end,
        )
    assert payload['task']['kind'] == 'binary_classification'
    assert payload['task']['target'] == {
        'column_name': 'TARGET',
        'dtype': 'bool',
        'classes': ['false', 'true'],
        'positive_class': 'true',
    }
    got = _context(payload, context)
    assert got == {
        key: compare(count, 1)
        for key, count in {0: 2, 1: 1, 2: 1, 3: 0, 4: 0}.items()
    }
    assert all(type(value) is bool for value in got.values())


@pytest.mark.parametrize('aggregate', ['SUM', 'AVG'])
def test_exact_decimal_cancellation_and_midpoints_are_rounded_once(aggregate):
    context = pd.Timestamp('2024-03-01')
    end = context + pd.Timedelta(days=30)
    terms = {
        0: ['1000000000000000', '0.1', '-1000000000000000'],
        1: ['1.000000059604644776'],
        2: ['1.000000059604644775'],
        3: ['2'],
    }
    with _database('DECIMAL(38,18)') as connection:
        rows = [
            (len_done, entity, value, context if entity == 4 else end)
            for len_done, (entity, value) in enumerate(
                [(k, v) for k, vs in terms.items() for v in vs] + [(4, '0')]
            )
        ]
        connection.executemany('INSERT INTO orders VALUES (?, ?, ?, ?)', rows)
        payload = _request(
            connection, f'{aggregate}(orders.value, 0, 30, days)', context, end
        )
    expected = {
        key: _float32_once(
            sum(map(Fraction, values))
            / (len(values) if aggregate == 'AVG' else 1)
        )
        for key, values in terms.items()
    }
    if aggregate == 'SUM':
        expected[4] = 0.0
    got = _context(payload, context)
    assert got == expected
    assert np.float32(got[1]).view(np.uint32) == 0x3F800001
    assert np.float32(got[2]).view(np.uint32) == 0x3F800000


def test_float32_compensation_crosses_the_public_driver_boundary():
    context = pd.Timestamp('2024-03-01')
    end = context + pd.Timedelta(days=30)
    with _database() as connection:
        values = [100_000_000.0] + [1.0] * 8 + [-100_000_000.0]
        connection.executemany(
            'INSERT INTO orders VALUES (?, 0, ?, ?)',
            [(i, value, end) for i, value in enumerate(values)],
        )
        connection.execute(
            'INSERT INTO orders VALUES (100, 4, 0, ?)', [context]
        )
        for aggregate, expected in [
            ('SUM', 8.0),
            ('AVG', float(np.float32(0.8))),
        ]:
            payload = _request(
                connection,
                f'{aggregate}(orders.value, 0, 30, days)',
                context,
                end,
            )
            assert _context(payload, context)[0] == expected


def test_all_null_average_refuses_but_sum_keeps_zero_labels():
    context = pd.Timestamp('2024-01-31')
    end = context + pd.DateOffset(months=1)
    with _database() as connection:
        connection.executemany(
            'INSERT INTO orders VALUES (?, 0, NULL, ?)',
            [(0, context), (1, end)],
        )
        summed = _request(
            connection, 'SUM(orders.value, 0, 1, months)', context, end
        )
        assert _context(summed, context) == dict.fromkeys(range(5), 0.0)
        with pytest.raises(StructuredDataError) as error:
            _request(
                connection, 'AVG(orders.value, 0, 1, months)', context, end
            )
        assert error.value.code == 'INVALID_REQUEST'
        assert 'collect any context examples' in str(error.value).lower()


@pytest.mark.parametrize(
    'aggregate,sql_type,values,other,symbol,constant,first,second',
    [
        ('SUM', 'DOUBLE', ['0.1', '0.2'], '2', '=', '0.3', True, False),
        ('MAX', 'DOUBLE', ['0.1'], '2', '=', '0.1', True, False),
        (
            'SUM',
            'DOUBLE',
            ['16777216', '9'],
            '16777226',
            '>',
            '16777224',
            False,
            True,
        ),
        ('MAX', 'BIGINT', ['7'], '8', '>', '7.5', False, True),
        (
            'AVG',
            'DOUBLE',
            ['1.00000011920928955078125'],
            '2',
            '=',
            '1.0000001',
            True,
            False,
        ),
        ('SUM', 'DECIMAL(38,18)', ['0.1', '0.2'], '1', '>', '0.3', False, True),
    ],
)
def test_aggregate_conditions_use_the_float32_label_representation(
    aggregate, sql_type, values, other, symbol, constant, first, second
):
    context = pd.Timestamp('2024-03-01')
    end = context + pd.Timedelta(days=30)
    with _database(sql_type) as connection:
        rows = [(i, 0, value, end) for i, value in enumerate(values)]
        rows += [(100, 1, other, end), (101, 4, '0', context)]
        connection.executemany('INSERT INTO orders VALUES (?, ?, ?, ?)', rows)
        payload = _request(
            connection,
            f'{aggregate}(orders.value, 0, 30, days) {symbol} {constant}',
            context,
            end,
        )
    expected = {0: first, 1: second}
    if aggregate == 'SUM':
        expected.update({2: False, 3: False, 4: False})
    assert _context(payload, context) == expected
    assert payload['task']['kind'] == 'binary_classification'
    assert all(type(v) is bool for v in _context(payload, context).values())


@pytest.mark.parametrize(
    'sql_type,values,expression,kind',
    [
        (
            'VARCHAR',
            ['b', 'a', 'c', 'é', 'b'],
            'feature',
            'multiclass_classification',
        ),
        (
            'BOOLEAN',
            [True, False, None, True, False],
            'feature',
            'multiclass_classification',
        ),
        (
            'DECIMAL(38,18)',
            ['1.5', '2', None, '1.500000000000000001', '-3'],
            'feature',
            'multiclass_classification',
        ),
        (
            'DOUBLE',
            [0.1, 16777217.0, None, -0.0, 1e-40],
            'feature',
            'regression',
        ),
        (
            'DOUBLE',
            [0.1, 0.2, None, 0.1000001, 0.0999999],
            'feature > 0.1',
            'binary_classification',
        ),
    ],
)
def test_static_labels_keep_types_exact_classes_and_exclude_scored_entities(
    sql_type, values, expression, kind
):
    context = pd.Timestamp('2024-03-01')
    indices = (
        [0, 2]
        if kind == 'regression'
        else [1, 2]
        if kind == 'binary_classification'
        else [0, 1]
    )
    with _database() as connection:
        connection.execute(f'ALTER TABLE users ADD COLUMN feature {sql_type}')
        connection.executemany(
            'UPDATE users SET feature=? WHERE user_id=?',
            [(value, i) for i, value in enumerate(values)],
        )
        connection.execute('INSERT INTO orders VALUES (0, 0, 1, ?)', [context])
        source = dict(
            connection.execute('SELECT user_id, feature FROM users').fetchall()
        )
        payload = _request(
            connection,
            'users.' + expression,
            context,
            context,
            indices=indices,
            numerical_feature=kind == 'regression',
        )
    got = _context(payload, context)
    assert set(got) == {
        key
        for key, value in source.items()
        if key not in indices and value is not None
    }
    assert payload['task']['kind'] == kind
    target = payload['task']['target']
    if kind == 'regression':
        assert np.float32(got[1]).view(np.uint32) == 0x4B800000
    elif kind == 'binary_classification':
        assert got[0] is False  # Both0.1 operands must be compared at float32.
    for entity, label in got.items():
        value = source[entity]
        if kind == 'regression':
            assert type(label) is float
            assert np.float32(label).tobytes() == np.float32(value).tobytes()
        elif kind == 'binary_classification':
            assert type(label) is bool
            assert label == bool(np.float32(value) > np.float32(0.1))
        else:
            assert type(label) is str
            assert label == str(value)
    if kind == 'multiclass_classification':
        assert target['dtype'] == 'string'
        classes = {str(value) for value in source.values() if value is not None}
        assert set(target['classes']) == classes
        assert len(target['classes']) == len(classes)
        if sql_type.startswith('DECIMAL'):
            assert (
                len(classes) == 4
            )  # The two distinct values near1.5 must not merge.
    assert (
        'feature'
        not in payload['context']['related_tables']['users']['columns']
    )
