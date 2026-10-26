# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The original metric-view time choice reaches the public warning surface.

The real native core reads a small metric YAML definition and typed source
rows from a bounded provider fixture; no SQL, account or model is contacted.
"""

from __future__ import annotations

import datetime as dt
import warnings
from typing import Any

import pyarrow as pa
import pytest
from sdm_connectors import DatabricksSource

from sdm_client import Schema, SDMWarning, _routing
from sdm_client._rust import providers


_VIEW = """
version: 0.1
source: cat.shop.orders
parameters:
  - {name: min_price, data_type: DOUBLE, default: 50}
filter: source.price >= min_price
joins:
  - name: users
    source: cat.shop.users
    on: source.user_id = users.user_id
dimensions:
  - {name: order_id, expr: source.order_id}
  - {name: user_id, expr: source.user_id}
  - {name: ts, expr: "DATE_TRUNC('MONTH', source.ts)"}
  - {name: price, expr: source.price}
measures:
  - {name: revenue, expr: SUM(source.price)}
"""


class _Connection:
    __module__ = 'databricks.sql.client'


_Connection.__name__ = 'Connection'


class _MetricProvider:
    """Answer only the genuine describe/preview requests in this journey."""

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def __call__(self, step: dict[str, Any], keys: Any = None) -> pa.Table:
        request = step['request']
        assert step['plan']['plan'] == 'sql'
        assert all(
            item['dialect'] == 'databricks'
            for item in step['plan']['statements']
        )
        if request['request'] == 'key_counts':
            # Every key in this fixed three-row source is unique and held.
            count = len(request['keys'])
            return pa.table(
                {
                    'k': list(range(count)),
                    'n': [3] * count,
                    'd': [3] * count,
                    'z': [0] * count,
                }
            )
        if request['request'] == 'describe':
            part = request['part']
            if part == 'view_definition':
                return pa.table(
                    {
                        'col_name': ['Type', 'View Text'],
                        'data_type': ['METRIC_VIEW', _VIEW],
                        'comment': pa.array([None, None], pa.string()),
                    }
                )
            assert part in ('columns', 'keys')
            if part == 'keys':
                return pa.table(
                    {
                        name: pa.array([], pa.string())
                        for name in ('kind', 'name', 'column')
                    }
                )
            table = request['tables'][0]['source'][-1]
            columns = (
                [
                    ('order_id', 'BIGINT'),
                    ('user_id', 'BIGINT'),
                    ('ts', 'TIMESTAMP'),
                    ('price', 'DOUBLE'),
                ]
                if table == 'orders'
                else [('user_id', 'BIGINT')]
            )
            return pa.table(
                {
                    'column_name': [name for name, _ in columns],
                    'full_data_type': [kind for _, kind in columns],
                    'is_nullable': ['NO'] * len(columns),
                    'table_catalog': ['cat'] * len(columns),
                    'table_schema': ['shop'] * len(columns),
                    'table_name': [table] * len(columns),
                    'table_type': ['MANAGED'] * len(columns),
                }
            )
        assert request['request'] == 'preview', request
        table = request['table']
        raw = pa.table(
            {
                'order_id': [10, 11, 12],
                'user_id': [1, 2, 3],
                'price': [60.0, 70.0, 80.0],
                'ts': pa.array(
                    [dt.datetime(2024, month, 3, 12) for month in (1, 2, 3)],
                    pa.timestamp('us'),
                ),
            }
        )
        selected = {}
        for name in request['columns']:
            binding = next(
                (c for c in table.get('columns', []) if c['name'] == name), None
            )
            if binding is not None and binding.get('expression') is not None:
                assert name == 'ts'
                selected[name] = pa.array(
                    [dt.datetime(2024, month, 1) for month in (1, 2, 3)],
                    pa.timestamp('us'),
                )
            else:
                selected[name] = raw[(binding or {}).get('column') or name]
        return pa.table(selected)


def _metric_source(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    monkeypatch.setattr(providers, 'DatabricksSqlProvider', _MetricProvider)
    return DatabricksSource(
        _Connection(), catalog='cat', schema='shop'
    ).semantic_model('metrics')


@pytest.mark.parametrize(
    ('tables', 'time_columns', 'named', 'chosen', 'warning', 'note'),
    [
        (None, None, 'orders', 'ts_raw', True, True),
        (
            {'sales': 'orders', 'people': 'users'},
            None,
            'sales',
            'ts_raw',
            True,
            True,
        ),
        (['orders'], None, 'orders', 'ts_raw', False, True),
        (['users'], None, 'users', None, False, False),
        (None, {'orders': 'ts'}, 'orders', 'ts', False, False),
        (None, {'orders': 'ts_raw'}, 'orders', 'ts_raw', False, True),
        (None, {'orders': None}, 'orders', None, False, False),
    ],
)
def test_the_public_metric_time_warning_matches_the_schema_that_survives(
    monkeypatch: pytest.MonkeyPatch,
    tables: Any,
    time_columns: Any,
    named: str,
    chosen: str | None,
    warning: bool,
    note: bool,
) -> None:
    source = _metric_source(monkeypatch)
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        schema = Schema.from_source(
            source,
            tables,
            time_columns=time_columns,
            check_keys=False,
        )
    assert schema[named].time_column == chosen
    said = [
        str(item.message)
        for item in shown
        if issubclass(item.category, SDMWarning)
        and 'was taken as the time column rather than' in str(item.message)
    ]
    if warning:
        assert said == [
            '[TIME_COLUMNS_GUESSED] Some time columns were inferred, or '
            'refused, on a guess about what a column holds:\n'
            f"- {named}.ts_raw was taken as the time column rather than 'ts', "
            'which truncates or casts it, since it says when a row happened '
            'more precisely. If a question should count by it, name it: '
            f"time_columns={{{named!r}: 'ts'}}."
        ]
    else:
        assert said == []
    time_notes = [n for n in schema.conversion_notes if 'time column' in n]
    assert bool(time_notes) is note
    if note:
        assert named in time_notes[0]


def test_the_public_warning_can_abort_and_the_source_can_be_read_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _metric_source(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter('error', SDMWarning)
        with pytest.raises(
            SDMWarning, match='was taken as the time column rather than'
        ):
            Schema.from_source(source, check_keys=False)
        schema = Schema.from_source(
            source,
            time_columns={'orders': 'ts'},
            check_keys=False,
        )
    assert schema['orders'].time_column == 'ts'
