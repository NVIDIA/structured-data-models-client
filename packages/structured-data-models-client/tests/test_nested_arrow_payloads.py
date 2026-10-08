# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nested warehouse features retain their existing public string cells."""

import datetime

import pyarrow as pa
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import PredictiveQuery, Schema

duckdb = pytest.importorskip('duckdb')


@pytest.mark.parametrize(
    'kind',
    [
        'struct',
        'map_sorted',
        'map_unsorted',
        'struct_list',
        'map_struct',
        'struct_temporal',
        'struct_deep',
    ],
)
def test_nested_warehouse_features_are_not_replaced_with_null(kind):
    if kind == 'struct':
        data_type = pa.struct([('number', pa.int64()), ('label', pa.string())])
        values = [
            {'number': 70, 'label': 'one'},
            {'number': 63, 'label': 'two'},
            {'number': None, 'label': 'three'},
            None,
            {'number': 65, 'label': None},
            {'number': 73, 'label': 'four'},
        ]
        expected = [
            "{'number': 70, 'label': 'one'}",
            "{'number': 63, 'label': 'two'}",
            "{'number': None, 'label': 'three'}",
            None,
            "{'number': 65, 'label': None}",
            "{'number': 73, 'label': 'four'}",
        ]
    elif kind == 'struct_list':
        data_type = pa.struct([('numbers', pa.list_(pa.int64()))])
        values = [
            {'numbers': [70, None]},
            {'numbers': []},
            {'numbers': None},
            None,
            {'numbers': [65]},
            {'numbers': [73, 74]},
        ]
        expected = [
            "{'numbers': [70, None]}",
            "{'numbers': []}",
            "{'numbers': None}",
            None,
            "{'numbers': [65]}",
            "{'numbers': [73, 74]}",
        ]
    elif kind == 'map_struct':
        data_type = pa.map_(pa.string(), pa.struct([('number', pa.int64())]))
        values = [
            [('one', {'number': 70})],
            [('one', {'number': None})],
            [],
            None,
            [('one', None)],
            [('one', {'number': 73})],
        ]
        expected = [
            "[['one', {'number': 70}]]",
            "[['one', {'number': None}]]",
            '[]',
            None,
            "[['one', None]]",
            "[['one', {'number': 73}]]",
        ]
    elif kind == 'struct_deep':
        nested_type = pa.int64()
        nested_value = 70
        for _ in range(48):
            nested_type = pa.list_(nested_type)
            nested_value = [nested_value]
        data_type = pa.struct([('number', pa.int64()), ('nested', nested_type)])
        values = [
            {'number': 70 + index, 'nested': nested_value} for index in range(6)
        ]
        expected = [repr(row) for row in values]
    elif kind == 'struct_temporal':
        data_type = pa.struct(
            [
                ('number', pa.int64()),
                ('day', pa.date32()),
                ('microsecond', pa.timestamp('us')),
                ('nanosecond', pa.timestamp('ns')),
                ('label', pa.string()),
            ]
        )
        day = datetime.date(2020, 1, 2)
        instant = datetime.datetime(2020, 1, 2, 3, 4, 5, 123456)
        values = [
            {
                'day': day,
                'microsecond': instant,
                'nanosecond': instant,
                'label': 'a\'b"c',
            },
            {
                'day': None,
                'microsecond': instant,
                'nanosecond': None,
                'label': 'line\nbreak',
            },
            {
                'day': day,
                'microsecond': None,
                'nanosecond': instant,
                'label': '\\slash',
            },
            None,
            {
                'day': None,
                'microsecond': None,
                'nanosecond': None,
                'label': None,
            },
            {
                'day': day,
                'microsecond': instant,
                'nanosecond': instant,
                'label': 'plain',
            },
        ]
        values = [
            None if row is None else {'number': 70 + index, **row}
            for index, row in enumerate(values)
        ]
        expected = [
            None
            if row is None
            else repr(
                {
                    **row,
                    'nanosecond': None
                    if row['nanosecond'] is None
                    else '2020-01-02T03:04:05.123456Z',
                }
            )
            for row in values
        ]
    else:
        data_type = pa.map_(
            pa.string(), pa.int64(), keys_sorted=kind == 'map_sorted'
        )
        values = [
            [('one', 70), ('two', 71)],
            [('one', 63), ('two', 64)],
            [],
            None,
            [('one', None)],
            [('one', 73), ('two', 74)],
        ]
        expected = [
            "[['one', 70], ['two', 71]]",
            "[['one', 63], ['two', 64]]",
            '[]',
            None,
            "[['one', None]]",
            "[['one', 73], ['two', 74]]",
        ]
    table = pa.table(
        {
            'product_id': pa.array(range(1, 7), type=pa.int64()),
            'category': ['books', 'games', 'tools', 'books', 'games', 'tools'],
            'price': pa.array(values, type=data_type),
        }
    )
    connection = duckdb.connect()
    try:
        connection.register('input_frame', table)
        connection.execute('CREATE TABLE products AS SELECT * FROM input_frame')
        connection.unregister('input_frame')
        schema = Schema.from_source(DuckDBSource(connection))
        schema['products'].primary_key = 'product_id'
        request = PredictiveQuery(
            'PREDICT products.category FOR EACH products.product_id', schema
        ).materialize(indices=list(range(1, 7)), random_seed=7)
        seen = {}
        for payload in request.payloads:
            for half in ('context', 'predict'):
                rows = payload[half]['related_tables']['products']
                key_position = rows['columns'].index('product_id')
                feature_position = rows['columns'].index('price')
                for row in rows['rows']:
                    identity = int(row[key_position])
                    assert row[feature_position] == expected[identity - 1]
                    seen[identity] = row[feature_position]
        assert seen == dict(enumerate(expected, start=1))
    finally:
        connection.close()
