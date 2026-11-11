#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise canonical requests and validate their wire behavior without snapshots."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sdm_client import PredictiveQuery, Request, Schema, SDMClient

_FIXED_REQUEST_ID = '<request-id>'


def _people(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'user_id': range(n),
            'age': [20.0 + i for i in range(n)],
            'segment': ['north' if i % 3 else 'south' for i in range(n)],
            'churn': [i % 2 for i in range(n)],
        }
    )


def _orders(n: int = 240) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'order_id': range(n),
            'user_id': [i % 12 for i in range(n)],
            'price': [10.0 + i for i in range(n)],
            'ordered_at': pd.date_range('2026-01-01', periods=n, freq='D'),
        }
    )


def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
    recorded = json.loads(json.dumps(payload, default=str))
    metadata = recorded.get('metadata')
    if isinstance(metadata, dict) and 'request_id' in metadata:
        assert (
            isinstance(metadata['request_id'], str) and metadata['request_id']
        )
        metadata['request_id'] = _FIXED_REQUEST_ID
    return recorded


def _output_sets(request: Request):
    sets = [('', ('prediction',))]
    if request.task.kind != 'regression':
        sets.append(('#default-outputs', None))
    return sets


def _capture(request: Request, model: str, outputs):
    """Record actual transport bytes; return a minimal valid scripted answer."""

    class Transport:
        url = 'http://snapshot.invalid'

        def __init__(self):
            self.calls = []

        def predict_body(self, body):
            payload = json.loads(body)
            self.calls.append(_normalize(payload))
            table = payload['predict']['instance_table']
            context = payload['context']['instance_table']
            target = payload['task']['target']['column_name']
            value = context['rows'][0][context['columns'].index(target)]
            predictions = []
            for i, row in enumerate(table['rows']):
                item = {'row_index': i, 'prediction': value}
                if model == 'kumo-relational':
                    item['id'] = str(row[table['columns'].index('instance_id')])
                if 'probabilities' in payload['output']['fields']:
                    item['probabilities'] = {str(value): 1.0}
                predictions.append(item)
            return json.dumps(
                {
                    'id': 'snapshot',
                    'model': model,
                    'metadata': {},
                    'predictions': predictions,
                }
            ).encode()

        def predict(self, payload):
            return json.loads(self.predict_body(json.dumps(payload).encode()))

        def close(self):
            pass

    transport = Transport()
    with SDMClient(url=transport.url) as client:
        client._transport.close()
        client._transport = transport
        result = client.predict(request, model=model, outputs=outputs)
    assert len(result) == request.predict_rows
    return transport.calls


def _tabular_cases() -> dict[str, list[dict[str, Any]]]:
    people = _people()
    unlabelled = people.head(3).drop(columns=['churn'])
    cases = {
        'tabular/direct-classification': Request(
            context=people,
            predict=unlabelled,
            target='churn',
            task='classification',
        ),
        'tabular/direct-regression': Request(
            context=people,
            predict=people.head(3).drop(columns=['age']),
            target='age',
            task='regression',
        ),
        'tabular/pql-predict-frame': PredictiveQuery(
            'PREDICT people.churn FOR EACH people.user_id',
            {'people': people},
        ).materialize(predict=unlabelled),
        # A query whose task is *derived* as a regression. Every other case
        # either states the task or derives a classification, so the branch a
        # continuous target takes was recorded nowhere: flipping its return
        # value moved no golden file.
        'tabular/pql-derived-regression': PredictiveQuery(
            'PREDICT people.age FOR EACH people.user_id',
            {'people': people},
        ).materialize(predict=people.head(3).drop(columns=['age'])),
        # The third derivation branch: a categorical target, classified
        # without any of the two-whole-numbers reasoning. With the four above
        # it, every path through _derive_task_kind is now on record.
        'tabular/pql-derived-classification': PredictiveQuery(
            'PREDICT people.segment FOR EACH people.user_id',
            {'people': people},
        ).materialize(predict=people.head(3).drop(columns=['segment'])),
    }
    recorded: dict[str, list[dict[str, Any]]] = {}
    for name, request in cases.items():
        for label, outputs in _output_sets(request):
            recorded[f'{name}{label}'] = _capture(
                request, 'kumo-tabular', outputs
            )
    return recorded


def _relational_cases() -> dict[str, list[dict[str, Any]]]:
    schema = Schema.from_data({'people': _people(), 'orders': _orders()})
    request = PredictiveQuery(
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH people.user_id',
        schema,
    ).materialize(indices=[0, 1, 2])
    recorded: dict[str, list[dict[str, Any]]] = {}
    for label, outputs in _output_sets(request):
        recorded[f'relational/pql-count-aggregation{label}'] = _capture(
            request, 'kumo-relational', outputs
        )
    return recorded


def collect() -> dict[str, Any]:
    recorded: dict[str, Any] = dict(_tabular_cases())
    recorded.update(_relational_cases())
    return recorded


def _rows(table, columns):
    assert table['format'] == 'arrays'
    assert table['columns'] == columns
    assert all(len(row) == len(columns) for row in table['rows'])
    types = {
        'age': float,
        'price': float,
        'TARGET': float,
        'segment': str,
        '__kumo_anchor_time': str,
        'ordered_at': str,
    }
    for row in table['rows']:
        assert all(
            type(value) is types.get(col, int)
            for col, value in zip(columns, row, strict=True)
        )
    return table['rows']


def assert_behavior(cases):
    """Expected source values and task rules, independent of captured outputs."""
    targets = {
        'direct-classification': ('churn', 'classification', 'int64'),
        'direct-regression': ('age', 'regression', 'float64'),
        'pql-predict-frame': ('churn', 'classification', 'int64'),
        'pql-derived-regression': ('age', 'regression', 'float64'),
        'pql-derived-classification': ('segment', 'classification', 'string'),
    }
    names = {'relational/pql-count-aggregation'}
    for name, (_, kind, _) in targets.items():
        names.add('tabular/' + name)
        if kind == 'classification':
            names.add('tabular/' + name + '#default-outputs')
    assert set(cases) == names
    columns = ['user_id', 'age', 'segment', 'churn']
    people = [
        [i, 20.0 + i, 'north' if i % 3 else 'south', i % 2] for i in range(12)
    ]
    sections = {
        'model',
        'task',
        'context',
        'predict',
        'schema',
        'output',
        'metadata',
    }
    for name, calls in cases.items():
        assert len(calls) == 1, name
        body = calls[0]
        if name.startswith('relational/'):
            assert set(body) == sections | {'inference'}, name
            _assert_relational(body, people)
            continue
        assert set(body) == sections, name
        assert body['metadata'] == {'request_id': _FIXED_REQUEST_ID}
        target, kind, dtype = targets[
            name.removeprefix('tabular/').split('#')[0]
        ]
        assert body['model'] == 'kumo-tabular'
        assert body['task'] == {
            'kind': kind,
            'target': {'column_name': target, 'dtype': dtype},
        }
        assert body['output'] == {
            'fields': (
                ['prediction', 'probabilities']
                if '#default-outputs' in name
                else ['prediction']
            )
        }
        start = 3 if '/pql-' in name else 0
        assert (
            _rows(body['context']['instance_table'], columns) == people[start:]
        )
        keep = [i for i, col in enumerate(columns) if col != target]
        assert _rows(
            body['predict']['instance_table'], [columns[i] for i in keep]
        ) == [[row[i] for i in keep] for row in people[:3]]
        assert set(body['schema']) == {'instance_table'}
        assert (
            set(body['context']) == set(body['predict']) == {'instance_table'}
        )
        schema = body['schema']['instance_table']
        assert set(schema) == {'primary_key', 'columns'}
        assert schema['primary_key'] == ['user_id']
        assert set(schema['columns']) == set(columns)
        for col, typ in zip(
            columns, ['int64', 'float64', 'string', 'int64'], strict=True
        ):
            stype = (
                'ID'
                if col == 'user_id'
                else 'categorical'
                if col == 'segment' or (col == 'churn' and target == 'churn')
                else 'numerical'
            )
            assert schema['columns'][col] == {'dtype': typ, 'stype': stype}


def _assert_relational(body, people):
    assert body['model'] == 'kumo-relational'
    assert body['task'] == {
        'kind': 'regression',
        'target': {'column_name': 'TARGET', 'dtype': 'float32'},
        'anchor_time_column': '__kumo_anchor_time',
        'entity_table_names': ['people'],
    }
    assert body['output'] == {
        'fields': ['prediction'],
        'prediction_statistic': 'median',
    }
    assert body['inference'] == {
        'run_mode': 'fast',
        'use_prediction_time': False,
    }
    assert body['metadata'] == {
        'source_query': (
            'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH people.user_id'
        ),
        'num_context_examples': 84,
        'num_prediction_examples': 3,
        'request_id': _FIXED_REQUEST_ID,
    }
    start = pd.Timestamp('2026-01-01', tz='UTC')
    seen = set()
    for half in ('context', 'predict'):
        cols = ['instance_id', '__kumo_entity_ref_0', '__kumo_anchor_time']
        if half == 'context':
            cols += ['TARGET']
        rows = _rows(body[half]['instance_table'], cols)
        assert len(rows) == (84 if half == 'context' else 3)
        assert (
            [r[0] for r in rows] == list(range(84))
            if half == 'context'
            else [r[0] for r in rows] == [84, 85, 86]
        )
        expected_people, expected_orders = [], []
        for row in rows:
            instance, user, stamp = row[:3]
            day = (pd.Timestamp(stamp) - start).days
            assert stamp == (start + pd.Timedelta(days=day)).strftime(
                '%Y-%m-%dT%H:%M:%S.000000Z'
            )
            assert 0 <= user < 12
            if half == 'context':
                assert (user, day) not in seen
                seen.add((user, day))
                assert row[3] == sum(
                    1
                    for order in range(240)
                    if order % 12 == user and day < order <= day + 30
                )
            else:
                assert user == instance - 84 and day == 239
            expected_people.append([instance, *people[user]])
            expected_orders.extend(
                [
                    instance,
                    order,
                    user,
                    10.0 + order,
                    (start + pd.Timedelta(days=order)).strftime(
                        '%Y-%m-%dT%H:%M:%S.000000Z'
                    ),
                ]
                for order in range(240)
                if order % 12 == user and order <= day
            )
        related = body[half]['related_tables']
        assert set(related) == {'people', 'orders'}
        assert (
            _rows(
                related['people'],
                ['instance_id', 'user_id', 'age', 'segment', 'churn'],
            )
            == expected_people
        )
        assert (
            _rows(
                related['orders'],
                ['instance_id', 'order_id', 'user_id', 'price', 'ordered_at'],
            )
            == expected_orders
        )
    assert seen == {
        (user, day) for user in range(12) for day in range(29, 210, 30)
    }
    context = body['context']['instance_table']['rows']
    # Independent NumPy PCG64 oracle for seeded context ordering.
    rng = np.random.default_rng(42)
    assert [r[1] for r in context] == [
        int(user) for _ in range(7) for user in rng.permutation(12)
    ]
    assert [r[2] for r in context] == sorted(
        [r[2] for r in context], reverse=True
    )
    schema = body['schema']
    assert schema['instance_table']['primary_key'] == 'instance_id'
    assert set(schema['related_tables']) == {'people', 'orders'}
    for name, key in [('people', 'user_id'), ('orders', 'order_id')]:
        assert schema['related_tables'][name]['primary_key'] == [
            'instance_id',
            key,
        ]
    layouts = {
        'instance_table': [
            'instance_id',
            '__kumo_entity_ref_0',
            '__kumo_anchor_time',
            'TARGET',
        ],
        'people': ['instance_id', 'user_id', 'age', 'segment', 'churn'],
        'orders': ['instance_id', 'order_id', 'user_id', 'price', 'ordered_at'],
    }
    for name, columns in layouts.items():
        table = (
            schema['instance_table']
            if name == 'instance_table'
            else schema['related_tables'][name]
        )
        assert set(table['columns']) == set(columns)
        for col in columns:
            dtype = (
                'timestamp[us]'
                if col in ('__kumo_anchor_time', 'ordered_at')
                else 'string'
                if col == 'segment'
                else 'float32'
                if col == 'TARGET'
                else 'float64'
                if col in ('age', 'price')
                else 'int64'
            )
            expected = {'dtype': dtype}
            if col in ('instance_id', '__kumo_anchor_time') or (name, col) in [
                ('people', 'user_id'),
                ('orders', 'order_id'),
            ]:
                expected['nullable'] = False
            if (name, col) not in [
                ('people', 'user_id'),
                ('orders', 'order_id'),
            ]:
                expected['stype'] = (
                    'ID'
                    if col in ('instance_id', '__kumo_entity_ref_0', 'user_id')
                    else 'timestamp'
                    if dtype == 'timestamp[us]'
                    else 'categorical'
                    if col == 'segment'
                    else 'numerical'
                )
            assert table['columns'][col] == expected, (name, col)
    assert schema['relationships'] == [
        {
            'source_columns': ['__kumo_entity_ref_0'],
            'target_columns': ['user_id'],
            'target_table': 'people',
        },
        {
            'source_columns': ['user_id'],
            'source_table': 'orders',
            'target_columns': ['user_id'],
            'target_table': 'people',
        },
    ]


def main() -> int:
    assert_behavior(collect())
    print('Canonical wire journeys satisfy source-value and task assertions')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
