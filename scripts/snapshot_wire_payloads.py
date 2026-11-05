#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record what each canonical request puts on the wire.

``metadata.request_id`` is generated per call and normalized; nothing else is.
``--write`` re-records after an intended change.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import pandas as pd
from sdm_client import PredictiveQuery, Request, Schema, SDMClient

SNAPSHOT = (
    pathlib.Path(__file__).resolve().parent.parent
    / 'tests'
    / 'golden'
    / 'wire_payloads.json'
)

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()

    current = collect()
    if args.write:
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(
            json.dumps(current, indent=2, sort_keys=True) + '\n'
        )
        print(f'wrote {SNAPSHOT} ({len(current)} cases)')
        return 0

    if not SNAPSHOT.exists():
        print(f'no snapshot at {SNAPSHOT}; run with --write', file=sys.stderr)
        return 1

    recorded = json.loads(SNAPSHOT.read_text())

    expected = set(recorded)

    if set(current) != expected:
        for name in sorted(expected - set(current)):
            print(f'case no longer produced: {name}', file=sys.stderr)
        for name in sorted(set(current) - expected):
            print(f'case not in the snapshot: {name}', file=sys.stderr)
        print(
            'the set of canonical requests changed; re-record with --write if '
            'that was intended',
            file=sys.stderr,
        )
        return 1

    differing = [name for name in current if recorded[name] != current[name]]
    if not differing:
        print(
            f'wire payloads match the recorded snapshot ({len(current)} cases)'
        )
        return 0

    for name in differing:
        print(f'payload changed: {name}', file=sys.stderr)
    print('wire payloads differ from the recorded snapshot', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
