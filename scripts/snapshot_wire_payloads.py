#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record the exact bytes each canonical request puts on the wire.

The public API snapshot records what a caller can *name*. This records what the
endpoint actually receives, which is the other half of the contract and the half
a refactor is far more likely to move without anyone noticing: a changed dtype
string, a column encoded one way instead of another, a task kind derived rather
than stated.

``metadata.request_id`` is generated per call and is replaced with a fixed value
so the snapshot compares. Nothing else is normalized -- the point is to notice.

Run with ``--write`` to re-record after an intended change.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import pandas as pd
from sdm_client import PredictiveQuery, Request, Schema
from sdm_client.adapters.relational import RelationalAdapter
from sdm_client.adapters.tabular import TabularAdapter

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
    # Long enough that a 30-day forward aggregation has history behind every
    # anchor the sampler picks; a short frame is refused by the engine, which
    # is correct and is not what this snapshot is for.
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


def _tabular_cases() -> dict[str, list[dict[str, Any]]]:
    people = _people()
    unlabelled = people.head(3).drop(columns=['churn'])
    adapter = TabularAdapter()
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
    }
    return {
        name: [
            _normalize(payload)
            for payload in adapter.prediction_plan(request, ('prediction',))
        ]
        for name, request in cases.items()
    }


def _relational_cases() -> dict[str, list[dict[str, Any]]]:
    schema = Schema.from_data({'people': _people(), 'orders': _orders()})
    request = PredictiveQuery(
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH people.user_id',
        schema,
    ).materialize(indices=[0, 1, 2])
    adapter = RelationalAdapter()
    return {
        'relational/pql-count-aggregation': [
            _normalize(payload)
            for payload in adapter.prediction_plan(request, ('prediction',))
        ]
    }


def collect() -> dict[str, Any]:
    recorded: dict[str, Any] = dict(_tabular_cases())
    try:
        import sdm_relational_engine  # noqa: F401
    except ImportError:  # pragma: no cover - the extra is not installed
        return recorded
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
    # Only compare the cases this environment can build, so an install without
    # the relational extra does not report a difference it cannot cause.
    comparable = {name: recorded[name] for name in current if name in recorded}
    if comparable == current and set(current) <= set(recorded):
        print(
            f'wire payloads match the recorded snapshot ({len(current)} cases)'
        )
        return 0

    print('wire payloads differ from the recorded snapshot', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
