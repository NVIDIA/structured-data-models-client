# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""sdm-client's smoke test, for a clean install (P9.wheels).

Run as ``python -I client_smoke.py`` in an environment where ``sdm-client``
was installed from wheels. It needs nothing but what that install brought:
no checkout, no network, no database. It exits non-zero on the first failure
and on success prints one line starting ``SDM_CLIENT_SMOKE_OK``.

It checks:
- ``sdm_client`` and ``sdm_core`` are the installed ones, and ``sdm-client``
  requires ``sdm-core`` at its own version and ``pyarrow``, neither as an
  extra;
- a single-table question (``Request``) and a relational one
  (``Schema.from_data``, ``PredictiveQuery.materialize``) are asked and
  answered, through ``SDMClient.predict`` and a stand-in transport, and
  their frames read;
- neither ``predictive_query`` nor ``sdm_relational_engine`` was imported.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any


def fail(message: str) -> None:
    print(f'SDM_CLIENT_SMOKE_FAILED {message}', file=sys.stderr)
    sys.exit(1)


def check(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


class _Response:
    def __init__(self, body: Any) -> None:
        self.status_code = 200
        self.headers = {'Content-Type': 'application/json'}
        self._body = json.dumps(body).encode('utf-8')

    def iter_content(self, chunk_size: int = 1) -> Any:
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class StandIn:
    r"""Answers each prediction with one item per row asked about."""

    def __init__(self) -> None:
        self.bodies: list[Any] = []

    def post(self, url: str, data: bytes, **kwargs: Any) -> _Response:
        payload = json.loads(data)
        self.bodies.append(payload)
        table = payload['predict']['instance_table']
        columns = table['columns']
        keys = payload['schema']['instance_table'].get('primary_key') or []
        classification = 'classification' in payload['task']['kind']
        items = []
        for index, row in enumerate(table['rows']):
            if 'instance_id' in columns:
                row_id = str(row[columns.index('instance_id')])
            else:
                row_id = json.dumps([row[columns.index(k)] for k in keys])
            items.append(
                {
                    'row_index': index,
                    'id': row_id,
                    'prediction': '1' if classification else 1.5,
                }
            )
        return _Response(
            {
                'object': 'prediction.response',
                'id': f'smoke-{len(self.bodies)}',
                'model': payload['model'],
                'metadata': {},
                'predictions': items,
            }
        )

    def delete(self, url: str, **kwargs: Any) -> _Response:
        return _Response({})

    def close(self) -> None:
        return None


def client(session: StandIn) -> Any:
    from sdm_client import SDMClient

    made = SDMClient(url='http://localhost:8000')
    made._transport._session.close()
    made._transport._session = session
    return made


def check_install() -> None:
    from importlib.metadata import requires, version

    import sdm_client
    import sdm_core

    for module in (sdm_client, sdm_core):
        check(
            'site-packages' in (module.__file__ or ''),
            f'{module.__name__} is not installed: {module.__file__}',
        )
    base = [
        r
        for r in requires('structured-data-models-client') or []
        if 'extra ==' not in r
    ]
    pinned = [
        r for r in base if re.match(r'sdm[-_]core\s*==', r.replace(' ', ''))
    ]
    check(
        pinned
        and pinned[0].replace(' ', '').split('==')[1].split(';')[0]
        == version('structured-data-models-client'),
        f'sdm-client does not require sdm-core=={version("structured-data-models-client")}: {base}',
    )
    check(
        version('sdm-core') == version('structured-data-models-client'),
        f'sdm-core {version("sdm-core")} is not sdm-client {version("structured-data-models-client")}',
    )
    check(
        any(re.match(r'pyarrow\b', r) for r in base),
        f'sdm-client does not require pyarrow: {base}',
    )


def check_flows() -> None:
    import pandas as pd
    import sdm_client

    people = pd.DataFrame(
        {
            'user_id': range(12),
            'age': [20.0 + i for i in range(12)],
            'segment': ['south', 'north', 'north'] * 4,
            'churn': [0, 1] * 6,
        }
    )
    session = StandIn()
    tabular = client(session)
    try:
        request = sdm_client.Request(
            context=people,
            predict=people.head(3).drop(columns=['churn']),
            target='churn',
            task='classification',
        )
        result = tabular.predict(request, model='kumo-tabular')
        frame = result.to_pandas()
    finally:
        tabular.close()
    check(len(session.bodies) == 1, f'{len(session.bodies)} tabular requests')
    check(len(frame) == 3, f'the tabular frame has {len(frame)} rows')

    users = pd.DataFrame({'user_id': range(1, 21), 'age': range(20, 40)})
    orders = pd.DataFrame(
        {
            'order_id': range(1, 201),
            'user_id': [1 + i % 20 for i in range(200)],
            'quantity': [1 + i % 5 for i in range(200)],
            'ordered_at': pd.date_range('2024-01-01', periods=200, freq='D'),
        }
    )
    schema = sdm_client.Schema.from_data({'users': users, 'orders': orders})
    query = sdm_client.PredictiveQuery(
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id', schema
    )
    request = query.materialize(indices=[1, 2, 3], random_seed=7)
    check(bool(request.payloads), 'the relational request gathered nothing')
    session = StandIn()
    relational = client(session)
    try:
        result = relational.predict(request, model='kumo-relational')
        frame = result.to_pandas()
    finally:
        relational.close()
    check(len(session.bodies) >= 1, 'no relational request was sent')
    check(len(frame) == 3, f'the relational frame has {len(frame)} rows')

    loaded = [
        m
        for m in sys.modules
        if m.split('.')[0] in ('predictive_query', 'sdm_relational_engine')
    ]
    check(not loaded, f'ported packages were imported: {sorted(loaded)[:5]}')


if __name__ == '__main__':
    from importlib.metadata import version

    check_install()
    check_flows()
    print(
        f'SDM_CLIENT_SMOKE_OK sdm-client {version("structured-data-models-client")} on {sys.platform}'
    )
