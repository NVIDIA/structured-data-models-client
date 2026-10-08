# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The Python bugs brief task 7b fixes in the core, and task 7c's stream,
each reproduced through public calls (``accept_fixes.py``).

    python _fixes.py --out FILE

runs every scenario in this process and writes each outcome: ``ok`` and a
value, or the error's class, whether it is a public ``sdm_client`` error,
its code and its message.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _switch import public_error_types, qualname  # noqa: E402

QUESTION = 'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id'


def _outcome(fn: Any) -> dict[str, Any]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            value = fn()
    except Exception as error:  # noqa: BLE001 - the outcome is the point
        return {
            'class': qualname(type(error)),
            'public': isinstance(error, public_error_types()),
            'code': getattr(error, 'code', None),
            'message': str(error)[:1000],
        }
    return {'ok': True, 'value': value}


def _shop() -> tuple[Any, Any]:
    import pandas as pd

    users = pd.DataFrame({'user_id': range(1, 21), 'age': range(20, 40)})
    orders = pd.DataFrame(
        {
            'order_id': range(1, 201),
            'user_id': [1 + i % 20 for i in range(200)],
            'quantity': [1 + i % 5 for i in range(200)],
            'ordered_at': pd.date_range('2024-01-01', periods=200, freq='D'),
        }
    )
    return users, orders


def _duckdb(frames: dict[str, Any]) -> Any:
    import duckdb

    connection = duckdb.connect()
    connection.execute('SET threads TO 1')
    for name, frame in frames.items():
        connection.register('frame', frame)
        connection.execute(f'CREATE TABLE {name} AS SELECT * FROM frame')
        connection.unregister('frame')
    return connection


def empty_table_from_source() -> Any:
    import duckdb
    from sdm_client import Schema
    from sdm_connectors import DuckDBSource

    connection = duckdb.connect()
    connection.execute(
        'CREATE TABLE users (user_id INTEGER PRIMARY KEY, age INTEGER)'
    )
    connection.execute(
        'CREATE TABLE orders (order_id INTEGER, user_id INTEGER, ordered_at TIMESTAMP)'
    )
    connection.execute('INSERT INTO users VALUES (1, 20), (2, 30)')
    Schema.from_source(DuckDBSource(connection))
    return None


def _odd_column(kind: str) -> Any:
    import pandas as pd
    from sdm_client import PredictiveQuery, Schema

    users, orders = _shop()
    if kind == 'complex':
        users['signal'] = [complex(i, 1) for i in range(20)]
    elif kind == 'period':
        users['joined'] = pd.period_range('2020-01', periods=20, freq='M')
    else:
        users['band'] = pd.interval_range(0, 20)
    schema = Schema.from_data({'users': users, 'orders': orders})
    PredictiveQuery(QUESTION, schema).materialize(indices=[1, 2], random_seed=7)
    return None


def table_emptied_after_read() -> Any:
    from sdm_client import PredictiveQuery, Schema
    from sdm_connectors import DuckDBSource

    users, orders = _shop()
    connection = _duckdb({'users': users, 'orders': orders})
    schema = Schema.from_source(DuckDBSource(connection))
    connection.execute('DELETE FROM orders')
    PredictiveQuery(QUESTION, schema).materialize(indices=[1, 2], random_seed=7)
    return None


def _reserved_name_added(name: str) -> Any:
    from sdm_client import PredictiveQuery, Schema

    users, orders = _shop()
    schema = Schema.from_data({'users': users, 'orders': orders})
    users[name] = 1
    request = PredictiveQuery(QUESTION, schema).materialize(
        indices=[1, 2], random_seed=7
    )
    return len(request.payloads or ())


def _dropped_after_schema(column: str) -> Any:
    import pandas as pd
    from sdm_client import PredictiveQuery, Schema

    people = pd.DataFrame(
        {
            'user_id': range(12),
            'age': [20.0 + i for i in range(12)],
            'segment': ['south', 'north', 'north'] * 4,
            'churn': [0, 1] * 6,
        }
    )
    schema = Schema.from_data({'people': people})
    scored = people.head(3).drop(columns=['churn'])
    people.drop(columns=[column], inplace=True)
    PredictiveQuery(
        'PREDICT people.churn FOR EACH people.user_id', schema
    ).materialize(predict=scored)
    return None


def time_of_day_column() -> Any:
    import duckdb
    from sdm_client import Schema
    from sdm_connectors import DuckDBSource

    connection = duckdb.connect()
    connection.execute(
        'CREATE TABLE shifts AS SELECT range AS shift_id, (range % 5) AS worker, '
        "TIME '08:00:00' + INTERVAL (range) MINUTE AS starts_at FROM range(50)"
    )
    return Schema.from_source(DuckDBSource(connection))['shifts'].time_column


def index_past_uint64() -> Any:
    from sdm_client import PredictiveQuery, Schema

    users, orders = _shop()
    schema = Schema.from_data({'users': users, 'orders': orders})
    PredictiveQuery(QUESTION, schema).materialize(
        indices=[2**64], random_seed=7
    )
    return None


def unseeded_runs() -> Any:
    from sdm_client import PredictiveQuery, Schema

    users, orders = _shop()
    schema = Schema.from_data({'users': users, 'orders': orders})
    query = PredictiveQuery(QUESTION, schema)
    asked = [
        query.materialize(
            indices=[1, 2, 3], num_context_samples=5, random_seed=None
        ),
        query.materialize(
            indices=[1, 2, 3], num_context_samples=5, random_seed=None
        ),
        PredictiveQuery(QUESTION, schema).materialize(
            indices=[1, 2, 3], num_context_samples=5, random_seed=None
        ),
    ]
    return [
        json.dumps(
            [p.get('context') for p in r.payloads or ()],
            default=str,
            sort_keys=True,
        )
        for r in asked
    ]


SCENARIOS = {
    'empty-table-from-source': empty_table_from_source,
    'complex-column': lambda: _odd_column('complex'),
    'period-column': lambda: _odd_column('period'),
    'interval-column': lambda: _odd_column('interval'),
    'table-emptied-after-read': table_emptied_after_read,
    'instance_id-added-after-schema': lambda: _reserved_name_added(
        'instance_id'
    ),
    '__kumo_anchor_time-added-after-schema': lambda: _reserved_name_added(
        '__kumo_anchor_time'
    ),
    'target-dropped-after-schema': lambda: _dropped_after_schema('churn'),
    'entity-dropped-after-schema': lambda: _dropped_after_schema('user_id'),
    'time-of-day-column': time_of_day_column,
    'index-past-uint64': index_past_uint64,
    'unseeded-runs': unseeded_runs,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    out = {name: _outcome(fn) for name, fn in SCENARIOS.items()}
    Path(args.out).write_text(json.dumps(out, indent=1, default=repr))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
