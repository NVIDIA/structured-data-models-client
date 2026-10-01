# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
import textwrap

import pytest

_SCRIPT = textwrap.dedent(
    """
    import hashlib
    import json

    import numpy as np
    import pandas as pd

    from sdm_relational_engine.rfm import Graph, KumoRelational

    rng = np.random.default_rng(0)
    users = pd.DataFrame({'user_id': np.arange(12), 'age': rng.integers(18, 70, 12)})
    orders = pd.DataFrame({
        'order_id': np.arange(60),
        'user_id': rng.integers(0, 12, 60),
        'date': pd.to_datetime('2025-01-01') + pd.to_timedelta(
            rng.integers(0, 60, 60), unit='D'),
        'amount': rng.uniform(5, 100, 60).round(2),
    })
    graph = Graph.from_data({'users': users, 'orders': orders}, verbose=False)
    model = KumoRelational(graph, verbose=False)
    query = 'PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.user_id'
    task = model._get_task_table(model._parse_query(query), indices=[1, 2],
                                 random_seed=42)
    requests = model.materialize_task(task, random_seed=42, verbose=False)
    blob = json.dumps([r.payload for r in requests], sort_keys=True, default=str)
    print(hashlib.sha256(blob.encode()).hexdigest())
    """
)


def _payload_hash() -> str:
    result = subprocess.run(
        [sys.executable, '-c', _SCRIPT],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().splitlines()[-1]


def test_payload_is_identical_across_processes() -> None:
    r"""A fixed ``random_seed`` must survive a fresh interpreter.

    Python randomizes string hashing per process, so iterating a ``set`` of
    column names leaks that ordering into the request. This has to run in
    subprocesses: within one process the ordering is stable and the regression
    is invisible.
    """
    assert _payload_hash() == _payload_hash()


_SQL_SCRIPT = textwrap.dedent(
    """
    import json
    import sqlite3
    import sys
    import warnings

    warnings.filterwarnings('ignore')

    import numpy as np
    import pandas as pd

    from sdm_relational_engine.rfm import Graph, KumoRelational

    backend, path = sys.argv[1], sys.argv[2]

    rng = np.random.default_rng(0)
    users = pd.DataFrame({
        'user_id': np.arange(12, dtype='int64'),
        'age': rng.integers(18, 70, 12).astype('int64'),
        'country': rng.choice(['US', 'DE', 'JP'], 12),
    })
    orders = pd.DataFrame({
        'order_id': np.arange(60, dtype='int64'),
        'user_id': rng.integers(0, 12, 60).astype('int64'),
        'date': pd.to_datetime('2025-01-01') + pd.to_timedelta(
            rng.integers(0, 60, 60).astype('int64'), unit='D'),
        'amount': rng.uniform(5, 100, 60).round(2),
    })

    if backend == 'duckdb':
        import duckdb
        con = duckdb.connect(path)
        con.execute('CREATE TABLE users (user_id BIGINT PRIMARY KEY, '
                    'age BIGINT, country VARCHAR)')
        con.execute('CREATE TABLE orders (order_id BIGINT PRIMARY KEY, '
                    'user_id BIGINT, date TIMESTAMP, amount DOUBLE)')
        con.register('_u', users)
        con.execute('INSERT INTO users SELECT * FROM _u')
        con.register('_o', orders)
        con.execute('INSERT INTO orders SELECT * FROM _o')
        con.execute('CREATE INDEX orders_user_id ON orders(user_id)')
        con.close()
    else:
        con = sqlite3.connect(path)
        con.execute('CREATE TABLE users (user_id BIGINT PRIMARY KEY, '
                    'age BIGINT, country TEXT)')
        con.execute('CREATE TABLE orders (order_id BIGINT PRIMARY KEY, '
                    'user_id BIGINT, date TEXT, amount REAL)')
        con.executemany('INSERT INTO users VALUES (?, ?, ?)',
                        [tuple(r) for r in users.itertuples(index=False)])
        rows = orders.copy()
        rows['date'] = rows['date'].astype(str)
        con.executemany('INSERT INTO orders VALUES (?, ?, ?, ?)',
                        [tuple(r) for r in rows.itertuples(index=False)])
        con.execute('CREATE INDEX orders_user_id ON orders(user_id)')
        con.commit()
        con.close()

    graph = getattr(Graph, 'from_' + backend)(path, verbose=False)
    edges = {(e.src_table, e.fkey, e.dst_table) for e in graph.edges}
    if ('orders', 'user_id', 'users') not in edges:
        graph.link('orders', 'user_id', 'users')
    graph.validate()

    model = KumoRelational(graph, verbose=False)
    query = 'PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.user_id'
    task = model._get_task_table(model._parse_query(query), indices=[1, 2],
                                 random_seed=42)
    payload = model.materialize_task(task, random_seed=42,
                                     verbose=False)[0].payload
    context = payload['context']
    order = {'instance_table': context['instance_table']['columns']}
    for name, table in sorted(context['related_tables'].items()):
        order[name] = table['columns']
    schema = payload['schema']
    order['schema'] = {
        name: list(table['columns'])
        for name, table in sorted(schema['related_tables'].items())
    }
    print(json.dumps(order))
    """
)


def _column_order(backend: str, hash_seed: str, tmp_path) -> dict:
    environment = dict(os.environ, PYTHONHASHSEED=hash_seed)
    database = tmp_path / f'{backend}-{hash_seed}.db'
    result = subprocess.run(
        [sys.executable, '-c', _SQL_SCRIPT, backend, str(database)],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize('backend', ['sqlite', 'duckdb'])
def test_sql_payload_column_order_is_identical_across_processes(
    backend,
    tmp_path,
) -> None:
    """The SQL samplers built their ``SELECT`` projection by iterating a
    ``set[str]``, so the column order of every table in the request, and the
    prediction derived from it, changed with the per-process string hash
    seed. The existing local-backend test could not catch this: the leak is
    in the SQL projection, and only a fresh interpreter re-randomises the
    seed, so the hash seed is pinned explicitly here rather than trusting the
    subprocesses to differ.
    """
    pytest.importorskip(f'adbc_driver_{backend}')

    orders = [
        _column_order(backend, seed, tmp_path) for seed in ('0', '1', '2')
    ]
    assert orders[0] == orders[1] == orders[2]
