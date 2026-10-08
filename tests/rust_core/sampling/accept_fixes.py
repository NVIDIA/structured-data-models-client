# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P7.fixes: task 1's three fixes to today's Python, each as a user meets
the bug. Every test here fails at BASE.

- Seeds of ``2**31`` and above fail on the in-memory backend, because the
  C++ sampler's ``seed`` takes a C ``int``; the public API accepts any seed
  from 0 up, and so must the in-memory backend, deterministically, with the
  seed deciding the neighbors drawn (reseeding, not ignoring it).
- With ``anchor_time='entity'``, the context examples are sorted by time
  with an unstable sort, so the order of entities sharing a time is not
  specified; it must be the order of the shuffle, a stable sort.
- The ``time_ranges`` statement lists its tables in set order, which follows
  ``PYTHONHASHSEED`` once two or more tables have a time column; it must
  follow the order the tables were given.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from _sampling import HERE, REPO

LARGE_SEEDS = (2**31, 2**32 - 1, 2**32, 2**63, 2**64 + 3, 10**30)


# --- Large seeds in memory -----------------------------------------------------


def _shop_schema() -> Any:
    from _parity.data import shop
    from sdm_client import Schema

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data(shop(0))


def _payloads(schema: Any, query: str, seed: int) -> list[Any]:
    from sdm_client import PredictiveQuery

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = PredictiveQuery(query, schema).materialize(
            indices=[1, 2], random_seed=seed, num_context_samples=30
        )
    return json.loads(json.dumps(request._payloads, default=str))


QUESTIONS = (
    'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
    'PREDICT products.category FOR EACH products.product_id',
)


@pytest.mark.parametrize('query', QUESTIONS)
@pytest.mark.parametrize('seed', LARGE_SEEDS)
def test_a_large_seed_works_in_memory(query: str, seed: int) -> None:
    schema = _shop_schema()
    first = _payloads(schema, query, seed)
    assert first, 'materialize gave no payload'
    assert _payloads(schema, query, seed) == first, (
        f'random_seed={seed} gives different payloads on a second call'
    )


def _static_sampler() -> Any:
    r"""An in-memory sampler over a static graph whose users have about 20
    orders each, so a hop of 3 draws from the C++ generator."""
    from sdm_relational_engine.rfm import Edge, Graph, LocalTable
    from sdm_relational_engine.rfm.backend.local import LocalSampler

    rng = np.random.RandomState(0)
    users = pd.DataFrame(
        {
            'user_id': np.arange(60),
            'score': rng.randint(0, 5, 60).astype(float),
        }
    )
    orders = pd.DataFrame(
        {
            'order_id': np.arange(1200),
            'user_id': rng.randint(0, 60, 1200),
            'amount': rng.rand(1200),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        graph = Graph(
            [
                LocalTable(users, 'users', primary_key='user_id'),
                LocalTable(orders, 'orders', primary_key='order_id'),
            ],
            [Edge('orders', 'user_id', 'users')],
        )
    return LocalSampler(graph, verbose=False)


def _drawn(sampler: Any, seed: int) -> str:
    r"""The rows one seeded subgraph sample holds."""
    out = sampler._sample_subgraph(
        'users',
        pd.Series([1, 2, 3]),
        pd.Series(pd.to_datetime(['2024-01-01'] * 3)),
        {'users': {'score'}, 'orders': {'amount'}},
        [3, 3],
        random_seed=seed,
    )
    return json.dumps(
        {name: df.to_dict('list') for name, df in out.df_dict.items()}
    )


def test_a_large_seed_restarts_the_neighbor_draws() -> None:
    sampler = _static_sampler()
    assert _drawn(sampler, 7) != _drawn(sampler, 8), (
        'the graph does not make the C++ sampler draw'
    )
    first = {seed: _drawn(sampler, seed) for seed in LARGE_SEEDS}
    # Each seed gives the same rows after others have drawn: it reseeds.
    for seed in reversed(LARGE_SEEDS):
        assert _drawn(sampler, seed) == first[seed], (
            f'random_seed={seed} draws other neighbors after other seeds drew'
        )
    # A seed is not ignored: the next seed up draws other rows.
    differ = [s for s in LARGE_SEEDS if _drawn(sampler, s + 1) != first[s]]
    assert len(differ) >= len(LARGE_SEEDS) - 1, (
        f'only seeds {differ} draw other neighbors than the seed after them'
    )


# --- A stable sort for anchor_time='entity' ----------------------------------


def _entity_model() -> tuple[Any, Any, pd.DataFrame]:
    from predictive_query.ast import Column, Condition, Constant
    from predictive_query.query import ValidatedPredictiveQuery
    from predictive_query.schema.typing import Dtype, Stype
    from sdm_relational_engine.rfm import Graph, KumoRelational, LocalTable

    rng = np.random.RandomState(0)
    n = 500
    users = pd.DataFrame(
        {
            'USER_ID': np.arange(n),
            # Four days for 500 users: most of them share a time.
            'SIGNUP': pd.Timestamp('2024-01-01')
            + pd.to_timedelta(rng.randint(0, 4, n), unit='D'),
            'SCORE': rng.randint(0, 100, n).astype('float64'),
        }
    )
    graph = Graph(
        tables=[
            LocalTable(
                users, 'USERS', primary_key='USER_ID', time_column='SIGNUP'
            )
        ],
        edges=[],
    )
    query = ValidatedPredictiveQuery(
        target_ast=Column(fqn='USERS.SCORE', stype_maybe=Stype.numerical),
        entity_ast=Column(fqn='USERS.USER_ID'),
        rfm_entity_ids=Condition(
            target=Column(fqn='USERS.USER_ID'),
            op='=',
            value=Constant(value='0', dtype_maybe=Dtype.int),
        ),
    )
    return KumoRelational(graph, verbose=False), query, users


@pytest.mark.parametrize('seed', [0, 7, 11, 2024, 99_991])
def test_entity_anchor_times_sort_stably(seed: int) -> None:
    from sdm_relational_engine.rfm.base.utils import random_state

    model, query, users = _entity_model()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        table = model._get_task_table(
            query, indices=[0], anchor_time='entity', random_seed=seed
        )
    got = table._context_df['ENTITY'].tolist()

    # The documented stream of sample_target: its first draw shuffles the
    # candidates, here every user, in table order.
    order = np.arange(len(users))
    random_state(seed).shuffle(order)
    times = users['SIGNUP'].astype('int64').to_numpy()
    # Latest first, and a tie in the order the shuffle left it.
    stable = sorted(order.tolist(), key=lambda row: -times[row])
    want = users['USER_ID'].to_numpy()[stable].tolist()

    assert len(got) == len(users), f'{len(got)} context examples'
    assert got == want, (
        'the context is not the shuffled users sorted latest first with ties '
        f'kept in shuffled order; first difference at position '
        f'{next(i for i, (a, b) in enumerate(zip(got, want, strict=True)) if a != b)}'
    )


# --- time_ranges in the given order ------------------------------------------

HASH_SEEDS = ('0', '1', '2', '3', '4242', '987654')


def _run_questions(hashseed: str) -> dict[str, Any]:
    env = dict(os.environ)
    env['PYTHONHASHSEED'] = hashseed
    proc = subprocess.run(
        [sys.executable, str(HERE / '_fixes_run.py')],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, (
        f'the questions failed under PYTHONHASHSEED={hashseed}:\n'
        f'{proc.stderr[-6000:]}'
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _time_ranges(statements: list[str]) -> list[list[str]]:
    r"""The tables each ``time_ranges`` statement reads, in order."""
    out = []
    for sql in statements:
        if 'UNION ALL' in sql and 'min_date' in sql and 'max_date' in sql:
            out.append(re.findall(r'FROM\s+"?([A-Za-z_]+)"?', sql))
    return out


def test_time_ranges_follow_the_given_table_order() -> None:
    runs = {seed: _run_questions(seed) for seed in HASH_SEEDS}
    first = runs[HASH_SEEDS[0]]
    differ = [
        seed
        for seed, run in runs.items()
        if run['questions'] != first['questions']
    ]
    assert not differ, (
        f'the SQL differs under PYTHONHASHSEED {differ} from {HASH_SEEDS[0]}'
    )
    given = first['tables']
    found = [
        tables
        for question in first['questions']
        for tables in _time_ranges(question)
    ]
    assert found, 'no time_ranges statement over several tables ran'
    for tables in found:
        assert len(tables) >= 2, f'a time_ranges statement over {tables}'
        want = [name for name in given if name in tables]
        assert tables == want, (
            f'time_ranges reads {tables}; the tables were given as {given}'
        )
