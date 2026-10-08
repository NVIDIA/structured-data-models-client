# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P7.graph-store (task 5): the Rust ``graph_store`` is ``LocalGraphStore``.

1,000 random graphs of in-memory tables (seed 20261012): keys of ``int64``,
``Int64``, ``float64`` and text, with nulls and repeats; times as
timestamps and as text, with nulls, sub-second parts and times before 1970;
foreign keys that miss. Each gives the same sanitization report, masks, key
maps, times in floored seconds, time ranges and CSC arrays (the stable sort
by destination and time), or the same refusal.

Then each graph whose stores agree is sampled through ``relationallib``
over today's arrays and through the Rust ``NeighborSampler`` over the core's,
from random entities of a table with a key.
"""

from __future__ import annotations

import random
from typing import Any

from _sampling import (
    NUM_STORES,
    STORE_SEED,
    build_graph,
    native,
    ordered,
    python_store,
    random_store,
    relationallib,
    report,
    run_sampler,
    rust_store,
    short,
    store_sampler_args,
)


def _graphs() -> list[Any]:
    rng = random.Random(STORE_SEED)
    out = []
    for _ in range(NUM_STORES):
        graph = build_graph(random_store(rng))
        if graph is not None:
            out.append(graph)
    return out


def _first_difference(want: Any, got: Any, path: str = '$') -> str:
    if isinstance(want, list) and isinstance(got, list):
        if len(want) != len(got):
            return f'{path}: length {len(got)}, today {len(want)}'
        for i, (a, b) in enumerate(zip(want, got, strict=True)):
            if a != b:
                return _first_difference(a, b, f'{path}[{i}]')
    return f'{path}: {short(got, 200)}, today {short(want, 200)}'


def _calls(rng: random.Random, store: dict[str, Any]) -> list[dict[str, Any]]:
    keyed = [name for name, ids in store['node_ids'].items() if ids]
    if not keyed or not store['colptr']:
        return []
    calls = []
    for _ in range(2):
        entity = rng.choice(keyed)
        nodes = [
            rng.choice(store['node_ids'][entity])
            for _ in range(rng.randint(1, 8))
        ]
        hops = rng.randint(1, 3)
        times = store['time'].get(entity)
        calls.append(
            {
                'num_neighbors': {
                    key: [rng.choice((-1, 1, 2, 4)) for _ in range(hops)]
                    for key in store['colptr']
                },
                'offsets': {},
                'entity': entity,
                'seed_node': nodes,
                'seed_time': [times[n] for n in nodes]
                if times and rng.random() < 0.5
                else [],
                'reseed': rng.choice((None, rng.randint(0, 2**31 - 1))),
            }
        )
    return calls


def test_the_graphs_reach_every_path() -> None:
    graphs = _graphs()
    assert len(graphs) >= 900, f'only {len(graphs)} graphs were built'
    stores = [python_store(g) for g in graphs]
    good = [s for s in stores if 'error' not in s]
    assert sum(bool(s['masks']) for s in good) >= 200
    assert sum(len(s['masks']) < len(s['report']) for s in good) >= 200
    assert sum(bool(s['colptr']) for s in good) >= 500
    assert sum(bool(s['time']) for s in good) >= 500
    assert sum('error' in s for s in stores) >= 3


def test_random_graphs_store_as_today() -> None:
    module = native('graph_store', 'NeighborSampler')
    rng = random.Random(STORE_SEED + 1)
    failures = []
    sampled = 0
    graphs = _graphs()
    for index, graph in enumerate(graphs):
        want = python_store(graph)
        got = rust_store(module, graph)
        if ordered(got) != ordered(want):
            failures.append(
                f'graph {index}: {_first_difference(ordered(want), ordered(got))}'
            )
            continue
        if 'error' in want:
            continue
        calls = _calls(rng, want)
        if not calls:
            continue
        python_graph = {
            **store_sampler_args(want, graph),
            'seed': 7,
            'calls': calls,
        }
        rust_graph = {
            **store_sampler_args(got, graph),
            'seed': 7,
            'calls': calls,
        }
        a = run_sampler(relationallib().NeighborSampler, python_graph)
        b = run_sampler(module.NeighborSampler, rust_graph)
        if a != b:
            failures.append(
                f'graph {index}: sampled {short(b, 300)}, today {short(a, 300)}'
            )
        sampled += sum(not (isinstance(x, dict) and 'error' in x) for x in a)
    report(failures, len(graphs), 'graph stores')
    # Samples that raise on both sides agree without comparing anything.
    assert sampled >= 1_000, f'only {sampled} samples were drawn without error'
