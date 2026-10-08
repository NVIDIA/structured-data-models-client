# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P7.graphs: the Rust ``NeighborSampler`` is a drop-in for
``relationallib.NeighborSampler``.

10,000 random graphs (seed 20261011), of one to four node types and up to
five edge types, static and temporal, with neighborhoods wider than the
fan-out (where the C++ draws random numbers), time ranges with and without a
start, one to three hops, and seed times given or taken from the entity
table. Each is sampled one to three times by one sampler, reseeded now and
then, so the random stream must carry over between calls; about one in a
hundred has a neighborhood out of time order, which the C++ refuses.

Then invalid inputs, compared by error class and text, and seeds outside a
C ``int``, which the drop-in must take or refuse as ``relationallib`` does.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from _sampling import (
    NUM_GRAPHS,
    draws_random_numbers,
    graphs,
    native,
    relationallib,
    report,
    run_sampler,
    short,
)


def _both(graph: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    want = run_sampler(relationallib().NeighborSampler, graph)
    got = run_sampler(native('NeighborSampler').NeighborSampler, graph)
    return want, got


def _difference(want: list[Any], got: list[Any]) -> str | None:
    if len(want) != len(got):
        return f'{len(got)} outcomes, relationallib {len(want)}'
    for i, (a, b) in enumerate(zip(want, got, strict=True)):
        if a != b:
            return (
                f'call {i}: gave {short(b, 300)}, relationallib {short(a, 300)}'
            )
    return None


def test_the_graphs_reach_every_path() -> None:
    drawn = list(graphs())
    assert len(drawn) == NUM_GRAPHS
    assert sum(map(draws_random_numbers, drawn)) >= 2_000
    assert sum(any(c['offsets'] for c in g['calls']) for g in drawn) >= 2_000
    assert (
        sum(
            any(c['seed_time'] for c in g['calls']) and bool(g['node_time'])
            for g in drawn
        )
        >= 2_000
    )
    assert (
        sum(
            any(
                len(v) > 1
                for c in g['calls']
                for v in c['num_neighbors'].values()
            )
            for g in drawn
        )
        >= 2_000
    )


def test_random_graphs_sample_as_relationallib_does() -> None:
    failures = []
    for index, graph in enumerate(graphs()):
        want, got = _both(graph)
        problem = _difference(want, got)
        if problem:
            failures.append(f'graph {index}: {problem}')
    report(failures, NUM_GRAPHS, 'graphs')


def _valid() -> dict[str, Any]:
    return {
        'node_types': ['users', 'orders'],
        'edge_types': [['orders', 'by', 'users'], ['users', 'of', 'orders']],
        'colptr': {
            'orders__by__users': [0, 2, 3],
            'users__of__orders': [0, 1, 2, 3],
        },
        'row': {'orders__by__users': [0, 1, 2], 'users__of__orders': [0, 0, 1]},
        'node_time': {'orders': [1, 2, 3], 'users': [0, 0]},
        'seed': 5,
        'calls': [
            {
                'num_neighbors': {'orders__by__users': [1, 1]},
                'offsets': {'users__of__orders': [[None, 0], [-1, 0]]},
                'entity': 'users',
                'seed_node': [0, 1],
                'seed_time': [3, 3],
                'reseed': None,
            }
        ],
    }


def _broken(change: str) -> dict[str, Any]:
    graph = _valid()
    call = graph['calls'][0]
    if change == 'unknown node type in node_time':
        graph['node_time']['nobody'] = [1]
    elif change == 'unknown edge type in colptr':
        graph['colptr']['x__y__z'] = [0]
    elif change == 'unknown entity table':
        call['entity'] = 'nobody'
    elif change == 'seed time size mismatch':
        call['seed_time'] = [3]
    elif change == 'edge in both dicts':
        call['num_neighbors']['users__of__orders'] = [1, 1]
    elif change == 'edge in neither dict':
        del call['num_neighbors']['orders__by__users']
    elif change == 'offsets without node time':
        call['offsets']['orders__by__users'] = call['offsets'][
            'users__of__orders'
        ]
        del call['num_neighbors']['orders__by__users']
        del graph['node_time']['orders']
        graph['node_time']['users'] = [0, 0]
    elif change == 'offsets not of shape (N, 2)':
        call['offsets']['users__of__orders'] = [[None, 0, 1], [-1, 0, 1]]
    elif change == 'unsorted temporal neighborhood':
        graph['row']['orders__by__users'] = [1, 0, 2]
        call['num_neighbors']['orders__by__users'] = [-1, -1]
    elif change == 'unknown edge type in num_neighbors':
        call['num_neighbors']['x__y__z'] = [1, 1]
    return graph


BROKEN = (
    'unknown node type in node_time',
    'unknown edge type in colptr',
    'unknown entity table',
    'seed time size mismatch',
    'edge in both dicts',
    'edge in neither dict',
    'offsets without node time',
    'offsets not of shape (N, 2)',
    'unsorted temporal neighborhood',
    'unknown edge type in num_neighbors',
)


@pytest.mark.parametrize('change', BROKEN)
def test_invalid_inputs_are_refused_as_relationallib_refuses_them(
    change: str,
) -> None:
    graph = _broken(change)
    want, got = _both(graph)
    assert any(
        'error' in outcome for outcome in want if isinstance(outcome, dict)
    ), f'relationallib takes the input with {change}; the case is not invalid'
    problem = _difference(want, got)
    assert problem is None, problem


def _typed(graph: dict[str, Any], **arrays: Any) -> tuple[Any, Any]:
    r"""Construct both samplers with some arrays replaced, and sample once."""
    out = []
    for cls in (
        relationallib().NeighborSampler,
        native('NeighborSampler').NeighborSampler,
    ):
        args = {
            'node_types': graph['node_types'],
            'edge_types': [tuple(e) for e in graph['edge_types']],
            'colptr': {
                k: np.array(v, dtype=np.int64)
                for k, v in graph['colptr'].items()
            },
            'row': {
                k: np.array(v, dtype=np.int64) for k, v in graph['row'].items()
            },
            'node_time': {
                k: np.array(v, dtype=np.int64)
                for k, v in graph['node_time'].items()
            },
        }
        args.update(arrays)
        try:
            cls(**args, seed=1)
            out.append('ok')
        except Exception as error:  # noqa: BLE001 - compared, not judged
            out.append(f'{type(error).__name__}: {error}')
    return out[0], out[1]


@pytest.mark.parametrize('dtype', ['int32', 'float64', 'uint64'])
def test_arrays_of_another_dtype_are_refused_alike(dtype: str) -> None:
    graph = _valid()
    colptr = {k: np.array(v, dtype=dtype) for k, v in graph['colptr'].items()}
    want, got = _typed(graph, colptr=colptr)
    assert want != 'ok', f'relationallib takes {dtype} arrays'
    assert got == want


def test_two_dimensional_arrays_are_refused_alike() -> None:
    graph = _valid()
    row = {k: np.array([v], dtype=np.int64) for k, v in graph['row'].items()}
    want, got = _typed(graph, row=row)
    assert want != 'ok'
    assert got == want


SEEDS = (
    -(2**31),
    -1,
    0,
    2**31 - 1,
    2**31,
    2**32 - 1,
    2**32,
    2**63 - 1,
    2**63,
    2**64 - 1,
    2**64,
    10**30,
)


@pytest.mark.parametrize('where', ['constructor', 'seed'])
@pytest.mark.parametrize('seed', SEEDS)
def test_seeds_are_taken_or_refused_as_relationallib_does(
    seed: int, where: str
) -> None:
    graph = copy.deepcopy(_valid())
    # A static graph whose neighborhoods are wider than the fan-out, so the
    # seed decides what is drawn.
    graph['node_time'] = {}
    graph['colptr'] = {
        'orders__by__users': [0, 3, 3],
        'users__of__orders': [0, 2, 4, 6],
    }
    graph['row'] = {
        'orders__by__users': [0, 1, 2],
        'users__of__orders': [0, 1, 0, 1, 1, 0],
    }
    graph['calls'][0]['num_neighbors'] = {
        'orders__by__users': [1, 1],
        'users__of__orders': [1, 1],
    }
    graph['calls'][0]['offsets'] = {}
    graph['calls'][0]['seed_time'] = []
    if where == 'constructor':
        graph['seed'] = seed
    else:
        graph['calls'][0]['reseed'] = seed
    want, got = _both(graph)
    problem = _difference(_type_errors_alike(want), _type_errors_alike(got))
    assert problem is None, problem


def _type_errors_alike(outcomes: list[Any]) -> list[Any]:
    r"""Outcomes with a ``TypeError``'s text dropped: pybind11 words it."""
    return [
        {'error': {'code': 'TypeError'}}
        if isinstance(o, dict) and o['error']['code'] == 'TypeError'
        else o
        for o in outcomes
    ]
