# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Hostile input to every entry point Phase 7 added to the binding.

Each call gets values that are huge, negative, zero, empty, of the wrong
type, or sizes no array can have, and must either answer or raise the
exception ``tests/rust_core/sampling/CONTRACT.md`` documents for it:
``ValueError`` for input it can't read, ``TypeError`` for an argument of the
wrong type, as Python's own functions raise, ``MemoryError`` for an array no
allocation can hold, as numpy raises, and for the neighbor sampler and the
graph store the class today's Python raises.

None may panic. Uncaught, pyo3 raises a panic as ``PanicException``, which
derives from ``BaseException``; caught, the binding raises it as a
``ValueError`` that says the core panicked, and that fails here too. An
allocation that fails must be an error rather than an abort, so the calls run
in a child process, and a crash fails the test with the call that made it.

The sampler and labels fixture operations get the same treatment over every
case of their governed fixtures, each mutated at random.
"""

from __future__ import annotations

import copy
import json
import random
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[4]
FIXTURES = REPO / 'tests' / 'parity' / 'fixtures'
FIXTURE_FILES = (
    'local-sampler.json',
    'local-graph-store.json',
    'sampler-utils.json',
    'pandas-executor.json',
)
MUTATION_SEED = 20261003

OK = 'ok'
VALUE = 'ValueError'
TYPE = 'TypeError'
MEMORY = 'MemoryError'
RUNTIME = 'RuntimeError'
KEY = 'KeyError'
OVERFLOW = 'OverflowError'

# Sizes no array can have: 2**59 int64 values (4 EiB) are past any address
# space, so allocating them fails; from 2**61 they are past isize::MAX bytes.
# Sizes between that and what fits in memory would be allocated and filled,
# so none is used.
NO_ROOM = 2**59
TOO_BIG = (2**61, 2**63, 2**64, 2**200)
NEGATIVE = (-1, -(2**31) - 1, -(2**63) - 1, -(2**200))
NOT_INTS: tuple[Any, ...] = (1.5, 'x', None, b'1', {})

Case = tuple[str, Callable[[], Any], frozenset[str]]


def expect(*outcomes: str) -> frozenset[str]:
    return frozenset(outcomes)


def random_cases(n: Any) -> Iterator[Case]:
    yield from (
        (f'RandomState({s!r})', lambda s=s: n.RandomState(s), expect(OK))
        for s in (0, 2**32 - 1, True, [1, 2], (1, 2), [7])
    )
    for seed in (2**32, *TOO_BIG, *NEGATIVE, [], [2**32], [-1]):
        yield (
            f'RandomState({seed!r})',
            lambda s=seed: n.RandomState(s),
            expect(VALUE),
        )
    for seed in (*NOT_INTS, [1.5], ['x'], iter([1])):
        yield (
            f'RandomState({seed!r})',
            lambda s=seed: n.RandomState(s),
            expect(TYPE),
        )
    for seed in (0, 2**64, 10**100, 2**200):
        yield (
            f'random_state({seed!r})',
            lambda s=seed: n.random_state(s),
            expect(OK),
        )
    for seed in NEGATIVE:
        yield (
            f'random_state({seed!r})',
            lambda s=seed: n.random_state(s),
            expect(VALUE),
        )
    for seed in NOT_INTS:
        yield (
            f'random_state({seed!r})',
            lambda s=seed: n.random_state(s),
            expect(TYPE),
        )

    def state() -> Any:
        return n.RandomState(1)

    yield 'next_u32()', lambda: state().next_u32(), expect(OK)
    yield 'next_u64()', lambda: state().next_u64(), expect(OK)
    for value, outcome in [
        (0, OK),
        (2**61, OK),
        (2**63 - 2, OK),
        (2**63 - 1, VALUE),
        *((v, VALUE) for v in (*TOO_BIG[1:], *NEGATIVE)),
        *((v, TYPE) for v in NOT_INTS),
    ]:
        yield (
            f'interval({value!r})',
            lambda v=value: state().interval(v),
            expect(outcome),
        )
    for args, outcome in [
        ((0, 10, 0), OK),
        ((-(2**63), 2**63 - 1, 3), OK),
        ((0, 10, -1), VALUE),
        ((0, 10, NO_ROOM), MEMORY),
        *(((0, 10, v), VALUE) for v in (*TOO_BIG, *NEGATIVE)),
        ((10, 0, 1), VALUE),
        ((5, 5, 1), VALUE),
        ((2**63, 2**64, 1), VALUE),
        ((-(2**63) - 1, 0, 1), VALUE),
        ((0, 2**63, 1), VALUE),
        *(((v, 10, 1), TYPE) for v in NOT_INTS),
        *(((0, v, 1), TYPE) for v in NOT_INTS),
        *(((0, 10, v), TYPE) for v in NOT_INTS),
    ]:
        yield (
            f'randint{args!r}',
            lambda a=args: state().randint(*a),
            expect(outcome),
        )
    for values, outcome in [
        ([], OK),
        ((3, 1, 2), OK),
        ([2**63 - 1, -(2**63)], OK),
        ([2**63], VALUE),
        ([-(2**63) - 1], VALUE),
        ([2**200], VALUE),
        ('abc', TYPE),
        (None, TYPE),
        (5, TYPE),
        ([1.5], TYPE),
        (['x'], TYPE),
        ([None], TYPE),
    ]:
        yield (
            f'shuffle({values!r})',
            lambda v=values: state().shuffle(v),
            expect(outcome),
        )
    for value, outcome in [
        (0, OK),
        (5, OK),
        # numpy's permutation(n) of a negative n is arange(n), empty, as
        # long as numpy can read n.
        *((v, OK) for v in NEGATIVE[:-1]),
        (NEGATIVE[-1], VALUE),
        (NO_ROOM, MEMORY),
        *((v, VALUE) for v in TOO_BIG),
        *((v, TYPE) for v in NOT_INTS),
    ]:
        yield (
            f'permutation({value!r})',
            lambda v=value: state().permutation(v),
            expect(outcome),
        )
    for args, outcome in [
        ((0, 0), OK),
        ((5, 0), OK),
        ((5, 5), OK),
        ((0, 1), VALUE),
        ((-1, 1), VALUE),
        ((5, 6), VALUE),
        ((5, -1), VALUE),
        ((NO_ROOM, 1), MEMORY),
        *(((v, 1), VALUE) for v in TOO_BIG),
        *(((v, 1), VALUE) for v in NEGATIVE),
        *(((5, v), VALUE) for v in (*TOO_BIG, *NEGATIVE)),
        *(((v, 1), TYPE) for v in NOT_INTS),
        *(((5, v), TYPE) for v in NOT_INTS),
    ]:
        yield (
            f'choice{args!r}',
            lambda a=args: state().choice(*a),
            expect(outcome),
        )
    for seed, outcome in [
        (0, OK),
        (2**64 - 1, OK),
        *((v, VALUE) for v in (*TOO_BIG[2:], *NEGATIVE)),
        *((v, TYPE) for v in NOT_INTS),
    ]:
        yield (
            f'Mt19937({seed!r})',
            lambda s=seed: n.Mt19937(s),
            expect(outcome),
        )
    yield 'Mt19937.next_u32()', lambda: n.Mt19937(1).next_u32(), expect(OK)
    for args, outcome in [
        ((5, 4), OK),
        ((5, 5), VALUE),
        ((5, 0), VALUE),
        ((0, 0), VALUE),
        ((-1, -2), VALUE),
        ((2**63 - 1, 1), OK),
        ((2**60, NO_ROOM), MEMORY),
        ((2**62, 2**61), VALUE),
        ((2**63, 1), VALUE),
        ((5, -(2**63) - 1), VALUE),
        *(((v, 1), TYPE) for v in NOT_INTS),
        *(((5, v), TYPE) for v in NOT_INTS),
    ]:
        yield (
            f'floyd{args!r}',
            lambda a=args: n.Mt19937(1).floyd(*a),
            expect(outcome),
        )


DAY = '{"n": 1, "kwds": {"days": 1}}'
UNREADABLE_JSON = ('', 'x', '{', '[1e400]', '{"n": 1}}')


def offset_cases(n: Any) -> Iterator[Case]:
    for sign, outcome in [
        (1, OK),
        (-1, OK),
        (True, OK),
        (0, VALUE),
        (2, VALUE),
        *((v, VALUE) for v in (*TOO_BIG, *NEGATIVE[1:])),
        *((v, TYPE) for v in NOT_INTS),
    ]:
        yield (
            f'date_offset_apply sign={sign!r}',
            lambda s=sign: n.date_offset_apply('[0]', DAY, s),
            expect(outcome),
        )
    extremes = [
        2**63 - 1,
        -(2**63),
        -(2**63) + 1,
        0,
    ]
    for stamps, outcome in [
        ('[]', OK),
        (json.dumps(extremes), OK),
        *((s, VALUE) for s in UNREADABLE_JSON),
        ('{}', VALUE),
        ('[1.5]', VALUE),
        ('["x"]', VALUE),
        ('[null]', VALUE),
        (f'[{2**64}]', VALUE),
        (f'[{-(2**63) - 1}]', VALUE),
        (None, TYPE),
        (0, TYPE),
    ]:
        yield (
            f'date_offset_apply timestamps={stamps!r}',
            lambda s=stamps: n.date_offset_apply(s, DAY, 1),
            expect(outcome),
        )
    readable_offsets = [
        {'n': n_, 'kwds': {unit: value}}
        for n_ in (0, 1, -1, 2**31, 2**62, 2**63 - 1, -(2**63))
        for unit in (
            'years',
            'months',
            'weeks',
            'days',
            'hours',
            'minutes',
            'seconds',
        )
        for value in (0, 1, -1, 2**31, 2**62, 2**63 - 1, -(2**63))
    ]
    readable_offsets.append(
        {
            'n': -7,
            'kwds': {'years': -(2**63), 'months': 2**63 - 1, 'seconds': 1},
        }
    )
    unreadable_offsets = [
        *UNREADABLE_JSON,
        '[]',
        '{}',
        '{"n": "x", "kwds": {}}',
        '{"n": 1, "kwds": {"fortnights": 1}}',
        '{"n": 1, "kwds": {"days": 1.5}}',
        '{"n": 1, "kwds": []}',
        # pandas reads DateOffset(n=3) as three days; the contract's offsets
        # name a unit, and the core refuses one that doesn't.
        '{"n": 3, "kwds": {}}',
        json.dumps({'n': 2**64, 'kwds': {'days': 1}}),
        json.dumps({'n': 1, 'kwds': {'days': -(2**63) - 1}}),
    ]
    stamps = json.dumps([0, 2**63 - 1, -(2**63) + 1])
    # pandas can't make every one of these: a timedelta of hours, minutes
    # and seconds past its range is an OverflowError, as is a product with n
    # past it. Their values are checked against pandas elsewhere; here, each
    # answers or raises that.
    made = expect(OK, OVERFLOW)
    for offset in readable_offsets:
        text = json.dumps(offset)
        yield (
            f'date_offset_apply offset={text}',
            lambda o=text: (
                n.date_offset_apply(stamps, o, 1),
                n.date_offset_apply(stamps, o, -1),
            ),
            made,
        )
        yield (
            f'date_offset_to_ns {text}',
            lambda o=text: n.date_offset_to_ns(o),
            expect(OK),
        )
        for bound in ('earliest', 'latest'):
            yield (
                f'date_offset_to_seconds {text} {bound}',
                lambda o=text, b=bound: n.date_offset_to_seconds(o, b),
                made,
            )
        yield (
            f'date_offset_pool [{text}]',
            lambda o=text: n.date_offset_pool(f'[{o}, {DAY}]', True),
            made,
        )
    for text in unreadable_offsets:
        yield (
            f'date_offset_apply offset={text!r}',
            lambda o=text: n.date_offset_apply('[0]', o, 1),
            expect(VALUE),
        )
        yield (
            f'date_offset_to_ns {text!r}',
            lambda o=text: n.date_offset_to_ns(o),
            expect(VALUE),
        )
        yield (
            f'date_offset_to_seconds {text!r}',
            lambda o=text: n.date_offset_to_seconds(o, 'earliest'),
            expect(VALUE),
        )
    for bound in ('', 'Earliest', 'x'):
        yield (
            f'date_offset_to_seconds bound={bound!r}',
            lambda b=bound: n.date_offset_to_seconds(DAY, b),
            expect(VALUE),
        )
    for args, outcome in [
        (('[]', True), VALUE),
        (('{}', True), VALUE),
        (('[1]', True), VALUE),
        *(((s, True), VALUE) for s in UNREADABLE_JSON),
        ((f'[{DAY}]', 'x'), TYPE),
        ((None, True), TYPE),
    ]:
        yield (
            f'date_offset_pool{args!r}',
            lambda a=args: n.date_offset_pool(*a),
            expect(outcome),
        )


def text_cases(n: Any) -> Iterator[Case]:
    for max_words in (None, 0, 3, *NEGATIVE, *TOO_BIG, NO_ROOM, True):
        # pyarrow takes the cut to 6 * max_words characters as a C long.
        cut = 6 * int(max_words or 0)
        yield (
            f'normalize_text max_words={max_words!r}',
            lambda m=max_words: n.normalize_text(
                ['A b, c <br/> d', None, '', ' ' * 5], m
            ),
            expect(OK if -(2**63) <= cut < 2**63 else OVERFLOW),
        )
        yield (
            f'normalize_text([], max_words={max_words!r})',
            lambda m=max_words: n.normalize_text([], m),
            expect(OK),
        )
    for values, max_words, outcome in [
        ([], 3, OK),
        ('abc', 3, TYPE),
        (None, 3, TYPE),
        ([1], 3, TYPE),
        ([b'x'], 3, TYPE),
        (['a'], 1.5, TYPE),
        (['a'], 'x', TYPE),
    ]:
        yield (
            f'normalize_text({values!r}, {max_words!r})',
            lambda v=values, m=max_words: n.normalize_text(v, m),
            expect(outcome),
        )


def sampler_cases(n: Any) -> Iterator[Case]:
    import numpy as np

    def arr(values: Any, dtype: Any = np.int64) -> Any:
        return np.array(values, dtype=dtype)

    def build(**changes: Any) -> Any:
        args = {
            'node_types': ['A', 'B'],
            'edge_types': [('A', 'to', 'B'), ('B', 'to', 'A')],
            'colptr': {
                'A__to__B': arr([0, 2, 4]),
                'B__to__A': arr([0, 1, 2, 3]),
            },
            'row': {'A__to__B': arr([0, 2, 0, 2]), 'B__to__A': arr([0, 1, 1])},
            'node_time': {'A': arr([1, 0, 3]), 'B': arr([2, 1])},
        }
        args.update(changes)
        return n.NeighborSampler(**args)

    def sample(sampler: Any = None, **changes: Any) -> Any:
        args = {
            'num_neighbors': {
                'A__to__B': arr([1, -1]),
                'B__to__A': arr([1, 1]),
            },
            'sampling_time_offsets': {},
            'entity_table': 'A',
            'seed_node': arr([0, 1, 2]),
            'seed_time': arr([3, 1, 3]),
        }
        args.update(changes)
        return (sampler or build()).sample(**args)

    yield 'NeighborSampler()', build, expect(OK)
    yield 'NeighborSampler.sample()', sample, expect(OK)
    constructors: list[tuple[str, dict[str, Any], str]] = [
        ('node_types str', {'node_types': 'AB'}, TYPE),
        ('node_types None', {'node_types': None}, TYPE),
        ('node_types ints', {'node_types': [1, 2]}, TYPE),
        ('edge_types pairs', {'edge_types': [('A', 'B')]}, TYPE),
        ('edge_types int', {'edge_types': 5}, TYPE),
        ('edge_types of ints', {'edge_types': [(1, 2, 3)]}, TYPE),
        ('colptr None', {'colptr': None}, TYPE),
        ('colptr list', {'colptr': [1]}, TYPE),
        ('colptr unknown type', {'colptr': {'X': arr([0])}}, RUNTIME),
        ('colptr floats', {'colptr': {'A__to__B': arr([0.5], float)}}, RUNTIME),
        ('colptr 2D', {'colptr': {'A__to__B': arr([[0, 1]])}}, RUNTIME),
        ('colptr text', {'colptr': {'A__to__B': 'x'}}, RUNTIME),
        ('colptr empty', {'colptr': {}}, KEY),
        ('row None', {'row': None}, TYPE),
        ('node_time int', {'node_time': 5}, TYPE),
        ('node_time unknown type', {'node_time': {'X': arr([1])}}, RUNTIME),
        ('node_time int32', {'node_time': {'A': arr([1], np.int32)}}, RUNTIME),
        ('seed 2**200', {'seed': 2**200}, OK),
        ('seed -2**31', {'seed': -(2**31)}, OK),
        ('seed -2**31-1', {'seed': -(2**31) - 1}, TYPE),
        ('seed -2**200', {'seed': -(2**200)}, TYPE),
        ('seed float', {'seed': 1.5}, TYPE),
        ('seed text', {'seed': 'x'}, TYPE),
        ('seed None', {'seed': None}, TYPE),
    ]
    for name, changes, outcome in constructors:
        yield (
            f'NeighborSampler({name})',
            lambda c=changes: build(**c),
            expect(outcome),
        )
    for seed, outcome in [
        (0, OK),
        (2**200, OK),
        (-(2**31) - 1, TYPE),
        (1.5, TYPE),
        (None, TYPE),
    ]:
        yield (
            f'NeighborSampler.seed({seed!r})',
            lambda s=seed: build().seed(s),
            expect(outcome),
        )
    # Out-of-range neighborhoods: relationallib reads past its arrays; the
    # port refuses them with the RuntimeError of its other bounds checks.
    wild = [
        ('colptr past row', {'A__to__B': arr([0, 2**62, 2**62 + 1])}),
        ('colptr negative', {'A__to__B': arr([-5, 2, 4])}),
        ('colptr extremes', {'A__to__B': arr([-(2**63), 2**63 - 1, 0])}),
        ('colptr descending', {'A__to__B': arr([4, 0, 2])}),
        ('colptr short', {'A__to__B': arr([0])}),
    ]
    TIMES = {'A': arr([1, 0, 3]), 'B': arr([2, 1])}  # noqa: N806
    for name, colptr in wild:
        full = {'B__to__A': arr([0, 1, 2, 3]), **colptr}
        for static in (True, False):
            yield (
                f'sample({name}, static={static})',
                lambda c=full, s=static: sample(
                    build(colptr=c, node_time={} if s else TIMES)
                ),
                expect(OK, RUNTIME),
            )

    def timed(offsets: Any, hops: int = 1) -> dict[str, Any]:
        return {
            'num_neighbors': {'B__to__A': arr([1] * hops)},
            'sampling_time_offsets': {'A__to__B': offsets},
        }

    samples: list[tuple[str, dict[str, Any], frozenset[str]]] = [
        ('entity unknown', {'entity_table': 'Z'}, expect(KEY, RUNTIME)),
        ('entity int', {'entity_table': 1}, expect(TYPE)),
        ('seed_node past', {'seed_node': arr([5, 2**62, -1])}, expect(RUNTIME)),
        (
            'seed_node extremes',
            {'seed_node': arr([2**63 - 1, -(2**63), 0])},
            expect(RUNTIME),
        ),
        ('seed_node text', {'seed_node': 'x'}, expect(TYPE)),
        ('seed_node list', {'seed_node': [0, 1, 2]}, expect(TYPE)),
        ('seed_time None', {'seed_time': None}, expect(TYPE)),
        ('seed_node floats', {'seed_node': arr([0.5], float)}, expect(RUNTIME)),
        (
            'seed_node empty',
            {'seed_node': arr([]), 'seed_time': arr([])},
            expect(OK, RUNTIME),
        ),
        ('seed_time short', {'seed_time': arr([3])}, expect(OK, RUNTIME)),
        (
            'seed_time extremes',
            {'seed_time': arr([2**63 - 1, -(2**63), 0])},
            expect(OK),
        ),
        (
            'num_neighbors huge',
            {
                'num_neighbors': {
                    'A__to__B': arr([2**63 - 1, 2**62]),
                    'B__to__A': arr([NO_ROOM, 1]),
                }
            },
            expect(OK),
        ),
        (
            'num_neighbors negative',
            {
                'num_neighbors': {
                    'A__to__B': arr([-(2**63), -5]),
                    'B__to__A': arr([-1, -1]),
                }
            },
            expect(OK),
        ),
        ('num_neighbors empty', {'num_neighbors': {}}, expect(KEY)),
        (
            'num_neighbors unknown',
            {'num_neighbors': {'X': arr([1])}},
            expect(RUNTIME),
        ),
        (
            'num_neighbors floats',
            {'num_neighbors': {'A__to__B': arr([1.5], float)}},
            expect(RUNTIME),
        ),
        ('num_neighbors None', {'num_neighbors': None}, expect(TYPE)),
        (
            'time offsets extremes',
            timed(
                np.array(
                    [[None, 2**63 - 1], [-(2**63), 2**63 - 1]], dtype=object
                )
            ),
            expect(OK, RUNTIME),
        ),
        (
            'time offsets floats',
            timed(arr([[0.5, 1.5]], float)),
            expect(RUNTIME),
        ),
        ('time offsets wide', timed(arr([[0, 1, 2]])), expect(OK, RUNTIME)),
        ('time offsets 1D', timed(arr([0, 1])), expect(RUNTIME)),
        (
            'time offsets empty rows',
            timed(arr([[]]).reshape(0, 2)),
            expect('IndexError'),
        ),
        (
            'time offsets text',
            timed(np.array([['a', 'b']], dtype=object)),
            expect(RUNTIME),
        ),
        (
            'time offsets past int64',
            timed(np.array([[None, 2**63]], dtype=object)),
            expect(RUNTIME),
        ),
        (
            'time offsets end None',
            timed(np.array([[0, None]], dtype=object)),
            expect(RUNTIME),
        ),
        (
            'time offsets fewer than hops',
            timed(arr([[0, 1]]), hops=2),
            expect('IndexError'),
        ),
        (
            'time offsets and counts',
            {'sampling_time_offsets': {'A__to__B': arr([[0, 1]])}},
            expect(RUNTIME),
        ),
    ]
    for name, changes, outcomes in samples:
        yield (
            f'NeighborSampler.sample({name})',
            lambda c=changes: sample(**c),
            outcomes,
        )


def graph_json(**changes: Any) -> str:
    graph: dict[str, Any] = {
        'tables': [
            {
                'name': 'users',
                'primary_key': 'id',
                'time_column': None,
                'end_time_column': None,
                'columns': [{'name': 'id', 'dtype': 'int', 'stype': 'ID'}],
            },
            {
                'name': 'orders',
                'primary_key': 'oid',
                'time_column': 'ts',
                'end_time_column': None,
                'columns': [
                    {'name': 'oid', 'dtype': 'int', 'stype': 'ID'},
                    {'name': 'uid', 'dtype': 'int', 'stype': 'ID'},
                    {'name': 'ts', 'dtype': 'time', 'stype': 'timestamp'},
                ],
            },
        ],
        'edges': [{'src_table': 'orders', 'fkey': 'uid', 'dst_table': 'users'}],
    }
    graph.update(changes)
    return json.dumps(graph)


def store_cases(n: Any) -> Iterator[Case]:
    import pyarrow as pa

    def tables(**changes: Any) -> dict[str, Any]:
        out = {
            'users': pa.table({'id': pa.array([1, 2, 3], pa.int64())}),
            'orders': pa.table(
                {
                    'oid': pa.array([1, 2], pa.int64()),
                    'uid': pa.array([1, 5], pa.int64()),
                    'ts': pa.array([0, 10**18], pa.timestamp('ns')),
                }
            ),
        }
        out.update(changes)
        return out

    users = json.loads(graph_json())['tables'][0]
    orders = json.loads(graph_json())['tables'][1]
    cases: list[tuple[str, Any, Any, frozenset[str]]] = [
        ('valid', tables(), graph_json(), expect(OK)),
        ('tables None', None, graph_json(), expect(TYPE)),
        ('table int', tables(users=5), graph_json(), expect(TYPE)),
        ('table dict', tables(users={'id': [1]}), graph_json(), expect(TYPE)),
        ('tables empty', {}, graph_json(), expect(KEY, VALUE)),
        ('graph None', tables(), None, expect(TYPE)),
        *(
            (f'graph {s!r}', tables(), s, expect(VALUE))
            for s in (*UNREADABLE_JSON, '[]', '{}', '{"tables": 5}')
        ),
        ('no tables', tables(), graph_json(tables=[], edges=[]), expect(OK)),
        (
            'edge to a missing table',
            tables(),
            graph_json(
                edges=[
                    {'src_table': 'orders', 'fkey': 'uid', 'dst_table': 'nope'}
                ]
            ),
            expect(KEY, VALUE),
        ),
        (
            'edge from a missing column',
            tables(),
            graph_json(
                edges=[
                    {
                        'src_table': 'orders',
                        'fkey': 'nope',
                        'dst_table': 'users',
                    }
                ]
            ),
            expect(KEY),
        ),
        (
            'edge to a table without a primary key',
            tables(),
            graph_json(tables=[{**users, 'primary_key': None}, orders]),
            expect(KEY),
        ),
        (
            'primary key missing',
            tables(),
            graph_json(tables=[{**users, 'primary_key': 'nope'}, orders]),
            expect(KEY),
        ),
        (
            'time column missing',
            tables(),
            graph_json(tables=[users, {**orders, 'time_column': 'nope'}]),
            expect(KEY),
        ),
        (
            'extreme keys',
            tables(
                users=pa.table(
                    {'id': pa.array([2**63 - 1, -(2**63), 0], pa.int64())}
                ),
                orders=pa.table(
                    {
                        'oid': pa.array([-(2**63), 2**63 - 1], pa.int64()),
                        'uid': pa.array([2**63 - 1, None], pa.int64()),
                        'ts': pa.array(
                            [-(2**63) + 1, 2**63 - 1], pa.timestamp('ns')
                        ),
                    }
                ),
            ),
            graph_json(),
            expect(OK),
        ),
        (
            'empty tables',
            tables(
                users=pa.table({'id': pa.array([], pa.int64())}),
                orders=pa.table(
                    {
                        'oid': pa.array([], pa.int64()),
                        'uid': pa.array([], pa.int64()),
                        'ts': pa.array([], pa.timestamp('ns')),
                    }
                ),
            ),
            graph_json(),
            # LocalGraphStore refuses a table with no valid rows.
            expect(VALUE),
        ),
        (
            'text times that do not parse',
            tables(
                orders=pa.table(
                    {
                        'oid': pa.array([1, 2], pa.int64()),
                        'uid': pa.array([1, 2], pa.int64()),
                        'ts': pa.array(['not a time', '99999-99-99']),
                    }
                )
            ),
            graph_json(),
            expect(OK, VALUE, 'DateParseError', 'OutOfBoundsDatetime'),
        ),
        (
            'keys of another type',
            tables(users=pa.table({'id': pa.array([[1], [2]])})),
            graph_json(),
            expect(OK, VALUE, TYPE),
        ),
        (
            'a column of nulls',
            tables(users=pa.table({'id': pa.array([None, None], pa.null())})),
            graph_json(),
            expect(OK, VALUE, TYPE),
        ),
    ]
    for name, data, graph, outcomes in cases:
        yield (
            f'graph_store({name})',
            lambda d=data, g=graph: n.graph_store(d, g),
            outcomes,
        )


LABELS_SCHEMA = {
    'tables': {
        'users': {
            'cols': [
                {
                    'name': 'user_id',
                    'stype': 'ID',
                    'dtype': 'int',
                    'timestamp_format': None,
                }
            ],
            'pkey': 'user_id',
            'time_col': None,
            'end_time_col': None,
        },
        'orders': {
            'cols': [
                {
                    'name': 'order_id',
                    'stype': 'ID',
                    'dtype': 'int',
                    'timestamp_format': None,
                },
                {
                    'name': 'user_id',
                    'stype': 'ID',
                    'dtype': 'int',
                    'timestamp_format': None,
                },
                {
                    'name': 'amount',
                    'stype': 'numerical',
                    'dtype': 'float',
                    'timestamp_format': None,
                },
                {
                    'name': 'ordered_at',
                    'stype': 'timestamp',
                    'dtype': 'time',
                    'timestamp_format': None,
                },
            ],
            'pkey': 'order_id',
            'time_col': 'ordered_at',
            'end_time_col': None,
        },
    },
    'col_groups': [
        {
            'columns': [
                {'table_name': 'orders', 'col_name': 'user_id'},
                {'table_name': 'users', 'col_name': 'user_id'},
            ]
        }
    ],
}
LABELS_QUERIES = (
    'PREDICT SUM(orders.amount, 0, 7, days) FOR EACH users.user_id',
    'PREDICT COUNT(orders.*, 0, 40000000, months) FOR EACH users.user_id',
    'PREDICT MIN(orders.amount, 0, 1, seconds) IS NULL FOR EACH users.user_id',
    'PREDICT AVG(orders.amount, 0, 7, days) > 2 FOR EACH users.user_id WHERE users.user_id > 1',
)


def labels_cases(n: Any) -> Iterator[Case]:
    import pyarrow as pa

    def tree(query: str) -> str:
        out = json.loads(
            n.pql_validate(query, json.dumps(LABELS_SCHEMA), 'RELATIONAL')
        )
        return json.dumps(out['tree'])

    times = pa.array([0, 86_400 * 10**9, 2**63 - 1], pa.timestamp('ns'))

    def args(**changes: Any) -> dict[str, Any]:
        out = {
            'feat': {
                'users': pa.table({'user_id': pa.array([1, 2], pa.int64())}),
                'orders': pa.table(
                    {
                        'order_id': pa.array([1, 2, 3], pa.int64()),
                        'user_id': pa.array([1, 1, 2], pa.int64()),
                        'amount': pa.array([1.0, float('nan'), 1e308]),
                        'ordered_at': times,
                    }
                ),
            },
            'times': {
                'orders': times,
                'users': pa.array([None, None], pa.timestamp('ns')),
            },
            'batches': {
                'orders': pa.array([0, 0, 1], pa.int64()),
                'users': pa.array([0, 1], pa.int64()),
            },
            'anchor': pa.array([-(2**63) + 1, 0], pa.timestamp('ns')),
        }
        out.update(changes)
        return out

    trees = [tree(q) for q in LABELS_QUERIES]
    for i, query in enumerate(trees):
        yield (
            f'labels_execute(query {i})',
            lambda q=query: n.labels_execute(q, **args()),
            expect(OK, VALUE, 'OutOfBoundsDatetime', 'OverflowError'),
        )
    query = trees[0]
    hostile: list[tuple[str, Any, dict[str, Any], frozenset[str]]] = [
        ('query None', None, {}, expect(TYPE)),
        *(
            (f'query {s!r}', s, {}, expect(VALUE))
            for s in (*UNREADABLE_JSON, '[]', '{}', 'null')
        ),
        ('feat None', query, {'feat': None}, expect(TYPE)),
        ('feat empty', query, {'feat': {}}, expect(KEY)),
        ('feat table int', query, {'feat': {'orders': 5}}, expect(TYPE)),
        ('times empty', query, {'times': {}}, expect(KEY)),
        (
            'times list',
            query,
            {'times': {'orders': [1, 2, 3]}},
            expect(TYPE, VALUE),
        ),
        (
            'times text',
            query,
            {
                'times': {
                    'orders': pa.array(['a', 'b', 'c']),
                    'users': pa.array(['a', 'b']),
                }
            },
            expect(VALUE, TYPE),
        ),
        ('batches empty', query, {'batches': {}}, expect(KEY)),
        (
            'batches text',
            query,
            {
                'batches': {
                    'orders': pa.array(['a', 'b', 'c']),
                    'users': pa.array(['a', 'b']),
                }
            },
            expect(TYPE, VALUE),
        ),
        (
            'batches past the anchors',
            query,
            {
                'batches': {
                    'orders': pa.array([2**63 - 1, -(2**63), 5], pa.int64()),
                    'users': pa.array([0, 1], pa.int64()),
                }
            },
            expect(OK, 'IndexError'),
        ),
        (
            'batches short',
            query,
            {
                'batches': {
                    'orders': pa.array([0], pa.int64()),
                    'users': pa.array([0, 1], pa.int64()),
                }
            },
            expect(OK, VALUE, 'IndexError'),
        ),
        ('anchor None', query, {'anchor': None}, expect(TYPE)),
        (
            'anchor empty',
            query,
            {'anchor': pa.array([], pa.timestamp('ns'))},
            expect(OK, 'IndexError'),
        ),
        (
            'anchor ints',
            query,
            {'anchor': pa.array([0, 1], pa.int64())},
            expect(OK, VALUE, TYPE),
        ),
        (
            'anchor text',
            query,
            {'anchor': pa.array(['x', 'y'])},
            expect(VALUE, TYPE),
        ),
    ]
    for name, q, changes, outcomes in hostile:
        yield (
            f'labels_execute({name})',
            lambda q=q, c=changes: n.labels_execute(q, **args(**c)),
            outcomes,
        )


HOSTILE_VALUES: tuple[Any, ...] = (
    None,
    True,
    0,
    -1,
    NO_ROOM,
    2**63 - 1,
    -(2**63),
    2**64,
    1e308,
    -1e308,
    'x',
    '',
    [],
    {},
)


def leaf_paths(
    value: Any, path: tuple[Any, ...] = ()
) -> Iterator[tuple[Any, ...]]:
    yield path
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaf_paths(item, (*path, key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from leaf_paths(item, (*path, i))


def replaced(value: Any, path: tuple[Any, ...], new: Any) -> Any:
    if not path:
        return new
    out = copy.deepcopy(value)
    at = out
    for step in path[:-1]:
        at = at[step]
    if new is DROP:
        del at[path[-1]]
    else:
        at[path[-1]] = new
    return out


DROP = object()


def op_cases(n: Any) -> Iterator[Case]:
    for name, op in (
        ('labels_op', n.labels_op),
        ('sampling_op', n.sampling_op),
    ):
        yield f'{name} unknown', lambda o=op: o('nope.op', '{}'), expect(VALUE)
        for text in (*UNREADABLE_JSON, '[]', '1', 'null'):
            yield (
                f'{name} input {text!r}',
                lambda o=op, t=text: o(
                    'labels.column'
                    if o is n.labels_op
                    else 'sampling.random_state',
                    t,
                ),
                expect(VALUE),
            )
        yield f'{name} op None', lambda o=op: o(None, '{}'), expect(TYPE)
    rng = random.Random(MUTATION_SEED)
    for file in FIXTURE_FILES:
        cases = json.loads((FIXTURES / file).read_text(encoding='utf-8'))[
            'cases'
        ]
        for case in cases:
            operation = case['operation']
            op = (
                n.labels_op
                if operation.startswith('labels.')
                else n.sampling_op
            )
            paths = list(leaf_paths(case['input']))
            for i in range(12):
                path = rng.choice(paths)
                new = (
                    DROP
                    if path and rng.random() < 0.15
                    else rng.choice(HOSTILE_VALUES)
                )
                mutated = replaced(case['input'], path, new)
                yield (
                    f'{operation} {case["id"]} #{i} at {list(path)}',
                    lambda o=op, a=operation, m=mutated: _op_output(o, a, m),
                    expect(OK, VALUE),
                )


def _op_output(op: Any, operation: str, value: Any) -> Any:
    answer = json.loads(op(operation, json.dumps(value)))
    assert isinstance(answer, dict) and set(answer) in (
        {'output'},
        {'error'},
    ), answer
    return answer


GROUPS: dict[str, tuple[Callable[[Any], Iterator[Case]], bool]] = {
    'random': (random_cases, False),
    'offsets': (offset_cases, False),
    'text': (text_cases, False),
    'sampler': (sampler_cases, True),
    'store': (store_cases, False),
    'labels': (labels_cases, False),
    'ops': (op_cases, False),
}


def _child(group: str) -> None:
    r"""Run one group's calls, writing a JSON line before and after each."""
    from sdm_core import _native

    # The case names print integers of any size (Python 3.10.7 and later).
    if hasattr(sys, 'set_int_max_str_digits'):
        sys.set_int_max_str_digits(0)
    cases, needs_numpy = GROUPS[group]
    if needs_numpy:
        try:
            import numpy  # noqa: F401
        except ImportError:
            print(json.dumps({'skip': 'numpy is not installed'}), flush=True)
            return
    for name, call, outcomes in cases(_native):
        print(json.dumps({'start': name}), flush=True)
        try:
            call()
        except BaseException as error:  # noqa: BLE001 - a panic is one
            got = type(error).__name__
            message = str(error)
        else:
            got, message = OK, ''
        print(
            json.dumps(
                {
                    'case': name,
                    'got': got,
                    'message': message[:300],
                    'want': sorted(outcomes),
                }
            ),
            flush=True,
        )


def _sweep(group: str) -> list[str]:
    child = subprocess.run(
        [sys.executable, __file__, group],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    failures = []
    started = None
    for line in child.stdout.splitlines():
        record = json.loads(line)
        if 'skip' in record:
            pytest.skip(record['skip'])
        if 'start' in record:
            started = record['start']
            continue
        started = None
        got, message = record['got'], record['message']
        if 'panicked' in message:
            failures.append(f'{record["case"]}: panicked: {message}')
        elif got not in record['want']:
            failures.append(
                f'{record["case"]}: {got}({message!r}), want {" or ".join(record["want"])}'
            )
    if child.returncode != 0:
        failures.append(
            f'the child ended with {child.returncode} during {started!r}: '
            f'{child.stderr[-2000:]}'
        )
    return failures


def _check(group: str) -> None:
    failures = _sweep(group)
    assert not failures, (
        f'{len(failures)} hostile call(s) misbehaved:\n'
        + '\n'.join(failures[:60])
    )


def test_random_numbers_refuse_hostile_input_without_panicking() -> None:
    _check('random')


def test_date_offsets_refuse_hostile_input_without_panicking() -> None:
    _check('offsets')


def test_text_normalization_takes_hostile_input_without_panicking() -> None:
    _check('text')


def test_the_neighbor_sampler_refuses_hostile_input_without_panicking() -> None:
    _check('sampler')


def test_the_graph_store_refuses_hostile_input_without_panicking() -> None:
    _check('store')


def test_labels_execute_refuses_hostile_input_without_panicking() -> None:
    _check('labels')


def test_mutated_fixture_operations_answer_without_panicking() -> None:
    _check('ops')


if __name__ == '__main__':
    _child(sys.argv[1])
