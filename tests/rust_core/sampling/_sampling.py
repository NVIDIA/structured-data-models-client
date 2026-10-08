# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What the Phase 7 acceptance checks share: building the binding, today's
Python answers (numpy's legacy ``RandomState``, the C++ ``std::mt19937``,
``relationallib``, the local graph store, pandas' ``DateOffset`` arithmetic
and the sampler's text normalization), the seeded generators, and the
comparisons.

The interface these rely on is ``CONTRACT.md``, beside this file.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FIXTURES = REPO / 'tests' / 'parity' / 'fixtures'
TRANSCRIPTS = REPO / 'tests' / 'parity' / 'transcripts'
CPP_REFERENCE = HERE / 'mt19937_reference.cpp'

for _path in (
    HERE,
    HERE.parent,
    REPO / 'tests' / 'parity',
    REPO / 'scripts' / 'parity',
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

#: Set to ``1`` in a process started after the wheel was built and installed.
INSTALLED = 'SDM_P7_WHEEL_INSTALLED'

#: The units an offset may name, as ``pd.DateOffset`` keywords.
OFFSET_UNITS = (
    'years',
    'months',
    'weeks',
    'days',
    'hours',
    'minutes',
    'seconds',
)

NS_MIN = -(2**63) + 1  # pd.Timestamp.min.value; -2**63 is NaT
NS_MAX = 2**63 - 1  # pd.Timestamp.max.value


# --- The Rust core, through the binding --------------------------------------

_NATIVE: dict[str, Any] = {}


def _wheel_cache() -> Path | None:
    r"""Where this commit's wheel is kept, when the worktree is clean.

    Every check builds the wheel once per process; a clean commit's wheel is
    the same bytes every time, so it is built once and reinstalled after.
    """
    from _rc import head_sha, worktree_clean

    if not worktree_clean():
        return None
    root = Path(
        os.environ.get(
            'SDM_RC_P7_CACHE',
            str(Path(tempfile.gettempdir()) / 'sdm-rust-core-p7'),
        )
    ).expanduser()
    return root / head_sha()


def _install() -> None:
    from _rc import build_local_wheel, install_wheel

    cache = _wheel_cache()
    if cache is not None:
        wheels = sorted(cache.glob('*.whl'))
        if (cache / 'DONE').exists() and len(wheels) == 1:
            install_wheel(wheels[0])
            return
        if cache.exists():
            shutil.rmtree(cache)
        cache.mkdir(parents=True)
        wheel = build_local_wheel(cache)
        (cache / 'DONE').write_text('built\n')
        install_wheel(wheel)
        return
    out = Path(tempfile.mkdtemp(prefix='sdm-core-p7-wheel-'))
    install_wheel(build_local_wheel(out))


def native(*names: str) -> Any:
    r"""``sdm_core._native``, freshly built from this checkout and installed
    into this environment, once per process, holding every one of ``names``.

    A build that fails fails every later call the same way, without building
    again.
    """
    if 'error' in _NATIVE:
        raise AssertionError(_NATIVE['error'])
    if 'module' not in _NATIVE:
        try:
            # A process the checks start after building the wheel themselves
            # imports what they installed rather than building it again.
            if os.environ.get(INSTALLED) != '1':
                _install()
            importlib.invalidate_caches()
            _NATIVE['module'] = importlib.import_module('sdm_core._native')
        except Exception as error:  # noqa: BLE001 - kept for every later call
            _NATIVE['error'] = f'the binding is not usable: {error}'
            raise AssertionError(_NATIVE['error']) from error
    module = _NATIVE['module']
    missing = [name for name in names if not hasattr(module, name)]
    assert not missing, f'sdm_core._native has no {", ".join(missing)}'
    return module


def loads(text: Any, what: str) -> Any:
    assert isinstance(text, str), f'{what} gave {type(text).__name__}, not str'
    try:
        return json.loads(text)
    except ValueError as error:
        raise AssertionError(
            f'{what} gave text that is not JSON: {error}'
        ) from error


def as_list(value: Any) -> list[Any]:
    r"""A list, a tuple, a numpy array or a pyarrow array, as a list."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, 'to_pylist'):
        return value.to_pylist()
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise AssertionError(f'{type(value).__name__} is not a sequence')


def error_of(error: BaseException) -> dict[str, str]:
    r"""``{"code", "message"}``: the class name and the text."""
    return {'code': type(error).__name__, 'message': str(error)}


def short(value: Any, limit: int = 400) -> str:
    text = json.dumps(value, default=repr)
    return text if len(text) <= limit else text[:limit] + '...'


def report(failures: list[str], total: int, what: str) -> None:
    assert not failures, (
        f'{len(failures)} of {total} {what} differ:\n'
        + '\n'.join(failures[:25])
    )


# --- Random numbers ------------------------------------------------------------

RNG_SEED = 20261010
NUM_SEEDS = 10_000

SEED_EDGES = (
    0,
    1,
    2,
    7,
    42,
    123,
    2**31 - 1,
    2**31,
    2**32 - 2,
    2**32 - 1,
    2**32,
    2**32 + 1,
    2**63 - 1,
    2**63,
    2**64 - 1,
    2**64,
    2**64 + 1,
    2**96,
    2**128 + 12345,
    10**40,
)


def seeds() -> list[int]:
    r"""The seeds of P7.rng, edges first: a quarter below 2**31, a quarter up
    to 2**32, a quarter up to 2**64 and a quarter up to 2**160."""
    rng = random.Random(RNG_SEED)
    out = list(SEED_EDGES)
    while len(out) < NUM_SEEDS:
        kind = len(out) % 4
        if kind == 0:
            out.append(rng.randrange(0, 2**31))
        elif kind == 1:
            out.append(rng.randrange(2**31, 2**32))
        elif kind == 2:
            out.append(rng.randrange(2**32, 2**64))
        else:
            out.append(rng.randrange(2**64, 2**160))
    return out


def seed_words(seed: int) -> list[int]:
    r"""The 32-bit words ``base/utils.random_state`` seeds a large seed with,
    least significant first."""
    words = []
    rest = seed
    while rest > 0:
        words.append(rest % 2**32)
        rest //= 2**32
    return words


#: ``(low, high)`` bounds of ``randint``, on both sides of 2**32.
RANDINT_BOUNDS = (
    (0, 1),
    (0, 2),
    (0, 7),
    (0, 10),
    (-50, 50),
    (0, 1000),
    (0, 2**31 - 1),
    (1, 2**31 - 1),
    (0, 2**31),
    (0, 2**32 - 1),
    (0, 2**32),
    (0, 2**32 + 1),
    (0, 2**40),
    (-(2**40), 2**40),
    (0, 2**62),
    (-(2**63), 2**63 - 1),
)

INTERVAL_MAXIMA = (
    0,
    1,
    2,
    3,
    6,
    100,
    2**31 - 1,
    2**31,
    2**32 - 2,
    2**32 - 1,
    2**32,
    2**33 + 5,
    2**40,
    2**62,
    2**63 - 2,
)


def rng_ops(rng: random.Random) -> list[list[Any]]:
    r"""A short sequence of draws from one stream."""
    ops: list[list[Any]] = []
    for _ in range(rng.randint(1, 5)):
        kind = rng.choice(
            (
                'u32',
                'u64',
                'interval',
                'randint',
                'shuffle',
                'permutation',
                'choice',
            )
        )
        if kind in ('u32', 'u64'):
            count = rng.choice((1, 2, 5, 17, 300, 623, 624, 625, 1300))
            ops.append([kind, count])
        elif kind == 'interval':
            ops.append([kind, rng.choice(INTERVAL_MAXIMA)])
        elif kind == 'randint':
            low, high = rng.choice(RANDINT_BOUNDS)
            ops.append([kind, low, high, rng.randint(1, 40)])
        elif kind == 'shuffle':
            n = rng.choice((0, 1, 2, 3, 10, 57, 1000, rng.randint(0, 3000)))
            offset = rng.randint(-1000, 1000)
            ops.append([kind, [offset + i for i in range(n)]])
        elif kind == 'permutation':
            ops.append(
                [kind, rng.choice((0, 1, 2, 9, 100, rng.randint(0, 3000)))]
            )
        else:
            a = rng.choice((1, 2, 5, 40, 1000, rng.randint(1, 5000)))
            ops.append([kind, a, rng.randint(0, a)])
    return ops


def numpy_draws(state: Any, ops: list[list[Any]]) -> list[Any]:
    r"""What numpy's legacy ``RandomState`` gives for ``ops``, in order."""
    import numpy as np

    out: list[Any] = []
    for op in ops:
        kind = op[0]
        if kind == 'u32':
            out.append(
                state.randint(0, 2**32, size=op[1], dtype=np.uint32).tolist()
            )
        elif kind == 'u64':
            out.append(
                state.randint(0, 2**64, size=op[1], dtype=np.uint64).tolist()
            )
        elif kind == 'interval':
            out.append(int(state.randint(0, op[1] + 1, dtype=np.int64)))
        elif kind == 'randint':
            out.append(
                state.randint(op[1], op[2], size=op[3], dtype=np.int64).tolist()
            )
        elif kind == 'shuffle':
            values = np.array(op[1], dtype=np.int64)
            state.shuffle(values)
            out.append(values.tolist())
        elif kind == 'permutation':
            out.append(state.permutation(op[1]).tolist())
        else:
            out.append(state.choice(op[1], size=op[2], replace=False).tolist())
    return out


def rust_draws(state: Any, ops: list[list[Any]]) -> list[Any]:
    r"""What the binding's ``RandomState`` gives for ``ops``, in order."""
    out: list[Any] = []
    for op in ops:
        kind = op[0]
        if kind == 'u32':
            out.append([state.next_u32() for _ in range(op[1])])
        elif kind == 'u64':
            out.append([state.next_u64() for _ in range(op[1])])
        elif kind == 'interval':
            out.append(state.interval(op[1]))
        elif kind == 'randint':
            out.append(as_list(state.randint(op[1], op[2], op[3])))
        elif kind == 'shuffle':
            out.append(as_list(state.shuffle(list(op[1]))))
        elif kind == 'permutation':
            out.append(as_list(state.permutation(op[1])))
        else:
            out.append(as_list(state.choice(op[1], op[2])))
    return out


def mt_jobs(seed_list: Sequence[int]) -> list[list[Any]]:
    r"""For each seed, raw draws and the neighbor sampler's picks, mixed."""
    rng = random.Random(RNG_SEED + 1)
    jobs = []
    for given in seed_list:
        # The C++ seeds with an unsigned 64-bit word at most.
        seed = given % 2**64
        ops: list[list[Any]] = []
        for _ in range(rng.randint(1, 4)):
            if rng.random() < 0.5:
                ops.append(['u', rng.choice((1, 3, 100, 623, 624, 625, 1500))])
            else:
                population = rng.choice(
                    (2, 3, 8, 50, 1000, rng.randint(2, 200_000))
                )
                count = rng.randint(1, min(population - 1, 300))
                ops.append(['f', population, count])
        jobs.append([seed, ops])
    return jobs


_CPP: dict[str, Path] = {}


def cpp_reference() -> Path:
    r"""``mt19937_reference.cpp``, compiled with the system's C++ compiler."""
    if 'binary' in _CPP:
        return _CPP['binary']
    compiler = (
        shutil.which('c++') or shutil.which('clang++') or shutil.which('g++')
    )
    assert compiler, 'no C++ compiler to build the std::mt19937 reference'
    out = Path(tempfile.mkdtemp(prefix='sdm-p7-mt19937-')) / 'mt19937_reference'
    proc = subprocess.run(
        [compiler, '-std=c++17', '-O2', str(CPP_REFERENCE), '-o', str(out)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f'the reference does not compile:\n{proc.stderr}'
    )
    _CPP['binary'] = out
    return out


def cpp_draws(jobs: list[list[Any]]) -> list[list[int]]:
    r"""What the C++ standard library's ``std::mt19937`` gives for each job."""
    lines = []
    for seed, ops in jobs:
        parts = [str(seed)]
        for op in ops:
            parts += [str(x) for x in op]
        lines.append(' '.join(parts))
    proc = subprocess.run(
        [str(cpp_reference())],
        input='\n'.join(lines) + '\n',
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.split('\n')
    assert len(out) >= len(jobs), 'the reference answered fewer jobs than asked'
    return [[int(x) for x in line.split()] for line in out[: len(jobs)]]


def rust_mt_draws(module: Any, job: list[Any]) -> list[int]:
    seed, ops = job
    rng = module.Mt19937(seed)
    out: list[int] = []
    for op in ops:
        if op[0] == 'u':
            out += [rng.next_u32() for _ in range(op[1])]
        else:
            out += as_list(rng.floyd(op[1], op[2]))
    return out


# --- relationallib's NeighborSampler -------------------------------------------

GRAPH_SEED = 20261011
NUM_GRAPHS = 10_000


def relationallib() -> Any:
    import sdm_relational_engine.relationallib as module

    return module


def _rel(edge: Sequence[str]) -> str:
    return '__'.join(edge)


def random_graph(rng: random.Random) -> dict[str, Any]:
    r"""A random heterogeneous graph in ``relationallib``'s CSC form, and the
    arguments of a few ``sample`` calls over it, as plain data.

    An edge type ``(a, rel, b)`` is sampled from ``b``'s nodes to ``a``'s,
    as the C++ reads it: its ``colptr`` has one entry per node of ``b`` and
    one more, and its ``row`` holds nodes of ``a``. A neighborhood of an edge
    whose ``a`` has times is sorted by time, oldest first, as the graph store
    builds it, except in about one graph in a hundred.
    """
    num_types = rng.randint(1, 4)
    names = rng.sample(
        ['users', 'orders', 'items', 'n3', 'Ünï', 'a b'], num_types
    )
    sizes = {name: rng.randint(0, 25) for name in names}
    entity = rng.choice(names)
    sizes[entity] = max(sizes[entity], 1)
    timed = [name for name in names if rng.random() < 0.6]
    span = rng.choice((3, 50, 10_000))
    node_time = {
        name: [rng.randint(-span, span) for _ in range(sizes[name])]
        for name in timed
    }
    unsorted = rng.random() < 0.01
    edges = []
    colptr: dict[str, list[int]] = {}
    row: dict[str, list[int]] = {}
    for index in range(rng.randint(0, 5)):
        a, b = rng.choice(names), rng.choice(names)
        edge = [a, f'r{index}', b]
        edges.append(edge)
        widest = rng.choice((1, 3, 8, 30))
        pointers, rows = [0], []
        for _ in range(sizes[b]):
            count = rng.randint(0, widest) if sizes[a] else 0
            hood = [rng.randrange(sizes[a]) for _ in range(count)]
            if a in node_time and not unsorted:
                hood.sort(key=lambda node, a=a: node_time[a][node])
            rows += hood
            pointers.append(len(rows))
        colptr[_rel(edge)] = pointers
        row[_rel(edge)] = rows
    hops = rng.choice((1, 1, 2, 2, 3))
    calls = []
    for _ in range(rng.randint(1, 3)):
        num_neighbors: dict[str, list[int]] = {}
        offsets: dict[str, list[list[int | None]]] = {}
        for edge in edges:
            if edge[0] in node_time and rng.random() < 0.35:
                ranges = []
                for _ in range(hops):
                    end = rng.randint(-span // 2 - 1, span // 4 + 1)
                    start = (
                        None
                        if rng.random() < 0.3
                        else end - rng.randint(0, span + 1)
                    )
                    ranges.append([start, end])
                offsets[_rel(edge)] = ranges
            else:
                num_neighbors[_rel(edge)] = [
                    rng.choice((-1, 0, 1, 1, 2, 2, 3, 5, 8))
                    for _ in range(hops)
                ]
        seed_nodes = [
            rng.randrange(sizes[entity]) for _ in range(rng.randint(1, 10))
        ]
        if rng.random() < 0.6:
            seed_time = [rng.randint(-span, span) for _ in seed_nodes]
        else:
            seed_time = []
        calls.append(
            {
                'num_neighbors': num_neighbors,
                'offsets': offsets,
                'entity': entity,
                'seed_node': seed_nodes,
                'seed_time': seed_time,
                'reseed': rng.choice((None, None, rng.randint(0, 2**31 - 1))),
            }
        )
    return {
        'node_types': names,
        'edge_types': edges,
        'colptr': colptr,
        'row': row,
        'node_time': node_time,
        'seed': rng.choice((0, 123, 2**31 - 1, rng.randint(0, 2**31 - 1))),
        'calls': calls,
    }


def graphs(
    count: int = NUM_GRAPHS, seed: int = GRAPH_SEED
) -> Iterator[dict[str, Any]]:
    rng = random.Random(seed)
    for _ in range(count):
        yield random_graph(rng)


def draws_random_numbers(graph: dict[str, Any]) -> bool:
    r"""Whether some static hop asks for fewer neighbors than a node has,
    which is when the C++ draws from its generator."""
    has_time = bool(graph['node_time'])
    for call in graph['calls']:
        for edge in graph['edge_types']:
            counts = call['num_neighbors'].get(_rel(edge))
            if counts is None:
                continue
            static = (
                not has_time
                or edge[0] not in graph['node_time']
                or not (
                    call['seed_time'] or call['entity'] in graph['node_time']
                )
            )
            pointers = graph['colptr'][_rel(edge)]
            widest = max(
                (
                    pointers[i + 1] - pointers[i]
                    for i in range(len(pointers) - 1)
                ),
                default=0,
            )
            if static and any(0 < c < widest for c in counts):
                return True
    return False


def _arrays(graph: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    def ints(values: Sequence[int]) -> Any:
        return np.array(values, dtype=np.int64)

    return {
        'node_types': list(graph['node_types']),
        'edge_types': [tuple(edge) for edge in graph['edge_types']],
        'colptr': {k: ints(v) for k, v in graph['colptr'].items()},
        'row': {k: ints(v) for k, v in graph['row'].items()},
        'node_time': {k: ints(v) for k, v in graph['node_time'].items()},
    }


def _call_args(call: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    return {
        'num_neighbors': {
            k: np.array(v, dtype=np.int64)
            for k, v in call['num_neighbors'].items()
        },
        'sampling_time_offsets': {k: v for k, v in call['offsets'].items()},
        'entity_table': call['entity'],
        'seed_node': np.array(call['seed_node'], dtype=np.int64),
        'seed_time': np.array(call['seed_time'], dtype=np.int64),
    }


def sampler_output(out: Any) -> Any:
    r"""A ``sample`` result as plain data: six dicts, each a list of
    ``[key, dtype, values]`` in the dict's order."""
    import numpy as np

    assert isinstance(out, tuple) and len(out) == 6, (
        f'sample returned {type(out).__name__}, not a 6-tuple'
    )
    shaped = []
    for part in out:
        assert isinstance(part, dict), f'sample gave {type(part).__name__}'
        items = []
        for key, value in part.items():
            kind = (
                str(value.dtype)
                if isinstance(value, np.ndarray)
                else type(value).__name__
            )
            items.append([key, kind, as_list(value)])
        shaped.append(items)
    return shaped


def run_sampler(cls: Any, graph: dict[str, Any]) -> list[Any]:
    r"""Every outcome of the graph's ``sample`` calls through ``cls``: a
    result, or the error the constructor or the call raised."""
    arrays = _arrays(graph)
    outcomes: list[Any] = []
    try:
        sampler = cls(
            arrays['node_types'],
            arrays['edge_types'],
            arrays['colptr'],
            arrays['row'],
            arrays['node_time'],
            seed=graph['seed'],
        )
    except Exception as error:  # noqa: BLE001 - compared, not judged
        return [{'error': error_of(error)}]
    for call in graph['calls']:
        try:
            if call['reseed'] is not None:
                sampler.seed(call['reseed'])
            outcomes.append(sampler_output(sampler.sample(**_call_args(call))))
        except Exception as error:  # noqa: BLE001 - compared, not judged
            outcomes.append({'error': error_of(error)})
    return outcomes


# --- The local graph store -----------------------------------------------------

STORE_SEED = 20261012
NUM_STORES = 1_000


def _key_values(kind: str, ints: list[int]) -> list[Any]:
    r"""Keys of ``kind`` standing for ``ints``."""
    if kind in ('int', 'Int64'):
        return [k * 7 - 20 for k in ints]
    if kind == 'float':
        return [k * 1.5 for k in ints]
    return [f'k{k}' for k in ints]


def _with_nulls(
    rng: random.Random, kind: str, values: list[Any], rate: float
) -> Any:
    import numpy as np
    import pandas as pd

    nulls = [rng.random() < rate for _ in values]
    if kind == 'int':
        return pd.Series(values, dtype='int64')
    if kind == 'Int64':
        return pd.Series(
            [pd.NA if z else v for v, z in zip(values, nulls, strict=True)],
            dtype='Int64',
        )
    if kind == 'float':
        return pd.Series(
            [np.nan if z else v for v, z in zip(values, nulls, strict=True)],
            dtype='float64',
        )
    return pd.Series(
        [None if z else v for v, z in zip(values, nulls, strict=True)],
        dtype=object,
    )


def _times(rng: random.Random, n: int, kind: str) -> Any:
    import pandas as pd

    base = rng.choice(
        (
            pd.Timestamp('2024-01-01').value,
            pd.Timestamp('1969-12-31 23:59:59').value,
            pd.Timestamp('1900-03-01').value,
        )
    )
    step = rng.choice((10**9, 86_400 * 10**9, 1, 999_999_999))
    values = []
    for _ in range(n):
        if rng.random() < 0.1:
            values.append(None)
        else:
            values.append(
                base
                + rng.randint(-5, 30) * step
                + rng.choice((0, 0, 1, 500_000_000))
            )
    if kind == 'iso':
        return pd.Series(
            [
                None
                if v is None
                else pd.Timestamp(v).strftime('%Y-%m-%d %H:%M:%S')
                for v in values
            ],
            dtype=object,
        )
    return pd.Series(
        [pd.NaT if v is None else pd.Timestamp(v) for v in values],
        dtype='datetime64[ns]',
    )


def random_store(rng: random.Random) -> dict[str, Any]:
    r"""A random graph of in-memory tables, as data frames and their keys:
    keys of several types, with nulls and repeats, times of several kinds,
    with nulls, and foreign keys that miss."""
    import numpy as np

    num_tables = rng.randint(1, 4)
    names = [f'T{i}' for i in range(num_tables)]
    kinds = {
        name: rng.choice(('int', 'Int64', 'float', 'str', None))
        for name in names
    }
    pools = {name: rng.randint(1, 80) for name in names}
    frames = {}
    times = {}
    for name in names:
        n = rng.randint(1, 40)
        columns: dict[str, Any] = {}
        kind = kinds[name]
        if kind is not None:
            rate = rng.choice((0.0, 0.0, 0.0, 0.1))
            pool = pools[name]
            if rng.random() < 0.5 and pool >= n:
                ints = rng.sample(range(pool), n)
            else:
                ints = [rng.randrange(pool) for _ in range(n)]
            columns['id'] = _with_nulls(
                rng,
                'Int64' if kind == 'int' and rate else kind,
                _key_values(kind, ints),
                rate,
            )
        time_kind = rng.choice((None, 'datetime', 'datetime', 'iso'))
        if time_kind is not None:
            columns['ts'] = _times(rng, n, time_kind)
            times[name] = 'ts'
        for other in names:
            if kinds[other] is not None and rng.random() < 0.6:
                fkind = kinds[other] if kinds[other] != 'int' else 'Int64'
                ints = [rng.randrange(pools[other] + 3) for _ in range(n)]
                columns[f'fk_{other}'] = _with_nulls(
                    rng, fkind, _key_values(kinds[other], ints), 0.1
                )
        columns['x'] = np.array([rng.random() for _ in range(n)])
        frames[name] = columns
    return {'names': names, 'kinds': kinds, 'frames': frames, 'times': times}


def build_graph(spec: dict[str, Any]) -> Any:
    r"""The engine's ``Graph`` of ``LocalTable``\ s for a ``random_store``
    spec, or ``None`` where the engine refuses it."""
    import warnings

    import pandas as pd
    from sdm_relational_engine.rfm import Edge, Graph, LocalTable

    tables = []
    edges = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            for name in spec['names']:
                frame = pd.DataFrame(spec['frames'][name])
                tables.append(
                    LocalTable(
                        frame,
                        name,
                        primary_key='id' if 'id' in frame.columns else None,
                        time_column=spec['times'].get(name),
                    )
                )
            for name in spec['names']:
                for column in spec['frames'][name]:
                    if column.startswith('fk_'):
                        edges.append(Edge(name, column, column[3:]))
            return Graph(tables, edges)
        except Exception:  # noqa: BLE001 - a refused graph is skipped
            return None


def graph_json(graph: Any) -> str:
    r"""What ``graph_store`` is told about a graph (``CONTRACT.md``)."""
    tables = []
    for table in graph.tables.values():
        tables.append(
            {
                'name': table.name,
                'primary_key': table._primary_key,
                'time_column': table._time_column,
                'end_time_column': table._end_time_column,
                'columns': [
                    {
                        'name': column.name,
                        'dtype': str(column.dtype),
                        'stype': str(column.stype),
                    }
                    for column in table.columns
                ],
            }
        )
    edges = [
        {'src_table': e.src_table, 'fkey': e.fkey, 'dst_table': e.dst_table}
        for e in graph.edges
    ]
    return json.dumps({'tables': tables, 'edges': edges})


def arrow_tables(graph: Any) -> dict[str, Any]:
    import pyarrow as pa

    return {
        name: pa.Table.from_pandas(table._data, preserve_index=False)
        for name, table in graph.tables.items()
    }


def _plain(value: Any) -> Any:
    r"""A key as JSON-comparable data: ints, floats and text by type."""
    if isinstance(value, float) and math.isnan(value):
        return {'$float': 'nan'}
    if hasattr(value, 'item'):
        return _plain(value.item())
    return value


def python_store(graph: Any) -> dict[str, Any]:
    r"""Today's ``LocalGraphStore`` of ``graph``, in the contract's shape."""
    from sdm_relational_engine.rfm.backend.local import LocalGraphStore

    try:
        store = LocalGraphStore(graph, verbose=False)
    except Exception as error:  # noqa: BLE001 - compared, not judged
        return {'error': error_of(error)}
    return {
        'report': {
            name: {
                'input_rows': r.input_rows,
                'output_rows': r.output_rows,
                'null_primary_key_rows': r.null_primary_key_rows,
                'duplicate_primary_key_rows': r.duplicate_primary_key_rows,
                'null_time_rows': r.null_time_rows,
            }
            for name, r in store.sanitization_report.tables.items()
        },
        'masks': {k: [bool(x) for x in v] for k, v in store.mask_dict.items()},
        'keys': {
            k: [_plain(x) for x in v.index.tolist()]
            for k, v in store.pkey_map_dict.items()
        },
        'node_ids': {
            k: [int(x) for x in v['arange'].tolist()]
            for k, v in store.pkey_map_dict.items()
        },
        'time': {
            k: [int(x) for x in v.tolist()] for k, v in store.time_dict.items()
        },
        'time_range': {
            k: [int(lo.value), int(hi.value)]
            for k, (lo, hi) in store.min_max_time_dict.items()
        },
        'colptr': {
            _rel(k): [int(x) for x in v.tolist()]
            for k, v in store.colptr_dict.items()
        },
        'row': {
            _rel(k): [int(x) for x in v.tolist()]
            for k, v in store.row_dict.items()
        },
    }


STORE_KEYS = (
    'report',
    'masks',
    'keys',
    'node_ids',
    'time',
    'time_range',
    'colptr',
    'row',
)


def rust_store(module: Any, graph: Any) -> dict[str, Any]:
    r"""The binding's ``graph_store`` of ``graph``, in the contract's shape."""
    return rust_store_of(module, arrow_tables(graph), graph_json(graph))


def rust_store_of(
    module: Any, tables: dict[str, Any], text: str
) -> dict[str, Any]:
    r"""The binding's ``graph_store`` of Arrow tables and a graph's JSON."""
    try:
        got = module.graph_store(tables, text)
    except Exception as error:  # noqa: BLE001 - compared, not judged
        return {'error': error_of(error)}
    assert isinstance(got, dict), f'graph_store gave {type(got).__name__}'
    assert set(got) == set(STORE_KEYS), (
        f'graph_store gave keys {sorted(got)}, not {sorted(STORE_KEYS)}'
    )
    out: dict[str, Any] = {}
    for key in STORE_KEYS:
        part = got[key]
        assert isinstance(part, dict), f'graph_store[{key!r}] is not a dict'
        if key == 'report':
            out[key] = {
                name: {field: int(v) for field, v in counts.items()}
                for name, counts in part.items()
            }
        elif key == 'keys':
            out[key] = {
                name: [_plain(x) for x in as_list(v)]
                for name, v in part.items()
            }
        elif key == 'masks':
            out[key] = {
                name: [bool(x) for x in as_list(v)] for name, v in part.items()
            }
        else:
            out[key] = {
                name: [int(x) for x in as_list(v)] for name, v in part.items()
            }
    return out


def ordered(value: Any) -> Any:
    r"""Dicts as lists of pairs, so their order is compared too."""
    if isinstance(value, dict):
        return [[k, ordered(v)] for k, v in value.items()]
    if isinstance(value, list):
        return [ordered(v) for v in value]
    return value


def store_sampler_args(store: dict[str, Any], graph: Any) -> dict[str, Any]:
    r"""``relationallib`` constructor arguments built from a store's arrays,
    as ``LocalSampler`` builds them.

    The edge types are the graph's, each followed by its reverse, as
    ``Sampler.edge_types`` lists them. They can't be read back from the
    store's keys: a reverse relation starts with ``__###REV###__``, so
    splitting ``src__rel__dst`` on ``__`` mangles it."""
    from sdm_relational_engine.rfm.inference.context import SampledSubgraph

    edge_types = []
    for edge in graph.edges:
        edge_type = (edge.src_table, edge.fkey, edge.dst_table)
        edge_types += [edge_type, SampledSubgraph.rev_edge_type(edge_type)]
    return {
        'node_types': list(graph.tables),
        'edge_types': [list(e) for e in edge_types],
        'colptr': store['colptr'],
        'row': store['row'],
        'node_time': store['time'],
    }


# --- DateOffset arithmetic -----------------------------------------------------

OFFSET_SEED = 20261013
NUM_PAIRS = 100_000


def offset_of(spec: dict[str, Any]) -> Any:
    import pandas as pd

    return pd.DateOffset(n=spec['n'], **spec['kwds'])


def random_offset(rng: random.Random) -> dict[str, Any]:
    r"""An offset with one to three units, any ``n``; mostly small, now and
    then large enough to leave the timestamp range."""
    kwds = {}
    for unit in rng.sample(OFFSET_UNITS, rng.choice((1, 1, 1, 2, 3))):
        scale = rng.choice((3, 12, 40, 40, 400, 5000, 200_000))
        kwds[unit] = rng.randint(-scale, scale)
    n = rng.choice((1, 1, 1, 2, 3, 4, 12, -1, -2, -3, 0, rng.randint(-60, 60)))
    return {'n': n, 'kwds': kwds}


_MONTH_ENDS = (
    '2024-01-31',
    '2024-02-29',
    '2023-02-28',
    '2024-03-31',
    '2024-04-30',
    '2000-02-29',
    '1900-02-28',
    '2100-02-28',
    '1970-01-31',
    '2023-12-31',
    '2024-02-28',
    '2023-03-29',
    '2262-03-31',
    '1677-10-31',
)


def random_timestamp(rng: random.Random) -> int:
    import pandas as pd

    kind = rng.random()
    if kind < 0.3:
        return rng.randint(NS_MIN, NS_MAX)
    if kind < 0.45:
        return NS_MIN + rng.randint(0, rng.choice((10**9, 10**15, 10**17)))
    if kind < 0.6:
        return NS_MAX - rng.randint(0, rng.choice((10**9, 10**15, 10**17)))
    if kind < 0.9:
        day = pd.Timestamp(rng.choice(_MONTH_ENDS)).value
        return day + rng.choice((0, 0, rng.randint(0, 86_400 * 10**9 - 1)))
    return rng.randint(-(10**18), 10**18)


def offset_pairs() -> list[dict[str, Any]]:
    r"""The P7.offsets jobs: an offset, a sign, timestamps, and how many
    times the offset is applied in a row."""
    rng = random.Random(OFFSET_SEED)
    jobs = []
    total = 0
    while total < NUM_PAIRS:
        size = rng.randint(1, 30)
        jobs.append(
            {
                'offset': random_offset(rng),
                'sign': rng.choice((1, -1)),
                'repeat': rng.choice((1, 1, 1, 1, 2, 3, 5)),
                'timestamps': [random_timestamp(rng) for _ in range(size)],
            }
        )
        total += size
    return jobs


def python_apply(ns: int, spec: dict[str, Any], sign: int) -> dict[str, Any]:
    r"""``pd.Timestamp(ns) + offset`` (or minus), as an outcome."""
    import pandas as pd

    offset = offset_of(spec)
    try:
        ts = pd.Timestamp(ns, unit='ns')
        moved = ts + offset if sign > 0 else ts - offset
        return {'ns': int(moved.value)}
    except Exception as error:  # noqa: BLE001 - every refusal is an outcome
        return {'error': error_of(error)}


def python_pool(specs: list[dict[str, Any]], earliest: bool) -> dict[str, Any]:
    from sdm_relational_engine.rfm.base.sampler import _pooled_offset

    pooled = _pooled_offset([offset_of(s) for s in specs], earliest=earliest)
    return {
        'n': int(pooled.n),
        'kwds': {k: int(v) for k, v in sorted(pooled.kwds.items())},
    }


def python_to_ns(spec: dict[str, Any]) -> dict[str, Any]:
    from sdm_relational_engine.rfm.rfm import _date_offset_to_ns

    try:
        return {'ns': int(_date_offset_to_ns(offset_of(spec)))}
    except Exception as error:  # noqa: BLE001 - every refusal is an outcome
        return {'error': error_of(error)}


# --- Text ----------------------------------------------------------------------

TEXT_SEED = 20261014
NUM_TEXTS = 20_000


def _unicode_classes() -> dict[str, list[str]]:
    r"""Every code point Python calls whitespace or that ``lower`` changes,
    with the punctuation and markup the normalizer treats specially."""
    spaces, cased = [], []
    for code in range(0x110000):
        if 0xD800 <= code <= 0xDFFF:
            continue
        char = chr(code)
        if char.isspace():
            spaces.append(char)
        if char.lower() != char or char.upper() != char:
            cased.append(char)
    return {
        'space': spaces,
        'cased': cased,
        'punct': list('\'".,()!?;:'),
        'markup': ['<br>', '<br/>', '<br />', '<BR>', '<br  />', '<bR\t/>'],
    }


def texts() -> list[str | None]:
    r"""Generated text: words of cased letters (every one of them appears),
    every whitespace character, punctuation, ``<br>`` tags, long runs, and
    nulls."""
    rng = random.Random(TEXT_SEED)
    classes = _unicode_classes()
    out: list[str | None] = [None, '', ' ', '\x1c', '\x1c\x1d\x1e\x1f', '\x85']
    out += [''.join(classes['space'])]
    cased = classes['cased']
    for start in range(0, len(cased), 40):
        out.append(' '.join(cased[start : start + 40]))
    while len(out) < NUM_TEXTS:
        if rng.random() < 0.02:
            out.append(None)
            continue
        parts = []
        for _ in range(rng.randint(0, rng.choice((5, 20, 120)))):
            kind = rng.random()
            if kind < 0.45:
                parts.append(
                    ''.join(rng.choice(cased) for _ in range(rng.randint(1, 8)))
                )
            elif kind < 0.6:
                parts.append(
                    ''.join(
                        rng.choice('abcdefgh XYZ0123')
                        for _ in range(rng.randint(1, 8))
                    )
                )
            elif kind < 0.85:
                parts.append(
                    ''.join(
                        rng.choice(classes['space'])
                        for _ in range(rng.randint(1, 3))
                    )
                )
            elif kind < 0.95:
                parts.append(rng.choice(classes['punct']))
            else:
                parts.append(rng.choice(classes['markup']))
        out.append(''.join(parts))
    return out


def python_normalize(
    values: list[str | None], max_words: int | None
) -> list[Any]:
    import pandas as pd
    from sdm_relational_engine.rfm.base.sampler import _normalize_text

    ser = pd.Series(values, dtype=object)
    return [list(words) for words in _normalize_text(ser, max_words).tolist()]


# --- Fixtures ------------------------------------------------------------------


def fixture_files() -> list[Path]:
    return sorted(FIXTURES.glob('*.json'))


def fixture_cases(prefix: str) -> list[dict[str, Any]]:
    r"""Every fixture case whose operation starts with ``prefix``."""
    out = []
    for path in fixture_files():
        data = json.loads(path.read_text(encoding='utf-8'))
        for case in data['cases']:
            if case['operation'].startswith(prefix):
                out.append(
                    {**case, '_file': path.name, '_source': data['source']}
                )
    return out


def python_fixture(case: dict[str, Any]) -> dict[str, Any]:
    r"""Today's Python answer to a fixture case, as ``{"output"}`` or
    ``{"error"}``."""
    from _fixture_ops import FixtureError, run

    try:
        return {
            'output': json.loads(
                json.dumps(run(case['operation'], case['input']))
            )
        }
    except FixtureError as error:
        return {'error': {'code': error.code, 'message': error.message}}


def fixture_problem(case: dict[str, Any], got: Any) -> str | None:
    r"""How an ``{"output"}``/``{"error"}`` answer misses the case's expected
    output (by value and JSON type) or error (code, and message where the
    case has one)."""
    from _fixture_ops import same

    if not isinstance(got, dict) or len(got) != 1:
        return f'gave {short(got)}, not one of output or error'
    if 'expected' in case:
        if 'output' not in got:
            return f'gave {short(got)}, expected {short(case["expected"])}'
        if not same(got['output'], case['expected']):
            return f'gave {short(got["output"])}, expected {short(case["expected"])}'
        return None
    want = case['expected_error']
    error = got.get('error')
    if not isinstance(error, dict) or set(error) != {'code', 'message'}:
        return f'gave {short(got)}, expected the error {short(want)}'
    if error['code'] != want['code']:
        return f'code {error["code"]!r}, expected {want["code"]!r}'
    if 'message' in want and error['message'] != want['message']:
        return f'message {error["message"]!r}, expected {want["message"]!r}'
    return None
