# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P75.debug (task 6b): debug and release builds agree, and neither panics.

A debug build of the binding (``maturin build`` without ``--release``) and
the release build answer the same jobs (seed 20261016) in separate
processes (``_debug_run.py``):

- the two inputs the Phase 7 audit found panicking in debug only: a
  temporal neighborhood whose ``colptr`` starts at ``i64::MIN``
  (``neighbor.rs``, ``end - count``), and an offset of ``n = 2**62`` with
  weeks of ``-2**63`` (``date_offset.rs``, ``RelativeDelta::fix``);
- 3,000 offsets with units and ``n`` at the edges of 32 and 64 bits, through
  ``sampling.date_offset_apply`` and ``sampling.date_offset_to_seconds``
  (``sampling_op``), and ``date_offset_apply``;
- 600 neighbor samplers with pointers, rows, times, counts, ranges and seed
  times at the edges of 64 bits, through ``sampling.neighbor_sample``.

Each job's answers are the same in both builds and none is a panic. Where
today's Python answers a ``sampling_op`` job (offsets; not the neighbor
samplers, where ``relationallib`` reads out of bounds), both builds give its
answer: so ``sampling.date_offset_to_seconds`` writes a result past int64 as
the exact integer.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from _driver import HERE, REPO, native
from _sampling import (
    NS_MAX,
    NS_MIN,
    OFFSET_UNITS,
    python_fixture,
    random_graph,
    random_timestamp,
)

SEED = 20261016
NUM_OFFSETS = 3_000
NUM_GRAPHS = 600
I64_MIN, I64_MAX = -(2**63), 2**63 - 1
BINDING = REPO / 'rust' / 'bindings' / 'python'

EDGES = (
    0,
    1,
    -1,
    2,
    2**31 - 1,
    -(2**31),
    2**31,
    2**32,
    2**53 + 1,
    2**62,
    -(2**62),
    I64_MAX,
    I64_MIN,
    I64_MAX - 1,
    I64_MIN + 1,
)

#: The audit's neighbor sampler: B__to__A's pointers start at i64::MIN.
#: relationallib reads out of bounds on it, so only the builds are compared.
AUDIT_NEIGHBORS = {
    'sampler': {
        'node_types': ['A', 'B'],
        'edge_types': [['A', 'to', 'B'], ['B', 'to', 'A']],
        'colptr': {
            'A__to__B': {'$ndarray': {'dtype': 'int64', 'values': [0, 2, 4]}},
            'B__to__A': {
                '$ndarray': {'dtype': 'int64', 'values': [I64_MIN, 0, 1, 2]}
            },
        },
        'row': {
            'A__to__B': {
                '$ndarray': {'dtype': 'int64', 'values': [0, 2, 0, 2]}
            },
            'B__to__A': {'$ndarray': {'dtype': 'int64', 'values': [0, 1, 1]}},
        },
        'node_time': {
            'A': {'$ndarray': {'dtype': 'int64', 'values': [1, 0, 3]}},
            'B': {'$ndarray': {'dtype': 'int64', 'values': [2, 1]}},
        },
    },
    'calls': [
        {
            'num_neighbors': {
                'A__to__B': {'$ndarray': {'dtype': 'int64', 'values': [1, 1]}},
                'B__to__A': {'$ndarray': {'dtype': 'int64', 'values': [1, 1]}},
            },
            'sampling_time_offsets': {},
            'entity_table': 'B',
            'seed_node': {'$ndarray': {'dtype': 'int64', 'values': [0, 1]}},
            'seed_time': {'$ndarray': {'dtype': 'int64', 'values': [1, 1]}},
        }
    ],
}

#: The audit's offset: RelativeDelta::fix carries past i128 in debug.
AUDIT_OFFSET = {
    'timestamps': [1667260800000000000],
    'offset': {
        'n': 2**62,
        'kwds': {'days': 86400, 'weeks': I64_MIN, 'seconds': -12},
    },
    'sign': 1,
}

#: Offsets whose seconds today's Python gives past int64.
PAST_INT64 = (
    {'n': 1, 'kwds': {'years': 10**12}},
    {'n': 1, 'kwds': {'days': 2**62}},
    {'n': 1, 'kwds': {'weeks': 2**62}},
    {'n': 1, 'kwds': {'months': -(2**62)}},
    {'n': 3, 'kwds': {'years': 2**40, 'days': -5}},
)


def _edge(rng: random.Random) -> int:
    r"""An integer at an edge of 32 or 64 bits, inside int64: what lies
    outside is refused when read, not overflowed in arithmetic."""
    value = rng.choice(EDGES) + rng.choice((0, 0, 0, 1, -1))
    return min(max(value, I64_MIN), I64_MAX)


def _offset(rng: random.Random) -> dict[str, Any]:
    kwds = {}
    for unit in rng.sample(OFFSET_UNITS, rng.choice((1, 2, 3))):
        kwds[unit] = (
            _edge(rng) if rng.random() < 0.7 else rng.randint(-5000, 5000)
        )
    n = _edge(rng) if rng.random() < 0.5 else rng.randint(-60, 60)
    return {'n': n, 'kwds': kwds}


def _ndarray(values: list[int]) -> dict[str, Any]:
    return {'$ndarray': {'dtype': 'int64', 'values': values}}


def _hostile(rng: random.Random, values: list[int]) -> list[int]:
    return [_edge(rng) if rng.random() < 0.3 else v for v in values]


def _neighbors(rng: random.Random) -> dict[str, Any]:
    graph = random_graph(rng)
    sampler = {
        'node_types': graph['node_types'],
        'edge_types': graph['edge_types'],
        'colptr': {
            k: _ndarray(_hostile(rng, v)) for k, v in graph['colptr'].items()
        },
        'row': {k: _ndarray(_hostile(rng, v)) for k, v in graph['row'].items()},
        'node_time': {
            k: _ndarray(_hostile(rng, v)) for k, v in graph['node_time'].items()
        },
        'seed': graph['seed'],
    }
    calls = []
    for call in graph['calls']:
        offsets = {
            k: [
                [None if s is None else _hostile(rng, [s])[0], _edge(rng)]
                if rng.random() < 0.3
                else [s, e]
                for s, e in ranges
            ]
            for k, ranges in call['offsets'].items()
        }
        made = {
            'num_neighbors': {
                k: _ndarray(_hostile(rng, v))
                for k, v in call['num_neighbors'].items()
            },
            'sampling_time_offsets': offsets,
            'entity_table': call['entity'],
            'seed_node': _ndarray(call['seed_node']),
            'seed_time': _ndarray(_hostile(rng, call['seed_time'])),
        }
        if call['reseed'] is not None:
            made['reseed'] = call['reseed']
        calls.append(made)
    return {'sampler': sampler, 'calls': calls}


def _op(operation: str, given: dict[str, Any]) -> dict[str, Any]:
    return {
        'call': 'sampling_op',
        'args': [operation, json.dumps(given)],
        'case': {'operation': operation, 'input': given},
    }


def jobs() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    audit_neighbors = _op('sampling.neighbor_sample', AUDIT_NEIGHBORS)
    audit_neighbors['crashes_python'] = True
    out = [audit_neighbors, _op('sampling.date_offset_apply', AUDIT_OFFSET)]
    for offset in PAST_INT64:
        for bound in ('earliest', 'latest'):
            out.append(
                _op(
                    'sampling.date_offset_to_seconds',
                    {'offset': offset, 'bound': bound},
                )
            )
    for _ in range(NUM_OFFSETS):
        offset = _offset(rng)
        stamps = [
            rng.choice((NS_MIN, NS_MAX, random_timestamp(rng)))
            for _ in range(rng.randint(1, 5))
        ]
        sign = rng.choice((1, -1))
        out.append(
            _op(
                'sampling.date_offset_apply',
                {'timestamps': stamps, 'offset': offset, 'sign': sign},
            )
        )
        out.append(
            {
                'call': 'date_offset_apply',
                'args': [json.dumps(stamps), json.dumps(offset), sign],
            }
        )
        bound = rng.choice(('earliest', 'latest'))
        out.append(
            _op(
                'sampling.date_offset_to_seconds',
                {'offset': offset, 'bound': bound},
            )
        )
        out.append(
            {
                'call': 'date_offset_to_seconds',
                'args': [json.dumps(offset), bound],
            }
        )
    for _ in range(NUM_GRAPHS):
        job = _op('sampling.neighbor_sample', _neighbors(rng))
        job['crashes_python'] = True
        out.append(job)
    return out


def _debug_build(root: Path) -> Path:
    maturin = shutil.which('maturin', path=os.path.dirname(sys.executable))
    maturin = maturin or shutil.which('maturin')
    assert maturin, 'maturin is not installed in this environment'
    wheels = root / 'wheel'
    proc = subprocess.run(
        [maturin, 'build', '--interpreter', sys.executable, '--out', wheels],
        cwd=BINDING,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert proc.returncode == 0, (
        f'the debug build failed:\n{proc.stderr[-6000:]}'
    )
    built = sorted(wheels.glob('*.whl'))
    assert len(built) == 1, f'expected one wheel, built {built}'
    site = root / 'site'
    with zipfile.ZipFile(built[0]) as wheel:
        wheel.extractall(site)
    return site


def _answers(given: list[dict[str, Any]], root: Path, path: Path | None):
    jobs_file = root / 'jobs.json'
    jobs_file.write_text(
        json.dumps([{'call': j['call'], 'args': j['args']} for j in given]),
        'utf-8',
    )
    out = root / f'answers-{"debug" if path else "release"}.json'
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    if path is not None:
        env['PYTHONPATH'] = str(path)
    env['RUST_BACKTRACE'] = '0'
    proc = subprocess.run(
        [sys.executable, str(HERE / '_debug_run.py'), jobs_file, out],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    assert proc.returncode == 0, (
        f'the {"debug" if path else "release"} build crashed '
        f'({proc.returncode}):\n{proc.stderr[-6000:]}'
    )
    found = json.loads(out.read_text('utf-8'))
    if path is not None:
        assert str(path) in found['module'], (
            f'the debug run imported {found["module"]}, not the debug build'
        )
    return found['answers']


def _panicked(answer: dict[str, Any]) -> bool:
    raised = answer.get('raised')
    return bool(raised) and (
        raised['code'] == 'PanicException' or 'panick' in raised['message']
    )


def _outcome(answer: dict[str, Any]) -> Any:
    if 'text' not in answer:
        return answer
    try:
        return json.loads(answer['text'])
    except ValueError:
        return answer


def test_debug_and_release_agree() -> None:
    native('sampling_op', 'date_offset_apply', 'date_offset_to_seconds')
    given = jobs()
    with tempfile.TemporaryDirectory(prefix='sdm-p75-debug-') as tmp:
        root = Path(tmp)
        debug = _answers(given, root, _debug_build(root))
        release = _answers(given, root, None)
    problems = []
    checked_python = 0
    for i, (job, a, b) in enumerate(zip(given, debug, release, strict=True)):
        what = f'job {i} ({job["call"]} {job["args"][0][:40]})'
        if _panicked(a) or _panicked(b):
            which = 'debug' if _panicked(a) else 'release'
            problems.append(f'{what}: the {which} build panicked: {a or b}')
            continue
        if json.dumps(_outcome(a)) != json.dumps(_outcome(b)):
            problems.append(
                f'{what}: debug {json.dumps(a)[:200]}, release '
                f'{json.dumps(b)[:200]}'
            )
            continue
        if 'case' in job and not job.get('crashes_python'):
            want = python_fixture(job['case'])
            if json.dumps(_outcome(b)) != json.dumps(want):
                problems.append(
                    f'{what}: {json.dumps(_outcome(b))[:200]} where today '
                    f'gives {json.dumps(want)[:200]}'
                )
            else:
                checked_python += 1
    assert not problems, f'{len(problems)} of {len(given)} jobs:\n' + '\n'.join(
        problems[:25]
    )
    assert checked_python >= 6_000, f'only {checked_python} checked'
