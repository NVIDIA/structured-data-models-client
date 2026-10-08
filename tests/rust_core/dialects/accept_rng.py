# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P6.rng: numpy's legacy ``RandomState``, as ``base/utils.random_state``
seeds it, drawn with ``randint``, from the Rust core.

10,000 generated seeds (seed 20261004), a third of them 2**32 or above,
where ``random_state`` seeds from 32-bit words instead, each:

- through SQLite's ``seeded_hash``, whose four coefficients are
  ``randint(0, 2**31 - 1, size=4)``, the bound the dialects use;
- through ``utils.random_state.randint`` with one of ``BOUNDS`` and a size
  from 1 to 12, compared with ``randint(...).tolist()``.
"""

from __future__ import annotations

import random

from _dialects import (
    BACKENDS,
    encode,
    mismatch,
    python_call,
    python_randint,
    rust_call,
)

SEED = 20261004
NUM_SEEDS = 10_000

#: ``(low, high)`` of the draws: the dialects' bound first, then spans on
#: both sides of 2**32, where numpy draws 32 or 64 bits at a time.
BOUNDS = (
    (0, 2**31 - 1),
    (0, 1),
    (0, 2),
    (0, 7),
    (0, 10),
    (0, 100),
    (-50, 50),
    (0, 1000),
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

EDGES = (
    0,
    1,
    2,
    42,
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
    r"""The seeds, edges first: small ones, any below 2**32, then 2**32 up to
    2**64, and up to 2**160."""
    rng = random.Random(SEED)
    out = list(EDGES)
    while len(out) < NUM_SEEDS:
        kind = len(out) % 4
        if kind == 0:
            out.append(rng.randrange(0, 1000))
        elif kind == 1:
            out.append(rng.randrange(0, 2**32))
        elif kind == 2:
            out.append(rng.randrange(2**32, 2**64))
        else:
            out.append(rng.randrange(2**64, 2**160))
    return out


def test_the_seeds_reach_both_seedings() -> None:
    drawn = seeds()
    assert len(drawn) == NUM_SEEDS
    assert sum(seed >= 2**32 for seed in drawn) >= NUM_SEEDS // 3
    assert sum(seed < 2**32 for seed in drawn) >= NUM_SEEDS // 3


def test_seeded_hash_matches_numpy() -> None:
    failures = []
    for seed in seeds():
        args = {'random_seed': encode(seed), 'ref': 'rowid'}
        want = python_call('sqlite', 'seeded_hash', args)
        try:
            got = rust_call('sqlite', 'seeded_hash', args)
        except Exception as error:  # noqa: BLE001 - reported per seed
            failures.append(f'seed {seed}: raised {error!r}')
            continue
        problem = mismatch(want, got)
        if problem:
            failures.append(f'seed {seed}: {problem}')
    assert not failures, (
        f'{len(failures)} of {NUM_SEEDS} seeds differ:\n'
        + '\n'.join(failures[:30])
    )


def test_randint_matches_numpy() -> None:
    failures = []
    for i, seed in enumerate(seeds()):
        low, high = BOUNDS[i % len(BOUNDS)]
        args = {
            'random_seed': encode(seed),
            'low': encode(low),
            'high': encode(high),
            'size': 1 + i % 12,
        }
        want = python_randint(args)
        backend = BACKENDS[i % len(BACKENDS)]
        try:
            got = rust_call(backend, 'utils.random_state.randint', args)
        except Exception as error:  # noqa: BLE001 - reported per seed
            failures.append(f'seed {seed} {args}: raised {error!r}')
            continue
        problem = mismatch(want, got)
        if problem:
            failures.append(f'seed {seed} {args}: {problem}')
    assert not failures, (
        f'{len(failures)} of {NUM_SEEDS} draws differ:\n'
        + '\n'.join(failures[:30])
    )
