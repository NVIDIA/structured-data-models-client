# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P7.rng: numpy's legacy ``RandomState`` and C++'s ``std::mt19937``, from
the Rust core.

10,000 seeds (seed 20261010): 0, 2**31 - 1, 2**32 - 1, 2**32 and other
edges first, then a quarter each below 2**31, below 2**32, below 2**64 and
below 2**160. For each seed, a random sequence of draws (32- and 64-bit
words, interval draws, ``randint``, ``shuffle``, ``permutation``,
``choice``) from one stream:

- through ``random_state(seed)``, against ``base/utils.random_state(seed)``;
- through ``RandomState(seed)`` for seeds below 2**32, and through
  ``RandomState(words)`` for the rest, against numpy seeded the same way;
- through ``RandomState(words)`` with word lists of up to 1,000 words,
  longer than the generator's state, against ``init_by_array``.

And for each seed, raw draws and the neighbor sampler's picks through
``Mt19937``, against ``std::mt19937`` compiled from
``mt19937_reference.cpp``.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from _sampling import (
    NUM_SEEDS,
    RNG_SEED,
    cpp_draws,
    mt_jobs,
    native,
    numpy_draws,
    report,
    rng_ops,
    rust_draws,
    rust_mt_draws,
    seed_words,
    seeds,
    short,
)


def test_the_seeds_cover_the_edges() -> None:
    drawn = seeds()
    assert len(drawn) == NUM_SEEDS
    for edge in (0, 2**31 - 1, 2**32 - 1, 2**32, 2**64):
        assert edge in drawn
    assert sum(s >= 2**32 for s in drawn) >= NUM_SEEDS // 3


def _compare(make_rust, make_numpy, label: str, seed_list) -> None:  # noqa: ANN001
    rng = random.Random(RNG_SEED + 7)
    failures = []
    for seed in seed_list:
        ops = rng_ops(rng)
        want = numpy_draws(make_numpy(seed), ops)
        try:
            got = rust_draws(make_rust(seed), ops)
        except Exception as error:  # noqa: BLE001 - reported per seed
            failures.append(
                f'{label} {seed} {short(ops, 120)}: raised {error!r}'
            )
            continue
        if got != want:
            at = next(
                i
                for i, (a, b) in enumerate(zip(got, want, strict=True))
                if a != b
            )
            failures.append(
                f'{label} {seed}: draw {at} {short(ops[at], 80)} gives '
                f'{short(got[at], 120)}, numpy {short(want[at], 120)}'
            )
    report(failures, len(seed_list), f'{label} streams')


def test_random_state_matches_utils_random_state() -> None:
    from sdm_relational_engine.rfm.base.utils import random_state

    module = native('random_state')
    _compare(module.random_state, random_state, 'random_state', seeds())


def test_random_state_by_number_and_by_words() -> None:
    module = native('RandomState')

    def rust(seed: int):  # noqa: ANN202
        return module.RandomState(seed if seed < 2**32 else seed_words(seed))

    def numpy(seed: int):  # noqa: ANN202
        if seed < 2**32:
            return np.random.RandomState(seed)
        return np.random.RandomState(
            np.array(seed_words(seed), dtype=np.uint32)
        )

    _compare(rust, numpy, 'RandomState', seeds())


def test_random_state_by_long_word_lists() -> None:
    module = native('RandomState')
    rng = random.Random(RNG_SEED + 3)
    lists = [
        [
            rng.randrange(2**32)
            for _ in range(rng.choice((1, 2, 623, 624, 625, 1000)))
        ]
        for _ in range(300)
    ]
    lists += [[0], [2**32 - 1], [0] * 700, [1, 0, 0]]
    index = {i: words for i, words in enumerate(lists)}

    _compare(
        lambda i: module.RandomState(index[i]),
        lambda i: np.random.RandomState(np.array(index[i], dtype=np.uint32)),
        'RandomState(words)',
        list(index),
    )


@pytest.mark.parametrize(
    'seed', [-1, 2**32, 2**64, [], [-1], [2**32], 'seed', 1.5]
)
def test_random_state_refuses_what_numpy_refuses(seed: object) -> None:
    module = native('RandomState')
    with pytest.raises((ValueError, TypeError)):
        module.RandomState(seed)


def test_mt19937_matches_the_cpp_standard_library() -> None:
    module = native('Mt19937')
    jobs = mt_jobs(seeds())
    want = cpp_draws(jobs)
    failures = []
    for job, expected in zip(jobs, want, strict=True):
        try:
            got = rust_mt_draws(module, job)
        except Exception as error:  # noqa: BLE001 - reported per seed
            failures.append(f'{short(job, 160)}: raised {error!r}')
            continue
        if got != expected:
            at = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(got, expected, strict=False))
                    if a != b
                ),
                min(len(got), len(expected)),
            )
            failures.append(
                f'{short(job, 160)}: number {at} is '
                f'{got[at] if at < len(got) else "missing"}, C++ gives '
                f'{expected[at] if at < len(expected) else "nothing"}'
            )
    report(failures, len(jobs), 'std::mt19937 jobs')
