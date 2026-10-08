# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P7.offsets: ``DateOffset`` arithmetic in the Rust core is pandas'.

100,000 timestamp and offset pairs (seed 20261013): timestamps anywhere in
pandas' range, near its 1677 and 2262 bounds, on month ends and leap days;
offsets of one to three units (years to seconds), negative ones, any ``n``,
and now and then large enough to leave the range. Each pair adds or
subtracts the offset, and a fifth of them again and again, up to five times,
so ``(ts - o) - o`` is checked beside ``ts - 2*o``. Every result, or error
by class and text, is pandas'.

Each pair's offset also goes through ``date_offset_to_seconds`` (both
bounds) and ``date_offset_to_ns``, and groups of them through
``_pooled_offset`` (both directions).
"""

from __future__ import annotations

import collections
import json
import random
from typing import Any

from _sampling import (
    NUM_PAIRS,
    OFFSET_SEED,
    loads,
    native,
    offset_of,
    offset_pairs,
    python_apply,
    python_pool,
    python_to_ns,
    random_offset,
    report,
    short,
)


def test_the_pairs_reach_every_path() -> None:
    jobs = offset_pairs()
    assert sum(len(j['timestamps']) for j in jobs) >= NUM_PAIRS
    kinds: collections.Counter[str] = collections.Counter()
    for job in jobs:
        for ts in job['timestamps']:
            outcome = python_apply(ts, job['offset'], job['sign'])
            kinds[outcome['error']['code'] if 'error' in outcome else 'ok'] += 1
    assert kinds['ok'] >= 60_000
    for code in ('OutOfBoundsDatetime', 'ValueError', 'OverflowError'):
        assert kinds[code] >= 200, f'only {kinds[code]} {code}s: {kinds}'
    assert sum(j['repeat'] > 1 for j in jobs) >= 1_000
    assert sum(j['offset']['n'] not in (0, 1) for j in jobs) >= 2_000
    assert sum('months' in j['offset']['kwds'] for j in jobs) >= 1_000


def test_pairs_match_pandas() -> None:
    module = native('date_offset_apply')
    failures = []
    checked = 0
    for index, job in enumerate(offset_pairs()):
        offset_json = json.dumps(job['offset'])
        current = list(job['timestamps'])
        alive = list(range(len(current)))
        for step in range(job['repeat']):
            got = loads(
                module.date_offset_apply(
                    json.dumps([current[i] for i in alive]),
                    offset_json,
                    job['sign'],
                ),
                'date_offset_apply',
            )
            if not isinstance(got, list) or len(got) != len(alive):
                failures.append(f'job {index}: gave {short(got)}')
                break
            still = []
            for i, outcome in zip(alive, got, strict=True):
                want = python_apply(current[i], job['offset'], job['sign'])
                checked += 1
                if outcome != want:
                    failures.append(
                        f'job {index} step {step}: {current[i]} '
                        f'{"+" if job["sign"] > 0 else "-"} {short(job["offset"])} '
                        f'gives {short(outcome)}, pandas {short(want)}'
                    )
                elif 'ns' in want:
                    current[i] = want['ns']
                    still.append(i)
            alive = still
            if not alive:
                break
    assert checked >= NUM_PAIRS
    report(failures, checked, 'offset applications')


def test_seconds_ns_and_pooling_match_pandas() -> None:
    from sdm_relational_engine.rfm.base.utils import date_offset_to_seconds

    module = native(
        'date_offset_to_seconds', 'date_offset_to_ns', 'date_offset_pool'
    )
    jobs = offset_pairs()
    failures = []
    for job in jobs:
        spec = job['offset']
        text = json.dumps(spec)
        for bound in ('earliest', 'latest'):
            want = date_offset_to_seconds(offset_of(spec), bound)
            got = module.date_offset_to_seconds(text, bound)
            if got != want or type(got) is not int:
                failures.append(
                    f'{short(spec)} {bound}: seconds {got!r}, today {want}'
                )
        got_ns = loads(module.date_offset_to_ns(text), 'date_offset_to_ns')
        if got_ns != python_to_ns(spec):
            failures.append(
                f'{short(spec)}: ns {short(got_ns)}, today {short(python_to_ns(spec))}'
            )
    rng = random.Random(OFFSET_SEED + 1)
    for _ in range(5_000):
        group: list[dict[str, Any]] = [
            random_offset(rng) for _ in range(rng.randint(1, 4))
        ]
        if rng.random() < 0.3:
            group.append(dict(group[0]))
        for earliest in (True, False):
            want = python_pool(group, earliest)
            got = loads(
                module.date_offset_pool(json.dumps(group), earliest),
                'date_offset_pool',
            )
            if got != want:
                failures.append(
                    f'pool {short(group)} earliest={earliest}: {short(got)}, today {short(want)}'
                )
    report(failures, len(jobs) * 3 + 10_000, 'conversions and pools')
