# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A ranking with a NaN score is ordered as today's Python orders it, at
every size.

``client._ranked`` sorts with ``list.sort(key=..., reverse=True)``, and a NaN
makes every comparison with it false, so the order it leaves is whatever
CPython's timsort does with those comparisons. Under 64 candidates that is
one run extended by binary insertion; from 64 on, runs are merged, galloping
when one run keeps winning. ``rust/crates/request/src/pysort.rs`` is that
sort, and here it is held to the interpreter's on rankings large enough to
merge, through the binding's fixture runner against the Python one.

The order is CPython 3.12's (3.10 and 3.11 agree); 3.13 rewrote
``count_run`` and can leave a NaN elsewhere, so this holds only where the
interpreter running it is one of those.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip('sdm_relational_engine')
native = pytest.importorskip('sdm_core._native')

PARITY = Path(__file__).resolve().parents[1] / 'parity'
if str(PARITY) not in sys.path:
    sys.path.insert(0, str(PARITY))

from _fixture_ops import FixtureError, run  # noqa: E402

SEED = 20261004
CASES = 400


def _rankings() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    out = []
    for _ in range(CASES):
        k = rng.randint(60, 400)
        nan = rng.choice([0.02, 0.1, 0.3, 0.6])
        pool = rng.choice([[0.1, 0.5, 0.9], [0.0, 1.0], None])
        candidates = rng.sample(range(10_000), k)
        rows = rng.randint(1, 3)
        scores = [
            [
                {'$float': 'nan'}
                if rng.random() < nan
                else rng.choice(pool)
                if pool
                else round(rng.random(), 3)
                for _ in range(rows)
            ]
            for _ in candidates
        ]
        out.append(
            {
                'candidates': candidates,
                'scores': scores,
                'top': rng.choice([None, 1, 5, k // 2, k + 1]),
            }
        )
    return out


@pytest.mark.skipif(
    sys.version_info >= (3, 13),
    reason='CPython 3.13 rewrote count_run, so a NaN is left elsewhere',
)
def test_a_nan_score_is_ordered_as_python_orders_it() -> None:
    problems = []
    for i, given in enumerate(_rankings()):
        try:
            want: Any = {
                'output': json.loads(json.dumps(run('client.ranked', given)))
            }
        except FixtureError as error:
            want = {'error': error.described()['message']}
        got = json.loads(
            native.request_fixture_op('client.ranked', json.dumps(given))
        )
        if 'error' in want:
            if (got.get('error') or {}).get('message') != want['error']:
                problems.append(f'ranking {i}: {got} for {want}')
        elif json.dumps(got.get('output'), sort_keys=True) != json.dumps(
            want['output'], sort_keys=True
        ):
            problems.append(
                f'ranking {i} of {len(given["candidates"])} candidates'
            )
    assert not problems, '\n'.join(problems[:20])


#: Sixty-four candidates scored 0.25, 0.5, 0.75 or NaN (``None`` here), and
#: the order ``list.sort(key=..., reverse=True)`` leaves them in on CPython
#: 3.10 to 3.12. CPython 3.13 and 3.14 leave another.
PINNED_SCORES = [
    0.5, None, 0.5, 0.5, 0.5, 0.25, 0.5, 0.25, 0.5, 0.75, 0.75, 0.5, None,
    None, 0.5, 0.25, 0.5, 0.25, 0.5, 0.75, 0.75, 0.25, 0.5, 0.75, 0.75, 0.5,
    0.5, 0.25, 0.75, 0.25, 0.5, None, 0.75, 0.5, 0.75, 0.25, 0.75, 0.75, 0.5,
    0.75, 0.5, 0.5, 0.25, None, 0.5, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75,
    0.25, 0.75, 0.5, 0.5, 0.75, 0.5, None, 0.5, 0.5, 0.25, None, 0.75,
]  # fmt: skip
PINNED_ORDER = [
    1, 9, 10, 12, 13, 19, 20, 23, 24, 28, 0, 2, 3, 4, 6, 8, 11, 14, 16, 18,
    22, 25, 26, 30, 5, 7, 15, 17, 21, 27, 29, 31, 32, 34, 36, 37, 39, 43, 48,
    49, 50, 51, 53, 56, 33, 38, 40, 41, 44, 54, 55, 57, 45, 52, 58, 59, 60,
    35, 42, 46, 47, 61, 62, 63,
]  # fmt: skip


def test_a_nan_score_is_ordered_as_3_12_orders_it_on_every_python() -> None:
    r"""Phase 9's decision: the core keeps CPython 3.12's order everywhere,
    so a ranking with a NaN score no longer depends on the interpreter (the
    client sorted with the running Python's ``list.sort`` before)."""
    given = {
        'candidates': list(range(len(PINNED_SCORES))),
        'scores': [
            [{'$float': 'nan'} if score is None else score]
            for score in PINNED_SCORES
        ],
        'top': None,
    }
    got = json.loads(
        native.request_fixture_op('client.ranked', json.dumps(given))
    )
    assert got['output'][0]['prediction'] == PINNED_ORDER
