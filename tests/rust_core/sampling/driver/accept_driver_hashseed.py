# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P75.hashseed: the boundary outputs don't follow ``PYTHONHASHSEED``.

- ``capture_sampling.py --out`` writes the same files under
  ``PYTHONHASHSEED`` 1 and 4242, and they are the committed recordings.
- The core's runs, driven from the recordings (``_driver_run.py``), ask the
  same queries and give the same outputs under both.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from _driver import RECORDINGS
from accept_driver_capture import compare_trees, run_capture, tree
from accept_driver_replay import run_core

SEEDS = ('1', '4242')


def test_the_capture_ignores_the_hash_seed() -> None:
    with tempfile.TemporaryDirectory(prefix='sdm-p75-hashseed-') as tmp:
        trees = {}
        for seed in SEEDS:
            out = Path(tmp) / seed
            run_capture(out, seed)
            trees[seed] = tree(out)
        compare_trees(trees[SEEDS[0]], trees[SEEDS[1]], 'the two hash seeds')
        compare_trees(tree(RECORDINGS), trees[SEEDS[0]], 'the committed files')


def _stable(found: dict) -> str:
    return json.dumps(
        {
            name: [
                {k: v for k, v in entry.items() if k != 'seconds'}
                for entry in entries
            ]
            for name, entries in found.items()
        },
        sort_keys=True,
    )


def test_the_core_ignores_the_hash_seed() -> None:
    runs = [run_core(seed) for seed in SEEDS]
    assert runs[0], 'no recording was driven'
    compared = sum(
        1
        for entries in runs[0].values()
        for entry in entries
        if entry['sql'] is None and entry['outputs'] is not None
    )
    assert compared >= 230, f'only {compared} calls ran to the end'
    assert _stable(runs[0]) == _stable(runs[1]), (
        'the core asks other queries or gives other outputs under '
        f'PYTHONHASHSEED {SEEDS[1]} than under {SEEDS[0]}'
    )
