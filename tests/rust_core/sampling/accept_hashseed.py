# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P7.hashseed: the corpus and the fixtures replay identically under two
``PYTHONHASHSEED`` values.

- The corpus replay and the fixture runner of ``tests/parity`` (every
  transcript and every fixture file, the new sampler fixtures included) pass
  under ``PYTHONHASHSEED`` 1 and 4242.
- The Rust core's answers to every ``labels.*`` and ``sampling.*`` fixture
  case and to the first generated inputs of every other check (random
  numbers, offsets, text, aggregations, graphs, graph stores and questions)
  are the same bytes under both.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from _jobs import array_jobs, python_jobs
from _sampling import HERE, INSTALLED, REPO, native

SEEDS = ('1', '4242')


@pytest.mark.parametrize('seed', SEEDS)
def test_the_corpus_and_fixtures_replay(seed: str) -> None:
    env = dict(os.environ)
    env['PYTHONHASHSEED'] = seed
    proc = subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            'tests/parity/test_corpus.py',
            'tests/parity/test_fixtures.py',
            '-q',
            '-p',
            'no:cacheprovider',
            '-m',
            'not live_nim and not live_warehouse',
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    assert proc.returncode == 0, (
        f'the replay fails under PYTHONHASHSEED={seed}:\n{proc.stdout[-6000:]}'
        f'\n{proc.stderr[-3000:]}'
    )


def _answers(jobs: Path, out: Path, seed: str) -> list[object]:
    env = dict(os.environ)
    env['PYTHONHASHSEED'] = seed
    env[INSTALLED] = '1'
    proc = subprocess.run(
        [sys.executable, str(HERE / '_jobs.py'), 'all', str(jobs), str(out)],
        cwd=HERE,
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    assert proc.returncode == 0, (
        f"the core's run under PYTHONHASHSEED={seed} exited {proc.returncode}\n"
        f'{proc.stderr[-6000:]}'
    )
    return json.loads(out.read_text())


def test_the_core_answers_the_same_under_both() -> None:
    native(
        'labels_op',
        'sampling_op',
        'NeighborSampler',
        'graph_store',
        'labels_execute',
    )
    with tempfile.TemporaryDirectory(prefix='sdm-p7-hashseed-') as tmp:
        root = Path(tmp)
        jobs = python_jobs(500) + array_jobs(500, root)
        given = root / 'jobs.json'
        given.write_text(json.dumps(jobs))
        first = _answers(given, root / 'a.json', SEEDS[0])
        second = _answers(given, root / 'b.json', SEEDS[1])
    assert len(first) == len(second) == len(jobs)
    raised = [a for a in first if isinstance(a, dict) and 'raised' in a]
    assert len(raised) < len(jobs) // 2, (
        f'{len(raised)} of {len(jobs)} jobs raised, such as {raised[:3]}'
    )
    differ = [
        f'{jobs[i]["kind"]} #{i}'
        for i, (a, b) in enumerate(zip(first, second, strict=True))
        if json.dumps(a) != json.dumps(b)
    ]
    assert not differ, (
        f'{len(differ)} answers change with PYTHONHASHSEED: {differ[:20]}'
    )
