# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P7.independent: the Rust core answers on its own, not by calling today's
Python back.

The other checks compare the core with today's Python in one process, where
a binding could pass by asking numpy, ``relationallib`` or pandas. Here the
same questions go to the core in a process where those can't be imported,
and the answers must be the ones it gives with everything importable:

- random numbers, ``std::mt19937``, offsets, text, and the ``labels.*`` and
  ``sampling.*`` operations (fixtures and generated aggregations), with
  numpy, pandas, dateutil, pytz, torch, ``sdm_relational_engine``,
  ``sdm_client`` and ``predictive_query`` unimportable;
- the neighbor sampler, the graph store and ``labels_execute``, with all of
  those but numpy unimportable, numpy and pyarrow carrying the arrays.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from _jobs import answer_all, array_jobs, python_jobs
from _sampling import HERE, INSTALLED, native


def _run(mode: str, jobs: list[dict], root: Path) -> list[object]:
    given, out = root / f'{mode}-jobs.json', root / f'{mode}-out.json'
    given.write_text(json.dumps(jobs))
    env = dict(os.environ)
    env[INSTALLED] = '1'
    proc = subprocess.run(
        [sys.executable, str(HERE / '_jobs.py'), mode, str(given), str(out)],
        cwd=HERE,
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    assert proc.returncode == 0, proc.stderr[-6000:]
    return json.loads(out.read_text())


def _compare(mode: str, jobs: list[dict], root: Path) -> None:
    native()
    here = answer_all(jobs)
    raised = [a for a in here if isinstance(a, dict) and 'raised' in a]
    assert len(raised) < len(jobs) // 2, (
        f'{len(raised)} of {len(jobs)} jobs raised with everything importable, '
        f'such as {raised[:3]}'
    )
    there = _run(mode, jobs, root)
    assert len(there) == len(here)
    differ = [
        f'{jobs[i]["kind"]} #{i}: {json.dumps(b)[:300]} instead of {json.dumps(a)[:300]}'
        for i, (a, b) in enumerate(zip(here, there, strict=True))
        if json.dumps(a) != json.dumps(b)
    ]
    assert not differ, (
        f'{len(differ)} of {len(jobs)} answers change without '
        f'{mode.removeprefix("no-")} Python:\n' + '\n'.join(differ[:15])
    )


def test_json_answers_need_no_python_numerics() -> None:
    with tempfile.TemporaryDirectory(prefix='sdm-p7-independent-') as tmp:
        _compare('no-python', python_jobs(1_000), Path(tmp))


def test_array_answers_need_no_engine() -> None:
    with tempfile.TemporaryDirectory(prefix='sdm-p7-independent-') as tmp:
        root = Path(tmp)
        _compare('no-engine', array_jobs(1_000, root), root)
