# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P65.hashseed: two ``PYTHONHASHSEED`` runs give identical schemas and
warnings.

``_hashseed_run.py`` writes the Rust core's outcome for every corpus schema,
every ``schema.from_data`` fixture, every describer shape, and the first
generated semantic views, metric views and date columns, with today's
Python's for the corpus and the fixtures. Run under ``PYTHONHASHSEED`` 1 and
4242, after the wheel is built once, the two files must be the same bytes:
no order is taken from a set or from a dict of hashed keys, in the core, its
host or the describer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from _schema import HERE, INSTALLED, native

SCRIPT = HERE / '_hashseed_run.py'


def _run(out: Path, seed: str) -> None:
    env = dict(os.environ)
    env['PYTHONHASHSEED'] = seed
    env[INSTALLED] = '1'
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(out)],
        cwd=HERE,
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    assert proc.returncode == 0, (
        f'the run under PYTHONHASHSEED={seed} exited {proc.returncode}\n'
        f'{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}'
    )


def test_two_hash_seeds_give_the_same_schemas(tmp_path: Path) -> None:
    native(
        'schema_from_data',
        'schema_from_source',
        'metric_view_parse',
        'dates_op',
    )
    first, second = tmp_path / 'seed-1.json', tmp_path / 'seed-4242.json'
    _run(first, '1')
    _run(second, '4242')
    a, b = json.loads(first.read_text()), json.loads(second.read_text())
    assert list(a) == list(b), 'the two runs compared different things'
    assert len(a) >= 3000, f'only {len(a)} answers written'
    differ = [name for name in a if json.dumps(a[name]) != json.dumps(b[name])]
    assert not differ, (
        f'{len(differ)} answers change with PYTHONHASHSEED, such as '
        + ', '.join(differ[:20])
    )
