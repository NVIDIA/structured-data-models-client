# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Heavy work through the slot wrapper only (T3.2 fix 4; the owner's
rule, 2026-10-22, "the five speed changes", change 3):

- ``heavy --priority`` jumps the queue, and a job admitted after waiting
  leaves a line in ``heavy-waits.log``;
- the command guard refuses ``cargo test``, a pytest run over a whole test
  directory and ``gate.sh`` outside ``heavy``, and lets them through behind
  it.

``lane.py``'s limit on active lanes is held by ``lane_selftest.py``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
ORCH = ROOT / 'scripts/rust_core/orchestration'


def _module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GUARD = _module(ORCH / 'guard_cmd.py', 't32_heavy_guard_cmd')


def _wait_for(check: object, seconds: float = 20) -> None:
    deadline = time.monotonic() + seconds
    while not check():  # type: ignore[operator]
        assert time.monotonic() < deadline, 'timed out'
        time.sleep(0.1)


def test_a_priority_job_takes_the_next_slot_and_every_wait_is_logged(
    tmp_path: Path,
) -> None:
    heavy = tmp_path / 'heavy'
    shutil.copy2(ORCH / 'heavy', heavy)
    conf = tmp_path / 'heavy.conf'
    conf.write_text(
        f'SLOTS=1\nMIN_FREE_MEM_GB=0\nMIN_FREE_DISK_GB=0\nDISK={tmp_path}\n'
        'POLL=1\nMAX_LANES=3\n'
    )
    state = tmp_path / 'state'
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith('SDM_')},
        'SDM_HEAVY_CONFIG': str(conf),
        'SDM_ORCH_STATE': str(state),
    }
    order = tmp_path / 'order'

    def start(*argv: str, **extra: str) -> subprocess.Popen[str]:
        return subprocess.Popen(
            ['bash', str(heavy), *argv],
            env={**env, **extra},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    holder = start(
        'sh', '-c',
        f'touch {tmp_path}/held; while [ ! -e {tmp_path}/release ]; do '
        'sleep 0.1; done',
    )  # fmt: skip
    jobs = [holder]
    try:
        _wait_for(lambda: (tmp_path / 'held').exists())
        jobs.append(start('sh', '-c', f'echo low >> {order}', SDM_LANE='x'))
        time.sleep(1.5)
        jobs.append(
            start(
                '--priority', 'sh', '-c', f'echo high >> {order}', SDM_LANE='y'
            )
        )
        _wait_for(lambda: any((state / 'heavy-priority').glob('*')))
        (tmp_path / 'release').touch()
        outs = [job.communicate(timeout=30)[0] for job in jobs]
    finally:
        for job in jobs:
            if job.poll() is None:
                job.kill()
    assert [j.returncode for j in jobs] == [0, 0, 0], outs
    assert order.read_text().split() == ['high', 'low'], outs
    waits = (state / 'heavy-waits.log').read_text().splitlines()
    assert any('who=y priority=yes' in w and 'echo high' in w for w in waits)
    assert any('who=x priority=no' in w and 'echo low' in w for w in waits)
    assert not any((state / 'heavy-priority').glob('*'))


@pytest.mark.parametrize(
    'command',
    [
        'cargo test -p sdm-testkit',
        'cd rust && cargo build --locked',
        'cargo +stable clippy --all-targets',
        'python -m pytest -q -n 2 tests/',
        '/venv/bin/python -I -m pytest contract/tests',
        'uv run pytest ./tests',
        'bash scripts/rust_core/gate.sh --base HEAD~1',
    ],
)
def test_the_guard_refuses_heavy_work_outside_heavy(command: str) -> None:
    verdict = GUARD.check_command(command, '.', None)
    assert verdict is not None and 'outside heavy' in verdict.rule, command
    assert 'heavy' in verdict.instead


@pytest.mark.parametrize(
    'command',
    [
        'heavy cargo test -p sdm-testkit',
        '$SDM_ORCH_STATE/bin/heavy --priority cargo build',
        "heavy bash -c 'cd rust && cargo clippy'",
        "perl -e 'alarm shift; exec @ARGV' 600 heavy python -m pytest tests",
        'python -m pytest -q tests/test_t32_tools.py',
        'cargo fmt --check',
    ],
)
def test_the_guard_lets_heavy_work_through_behind_heavy(command: str) -> None:
    assert GUARD.check_command(command, '.', None) is None, command
