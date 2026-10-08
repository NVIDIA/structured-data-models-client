# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P2.offline: the full replay runs without network or credentials in under
10 minutes."""

from __future__ import annotations

import time
from pathlib import Path

from _support import (
    RUN_CORPUS,
    offline_env,
    output,
    recorded,
    replay_lines,
    run,
)

LIMIT_SECONDS = 600


def test_full_replay_is_offline_and_fast(tmp_path: Path) -> None:
    scenarios = recorded()
    env = offline_env(tmp_path / 'home')
    started = time.monotonic()
    proc = run(RUN_CORPUS, 'replay', env=env, timeout=LIMIT_SECONDS)
    elapsed = time.monotonic() - started
    assert proc.returncode == 0, output(proc)
    lines = replay_lines(proc.stdout)
    missing = [
        e['name']
        for e in scenarios
        if lines.get(e['name'], ('missing',))[0] != 'PASS'
    ]
    assert not missing, f'not replayed offline: {missing}\n{output(proc)}'
    assert elapsed < LIMIT_SECONDS, f'replay took {elapsed:.0f}s'
    print(f'offline replay of {len(scenarios)} scenarios: {elapsed:.1f}s')
