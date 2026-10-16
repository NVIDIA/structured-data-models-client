# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``scripts/rust_core/orchestration/codex_pane.py``: its self-test, against
a fake herdr and a throwaway repository, with no network, run where CI's
``pytest tests`` jobs run it."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SELFTEST = (
    Path(__file__).resolve().parents[1]
    / 'scripts/rust_core/orchestration/codex_pane_selftest.py'
)


def test_the_codex_pane_self_test_passes():
    env = {k: v for k, v in os.environ.items() if not k.startswith('SDM_')}
    done = subprocess.run(
        [sys.executable, str(SELFTEST)],
        capture_output=True,
        text=True,
        env={**env, 'GIT_TERMINAL_PROMPT': '0'},
        stdin=subprocess.DEVNULL,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.rstrip().endswith('all passed'), done.stdout
