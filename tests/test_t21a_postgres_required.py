# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1a's seeded fault for H26 (the independent test author's): with
``SDM_REQUIRE_POSTGRES`` set, ``contract/tests/test_postgres_required.py``
fails, never stands down, when the Postgres service is unreachable or not
named; without it, it passes.

Each run is the real file under pytest in a child process, with the
environment set as a Postgres job sets it. The unreachable server is a
localhost port that was just closed, so the refusal is immediate.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / 'contract/tests/test_postgres_required.py'


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return int(probe.getsockname()[1])


def _run(**environment: str) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ('SDM_REQUIRE_POSTGRES', 'SDM_POSTGRES_TEST_DSN')
    }
    env.update(environment)
    return subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            '-q',
            '-p',
            'no:cacheprovider',
            '-rs',
            str(TEST),
        ],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_not_required_passes() -> None:
    r"""Control: no Postgres job, nothing required."""
    done = _run()
    assert done.returncode == 0, done.stdout
    assert '1 passed' in done.stdout


def test_an_unreachable_service_fails_where_required() -> None:
    r"""Seeded (row: "an unreachable Postgres service with
    SDM_REQUIRE_POSTGRES set ... fail"): a failure, not a skip."""
    dsn = f'postgresql://postgres@127.0.0.1:{_closed_port()}/sdm_test'
    done = _run(SDM_REQUIRE_POSTGRES='1', SDM_POSTGRES_TEST_DSN=dsn)
    assert done.returncode == 1, done.stdout
    assert '1 failed' in done.stdout and 'skipped' not in done.stdout
    assert 'unreachable' in done.stdout


def test_a_missing_dsn_fails_where_required() -> None:
    r"""Seeded: required, and no server named."""
    done = _run(SDM_REQUIRE_POSTGRES='1')
    assert done.returncode == 1, done.stdout
    assert 'SDM_POSTGRES_TEST_DSN is not' in done.stdout
