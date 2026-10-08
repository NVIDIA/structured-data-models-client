# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The parity corpus in CI: it replays, it holds no secret, and the diff
tool still finds what it is seeded with.

Each runs its tool as a user would, from the repository root, with no
database and no credentials. See ``tests/parity/FORMAT.md``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / 'scripts' / 'parity'


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=900,
    )


def _recorded(tabular: bool) -> list[str]:
    proc = _run(str(SCRIPTS / 'run_corpus.py'), 'list', '--json')
    assert proc.returncode == 0, proc.stderr
    return [
        entry['name']
        for entry in json.loads(proc.stdout)
        if entry['status'] == 'recorded'
        and (entry['backend'] == 'tabular') == tabular
    ]


def _replays(names: list[str]) -> None:
    assert names
    proc = _run(
        str(SCRIPTS / 'run_corpus.py'), 'replay', '--only', ','.join(names)
    )
    failed = [
        line for line in proc.stdout.splitlines() if line.startswith('FAIL')
    ]
    assert proc.returncode == 0 and not failed, '\n'.join(failed) or proc.stderr


def test_every_tabular_scenario_replays() -> None:
    r"""The tabular cases need no relational engine, so they run in every
    job that runs this file."""
    _replays(_recorded(tabular=True))


def test_every_relational_scenario_replays() -> None:
    pytest.importorskip(
        'sdm_relational_engine',
        reason='the relational scenarios replay through the engine',
    )
    _replays(_recorded(tabular=False))


def test_every_scenario_is_recorded_or_pending_live() -> None:
    proc = _run(str(SCRIPTS / 'run_corpus.py'), 'list')
    assert proc.returncode == 0, proc.stderr
    unrecorded = [
        line
        for line in proc.stdout.splitlines()
        if line.startswith('unrecorded')
    ]
    assert not unrecorded, unrecorded


def test_the_stored_corpus_holds_no_secret() -> None:
    proc = _run(str(SCRIPTS / 'scrub.py'), 'scan')
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_the_diff_tool_finds_its_seeded_changes() -> None:
    proc = _run(str(SCRIPTS / 'diff.py'), '--self-test')
    assert proc.returncode == 0, proc.stdout + proc.stderr
