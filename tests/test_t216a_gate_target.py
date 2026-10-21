# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.16a fix 3 (the Claude review of round 1, P1-1): ``gate.sh`` builds
rust-host from a judging worktree with the checkpoint's observation writer
overlaid. Built into the candidate's own target directory, Cargo (judging
freshness by modification times) took that build for the candidate's
afterwards, and CI's SQLite and Postgres cross-host ran the checkpoint's
writer. The overlaid build goes to a target directory of its own; the
candidate's own observation still uses the candidate's.

The gate's ``observe`` function is run as written, with ``git`` and
``cargo`` standing in, each recording what it was given.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / 'scripts/rust_core/gate.sh'


def _observe() -> str:
    text = GATE.read_text()
    found = re.search(r'^observe\(\) \{\n.*?^\}\n', text, re.M | re.S)
    assert found, 'gate.sh has no observe function'
    return found.group(0)


def _target_dir(tmp_path: Path, differs: int) -> str:
    candidate = tmp_path / f'candidate-{differs}'
    (candidate / 'rust').mkdir(parents=True)
    scratch = tmp_path / f'scratch-{differs}'
    scratch.mkdir()
    log = tmp_path / f'cargo-{differs}.log'
    script = f"""
set -euo pipefail
git() {{ mkdir -p "${{@: -2:1}}/rust"; }}
cargo() {{ echo "$CARGO_TARGET_DIR" >>'{log}'; }}
here='{tmp_path}'
candidate='{candidate}'
scratch='{scratch}'
head=HEAD
made='{tmp_path}/made-{differs}'
mkdir -p "$made"
runs=rust-host
differs={differs}
overlay=()
worktrees=()
{_observe()}
observe
"""
    done = subprocess.run(
        ['bash', '-c', script], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    return log.read_text().strip()


def test_the_overlaid_host_is_built_apart_from_the_candidate_s(tmp_path):
    if shutil.which('bash') is None:
        pytest.skip('no bash')
    own = _target_dir(tmp_path, 0)
    assert own == str(tmp_path / 'candidate-0' / 'rust' / 'target')
    overlaid = _target_dir(tmp_path, 1)
    shared = str(tmp_path / 'candidate-1' / 'rust' / 'target')
    assert overlaid and overlaid != shared
