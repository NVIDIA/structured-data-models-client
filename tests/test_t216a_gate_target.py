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
import shlex
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
database=''
database_args=()
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


def test_checkpoint_overlay_owns_the_runners_loaded_semantic_readers(tmp_path):
    text = GATE.read_text()
    found = re.search(r'^overlay=\(\n(.*?)^\)', text, re.M | re.S)
    assert found
    paths = shlex.split(found.group(1))
    assert set(paths) == {
        'contract/runners/run_python.py',
        'contract/runners/prediction_observation.py',
        'contract/runners/observation_publication.py',
        'rust/crates/sdm-testkit/src/case.rs',
        'rust/crates/sdm-testkit/src/observation.rs',
    }
    checkpoint, candidate, scratch = (
        tmp_path / name for name in ('checkpoint', 'candidate', 'scratch')
    )
    for path in paths:
        for root, producer in (
            (checkpoint, 'checkpoint'),
            (candidate, 'candidate'),
        ):
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(producer + ':' + path)
        (scratch / 'judge' / path).parent.mkdir(parents=True, exist_ok=True)
    log = tmp_path / 'observed'
    script = f"""
set -euo pipefail
git() {{ :; }}
python_record() {{
  local directory
  directory=$(dirname "$2")
  cat "$directory/prediction_observation.py" >>'{log}'
  cat "$directory/observation_publication.py" >>'{log}'
}}
here='{checkpoint}'
candidate='{candidate}'
scratch='{scratch}'
head=HEAD
made='{tmp_path / 'made'}'
mkdir -p "$made"
PYTHON=python_record
runs=python-core,python-old
differs=1
database_args=()
overlay=({' '.join(paths)})
worktrees=()
{_observe()}
observe
"""
    done = subprocess.run(
        ['bash', '-c', script], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    expected = ''.join(
        'checkpoint:' + path
        for path in paths
        if path.startswith('contract/runners/')
        and not path.endswith('run_python.py')
    )
    assert log.read_text() == expected * 2
    assert all(
        (candidate / path).read_text() == 'candidate:' + path for path in paths
    )
