# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.16a: a covered function the candidate takes out of the freeze passes
the checkpoint's ``bridge_freeze.py`` when the candidate declares it
``gone``, and only then.

The gate judges a candidate with its checkpoint's ``bridge_freeze.py``, so
a candidate lowering the freeze (T2.14c: a function the core route no
longer reaches leaves ``COVERED``, and its baseline entry with it) is
judged by a copy that still covers the function. That copy, judging
another tree (not strict), takes the entry missing from the candidate's
baseline when ``bridge_changes.json`` declares it ``gone``, and refuses it,
naming the entry, when nothing declares it. The function itself stays in
its file, unchanged.

The history: the phase's base (P0) holds the full baseline; the package
commits the lowered freeze on it, with its own copy of the script.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Gate tooling (T3.0): CI runs it in the tooling jobs, not in every matrix job.
pytestmark = pytest.mark.tooling

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/rust_core/bridge_freeze.py'
BASE = 'origin/rust-core-phase-2'

_spec = importlib.util.spec_from_file_location('t216a_bridge_freeze', SCRIPT)
assert _spec is not None and _spec.loader is not None
freeze = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = freeze  # its dataclass looks its module up
_spec.loader.exec_module(freeze)

#: A file covered for the functions COVERED names, and a top-level one of
#: them, which the candidate takes out of the freeze.
TARGET = next(rel for rel in freeze.LISTED if 'materializers' in str(rel))
NAME = next(name for name in freeze.COVERED[TARGET] if '.' not in name)
KEY = f'{TARGET}::{NAME}'
#: Where the candidate keeps its own copy of the script.
OWN_COPY = 'scripts/rust_core/bridge_freeze.py'


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), '-c', 'user.name=Freeze Test',
         '-c', 'user.email=freeze@example.com', *args],
        capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL,
    ).stdout.strip()  # fmt: skip


def commit(repo: Path, subject: str) -> str:
    git(repo, 'add', '-A')
    git(repo, 'commit', '-q', '--allow-empty', '-m', subject)
    return git(repo, 'rev-parse', 'HEAD')


def declare(repo: Path, changes: dict[str, dict[str, str]]) -> None:
    path = repo / freeze.CHANGES
    data = json.loads(path.read_text())
    data['changes'] = changes
    path.write_text(json.dumps(data, indent=2) + '\n')


def run(repo: Path) -> tuple[int, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(('GITHUB_', 'SDM_', 'GIT_'))
    }
    done = subprocess.run(
        [sys.executable, str(SCRIPT), '--root', str(repo), '--base', BASE],
        capture_output=True, text=True, check=False, cwd=repo, env=env,
        stdin=subprocess.DEVNULL, timeout=300,
    )  # fmt: skip
    return done.returncode, done.stdout + done.stderr


def lowered(tmp_path: Path) -> Path:
    r"""A repository whose base freezes ``KEY`` and whose candidate commit
    takes it out of its baseline and its own copy's ``COVERED``, the
    function left in its file, nothing declared yet."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    git(repo, 'init', '-q', '-b', 'trunk')
    for rel in (*freeze.FROZEN, freeze.CHANGES):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, repo / rel)
    declare(repo, {})
    freeze.write(repo)
    held = json.loads((repo / freeze.BASELINE).read_text())
    assert NAME in held['files'][str(TARGET)], 'the base freezes the function'
    git(repo, 'update-ref', f'refs/remotes/{BASE}', commit(repo, 'Land T2.0'))
    code, text = run(repo)
    assert code == 0 and 'freeze: ok' in text, text  # control: the base

    # The candidate lowers the freeze: its copy covers the function no
    # longer (another file than the checkpoint's, so the checkpoint's copy
    # judges it as another tree), and its baseline drops the entry.
    copy = repo / OWN_COPY
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(
        SCRIPT.read_text() + f'\n# The candidate covers {NAME} no longer.\n'
    )
    del held['files'][str(TARGET)][NAME]
    (repo / freeze.BASELINE).write_text(json.dumps(held, indent=2) + '\n')
    assert (repo / TARGET).read_text() == (ROOT / TARGET).read_text()
    return repo


def test_a_covered_function_dropped_from_the_freeze_passes_declared_gone(
    tmp_path: Path,
) -> None:
    repo = lowered(tmp_path)
    declare(repo, {KEY: {'gone': f'T2.14c: {NAME} leaves the freeze'}})
    commit(repo, 'T2.14c: the freeze lowered, the entry declared gone')
    code, text = run(repo)
    assert code == 0 and 'freeze: ok' in text, text


def test_a_covered_function_dropped_from_the_freeze_fails_undeclared(
    tmp_path: Path,
) -> None:
    repo = lowered(tmp_path)
    commit(repo, 'T2.14c: the freeze lowered, nothing declared')
    code, text = run(repo)
    assert code == 1, text
    assert KEY in text, text


def test_a_gone_declaration_does_not_excuse_a_changed_function(
    tmp_path: Path,
) -> None:
    # Control: the declaration takes the entry out of the baseline only; the
    # function's skeleton entry still holds, so the module changed under it
    # is still refused.
    repo = lowered(tmp_path)
    declare(repo, {KEY: {'gone': f'T2.14c: {NAME} leaves the freeze'}})
    path = repo / TARGET
    path.write_text(path.read_text() + '\n\nADDED_AT_MODULE_SCOPE = 1\n')
    commit(repo, 'T2.14c: the freeze lowered, the module changed')
    code, text = run(repo)
    assert code == 1, text
    assert f'{TARGET}::{freeze.MODULE}' in text, text
