# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.0g: a covered function that ``main`` deleted passes the bridge freeze
when it is declared ``main``, naming the commit that deleted it.

The gate judges a candidate with its checkpoint's ``bridge_freeze.py``, so a
candidate merging ``main``'s deletion of a covered function (T2.0c, main's
1fc5a686 deleting ``_check_entity_count``) cannot take it out of ``COVERED``
first. The history is T2.0c's shape: main deletes the function (M1), the
phase's base (P0) forks before it, and the package merges main into it.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Gate tooling (T3.0): CI runs it in the tooling jobs, not in every matrix job.
pytestmark = pytest.mark.gate_tooling

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/rust_core/bridge_freeze.py'
MAIN_REF = 'refs/remotes/origin/main'
BASE = 'origin/rust-core-phase-2'

_spec = importlib.util.spec_from_file_location('t20g_bridge_freeze', SCRIPT)
assert _spec is not None and _spec.loader is not None
freeze = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = freeze  # its dataclass looks its module up
_spec.loader.exec_module(freeze)

#: A file covered for the functions COVERED names, and a top-level one of
#: them, which main's commit deletes.
TARGET = next(rel for rel in freeze.LISTED if 'materializers' in str(rel))
NAME = next(name for name in freeze.COVERED[TARGET] if '.' not in name)
KEYS = (f'{TARGET}::{NAME}', f'{TARGET}::{freeze.MODULE}')


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
        stdin=subprocess.DEVNULL,
    )  # fmt: skip
    return done.returncode, done.stdout + done.stderr


def delete_function(path: Path, name: str) -> None:
    source = path.read_text()
    (node,) = (
        n
        for n in ast.parse(source).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == name
    )
    first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
    lines = source.splitlines(keepends=True)
    path.write_text(''.join(lines[: first - 1] + lines[node.end_lineno :]))


def test_a_covered_function_main_deleted_passes_declared_main(
    tmp_path: Path,
) -> None:
    repo = tmp_path / 'repo'
    repo.mkdir()
    git(repo, 'init', '-q', '-b', 'trunk')
    for rel in (*freeze.FROZEN, freeze.CHANGES):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, repo / rel)
    declare(repo, {})
    freeze.write(repo)
    m0 = commit(repo, 'Start the client (#1)')
    delete_function(repo / TARGET, NAME)
    m1 = commit(repo, 'Delete a covered function (#2)')
    git(repo, 'update-ref', MAIN_REF, m1)
    git(repo, 'checkout', '-q', '-b', 'lane', m0)
    git(repo, 'update-ref', f'refs/remotes/{BASE}', commit(repo, 'Land T2.0'))
    git(repo, 'merge', '-q', '--no-ff', '-m', 'T2.0c: merge main', m1)
    freeze.write(repo)
    commit(repo, 'T2.0c: the baseline')

    # Refused: deleted, and not declared; the missing covered name fails by
    # name.
    code, text = run(repo)
    assert code == 1 and 'COVERED names it' in text, text
    assert all(key in text for key in KEYS), text

    # Passes: declared main, naming the commit that deleted it.
    reason = f'{m1} ' + git(repo, 'log', '-1', '--format=%s', m1)
    declare(repo, {key: {'main': reason} for key in KEYS})
    commit(repo, 'T2.0c: the freeze declares what main deleted')
    code, text = run(repo)
    assert code == 0 and 'freeze: ok' in text, text
