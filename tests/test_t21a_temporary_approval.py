# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1a's acceptance tests for the temporary-approval rule (the
independent test author's).

The design (T2.1a's row; locked check 1): the decision rows that approve
only temporarily, KO2-45 and KO2-50, are named once, in a short list in
``register.py``, and the locked run's ratchet step fails, naming each
entry, while any approved register entry cites one of them (its
``decision_ref``). An entry its package retires (marked ``rejected``, or
pointed at a permanent decision row; IDs are kept) no longer fails it. A
landing's own gate, which is not the locked run, keeps passing with them:
the entries are approved until their packages land.

Each run is ``ratchet.py check`` on a real git history, built as
``tests/test_contract_ratchet.py`` builds it (its helpers, reused), with
one commit on top adding a register entry; the locked run is the same
check with ``--locked``, as ``gate.sh --locked`` passes it. Every seeded
fault is shown beside its unseeded twin, so a refusal is the seed's.

Gating: :data:`PENDING`, a strict xfail; the implementer deletes its one
definition. Unmarked tests are controls that pass today.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

# Gate tooling (T3.0): CI runs it in the tooling jobs, not in every matrix job.
pytestmark = pytest.mark.gate_tooling

ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'
DECISIONS = 'docs/design/rust-core/phase-2/decisions.md'


# --- the interface assumed (the implementer matches it, or asks) ---------------
#
#: ``register.py``'s ``TEMPORARY``: the temporary decision rows, the one
#: place they are named.
TEMPORARY = ('KO2-45', 'KO2-50', 'KO2-56')
#: ``ratchet.py check ... --locked``: the locked run's ratchet step, which
#: fails, naming each approved entry whose ``decision_ref`` row is one of
#: them; without ``--locked`` the rule does not apply.
# ---------------------------------------------------------------------------------


def _module(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


helpers = _module(ROOT / 'tests/test_contract_ratchet.py', 'ratchet_tests')
git, commit = helpers.git, helpers.commit

#: The seeded entry's ID: a new one, of no existing prefix's numbering.
SEED = 'T21A-1'


@pytest.fixture(scope='module')
def origin(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    repo, landed = helpers.build(tmp_path_factory.mktemp('origin'))
    return repo, landed


@pytest.fixture
def tree(origin: tuple[Path, str], tmp_path: Path) -> Path:
    repo, landed = origin
    tree = tmp_path / 'tree'
    git(repo, 'worktree', 'add', '-q', '--detach', str(tree), landed)
    return tree


def _entry(row: str, status: str = 'approved') -> Callable[[Path], None]:
    def edit(tree: Path) -> None:
        helpers.append_entry(
            tree,
            {
                'id': SEED,
                'title': 'A seeded temporary approval',
                'kind': 'change',
                'cases': [],
                'refs': ['T2.1a tests'],
                'status': status,
                'approved_by': None if status == 'proposed' else 'owner',
                'decision_ref': {'file': DECISIONS, 'row': row},
            },
        )

    return edit


def _check(tree: Path, checkpoint: str, locked: bool) -> tuple[int, str]:
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'scripts/rust_core/ratchet.py'),
            'check',
            '--repo',
            str(tree),
            '--checkpoint',
            checkpoint,
            *(['--locked'] if locked else []),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return done.returncode, done.stdout + done.stderr


def _failures(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith('FAILS ')]


def _seeded(tree: Path, row: str, status: str = 'approved') -> str:
    checkpoint = git(tree, 'rev-parse', 'HEAD')
    commit(tree, 'T2.1a: a seeded register entry', _entry(row, status))
    return checkpoint


def _approved_temporary(path: Path) -> list[str]:
    import yaml

    entries: list[dict[str, Any]] = yaml.safe_load(path.read_text())['entries']
    return [
        entry['id']
        for entry in entries
        if entry.get('status') == 'approved'
        and (entry.get('decision_ref') or {}).get('row') in TEMPORARY
    ]


def test_the_seed_s_rows_are_the_temporary_owner_rows() -> None:
    r"""Control: KO2-45, KO2-50 and KO2-56 are rows of the phase's decisions file,
    each approving until a package lands."""
    text = (ROOT / DECISIONS).read_text()
    for row in TEMPORARY:
        [line] = [x for x in text.splitlines() if x.startswith(f'| {row} |')]
        assert 'approved by owner' in line


def test_the_temporary_rows_are_named_once_in_register_py() -> None:
    r"""Exit: the list lives in ``register.py``, and no other harness file
    under ``scripts/rust_core`` spells a temporary row."""
    register = _module(ROOT / 'scripts/rust_core/register.py', 't21a_register')
    assert tuple(sorted(register.TEMPORARY)) == TEMPORARY
    spelled = [
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / 'scripts/rust_core').rglob('*.py'))
        if path.name != 'register.py'
        and re.search(r'KO2-(45|50)\b', path.read_text())
    ]
    assert not spelled, spelled


@pytest.mark.parametrize('row', TEMPORARY)
def test_the_locked_run_fails_naming_an_approved_temporary_entry(
    row: str, tree: Path
) -> None:
    r"""Exit and seeded fault ("a temporary entry left after its package
    lands fails the locked run by name"): the locked ratchet fails, naming
    the entry; the same candidate's landing ratchet does not name it."""
    checkpoint = _seeded(tree, row)
    status, output = _check(tree, checkpoint, locked=True)
    named = [line for line in _failures(output) if SEED in line]
    assert status == 1 and named, output
    assert any(row in line for line in named), named
    _, unlocked = _check(tree, checkpoint, locked=False)
    assert not [line for line in _failures(unlocked) if SEED in line], unlocked


@pytest.mark.parametrize(
    ('row', 'status'),
    [
        ('KO2-45', 'rejected'),
        ('KO2-35', 'approved'),
    ],
    ids=['retired as rejected', 'pointed at a permanent row'],
)
def test_a_retired_entry_passes_the_locked_run(
    row: str, status: str, tree: Path
) -> None:
    r"""Control: an entry its package retired (rejected, or citing a
    permanent row) is not named by the locked run."""
    checkpoint = _seeded(tree, row, status)
    code, output = _check(tree, checkpoint, locked=True)
    # A judgement (0 or 1), not a usage error (argparse's 2).
    assert code in (0, 1), output
    assert not [line for line in _failures(output) if SEED in line], output


def test_the_locked_run_names_every_approved_temporary_entry_of_the_register(
    tree: Path,
) -> None:
    r"""Exit: on the tree as it stands (after T2.0c's merge, its KO2-45
    entries), the locked ratchet names every approved entry citing a
    temporary row, and passes the rule when there is none. Judged on an
    empty commit, so nothing but the register as committed is seen."""
    checkpoint = git(tree, 'rev-parse', 'HEAD')
    commit(tree, 'T2.1a: nothing')
    expected = _approved_temporary(tree / 'docs/design/rust-core/register.yaml')
    code, output = _check(tree, checkpoint, locked=True)
    assert code in (0, 1), output
    failures = _failures(output)
    for entry in expected:
        assert any(
            re.search(rf'\b{re.escape(entry)}\b', f) for f in failures
        ), (
            entry,
            output,
        )
    if expected:
        assert code == 1, output
