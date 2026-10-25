# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.0's acceptance tests for exit 2 (the independent test author's):
every phase's decisions count.

``ratchet.py`` (policy and availability relaxations), ``coverage.py``
(rule 8) and ``contract/runners/capture_check.py`` (the comparison
exclusions) read the owner's approvals from every
``docs/design/rust-core/phase-*/decisions.md``: Phase 1's rows stay in
force, and a row only in ``phase-2/decisions.md`` counts. Each is held on
the real tools: the ratchet and coverage on temporary repositories built as
``tests/test_contract_ratchet.py`` and ``tests/test_contract_coverage.py``
build them (their helpers, reused), the capture check on a copy of its
file beside the files it reads.

Seeded faults, each failing closed and naming the file: an approval only in
a phase file that is not UTF-8 (malformed); Phase 1's file missing while
Phase 2's is there (missing). Controls: a Phase 2 row the owner did not
approve, and one naming another file, allow nothing; Phase 1's approvals
still count.

The interface assumed (see the constants below): the files' path pattern
is the brief's; "malformed" is a file that is not UTF-8 text; "missing" is
Phase 1's file absent at a checkpoint that holds Phase 2's, since Phase
1's rows stay in force. A failure names the file by its path under
``docs/design/rust-core/`` (``phase-2/decisions.md``) in the tool's output.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

# Gate tooling (T3.0): CI runs it in the tooling jobs, not in every matrix job.
pytestmark = pytest.mark.tooling

ROOT = Path(__file__).resolve().parents[1]

# --- the interface assumed ----------------------------------------------------------
DESIGN = 'docs/design/rust-core'
PHASE1_DECISIONS = f'{DESIGN}/phase-1/decisions.md'
PHASE2_DECISIONS = f'{DESIGN}/phase-2/decisions.md'
#: How a failure names a phase file: by this part of its path.
NAMED = {
    PHASE1_DECISIONS: 'phase-1/decisions.md',
    PHASE2_DECISIONS: 'phase-2/decisions.md',
}
#: A malformed phase file: these bytes, not UTF-8, around its rows.
NOT_UTF8 = b'\xff\xfe\x80 not text\n'
# --------------------------------------------------------------------------------------

APPROVED = 'approved by owner 2026-10-16'
HEADER = (
    '# Phase 2 decisions (seeded by the T2.0 tests)\n\n'
    '| ID | Decision | Why | Status |\n|---|---|---|---|\n'
)


def _module(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


#: The ratchet tests' helpers, under the name the other test files share.
helpers = _module(ROOT / 'tests/test_contract_ratchet.py', 'ratchet_tests')
coverage = _module(
    ROOT / 'tests/test_contract_coverage.py', 't20_coverage_tests'
)
git, commit, land = helpers.git, helpers.commit, helpers.land
RATCHET = ROOT / 'scripts/rust_core/ratchet.py'


def row(ident: str, names: str, status: str = APPROVED) -> str:
    return f'| {ident} | A seeded relaxation of {names} | Seeded | {status} |\n'


def phase2(*rows: str, malformed: bool = False) -> Callable[[Path], None]:
    r"""Adds ``rows`` beside existing Phase 2 approvals; malformed, its
    text is not UTF-8."""

    def edit(repo: Path) -> None:
        path = repo / PHASE2_DECISIONS
        path.parent.mkdir(parents=True, exist_ok=True)
        before = path.read_bytes() if path.exists() else HEADER.encode('utf-8')
        text = before.rstrip(b'\n') + b'\n' + ''.join(rows).encode('utf-8')
        path.write_bytes(NOT_UTF8 + text if malformed else text)

    return edit


def phase1_row(*rows: str) -> Callable[[Path], None]:
    def edit(repo: Path) -> None:
        path = repo / PHASE1_DECISIONS
        path.write_text(
            path.read_text(encoding='utf-8').rstrip('\n')
            + '\n'
            + ''.join(rows),
            encoding='utf-8',
        )

    return edit


def both(*edits: Callable[[Path], None]) -> Callable[[Path], None]:
    def edit(repo: Path) -> None:
        for each in edits:
            each(repo)

    return edit


def test_seeding_a_phase_2_decision_preserves_existing_owner_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / PHASE2_DECISIONS
    path.parent.mkdir(parents=True)
    original = HEADER + row('KO2-existing', '`register.yaml`')
    path.write_text(original, encoding='utf-8')
    seeded = row('P2-900', '`compare-policy.yaml`')
    phase2(seeded)(tmp_path)
    assert path.read_text(encoding='utf-8') == original + seeded


# --- ratchet.py: policy and availability relaxations ---------------------------------

POLICY = 'contract/compare-policy.yaml'
CHANGE = 'rule sql.result: gate to diagnostic'


def relaxed(decision: str) -> Callable[[Path], None]:
    def change(document: dict[str, Any]) -> None:
        document['rules']['sql.result'] = 'diagnostic'
        document['relaxations'] = [{'change': CHANGE, 'decision': decision}]

    return helpers.edit_yaml(POLICY, change)


@pytest.fixture(scope='module')
def ratchet_origin(
    owned_module_dir: Callable[[str], Path],
) -> tuple[Path, str]:
    return helpers.build(owned_module_dir('origin'))  # type: ignore[no-any-return]


@pytest.fixture
def repo(ratchet_origin: tuple[Path, str], tmp_path: Path) -> Path:
    return helpers.repo.__wrapped__(ratchet_origin, tmp_path)


def ratchet_check(
    repo: Path, checkpoint: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(RATCHET),
            'check',
            '--repo',
            str(repo),
            '--checkpoint',
            checkpoint,
            '--protected-paths',
            '',
        ],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )


def policy_fails(done: subprocess.CompletedProcess[str]) -> bool:
    return any(
        line.startswith('FAILS policy:') for line in done.stdout.splitlines()
    )


def named_outside_the_register(output: str, path: str) -> bool:
    r"""A line names the file, other than the register rule's, which names
    Phase 1's file for each entry's ``decision_ref`` whenever it is gone
    (not this rule's failure)."""
    return any(
        NAMED[path] in line and not line.startswith('FAILS register:')
        for line in output.splitlines()
    )


def test_ratchet_an_approval_only_in_phase_2_counts(repo: Path) -> None:
    r"""Exit 2: the owner's row in ``phase-2/decisions.md`` at the
    checkpoint authorizes the relaxation that names it."""
    checkpoint = land(
        repo, 'T2.1', phase2(row('P2-900', '`compare-policy.yaml`'))
    )
    commit(repo, 'T2.2: relaxed', relaxed('P2-900'))
    done = ratchet_check(repo, checkpoint)
    assert not policy_fails(done), done.stdout + done.stderr


@pytest.mark.parametrize(
    ('status', 'names'),
    [
        ('owner to confirm', '`compare-policy.yaml`'),
        (APPROVED, '`availability.yaml`'),
    ],
    ids=['not approved', 'another file'],
)
def test_ratchet_a_phase_2_row_that_does_not_approve_allows_nothing(
    repo: Path, status: str, names: str
) -> None:
    r"""Control: a Phase 2 row the owner did not approve, or naming another
    file, authorizes nothing."""
    checkpoint = land(repo, 'T2.1', phase2(row('P2-900', names, status)))
    commit(repo, 'T2.2: relaxed', relaxed('P2-900'))
    done = ratchet_check(repo, checkpoint)
    assert done.returncode == 1 and policy_fails(done), done.stdout


def test_ratchet_a_phase_2_row_the_candidate_adds_allows_nothing(
    repo: Path,
) -> None:
    r"""Control: the candidate's own Phase 2 row never authorizes it."""
    checkpoint = git(repo, 'rev-parse', 'HEAD')
    commit(
        repo,
        'T2.2: relaxed',
        both(phase2(row('P2-900', '`compare-policy.yaml`')), relaxed('P2-900')),
    )
    done = ratchet_check(repo, checkpoint)
    assert done.returncode == 1 and policy_fails(done), done.stdout


def test_ratchet_phase_1_rows_stay_in_force_beside_phase_2_s(
    repo: Path,
) -> None:
    r"""Control, exit 2: with a Phase 2 file present, Phase 1's approval
    still counts."""
    checkpoint = land(
        repo,
        'T2.1',
        both(
            phase1_row(row('E900', '`compare-policy.yaml`')),
            phase2(row('P2-901', '`availability.yaml`')),
        ),
    )
    commit(repo, 'T2.2: relaxed', relaxed('E900'))
    done = ratchet_check(repo, checkpoint)
    assert not policy_fails(done), done.stdout + done.stderr


def test_ratchet_an_approval_only_in_a_malformed_phase_2_file_fails_by_name(
    repo: Path,
) -> None:
    r"""Seeded: the approving row is in a ``phase-2/decisions.md`` that is
    not UTF-8: the check fails and names the file."""
    checkpoint = land(
        repo,
        'T2.1',
        phase2(row('P2-900', '`compare-policy.yaml`'), malformed=True),
    )
    commit(repo, 'T2.2: relaxed', relaxed('P2-900'))
    done = ratchet_check(repo, checkpoint)
    output = done.stdout + done.stderr
    assert done.returncode == 1, output
    assert named_outside_the_register(output, PHASE2_DECISIONS), output
    assert 'Traceback' not in output, output


def test_ratchet_phase_1_s_file_missing_fails_by_name(repo: Path) -> None:
    r"""Seeded: Phase 1's file is gone at a checkpoint holding Phase 2's,
    whose rows would otherwise stand alone: the check fails and names the
    missing file, even for a relaxation Phase 2's file approves."""
    checkpoint = land(
        repo,
        'T2.1',
        both(
            lambda r: (r / PHASE1_DECISIONS).unlink(),
            phase2(row('P2-900', '`compare-policy.yaml`')),
        ),
    )
    commit(repo, 'T2.2: relaxed', relaxed('P2-900'))
    done = ratchet_check(repo, checkpoint)
    output = done.stdout + done.stderr
    assert done.returncode == 1, output
    assert named_outside_the_register(output, PHASE1_DECISIONS), output


# --- coverage.py: rule 8 ---------------------------------------------------------------


@pytest.fixture(scope='module')
def coverage_origin(
    owned_module_dir: Callable[[str], Path],
) -> tuple[Path, str]:
    return coverage.build(owned_module_dir('coverage'))  # type: ignore[no-any-return]


@pytest.fixture
def manifests(coverage_origin: tuple[Path, str], tmp_path: Path) -> Path:
    return helpers.repo.__wrapped__(coverage_origin, tmp_path)


def dropped(repo: Path) -> None:
    coverage.manifest(coverage.PYTHON, coverage.drop('paths', coverage.PATH))(
        repo
    )


def test_coverage_an_approval_only_in_phase_2_allows_a_dropped_path(
    manifests: Path,
) -> None:
    r"""Exit 2, rule 8: the owner's Phase 2 row naming the path allows its
    removal."""
    checkpoint = land(
        manifests, 'T2.1', phase2(row('P2-900', f'`{coverage.PATH}`'))
    )
    commit(manifests, 'T2.2: the regression', dropped)
    coverage.passing(coverage.check(manifests, checkpoint))


def test_coverage_a_phase_2_row_the_owner_did_not_approve_allows_nothing(
    manifests: Path,
) -> None:
    r"""Control."""
    checkpoint = land(
        manifests,
        'T2.1',
        phase2(row('P2-900', f'`{coverage.PATH}`', 'owner to confirm')),
    )
    commit(manifests, 'T2.2: the regression', dropped)
    coverage.failing(
        coverage.check(manifests, checkpoint), 'rule 8', coverage.PATH
    )


def test_coverage_an_approval_only_in_a_malformed_phase_2_file_fails_by_name(
    manifests: Path,
) -> None:
    r"""Seeded: the approving row only in a ``phase-2/decisions.md`` that is
    not UTF-8: rule 8 still fails, and the file is named."""
    checkpoint = land(
        manifests,
        'T2.1',
        phase2(row('P2-900', f'`{coverage.PATH}`'), malformed=True),
    )
    commit(manifests, 'T2.2: the regression', dropped)
    done = coverage.check(manifests, checkpoint)
    output = done.stdout + done.stderr
    assert done.returncode == 1, output
    assert NAMED[PHASE2_DECISIONS] in output, output
    assert 'Traceback' not in output, output


# --- capture_check.py: the comparison exclusions -----------------------------------------

E35_STATUS = 'approved by owner 2026-10-10'


def capture_tree(tmp_path: Path) -> tuple[Path, str]:
    r"""``capture_check.py`` and the files it reads, copied, with E35's row
    taken out of Phase 1's decisions; and that row."""
    tree = tmp_path / 'copy'
    for name in (
        'contract/runners/capture_check.py',
        'contract/runners/captures.py',
        'contract/compare-policy.yaml',
        PHASE1_DECISIONS,
    ):
        (tree / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, tree / name)
    path = tree / PHASE1_DECISIONS
    lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    [e35] = [line for line in lines if line.startswith('| E35 |')]
    assert E35_STATUS in e35
    path.write_text(''.join(x for x in lines if x is not e35), encoding='utf-8')
    return tree, e35


def capture_policy(tree: Path) -> subprocess.CompletedProcess[str]:
    r"""The copy's ``policy()``, with its defaults, as the comparator calls
    it, in a process of its own."""
    return subprocess.run(
        [
            sys.executable,
            '-c',
            'import sys; sys.path.insert(0, sys.argv[1]); '
            'import capture_check; capture_check.policy(); print("policy ok")',
            str(tree / 'contract/runners'),
        ],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )


def test_capture_check_the_control_without_the_row_is_refused(
    tmp_path: Path,
) -> None:
    r"""Control: with E35's row in no decisions file, the exclusion is
    refused (passes on the base too: the copy is sound)."""
    tree, _ = capture_tree(tmp_path)
    done = capture_policy(tree)
    assert done.returncode != 0, done.stdout
    assert 'Traceback' not in done.stderr, done.stderr
    assert 'exclusion request-id' in done.stderr, done.stderr


def test_capture_check_an_exclusion_approved_only_in_phase_2_counts(
    tmp_path: Path,
) -> None:
    r"""Exit 2: E35's row moved to ``phase-2/decisions.md`` still approves
    the exclusion."""
    tree, e35 = capture_tree(tmp_path)
    phase2(e35)(tree)
    done = capture_policy(tree)
    assert done.returncode == 0, done.stdout + done.stderr
    assert 'policy ok' in done.stdout


def test_capture_check_an_exclusion_approved_only_in_a_malformed_file_fails_by_name(
    tmp_path: Path,
) -> None:
    r"""Seeded: E35's row only in a ``phase-2/decisions.md`` that is not
    UTF-8: refused, naming the file."""
    tree, e35 = capture_tree(tmp_path)
    phase2(e35, malformed=True)(tree)
    done = capture_policy(tree)
    assert done.returncode != 0, done.stdout
    assert 'Traceback' not in done.stderr, done.stderr
    assert NAMED[PHASE2_DECISIONS] in done.stderr, done.stderr
