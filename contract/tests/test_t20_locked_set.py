# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.0's acceptance tests for exit 5 (the independent test author's):
``gate.sh --locked`` for any phase, its locked set read as data from the
checkpoint, and ``phase_end.sh``'s inventory not Phase 1's by default.

Through the real ``scripts/rust_core/gate.sh`` on the temporary repository
and stubs ``contract/tests/test_gate_modes.py`` builds (its fixtures,
reused: the checkpoint holding the locked steps' stubs, a ``cargo`` stub,
the Python runners' real observations). A fixture Phase 2 locked-set file,
landed at the checkpoint, names a half ``2a`` and its cases (Phase 2's real
file is the design's, T2.0a, not this package's):

- with the file, ``--locked 2a`` is accepted and passes on the stubs while
  every case it names is an acceptance case, and the evidence records the
  half;
- seeded: one of its cases diagnostic at the checkpoint fails the half,
  naming exactly that case;
- seeded: a half the file does not name (``2z``), and a half of a phase
  with no locked-set file (``3a``: Phase 3 has none; Phase 2's is the
  design's, T2.0a) are usage errors naming the half;
- Phase 1's ``1a`` and ``1b`` stay as they are: ``test_gate_modes.py``'s
  own tests hold them, unchanged.

``phase_end.sh --phase 2`` with no ``SDM_INVENTORY`` (a dry run) no longer
reads Phase 1b's inventory; ``--phase 1`` still does (control).

The interface assumed, in one place: :data:`LOCKED_SET` (the file's path,
per phase) and :func:`locked_set_text` (its shape). The design may name the
file otherwise; the implementer then changes these two and nothing else.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


# --- the interface assumed ------------------------------------------------------------
def LOCKED_SET(phase: int) -> str:  # noqa: N802 - the assumption, named as a constant
    r"""Where a phase's locked set lives, in the checkpoint's tree."""
    return f'docs/design/rust-core/phase-{phase}/locked-set.yaml'


def locked_set_text(halves: dict[str, list[str]]) -> str:
    r"""The file: each half by name, the cases that must be acceptance
    cases for its locked checks to pass."""
    return yaml.safe_dump(
        {'halves': {half: {'cases': cases} for half, cases in halves.items()}},
        sort_keys=False,
    )


# ----------------------------------------------------------------------------------------


def _module(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GM = _module(ROOT / 'contract/tests/test_gate_modes.py', 't20_gate_modes')
helpers = GM.helpers
git, commit, land = helpers.git, helpers.commit, helpers.land

# test_gate_modes' fixtures, used here as there.
origin = GM.origin
observed = GM.observed
stubbed = GM.stubbed
stubs = GM.stubs

#: Phase 2's fixture half and the acceptance cases it locks.
HALF = '2a'
CASES = [
    's0.accept.binary-classification',
    's0.accept.multiclass-classification',
    's0.accept.sum-window',
]


def with_locked_set(halves: dict[str, list[str]]) -> Callable[[Path], None]:
    def edit(repo: Path) -> None:
        path = repo / LOCKED_SET(2)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(locked_set_text(halves), encoding='utf-8')

    return edit


def run_locked(
    tree: Path, stubs: dict[str, Any], observed: Any, tmp_path: Path, half: str
) -> subprocess.CompletedProcess[str]:
    return GM.locked(  # type: ignore[no-any-return]
        tree, stubs, half, '--run', 'rust-host', *GM.files(observed, tmp_path)
    )


def test_the_fixture_s_cases_are_acceptance_cases() -> None:
    r"""The seed's premise (passes on the base): each case the fixture locks
    is an acceptance case here, so a failure below is the seed's."""
    status = {
        json.loads(p.read_text())['id']: json.loads(p.read_text())['status']
        for p in (ROOT / 'contract/cases').rglob('*.json')
    }
    assert {case: status.get(case) for case in CASES} == dict.fromkeys(
        CASES, 'acceptance'
    )


def test_a_phase_2_half_named_by_its_locked_set_passes_and_is_recorded(
    stubbed: Path,
    tmp_path: Path,
    observed: Any,
    stubs: dict[str, Any],
) -> None:
    r"""Exit 5: the checkpoint's Phase 2 file names ``2a``; ``--locked 2a``
    is accepted, passes on the stubs, and its evidence says so."""
    land(stubbed, 'T2.0a', with_locked_set({HALF: CASES}), protected='docs/**')
    commit(stubbed)
    done = run_locked(stubbed, stubs, observed, tmp_path, HALF)
    assert done.returncode == 0, done.stdout + done.stderr
    assert GM.last(done) == f'gate: passed (--locked {HALF})', done.stdout
    half = GM.evidence_of(stubbed, stubs)['halves'][HALF]
    assert half['passed'] is True, half


def test_a_phase_2_case_diagnostic_at_the_checkpoint_fails_its_half_by_name(
    stubbed: Path,
    tmp_path: Path,
    observed: Any,
    stubs: dict[str, Any],
) -> None:
    r"""Seeded: one locked case diagnostic at the checkpoint fails ``2a``,
    naming exactly that case."""
    seeded = CASES[1]
    land(
        stubbed,
        'T2.0a',
        lambda r: (
            with_locked_set({HALF: CASES})(r),
            GM._diagnostic(seeded)(r),
        ),
        protected='docs/** contract/cases/**',
    )
    commit(stubbed)
    done = run_locked(stubbed, stubs, observed, tmp_path, HALF)
    assert done.returncode != 0, done.stdout + done.stderr
    naming = [x for x in done.stdout.splitlines() if seeded in x]
    assert naming, done.stdout
    assert not any(c in x for x in naming for c in (CASES[0], CASES[2])), naming
    assert GM.evidence_of(stubbed, stubs)['halves'][HALF]['passed'] is False


def test_a_half_the_locked_set_does_not_name_is_a_usage_error(
    stubbed: Path,
    tmp_path: Path,
    observed: Any,
    stubs: dict[str, Any],
) -> None:
    r"""Seeded: ``2z`` is in no locked set: refused before any step."""
    land(stubbed, 'T2.0a', with_locked_set({HALF: CASES}), protected='docs/**')
    commit(stubbed)
    done = run_locked(stubbed, stubs, observed, tmp_path, '2z')
    assert done.returncode == 2, done.stdout + done.stderr
    assert '2z' in done.stdout + done.stderr
    assert 'gate summary' not in done.stdout


#: A phase with no locked-set file yet, and a half of it.
NO_SET_PHASE = 3
NO_SET_HALF = '3a'


def test_a_half_of_a_phase_without_its_locked_set_is_a_usage_error(
    stubbed: Path,
    tmp_path: Path,
    observed: Any,
    stubs: dict[str, Any],
) -> None:
    r"""Seeded: a half of a phase whose locked-set file does not exist
    (Phase 3's, before its design lands one) is refused naming the half:
    no locked set, no locked checks. Written for Phase 2 until T2.0a
    landed its file; Phase 3 keeps the premise true."""
    commit(stubbed)
    assert not (stubbed / LOCKED_SET(NO_SET_PHASE)).exists()
    done = run_locked(stubbed, stubs, observed, tmp_path, NO_SET_HALF)
    assert done.returncode == 2, done.stdout + done.stderr
    assert NO_SET_HALF in done.stdout + done.stderr
    assert 'gate summary' not in done.stdout


# --- phase_end.sh's inventory --------------------------------------------------------------

PHASE_END = ROOT / 'scripts/rust_core/orchestration/phase_end.sh'
PHASE_1B_INVENTORY = 'phase-0/phase-1b-inventory.yaml'


def phase_end(phase: int, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    r"""``phase_end.sh``'s dry run (nothing built, run or written) for
    ``phase``, from this checkout's HEAD, with no ``SDM_INVENTORY``."""
    sha = subprocess.run(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    env = {k: v for k, v in os.environ.items() if not k.startswith('SDM_')}
    return subprocess.run(
        [
            'bash', str(PHASE_END), '--phase', str(phase), '--sha', sha,
            '--work', str(tmp_path / f'phase-{phase}'),
        ],
        env={**env, 'SDM_REPO': str(ROOT), 'GH_PROMPT_DISABLED': '1'},
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=300,
    )  # fmt: skip


def test_phase_end_for_phase_2_does_not_default_to_phase_1b_s_inventory(
    tmp_path: Path,
) -> None:
    r"""Exit 5: the inventory is phase-derived or required: Phase 2's end
    run never reads Phase 1b's by default (it names Phase 2's, or stops
    asking for SDM_INVENTORY)."""
    done = phase_end(2, tmp_path)
    output = done.stdout + done.stderr
    assert PHASE_1B_INVENTORY not in output, output[-3000:]
    if done.returncode != 0:
        assert 'SDM_INVENTORY' in output or 'phase-2' in output, output[-3000:]


def test_phase_end_for_phase_1_still_reads_phase_1b_s_inventory(
    tmp_path: Path,
) -> None:
    r"""Control: Phase 1's end run is unchanged (passes on the base)."""
    done = phase_end(1, tmp_path)
    assert PHASE_1B_INVENTORY in done.stdout + done.stderr, done.stdout[-3000:]
