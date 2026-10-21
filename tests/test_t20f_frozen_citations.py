# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.0f's tests of the frozen citing files (the independent test
author's), on the orchestrator's decision for the rebase onto T2.0e.

``tests/golden/t132g-fixture-cases/python-ref.jsonl`` and
``tests/golden/t132g-later-stage/python-ref.jsonl`` are frozen historical
snapshots of one gate run: they keep naming the reference they were
recorded at (b413e9ad) after the reference is re-recorded. The ratchet
lists them explicitly as frozen, each with its reason, out of
:data:`ratchet.CITATIONS` (the files and fields the reference SHA
consistency tests read) and exempt from the rule that no file outside
``docs/`` keeps the old SHA.

Each test runs ``ratchet.py``'s checks on the real git history T2.0e's
tests build (``tests/test_t20e_reference_change.py``, its helpers and
fixtures reused): every file of this checkout citing the reference SHA,
the owner's row approving a fake new SHA at the checkpoint, and a
candidate re-recording the reference at it.

- An approved re-record that leaves the two frozen files naming the old
  SHA passes, every other citation moved.
- Seeded, on that twin: any other citing file left naming the old SHA, a
  file the consistency tests read or a provenance file outside the
  table, is refused, naming it.
- The frozen list can't hide a citing file the consistency tests read:
  it is exactly the two files, each with a reason, none in the citation
  table, each in this checkout; and a table file added to the frozen list
  (seeded) is still refused when left at the old SHA.

Gating: the tests that need the feature are marked :data:`PENDING`, a
strict xfail; the implementer deletes the marker (its one definition below)
in the commit that makes them pass.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]

# --- the interface assumed (the implementer matches it, or asks) ---------------
#
#: The ratchet's frozen list: ``ratchet.FROZEN``, each frozen file's path
#: (as CITATIONS keys are written) mapped to its reason, a non-empty string.
FROZEN_NAME = 'FROZEN'
#: The orchestrator's decision: exactly these two files.
FROZEN_FILES = (
    'tests/golden/t132g-fixture-cases/python-ref.jsonl',
    'tests/golden/t132g-later-stage/python-ref.jsonl',
)
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


T20E = _module(
    ROOT / 'tests/test_t20e_reference_change.py', 't20f_t20e_reference_change'
)
ratchet = T20E.ratchet
git, commit = T20E.git, T20E.commit
NEW = T20E.NEW

# T2.0e's tests' fixtures, used here as there.
origin = T20E.origin
repo = T20E.repo

#: A citing file the consistency tests read (in the citation table).
TABLE_FILE = 'contract/decimal-vectors.json'
#: A citing file outside the table (provenance), moved by the old-SHA rule.
PROSE_FILE = 'rust/crates/sdm-core/tests/pql/corpus.json'


def _frozen() -> dict[str, str]:
    found: dict[str, str] = getattr(ratchet, FROZEN_NAME)
    return found


def _rerecord_leaving(*left: str) -> Callable[[Path], None]:
    r"""The re-record at ``NEW``, every citation of the old SHA moved but
    in ``left``, which keep naming the old SHA."""
    edit: Callable[[Path], None] = T20E.rerecord(NEW, skip=tuple(left))
    return edit


def _twin(repo: Path) -> str:
    r"""The checkpoint approving ``NEW``, and the twin: the frozen files
    left naming the old SHA, every other citation moved, passing; the
    worktree back at the checkpoint."""
    old = T20E.old_sha(repo)
    for name in FROZEN_FILES:
        assert old in (repo / name).read_text(), name
    checkpoint: str = T20E.approved_checkpoint(repo)
    commit(
        repo,
        'T2.0c: the reference re-recorded, the frozen files kept',
        _rerecord_leaving(*FROZEN_FILES),
    )
    for name in FROZEN_FILES:
        assert old in (repo / name).read_text(), name
        assert NEW not in (repo / name).read_text(), name
    found = T20E.run(repo, checkpoint).problems
    assert found == [], '\n'.join(found)
    git(repo, 'reset', '-q', '--hard', checkpoint)
    return checkpoint


@pytest.mark.gate_tooling
def test_the_frozen_files_keep_the_old_sha_after_an_approved_rerecord(
    repo: Path,
) -> None:
    r"""The decision: an approved re-record leaving the two frozen goldens
    naming the old SHA passes."""
    _twin(repo)


@pytest.mark.gate_tooling
@pytest.mark.parametrize('left', [TABLE_FILE, PROSE_FILE])
def test_any_other_file_left_at_the_old_sha_is_refused_by_name(
    repo: Path, left: str
) -> None:
    r"""Seeded, on the twin: a citing file not frozen, one the consistency
    tests read or a provenance file outside the table, left naming the
    old SHA, is refused, naming it."""
    checkpoint = _twin(repo)
    old = T20E.old_sha(repo)
    commit(
        repo,
        f'T2.0c: {left} left behind',
        _rerecord_leaving(*FROZEN_FILES, left),
    )
    assert old in (repo / left).read_text()
    found = T20E.run(repo, checkpoint).problems
    assert any(left in p for p in found), found
    assert not any(name in p for p in found for name in FROZEN_FILES), found


def test_the_frozen_list_is_the_two_goldens_each_with_a_reason() -> None:
    r"""The frozen list is exactly the decision's two files, each with a
    reason, each in this checkout, none in the citation table the
    consistency tests' fields are read from."""
    frozen = _frozen()
    assert sorted(frozen) == sorted(FROZEN_FILES)
    for name, reason in frozen.items():
        assert isinstance(reason, str) and reason.strip(), name
        assert (ROOT / name).is_file(), name
    assert not set(frozen) & set(ratchet.CITATIONS)


def test_the_frozen_list_can_t_hide_a_file_the_consistency_tests_read(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""Seeded: a file of the citation table added to the frozen list and
    left naming the old SHA after an approved re-record is still refused,
    naming it: the table's check comes first."""
    assert TABLE_FILE in ratchet.CITATIONS
    checkpoint = _twin(repo)
    monkeypatch.setitem(_frozen(), TABLE_FILE, 'seeded: frozen by mistake')
    commit(
        repo,
        f'T2.0c: {TABLE_FILE} left behind',
        _rerecord_leaving(*FROZEN_FILES, TABLE_FILE),
    )
    found = T20E.run(repo, checkpoint).problems
    assert any(TABLE_FILE in p for p in found), found
