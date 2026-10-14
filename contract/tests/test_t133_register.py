# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.33's register reconciliation, and the owner's answers it records.

Written by the package's independent test author, before the change.

    python -m pytest contract/tests/test_t133_register.py -q

Exits of T1.33 (phase-1 design.md's T1.33 row and tracker.md's;
design.md "Register v2"; H19), held on ``docs/design/rust-core/
register.yaml`` and ``phase-1/decisions.md``:

- "no proposed entry left without its approver": every entry still
  ``proposed`` carries ``approver``, ``owner`` or ``reviewer``, who must
  approve it (the register's ``approved_by`` stays null until then);
- "every changed case's entry complete": every case flagged ``register``
  (a case whose answer the core changes) is named, in ``cases`` or
  ``case_outcomes``, by an entry that is not rejected and is not one of
  the umbrellas P1B-1 to P1B-4;
- "P1B-1 replaced case by case": each of P1B-1's cases is named by an
  approved entry of its own;
- the owner's answers of 2026-10-13 night
  (the owner's written answers, items 8 to 15 and 17): P1B-36, P1B-5
  and P1B-6 approved; DT-14 approved; P1B-1 to P1B-4 closed as superseded
  (per-case entries and the Phase 2 class, E92), which the register says
  with status ``approved`` or ``superseded``, ``approved_by: owner`` and a
  note saying superseded; C9 and H11 confirmed as written; DT-7 not an
  owner item (the name ``text_times`` kept), so its approver is the
  reviewer; ORD-1 approved.

Gating (``contract/tests/t133p2_gate.py``): a test named in
``t133p2-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``)
failed when written. It still runs and must still fail; once it passes it
fails, naming itself, until it is taken out of the list.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'
DECISIONS = ROOT / 'docs/design/rust-core/phase-1/decisions.md'
CASES = ROOT / 'contract/cases'
#: The umbrella entries the owner closed as superseded.
UMBRELLAS = ('P1B-1', 'P1B-2', 'P1B-3', 'P1B-4')
APPROVERS = ('owner', 'reviewer')


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


G = _module('t133p2_gate', ROOT / 'contract/tests/t133p2_gate.py')


def _entries() -> list[dict[str, Any]]:
    loaded = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))
    entries: list[dict[str, Any]] = loaded['entries']
    return entries


def _entry(entry_id: str) -> dict[str, Any]:
    for entry in _entries():
        if entry['id'] == entry_id:
            return entry
    raise AssertionError(f'register: no entry {entry_id}')


def _named(entry: dict[str, Any]) -> set[str]:
    r"""The cases an entry names, in ``cases`` or ``case_outcomes``."""
    names = {str(c) for c in entry.get('cases') or []}
    names |= {str(o.get('case')) for o in entry.get('case_outcomes') or []}
    return names


def _cases() -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding='utf-8'))
        for path in sorted(CASES.rglob('*.json'))
    ]


def _held(name: str, check: Any) -> None:
    G.held('exits', f'py:{name}', check)


# --- completeness ----------------------------------------------------------------


def test_no_entry_is_proposed_without_its_approver() -> None:
    def check() -> None:
        missing = [
            f'{e["id"]}: approver {e.get("approver")!r}'
            for e in _entries()
            if e['status'] == 'proposed' and e.get('approver') not in APPROVERS
        ]
        assert missing == [], missing

    _held('test_no_entry_is_proposed_without_its_approver', check)


def test_every_changed_case_is_named_by_an_entry_of_its_own() -> None:
    entries = [
        e
        for e in _entries()
        if e['status'] != 'rejected' and e['id'] not in UMBRELLAS
    ]
    flagged = [c['id'] for c in _cases() if c.get('register')]
    assert flagged, 'no case is flagged register'
    unnamed = [
        case for case in flagged if not any(case in _named(e) for e in entries)
    ]
    assert unnamed == [], unnamed


def test_p1b_1_is_replaced_case_by_case() -> None:
    def check() -> None:
        cases = _entry('P1B-1')['cases']
        assert cases
        approved = [
            e
            for e in _entries()
            if e['status'] == 'approved' and e['id'] not in UMBRELLAS
        ]
        alone = [c for c in cases if not any(c in _named(e) for e in approved)]
        assert alone == [], f'P1B-1 cases no approved entry names: {alone}'

    _held('test_p1b_1_is_replaced_case_by_case', check)


# --- the owner's answers of 2026-10-13 night ----------------------------------------

#: Entries the owner approved (items 8, 9, 10, 11 and 17).
APPROVED_BY_OWNER = ['P1B-36', 'P1B-5', 'P1B-6', 'DT-14', 'ORD-1']


@pytest.mark.parametrize('entry_id', APPROVED_BY_OWNER)
def test_the_owner_approved_it(entry_id: str) -> None:
    def check() -> None:
        entry = _entry(entry_id)
        assert (entry['status'], entry['approved_by']) == (
            'approved',
            'owner',
        ), f'{entry_id}: {entry["status"]} by {entry["approved_by"]}'

    _held(f'test_the_owner_approved_it[{entry_id}]', check)


@pytest.mark.parametrize('entry_id', UMBRELLAS)
def test_the_owner_closed_the_umbrella_as_superseded(entry_id: str) -> None:
    def check() -> None:
        entry = _entry(entry_id)
        assert entry['status'] in ('approved', 'superseded'), entry['status']
        assert entry['approved_by'] == 'owner'
        assert 'supersed' in ' '.join(str(entry.get('notes')).split()).lower()

    _held(
        f'test_the_owner_closed_the_umbrella_as_superseded[{entry_id}]', check
    )


def test_dt_7_is_the_reviewer_s() -> None:
    def check() -> None:
        entry = _entry('DT-7')
        assert entry['status'] in ('proposed', 'approved'), entry['status']
        if entry['status'] == 'proposed':
            assert entry.get('approver') == 'reviewer'
        else:
            assert entry['approved_by'] == 'reviewer'

    _held('test_dt_7_is_the_reviewer_s', check)


def _decision(row: str) -> list[str]:
    r"""A decisions.md row's cells: decision, reason, status."""
    for line in DECISIONS.read_text(encoding='utf-8').splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if cells and cells[0] == row:
            return cells[1:]
    raise AssertionError(f'decisions.md: no row {row}')


@pytest.mark.parametrize('row', ['C9', 'H11'])
def test_the_owner_confirmed_the_row_as_written(row: str) -> None:
    def check() -> None:
        status = _decision(row)[-1]
        assert 'to confirm' not in status, status
        assert 'owner' in status and '2026-10-13' in status, status

    _held(f'test_the_owner_confirmed_the_row_as_written[{row}]', check)
