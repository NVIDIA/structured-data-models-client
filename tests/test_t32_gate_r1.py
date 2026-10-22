# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T3.2 fix 5, Claude's round 1 on the coverage and ratchet readings:

- G12: ``tables: []`` names no table, as the client reads it (it chooses
  none and refuses), never every table;
- U4: the owner's approval is read by allowlist over its whole qualifier,
  every clause of every group after the date, so a later clause or group
  saying no voids it.

The gate's per-leg comparison is ``contract/tests/test_t32_gate_legs.py``.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/rust_core'))


def _module(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


COVERAGE = _module(ROOT / 'scripts/rust_core/coverage.py', 't32r1_coverage')
RATCHET = _module(ROOT / 'scripts/rust_core/ratchet.py', 't32r1_ratchet')


def test_an_empty_table_list_names_no_table() -> None:
    reference = next((ROOT / 'contract/reference').glob('*.jsonl'))
    line = next(
        json.loads(text)
        for text in reference.read_text(encoding='utf-8').splitlines()
        if text.strip() and json.loads(text)['case'] == 'base.binary'
    )
    case = json.loads(
        (ROOT / 'contract/cases/s1/base.binary.json').read_text(
            encoding='utf-8'
        )
    )
    assert COVERAGE.observed(case, line), 'the case claims its cells'
    empty = copy.deepcopy(case)
    empty['input'].setdefault('from_source', {})['tables'] = []
    assert COVERAGE.observed(empty, line) == set()


@pytest.mark.parametrize(
    ('status', 'approved'),
    [
        # Claude r1 P2-5's vectors: a later group or clause saying no.
        ('approved by owner 2026-10-22 (with KO3-17) (declined)', False),
        (
            'approved by owner 2026-10-22 (T3.4, on hold until the review)',
            False,
        ),
        (
            'approved by owner 2026-10-22 (recorded at T1.16; the owner said no)',
            False,
        ),
        (
            'approved by owner 2026-10-22 (relayed by the orchestrator) (maybe)',
            False,
        ),
        ('approved by owner 2026-10-22 (with KO3-17', False),
        # The forms the decisions files write still approve.
        ('approved by owner 2026-10-22 (with KO3-17)', True),
        ('approved by owner 2026-10-12; recorded at T1.16', True),
        ('approved by owner 2026-10-12, including the path', True),
        (
            'approved by owner 2026-10-14 (items 22, 23 and 27, relayed by the '
            "orchestrator; recorded at T1.36, never the implementer's approval)",
            True,
        ),
        ('approved by owner 2026-10-09 ((a) to (c))', True),
    ],
)
def test_the_whole_qualifier_is_read_by_allowlist(
    status: str, approved: bool
) -> None:
    assert RATCHET.owner_approved(status) is approved
