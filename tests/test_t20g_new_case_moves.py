# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.0g: a case new in a change, whose reference line refuses and whose
core answer an approved entry moves from ``none`` past that refusal, counts
as moved past the reference's stop (``ratchet.py``'s ``moves``), as T2.0c's
``decimal.sum-past-38`` and ``decimal.class-target-collide`` need.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    't20g_ratchet', ROOT / 'scripts/rust_core/ratchet.py'
)
assert _spec is not None and _spec.loader is not None
ratchet = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ratchet
_spec.loader.exec_module(ratchet)

CASE = 'decimal.new-case'
REFUSED = {
    'case': CASE,
    'outcome': {
        'result': 'refusal',
        'stage': 'materialize',
        'error': 'StructuredDataError',
        'code': 'INVALID_REQUEST',
    },
}
ANSWERED = {
    'outcome.result': 'success',
    'outcome.stage': 'predict',
    'outcome.error': None,
    'outcome.code': None,
}
ENTRY = {
    'id': 'P2E-1',
    'status': 'approved',
    'approved_by': 'owner',
    'case_outcomes': [
        {
            'case': CASE,
            'cell': ratchet.CELL,
            'runner': 'python-core',
            'old': 'none',
            'new': 'success at predict',
            'summary': 'more exact than the reference',
        }
    ],
}


def test_a_new_case_moved_by_this_change_s_entry_counts_as_moved() -> None:
    projected = {CASE: {'python-core': ANSWERED}}
    # New in this change: no line at the checkpoint, its new line refuses.
    found = ratchet.moves(
        projected, [ENTRY], [ENTRY], {}, {CASE: REFUSED}, None
    )
    assert found == {CASE: ['python-core']}
    # Only an entry this change adds or changes moves a new case.
    assert (
        ratchet.moves(projected, [ENTRY], [], {}, {CASE: REFUSED}, None) == {}
    )
