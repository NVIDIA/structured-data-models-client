# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1q: ``ratchet.py``'s ``moves`` reports the same moves whether the
checkpoint's and this change's copies of a reference line are one object,
two equal values, or two values that differ only where the semantic
projection never looks (identity, a statement's text); and a line that
changes semantically is still judged as re-recorded, from its new stop,
with only this change's entries counting.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    't21q_ratchet', ROOT / 'scripts/rust_core/ratchet.py'
)
assert _spec is not None and _spec.loader is not None
ratchet = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ratchet
_spec.loader.exec_module(ratchet)

CASE = 'decimal.equal-copies'
RUNNER = 'python-core'


def _line(stage: str, code: str, identity: str, sql: str) -> dict[str, Any]:
    return {
        'case': CASE,
        'identity': {'recorded': identity},
        'outcome': {
            'result': 'refusal',
            'stage': stage,
            'error': 'StructuredDataError',
            'code': code,
        },
        'driver': {
            'capture': {'statements': [{'stage': 'from_source', 'sql': sql}]}
        },
    }


LINE = _line('materialize', 'INVALID_REQUEST', 'run-1', 'SELECT 1')
ANSWERED = {
    'outcome.result': 'success',
    'outcome.stage': 'predict',
    'outcome.error': None,
    'outcome.code': None,
}
PROJECTED = {CASE: {RUNNER: ANSWERED}}


def _entry(old: str) -> dict[str, Any]:
    return {
        'id': 'P2E-1',
        'status': 'approved',
        'approved_by': 'owner',
        'case_outcomes': [
            {
                'case': CASE,
                'cell': ratchet.CELL,
                'runner': RUNNER,
                'old': old,
                'new': 'success at predict',
                'summary': 'more exact than the reference',
            }
        ],
    }


def _moves(
    before: dict[str, Any],
    after: dict[str, Any],
    entries: list[dict[str, Any]],
    this_change: list[dict[str, Any]],
) -> dict[str, list[str]]:
    return ratchet.moves(
        PROJECTED, entries, this_change, {CASE: before}, {CASE: after}, None
    )


def test_identical_and_equal_copies_report_the_same_moves() -> None:
    held = _entry(
        'refusal INVALID_REQUEST at materialize (StructuredDataError)'
    )
    unchanged = [
        LINE,  # the same object
        copy.deepcopy(LINE),  # an equal value
        # Equal in the semantic projection only.
        _line('materialize', 'INVALID_REQUEST', 'run-2', 'SELECT 2'),
    ]
    for now in unchanged:
        # Unchanged: an entry held since the checkpoint counts ...
        assert _moves(LINE, now, [held], []) == {CASE: [RUNNER]}
        # ... from the checkpoint's stop, not another one.
        stale = _entry(
            'refusal INVALID_REQUEST at predict (StructuredDataError)'
        )
        assert _moves(LINE, now, [stale], [stale]) == {}
    assert LINE == _line('materialize', 'INVALID_REQUEST', 'run-1', 'SELECT 1')


def test_a_semantic_change_is_still_judged_as_re_recorded() -> None:
    # The new line refuses at another stage with another code.
    now = _line('from_source', 'UNSUPPORTED', 'run-1', 'SELECT 1')
    moved = _entry('refusal UNSUPPORTED at from_source (StructuredDataError)')
    # Only an entry this change adds or changes counts ...
    assert _moves(LINE, now, [moved], [moved]) == {CASE: [RUNNER]}
    assert _moves(LINE, now, [moved], []) == {}
    # ... and from the new line's stop, not the checkpoint's.
    held = _entry(
        'refusal INVALID_REQUEST at materialize (StructuredDataError)'
    )
    assert _moves(LINE, now, [held], [held]) == {}
