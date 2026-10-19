# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.12 (KO2-68): a refusal the core keeps at the reference's call, with
the reference's code, reading less before it.

Where an owner-approved entry's ``case_outcomes`` row keeps the reference's
refusal at a call and names that call under ``statements_differ`` (the
owner's KO2-68 on rule.composite-key-given, whose fold the core no longer
reads), that call's ``sql.count`` and ``sql.result`` against the reference
are reported as diagnostic; register.py takes the opt-in on such a row only
at that call and with one code, and compare.py only where the runner's
expectation refuses there as the reference does.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KEY = 'rule.composite-key-given'


def _load(name: str, path: str) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_a_kept_refusal_opts_in_only_its_own_call_and_code() -> None:
    c = _load('t212_compare', 'scripts/rust_core/compare.py')
    report = c.Report()
    report.statements_differ = {(KEY, 'rust-host'): ('materialize',)}
    reference = {
        'outcome': {
            'result': 'refusal',
            'stage': 'materialize',
            'code': 'INVALID_REQUEST',
            'error': 'StructuredDataError',
        }
    }
    kept = {'result': 'refusal', 'stage': None, 'code': 'INVALID_REQUEST'}
    assert c.statements_kept(report, KEY, 'rust-host', kept, reference) == (
        'materialize'
    )
    # No expectation of its own: the reference's refusal is the one held.
    assert c.statements_kept(report, KEY, 'rust-host', None, reference) == (
        'materialize'
    )
    for wrong in (
        {**kept, 'code': 'UNSUPPORTED_REQUEST'},
        {**kept, 'stage': 'predict'},
        {'result': 'success', 'stage': None, 'code': None},
    ):
        assert (
            c.statements_kept(report, KEY, 'rust-host', wrong, reference)
            is None
        )
    # A row naming no call, or another runner's, opts in nothing.
    assert (
        c.statements_kept(report, KEY, 'python-core', kept, reference) is None
    )
    report.statements_differ = {(KEY, 'rust-host'): ('from_source',)}
    assert c.statements_kept(report, KEY, 'rust-host', kept, reference) is None

    register = _load('t212_register', 'scripts/rust_core/register.py')
    entry = {'id': 'X-1', 'status': 'approved', 'approved_by': 'owner'}
    refused = 'refusal INVALID_REQUEST at materialize (StructuredDataError)'
    row: dict[str, Any] = {
        'case': KEY,
        'runner': 'python-core',
        'old': refused,
        'new': refused,
        'statements_differ': ['materialize'],
    }
    corpus: dict[str, Any] = {KEY: {}}
    assert register._outcome_problems(entry, row, corpus, {}) == []
    for wrong in (
        {**row, 'statements_differ': ['from_source']},
        {**row, 'new': 'refusal UNSUPPORTED_REQUEST at materialize (X)'},
        {**row, 'new': 'refusal INVALID_REQUEST at predict (X)'},
    ):
        assert register._outcome_problems(entry, wrong, corpus, {})
