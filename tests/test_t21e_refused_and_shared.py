# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1e: an approved refusal where the reference answered, and the two
core hosts on a database line that stops early.

- ``statements_differ`` on an owner-approved row moving the reference's
  answer to the runner's refusal at a call (the owner's KO2-56 on
  P2B-16's s0.boundary.*@sqlite lines) holds the reference's statements
  only through that call, reports that call's ``sql.count`` and
  ``sql.result`` as diagnostic, and the line's ``warnings``; without the
  opt-in, or on another call, nothing changes.
- On a database line stopping at a call, where python-core was refused
  there by the client's check and rust-host's core went on into that
  call, the ``core`` rule holds the two on the operations both ran; a
  difference there still fails, and DuckDB's lines are held whole.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KEY = 's0.boundary.deepest@sqlite'


def _module(name: str, path: str) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _compare() -> ModuleType:
    return _module('t21e_compare_refused', 'scripts/rust_core/compare.py')


def test_an_approved_refusal_opts_in_its_call_alone() -> None:
    c = _compare()
    report = c.Report()
    report.statements_differ = {(KEY, 'python-core'): ('materialize',)}
    answered = {'outcome': {'result': 'success', 'stage': 'materialize'}}
    refused = {'result': 'refusal', 'stage': 'materialize'}
    refused_at = c.statements_refused
    assert refused_at(report, KEY, 'python-core', refused, answered) == (
        'materialize'
    )
    # rust-host's expectation names no stage: the row's one call.
    report.statements_differ[(KEY, 'rust-host')] = ('materialize',)
    assert refused_at(report, KEY, 'rust-host', {'result': 'refusal'}, answered)
    # Another call, no opt-in, a reference that refused, an answer: none.
    other = {'result': 'refusal', 'stage': 'predict'}
    assert refused_at(report, KEY, 'python-core', other, answered) is None
    assert refused_at(c.Report(), KEY, 'python-core', refused, answered) is None
    refusing = {'outcome': {'result': 'refusal', 'stage': 'materialize'}}
    assert refused_at(report, KEY, 'python-core', refused, refusing) is None
    assert (
        refused_at(report, KEY, 'python-core', {'result': 'success'}, answered)
        is None
    )
    r = _module('t21e_register_refused', 'scripts/rust_core/register.py')
    entry = {'id': 'X-1', 'status': 'approved', 'approved_by': 'owner'}
    row = {
        'case': KEY,
        'runner': 'python-core',
        'old': 'success at materialize',
        'new': 'refusal UNSUPPORTED_REQUEST at materialize (UnsupportedRequestError)',
        'statements_differ': ['materialize'],
    }
    corpus = {'s0.boundary.deepest': {}}
    assert r._outcome_problems(entry, row, corpus, {}) == []
    wrong = {**row, 'statements_differ': ['predict']}
    assert r._outcome_problems(entry, wrong, corpus, {})


def _host(operations: list[str], core: dict[str, Any]) -> dict[str, Any]:
    return {
        'identity': {'operations': operations, 'contract': 26},
        'core': {
            'inputs': [{'op': op} for op in operations],
            'fetches': [{'request': op} for op in operations],
            'described': {'tables': []},
            'accepted': None,
            'materialized': None,
            'predicted': None,
            'error': None,
            **core,
        },
    }


def _agree(
    python: dict[str, Any], rust: dict[str, Any], stop: str | None
) -> list[str]:
    c = _compare()
    report = c.Report()
    c._core_agree(report, {'id': 'decimal.sum-10-2@sqlite'}, python, rust, stop)
    return [text for _, text in report.failures]


def test_the_hosts_are_held_on_the_operations_both_ran() -> None:
    error = {'error': {'kind': 'invalid_request', 'code': 'INVALID_REQUEST'}}
    python = _host(['describe_source'], {})
    rust = _host(['describe_source', 'materialize'], error)
    assert _agree(python, rust, 'materialize') == []
    # A difference in what both ran still fails.
    other = _host(
        ['describe_source', 'materialize'], {**error, 'described': {}}
    )
    assert _agree(python, other, 'materialize')
    # The extra operation must be of the stopping call or later.
    assert _agree(python, rust, 'from_source') == []
    assert _agree(
        _host(['validate_query'], {}),
        _host(['validate_query', 'describe_source'], {}),
        'materialize',
    )
    # DuckDB's lines, and a line that answers, are held whole.
    assert _agree(python, rust, None)
