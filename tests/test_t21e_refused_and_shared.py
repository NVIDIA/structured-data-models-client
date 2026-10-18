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

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

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
    # The contract versions are held whichever operations each ran.
    newer = _host(['describe_source', 'materialize'], error)
    newer['identity']['contract'] = 27
    assert any(
        'identity.contract' in t for t in _agree(python, newer, 'materialize')
    )


REFERENCE = (
    ROOT / 'contract/reference/6298913c3b66da9459bb605d0127f8493adaa1fc.jsonl'
)


def _reference_line() -> dict[str, Any]:
    for line in REFERENCE.read_text(encoding='utf-8').splitlines():
        observation = json.loads(line)
        if observation['case'] == KEY:
            return dict(observation)
    raise AssertionError(KEY)


def _compare_refused(
    tmp: Path, opted_in: bool, from_source_changed: bool = False
) -> list[tuple[str | None, str]]:
    r"""``compare()`` on python-core refused at materialize where the
    reference answered ``KEY``, the effective expectation that refusal
    (an owner-approved move), its register's row naming materialize under
    ``statements_differ`` or not; the line's gating failures."""
    c = _compare()
    reference = _reference_line()
    mine = copy.deepcopy(reference)
    mine['identity']['runner'] = 'python-core'
    refusal = {
        'code': 'UNSUPPORTED_REQUEST',
        'error': 'UnsupportedRequestError',
        'result': 'refusal',
        'stage': 'materialize',
    }
    mine['outcome'] = dict(refusal)
    mine['host']['warnings'] = []
    statements = mine['driver']['capture']['statements']
    kept = [s for s in statements if s['stage'] == 'from_source']
    kept += [s for s in statements if s['stage'] == 'materialize'][:2]
    if from_source_changed:
        values = kept[0]['results'][0]['values']
        values['ordered'] = values['unordered'] = '0' * 64
    mine['driver']['capture']['statements'] = kept
    (tmp / 'ref.jsonl').write_text(json.dumps(reference) + '\n')
    (tmp / 'core.jsonl').write_text(json.dumps(mine) + '\n')
    expect = {KEY: {'python-core': refusal}, c.MOVED: {KEY: ['python-core']}}
    (tmp / 'expect.json').write_text(json.dumps(expect))
    row: dict[str, Any] = {
        'case': KEY,
        'cell': 'ref-pandas3',
        'runner': 'python-core',
        'old': 'success at materialize',
        'new': 'refusal UNSUPPORTED_REQUEST at materialize (UnsupportedRequestError)',
        'summary': 'the wiring test',
    }
    if opted_in:
        row['statements_differ'] = ['materialize']
    register = {
        'format': 'sdm-register/2',
        'entries': [
            {
                'id': 'X-1',
                'status': 'approved',
                'approved_by': 'owner',
                'decision_ref': {
                    'file': 'docs/design/rust-core/phase-2/decisions.md',
                    'row': 'KO2-56',
                },
                'case_outcomes': [row],
            }
        ],
    }
    (tmp / 'register.yaml').write_text(json.dumps(register))
    report = c.compare(
        [ROOT / 'contract/cases'],
        tmp / 'ref.jsonl',
        [tmp / 'core.jsonl'],
        expect_file=tmp / 'expect.json',
        register_file=tmp / 'register.yaml',
        database='sqlite',
    )
    return [(rule, text) for rule, text in report.failures if KEY in text]


def test_compare_wires_the_refusal_opt_in_to_the_refused_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The record's completeness is its own rule; here only the wiring.
    monkeypatch.setattr(_compare(), '_complete', lambda *_: True)

    def ruled(failures: list[tuple[str | None, str]], rule: str) -> list[str]:
        return [text for r, text in failures if r == rule]

    held = _compare_refused(tmp_path, opted_in=False)
    # Without the opt-in the refused call's statements and the warnings gate.
    assert any('materialize' in t for t in ruled(held, 'sql.count')), held
    assert ruled(held, 'warnings'), held
    opted = _compare_refused(tmp_path, opted_in=True)
    # With it, the reference's statements are held only through that call,
    # its own reported, and the warnings demoted.
    assert not any('materialize' in t for t in ruled(opted, 'sql.count')), opted
    assert not any('materialize' in t for t in ruled(opted, 'sql.result'))
    assert not ruled(opted, 'warnings'), opted
    # An earlier call still gates.
    earlier = _compare_refused(
        tmp_path, opted_in=True, from_source_changed=True
    )
    assert any('from_source' in t for t in ruled(earlier, 'sql.result')), (
        earlier
    )
