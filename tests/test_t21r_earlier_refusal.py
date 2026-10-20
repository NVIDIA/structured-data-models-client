# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1r (added scope): an owner-approved refusal at an earlier call.

The reference may answer a call and refuse at a later one, where the core
refuses earlier with its own code: T2.10's target.aggregation.list-distinct
(register P2E-19, KO2-75) materializes with 8 statements on the reference
and is refused at predict, while the core refuses UNSUPPORTED_REQUEST at
materialize. An owner-approved ``case_outcomes`` row moving the old
refusal to that earlier call, naming it under ``statements_differ``,
passes register.py, and compare.py then holds the line only up to that
call, reporting none of the reference's later calls. A reviewer-approved
row doing the same is refused by name; a refusal moved to a later call,
and a ``statements_differ`` naming another call, still fail.

T2.10's case is not in this corpus yet, so the line here is the
reference's multicall.fail-call-0 (DuckDB and SQLite), which materializes
and is refused at predict as that case is.
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
CASE = 'multicall.fail-call-0'
REFERENCE = (
    ROOT / 'contract/reference/6298913c3b66da9459bb605d0127f8493adaa1fc.jsonl'
)
OLD = 'refusal - at predict (NimRequestError)'
NEW = 'refusal UNSUPPORTED_REQUEST at materialize (UnsupportedRequestError)'
REFUSAL = {
    'code': 'UNSUPPORTED_REQUEST',
    'error': 'UnsupportedRequestError',
    'result': 'refusal',
    'stage': 'materialize',
}


def _module(name: str, path: str) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _compare() -> ModuleType:
    return _module('t21r_compare_earlier', 'scripts/rust_core/compare.py')


def _register() -> ModuleType:
    return _module('t21r_register_earlier', 'scripts/rust_core/register.py')


def _row(**changes: Any) -> dict[str, Any]:
    return {
        'case': CASE,
        'cell': 'ref-pandas3',
        'runner': 'python-core',
        'old': OLD,
        'new': NEW,
        'summary': 'the earlier-refusal test',
        'statements_differ': ['materialize'],
        **changes,
    }


def _entry(approved_by: str) -> dict[str, Any]:
    return {'id': 'X-1', 'status': 'approved', 'approved_by': approved_by}


def _problems(approved_by: str, row: dict[str, Any]) -> list[str]:
    found: list[str] = _register()._outcome_problems(
        _entry(approved_by), row, {CASE: {}}, {}
    )
    return found


def test_register_takes_an_owner_s_earlier_refusal() -> None:
    r"""Exit: an owner-approved row moving the refusal earlier passes."""
    assert _problems('owner', _row()) == []


def test_register_refuses_a_reviewer_s_earlier_refusal_by_name() -> None:
    r"""Exit: the same row approved by a reviewer is refused, naming the
    owner."""
    found = _problems('reviewer', _row())
    assert found
    assert any('owner' in text for text in found), found


def test_register_still_refuses_a_later_move_or_another_call() -> None:
    r"""Exit: nothing else loosens, for the owner's entry too."""
    # The core's refusal at a later call than the reference's.
    later = _row(
        old='refusal - at materialize (NimRequestError)',
        new='refusal UNSUPPORTED_REQUEST at predict (UnsupportedRequestError)',
    )
    for calls in (['materialize'], ['predict']):
        assert _problems('owner', {**later, 'statements_differ': calls})
    # The earlier refusal, naming another call than the one it moves to.
    for calls in (['predict'], ['from_source'], ['materialize', 'predict']):
        assert _problems('owner', _row(statements_differ=calls))


def _reference(key: str) -> dict[str, Any]:
    for line in REFERENCE.read_text(encoding='utf-8').splitlines():
        observation: dict[str, Any] = json.loads(line)
        if observation['case'] == key:
            return observation
    raise AssertionError(key)


def _refused_early(
    reference: dict[str, Any], key: str, changed: bool = False
) -> dict[str, Any]:
    r"""python-core's line refused at materialize, as the core refuses
    P2E-19's: the reference's from_source statements and its first two at
    materialize, no warnings, no predict call; its core described the
    source and refused materialize. ``changed``: its first from_source
    statement's result differs from the reference's."""
    case = json.loads(
        (ROOT / f'contract/cases/s1/{CASE}.json').read_text(encoding='utf-8')
    )
    dialect = key.partition('@')[2] or 'duckdb'
    core = {
        op: (
            {**spec, 'provider': {**spec['provider'], 'dialect': dialect}}
            if 'provider' in spec
            else spec
        )
        for op, spec in case['core'].items()
    }
    described: dict[str, Any] = {'tables': []}
    mine = copy.deepcopy(reference)
    mine['identity']['runner'] = 'python-core'
    mine['identity']['native_loaded'] = True
    mine['identity']['native'] = {'sha256': '0' * 64}
    mine['identity']['operations'] = ['describe_source', 'materialize']
    mine['core'] = {
        'inputs': [
            {'op': 'describe_source', **core['describe_source']},
            {
                'op': 'materialize',
                'query': case['input']['pql'],
                **core['materialize'],
                'schema': described,
            },
        ],
        'fetches': [{'request': {}, 'plan': {}}],
        'accepted': None,
        'described': described,
        'materialized': None,
        'predicted': None,
        'error': {'kind': 'unsupported', 'code': 'UNSUPPORTED_REQUEST'},
    }
    mine['outcome'] = dict(REFUSAL)
    mine['host']['warnings'] = []
    mine['public']['payloads'] = []
    statements = mine['driver']['capture']['statements']
    kept = [s for s in statements if s['stage'] == 'from_source']
    kept += [s for s in statements if s['stage'] == 'materialize'][:2]
    if changed:
        values = kept[0]['results'][0]['values']
        values['ordered'] = values['unordered'] = '0' * 64
    mine['driver']['capture']['statements'] = kept
    mine['driver']['capture']['bindings'] = [
        b
        for b in mine['driver']['capture'].get('bindings', [])
        if b['stage'] == 'from_source'
    ]
    return mine


def _gating(
    tmp: Path,
    database: str,
    approved_by: str = 'owner',
    calls: list[str] | None = None,
    changed: bool = False,
) -> list[tuple[str | None, str]]:
    r"""``compare()``'s gating failures on python-core's line refused at
    materialize where the reference materialized and was refused at
    predict, its register's owner (or reviewer) row moving it there and
    naming ``calls`` (materialize by default) under
    ``statements_differ``; ``changed`` as :func:`_refused_early`."""
    c = _compare()
    key = CASE if database == 'duckdb' else f'{CASE}@{database}'
    reference = _reference(key)
    mine = _refused_early(reference, key, changed)
    (tmp / 'ref.jsonl').write_text(json.dumps(reference) + '\n')
    (tmp / 'core.jsonl').write_text(json.dumps(mine) + '\n')
    expect = {key: {'python-core': REFUSAL}, c.MOVED: {key: ['python-core']}}
    (tmp / 'expect.json').write_text(json.dumps(expect))
    row = _row(case=key, statements_differ=calls or ['materialize'])
    register = {
        'format': 'sdm-register/2',
        'entries': [
            {
                **_entry(approved_by),
                'decision_ref': {
                    'file': 'docs/design/rust-core/phase-2/decisions.md',
                    'row': 'KO2-75',
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
        database=database,
    )
    return [
        (rule, text)
        for rule, text in report.failures
        if text.startswith(f'{key}: ')
    ]


@pytest.mark.parametrize('database', ['duckdb', 'sqlite'])
def test_compare_holds_an_earlier_refusal_up_to_its_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, database: str
) -> None:
    r"""Exit: with the owner's row the line is held up to materialize, and
    none of the reference's later calls (predict's route and inputs, the
    payload, materialize's statement count, the warnings) is reported."""
    # The record's completeness is its own rule; here only the wiring.
    monkeypatch.setattr(_compare(), '_complete', lambda *_: True)
    assert _gating(tmp_path, database) == []


@pytest.mark.parametrize('database', ['duckdb', 'sqlite'])
def test_compare_still_gates_without_the_owner_s_row_on_that_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, database: str
) -> None:
    r"""Exit: nothing else loosens: a reviewer's row, or the owner's naming
    another call, leaves the reference's statements at materialize gating,
    and a change to an earlier call's statements still gates."""
    monkeypatch.setattr(_compare(), '_complete', lambda *_: True)

    def counted(failures: list[tuple[str | None, str]]) -> bool:
        return any(
            rule == 'sql.count' and 'materialize' in text
            for rule, text in failures
        )

    assert counted(_gating(tmp_path, database, approved_by='reviewer'))
    for calls in (['predict'], ['from_source']):
        assert counted(_gating(tmp_path, database, calls=calls))
    # With the owner's row, the calls before materialize still gate.
    earlier = _gating(tmp_path, database, changed=True)
    assert any(
        rule == 'sql.result' and 'from_source' in text for rule, text in earlier
    ), earlier


def _cell_warnings(tmp: Path, opted_in: bool) -> list[str]:
    r"""``compare()``'s gating ``warnings`` failures on the DuckDB line in
    a cell other than the reference's (dbr17-pandas2), where python-core,
    refused at materialize, shows none of the warnings python-old shows
    (two UserWarnings, as T2.10's list-distinct line), its register's
    owner row naming materialize under ``statements_differ`` or naming no
    call."""
    c = _compare()
    key = CASE
    reference = _reference(key)
    reference['host']['warnings'] = ['UserWarning', 'UserWarning']
    mine = _refused_early(reference, key)
    mine['identity']['cell'] = 'dbr17-pandas2'
    mine['host']['warnings'] = []
    old = copy.deepcopy(reference)
    old['identity']['runner'] = 'python-old'
    old['identity']['cell'] = 'dbr17-pandas2'
    (tmp / 'ref.jsonl').write_text(json.dumps(reference) + '\n')
    (tmp / 'core.jsonl').write_text(json.dumps(mine) + '\n')
    (tmp / 'old.jsonl').write_text(json.dumps(old) + '\n')
    expect = {key: {'python-core': REFUSAL}, c.MOVED: {key: ['python-core']}}
    (tmp / 'expect.json').write_text(json.dumps(expect))
    row = _row(case=key)
    if not opted_in:
        del row['statements_differ']
    register = {
        'format': 'sdm-register/2',
        'entries': [
            {
                **_entry('owner'),
                'decision_ref': {
                    'file': 'docs/design/rust-core/phase-2/decisions.md',
                    'row': 'KO2-75',
                },
                'case_outcomes': [row],
            }
        ],
    }
    (tmp / 'register.yaml').write_text(json.dumps(register))
    report = c.compare(
        [ROOT / 'contract/cases'],
        tmp / 'ref.jsonl',
        [tmp / 'core.jsonl', tmp / 'old.jsonl'],
        expect_file=tmp / 'expect.json',
        register_file=tmp / 'register.yaml',
        database='duckdb',
    )
    return [
        text
        for rule, text in report.failures
        if rule == 'warnings' and text.startswith(f'{key}: ')
    ]


def test_an_earlier_refusal_s_warnings_against_python_old_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""Exit: in a cell other than the reference's (H1, python-core against
    python-old, as the dbr17-pandas2 and floor-pandas2.2 DuckDB jobs), the
    warnings of the call refused under the owner's row are reported, not
    gating; without the row they still gate."""
    monkeypatch.setattr(_compare(), '_complete', lambda *_: True)
    assert _cell_warnings(tmp_path, opted_in=False)
    assert _cell_warnings(tmp_path, opted_in=True) == []
