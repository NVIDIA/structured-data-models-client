# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P2.pql: at least 150 valid and 70 invalid PQL cases, all passing.

Each case pairs a query with a schema. Counts are of distinct inputs, so a
case copied under another id doesn't count twice. Copies with a
changed tree, code or span must fail, so the runner is known to compare all
three.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from _support import PARITY, changed, deep, get, leaves, put, ran, run_junit

CORPUS = PARITY / 'pql'
RUNNER = PARITY / 'test_pql_corpus.py'
ID = re.compile(r'^[A-Za-z0-9_./-]+$')
SPAN_KEYS = {'start_row', 'start_col', 'end_row', 'end_col'}


def cases() -> list[dict[str, Any]]:
    out = []
    for path in sorted(CORPUS.rglob('*.json')):
        document = json.loads(path.read_text())
        assert isinstance(document, dict), path
        assert isinstance(document['cases'], list), path
        out += document['cases']
    return out


def errors(case: dict[str, Any]) -> list[dict[str, Any]]:
    if 'expected_error' in case:
        return [case['expected_error']]
    return list(case.get('expected_errors', []))


_OUTCOME = ('id', 'expected_tree', 'expected_error', 'expected_errors')


def _pair(case: dict[str, Any]) -> str:
    r"""The case's inputs: query, schema and anything else but the outcome."""
    inputs = {k: v for k, v in case.items() if k not in _OUTCOME}
    return json.dumps(inputs, sort_keys=True)


def test_case_shape() -> None:
    all_cases = cases()
    ids = [case['id'] for case in all_cases]
    assert len(ids) == len(set(ids)), 'case ids are not unique'
    for case in all_cases:
        assert ID.match(case['id']), case['id']
        assert isinstance(case['query'], str) and case['query'], case['id']
        assert 'schema' in case, case['id']
        valid = 'expected_tree' in case
        assert valid != bool(errors(case)), case['id']
        if valid:
            assert isinstance(case['expected_tree'], (dict, list)), case['id']
        for error in errors(case):
            assert isinstance(error['code'], str) and error['code'], case['id']
            span = error['span']
            if span is not None:
                assert set(span) == SPAN_KEYS, case['id']
                assert all(type(span[k]) is int for k in SPAN_KEYS), case['id']


def test_counts() -> None:
    all_cases = cases()
    valid = {_pair(c) for c in all_cases if 'expected_tree' in c}
    invalid = {_pair(c) for c in all_cases if errors(c)}
    assert len(valid) >= 150, f'{len(valid)} distinct valid cases'
    assert len(invalid) >= 70, f'{len(invalid)} distinct invalid cases'
    assert not valid & invalid, 'a query is both valid and invalid'
    with_span = [c for c in all_cases if any(e['span'] for e in errors(c))]
    assert len(with_span) >= 10, (
        f'only {len(with_span)} invalid cases record a span'
    )


def test_every_case_passes_as_its_own_test(tmp_path: Path) -> None:
    code, outcomes, tail = run_junit(RUNNER, tmp_path / 'junit.xml')
    assert code == 0, tail
    bad = {n: o for n, o in outcomes.items() if o != 'passed'}
    assert not bad, f'{len(bad)} not passing, e.g. {list(bad.items())[:10]}'
    unrun = [c['id'] for c in cases() if ran(outcomes, c['id']) != ['passed']]
    assert not unrun, f'cases without exactly one passing test: {unrun[:10]}'


def _broken() -> list[dict[str, Any]]:
    all_cases = cases()
    out = []
    valid = [c for c in all_cases if 'expected_tree' in c]
    for case in valid[:: max(1, len(valid) // 20)]:
        case = deep(case)
        case['id'] += '.tree'
        tree = case['expected_tree']
        path = leaves(tree)[0]
        put(tree, path, changed(get(tree, path)))
        out.append(case)
    seen_codes = set()
    spans = 0
    for case in all_cases:
        if not errors(case):
            continue
        code = errors(case)[0]['code']
        if code not in seen_codes:
            seen_codes.add(code)
            broken = deep(case)
            broken['id'] += '.code'
            errors(broken)[0]['code'] += '_MUTATED'
            out.append(broken)
        if errors(case)[0]['span'] and spans < 10:
            spans += 1
            broken = deep(case)
            broken['id'] += '.span'
            errors(broken)[0]['span']['end_col'] += 1
            out.append(broken)
    return out


def test_a_wrong_tree_code_or_span_fails(tmp_path: Path) -> None:
    broken = _broken()
    scratch = tmp_path / 'pql'
    scratch.mkdir()
    (scratch / 'broken.json').write_text(json.dumps({'cases': broken}))
    env = dict(os.environ, SDM_PARITY_PQL_DIR=str(scratch))
    code, outcomes, tail = run_junit(RUNNER, tmp_path / 'junit.xml', env)
    assert code != 0, tail
    for case in broken:
        outcome = ran(outcomes, case['id'])
        assert outcome, f'{case["id"]} did not run\n{tail}'
        assert all(o in ('failure', 'error') for o in outcome), (
            f'{case["id"]} passed with a wrong expectation'
        )
