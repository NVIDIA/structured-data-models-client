# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P2.fixtures: at least 300 language-neutral fixture cases, all passing.

A runner that passes is only evidence if it fails on a wrong expectation, so
one case of every operation is copied with its expectation changed, and each
copy must fail.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from _support import (
    PARITY,
    REPO,
    changed,
    deep,
    get,
    leaves,
    put,
    ran,
    run_junit,
)

FIXTURES = PARITY / 'fixtures'
RUNNER = PARITY / 'test_fixtures.py'
ID = re.compile(r'^[A-Za-z0-9_./-]+$')
SOURCES = (
    'packages/sdm-client/tests/test_wire_integer_exactness.py',
    'packages/sdm-client/tests/test_inferred_time_and_keys.py',
    'packages/sdm-client/tests/test_inferred_relationships.py',
    'packages/sdm-client/tests/test_decomposition_is_faithful.py',
    'packages/sdm-relational-engine/tests/test_json_payload.py',
    'packages/sdm-relational-engine/tests/test_payload_dtype_fidelity.py',
    'packages/sdm-relational-engine/tests/test_composite_primary_key.py',
    'packages/sdm-relational-engine/tests/query/test_pandas_executor.py',
    'packages/sdm-client/tests/test_dates_the_column_decides.py',
    'packages/sdm-client/tests/test_a_binary_target_is_a_classification.py',
)


def cases(root: Path = FIXTURES) -> list[dict[str, Any]]:
    out = []
    for path in sorted(root.rglob('*.json')):
        document = json.loads(path.read_text())
        assert isinstance(document, dict), path
        assert isinstance(document['cases'], list), path
        out += document['cases']
    return out


def junit(
    tmp_path: Path, env: dict[str, str] | None = None
) -> tuple[int, dict[str, str], str]:
    return run_junit(RUNNER, tmp_path / 'junit.xml', env)


def test_case_shape_and_count() -> None:
    all_cases = cases()
    ids = [case['id'] for case in all_cases]
    assert len(ids) == len(set(ids)), 'case ids are not unique'
    for case in all_cases:
        assert ID.match(case['id']), case['id']
        assert isinstance(case['operation'], str), case['id']
        assert 'input' in case, case['id']
        assert ('expected' in case) != ('expected_error' in case), case['id']
        if 'expected_error' in case:
            assert isinstance(case['expected_error']['code'], str), case['id']
            assert case['expected_error']['code'], case['id']
        assert (REPO / case['source']).is_file(), case['source']
    distinct = {
        json.dumps([c['operation'], c['input']], sort_keys=True)
        for c in all_cases
    }
    assert len(distinct) >= 300, f'{len(distinct)} distinct cases, need 300'


def test_every_named_test_file_contributes() -> None:
    sources = {case['source'] for case in cases()}
    missing = [s for s in SOURCES if s not in sources]
    assert not missing, missing


def test_every_case_passes_as_its_own_test(tmp_path: Path) -> None:
    code, outcomes, tail = junit(tmp_path)
    assert code == 0, tail
    assert outcomes, 'the runner ran nothing'
    bad = {n: o for n, o in outcomes.items() if o != 'passed'}
    assert not bad, f'{len(bad)} not passing, e.g. {list(bad.items())[:10]}'
    unrun = [c['id'] for c in cases() if ran(outcomes, c['id']) != ['passed']]
    assert not unrun, f'cases without exactly one passing test: {unrun[:10]}'


def _broken(case: dict[str, Any]) -> dict[str, Any]:
    case = deep(case)
    case['id'] = f'{case["id"]}.mutated'
    if 'expected_error' in case:
        case['expected_error']['code'] += '_MUTATED'
        return case
    expected = case['expected']
    paths = leaves(expected)
    if paths == [[]]:
        case['expected'] = changed(expected)
    elif paths:
        put(expected, paths[0], changed(get(expected, paths[0])))
    elif isinstance(expected, list):
        expected.append('mutated')
    else:
        expected['mutated'] = 'mutated'
    return case


def test_a_wrong_expectation_fails(tmp_path: Path) -> None:
    by_operation: dict[tuple[str, bool], dict[str, Any]] = {}
    for case in cases():
        key = (case['operation'], 'expected_error' in case)
        by_operation.setdefault(key, case)
    broken = [_broken(case) for case in by_operation.values()]
    scratch = tmp_path / 'fixtures'
    scratch.mkdir()
    (scratch / 'broken.json').write_text(json.dumps({'cases': broken}))
    env = dict(os.environ, SDM_PARITY_FIXTURES_DIR=str(scratch))
    code, outcomes, tail = junit(tmp_path, env)
    assert code != 0, tail
    for case in broken:
        outcome = ran(outcomes, case['id'])
        assert outcome, f'{case["id"]} did not run\n{tail}'
        assert all(o in ('failure', 'error') for o in outcome), (
            f'{case["id"]} passed with a wrong expectation'
        )
