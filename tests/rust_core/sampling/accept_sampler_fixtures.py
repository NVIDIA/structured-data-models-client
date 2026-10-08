# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P7.fixtures, sampler half: task 8's new sampler fixtures exist, today's
Python still gives every expected output, and the Rust core gives it too,
through ``sampling_op``.

``rust/crates/sampling/tests/accept_fixtures.rs`` checks the same cases
inside the crate; ``../labels/accept_executor_fixtures.py`` checks
``pandas-executor.json``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from _sampling import (
    fixture_cases,
    fixture_files,
    fixture_problem,
    loads,
    native,
    python_fixture,
)

SOURCES = (
    'packages/sdm-relational-engine/tests/backend/local/test_local_sampler.py',
    'packages/sdm-relational-engine/tests/backend/local/test_local_graph_store.py',
)

CASES = fixture_cases('sampling.')


def test_the_sampler_fixtures_exist() -> None:
    assert len(CASES) >= 20, (
        f'{len(CASES)} sampling.* fixture cases, fewer than 20'
    )
    sources = {case['_source'] for case in CASES}
    missing = [s for s in SOURCES if s not in sources]
    assert not missing, f'no sampler fixtures from {missing}'
    for path in fixture_files():
        data = json.loads(path.read_text(encoding='utf-8'))
        operations = {case['operation'] for case in data['cases']}
        sampling = {op for op in operations if op.startswith('sampling.')}
        assert not sampling or sampling == operations, (
            f'{path.name} mixes sampling.* operations with {operations - sampling}'
        )
    assert len({case['id'] for case in CASES}) == len(CASES), 'repeated ids'


@pytest.mark.parametrize('case', CASES, ids=[c['id'] for c in CASES])
def test_sampler_fixture(case: dict[str, Any]) -> None:
    today = python_fixture(case)
    assert fixture_problem(case, today) is None, (
        f"today's Python no longer gives the fixture: {fixture_problem(case, today)}"
    )
    module = native('sampling_op')
    got = loads(
        module.sampling_op(case['operation'], json.dumps(case['input'])),
        'sampling_op',
    )
    problem = fixture_problem(case, got)
    assert problem is None, problem


@pytest.mark.parametrize(
    'args',
    [
        ('sampling.no_such_operation', '{}'),
        ('sampling.no_such_operation', 'not json'),
    ],
)
def test_sampling_op_refuses_what_it_cannot_read(args: tuple[str, str]) -> None:
    module = native('sampling_op')
    with pytest.raises(ValueError):
        module.sampling_op(*args)


def test_sampling_op_refuses_an_unreadable_input_of_every_operation() -> None:
    module = native('sampling_op')
    for operation in sorted({case['operation'] for case in CASES}):
        with pytest.raises(ValueError):
            module.sampling_op(operation, 'not json')
