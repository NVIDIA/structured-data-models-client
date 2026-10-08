# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P7.fixtures, labels half: every case of ``pandas-executor.json`` gives
its expected output through ``labels_op``, as it does through today's
Python.

``rust/crates/labels/tests/accept_fixtures.rs`` checks the same cases inside
the crate.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from _labels import REPO, native
from _sampling import fixture_problem, loads, python_fixture

FIXTURE = REPO / 'tests' / 'parity' / 'fixtures' / 'pandas-executor.json'
CASES = json.loads(FIXTURE.read_text(encoding='utf-8'))['cases']


def test_every_operation_is_covered() -> None:
    assert len(CASES) >= 57
    assert {c['operation'] for c in CASES} == {
        'labels.column',
        'labels.rel_op',
        'labels.member_op',
        'labels.bool_op',
        'labels.aggregation_type',
        'labels.aggregation',
    }


@pytest.mark.parametrize('case', CASES, ids=[c['id'] for c in CASES])
def test_pandas_executor_fixture(case: dict[str, Any]) -> None:
    today = python_fixture(case)
    assert fixture_problem(case, today) is None, fixture_problem(case, today)
    module = native('labels_op')
    got = loads(
        module.labels_op(case['operation'], json.dumps(case['input'])),
        'labels_op',
    )
    problem = fixture_problem(case, got)
    assert problem is None, problem


@pytest.mark.parametrize(
    'args',
    [
        ('labels.no_such_operation', '{}'),
        ('labels.column', 'not json'),
        ('labels.aggregation_type', '[]'),
    ],
)
def test_labels_op_refuses_what_it_cannot_read(args: tuple[str, str]) -> None:
    module = native('labels_op')
    with pytest.raises(ValueError):
        module.labels_op(*args)
