# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ratchet's reading of an owner's approval for a relaxation (T1.15 fix
round 1, a bug fix under E49): a checkpoint row authorizes relaxing
``compare-policy.yaml`` only when its status starts with "approved by
owner" and a date, followed by nothing, a parenthesis or a note that does
not withdraw it, the rule ``coverage.py``'s rule 8 shares
(``ratchet.owner_approved``); "not approved by owner ...", a rejection or an
approval left to confirm never authorizes. On ``tests/test_contract_ratchet.py``'s
temporary repository, whose helpers this reuses, and the post-landing
state: the candidate equal to its checkpoint, judged by this harness,
passes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if 'ratchet_tests' in sys.modules:
    tests = sys.modules['ratchet_tests']
else:
    spec = importlib.util.spec_from_file_location(
        'ratchet_tests', ROOT / 'tests/test_contract_ratchet.py'
    )
    assert spec is not None and spec.loader is not None
    tests = importlib.util.module_from_spec(spec)
    sys.modules['ratchet_tests'] = tests
    spec.loader.exec_module(tests)

origin, repo = tests.origin, tests.repo
POLICY = 'contract/compare-policy.yaml'

APPROVING = [
    'approved by owner 2026-10-12',
    'approved by owner 2026-10-12 (answer 3)',
    'approved by owner 2026-10-12; recorded at T1.15',
]
NOT_APPROVING = [
    'not approved by owner 2026-10-12',
    'rejected; not approved by owner 2026-10-12',
    'rejected by owner 2026-10-12',
    'approved by owner 2026-10-12; rejected by owner 2026-10-13',
    'approved by owner 2026-10-12, owner to confirm',
    'owner to confirm',
    'orchestrator, owner to confirm',
    'approved by owner 2026-10-12 later',
]


def _relax(document: dict[str, Any]) -> None:
    document['rules']['sql.result'] = 'diagnostic'
    document['relaxations'] = [
        {'change': 'rule sql.result: gate to diagnostic', 'decision': 'E900'}
    ]


def _relaxed(repo: Path, status: str) -> list[str]:
    checkpoint = tests.land(
        repo,
        'T1.15',
        lambda r: tests.owner_row(r, 'E900', 'compare-policy.yaml', status),
    )
    tests.commit(repo, 'T1.16: relaxed', tests.edit_yaml(POLICY, _relax))
    problems: list[str] = tests.check(repo, checkpoint)
    return problems


@pytest.mark.parametrize('status', APPROVING)
def test_an_affirmative_owner_approval_authorizes_the_relaxation(
    repo: Path, status: str
) -> None:
    assert 'policy' not in tests.rules(_relaxed(repo, status))


@pytest.mark.parametrize('status', NOT_APPROVING)
def test_a_status_that_is_not_the_owner_s_approval_authorizes_nothing(
    repo: Path, status: str
) -> None:
    assert 'policy' in tests.rules(_relaxed(repo, status))


def test_the_post_landing_state_passes(repo: Path) -> None:
    r"""The checkpoint's harness, this tree's, judging a candidate that
    changes nothing passes."""
    landed = tests.checkpoint_of(repo)
    tests.commit(repo)
    assert tests.check(repo, landed) == []
