# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rule 8's reading of an owner's approval (T1.15 fix round 1, Codex P2):
a checkpoint row authorizes a removed path only when its status starts with
"approved by owner" and a date, followed by nothing, a parenthesis or a
note that does not withdraw it; "not approved by owner ...", a rejection,
a revocation or an approval left to confirm never authorizes. Each status
on the same temporary repository as ``tests/test_contract_coverage.py``'s,
whose helpers this reuses."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Gate tooling (T3.0): CI runs it in the tooling jobs, not in every matrix job.
pytestmark = pytest.mark.tooling

ROOT = Path(__file__).resolve().parents[1]
if 'coverage_tests' in sys.modules:
    tests = sys.modules['coverage_tests']
else:
    spec = importlib.util.spec_from_file_location(
        'coverage_tests', ROOT / 'tests/test_contract_coverage.py'
    )
    assert spec is not None and spec.loader is not None
    tests = importlib.util.module_from_spec(spec)
    sys.modules['coverage_tests'] = tests
    spec.loader.exec_module(tests)

origin, repo = tests.origin, tests.repo
PATH = 'source.open/explicit/duckdb'

APPROVING = [
    'approved by owner 2026-10-12',
    'approved by owner 2026-10-12 (answer 3)',
    'approved by owner 2026-10-12; recorded at T1.16',
    'approved by owner 2026-10-12, including the path',
]
NOT_APPROVING = [
    'not approved by owner 2026-10-12',
    'unapproved by owner 2026-10-12',
    'rejected by owner 2026-10-12',
    'rejected; approved by owner 2026-10-12 earlier',
    'approved by owner 2026-10-12; rejected by owner 2026-10-13',
    'approved by owner 2026-10-12 (revoked 2026-10-13)',
    'approved by owner 2026-10-12; withdrawn',
    'approved by owner 2026-10-12, superseded by E901',
    'approved by owner 2026-10-12, owner to confirm',
    'approved by owner 2026-10-12 later',
    'approved by owner 2026-10-1',
    'owner to confirm',
    'implementer (T1.16), owner to confirm',
]


def _dropped(repo: Path, status: str) -> object:
    checkpoint = tests.land(
        repo, 'T1.16', tests.owner_row('E900', f'`{PATH}`', status)
    )
    tests.commit(
        repo,
        'T1.17: the regression',
        tests.manifest(tests.PYTHON, tests.drop('paths', PATH)),
    )
    return tests.check(repo, checkpoint)


@pytest.mark.parametrize('status', APPROVING)
def test_an_affirmative_owner_approval_allows_the_removal(
    repo: Path, status: str
) -> None:
    tests.passing(_dropped(repo, status))


@pytest.mark.parametrize('status', NOT_APPROVING)
def test_a_status_that_is_not_the_owner_s_approval_allows_nothing(
    repo: Path, status: str
) -> None:
    found = tests.failing(_dropped(repo, status), 'rule 8', PATH)
    assert tests.rules(found) == {'rule 8'}, found
