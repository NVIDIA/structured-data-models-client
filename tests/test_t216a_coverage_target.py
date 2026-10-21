# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T2.16a's exit on ``scripts/rust_core/coverage.py`` (the package's test
author's): rule 6 places the target of an EVALUATE, an EXPLAIN and a
named-entity query (``FOR t.c = v``, ``FOR t.c IN (...)``), the forms of
the eight cases T2.13's tests dropped the ``float64/target`` claim from
(1e89eacc: "coverage cannot place"). The claims come back in T2.14c
(``tests/test_t214c_coverage_tags.py``), once this reading has landed: the
gate judges a candidate with the checkpoint's coverage.py.

Each seed is this repository's HEAD with case files changed, written as a
commit whose objects live in a temporary object directory (nothing of the
repository is written, as ``tests/test_coverage_rule6_evidence.py``
seeds), and judged by ``coverage.py --check``. The ninth case 1e89eacc
touched, ``date.day-first-then-packed``, predicts ``COUNT(orders.*)``: no
target column, so the claim stays off it, and claiming it fails rule 6.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / 'scripts/rust_core/coverage.py'
CLAIM = 'float64/target'
#: The eight cases whose claim comes back, by the query form each holds.
RESTORED = {
    'evaluate.regression': 'EVALUATE PREDICT',
    'explain.answered': 'EXPLAIN PREDICT',
    'explain.query': 'EXPLAIN PREDICT',
    'explain.two-entities': 'EXPLAIN PREDICT',
    'ids.absent': 'FOR users.user_id =',
    'ids.in-list-other-indices': 'FOR users.user_id IN',
    'ids.one': 'FOR users.user_id =',
    'ids.wrong-type': 'FOR users.user_id =',
}
NO_TARGET = 'date.day-first-then-packed'


def _path(case_id: str) -> str:
    return f'contract/cases/s1/{case_id}.json'


def _git(env: dict[str, str], *args: str, text_in: str | None = None) -> str:
    return subprocess.run(
        ['git', '-C', str(ROOT), *args],
        input=text_in,
        capture_output=True,
        text=True,
        check=True,
        env=env,
        timeout=120,
    ).stdout.strip()


def _environment(tmp: Path) -> dict[str, str]:
    common = Path(_git(dict(os.environ), 'rev-parse', '--git-common-dir'))
    if not common.is_absolute():
        common = ROOT / common
    (tmp / 'objects').mkdir()
    return {
        **os.environ,
        'GIT_OBJECT_DIRECTORY': str(tmp / 'objects'),
        'GIT_ALTERNATE_OBJECT_DIRECTORIES': str(common / 'objects'),
        'GIT_INDEX_FILE': str(tmp / 'index'),
        'GIT_AUTHOR_NAME': 'seed',
        'GIT_AUTHOR_EMAIL': 'seed@example.invalid',
        'GIT_COMMITTER_NAME': 'seed',
        'GIT_COMMITTER_EMAIL': 'seed@example.invalid',
    }


def _case(case_id: str) -> dict[str, Any]:
    return json.loads(_git(dict(os.environ), 'show', f'HEAD:{_path(case_id)}'))


def _claiming(case: dict[str, Any]) -> dict[str, Any]:
    if CLAIM not in case['cells']:
        case['cells'] = [*case['cells'], CLAIM]
    return case


def seeded(
    tmp: Path, cases: dict[str, dict[str, Any]]
) -> subprocess.CompletedProcess[str]:
    r"""``coverage.py --check`` of HEAD with ``cases`` written over theirs."""
    env = _environment(tmp)
    _git(env, 'read-tree', 'HEAD')
    for case_id, case in cases.items():
        blob = _git(
            env,
            'hash-object',
            '-w',
            '--stdin',
            text_in=json.dumps(case, indent=2) + '\n',
        )
        _git(
            env,
            'update-index',
            '--cacheinfo',
            f'100644,{blob},{_path(case_id)}',
        )
    commit = _git(
        env, 'commit-tree', _git(env, 'write-tree'), '-p', 'HEAD', '-m', 'seed'
    )
    return subprocess.run(
        [
            sys.executable,
            str(COVERAGE),
            '--check',
            '--repo',
            str(ROOT),
            '--rev',
            commit,
            '--checkpoint',
            'HEAD',
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=600,
    )


def _rule_6(done: subprocess.CompletedProcess[str]) -> list[str]:
    output = done.stdout + done.stderr
    assert 'Traceback' not in output, output
    return [line for line in output.splitlines() if line.startswith('rule 6:')]


def test_the_ninth_case_has_no_target_column_and_claims_none() -> None:
    case = _case(NO_TARGET)
    assert 'COUNT(orders.*' in case['input']['pql']
    assert CLAIM not in case['cells']


@pytest.mark.parametrize('case_id', sorted(RESTORED))
def test_rule_6_finds_each_query_form_s_target(case_id: str) -> None:
    # The target each form names, read by the rule's own reader.
    spec = importlib.util.spec_from_file_location('coverage_t216a', COVERAGE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    case = _case(case_id)
    assert RESTORED[case_id] in case['input']['pql'], case_id
    assert module._target(case) == ('orders', 'price')


def test_claiming_the_target_of_a_count_still_fails_rule_6(
    tmp_path: Path,
) -> None:
    done = seeded(tmp_path, {NO_TARGET: _claiming(_case(NO_TARGET))})
    assert done.returncode == 1, done.stdout + done.stderr
    assert any(NO_TARGET in line and CLAIM in line for line in _rule_6(done))


def test_a_claim_on_a_query_with_no_target_column_fails_rule_6(
    tmp_path: Path,
) -> None:
    # Control: the reader does not take a column from past the target, as
    # the entity a FOR names, as the target.
    case = _case('ids.one')
    case['input']['pql'] = (
        'PREDICT COUNT(orders.*, 0, 30, days) FOR users.user_id = 1'
    )
    done = seeded(tmp_path, {'ids.one': _claiming(case)})
    assert any('ids.one' in line for line in _rule_6(done)), done.stdout
