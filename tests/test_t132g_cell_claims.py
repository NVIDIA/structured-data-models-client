# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32g part 1: coverage rule 6, "a contradicted cell claim fails"
(scope-format.md, "Checks"; design, "Coverage generator": a cell claim
names the observed boundary and stage, the received Arrow type, admitted
type, declared role and inferred role if reached, and a refusal covers its
rejected input without claiming a later stage).

Written by the package's independent test author, before the
implementation, beside the three families' seeds
(``tests/test_t132a_cell_claims.py``, ``test_t132c_cell_claims.py`` and
``test_t132d_cell_claims.py``: a contradicted type, role, key and stage).
These are what those leave open. The check is the gate's coverage check,
``scripts/rust_core/coverage.py --check``, which reads a commit: each seed
is this repository's HEAD with case files added or changed, written as a
commit whose objects live in a temporary object directory (nothing of the
repository is written).

- **No evidence fails closed**: a case no runner has observed (a copy of
  ``key.uint8`` under a new id, with no reference line or other
  observation) claiming ``key.uint8``'s own cells fails with a ``rule 6:``
  problem naming it. The claims are true of ``key.uint8``, so only the
  missing evidence can fail them. Its control: the same case with no
  claims passes, since rule 6 judges claims and not cases.
- **Every case's claims are judged**: a contradicted claim fails on an
  acceptance case and on a diagnostic one alike. The families' seeds are
  diagnostic cases until part 2 admits them, and the brief has them pass
  at part 1. Seeding the status keeps both judged after the admission.
- **The post-landing state passes** (standing rule): HEAD judged against
  itself, as the gate judges the phase branch at the landing, passes. Every
  case's own claims hold on the evidence, so a case recorded at
  ``timestamp[s]`` that claims ``timestamp-us/time-column``
  (``time.timestamp-s``) passes as the type the core receives (scope-format.md:
  "a cell names the type the core receives").

What a claim is held to (the families' seeds, as corrected here): a claim
``<type>/<role>`` is contradicted when no column of the case's observed
source holds that type in that role at a stage the case reached. No case
field names one column as the case's subject. So a claim that some other
column of ``shop`` makes true, ``int64/primary-key`` from
``orders.order_id`` or ``utf8/feature`` from ``users.country``, is no
seed of a contradiction. Each seed claims a cell that no column of its
case holds.

Gating (``contract/tests/t132g_gate.py``): a test named in
``t132g-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``) failed
when written. It still runs and must still fail; once it passes it fails,
naming itself, until it is taken out of the list.
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
UNOBSERVED = 'key.uint8-unobserved'


def _gate() -> Any:
    name = 't132g_gate'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'contract/tests/t132g_gate.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


G = _gate()


def _git(env: dict[str, str], *args: str, text_in: str | None = None) -> str:
    return subprocess.run(
        ['git', '-C', str(ROOT), *args],
        input=text_in,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout.strip()


def _environment(tmp: Path) -> dict[str, str]:
    r"""Git writing new objects to ``tmp`` and reading the repository's as
    alternates, with an index of its own: nothing of the repository is
    changed."""
    common = Path(
        subprocess.run(
            ['git', '-C', str(ROOT), 'rev-parse', '--git-common-dir'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    if not common.is_absolute():
        common = ROOT / common
    objects = tmp / 'objects'
    objects.mkdir(parents=True)
    return {
        **os.environ,
        'GIT_OBJECT_DIRECTORY': str(objects),
        'GIT_ALTERNATE_OBJECT_DIRECTORIES': str(common / 'objects'),
        'GIT_INDEX_FILE': str(tmp / 'index'),
        'GIT_AUTHOR_NAME': 'seed',
        'GIT_AUTHOR_EMAIL': 'seed@example.invalid',
        'GIT_COMMITTER_NAME': 'seed',
        'GIT_COMMITTER_EMAIL': 'seed@example.invalid',
    }


def _case(case: str) -> dict[str, Any]:
    r"""``case``'s file at HEAD."""
    loaded: dict[str, Any] = json.loads(
        _git(dict(os.environ), 'show', f'HEAD:contract/cases/s1/{case}.json')
    )
    return loaded


def seeded(
    tmp: Path, cases: dict[str, dict[str, Any]]
) -> tuple[str, dict[str, str]]:
    r"""HEAD with each of ``cases`` (by id) written as its case file, as a
    commit, and the environment that reads it."""
    env = _environment(tmp)
    _git(env, 'read-tree', 'HEAD')
    for case, loaded in cases.items():
        blob = _git(
            env,
            'hash-object',
            '-w',
            '--stdin',
            text_in=json.dumps(loaded, indent=2) + '\n',
        )
        _git(
            env,
            'update-index',
            '--add',
            '--cacheinfo',
            f'100644,{blob},contract/cases/s1/{case}.json',
        )
    tree = _git(env, 'write-tree')
    commit = _git(env, 'commit-tree', tree, '-p', 'HEAD', '-m', 'seed')
    return commit, env


def coverage(rev: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(COVERAGE),
            '--check',
            '--repo',
            str(ROOT),
            '--rev',
            rev,
            '--checkpoint',
            'HEAD',
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )


def fails_naming(done: subprocess.CompletedProcess[str], case: str) -> None:
    r"""The check failed by rule 6, naming ``case``, and did not crash."""
    output = done.stdout + done.stderr
    assert 'Traceback' not in output, output
    assert done.returncode == 1, output
    assert any(
        line.startswith('rule 6:') and case in line
        for line in output.splitlines()
    ), output


def _unobserved(cells: list[str]) -> dict[str, Any]:
    r"""``key.uint8`` under an id no runner has observed."""
    loaded = _case('key.uint8')
    loaded['id'] = UNOBSERVED
    loaded['cells'] = cells
    return loaded


# --- the post-landing state ----------------------------------------------------


def test_the_post_landing_state_passes(tmp_path: Path) -> None:
    # The phase branch at the landing, judged against itself: every case's
    # own claims hold on the evidence. Passes before rule 6 and must after.
    done = coverage('HEAD', _environment(tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr


# --- no evidence -----------------------------------------------------------------


def test_an_unobserved_case_with_no_claims_passes(tmp_path: Path) -> None:
    # The control of the next seed: the new case itself is no problem.
    rev, env = seeded(tmp_path, {UNOBSERVED: _unobserved([])})
    done = coverage(rev, env)
    assert done.returncode == 0, done.stdout + done.stderr


def test_a_claim_with_no_evidence_fails_closed(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        cells = _case('key.uint8')['cells']
        assert cells, 'key.uint8 claims cells'
        rev, env = seeded(tmp_path, {UNOBSERVED: _unobserved(cells)})
        fails_naming(coverage(rev, env), UNOBSERVED)

    G.held('exits', f'py:{request.node.name}', check)


# --- every case's claims -----------------------------------------------------------


@pytest.mark.parametrize('status', ['acceptance', 'diagnostic'])
def test_a_contradicted_claim_fails_whatever_the_case_s_status(
    status: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    # key.uint8's keys are uint8; no column of it is a uint16.
    def check() -> None:
        loaded = _case('key.uint8')
        loaded['status'] = status
        loaded['cells'] = ['uint16/primary-key', 'uint16/foreign-key']
        rev, env = seeded(tmp_path, {'key.uint8': loaded})
        fails_naming(coverage(rev, env), 'key.uint8')

    G.held('exits', f'py:{request.node.name}', check)


@pytest.mark.parametrize('status', ['acceptance', 'diagnostic'])
def test_a_case_s_own_claims_pass_whatever_its_status(
    status: str, tmp_path: Path
) -> None:
    # The control of the seed above: only the claims changed it.
    loaded = _case('key.uint8')
    loaded['status'] = status
    rev, env = seeded(tmp_path, {'key.uint8': loaded})
    done = coverage(rev, env)
    assert done.returncode == 0, done.stdout + done.stderr
