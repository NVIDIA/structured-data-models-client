# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32b's exit "a contradicted cell claim fails" (T1.32a's exits applied
to the decimal family; coverage rule 6; scope-format.md, "Checks";
design, "Coverage generator": a cell claim names the observed boundary and
stage, the received Arrow type and the declared role among them).

Written by the package's independent test author, before the
implementation. Rule 6 arrives with T1.32a's part 2, which has not landed
on the phase head this was written on (ca9a8b5f): until it does, the seeds
pass the check and are pending. The check is the gate's coverage check,
``scripts/rust_core/coverage.py --check``, which reads a commit: each seed
is this repository's HEAD with one case's ``cells`` changed to a claim its
own input contradicts, written as a commit whose objects live in a
temporary object directory (nothing of the repository is written), and the
check must fail with a ``rule 6:`` problem naming the case. The controls,
HEAD itself and a seed of the case's own cells, pass.

- ``decimal.feature-38-0-wide`` (a DECIMAL(38, 0) feature, received as
  ``decimal128``) claiming ``binary/feature``: the type contradicted, a
  type no column of its source has (the shop base's ``price`` is a DOUBLE
  feature, so under E96(a) a ``float64/feature`` claim is true and could
  never fail; reviewer's P3);
- ``decimal.feature-38-0-wide`` claiming ``decimal128/target``: the role
  contradicted (its question's target is ``orders.price``, a DOUBLE sum);
- ``decimal.target-38-18`` (a summed DECIMAL(38, 18) target) claiming
  ``decimal128/primary-key``: a decimal claimed as the key (the key is
  ``int64``).

Gating (``contract/tests/t132b_gate.py``): a test named in
``t132b-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``) failed
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


def _gate() -> Any:
    name = 't132b_gate'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'contract/tests/t132b_gate.py'
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
    objects.mkdir()
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


def seeded(
    tmp: Path, case: str, cells: list[str]
) -> tuple[str, dict[str, str]]:
    r"""HEAD with ``case``'s cells replaced, as a commit, and the
    environment that reads it."""
    env = _environment(tmp)
    path = f'contract/cases/s1/{case}.json'
    loaded = json.loads(_git(env, 'show', f'HEAD:{path}'))
    loaded['cells'] = cells
    blob = _git(
        env,
        'hash-object',
        '-w',
        '--stdin',
        text_in=json.dumps(loaded, indent=2) + '\n',
    )
    _git(env, 'read-tree', 'HEAD')
    _git(env, 'update-index', '--cacheinfo', f'100644,{blob},{path}')
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


def test_the_control_head_passes(tmp_path: Path) -> None:
    done = coverage('HEAD', _environment(tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr


def test_a_seed_with_the_case_s_own_claims_passes(tmp_path: Path) -> None:
    # The seeding itself changes nothing the check reads but the cells.
    head = json.loads(
        (ROOT / 'contract/cases/s1/decimal.feature-38-0-wide.json').read_text(
            encoding='utf-8'
        )
    )
    assert head['cells'] == ['decimal128/feature']
    rev, env = seeded(tmp_path, 'decimal.feature-38-0-wide', head['cells'])
    done = coverage(rev, env)
    assert done.returncode == 0, done.stdout + done.stderr


SEEDS = {
    'type': ('decimal.feature-38-0-wide', ['binary/feature']),
    'role': ('decimal.feature-38-0-wide', ['decimal128/target']),
    'decimal-as-key': ('decimal.target-38-18', ['decimal128/primary-key']),
}


@pytest.mark.parametrize('seed', list(SEEDS))
def test_a_contradicted_cell_claim_fails(seed: str, tmp_path: Path) -> None:
    def check() -> None:
        case, cells = SEEDS[seed]
        rev, env = seeded(tmp_path, case, cells)
        done = coverage(rev, env)
        output = done.stdout + done.stderr
        assert 'Traceback' not in output, output
        assert done.returncode == 1, output
        assert any(
            line.startswith('rule 6:') and case in line
            for line in output.splitlines()
        ), output

    G.held('exits', f'py:test_a_contradicted_cell_claim_fails[{seed}]', check)
