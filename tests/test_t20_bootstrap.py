# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.0's bootstrap of a phase's first checkpoint through ``gate.sh`` (the
implementer's own tests; the acceptance tests are
``contract/tests/test_t20_checkpoints.py``).

Phase 2's fork point is ``rust-core``'s head, no landing commit, and its
own ``ratchet.py`` is Phase 1's, which refuses any checkpoint that is no
landing commit, itself included. ``gate.sh`` then judges with the merged
phase's last landing (``ratchet.py stand-in``), whose harness resolves
that landing as the base. One temporary history, shaped as the real one:
``rust-core-phase-1``'s landings (``tests/test_contract_ratchet.py``'s
``build()``), ``rust-core`` merging them, a handbook commit, then the fork
point, whose ``ratchet.py`` refuses every checkpoint as Phase 1's refuses
the fork point.

- ``stand-in`` names the merged phase's last landing, and nothing for a
  commit holding no merged phase.
- The candidate on the fork point passes ``gate.sh --only ratchet``,
  judged by the stand-in's harness, which the gate says.
- Control: when the fork point's own ``ratchet.py`` resolves it, the fork
  point's harness judges, no stand-in.
- Seeded: a candidate on an unlanded commit of the phase branch is still
  refused, before any harness runs.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
PHASE1, PHASE2, EPIC = 'rust-core-phase-1', 'rust-core-phase-2', 'rust-core'
#: Phase 1's ratchet at the fork point, as far as gate.sh asks it: it
#: refuses the fork point, which is no landing commit.
REFUSING = (
    'import sys\n'
    "print('checkpoint: the checkpoint is not a landing commit', "
    'file=sys.stderr)\n'
    'sys.exit(1)\n'
)


def _module(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


helpers = _module(ROOT / 'tests/test_contract_ratchet.py', 'ratchet_tests')
git, commit = helpers.git, helpers.commit


def env() -> dict[str, str]:
    kept = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(('GITHUB_', 'SDM_', 'GIT_'))
    }
    return {**kept, 'PYTHON': sys.executable}


def world(base: Path, refusing: bool) -> dict[str, Any]:
    r"""The history above in a clone of a bare origin; the SHAs by name."""
    work = base / 'work'
    work.mkdir()
    helpers.build(work)
    shas = {'L1': git(work, 'rev-parse', 'HEAD')}
    root = git(work, 'rev-list', '--max-parents=0', 'HEAD')
    tree = git(work, 'rev-parse', root + '^{tree}')
    shas['R0'] = git(work, 'commit-tree', tree, '-m', 'Merge Phase 0')
    git(work, 'checkout', '-q', '-b', EPIC, shas['R0'])
    git(
        work, 'merge', '-q', '--no-ff', '--allow-unrelated-histories',
        '-m', 'Merge Phase 1: the harness', PHASE1,
    )  # fmt: skip
    shas['H1'] = commit(
        work,
        'Handoff: the handbook',
        lambda r: (r / 'docs/design/rust-core/handbook.md').write_text('# h\n'),
    )

    def fork(repo: Path) -> None:
        if refusing:
            (repo / 'scripts/rust_core/ratchet.py').write_text(REFUSING)

    shas['F'] = commit(work, 'Handoff: the fork point', fork)
    git(work, 'branch', PHASE2, shas['F'])
    origin = base / 'origin.git'
    git(base, 'init', '-q', '--bare', str(origin))
    git(work, 'push', '-q', str(origin), PHASE1, EPIC, PHASE2)
    clone = base / 'clone'
    git(base, 'clone', '-q', '--no-checkout', str(origin), str(clone))
    git(clone, 'checkout', '-q', '--detach', shas['F'])
    return {'clone': clone, 'shas': shas}


def work(repo: Path) -> None:
    r"""The candidate's change: this tree's ratchet.py, as T2.0's."""
    (repo / 't20.txt').write_text('x')
    (repo / 'scripts/rust_core/ratchet.py').write_bytes(
        (ROOT / 'scripts/rust_core/ratchet.py').read_bytes()
    )


def gate(clone: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(clone / 'scripts/rust_core/gate.sh'), *arguments],
        capture_output=True,
        text=True,
        check=False,
        cwd=clone,
        env={**env(), 'SDM_PHASE_BRANCH': PHASE2},
        stdin=subprocess.DEVNULL,
    )


def stand_in(clone: Path, sha: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / 'scripts/rust_core/ratchet.py'),
            'stand-in',
            '--repo',
            str(clone),
            sha,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env(),
        stdin=subprocess.DEVNULL,
    )


@pytest.fixture(scope='module')
def refusing(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return world(tmp_path_factory.mktemp('t20-refusing'), refusing=True)


def test_stand_in_names_the_merged_phase_s_last_landing(
    refusing: dict[str, Any],
) -> None:
    shas = refusing['shas']
    done = stand_in(refusing['clone'], shas['F'])
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == [shas['L1']]
    # Seeded: Phase 1's own history holds no merged phase.
    done = stand_in(refusing['clone'], shas['R0'])
    assert done.returncode == 1
    assert 'holds no merged phase' in done.stderr


def test_the_fork_point_s_candidate_is_judged_by_the_stand_in(
    refusing: dict[str, Any],
) -> None:
    clone, shas = refusing['clone'], refusing['shas']
    git(clone, 'checkout', '-q', '--detach', shas['F'])
    commit(clone, 'T2.0: the work', work)
    done = gate(clone, '--base', f'origin/{PHASE2}', '--only', 'ratchet')
    assert done.returncode == 0, done.stdout + done.stderr
    assert f'last landing {shas["L1"]} stands in' in done.stderr
    assert f'checkpoint {shas["L1"]}' in done.stdout
    assert 'gate: passed (--only ratchet' in done.stdout


def test_an_unlanded_commit_on_the_phase_branch_is_still_refused(
    refusing: dict[str, Any],
) -> None:
    clone, shas = refusing['clone'], refusing['shas']
    git(clone, 'checkout', '-q', '--detach', shas['F'])
    unlanded = commit(
        clone, 'T2.9: pushed unlanded', lambda r: (r / 'u.txt').write_text('u')
    )
    git(clone, 'update-ref', f'refs/remotes/origin/{PHASE2}', unlanded)
    try:
        commit(clone, 'T2.0: on it', work)
        done = gate(clone, '--base', f'origin/{PHASE2}', '--only', 'ratchet')
    finally:
        git(clone, 'update-ref', f'refs/remotes/origin/{PHASE2}', shas['F'])
    assert done.returncode == 1, done.stdout + done.stderr
    assert 'not a landing commit' in done.stderr
    assert 'stands in' not in done.stderr


def test_a_fork_point_whose_ratchet_resolves_it_judges_itself(
    tmp_path: Path,
) -> None:
    built = world(tmp_path, refusing=False)
    clone, shas = built['clone'], built['shas']
    commit(clone, 'T2.0: the work', work)
    done = gate(clone, '--base', f'origin/{PHASE2}', '--only', 'ratchet')
    assert done.returncode == 0, done.stdout + done.stderr
    assert 'stands in' not in done.stderr
    assert f'checkpoint {shas["F"]}' in done.stdout
