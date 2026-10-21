# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.0's acceptance tests for exits 3 and 4 (the independent test
author's): the first checkpoint of a phase, and a ratchet base for
``rust-core``.

One temporary history, shaped as the real one is: ``rust-core-phase-1``
holds two landings (``tests/test_contract_ratchet.py``'s ``build()``, the
harness copied from this checkout, the second landing adding
``ratchet.py``); ``rust-core`` merges it (a merge commit, no landing) and
then takes two handbook commits that change documents only; and
``rust-core-phase-2`` starts at ``rust-core``'s head, the fork point, which
is no landing commit. A bare origin holds the three branches, and each test
works in a clone of it, as CI does (``origin/<branch>``, with local
branches of the same names too).

Exit 3: a candidate on the fork point is judged against the fork point (by
``ratchet.py checkpoint --base origin/rust-core-phase-2``, and ``effective``
with it); the first landing's phase push and its post-landing lane resolve
to it; after it, the next package is judged against that landing.
Seeded faults: a non-landing commit that is not the fork point (an
unlanded commit on the Phase 2 branch, an older ``rust-core`` commit, the
Phase 1 merge commit) is still refused as "not a landing commit", and a
first landing recording another base is refused on its phase push.

Exit 4: on a push to ``rust-core`` (GitHub's environment, and ci.yml's own
workflow-level settings), ``ratchet.py where`` names a base; its
checkpoint is a landing commit ``rust-core`` holds; the ratchet's check
passes on ``rust-core``'s head with the real ``python-core`` observations
of the cases; those observations, compared with the reference raw, differ
(the red CI runs since the Phase 1 merge: every approved difference fails),
and compared with the effective expectations of ``where``'s checkpoint,
pass, through ``compare.py`` as CI's steps call it and through
``contract/tests/test_runners.py``'s ``expectations()``. Seeded: an
outcome change no entry approves still fails on ``rust-core``.

The interface assumed: ``SDM_PHASE_BRANCH`` names the phase for exit 3's
local runs (see :data:`PHASE_ENV`); ``ratchet.py``'s commands keep their
command lines and output (``checkpoint`` prints the checkpoint, the base
and ``harness``); exit 4 assumes nothing about what ``where`` prints for
``rust-core``, only that it is the gate's arguments, as for a lane.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

# Gate tooling (T3.0): CI runs it in the tooling jobs, not in every matrix job.
pytestmark = pytest.mark.gate_tooling

ROOT = Path(__file__).resolve().parents[2]
RATCHET = ROOT / 'scripts/rust_core/ratchet.py'
COMPARE = ROOT / 'scripts/rust_core/compare.py'

# --- the interface assumed ------------------------------------------------------------
PHASE_ENV = 'SDM_PHASE_BRANCH'
PHASE1 = 'rust-core-phase-1'
PHASE2 = 'rust-core-phase-2'
EPIC = 'rust-core'
# ----------------------------------------------------------------------------------------

#: An acceptance case whose python-core baseline is the reference's outcome,
#: the seed for an unapproved change (test_contract_ratchet's ACCEPTED).
SEEDED = 's0.accept.sum-window'


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


def landing(repo: Path, package: str, base: str) -> str:
    r"""An empty landing commit on HEAD, as land.py makes it."""
    return commit(
        repo,
        f'Land {package}: the package',
        trailers={
            'Reviewed-by': f'independent reviewer and Codex, {package}',
            'Protected-paths': 'none',
            'Base-SHA': base,
            'Checkpoint-SHA': base,
        },
    )


def handbook(name: str) -> Callable[[Path], None]:
    def edit(repo: Path) -> None:
        path = repo / 'docs/design/rust-core/orchestration' / f'{name}.md'
        path.write_text(f'# {name}\n', encoding='utf-8')

    return edit


@pytest.fixture(scope='module')
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    r"""The history above, pushed to a bare origin; the SHAs by name."""
    base = tmp_path_factory.mktemp('t20-world')
    work = base / 'work'
    work.mkdir()
    helpers.build(work)
    shas: dict[str, Any] = {'L1': git(work, 'rev-parse', 'HEAD')}
    shas['L0'] = git(work, 'rev-list', '--max-parents=0', 'HEAD')
    # rust-core before the merge, with the first landing's tree.
    tree = git(work, 'rev-parse', shas['L0'] + '^{tree}')
    shas['R0'] = git(work, 'commit-tree', tree, '-m', 'Merge Phase 0: prepare')
    git(work, 'checkout', '-q', '-b', EPIC, shas['R0'])
    git(
        work, 'merge', '-q', '--no-ff', '--allow-unrelated-histories',
        '-m', 'Merge Phase 1: the harness', PHASE1,
    )  # fmt: skip
    shas['M'] = git(work, 'rev-parse', 'HEAD')
    shas['H1'] = commit(work, 'Handoff: the handbook', handbook('handbook'))
    shas['F'] = commit(work, 'Handoff fix 1: the review', handbook('review'))
    git(work, 'branch', PHASE2, shas['F'])
    origin = base / 'origin.git'
    git(base, 'init', '-q', '--bare', str(origin))
    git(work, 'push', '-q', str(origin), PHASE1, EPIC, PHASE2)
    return {'work': work, 'origin': origin, 'shas': shas}


@pytest.fixture
def clone(world: dict[str, Any], tmp_path: Path) -> Path:
    r"""A clone of the origin, at the fork point, with local branches."""
    where = tmp_path / 'clone'
    git(
        tmp_path,
        'clone',
        '-q',
        '--no-checkout',
        str(world['origin']),
        str(where),
    )
    for branch in (PHASE1, EPIC, PHASE2):
        git(where, 'branch', '-q', '-f', branch, f'origin/{branch}')
    git(where, 'checkout', '-q', '--detach', world['shas']['F'])
    return where


def move(repo: Path, branch: str, sha: str) -> None:
    r"""``branch`` moved to ``sha``, locally and as last fetched."""
    git(repo, 'update-ref', f'refs/remotes/origin/{branch}', sha)
    git(repo, 'update-ref', f'refs/heads/{branch}', sha)


def work_on(repo: Path, at: str, subject: str, name: str) -> str:
    git(repo, 'checkout', '-q', '--detach', at)
    return commit(repo, subject, lambda r: (r / f'{name}.txt').write_text(name))


def clean_env(**extra: str) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(('GITHUB_', 'SDM_', 'GIT_'))
    }
    return {**env, **extra}


def ratchet(
    repo: Path, *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RATCHET), *arguments],
        cwd=repo,
        env=env if env is not None else clean_env(**{PHASE_ENV: PHASE2}),
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )


def checkpoint(repo: Path, *where: str) -> list[str]:
    done = ratchet(repo, 'checkpoint', '--repo', '.', *where)
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout.split()


def refused(repo: Path, *where: str) -> str:
    done = ratchet(repo, 'checkpoint', '--repo', '.', *where)
    assert done.returncode == 1, done.stdout + done.stderr
    return done.stderr


# --- exit 3: the first checkpoint --------------------------------------------------------


def test_a_candidate_on_the_fork_point_is_judged_against_it(
    clone: Path, world: dict[str, Any], tmp_path: Path
) -> None:
    r"""Exit 3: the Phase 2 branch's fork point from ``rust-core``, no
    landing commit, is the first checkpoint, and its harness judges;
    ``effective`` writes its expectations."""
    fork = world['shas']['F']
    work_on(clone, fork, 'T2.0: the work', 't20')
    assert checkpoint(clone, '--base', f'origin/{PHASE2}') == [
        fork,
        fork,
        'harness',
    ]
    out = tmp_path / 'effective.json'
    done = ratchet(
        clone, 'effective', '--repo', '.', '--base', f'origin/{PHASE2}',
        '--out', str(out),
    )  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(out.read_text()), 'no effective expectations'


def test_the_fork_point_stays_the_checkpoint_when_rust_core_moves_on(
    clone: Path, world: dict[str, Any]
) -> None:
    r"""Exit 3: ``rust-core`` taking a commit after the fork (the update
    from ``main``) does not change the fork point."""
    fork = world['shas']['F']
    later = work_on(clone, fork, 'Merge main: the update', 'update')
    move(clone, EPIC, later)
    work_on(clone, fork, 'T2.0: the work', 't20')
    assert checkpoint(clone, '--base', f'origin/{PHASE2}')[:2] == [fork, fork]


def test_an_unlanded_commit_on_the_phase_2_branch_is_still_refused(
    clone: Path, world: dict[str, Any]
) -> None:
    r"""Seeded: a commit pushed to the Phase 2 branch by hand, on the fork
    point, is no checkpoint for a candidate built on it."""
    unlanded = work_on(
        clone, world['shas']['F'], 'T2.9: pushed unlanded', 'unlanded'
    )
    move(clone, PHASE2, unlanded)
    work_on(clone, unlanded, 'T2.0: built on it', 'on-it')
    assert 'not a landing commit' in refused(
        clone, '--base', f'origin/{PHASE2}'
    )


@pytest.mark.parametrize('at', ['H1', 'M'])
def test_a_rust_core_commit_that_is_not_the_fork_point_is_still_refused(
    clone: Path, world: dict[str, Any], at: str
) -> None:
    r"""Seeded: a candidate built on an older ``rust-core`` commit (a
    handbook commit, the Phase 1 merge), whose merge base with the Phase 2
    branch is that commit: no landing, not the fork point, refused."""
    work_on(clone, world['shas'][at], 'T2.0: on an older rust-core', 'older')
    assert 'not a landing commit' in refused(
        clone, '--base', f'origin/{PHASE2}'
    )


def test_the_first_landing_s_phase_push_and_its_lane_resolve_to_the_fork_point(
    clone: Path, world: dict[str, Any]
) -> None:
    r"""Exit 3, the landing's own runs: the push landing T2.0 on Phase 2
    (``--phase-push`` the fork point) and the lane after it (the landing
    as the phase head) resolve to the fork point; the post-landing state
    judged against itself."""
    fork = world['shas']['F']
    work_on(clone, fork, 'T2.0: the work', 't20')
    landed = landing(clone, 'T2.0', fork)
    assert checkpoint(clone, '--phase-push', fork)[:2] == [fork, fork]
    move(clone, PHASE2, landed)
    assert checkpoint(clone, '--base', f'origin/{PHASE2}')[:2] == [fork, fork]


def test_after_the_first_landing_the_next_package_is_judged_against_it(
    clone: Path, world: dict[str, Any]
) -> None:
    r"""Control: once T2.0 has landed, the next package's checkpoint is
    that landing, as in Phase 1."""
    fork = world['shas']['F']
    work_on(clone, fork, 'T2.0: the work', 't20')
    landed = landing(clone, 'T2.0', fork)
    move(clone, PHASE2, landed)
    work_on(clone, landed, 'T2.0a: the design', 'design')
    assert checkpoint(clone, '--base', f'origin/{PHASE2}')[:2] == [
        landed,
        landed,
    ]


def test_a_first_landing_recording_another_base_is_refused_on_its_push(
    clone: Path, world: dict[str, Any]
) -> None:
    r"""Seeded: a landing whose Base-SHA is an older ``rust-core`` commit,
    pushed onto the Phase 2 branch from there, is refused."""
    older = world['shas']['H1']
    work_on(clone, older, 'T2.0: the work', 't20')
    landing(clone, 'T2.0', older)
    done = ratchet(clone, 'checkpoint', '--repo', '.', '--phase-push', older)
    assert done.returncode == 1, done.stdout + done.stderr


# --- exit 4: rust-core's ratchet base ------------------------------------------------------


def ci_settings() -> dict[str, str]:
    r"""ci.yml's workflow-level settings that are plain values, as every job
    has them (an expression is GitHub's to evaluate, not here)."""
    document = yaml.safe_load(
        (ROOT / '.github/workflows/ci.yml').read_text(encoding='utf-8')
    )
    return {
        str(k): str(v)
        for k, v in (document.get('env') or {}).items()
        if '${{' not in str(v)
    }


def epic_push(tmp_path: Path, before: str) -> dict[str, str]:
    r"""The environment of a CI job on a push to ``rust-core``."""
    event = tmp_path / 'event.json'
    event.write_text(
        json.dumps({'before': before, 'ref': f'refs/heads/{EPIC}'})
    )
    return clean_env(
        **ci_settings(),
        GITHUB_ACTIONS='true',
        GITHUB_EVENT_NAME='push',
        GITHUB_REF_NAME=EPIC,
        GITHUB_REF=f'refs/heads/{EPIC}',
        GITHUB_EVENT_PATH=str(event),
    )


def epic_where(clone: Path, world: dict[str, Any], tmp_path: Path) -> list[str]:
    git(clone, 'checkout', '-q', '--detach', world['shas']['F'])
    done = ratchet(clone, 'where', env=epic_push(tmp_path, world['shas']['H1']))
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout.split()


@pytest.fixture(scope='module')
def observed(
    world: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
) -> list[dict[str, Any]]:
    r"""python-core's real observations of the cases at ``rust-core``'s
    head, as contract-python's gate step makes them."""
    work = world['work']
    out = tmp_path_factory.mktemp('t20-observed') / 'python-core.jsonl'
    done = subprocess.run(
        [
            sys.executable,
            '-I',
            str(work / 'contract/runners/run_python.py'),
            '--mode',
            'python-core',
            '--out',
            str(out),
            str(work / 'contract/cases'),
        ],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    assert done.returncode == 0, done.stderr
    return [json.loads(line) for line in out.read_text().splitlines()]


def written(
    lines: list[dict[str, Any]],
    where: Path,
    change: Callable[[list[dict[str, Any]]], None] | None = None,
) -> Path:
    lines = copy.deepcopy(lines)
    if change is not None:
        change(lines)
    where.parent.mkdir(parents=True, exist_ok=True)
    where.write_text(''.join(json.dumps(x) + '\n' for x in lines))
    return where


def unapproved(lines: list[dict[str, Any]]) -> None:
    r"""The seed: one acceptance case's outcome moved to another stage,
    which no register entry approves."""
    [line] = [x for x in lines if x['case'] == SEEDED]
    assert line['outcome']['stage'] != 'predict'
    line['outcome']['stage'] = 'predict'


def compare(
    clone: Path, observations: Path, expect: list[str]
) -> subprocess.CompletedProcess[str]:
    reference = next((clone / 'contract/reference').glob('*.jsonl'))
    return subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            '--cases',
            str(clone / 'contract/cases'),
            '--reference',
            str(reference),
            *expect,
            str(observations),
        ],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )


def epic_expectations(
    clone: Path, world: dict[str, Any], tmp_path: Path
) -> list[str]:
    where = epic_where(clone, world, tmp_path)
    out = tmp_path / 'effective.json'
    done = ratchet(
        clone, 'effective', '--repo', '.', *where, '--out', str(out),
        env=epic_push(tmp_path, world['shas']['H1']),
    )  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    return ['--expect', str(out)]


def test_where_names_a_base_on_a_push_to_rust_core(
    clone: Path, world: dict[str, Any], tmp_path: Path
) -> None:
    r"""Exit 4: ``rust-core`` has a ratchet base."""
    assert epic_where(clone, world, tmp_path), 'no base on rust-core'


def test_rust_core_s_checkpoint_is_a_landing_it_holds(
    clone: Path, world: dict[str, Any], tmp_path: Path
) -> None:
    r"""Exit 4: the base ``where`` names resolves, from ``rust-core``'s
    head, to a landing commit that ``rust-core`` holds, with a harness."""
    where = epic_where(clone, world, tmp_path)
    done = ratchet(
        clone, 'checkpoint', '--repo', '.', *where,
        env=epic_push(tmp_path, world['shas']['H1']),
    )  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    found, _, judged = done.stdout.split()
    assert git(clone, 'log', '-1', '--format=%s', found).startswith('Land ')
    assert git(clone, 'merge-base', found, world['shas']['F']) == found
    assert judged == 'harness'


def test_the_ratchet_check_passes_on_rust_core_s_head(
    clone: Path,
    world: dict[str, Any],
    tmp_path: Path,
    observed: list[dict[str, Any]],
) -> None:
    r"""Exit 4: the gate's ratchet step on ``rust-core``'s head, with the
    real python-core observations, passes: no comparison relaxed. Seeded:
    the unapproved outcome change fails it."""
    where = epic_where(clone, world, tmp_path)
    env = epic_push(tmp_path, world['shas']['H1'])
    done = ratchet(clone, 'checkpoint', '--repo', '.', *where, env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    found = done.stdout.split()[0]
    sound = written(observed, tmp_path / 'sound/python-core.jsonl')
    done = ratchet(
        clone, 'check', '--repo', '.', '--checkpoint', found,
        '--observations', str(sound), env=env,
    )  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    seeded = written(
        observed, tmp_path / 'seeded/python-core.jsonl', unapproved
    )
    done = ratchet(
        clone, 'check', '--repo', '.', '--checkpoint', found,
        '--observations', str(seeded), env=env,
    )  # fmt: skip
    assert done.returncode == 1, done.stdout
    assert SEEDED in done.stdout, done.stdout


def test_on_rust_core_the_approved_differences_pass_and_an_unapproved_one_fails(
    clone: Path,
    world: dict[str, Any],
    tmp_path: Path,
    observed: list[dict[str, Any]],
) -> None:
    r"""Exit 4, the comparison CI's steps make on ``rust-core`` (the
    cross-host step's ``effective $where`` and ``--expect``): raw, the
    real observations differ from the reference (the red runs since the
    Phase 1 merge, reproduced); held to ``rust-core``'s effective
    expectations they pass; seeded, an unapproved change still fails."""
    sound = written(observed, tmp_path / 'sound/python-core.jsonl')
    raw = compare(clone, sound, [])
    assert raw.returncode != 0, 'the reproduction: no approved difference'
    expect = epic_expectations(clone, world, tmp_path)
    done = compare(clone, sound, expect)
    assert done.returncode == 0, done.stdout[-3000:]
    seeded = written(
        observed, tmp_path / 'seeded/python-core.jsonl', unapproved
    )
    done = compare(clone, seeded, expect)
    assert done.returncode != 0, done.stdout[-3000:]
    assert SEEDED in done.stdout, done.stdout[-3000:]


def test_test_runners_holds_rust_core_to_its_effective_expectations(
    clone: Path,
    world: dict[str, Any],
    tmp_path: Path,
    observed: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""Exit 4, ``contract/tests/test_runners.py``'s comparison: on a push
    to ``rust-core`` its ``expectations()`` gives the comparator the
    effective expectations, with which the real observations pass."""
    runners = _module(
        ROOT / 'contract/tests/test_runners.py', 't20_test_runners'
    )
    git(clone, 'checkout', '-q', '--detach', world['shas']['F'])
    for key in [k for k in os.environ if k.startswith(('GITHUB_', 'SDM_'))]:
        monkeypatch.delenv(key)
    for key, value in epic_push(tmp_path, world['shas']['H1']).items():
        if key.startswith(('GITHUB_', 'SDM_')) or key in ci_settings():
            monkeypatch.setenv(key, value)
    monkeypatch.chdir(clone)
    expect = runners.expectations(tmp_path, repo=clone)
    assert expect[:1] == ['--expect'], expect
    sound = written(observed, tmp_path / 'sound/python-core.jsonl')
    done = compare(clone, sound, expect)
    assert done.returncode == 0, done.stdout[-3000:]
