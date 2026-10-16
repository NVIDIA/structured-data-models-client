# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.0e's acceptance tests for exits 4 and 5 (the independent test
author's): a merge of ``main`` can land, and nothing else rides in with it.

Real git everywhere, with ``gh`` faked: ``scripts/rust_core/lane_selftest.py``'s
``World`` (a bare remote, a seed clone playing everyone else, the clone the
tools run in), its phase set to ``rust-core-phase-2``. The remote's
``main`` gains commits whose subjects carry no package ID (as ``main``'s
do); package ``T2.0c`` is started on a lane, which merges the remote's
``main`` (fetched) in one merge commit titled with the ID, beside commits of
its own. ``lane.py check`` (what ``prepush.sh`` runs), ``lane.py check
--pushed-refs`` (what its pre-push hook runs) and ``land.py`` (the
backstop) judge it.

Exit 4: the commits reachable from the remote's ``main``, brought in by the
one merge commit titled with the ID, are exempt from the subject rule;
every other commit still needs the ID. Seeded faults, each refused naming
the commit (and, for ``land.py``, nothing moved): a non-``main`` commit
without the ID hidden behind the merge; a merge of a branch that is not
``main``; a ``main`` commit rewritten (another SHA); the merge commit not
titled with the ID; a commit without the ID beside the merge; a commit only
the local ``main`` branch holds (never a local ref); two merges of
``main`` (one merge commit); and, for ``land.py``, a remote-tracking ref of
``main`` moved by hand (``main`` as the remote holds it at landing time).
Exit 5: the landing verifies, the lane is at the phase head after it, and a
Phase 1 landing still verifies.

The route assumed is the package's lane (the brief lets the implementation
choose the lane or a topic branch, the lightest; the lane already has
every ownership check, so it is assumed here). If the implementation
chooses the topic branch, it says so and :data:`ROUTE` changes with it.

Gating: the tests that need the feature are marked :data:`PENDING`, a
strict xfail; the implementer deletes the marker (its one definition below)
in the commit that makes them pass. Unmarked tests are controls that pass
on the base.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts/rust_core'

#: The marker the implementer deletes (see the module docstring).
PENDING = pytest.mark.xfail(
    strict=True,
    reason='T2.0e: landing a merge of main is not implemented yet',
)

# --- the interface assumed (the implementer matches it, or asks) ---------------
PHASE1 = 'rust-core-phase-1'
PHASE2 = 'rust-core-phase-2'
#: The route a merge of main lands from: the package's lane.
ROUTE = f'{PHASE2}-lane-1b'
PACKAGE = 'T2.0c'
#: Subjects of main's commits: no package ID, as main's are.
MAIN_SUBJECTS = ('Fix the parser (#101)', 'Document the client (#102)')
# ---------------------------------------------------------------------------------


def _module(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LS = _module(SCRIPTS / 'lane_selftest.py', 't20e_lane_selftest')


@pytest.fixture
def w(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    r"""lane_selftest's world on Phase 2, the remote's main advanced by
    :data:`MAIN_SUBJECTS`, and ``T2.0c`` started on the lane."""
    monkeypatch.setattr(LS, 'PHASE', PHASE2)
    monkeypatch.setattr(LS, 'BRANCH', ROUTE)
    monkeypatch.setattr(LS, 'PROTECTED', ('main', 'rust-core', PHASE2))
    world = LS.World(tmp_path)
    world.env['SDM_PHASE_BRANCH'] = PHASE2
    world.main = advance(world, 'main', 'main', *MAIN_SUBJECTS)
    ran = world.lane(
        'start', '--lane', '1b', '--package', PACKAGE,
        '--worktree', str(world.worktree),
    )  # fmt: skip
    assert ran.code == 0, ran.text
    world.git('fetch', '--quiet', 'origin', cwd=world.worktree)
    return world


def advance(w: Any, branch: str, start: str, *subjects: str) -> list[str]:
    r"""Commits ``subjects`` from the seed on the remote's ``start`` and
    pushes them as the remote's ``branch``; their SHAs."""
    w.git('fetch', '--quiet', 'origin', cwd=w.seed)
    w.git('checkout', '--quiet', '-B', f'work-{branch}', f'origin/{start}', cwd=w.seed)  # fmt: skip
    shas = [w.commit(w.seed, subject) for subject in subjects]
    w.git('push', '--quiet', 'origin', f'HEAD:refs/heads/{branch}', cwd=w.seed)  # fmt: skip
    return shas


def merge(
    w: Any, ref: str, subject: str = f'{PACKAGE}: merge origin/main'
) -> str:
    w.git('merge', '--quiet', '--no-ff', '-m', subject, ref, cwd=w.worktree)
    return w.git('rev-parse', 'HEAD', cwd=w.worktree).strip()


def own(w: Any, subject: str = f'{PACKAGE}: the work') -> str:
    return w.commit(w.worktree, subject)


def valid(w: Any) -> str:
    r"""The lane as T2.0c's candidate is: a commit of its own, the merge
    of the remote's main, another commit of its own."""
    own(w, f'{PACKAGE}: before the merge')
    merge(w, 'origin/main')
    return own(w, f'{PACKAGE}: after the merge')


def check(w: Any) -> Any:
    return w.lane('check', cwd=w.worktree)


def pushed(w: Any, sha: str) -> Any:
    refs = w.tmp / 'pushed-refs'
    refs.write_text(f'refs/heads/{ROUTE} {sha} refs/heads/{ROUTE} {w.phase}\n')
    return w.lane('check', '--pushed-refs', str(refs), cwd=w.worktree)


def publish(w: Any) -> str:
    sha = w.git('rev-parse', 'HEAD', cwd=w.worktree).strip()
    w.git('push', '--quiet', '--no-verify', 'origin', f'HEAD:{ROUTE}', cwd=w.worktree)  # fmt: skip
    w.set_runs(green_on=[ROUTE, PHASE2])
    return sha


def land_args(w: Any, sha: str) -> list[str]:
    reviews = []
    for name in ('a', 'b'):
        path = w.tmp / f'review-{name}.txt'
        path.write_text(f'APPROVE\nReviewer {name} approves {sha}.\n')
        reviews += ['--review', str(path)]
    return [
        '--candidate', sha, '--topic-branch', ROUTE,
        '--title', f'Land {PACKAGE}: main merged',
        '--protected-paths', 'none', '--phase-branch', PHASE2, *reviews,
    ]  # fmt: skip


# --- the seeded faults ------------------------------------------------------------


def hidden_behind_the_merge(w: Any) -> str:
    stray = advance(w, 'side', 'main', 'Unrelated change')[0]
    w.git('fetch', '--quiet', 'origin', cwd=w.worktree)
    merge(w, 'origin/side')
    return stray


def a_branch_that_is_not_main(w: Any) -> str:
    w.git('push', '--quiet', 'origin', f'{w.base}:refs/heads/feature-x', cwd=w.seed)  # fmt: skip
    stray = advance(w, 'feature-x', 'feature-x', 'Feature work (#200)')[0]
    w.git('fetch', '--quiet', 'origin', cwd=w.worktree)
    merge(w, 'origin/feature-x')
    return stray


def a_main_commit_rewritten(w: Any) -> str:
    first = w.main[0]
    env = {**w.env, 'GIT_COMMITTER_DATE': '2001-01-01T00:00:00Z'}
    done = LS.run(
        ['git', 'commit-tree', f'{first}^{{tree}}', '-p', f'{first}^',
         '-m', MAIN_SUBJECTS[0]],
        w.worktree, env,
    )  # fmt: skip
    assert done.code == 0, done.text
    stray = done.text.strip()
    assert stray != first
    merge(w, stray)
    return stray


def the_merge_without_the_id(w: Any) -> str:
    own(w)
    return merge(w, 'origin/main', "Merge remote-tracking branch 'origin/main'")


def a_stray_commit_beside_the_merge(w: Any) -> str:
    stray = own(w, 'wip')
    merge(w, 'origin/main')
    own(w)
    return stray


def only_on_a_local_main(w: Any) -> str:
    w.git('merge', '--quiet', '--ff-only', 'origin/main', cwd=w.clone)
    stray = w.commit(w.clone, 'Only on the local main (#300)')
    merge(w, 'refs/heads/main')
    return stray


def two_merges_of_main(w: Any) -> str:
    merge(w, w.main[0], f'{PACKAGE}: merge main, first part')
    return merge(w, 'origin/main', f'{PACKAGE}: merge main, second part')


FAULTS: dict[str, Callable[[Any], str]] = {
    'a non-main commit hidden behind the merge': hidden_behind_the_merge,
    'a merge of a branch that is not main': a_branch_that_is_not_main,
    'a main commit rewritten': a_main_commit_rewritten,
    'the merge commit not titled with the ID': the_merge_without_the_id,
    'a commit without the ID beside the merge': a_stray_commit_beside_the_merge,
    'a commit only the local main holds': only_on_a_local_main,
    'two merges of main': two_merges_of_main,
}


# --- controls -----------------------------------------------------------------------


def test_a_lane_without_a_merge_still_passes(w: Any) -> None:
    r"""Control: the lane's own commits pass ``lane.py check`` as before."""
    own(w)
    ran = check(w)
    assert ran.code == 0, ran.text


@pytest.mark.parametrize('seed', sorted(FAULTS))
def test_each_seed_puts_its_commit_on_the_lane(w: Any, seed: str) -> None:
    r"""Not vacuous (passes on the base): each seed's commit is one the
    lane adds to the phase head and the remote's main does not hold (the
    commits main's exemption must not reach); two merges are two merge
    commits."""
    stray = FAULTS[seed](w)
    added = w.git('rev-list', f'origin/{PHASE2}..HEAD', cwd=w.worktree).split()
    assert stray in added
    remote_main = w.git('ls-remote', 'origin', 'refs/heads/main').split()[0]
    in_main = w.git_ok(
        'merge-base', '--is-ancestor', stray, remote_main, cwd=w.worktree
    )
    assert in_main.code == 1, seed
    merges = w.git(
        'rev-list', '--merges', f'origin/{PHASE2}..HEAD', cwd=w.worktree
    ).split()
    assert len(merges) == (2 if seed == 'two merges of main' else 1), seed


# --- exit 4: lane.py check, and the pre-push hook ---------------------------------


@PENDING
def test_lane_check_passes_a_merge_of_main(w: Any) -> None:
    r"""Exit 4: ``lane.py check`` passes the lane, main's commits (no ID)
    brought in by the ID-titled merge."""
    valid(w)
    log = w.git('log', '--format=%s', f'origin/{PHASE2}..HEAD', cwd=w.worktree)
    assert set(MAIN_SUBJECTS) <= set(log.splitlines())
    ran = check(w)
    assert ran.code == 0, ran.text


@PENDING
def test_the_pre_push_check_passes_a_merge_of_main(w: Any) -> None:
    r"""Exit 4: ``lane.py check --pushed-refs`` (the pre-push hook's
    check) passes the same push."""
    head = valid(w)
    ran = pushed(w, head)
    assert ran.code == 0, ran.text


@PENDING
@pytest.mark.parametrize('seed', sorted(FAULTS))
def test_lane_check_refuses_what_is_not_main_s(w: Any, seed: str) -> None:
    r"""Exit 4's seeded faults, refused by ``lane.py check`` and the
    pre-push check, naming the commit; the twin (the valid merge) passes
    on the same world first."""
    valid(w)
    assert check(w).code == 0
    w.git('reset', '--quiet', '--hard', f'origin/{PHASE2}', cwd=w.worktree)
    stray = FAULTS[seed](w)
    head = w.git('rev-parse', 'HEAD', cwd=w.worktree).strip()
    for ran in (check(w), pushed(w, head)):
        assert ran.code == 2, ran.text
        assert stray[:12] in ran.text, ran.text


# --- exit 4 and 5: land.py ----------------------------------------------------------


@PENDING
def test_land_lands_a_merge_of_main_and_it_verifies(w: Any) -> None:
    r"""Exits 4 and 5: ``land.py`` lands the lane; the phase branch and the
    lane are at the landing, whose parent is the candidate and which holds
    main's commits; ``land.py verify`` passes; the lane, back at the phase
    head, passes ``lane.py check``."""
    valid(w)
    sha = publish(w)
    ran = w.land(*land_args(w, sha))
    assert ran.code == 0, ran.text
    landing = w.tip(PHASE2)
    assert landing == w.tip(ROUTE) != sha
    assert w.git('rev-parse', f'{landing}^', cwd=w.worktree).strip() == sha
    for main_sha in w.main:
        w.git('merge-base', '--is-ancestor', main_sha, landing, cwd=w.worktree)
    ran = w.land('verify', landing, '--phase-branch', PHASE2)
    assert ran.code == 0, ran.text
    assert f'VERIFIED {landing}' in ran.text
    w.git('fetch', '--quiet', 'origin', cwd=w.worktree)
    ran = check(w)
    assert ran.code == 0, ran.text


@PENDING
@pytest.mark.parametrize('seed', sorted(FAULTS))
def test_land_refuses_what_is_not_main_s(w: Any, seed: str) -> None:
    r"""Exit 4's seeded faults, refused by ``land.py`` naming the commit,
    nothing moved; the twin passes ``lane.py check`` first."""
    valid(w)
    assert check(w).code == 0
    w.git('reset', '--quiet', '--hard', f'origin/{PHASE2}', cwd=w.worktree)
    stray = FAULTS[seed](w)
    sha = publish(w)
    before = w.refs()
    ran = w.land(*land_args(w, sha))
    assert ran.code == 2, ran.text
    assert stray[:12] in ran.text, ran.text
    assert w.refs() == before


@PENDING
def test_land_reads_main_from_the_remote_not_a_tracking_ref_moved_by_hand(
    w: Any,
) -> None:
    r"""Exit 4, "ancestors of the remote's main at landing time, fetched,
    never a local ref": the clone's ``refs/remotes/origin/main`` moved by
    hand onto a commit without the ID, which the merge brings in; land.py
    refuses it, naming it, nothing moved. The twin passes ``lane.py check``
    first."""
    valid(w)
    assert check(w).code == 0
    w.git('reset', '--quiet', '--hard', f'origin/{PHASE2}', cwd=w.worktree)
    w.git('checkout', '--quiet', '--detach', 'origin/main', cwd=w.clone)
    stray = w.commit(w.clone, 'Not on the remote (#400)')
    w.git('update-ref', 'refs/remotes/origin/main', stray, cwd=w.clone)
    merge(w, 'origin/main')
    sha = publish(w)
    before = w.refs()
    ran = w.land(*land_args(w, sha))
    assert ran.code == 2, ran.text
    assert stray[:12] in ran.text, ran.text
    assert w.refs() == before


def test_a_phase_1_landing_still_verifies(tmp_path: Path) -> None:
    r"""Control, exit 5: a landing made on ``rust-core-phase-1`` still
    verifies, while the setting names Phase 2."""
    world = LS.World(tmp_path)
    assert LS.PHASE == PHASE1
    sha = LS.candidate(world)
    ran = world.land(*LS.land_args(world, sha, 'Land T1.11e: the work'))
    assert ran.code == 0, ran.text
    landing = world.tip(PHASE1)
    world.env['SDM_PHASE_BRANCH'] = PHASE2
    ran = world.land('verify', landing, '--phase-branch', PHASE1)
    assert ran.code == 0, ran.text
    assert f'VERIFIED {landing}' in ran.text
