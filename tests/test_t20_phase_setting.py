# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.0's acceptance tests for exits 1 and 6 (the independent test
author's): the phase branch is one setting, and Phase 2's package IDs pass.

Exit 1: every landing tool and CI pattern takes the phase from one setting,
so ``rust-core-phase-2`` works end to end: ``lane.py start`` makes a Phase 2
lane, ``land.py`` lands on ``rust-core-phase-2`` (by ``--phase-branch`` and
by ``SDM_PHASE_BRANCH`` alone), ``ratchet.py where`` names Phase 2's bases,
ci.yml's gate, skip, phase-check, concurrency and cache rules treat Phase 2
branches as they treat Phase 1's, and ``prepush.sh``, ``prune_caches.py``,
``replay_jobs.py`` and ``bridge_freeze.py`` default to the configured
phase. Negative controls: a Phase 1 lane under a Phase 2 setting, and a
phase branch that is no phase branch, are refused by name; a Phase 1
landing still verifies.

Exit 6: ``T2.0``, ``T2.0a``, ``T2.15`` and ``CI5`` pass ``land.py``,
``lane.py`` and ``ratchet.py``; malformed IDs are refused. The ID checks
already pass on the base (the brief asks to confirm them); the end-to-end
landing of each ID on Phase 2 fails there.

Real git everywhere, with ``gh`` faked: ``scripts/rust_core/lane_selftest.py``'s
``World`` (a bare remote, a clone, a fake ``gh``), with its phase set to
Phase 2; ``tests/test_ci_workflow.py``'s and
``tests/test_contract_ci_gate.py``'s harnesses run ci.yml's step texts.

The interface assumed, in one place (:data:`PHASE_ENV` and the notes on
it): see the constants below.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts/rust_core'
CI = ROOT / '.github/workflows/ci.yml'

# --- the interface assumed (the implementer matches it, or asks) ---------------
#
#: The setting every tool reads the phase branch from when no flag gives
#: it: the variable the orchestration tools already require
#: (scripts/rust_core/orchestration/README.md). ``land.py``, ``lane.py``,
#: ``ratchet.py`` (``where``), ``prepush.sh``, ``prune_caches.py``,
#: ``replay_jobs.py`` and ``bridge_freeze.py``/``l0_deps.py`` read it.
PHASE_ENV = 'SDM_PHASE_BRANCH'
#: ``land.py --phase-branch`` overrides the setting, as today's flag reads.
#: ``ratchet.py where`` keeps its output's form: ``--base origin/<phase>``
#: on a lane and ``--phase-push <before>`` on the phase branch.
#: ci.yml's conditions stay GitHub expressions over ``github.*`` contexts.
PHASE1 = 'rust-core-phase-1'
PHASE2 = 'rust-core-phase-2'
LANE2 = f'{PHASE2}-lane-1b'
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


LS = _module(SCRIPTS / 'lane_selftest.py', 't20_lane_selftest')
WF = _module(ROOT / 'tests/test_ci_workflow.py', 't20_ci_workflow')
GATE = _module(ROOT / 'tests/test_contract_ci_gate.py', 't20_ci_gate')


# --- lane.py and land.py on Phase 2 (exit 1) --------------------------------------


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    r"""lane_selftest's world with its phase branch ``rust-core-phase-2``,
    and the setting naming it."""
    monkeypatch.setattr(LS, 'PHASE', PHASE2)
    monkeypatch.setattr(LS, 'BRANCH', LANE2)
    monkeypatch.setattr(LS, 'PROTECTED', ('main', 'rust-core', PHASE2))
    w = LS.World(tmp_path)
    w.env[PHASE_ENV] = PHASE2
    return w


def start(w: Any, package: str) -> Any:
    return w.lane(
        'start', '--lane', '1b', '--package', package,
        '--worktree', str(w.worktree),
    )  # fmt: skip


def candidate(w: Any, package: str, subject: str | None = None) -> str:
    r"""``package`` started on the Phase 2 lane, one commit of its own
    pushed, green on the lane and the phase branch."""
    ran = start(w, package)
    assert ran.code == 0, ran.text
    sha = w.commit(w.worktree, subject or f'{package}: the work')
    w.git('push', '--quiet', 'origin', f'HEAD:{LANE2}', cwd=w.worktree)
    w.set_runs(green_on=[LANE2, PHASE2])
    return sha


def land_args(w: Any, sha: str, package: str, *extra: str) -> list[str]:
    reviews = []
    for name in ('a', 'b'):
        path = w.tmp / f'review-{name}.txt'
        path.write_text(f'APPROVE\nReviewer {name} approves {sha}.\n')
        reviews += ['--review', str(path)]
    return [
        '--candidate', sha, '--topic-branch', LANE2,
        '--title', f'Land {package}: the work',
        '--protected-paths', 'none', *reviews, *extra,
    ]  # fmt: skip


def test_lane_start_makes_a_phase_2_lane_at_the_phase_2_head(
    world: Any,
) -> None:
    r"""Exit 1: ``lane.py`` under the setting starts a lane of Phase 2."""
    ran = start(world, 'T2.0')
    assert ran.code == 0, ran.text
    assert world.tip(LANE2) == world.phase
    assert world.package(world.worktree) == 'T2.0'
    assert world.tip(f'{PHASE1}-lane-1b') is None


def test_land_lands_on_phase_2_with_the_flag(world: Any) -> None:
    r"""Exit 1: ``land.py --phase-branch rust-core-phase-2`` lands, the
    phase branch and the lane both at the landing commit."""
    sha = candidate(world, 'T2.0')
    ran = world.land(*land_args(world, sha, 'T2.0', '--phase-branch', PHASE2))
    assert ran.code == 0, ran.text
    landing = world.git('rev-parse', 'HEAD', cwd=world.worktree).strip()
    assert landing != sha
    assert world.tip(PHASE2) == world.tip(LANE2) == landing
    assert world.tip(PHASE1) is None


def test_land_takes_the_phase_from_the_setting_alone(world: Any) -> None:
    r"""Exit 1, "the override actually takes effect": with no
    ``--phase-branch``, ``SDM_PHASE_BRANCH`` decides where ``land.py``
    lands."""
    sha = candidate(world, 'T2.0')
    ran = world.land(*land_args(world, sha, 'T2.0'))
    assert ran.code == 0, ran.text
    landing = world.git('rev-parse', 'HEAD', cwd=world.worktree).strip()
    assert world.tip(PHASE2) == landing


def test_a_phase_1_lane_is_refused_under_the_phase_2_setting(
    world: Any,
) -> None:
    r"""Seeded: the topic branch is Phase 1's lane while the phase is
    Phase 2: refused for its topic branch, nothing moved."""
    sha = candidate(world, 'T2.0')
    before = world.refs()
    args = land_args(world, sha, 'T2.0', '--phase-branch', PHASE2)
    args[args.index('--topic-branch') + 1] = f'{PHASE1}-lane-1b'
    ran = world.land(*args)
    assert ran.code == 2, ran.text
    assert 'topic branch' in ran.text, ran.text
    assert world.refs() == before


def test_a_phase_2_lane_is_refused_under_the_phase_1_setting(
    world: Any,
) -> None:
    r"""Seeded, the other way: the setting says Phase 1, so Phase 2's lane
    is not a topic branch of the phase and nothing lands."""
    sha = candidate(world, 'T2.0')
    before = world.refs()
    world.env[PHASE_ENV] = PHASE1
    ran = world.land(*land_args(world, sha, 'T2.0'))
    assert ran.code == 2, ran.text
    assert world.refs() == before


@pytest.mark.parametrize(
    'phase',
    [
        'main',
        'rust-core',
        f'{PHASE2}-lane-1b',
        f'{PHASE2}-archive-1b',
        'rust-core-phase-',
        'rust-core-phase-x',
        'rust-core-phase-2a',
    ],
)
def test_a_phase_branch_that_is_no_phase_branch_is_refused(
    world: Any, phase: str
) -> None:
    r"""Seeded: only ``rust-core-phase-<N>`` is a phase branch."""
    sha = candidate(world, 'T2.0')
    before = world.refs()
    ran = world.land(*land_args(world, sha, 'T2.0', '--phase-branch', phase))
    assert ran.code == 2, ran.text
    assert 'phase branch' in ran.text, ran.text
    assert world.refs() == before


def test_a_phase_2_landing_verifies(world: Any) -> None:
    r"""Exit 1: ``land.py verify`` on a Phase 2 landing."""
    sha = candidate(world, 'T2.0')
    ran = world.land(*land_args(world, sha, 'T2.0', '--phase-branch', PHASE2))
    assert ran.code == 0, ran.text
    landing = world.tip(PHASE2)
    ran = world.land('verify', landing, '--phase-branch', PHASE2)
    assert ran.code == 0, ran.text
    assert f'VERIFIED {landing}' in ran.text


def test_a_phase_1_landing_still_verifies_under_the_phase_2_setting(
    tmp_path: Path,
) -> None:
    r"""Control, exit 1: "Landing SHAs from Phase 1 must still verify":
    a landing made on ``rust-core-phase-1`` verifies with
    ``--phase-branch rust-core-phase-1`` while the setting names Phase 2.
    (It passes on the base too; it guards the change.)"""
    w = LS.World(tmp_path)
    assert LS.PHASE == PHASE1
    sha = LS.candidate(w)
    ran = w.land(*LS.land_args(w, sha, 'Land T1.11e: the work'))
    assert ran.code == 0, ran.text
    landing = w.tip(PHASE1)
    w.env[PHASE_ENV] = PHASE2
    ran = w.land('verify', landing, '--phase-branch', PHASE1)
    assert ran.code == 0, ran.text
    assert f'VERIFIED {landing}' in ran.text


# --- package IDs (exit 6) ----------------------------------------------------------

#: Phase 2's IDs the brief names, and IDs that are none.
IDS = ('T2.0', 'T2.0a', 'T2.15', 'CI5')
NOT_IDS = ('T2', 'T2.', 'T2.0A', 'T2.0ab', 'T2.0.1', 'CI', 'C5', 't2.0', 'X2.0')


@pytest.mark.parametrize('package', IDS)
def test_phase_2_ids_are_package_ids_in_each_tool(package: str) -> None:
    r"""Exit 6: confirmed (passes on the base): each tool's ID pattern, and
    the ratchet's landing subject, take the ID."""
    land = _module(SCRIPTS / 'land.py', 'land')
    lane = _module(SCRIPTS / 'lane.py', 't20_lane')
    ratchet = _module(SCRIPTS / 'ratchet.py', 't20_ratchet')
    assert land.PACKAGE_ID.fullmatch(package)
    assert lane.package_id(package) == package
    assert land.owner_of(f'{package}: the work') == package
    assert land.owner_of(f'{package} fix 1: the review') == package
    assert land.LANDING_SUBJECT.match(f'Land {package}: the work')
    assert ratchet.LANDING.match(f'Land {package}: the work')


@pytest.mark.parametrize('package', NOT_IDS)
def test_malformed_ids_are_refused(package: str) -> None:
    r"""Exit 6's negative controls."""
    land = _module(SCRIPTS / 'land.py', 'land')
    lane = _module(SCRIPTS / 'lane.py', 't20_lane')
    assert not land.PACKAGE_ID.fullmatch(package)
    with pytest.raises(land.Refusal):
        lane.package_id(package)


@pytest.mark.parametrize('package', IDS)
def test_each_phase_2_id_lands_on_phase_2(world: Any, package: str) -> None:
    r"""Exit 6, end to end: each ID is started, owns its commits and lands
    on Phase 2."""
    sha = candidate(world, package)
    ran = world.land(*land_args(world, sha, package, '--phase-branch', PHASE2))
    assert ran.code == 0, ran.text
    assert world.tip(PHASE2) == world.tip(LANE2) != sha


# --- ratchet.py where (exit 1) -------------------------------------------------------


def where(environ: dict[str, str]) -> list[str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(('GITHUB_', 'SDM_'))
    }
    done = subprocess.run(
        [sys.executable, str(SCRIPTS / 'ratchet.py'), 'where'],
        env={**env, **environ},
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout.split()


def test_where_names_phase_2_s_bases(tmp_path: Path) -> None:
    r"""Exit 1: on a push to Phase 2's lane, its base; on a push to the
    Phase 2 branch, its previous tip; in a local replay, the setting's."""
    event = tmp_path / 'event.json'
    before = 'a' * 40
    event.write_text(json.dumps({'before': before}))
    push = {
        'GITHUB_EVENT_NAME': 'push',
        'GITHUB_EVENT_PATH': str(event),
        PHASE_ENV: PHASE2,
    }
    assert where({**push, 'GITHUB_REF_NAME': LANE2}) == [
        '--base',
        f'origin/{PHASE2}',
    ]
    assert where({**push, 'GITHUB_REF_NAME': f'{PHASE2}-t2.0'}) == [
        '--base',
        f'origin/{PHASE2}',
    ]
    assert where({**push, 'GITHUB_REF_NAME': PHASE2}) == [
        '--phase-push',
        before,
    ]
    assert where({PHASE_ENV: PHASE2}) == ['--base', f'origin/{PHASE2}']
    # Controls: no ratchet on main, nor on a pull request's run.
    assert where({**push, 'GITHUB_REF_NAME': 'main'}) == []
    assert (
        where(
            {
                **push,
                'GITHUB_REF_NAME': LANE2,
                'GITHUB_EVENT_NAME': 'pull_request',
            }
        )
        == []
    )


# --- ci.yml (exit 1) -------------------------------------------------------------------


def test_ci_yml_names_no_phase_1_branch_outside_comments() -> None:
    r"""Exit 1: ci.yml's phase is a pattern, not Phase 1's name: no line of
    it but a comment names ``rust-core-phase-1``."""
    named = [
        f'{number}: {line.strip()}'
        for number, line in enumerate(
            CI.read_text(encoding='utf-8').splitlines(), 1
        )
        if PHASE1 in line and not line.lstrip().startswith('#')
    ]
    assert named == [], '\n'.join(named)


def _pushed(branch: str) -> bool:
    r"""Whether a push to ``branch`` runs ci.yml, by its branch filters
    (GitHub's order: the last pattern matching decides)."""
    on = yaml.safe_load(CI.read_text(encoding='utf-8'))
    # PyYAML reads the key ``on`` as True.
    trigger = on.get('on', on.get(True))
    decided = False
    for pattern in trigger['push']['branches']:
        negated = pattern.startswith('!')
        if fnmatch.fnmatchcase(branch, pattern.lstrip('!')):
            decided = not negated
    return decided


@pytest.mark.parametrize(
    ('branch', 'runs'),
    [
        (PHASE2, True),
        (LANE2, True),
        (f'{PHASE2}-t2.0', True),
        (f'{PHASE2}-archive-1b-T2.0-abc1234', False),
        (f'{PHASE1}-archive-1a-T1.20-e587c32', False),
        ('rust-core', True),
    ],
)
def test_ci_runs_on_phase_2_pushes_but_not_its_archives(
    branch: str, runs: bool
) -> None:
    r"""Exit 1: Phase 2's archive branches are left out as Phase 1's are."""
    assert _pushed(branch) is runs


def _evaluate(condition: object, context: dict[str, str]) -> bool:
    r"""``tests/test_ci_workflow.py``'s evaluator, with ci.yml's one use of
    ``contains(fromJSON('[...]'), X)`` read as X equal to one of them."""

    def listed(match: re.Match[str]) -> str:
        names = json.loads(match['names'])
        return (
            '('
            + ' || '.join(f"{match['what']} == '{name}'" for name in names)
            + ')'
        )

    text = re.sub(
        r"contains\(fromJSON\('(?P<names>\[[^']*\])'\), (?P<what>[\w.]+)\)",
        listed,
        str(condition),
    )
    found: bool = WF._evaluate(text, context)
    return found


@pytest.mark.parametrize(
    ('ref', 'cancels'),
    [
        (PHASE2, False),
        (LANE2, False),
        (f'{PHASE2}-t2.0', True),
        ('main', True),
    ],
)
def test_a_phase_2_run_in_progress_is_kept_as_phase_1_s(
    ref: str, cancels: bool
) -> None:
    r"""Exit 1, ``cancel-in-progress``: the phase branch's and its lanes'
    runs are never cancelled (the evidence a landing stands on)."""
    condition = WF._workflow()['concurrency']['cancel-in-progress']
    context = {
        'github.event_name': 'push',
        'github.ref_name': ref,
        'github.ref': f'refs/heads/{ref}',
    }
    assert _evaluate(condition, context) is cancels, condition


@pytest.mark.parametrize(
    ('ref', 'saves'),
    [
        (LANE2, True),
        (PHASE2, False),
        (f'{PHASE2}-t2.0', False),
        ('rust-core', True),
    ],
)
def test_phase_2_lanes_save_caches_as_phase_1_s_did(
    ref: str, saves: bool
) -> None:
    r"""Exit 1, ``SAVE_CACHES``: the lanes save, the phase branch and topic
    branches do not."""
    condition = WF._workflow()['env']['SAVE_CACHES']
    context = {
        'github.event_name': 'push',
        'github.ref_name': ref,
        'github.ref': f'refs/heads/{ref}',
    }
    assert _evaluate(condition, context) is saves, condition


@pytest.mark.parametrize(
    ('event', 'ref', 'runs'),
    [
        ('push', PHASE2, True),
        ('push', LANE2, False),
        ('push', f'{PHASE2}-archive-1b-t2.0', False),
        ('pull_request', PHASE2, False),
    ],
)
def test_the_phase_check_runs_on_a_phase_2_push(
    event: str, ref: str, runs: bool
) -> None:
    r"""Exit 1: the landing job's phase check (``ratchet.py checkpoint
    --phase-push``) runs on every push to Phase 2's branch."""
    _, step = WF._phase_check_step()
    context = {
        'github.event_name': event,
        'github.ref_name': ref,
        'github.ref': f'refs/heads/{ref}',
        'github.base_ref': '',
        'github.head_ref': '',
    }
    assert _evaluate(step.get('if', ''), context) is runs


@pytest.mark.parametrize(
    ('job', 'step', 'job_env'),
    GATE.STEPS,
    ids=[name for name, _, _ in GATE.STEPS],
)
@pytest.mark.parametrize('ref', [PHASE2, LANE2])
def test_on_phase_2_the_gate_cannot_be_skipped(
    tmp_path: Path,
    job: str,
    step: dict[str, Any],
    job_env: dict[str, Any],
    ref: str,
) -> None:
    r"""Exit 1: on a push to Phase 2's branch or a branch under it an empty
    ``ratchet.py where`` fails each gate step, as on Phase 1's."""
    done, called = GATE._run(tmp_path, step, job_env, '', 'push', ref)
    assert done.returncode != 0, done.stdout + done.stderr
    assert not called.exists()


@pytest.mark.parametrize('ref', [PHASE2, LANE2])
def test_a_phase_2_landing_is_skipped_on_its_candidate_s_green_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ref: str
) -> None:
    r"""Exit 1: ci.yml's skip step treats a landing pushed to Phase 2 as
    one pushed to Phase 1: skipped on its candidate's green run on the
    lane, and run in full without one (the seeded control)."""
    monkeypatch.setattr(WF, 'PHASE', PHASE2)
    for name in ('green', 'untested'):
        (tmp_path / name).mkdir()
    clone, shas = WF._untested_push(tmp_path / 'green', ref, 1)
    green = [WF._run(shas['C'], 1, branch=LANE2)]
    found, calls = WF._decide(clone, tmp_path, 'push', ref, green)
    assert found == 'true', (found, calls)
    clone, shas = WF._untested_push(tmp_path / 'untested', ref, 1)
    found, calls = WF._decide(clone, tmp_path, 'push', ref, [])
    assert found in ('false', 'failed'), (found, calls)


def test_the_phase_check_judges_a_phase_2_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""Exit 1: the phase check's step text, on a push landing one package
    on Phase 2, resolves its checkpoint (the previous tip); a push of an
    unlanded commit there fails (the seeded control)."""
    monkeypatch.setattr(WF, 'PHASE', PHASE2)
    _, step = WF._phase_check_step()
    for name in ('sound', 'unlanded'):
        (tmp_path / name).mkdir()
    phase = WF._Phase(tmp_path / 'sound')
    before = WF._phase_scenario(phase, 'sound')
    done, _ = phase.run(before, str(step['run']))
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.split()[:2] == [before, before], done.stdout
    phase = WF._Phase(tmp_path / 'unlanded')
    before = WF._phase_scenario(phase, 'not-a-landing')
    done, _ = phase.run(before, str(step['run']))
    assert done.returncode != 0, done.stdout + done.stderr


# --- the other tools (exit 1) ---------------------------------------------------------


def _python(code: str, **environ: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith('SDM_')}
    done = subprocess.run(
        [sys.executable, '-c', code],
        cwd=SCRIPTS,
        env={**env, **environ},
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout.strip()


def test_the_freeze_base_follows_the_setting() -> None:
    r"""Exit 1: ``bridge_freeze.py`` (and ``l0_deps.py``, which takes its
    base from it) default to the configured phase; ``SDM_FREEZE_BASE``
    still wins (control)."""
    code = 'import bridge_freeze; print(bridge_freeze.base_ref())'
    assert _python(code, **{PHASE_ENV: PHASE2}) == f'origin/{PHASE2}'
    assert (
        _python(code, **{PHASE_ENV: PHASE2, 'SDM_FREEZE_BASE': 'x/y'}) == 'x/y'
    )


def test_prune_caches_keeps_phase_2_lanes_and_prunes_its_landed_topics() -> (
    None
):
    r"""Exit 1: ``prune_caches.py``'s stale list under the Phase 2 setting:
    a lane's cache is kept, a landed Phase 2 topic branch's is pruned."""
    code = (
        'import json, prune_caches\n'
        f'lane = {{"ref": "refs/heads/{LANE2}", "key": "k1", "id": 1}}\n'
        f'topic = {{"ref": "refs/heads/{PHASE2}-t2.0", "key": "k2", "id": 2}}\n'
        'found = prune_caches.stale([lane, topic], lambda b: True, lambda b: True)\n'
        'print(json.dumps([[c["id"], why] for c, why in found]))\n'
    )
    found = json.loads(_python(code, **{PHASE_ENV: PHASE2}))
    assert [entry[0] for entry in found] == [2], found
    assert PHASE2 in found[0][1], found


def test_replay_jobs_measures_from_the_configured_phase() -> None:
    r"""Exit 1: ``replay_jobs.py``'s default base is the merge base with
    the configured phase (its ``--help`` names it)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith('SDM_')}
    done = subprocess.run(
        [sys.executable, str(SCRIPTS / 'replay_jobs.py'), '--help'],
        env={**env, PHASE_ENV: PHASE2},
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert f'origin/{PHASE2}' in done.stdout, done.stdout
    assert PHASE1 not in done.stdout, done.stdout


def test_prepush_measures_a_phase_2_lane_from_phase_2(tmp_path: Path) -> None:
    r"""Exit 1: ``prepush.sh``'s cheap tier on a Phase 2 lane that changes
    no tool skips the self-tests, naming its merge base with Phase 2."""
    prepush = _module(ROOT / 'tests/test_prepush_selftests.py', 't20_prepush')
    repo, env = prepush._lane(tmp_path)
    git = {'cwd': repo, 'env': env, 'check': True, 'capture_output': True}
    subprocess.run(['git', 'branch', '-m', PHASE1, PHASE2], **git)
    subprocess.run(['git', 'branch', '-m', f'{PHASE1}-lane-1a', LANE2], **git)
    env[PHASE_ENV] = PHASE2
    out, ran = prepush._prepush(repo, env)
    assert 'lane_selftest.py' not in ran, ran
    skipped = [line for line in out.splitlines() if line.startswith('skipped')]
    assert len(skipped) == 1, out
    assert PHASE2 in skipped[0], skipped
