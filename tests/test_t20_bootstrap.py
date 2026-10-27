# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.0's bootstrap and the epic branch's protected changes (the
implementer's own tests; the acceptance tests are
``contract/tests/test_t20_checkpoints.py``).

The bootstrap: Phase 2's fork point is ``rust-core``'s head, no landing
commit, and its own ``ratchet.py`` is Phase 1's, which refuses any
checkpoint that is no landing commit, itself included. ``gate.sh`` then
judges with Phase 1's last landing, for that one pinned fork point alone
and only when every condition it reads from git holds. One temporary
history, shaped as the real one: ``rust-core-phase-1``'s landings
(``tests/test_contract_ratchet.py``'s ``build()``), the last landing
holding a ``ratchet.py`` that knows no fork point (as Phase 1's), then
``rust-core`` merging them, a handbook commit and the fork point. gate.sh's
pin is one line nothing overrides; the fixture's candidates carry a copy of
gate.sh with that line rewritten to the fixture's fork point and stand-in.
One test runs the real pin on the real history.

- T2.0's candidate on the fork point and T2.0's first landing itself (a
  replay of its tip and its phase push) are judged by the stand-in's
  harness, which the gate says.
- Seeded, each refused: a candidate above a landing with the fork point as
  its base (the reviewer's round 1 repro); a candidate whose
  ``ratchet.py`` names another non-landing checkpoint (Codex's round 1
  repro: it gets that commit's own harness, never the stand-in); a fork
  point that differs from the stand-in in a judging file; a remote epic
  branch whose fork point is not the pinned one; a first landing recording
  another Checkpoint-SHA; another package's candidate, first landing and
  phase push (Codex's round 2 P2-2); a first landing with no Reviewed-by.

The epic branch (rule ``epic``, ``--epic``, as ``where`` names it for a push
to ``rust-core``): every protected path, by the checkpoint L's list or the
head's, has the same tree entry (presence, mode, type and object ID) in
the head as in L, read with ``git ls-tree``, which no configuration
changes. Fixtures: Phase 2 cut at the fork point, a landing L2, and
``rust-core`` merging it ``--no-ff``. A clean owner merge, and a head
differing from L2 only in unprotected paths, pass. Seeded, each failing: a
direct protected edit under any name; a mode-only change; a shrunk
protected list; a protected submodule change git is told to ignore; an
epic edit carried through the landing's rename; an earlier merge's kept
epic version after a later landing; the real rust-core against Phase 1's
landing. ``where`` refuses a landing commit on rust-core's first-parent
history (a fast-forward, or a landing-shaped head) and a merged landing
record with no Reviewed-by; ``check --epic`` refuses a checkpoint other
than the merged landing; a run without ``--epic`` on the epic head says
the rule did not run.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

# Gate tooling (T3.0): CI runs it in the tooling jobs, not in every matrix job.
pytestmark = pytest.mark.tooling

ROOT = Path(__file__).resolve().parents[1]
RATCHET = ROOT / 'scripts/rust_core/ratchet.py'
PHASE1, PHASE2, EPIC = 'rust-core-phase-1', 'rust-core-phase-2', 'rust-core'
#: A ratchet that refuses every checkpoint.
REFUSING = (
    'import sys\n'
    "print('checkpoint: the checkpoint is not a landing commit', "
    'file=sys.stderr)\n'
    'sys.exit(1)\n'
)
#: The real bootstrap, as gate.sh pins it.
REAL_FORK = '9604074af96b8a05a5532c91249470595b38a4c5'
REAL_STAND = 'bd44b34270ca59679ddf1fb4104d8aaa0373ddfb'
#: Phase 1's ratchet, as the stand-in and the fork point hold it: this
#: tree's, but knowing no fork point, so it refuses any checkpoint that is
#: no landing commit, the fork point included.
FORK_LINE = '        fork, read = fork_point(repo, phase, epic)\n'
OLD_RATCHET = (ROOT / 'scripts/rust_core/ratchet.py').read_text(
    encoding='utf-8'
)
assert OLD_RATCHET.count(FORK_LINE) == 1
OLD_RATCHET = OLD_RATCHET.replace(
    FORK_LINE, "        fork, read = None, 'Phase 1 knew no fork point'\n"
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


def env(**extra: str) -> dict[str, str]:
    kept = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(('GITHUB_', 'SDM_', 'GIT_'))
    }
    return {**kept, 'PYTHON': sys.executable, **extra}


def write(path: str, text: str) -> Callable[[Path], None]:
    def edit(repo: Path) -> None:
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text, encoding='utf-8')

    return edit


def world(
    base: Path,
    fork_edit: Callable[[Path], None] | None = None,
    handbook: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    r"""The history above in a clone of a bare origin; the SHAs by name.
    ``fork_edit`` changes the fork point beside its refusing ratchet."""
    work = base / 'work'
    work.mkdir()
    helpers.build(work)
    # Phase 1's last landing holds the ratchet that knows no fork point.
    helpers.land(
        work,
        'T1.99',
        write('scripts/rust_core/ratchet.py', OLD_RATCHET),
        protected='scripts/rust_core/**',
    )
    shas = {'L1': git(work, 'rev-parse', 'HEAD')}
    root = git(work, 'rev-list', '--max-parents=0', 'HEAD')
    tree = git(work, 'rev-parse', root + '^{tree}')
    shas['R0'] = git(work, 'commit-tree', tree, '-m', 'Merge Phase 0')
    git(work, 'checkout', '-q', '-b', EPIC, shas['R0'])
    git(
        work, 'merge', '-q', '--no-ff', '--allow-unrelated-histories',
        '-m', 'Merge Phase 1: the harness', PHASE1,
    )  # fmt: skip
    shas['M'] = git(work, 'rev-parse', 'HEAD')
    shas['H1'] = commit(
        work,
        'Handoff: the handbook',
        handbook or write('docs/design/rust-core/handbook.md', '# h\n'),
    )

    def fork(repo: Path) -> None:
        (repo / 'fork.txt').write_text('the fork point\n')
        if fork_edit is not None:
            fork_edit(repo)

    shas['F'] = commit(work, 'Handoff: the fork point', fork)
    git(work, 'branch', PHASE2, shas['F'])
    origin = base / 'origin.git'
    git(base, 'init', '-q', '--bare', str(origin))
    git(work, 'push', '-q', str(origin), PHASE1, EPIC, PHASE2)
    clone = base / 'clone'
    git(base, 'clone', '-q', '--no-checkout', str(origin), str(clone))
    git(clone, 'checkout', '-q', '--detach', shas['F'])
    return {'clone': clone, 'shas': shas}


GATE = ROOT / 'scripts/rust_core/gate.sh'
#: gate.sh's pin, the one line the fixture's copy rewrites.
PIN = (
    "bootstrap='T2.0 rust-core-phase-2 9604074af96b8a05a5532c91249470595b38a4c5 "
    "bd44b34270ca59679ddf1fb4104d8aaa0373ddfb'\n"
)


def pinned_gate(shas: dict[str, str]) -> Callable[[Path], None]:
    r"""This tree's gate.sh, its pin rewritten to the fixture's fork point
    and stand-in: a disposable copy in the candidate's commit, as nothing
    overrides the pin."""
    text = GATE.read_text(encoding='utf-8')
    assert text.count(PIN) == 1
    copy = text.replace(
        PIN, f"bootstrap='T2.0 {PHASE2} {shas['F']} {shas['L1']}'\n"
    )

    def edit(repo: Path) -> None:
        path = repo / 'scripts/rust_core/gate.sh'
        path.write_text(copy, encoding='utf-8')
        path.chmod(0o755)

    return edit


def work_for(shas: dict[str, str], name: str = 't20') -> Callable[[Path], None]:
    r"""The candidate's change: this tree's ratchet.py, as T2.0's, and the
    pinned gate.sh copy."""

    def edit(repo: Path) -> None:
        (repo / f'{name}.txt').write_text('x')
        (repo / 'scripts/rust_core/ratchet.py').write_bytes(
            RATCHET.read_bytes()
        )
        pinned_gate(shas)(repo)

    return edit


def landing(
    repo: Path,
    base: str,
    checkpoint: str | None = None,
    package: str = 'T2.0',
    reviewed: bool = True,
) -> str:
    return commit(
        repo,
        f'Land {package}: the package',
        trailers={
            **(
                {'Reviewed-by': f'independent reviewer and Codex, {package}'}
                if reviewed
                else {}
            ),
            'Protected-paths': 'scripts/rust_core/**',
            'Base-SHA': base,
            'Checkpoint-SHA': checkpoint or base,
        },
    )


def gate(
    clone: Path, shas: dict[str, str], *arguments: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(clone / 'scripts/rust_core/gate.sh'), *arguments],
        capture_output=True,
        text=True,
        check=False,
        cwd=clone,
        env=env(SDM_PHASE_BRANCH=PHASE2),
        stdin=subprocess.DEVNULL,
    )


def stood_in(
    done: subprocess.CompletedProcess[str], shas: dict[str, str]
) -> bool:
    return (
        done.returncode == 0
        and f'last landing {shas["L1"]} stands in' in done.stderr
        and f'checkpoint {shas["L1"]}' in done.stdout
        and 'gate: passed (--only ratchet' in done.stdout
    )


@pytest.fixture(scope='module')
def built(owned_module_dir: Callable[[str], Path]) -> dict[str, Any]:
    return world(owned_module_dir('t20-bootstrap'))


def at_fork(built: dict[str, Any]) -> tuple[Path, dict[str, str]]:
    clone, shas = built['clone'], built['shas']
    git(clone, 'checkout', '-q', '--detach', shas['F'])
    return clone, shas


def test_the_fork_point_s_candidate_is_judged_by_the_stand_in(
    built: dict[str, Any],
) -> None:
    clone, shas = at_fork(built)
    commit(clone, 'T2.0: the work', work_for(shas))
    done = gate(clone, shas, '--base', f'origin/{PHASE2}', '--only', 'ratchet')
    assert stood_in(done, shas), done.stdout + done.stderr


def test_the_first_landing_its_tip_and_its_phase_push_are_judged(
    built: dict[str, Any],
) -> None:
    r"""Codex's P2: the empty ``Land T2.0`` commit (Base-SHA and
    Checkpoint-SHA the fork point), replayed as the phase head and as the
    phase push from the fork point, is judged by the stand-in, as its
    parent's tree under its trailers."""
    clone, shas = at_fork(built)
    commit(clone, 'T2.0: the work', work_for(shas))
    landing(clone, shas['F'])
    for where in (['--base', f'origin/{PHASE2}'], ['--phase-push', shas['F']]):
        done = gate(clone, shas, *where, '--only', 'ratchet')
        assert stood_in(done, shas), (where, done.stdout + done.stderr)


def test_a_first_landing_recording_another_checkpoint_is_refused(
    built: dict[str, Any],
) -> None:
    clone, shas = at_fork(built)
    commit(clone, 'T2.0: the work', work_for(shas))
    landing(clone, shas['F'], checkpoint=shas['H1'])
    done = gate(clone, shas, '--base', f'origin/{PHASE2}', '--only', 'ratchet')
    assert done.returncode == 1, done.stdout + done.stderr
    assert "first landing's Checkpoint-SHA is not the fork point" in done.stderr
    assert 'stands in' not in done.stderr


def test_a_candidate_above_a_landing_with_the_fork_point_as_base_is_refused(
    built: dict[str, Any],
) -> None:
    r"""The reviewer's P2-1 repro: once T2.0 has landed, a candidate on
    that landing given ``--base`` the fork point is never judged by the
    stand-in."""
    clone, shas = at_fork(built)
    commit(clone, 'T2.0: the work', work_for(shas))
    landing(clone, shas['F'])
    commit(clone, 'T2.0a: the design', write('design.txt', 'd'))
    done = gate(clone, shas, '--base', shas['F'], '--only', 'ratchet')
    assert done.returncode == 1, done.stdout + done.stderr
    assert 'holds a landing commit after the fork point' in done.stderr
    assert 'stands in' not in done.stderr


def test_a_candidate_ratchet_naming_another_checkpoint_never_gets_the_stand_in(
    built: dict[str, Any],
) -> None:
    r"""Codex's P1 repro: after the first landing, a non-landing commit X
    whose ratchet refuses everything, and a candidate whose ratchet names X
    as its checkpoint: X's own harness judges, and refuses; the stand-in is
    not the candidate's to choose."""
    clone, shas = at_fork(built)
    commit(clone, 'T2.0: the work', work_for(shas))
    landed = landing(clone, shas['F'])
    x = commit(
        clone,
        'T2.1: X refuses',
        write('scripts/rust_core/ratchet.py', REFUSING),
    )
    fake = (
        'import sys\n'
        f"if sys.argv[1:2] == ['checkpoint']: print('{x} {landed} harness'); sys.exit(0)\n"
        f"print('{shas['L1']}'); sys.exit(0)\n"
    )
    commit(
        clone,
        'T2.1: the candidate',
        write('scripts/rust_core/ratchet.py', fake),
    )
    done = gate(clone, shas, '--base', landed, '--only', 'ratchet')
    assert done.returncode != 0, done.stdout + done.stderr
    assert 'stands in' not in done.stderr
    assert 'gate: passed' not in done.stdout
    # And naming the pinned fork point from above the landing is refused.
    fake = (
        f"import sys\nprint('{shas['F']} {shas['F']} harness'); sys.exit(0)\n"
    )
    commit(
        clone,
        'T2.1: the candidate, again',
        write('scripts/rust_core/ratchet.py', fake),
    )
    done = gate(clone, shas, '--base', landed, '--only', 'ratchet')
    assert done.returncode == 1, done.stdout + done.stderr
    assert 'holds a landing commit after the fork point' in done.stderr


def test_a_fork_point_differing_from_the_stand_in_in_judging_data_is_refused(
    tmp_path: Path,
) -> None:
    built = world(
        tmp_path,
        fork_edit=write('docs/design/rust-core/register.yaml', 'entries: []\n'),
    )
    clone, shas = built['clone'], built['shas']
    commit(clone, 'T2.0: the work', work_for(shas))
    done = gate(clone, shas, '--base', f'origin/{PHASE2}', '--only', 'ratchet')
    assert done.returncode == 1, done.stdout + done.stderr
    assert 'differ in more than added files' in done.stderr


def test_a_remote_fork_point_that_is_not_the_pinned_one_is_refused(
    built: dict[str, Any],
) -> None:
    clone, shas = at_fork(built)
    # A candidate whose ratchet names the pinned fork point regardless.
    fake = f"print('{shas['F']} {shas['F']} harness')\n"
    commit(
        clone,
        'T2.0: the work',
        lambda r: (
            write('scripts/rust_core/ratchet.py', fake)(r),
            pinned_gate(shas)(r),
        ),
    )
    git(clone, 'update-ref', f'refs/remotes/origin/{EPIC}', shas['H1'])
    try:
        done = gate(clone, shas, '--base', shas['F'], '--only', 'ratchet')
    finally:
        git(clone, 'update-ref', f'refs/remotes/origin/{EPIC}', shas['F'])
    assert done.returncode == 1, done.stdout + done.stderr
    assert f'from origin/{EPIC} is {shas["H1"]}' in done.stderr


def test_the_pin_is_the_real_bootstrap_and_nothing_overrides_it() -> None:
    r"""Codex's round 2 P2-1: the pin is one line of gate.sh, the real
    package, phase, fork point and stand-in, read from no variable."""
    text = GATE.read_text(encoding='utf-8')
    assert text.count(PIN) == 1
    assert 'SDM_GATE_BOOTSTRAP' not in text
    assert text.count('"$bootstrap"') == 1


def test_another_package_never_gets_the_stand_in(built: dict[str, Any]) -> None:
    r"""Codex's round 2 P2-2: a candidate titled for another package on
    the fork point, its first landing, and its phase push are refused."""
    clone, shas = at_fork(built)
    commit(clone, 'T2.7: another package', work_for(shas))
    done = gate(clone, shas, '--base', f'origin/{PHASE2}', '--only', 'ratchet')
    assert done.returncode == 1, done.stdout + done.stderr
    assert "is not T2.0's commit: T2.7: another package" in done.stderr
    landing(clone, shas['F'], package='T2.7')
    done = gate(clone, shas, '--phase-push', shas['F'], '--only', 'ratchet')
    assert done.returncode == 1, done.stdout + done.stderr
    assert 'stands in' not in done.stderr
    # T2.0's own work under another package's landing.
    git(clone, 'checkout', '-q', '--detach', shas['F'])
    commit(clone, 'T2.0: the work', work_for(shas))
    landing(clone, shas['F'], package='T2.7')
    done = gate(clone, shas, '--phase-push', shas['F'], '--only', 'ratchet')
    assert done.returncode == 1, done.stdout + done.stderr
    assert "the first landing is not T2.0's" in done.stderr


def test_a_first_landing_without_reviewed_by_is_refused(
    built: dict[str, Any],
) -> None:
    r"""The reviewer's round 2 P3-1: the first landing needs Reviewed-by
    as every later landing does."""
    clone, shas = at_fork(built)
    commit(clone, 'T2.0: the work', work_for(shas))
    landing(clone, shas['F'], reviewed=False)
    done = gate(clone, shas, '--phase-push', shas['F'], '--only', 'ratchet')
    assert done.returncode == 1, done.stdout + done.stderr
    assert 'the first landing has no Reviewed-by trailer' in done.stderr


def test_the_real_bootstrap_on_the_real_history(tmp_path: Path) -> None:
    r"""The pinned bootstrap itself: a candidate on 9604074a holding this
    tree is judged by bd44b342's harness, which the gate says."""
    for sha in (REAL_FORK, REAL_STAND):
        held = subprocess.run(
            ['git', '-C', str(ROOT), 'cat-file', '-e', f'{sha}^{{commit}}'],
            capture_output=True,
            check=False,
        )
        assert held.returncode == 0, (
            f'{sha} is not in this checkout: the job must check out every '
            'commit (actions/checkout fetch-depth: 0)'
        )
    clone = tmp_path / 'real'
    git(
        tmp_path,
        'clone',
        '-q',
        '--shared',
        '--no-checkout',
        str(ROOT),
        str(clone),
    )
    for ref in (PHASE2, EPIC):
        git(clone, 'update-ref', f'refs/remotes/origin/{ref}', REAL_FORK)
    tree = git(ROOT, 'rev-parse', 'HEAD^{tree}')
    candidate = git(
        clone, 'commit-tree', tree, '-p', REAL_FORK, '-m', 'T2.0: probe'
    )
    git(clone, 'checkout', '-q', '--detach', candidate)
    done = subprocess.run(
        [str(clone / 'scripts/rust_core/gate.sh'), '--base',
         f'origin/{PHASE2}', '--only', 'ratchet'],
        capture_output=True, text=True, check=False, cwd=clone,
        env=env(SDM_PHASE_BRANCH=PHASE2), stdin=subprocess.DEVNULL,
    )  # fmt: skip
    assert f'last landing {REAL_STAND} stands in' in done.stderr, done.stderr
    assert f'at landing {REAL_STAND[:7]}' in done.stdout, done.stdout


# --- the epic branch's protected paths (rule epic, tree equality) -------------------


def merged_world(
    base: Path,
    epic_edit: Callable[[Path], None] | None = None,
    landing_edit: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    r"""The fork point's world, then: ``rust-core`` takes ``epic_edit`` (as
    U, when given); Phase 2 lands L2 (T2.0's ratchet, and
    ``landing_edit``); ``rust-core`` merges L2 (``--no-ff``, clean), its
    head M2."""
    built = world(base)
    work = base / 'work'
    shas = dict(built['shas'])
    if epic_edit is not None:
        git(work, 'checkout', '-q', EPIC)
        shas['U'] = commit(work, 'Update: an epic change', epic_edit)
    git(work, 'checkout', '-q', PHASE2)

    def package(repo: Path) -> None:
        (repo / 'scripts/rust_core/ratchet.py').write_bytes(
            RATCHET.read_bytes()
        )
        if landing_edit is not None:
            landing_edit(repo)

    shas['L2'] = helpers.land(
        work, 'T2.0', package, protected='scripts/rust_core/**'
    )
    git(work, 'checkout', '-q', EPIC)
    git(work, 'merge', '-q', '--no-ff', '-m', 'Merge T2.0', shas['L2'])
    shas['M2'] = git(work, 'rev-parse', 'HEAD')
    return {'work': work, 'shas': shas}


def epic_check(repo: Path, checkpoint: str) -> subprocess.CompletedProcess[str]:
    r"""The ratchet's check in the epic mode, as ``where`` names it for a
    push to rust-core (``--epic``), from no CI environment at all: a local
    replay runs rule epic too."""
    return subprocess.run(
        [sys.executable, str(RATCHET), 'check', '--repo', '.',
         '--checkpoint', checkpoint, '--epic'],
        capture_output=True, text=True, check=False, cwd=repo, env=env(),
        stdin=subprocess.DEVNULL,
    )  # fmt: skip


def ratchet_run(
    repo: Path, *arguments: str, **environ: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RATCHET), *arguments],
        capture_output=True, text=True, check=False, cwd=repo,
        env=env(**environ), stdin=subprocess.DEVNULL,
    )  # fmt: skip


def epic_failures(done: subprocess.CompletedProcess[str]) -> list[str]:
    r"""The paths rule epic names, in order."""
    marker = 'FAILS epic: '
    tail = ': differs on the epic branch from the merged landing'
    found = []
    for chunk in done.stdout.split(marker)[1:]:
        found.append(chunk.split(tail, 1)[0])
    return found


@pytest.fixture(scope='module')
def merged(owned_module_dir: Callable[[str], Path]) -> dict[str, Any]:
    r"""A clean owner merge of a landing, nothing else on rust-core."""
    return merged_world(owned_module_dir('t20-epic-merged'))


def at_merge(merged: dict[str, Any]) -> tuple[Path, dict[str, str]]:
    work, shas = merged['work'], merged['shas']
    git(work, 'checkout', '-q', '--detach', shas['M2'])
    return work, shas


def test_a_clean_owner_merge_of_a_landing_passes(
    merged: dict[str, Any],
) -> None:
    work, shas = at_merge(merged)
    where = ratchet_run(
        work,
        'where',
        '--repo',
        '.',
        GITHUB_EVENT_NAME='push',
        GITHUB_REF_NAME=EPIC,
    )
    assert where.stdout.split() == ['--base', shas['L2'], '--epic'], (
        where.stderr
    )
    done = epic_check(work, shas['L2'])
    assert done.returncode == 0, done.stdout


def test_a_head_differing_from_the_landing_only_unprotected_passes(
    merged: dict[str, Any],
) -> None:
    work, shas = at_merge(merged)
    commit(
        work, 'Handoff: notes', write('docs/design/rust-core/notes.md', 'n\n')
    )
    done = epic_check(work, shas['L2'])
    assert done.returncode == 0, done.stdout


@pytest.mark.parametrize(
    'name', ['tool.py', 'café.py', 'tab\there.py', 'new\nline.py']
)
def test_an_unapproved_direct_protected_edit_fails(
    merged: dict[str, Any], name: str
) -> None:
    r"""Any name, one git would quote or one holding a newline included."""
    work, shas = at_merge(merged)
    commit(work, 'Handoff: a tool', write(f'scripts/rust_core/{name}', 'x\n'))
    done = epic_check(work, shas['L2'])
    assert done.returncode == 1, done.stdout
    assert epic_failures(done) == [f'scripts/rust_core/{name}'], done.stdout


def test_a_mode_only_change_fails(merged: dict[str, Any]) -> None:
    work, shas = at_merge(merged)
    commit(
        work,
        'Handoff: executable',
        lambda r: (r / 'scripts/rust_core/phases.py').chmod(0o755),
    )
    assert epic_failures(epic_check(work, shas['L2'])) == [
        'scripts/rust_core/phases.py'
    ]


@pytest.mark.parametrize('ignored_by', ['.gitmodules', 'diff.ignoreSubmodules'])
def test_a_protected_submodule_change_fails_whatever_git_is_told_to_ignore(
    merged: dict[str, Any], ignored_by: str
) -> None:
    r"""Codex's round 7 P1-1 (the reviewer's P3-1): a submodule pinned at a
    protected path, git told to ignore submodules (``ignore = all`` in
    .gitmodules, or ``diff.ignoreSubmodules=all``), still differs from L."""
    work, shas = at_merge(merged)
    path = 'scripts/rust_core/sub'
    (work / '.gitmodules').write_text(
        f'[submodule "sub"]\n\tpath = {path}\n\turl = ./sub\n'
        + ('\tignore = all\n' if ignored_by == '.gitmodules' else ''),
        encoding='utf-8',
    )
    git(work, 'add', '.gitmodules')
    git(
        work,
        'update-index',
        '--add',
        '--cacheinfo',
        f'160000,{shas["L1"]},{path}',
    )
    git(work, 'commit', '-q', '-m', 'Handoff: a dependency')
    environ = (
        {
            'GIT_CONFIG_COUNT': '1',
            'GIT_CONFIG_KEY_0': 'diff.ignoreSubmodules',
            'GIT_CONFIG_VALUE_0': 'all',
        }
        if ignored_by == 'diff.ignoreSubmodules'
        else {}
    )
    done = ratchet_run(
        work, 'check', '--repo', '.', '--checkpoint', shas['L2'], '--epic',
        **environ,
    )  # fmt: skip
    assert done.returncode == 1, done.stdout
    assert path in epic_failures(done), done.stdout


def test_check_epic_by_hand_refuses_a_checkpoint_other_than_the_merged_landing(
    merged: dict[str, Any],
) -> None:
    r"""The reviewer's round 7 P3-3: the head judged against itself, or
    against a commit no phase merge brings in, is refused."""
    work, shas = at_merge(merged)
    head = commit(
        work, 'Handoff: a tool', write('scripts/rust_core/t.py', 'x\n')
    )
    for checkpoint in (head, shas['M2']):
        done = epic_check(work, checkpoint)
        assert done.returncode == 1, (checkpoint, done.stdout)
        assert "is not the merged phase's last landing the head holds" in (
            done.stdout
        ), done.stdout


def test_rust_core_fast_forwarded_to_a_landing_is_refused_by_name(
    merged: dict[str, Any],
) -> None:
    r"""The reviewer's round 7 note: rust-core fast-forwarded to a landing
    instead of merged --no-ff says so."""
    work, shas = merged['work'], merged['shas']
    git(work, 'checkout', '-q', '--detach', shas['L2'])
    where = ratchet_run(
        work,
        'where',
        '--repo',
        '.',
        GITHUB_EVENT_NAME='push',
        GITHUB_REF_NAME=EPIC,
    )
    assert where.returncode == 1, where.stdout
    assert 'must be a --no-ff merge' in where.stdout + where.stderr


def test_a_shrunk_protected_list_hides_nothing(merged: dict[str, Any]) -> None:
    r"""The head drops scripts/rust_core/** from its list and adds a file
    there: the list itself, and the file by L's list, fail."""
    work, shas = at_merge(merged)

    def shrink(repo: Path) -> None:
        path = repo / 'contract/protected-paths.txt'
        text = path.read_text(encoding='utf-8')
        assert 'scripts/rust_core/**\n' in text
        path.write_text(
            text.replace('scripts/rust_core/**\n', ''), encoding='utf-8'
        )
        write('scripts/rust_core/hidden.py', 'x\n')(repo)

    commit(work, 'Handoff: a smaller list', shrink)
    assert epic_failures(epic_check(work, shas['L2'])) == [
        'contract/protected-paths.txt',
        'scripts/rust_core/hidden.py',
    ]


def test_an_epic_edit_carried_through_a_landing_s_rename_fails(
    tmp_path: Path,
) -> None:
    r"""The reviewer's round 6 P1-1: U edits a protected file on rust-core,
    the landing renames it, and the clean merge carries U's text into the
    moved file, which then differs from the landing's."""
    lock = 'scripts/rust_core/contract_lock.py'
    moved = 'scripts/rust_core/contract_lock_moved.py'

    def edit(repo: Path) -> None:
        path = repo / lock
        path.write_text(path.read_text(encoding='utf-8') + '# epic edit\n')

    def rename(repo: Path) -> None:
        (repo / lock).rename(repo / moved)

    built = merged_world(tmp_path, epic_edit=edit, landing_edit=rename)
    work, shas = built['work'], built['shas']
    git(work, 'checkout', '-q', '--detach', shas['M2'])
    assert '# epic edit' in (work / moved).read_text(encoding='utf-8')
    assert epic_failures(epic_check(work, shas['L2'])) == [moved]


def test_an_earlier_merge_s_kept_epic_version_fails_after_a_later_landing(
    tmp_path: Path,
) -> None:
    r"""The reviewer's round 6 P2-1: rust-core's U sets a protected setting
    to 20, landing L' sets 11, and the owner's merge of L' keeps 20; a
    later landing L, merged cleanly, is the checkpoint, and 20 still
    differs from L's 11."""
    setting = 'scripts/rust_core/merge_setting.py'
    world(tmp_path)
    work = tmp_path / 'work'
    git(work, 'checkout', '-q', EPIC)
    commit(work, 'Update: the setting', write(setting, 'LIMIT = 20\n'))
    git(work, 'checkout', '-q', PHASE2)

    def first(repo: Path) -> None:
        (repo / 'scripts/rust_core/ratchet.py').write_bytes(
            RATCHET.read_bytes()
        )
        write(setting, 'LIMIT = 11\n')(repo)

    first_landing = helpers.land(work, 'T2.0', first, protected='scripts/**')
    git(work, 'checkout', '-q', EPIC)
    subprocess.run(
        ['git', '-C', str(work), *helpers.IDENTITY, 'merge', '-q', '--no-ff',
         '--no-commit', first_landing],
        capture_output=True, check=False, stdin=subprocess.DEVNULL,
    )  # fmt: skip
    write(setting, 'LIMIT = 20\n')(work)
    git(work, 'add', '-A')
    git(work, 'commit', '-q', '-m', 'Merge T2.0')
    git(work, 'checkout', '-q', PHASE2)
    later = helpers.land(work, 'T2.1', write('docs/later.md', 'l\n'))
    git(work, 'checkout', '-q', EPIC)
    git(work, 'merge', '-q', '--no-ff', '-m', 'Merge T2.1', later)
    where = ratchet_run(
        work,
        'where',
        '--repo',
        '.',
        GITHUB_EVENT_NAME='push',
        GITHUB_REF_NAME=EPIC,
    )
    assert where.stdout.split() == ['--base', later, '--epic'], where.stderr
    assert epic_failures(epic_check(work, later)) == [setting]


def test_the_real_rust_core_judged_against_phase_1_s_landing(
    tmp_path: Path,
) -> None:
    r"""The real history: rust-core's head 9604074a judged against
    bd44b342 fails on the handoff commits' protected changes (the owner's
    merge of T2.0's landing makes that landing, which holds them, the
    checkpoint)."""
    for sha in (REAL_FORK, REAL_STAND):
        held = subprocess.run(
            ['git', '-C', str(ROOT), 'cat-file', '-e', f'{sha}^{{commit}}'],
            capture_output=True,
            check=False,
        )
        assert held.returncode == 0, (
            f'{sha} is not in this checkout: the job must check out every '
            'commit (actions/checkout fetch-depth: 0)'
        )
    clone = tmp_path / 'real'
    git(
        tmp_path,
        'clone',
        '-q',
        '--shared',
        '--no-checkout',
        str(ROOT),
        str(clone),
    )
    git(clone, 'checkout', '-q', '--detach', REAL_FORK)
    done = epic_check(clone, REAL_STAND)
    assert done.returncode == 1, done.stdout
    found = epic_failures(done)
    assert 'scripts/rust_core/orchestration/auto_land.py' in found, found
    assert all(
        p.startswith(('scripts/', 'tests/test_contract_')) for p in found
    ), found


def test_a_landing_shaped_head_never_chooses_its_own_base(
    merged: dict[str, Any],
) -> None:
    r"""The reviewer's round 3 P1-1 (a): a head titled and trailered as a
    landing on Phase 1's last landing never chooses its own base: where
    and --epic refuse it by name, and its change fails."""
    work, shas = at_merge(merged)
    commit(
        work,
        'Land T9.9: old base',
        write('contract/protected-paths.txt', '# nothing\n'),
        trailers={
            'Reviewed-by': 'nobody',
            'Protected-paths': 'contract/ scripts/',
            'Base-SHA': shas['L1'],
            'Checkpoint-SHA': shas['L1'],
        },
    )
    where = ratchet_run(
        work,
        'where',
        '--repo',
        '.',
        GITHUB_EVENT_NAME='push',
        GITHUB_REF_NAME=EPIC,
    )
    # A landing commit on the first-parent history is refused (fix 8:
    # rust-core moves only by --no-ff merges), never taken as the base.
    assert where.returncode == 1, where.stdout
    assert 'must be a --no-ff merge' in where.stdout + where.stderr
    for base in (shas['L1'], shas['L2']):
        done = ratchet_run(
            work, 'checkpoint', '--repo', '.', '--base', base, '--epic'
        )
        assert done.returncode == 1, (base, done.stdout)
    assert epic_check(work, shas['L2']).returncode == 1


def test_a_landing_record_without_reviewed_by_is_no_epic_base(
    merged: dict[str, Any],
) -> None:
    r"""The reviewer's round 3 P1-1 (b): a landing-shaped commit with no
    Reviewed-by, merged as a phase merge's second parent, is refused as
    the epic branch's base, by name, never skipped for an older one."""
    work, shas = merged['work'], merged['shas']
    git(work, 'checkout', '-q', '--detach', shas['L2'])
    fake = commit(
        work,
        'Land T9.9: fake',
        write('contract/protected-paths.txt', '# nothing\n'),
        trailers={'Base-SHA': shas['L2'], 'Checkpoint-SHA': shas['L2']},
    )
    git(work, 'checkout', '-q', '--detach', shas['M2'])
    git(work, 'merge', '-q', '--no-ff', '-m', 'Merge the fake', fake)
    where = ratchet_run(
        work,
        'where',
        '--repo',
        '.',
        GITHUB_EVENT_NAME='push',
        GITHUB_REF_NAME=EPIC,
    )
    assert where.returncode == 1, where.stdout
    assert 'a landing record with no Reviewed-by' in where.stdout + where.stderr


def test_a_local_replay_through_the_gate_runs_rule_epic(
    merged: dict[str, Any],
) -> None:
    r"""gate.sh --epic, from no CI environment, judges rule epic; an older
    push replayed is judged as that push whatever origin/rust-core is."""
    work, shas = at_merge(merged)
    commit(work, 'Handoff: a tool', write('scripts/rust_core/other.py', 'x\n'))
    bad = git(work, 'rev-parse', 'HEAD')
    commit(work, 'Handoff: later', write('later.md', 'l\n'))
    git(work, 'update-ref', f'refs/remotes/origin/{EPIC}', 'HEAD')
    git(work, 'checkout', '-q', '--detach', bad)
    done = subprocess.run(
        [str(work / 'scripts/rust_core/gate.sh'), '--base', shas['L2'],
         '--epic', '--only', 'ratchet'],
        capture_output=True, text=True, check=False, cwd=work, env=env(),
        stdin=subprocess.DEVNULL,
    )  # fmt: skip
    assert done.returncode == 1, done.stdout + done.stderr
    assert (
        'scripts/rust_core/other.py: differs on the epic branch' in done.stdout
    )


def test_a_run_without_epic_on_the_epic_head_says_the_rule_did_not_run(
    merged: dict[str, Any],
) -> None:
    r"""Codex's round 4 P3-2: never silent."""
    work, shas = at_merge(merged)
    git(work, 'update-ref', f'refs/remotes/origin/{EPIC}', shas['M2'])
    done = ratchet_run(work, 'check', '--repo', '.', '--checkpoint', shas['L2'])
    assert "epic: not run: this is the epic branch's head" in done.stdout, (
        done.stdout
    )
    assert '--epic' in done.stdout


# --- a malformed phase setting is a named error (fix 1, P3) --------------------------


@pytest.mark.parametrize(
    ('arguments', 'code'),
    [(['bridge_freeze.py'], 1), (['replay_jobs.py', '--help'], 2)],
)
def test_a_malformed_phase_setting_is_named_not_a_traceback(
    arguments: list[str], code: int
) -> None:
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'scripts/rust_core' / arguments[0]),
            *arguments[1:],
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        env=env(SDM_PHASE_BRANCH='rust-core-phase-two'),
        stdin=subprocess.DEVNULL,
    )
    output = done.stdout + done.stderr
    assert done.returncode == code, output
    assert "'rust-core-phase-two' is not a phase branch" in output
    assert 'Traceback' not in output
