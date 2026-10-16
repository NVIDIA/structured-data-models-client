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
``rust-core`` merging them, a handbook commit and the fork point. The pin is the
fixture's (``SDM_GATE_BOOTSTRAP``, which ci.yml never sets); one test runs
the real pin on the real history.

- The candidate on the fork point, the first landing itself (a replay of
  its tip and its phase push) are judged by the stand-in's harness, which
  the gate says.
- Seeded, each refused: a candidate above a landing with the fork point as
  its base (the reviewer's repro); a candidate whose ``ratchet.py`` names
  another non-landing checkpoint (Codex's repro: it gets that commit's own
  harness, never the stand-in); a fork point that differs from the
  stand-in in a judging file; a remote epic branch whose fork point is not
  the pinned one; a first landing recording another Checkpoint-SHA.

The epic branch (rule ``epic``): on ``rust-core``'s head a commit after the
merged phase's last landing that changes a protected path fails unless an
owner-approved row of the head's decisions names it, by SHA prefix or by a
range; a row that is not the owner's names nothing.
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


def work(repo: Path) -> None:
    r"""The candidate's change: this tree's ratchet.py, as T2.0's."""
    (repo / 't20.txt').write_text('x')
    (repo / 'scripts/rust_core/ratchet.py').write_bytes(RATCHET.read_bytes())


def landing(repo: Path, base: str, checkpoint: str | None = None) -> str:
    return commit(
        repo,
        'Land T2.0: the package',
        trailers={
            'Reviewed-by': 'independent reviewer and Codex, T2.0',
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
        env=env(
            SDM_PHASE_BRANCH=PHASE2,
            SDM_GATE_BOOTSTRAP=f'{PHASE2} {shas["F"]} {shas["L1"]}',
        ),
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
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return world(tmp_path_factory.mktemp('t20-bootstrap'))


def at_fork(built: dict[str, Any]) -> tuple[Path, dict[str, str]]:
    clone, shas = built['clone'], built['shas']
    git(clone, 'checkout', '-q', '--detach', shas['F'])
    return clone, shas


def test_the_fork_point_s_candidate_is_judged_by_the_stand_in(
    built: dict[str, Any],
) -> None:
    clone, shas = at_fork(built)
    commit(clone, 'T2.0: the work', work)
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
    commit(clone, 'T2.0: the work', work)
    landing(clone, shas['F'])
    for where in (['--base', f'origin/{PHASE2}'], ['--phase-push', shas['F']]):
        done = gate(clone, shas, *where, '--only', 'ratchet')
        assert stood_in(done, shas), (where, done.stdout + done.stderr)


def test_a_first_landing_recording_another_checkpoint_is_refused(
    built: dict[str, Any],
) -> None:
    clone, shas = at_fork(built)
    commit(clone, 'T2.0: the work', work)
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
    commit(clone, 'T2.0: the work', work)
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
    commit(clone, 'T2.0: the work', work)
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
    commit(clone, 'T2.0: the work', work)
    done = gate(clone, shas, '--base', f'origin/{PHASE2}', '--only', 'ratchet')
    assert done.returncode == 1, done.stdout + done.stderr
    assert 'differ in more than added files' in done.stderr


def test_a_remote_fork_point_that_is_not_the_pinned_one_is_refused(
    built: dict[str, Any],
) -> None:
    clone, shas = at_fork(built)
    # A candidate whose ratchet names the pinned fork point regardless.
    fake = f"print('{shas['F']} {shas['F']} harness')\n"
    commit(clone, 'T2.0: the work', write('scripts/rust_core/ratchet.py', fake))
    git(clone, 'update-ref', f'refs/remotes/origin/{EPIC}', shas['H1'])
    try:
        done = gate(clone, shas, '--base', shas['F'], '--only', 'ratchet')
    finally:
        git(clone, 'update-ref', f'refs/remotes/origin/{EPIC}', shas['F'])
    assert done.returncode == 1, done.stdout + done.stderr
    assert f'from origin/{EPIC} is {shas["H1"]}' in done.stderr


def test_ci_never_sets_the_bootstrap_override() -> None:
    assert 'SDM_GATE_BOOTSTRAP' not in (
        ROOT / '.github/workflows/ci.yml'
    ).read_text(encoding='utf-8')


def test_the_real_bootstrap_on_the_real_history(tmp_path: Path) -> None:
    r"""The pinned bootstrap itself: a candidate on 9604074a holding this
    tree is judged by bd44b342's harness, which the gate says."""
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


# --- the epic branch's protected changes ---------------------------------------------


def epic_check(
    clone: Path, shas: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RATCHET), 'check', '--repo', '.',
         '--checkpoint', shas['L1']],
        capture_output=True, text=True, check=False, cwd=clone,
        env=env(), stdin=subprocess.DEVNULL,
    )  # fmt: skip


def row(text: str, status: str = 'approved by owner 2026-10-16') -> str:
    return (
        '# Phase 2 decisions\n\n| # | Decision | Reason | Status |\n'
        f'|---|---|---|---|\n| X2-90 | {text} | the handoff | {status} |\n'
    )


@pytest.fixture(scope='module')
def epic(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    r"""The history above with a handoff commit H1 changing a protected
    path, ``rust-core`` at the fork point."""
    built = world(
        tmp_path_factory.mktemp('t20-epic'),
        handbook=write('scripts/rust_core/handoff_tool.py', 'print(1)\n'),
    )
    return built


def epic_head(
    epic: dict[str, Any], decisions: Callable[[str], str] | None
) -> tuple[Path, dict[str, str]]:
    r"""``rust-core``'s head, the remote tip: on the fork point, a commit
    giving it T2.0's ratchet (a protected change), then one writing Phase
    2's decisions, ``decisions(<that commit>)`` (no protected change)."""
    clone, shas = epic['clone'], epic['shas']
    git(clone, 'checkout', '-q', '--detach', shas['F'])
    shas = dict(shas)
    shas['T'] = commit(
        clone,
        'Handoff: the ratchet',
        lambda r: (r / 'scripts/rust_core/ratchet.py').write_bytes(
            RATCHET.read_bytes()
        ),
    )
    if decisions is not None:
        commit(
            clone,
            "Handoff: the owner's rows",
            write(
                'docs/design/rust-core/phase-2/decisions.md',
                decisions(shas['T']),
            ),
        )
    head = git(clone, 'rev-parse', 'HEAD')
    git(clone, 'update-ref', f'refs/remotes/origin/{EPIC}', head)
    return clone, shas


def failures(done: subprocess.CompletedProcess[str]) -> list[str]:
    return [x for x in done.stdout.splitlines() if x.startswith('FAILS')]


def test_an_unnamed_protected_change_on_rust_core_fails(
    epic: dict[str, Any],
) -> None:
    clone, shas = epic_head(epic, None)
    done = epic_check(clone, shas)
    assert done.returncode == 1, done.stdout
    assert (
        f'scripts/rust_core/handoff_tool.py: changed on the epic branch by '
        f'{shas["H1"][:12]}' in done.stdout
    ), done.stdout


def test_a_row_naming_one_commit_names_it_and_no_other(
    epic: dict[str, Any],
) -> None:
    h1 = epic['shas']['H1']
    clone, shas = epic_head(epic, lambda _: row(f'The handoff `{h1[:10]}`'))
    found = failures(epic_check(clone, shas))
    assert not any(h1[:12] in x for x in found), found
    assert any(shas['T'][:12] in x for x in found), found


def test_a_row_that_is_not_the_owner_s_names_nothing(
    epic: dict[str, Any],
) -> None:
    h1 = epic['shas']['H1']
    clone, shas = epic_head(
        epic,
        lambda _: row(
            f'The handoff `{h1[:10]}`', status='orchestrator, owner to confirm'
        ),
    )
    found = failures(epic_check(clone, shas))
    assert any(h1[:12] in x for x in found), found


def test_an_epic_head_whose_protected_changes_a_range_names_passes(
    epic: dict[str, Any],
) -> None:
    m = epic['shas']['M']
    clone, shas = epic_head(
        epic, lambda t: row(f'The handoff commits `{m[:10]}..{t[:10]}`')
    )
    done = epic_check(clone, shas)
    assert done.returncode == 0, done.stdout
    assert 'named by an owner-approved row' in done.stdout


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
