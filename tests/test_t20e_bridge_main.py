# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.0e's acceptance tests for exits 3 and 5 (the independent test
author's): the bridge freeze takes ``main`` as a reason, and the reason
holds in the post-landing state.

``bridge_freeze.py`` runs, as ``l0_deps.py`` runs it, on a temporary git
repository holding this checkout's frozen files, with the history a merge
of ``main`` makes (the commits are the SHAs the reasons name):

    M0 (main, the phase's fork) -- P0 (origin/rust-core-phase-2, the base)
      \                              \
       M1 (a rule) -- M2 ------------ merge (T2.0c) -- declare (T2.0c)
                       \
                        M3 (refs/remotes/origin/main, after the merge)

M1 adds a function to a frozen bridge file (a new entry and a changed
skeleton); the candidate merges main up to M2 and declares both entries
``{"main": "<M1's SHA> <M1's title>"}``. The reason is valid only when its
commit is reachable from ``origin/main`` (the remote-tracking ref, never a
local branch) and was brought in by this merge: an ancestor of HEAD and not
of the merge base with the base branch.

Exit 3: that declaration passes; seeded faults: a SHA not in the merge (on
main after it, M3; no commit at all), a SHA already in the checkpoint (M0),
a malformed SHA (abbreviated, not hex, upper case, none), a commit of the
candidate that is not main's, and a commit only a local ``main`` branch
holds. The reason is documented next to the others.
Exit 5: after the landing (the base branch at the candidate), the
declarations are held as they stand, and a later change to an entry the
spent ``main`` declaration covers must state its own reason.

Gating: the tests that need the feature are marked :data:`PENDING`, a
strict xfail; the implementer deletes the marker (its one definition below)
in the commit that makes them pass. Unmarked tests are controls that pass
on the base.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/rust_core/bridge_freeze.py'

#: The marker the implementer deletes (see the module docstring).
PENDING = pytest.mark.xfail(
    strict=True,
    reason="T2.0e: the bridge freeze's main reason is not implemented yet",
)

# --- the interface assumed (the implementer matches it, or asks) ---------------
#
#: The reason's key, and its text: the full SHA of the main commit, then
#: its title (the brief's example).
REASON = 'main'
#: Where main is read: the remote-tracking ref, as last fetched.
MAIN_REF = 'refs/remotes/origin/main'
PHASE = 'rust-core-phase-2'
BASE = f'origin/{PHASE}'
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


freeze = _module(SCRIPT, 't20e_bridge_freeze')
#: The frozen file main's commit changes, covered whole.
TARGET = str(freeze.WHOLE[-1])
ADDED = '_a_rule_main_added'
KEYS = (f'{TARGET}::<module>', f'{TARGET}::{ADDED}')
#: The function main's commit appends, whole, so a later edit names it.
ADDED_SOURCE = f'\n\ndef {ADDED}(value):\n    return value\n'
IDENTITY = [
    '-c',
    'user.name=Freeze Test',
    '-c',
    'user.email=freeze@example.com',
]


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ['git', '-C', str(repo), *IDENTITY, *args],
        capture_output=True,
        text=True,
        check=True,
        stdin=subprocess.DEVNULL,
    )
    return done.stdout.strip()


def commit(repo: Path, subject: str) -> str:
    git(repo, 'add', '-A')
    git(repo, 'commit', '-q', '--allow-empty', '-m', subject)
    return git(repo, 'rev-parse', 'HEAD')


def declare(repo: Path, changes: dict[str, dict[str, str]]) -> None:
    path = repo / freeze.CHANGES
    data = json.loads(path.read_text())
    data['changes'] = changes
    path.write_text(json.dumps(data, indent=2) + '\n')


def run(repo: Path) -> tuple[int, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(('GITHUB_', 'SDM_', 'GIT_'))
    }
    done = subprocess.run(
        [sys.executable, str(SCRIPT), '--root', str(repo), '--base', BASE],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo,
        env=env,
        stdin=subprocess.DEVNULL,
    )
    return done.returncode, done.stdout + done.stderr


@dataclass
class History:
    repo: Path
    m0: str
    m1: str
    m3: str
    p0: str
    prep: str
    merge: str
    local_main: str

    def title(self, sha: str) -> str:
        return git(self.repo, 'log', '-1', '--format=%s', sha)

    def reason(
        self, text: str, kind: str = REASON
    ) -> dict[str, dict[str, str]]:
        return {key: {kind: text} for key in KEYS}

    def main_reason(self, sha: str) -> dict[str, dict[str, str]]:
        return self.reason(f'{sha} {self.title(sha)}')


def history(tmp: Path) -> History:
    r"""The history in the module docstring, HEAD at the merge, nothing
    declared yet."""
    repo = tmp / 'repo'
    repo.mkdir()
    git(repo, 'init', '-q', '-b', 'trunk')
    for rel in freeze.FROZEN:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, repo / rel)
    (repo / freeze.CHANGES).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / freeze.CHANGES, repo / freeze.CHANGES)
    declare(repo, {})
    freeze.write(repo)
    m0 = commit(repo, 'Start the client (#1)')
    # Main: a rule added to a frozen bridge file, then more of main.
    target = repo / TARGET
    target.write_text(target.read_text() + ADDED_SOURCE)
    m1 = commit(repo, 'Add a rule to the bridge (#2)')
    m2 = commit(repo, 'Document the rule (#3)')
    m3 = commit(repo, 'A later change on main (#4)')
    git(repo, 'update-ref', MAIN_REF, m3)
    # The phase branch, forked at M0.
    git(repo, 'checkout', '-q', '-b', 'lane', m0)
    p0 = commit(repo, 'Land T2.0: the phase base')
    git(repo, 'update-ref', f'refs/remotes/{BASE}', p0)
    # A local main that is not the remote's: one commit main never had.
    git(repo, 'branch', 'main', m2)
    git(repo, 'checkout', '-q', 'main')
    local_main = commit(repo, 'Only on the local main (#5)')
    git(repo, 'checkout', '-q', 'lane')
    prep = commit(repo, 'T2.0c: before the merge')
    git(repo, 'merge', '-q', '--no-ff', '-m', 'T2.0c: merge main', m2)
    merge = git(repo, 'rev-parse', 'HEAD')
    freeze.write(repo)
    return History(repo, m0, m1, m3, p0, prep, merge, local_main)


@pytest.fixture
def h(tmp_path: Path) -> History:
    return history(tmp_path)


def declared(h: History, changes: dict[str, dict[str, str]]) -> None:
    declare(h.repo, changes)
    commit(h.repo, 'T2.0c: the freeze declares what main changed')


def refused(code: int, text: str) -> bool:
    return code == 1 and 'freeze: FAIL' in text and all(k in text for k in KEYS)


# --- the history itself ------------------------------------------------------------


def test_the_merge_changes_the_freeze_and_an_undeclared_change_fails(
    h: History,
) -> None:
    r"""Not vacuous: the merge changes both entries, and declaring nothing
    fails on each (passes on the base)."""
    declared(h, {})
    code, text = run(h.repo)
    assert refused(code, text), text
    assert git(h.repo, 'merge-base', '--is-ancestor', h.m1, 'HEAD') == ''


def test_a_refactor_reason_still_passes(h: History) -> None:
    r"""Control: the reasons there were still work on the same history."""
    declared(h, h.reason('behaviour kept', kind='refactor'))
    code, text = run(h.repo)
    assert code == 0, text


# --- exit 3: the main reason --------------------------------------------------------


def test_main_is_a_documented_reason() -> None:
    r"""Exit 3: ``main`` is a reason, documented next to the others (the
    module docstring's list, and the declarations' ``about``)."""
    assert REASON in freeze.REASONS
    assert '{"main": ' in (freeze.__doc__ or '')
    assert REASON in freeze.CHANGES_ABOUT


def test_a_main_commit_the_merge_brought_in_is_a_reason(h: History) -> None:
    r"""Exit 3: M1, on ``origin/main`` and brought in by the candidate's
    merge, is a valid reason for the entries it changed."""
    declared(h, h.main_reason(h.m1))
    code, text = run(h.repo)
    assert code == 0, text
    assert 'freeze: ok' in text


@pytest.mark.parametrize(
    'seed',
    [
        'on main after the merge',
        'no such commit',
        'already in the checkpoint',
        'abbreviated',
        'not hex',
        'upper case',
        'no SHA',
        "the candidate's own commit",
        'the merge commit',
        'only on a local main',
    ],
)
def test_a_main_reason_that_is_not_the_merge_s_is_refused(
    h: History, seed: str
) -> None:
    r"""Exit 3's seeded faults, each refused naming the entries; the twin
    (M1's reason) passes on the same history first."""
    declared(h, h.main_reason(h.m1))
    code, text = run(h.repo)
    assert code == 0, text
    git(h.repo, 'reset', '-q', '--hard', h.merge)
    text_of = {
        'on main after the merge': f'{h.m3} {h.title(h.m3)}',
        'no such commit': f'{"ab" * 20} Add a rule to the bridge (#2)',
        'already in the checkpoint': f'{h.m0} {h.title(h.m0)}',
        'abbreviated': f'{h.m1[:12]} {h.title(h.m1)}',
        'not hex': f'{"zz" * 20} {h.title(h.m1)}',
        'upper case': f'{h.m1.upper()} {h.title(h.m1)}',
        'no SHA': h.title(h.m1),
        "the candidate's own commit": f'{h.prep} {h.title(h.prep)}',
        'the merge commit': f'{h.merge} {h.title(h.merge)}',
        'only on a local main': f'{h.local_main} {h.title(h.local_main)}',
    }
    declared(h, h.reason(text_of[seed]))
    code, text = run(h.repo)
    assert refused(code, text), text


# --- exit 5: after the landing -------------------------------------------------------


@PENDING
def test_after_the_landing_the_declarations_are_held_and_spent(
    h: History,
) -> None:
    r"""Exit 5: the base branch at the landed candidate, the check passes
    there (the merge base holds the declarations as they stand); a later
    change to the entry, declared only by the spent ``main`` declaration,
    is refused, and passes with its own reason."""
    declared(h, h.main_reason(h.m1))
    landed = commit(h.repo, 'Land T2.0c: main merged')
    git(h.repo, 'update-ref', f'refs/remotes/{BASE}', landed)
    code, text = run(h.repo)
    assert code == 0, text
    target = h.repo / TARGET
    source = target.read_text()
    # The edit names the function it targets: the file's own functions
    # (``_copied`` returns ``value`` too) stay as they are.
    assert source.count(ADDED_SOURCE) == 1
    target.write_text(
        source.replace(
            ADDED_SOURCE, ADDED_SOURCE.replace('return value', 'return 1')
        )
    )
    assert freeze.write(h.repo) == [f'{TARGET}::{ADDED}']
    commit(h.repo, 'T2.1: the rule changed')
    code, text = run(h.repo)
    assert code == 1 and f'{TARGET}::{ADDED}' in text, text
    assert 'spent' in text, text
    declare(h.repo, {f'{TARGET}::{ADDED}': {'refactor': 'a later edit'}})
    commit(h.repo, 'T2.1: declared')
    code, text = run(h.repo)
    assert code == 0, text
