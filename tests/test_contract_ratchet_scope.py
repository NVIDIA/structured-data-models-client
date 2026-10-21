# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ratchet's ``cases`` rule scopes an entry to its own change (T1.29
tests, for the bug fix T1.29's part 1 makes).

The hole: ``ratchet.py check`` passed a changed case file, or a changed
base, under any approved register entry whose ``case_outcomes`` name the
case (or a case built on the base). Once an admission entry is approved
(P1B-26, which names the four baseline cases), every later change to those
case files, or to the ``shop`` and ``adventureworks`` bases, passed for
good. The fix: an entry authorizes a change only in the change that adds or
changes that entry. Each claim on ``tests/test_contract_ratchet.py``'s
temporary repository, through the real ``ratchet.check``:

- a case file or a base changed under an approved entry the checkpoint
  already holds, and the change leaves untouched, fails;
- the same change with the entry added in it passes, and with the entry
  changed in it (the checkpoint holding it) passes;
- the post-landing state: the landing of a change and its entry, judged as
  the phase push judges it, from its base, passes; the next package,
  touching neither, passes against that landing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

# Gate tooling (T3.0): CI runs it in the tooling jobs, not in every matrix job.
pytestmark = pytest.mark.gate_tooling

ROOT = Path(__file__).resolve().parents[1]
if 'ratchet_tests' in sys.modules:
    helpers = sys.modules['ratchet_tests']
else:
    spec = importlib.util.spec_from_file_location(
        'ratchet_tests', ROOT / 'tests/test_contract_ratchet.py'
    )
    assert spec is not None and spec.loader is not None
    helpers = importlib.util.module_from_spec(spec)
    sys.modules['ratchet_tests'] = helpers
    spec.loader.exec_module(helpers)

origin, repo = helpers.origin, helpers.repo
git, commit, land = helpers.git, helpers.commit, helpers.land
check, rules = helpers.check, helpers.rules
append_entry, outcome = helpers.append_entry, helpers.outcome

#: An acceptance case built on the ``shop`` base.
CASE = helpers.ACCEPTED
BASE = 'shop'
REGISTER = 'docs/design/rust-core/register.yaml'
NOTES = 'Seeded by the scope tests: as first approved.'


def _entry() -> dict[str, object]:
    r"""An approved entry naming :data:`CASE` on every baselined runner,
    its outcomes unchanged (as an admission's are)."""
    return {
        'id': 'SEED-26',
        'cases': [CASE],
        'notes': NOTES,
        'case_outcomes': [
            outcome(CASE, runner, old, old)
            for runner, old in (
                ('python-core', 'success at materialize'),
                ('python-old', 'success at materialize'),
                ('rust-host', 'success'),
            )
        ],
    }


def add_entry(r: Path) -> None:
    append_entry(r, _entry())


def change_entry(r: Path) -> None:
    r"""The entry the checkpoint holds, changed: its notes rewritten."""
    path = r / REGISTER
    text = path.read_text()
    assert text.count(NOTES) == 1, 'the entry is not the checkpoint’s'
    path.write_text(
        text.replace(NOTES, 'Seeded by the scope tests: changed again.')
    )


def change_case(r: Path) -> None:
    path = r / f'contract/cases/s0/{CASE}.json'
    case = json.loads(path.read_text())
    assert case['input']['base'] == BASE
    # A valid case still, its input changed: one entity of the two.
    case['input']['materialize']['indices'] = [0]
    path.write_text(json.dumps(case, indent=1))


def change_base(r: Path) -> None:
    path = r / 'contract/bases.yaml'
    bases = yaml.safe_load(path.read_text())
    bases['bases'][BASE] = bases['bases'][BASE][:1]
    path.write_text(yaml.safe_dump(bases))


CHANGES: dict[str, tuple[Callable[[Path], None], str]] = {
    'a case file': (change_case, f'cases: {CASE}:'),
    'a base': (change_base, f'cases: base {BASE}:'),
}


def both(*edits: Callable[[Path], None]) -> Callable[[Path], None]:
    def edit(r: Path) -> None:
        for each in edits:
            each(r)

    return edit


def landed_with_the_entry(repo: Path) -> str:
    r"""A landing adding the approved entry, and nothing else: the
    checkpoint of a later change, which then holds the entry."""
    return land(repo, 'T1.28', add_entry, protected=REGISTER)


@pytest.mark.parametrize('what', sorted(CHANGES))
def test_a_change_under_an_entry_the_checkpoint_holds_fails(
    repo: Path, what: str
) -> None:
    edit, problem = CHANGES[what]
    checkpoint = landed_with_the_entry(repo)
    # The control: the later package touching neither passes the rule.
    commit(repo, 'T1.29: nothing of the case')
    assert 'cases' not in rules(check(repo, checkpoint))
    git(repo, 'reset', '-q', '--hard', checkpoint)
    commit(repo, 'T1.29: the case changed, the entry untouched', edit)
    found = check(repo, checkpoint)
    assert any(p.startswith(problem) for p in found), found


@pytest.mark.parametrize('what', sorted(CHANGES))
def test_the_change_with_its_entry_added_passes(repo: Path, what: str) -> None:
    edit, _ = CHANGES[what]
    checkpoint = git(repo, 'rev-parse', 'HEAD')
    commit(repo, 'T1.29: the case and its entry', both(edit, add_entry))
    assert 'cases' not in rules(check(repo, checkpoint))


@pytest.mark.parametrize('what', sorted(CHANGES))
def test_the_change_with_its_entry_changed_passes(
    repo: Path, what: str
) -> None:
    edit, _ = CHANGES[what]
    checkpoint = landed_with_the_entry(repo)
    commit(repo, 'T1.29: the case and its entry', both(edit, change_entry))
    assert 'cases' not in rules(check(repo, checkpoint))


@pytest.mark.parametrize('what', sorted(CHANGES))
def test_the_post_landing_state_passes(repo: Path, what: str) -> None:
    r"""The landing of the change and its entry judged from its base, as
    CI judges the phase push (every rule, the protected paths named by its
    trailers), then the next package against that landing."""
    edit, _ = CHANGES[what]
    base = git(repo, 'rev-parse', 'HEAD')
    touched = (
        f'contract/cases/s0/{CASE}.json'
        if what == 'a case file'
        else 'contract/bases.yaml'
    )
    landing = land(
        repo, 'T1.29', both(edit, add_entry), protected=f'{touched} {REGISTER}'
    )
    assert check(repo, base) == []
    commit(repo, 'T1.30: a later package')
    assert check(repo, landing) == []


def _named_by_the_register(repo: Path) -> list[str]:
    r"""The cases this tree's approved register entries name in the
    ratchet's cell: each one an approved entry already authorized once."""
    register = yaml.safe_load((repo / REGISTER).read_text())
    return sorted(
        {
            item['case']
            for entry in register['entries']
            if entry.get('status') == 'approved'
            for item in entry.get('case_outcomes') or []
            if item.get('cell') == helpers.CELL
        }
    )


def _case_file(repo: Path, case: str) -> Path:
    [path] = [
        p
        for p in (repo / 'contract/cases').glob('*/*.json')
        if json.loads(p.read_text())['id'] == case
    ]
    return path


def test_no_entry_of_this_tree_s_register_authorizes_a_later_change(
    repo: Path,
) -> None:
    r"""The tree as landed is the checkpoint: a case its approved entries
    name (from T1.29's part 2, ``base.regression`` under P1B-26), its file
    changed with the register untouched, fails."""
    corpus = {
        json.loads(p.read_text())['id']
        for p in (repo / 'contract/cases').glob('*/*.json')
    }
    named = [c for c in _named_by_the_register(repo) if c in corpus]
    if not named:
        # Before T1.29's part 2 no approved entry names a case of the
        # corpus; the seeded tests above hold the rule meanwhile.
        pytest.skip('no approved entry names a case of the corpus yet')
    case = 'base.regression' if 'base.regression' in named else named[0]
    checkpoint = git(repo, 'rev-parse', 'HEAD')

    def change(r: Path) -> None:
        path = _case_file(r, case)
        path.write_text(path.read_text().rstrip('\n') + '\n\n')

    commit(repo, 'T1.30: the case file changed alone', change)
    found = check(repo, checkpoint)
    assert any(p.startswith(f'cases: {case}:') for p in found), (case, found)


# --- a checkpoint without its register ---------------------------------------


def _without_the_register(repo: Path) -> tuple[str, str]:
    r"""A checkpoint holding the ratchet but no register, and a candidate
    restoring the register with an approved entry naming :data:`CASE`
    added, and the case changed; with the checkpoint's SHA and the
    register's text."""
    saved = (repo / REGISTER).read_text()
    checkpoint = commit(
        repo,
        'T1.28: the register missing',
        lambda r: (r / REGISTER).unlink(),
    )
    assert (repo / 'scripts/rust_core/ratchet.py').is_file()

    def restore(r: Path) -> None:
        (r / REGISTER).write_text(saved)
        add_entry(r)
        change_case(r)

    commit(repo, 'T1.29: the register restored and the case changed', restore)
    return checkpoint, saved


def test_a_checkpoint_missing_its_register_authorizes_nothing(
    repo: Path,
) -> None:
    r"""Codex's review of part 1: outside the bootstrap a missing
    checkpoint register is no license, so the case's change under the
    entry the candidate adds is refused (the register rule may refuse the
    missing predecessor too)."""
    checkpoint, _ = _without_the_register(repo)
    found = helpers.ratchet.check(repo, checkpoint, [], None, False).problems
    assert found, 'the change passed with no checkpoint register'
    assert any(p.startswith(f'cases: {CASE}:') for p in found), found


def test_the_real_bootstrap_without_a_register_passes(repo: Path) -> None:
    r"""The control: the same history judged in explicit bootstrap mode,
    where no register precedes the candidate and every approved entry of
    its own counts."""
    checkpoint, _ = _without_the_register(repo)
    found = helpers.ratchet.check(repo, checkpoint, [], None, True).problems
    assert not {'cases', 'register'} & rules(found), found
