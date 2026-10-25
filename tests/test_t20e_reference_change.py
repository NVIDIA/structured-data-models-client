# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.0e's acceptance tests for exits 1, 2 and 5 (the independent test
author's): the ratchet accepts a new reference recording only on the
owner's recorded approval, an agent can't forge that approval, and the
rule passes in its post-landing state.

Each test runs ``ratchet.py``'s checks on a real git history, built as
``tests/test_contract_ratchet.py`` builds it (its helpers, reused): the
harness, cases, reference and register copied from this checkout and
committed as landings, plus every other file of this checkout that cites
the reference SHA in full outside ``docs/`` (the vector files the
reference SHA consistency tests check, wherever they are), so the
temporary repository holds every citation the real one does. A test lands
an owner row in ``phase-2/decisions.md`` at the checkpoint (the
orchestrator's decision: the ratchet reads the approval only there, never
from the candidate), then commits a candidate that re-records the
reference under a fake new SHA: the old file replaced by one named for the
new SHA, its lines' identity moved to the new SHA, and every citation of
the old SHA outside ``docs/`` (``reference-env.txt`` and the vector files)
moved with it. ``docs/`` keeps the old SHA: it is history (Phase 1's
evidence), which no consistency test checks.

Every seeded fault is shown on a candidate whose unseeded twin passes, so
a refusal is the seed's (the ratchet tests' rule).

Exit 1: an owner row naming the full new SHA and the word "reference"
admits the replacement; without one, today's refusal stays by name; the
file's name, ``reference-env.txt`` and every vector file must agree, each
disagreement named; the old file must be gone (one reference file); the new
recording carries every case the old one did, a dropped case named; a case
whose recording differs is named in the report for the reviewers.
Review round 1 added: a duplicate line naming another reference commit,
named by file and line; a vector file whose reference field names another
SHA (the approved one only in a note), or which is deleted, named; the new
recording under any path but ``contract/reference/<SHA>.jsonl``, refused.
Review round 2 added: a vector file truncated, blank or malformed, and a
JSON-lines file whose later line still cites another SHA, each named; a
citing file with dots and brackets in its keys, moved whole, passes.
Exit 2 (seeded faults): a row without the SHA, a row naming another SHA, a
row that is not an owner approval, an abbreviated SHA, a row without the
word "reference", a row outside a ``phase-*/decisions.md``, a row the
candidate writes itself, two reference files left behind.
Exit 5: after the change lands, the next package passes, the per-case rules
hold the new recording, and the row does not admit a further replacement.

Gating: the tests that need the feature are marked :data:`PENDING`, a
strict xfail, so CI stays green until the implementation lands and turns
red if one passes before it (a vacuous test). The implementer deletes the
``PENDING`` marker (the one definition below) in the commit that makes them
pass. The unmarked tests are controls that pass on the base and must keep
passing.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
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

# --- the interface assumed (the implementer matches it, or asks) ---------------
#
#: Where the owner's approval is: a row of a phase's decisions file, read
#: as ``ratchet.py`` reads every phase's (T2.0), its status the owner's
#: approval as ``ratchet.owner_approved`` reads it, its text naming the
#: full new reference SHA (40 hex) and the word "reference".
DECISIONS = 'docs/design/rust-core/phase-2/decisions.md'
HEADER = (
    '# Phase 2 decisions (seeded by the T2.0e tests)\n\n'
    '| # | Decision | Reason | Status |\n|---|---|---|---|\n'
)
APPROVED = 'approved by owner 2026-10-16'
#: The refusal that stays, by name, when no row admits the change (today's
#: ``check_reference`` text).
REFUSAL = 'new reference commit'
#: A problem is ``<rule>: <text>``; the reference rule's name.
RULE = 'reference'
# ---------------------------------------------------------------------------------

NEW = '5a' * 20
OTHER = '6c' * 20
THIRD = '7d' * 20
REFERENCE = 'contract/reference'
ENV = 'contract/reference-env.txt'
#: A vector file a consistency test checks (test_t132b_decimal_vectors.py).
VECTORS = 'contract/decimal-vectors.json'
#: A case of the recording, for the per-case seeds.
CASE = 'base.regression'


def _module(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


#: The ratchet tests' helpers, under the name the other test files share.
helpers = _module(ROOT / 'tests/test_contract_ratchet.py', 'ratchet_tests')
ratchet = helpers.ratchet
git, commit, land = helpers.git, helpers.commit, helpers.land


def old_sha(repo: Path) -> str:
    (path,) = (repo / REFERENCE).glob('*.jsonl')
    return path.stem


def citing(root: Path, sha: str) -> list[str]:
    r"""The tracked files of ``root`` that cite ``sha`` in full, but
    ``docs/`` (history) and the reference recordings themselves."""
    done = subprocess.run(
        ['git', '-C', str(root), 'grep', '-l', '-F', sha],
        capture_output=True,
        text=True,
        check=False,
    )
    return sorted(
        name
        for name in done.stdout.splitlines()
        if not name.startswith(('docs/', f'{REFERENCE}/'))
    )


def _cited_at(text: str) -> str:
    r"""The reference a JSON-lines recording names, by its first line."""
    first = json.loads(text.splitlines()[0])
    sha: str = first['identity']['reference']['commit']['sha']
    return sha


def build(tmp: Path) -> tuple[Path, str]:
    r"""The ratchet tests' repository, plus a landing that adds every file
    of this checkout citing the reference SHA that it lacks, so it holds
    every citation the real repository does; and the ratchet's frozen
    recordings (``ratchet.FROZEN``), as they were at the move they were
    frozen across: naming the reference the repository records. They
    keep naming the reference they were recorded at (b413e9ad) since T2.0c
    moved it to 6298913c, so this checkout's copies no longer cite it; the
    landing names it in their place, every other byte theirs."""
    repo, _ = helpers.build(tmp)
    old = old_sha(repo)
    wanted = [name for name in citing(ROOT, old) if not (repo / name).exists()]
    frozen = [
        name
        for name in getattr(ratchet, 'FROZEN', {})
        if name not in wanted and not (repo / name).exists()
    ]

    def add(r: Path) -> None:
        for name in wanted:
            (r / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, r / name)
        for name in frozen:
            text = (ROOT / name).read_text(encoding='utf-8')
            recorded = _cited_at(text)
            assert recorded != old and recorded in text, name
            (r / name).parent.mkdir(parents=True, exist_ok=True)
            (r / name).write_text(text.replace(recorded, old), encoding='utf-8')

    return repo, land(repo, 'T1.16', add)


@pytest.fixture(scope='module')
def origin(owned_module_dir: Callable[[str], Path]) -> tuple[Path, str]:
    return build(owned_module_dir('origin'))


@pytest.fixture
def repo(origin: tuple[Path, str], tmp_path: Path) -> Path:
    source, landed = origin
    tree = tmp_path / 'tree'
    git(source, 'worktree', 'add', '-q', '--detach', str(tree), landed)
    return tree


def row(text: str, status: str = APPROVED, ident: str = 'P2E-901') -> str:
    return f'| {ident} | {text} | The owner moved the reference | {status} |\n'


def approval(sha: str) -> str:
    r"""The row's text in the form assumed: the word and the full SHA."""
    return (
        f'The reference moves to `{sha}`: its recording replaces the '
        'recorded one'
    )


def decisions(*rows: str, path: str = DECISIONS) -> Callable[[Path], None]:
    def edit(repo: Path) -> None:
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        before = target.read_text() if target.is_file() else HEADER
        target.write_text(before + ''.join(rows))

    return edit


def rerecord(
    new: str,
    *,
    name: str | None = None,
    keep_old: bool = False,
    skip: tuple[str, ...] = (),
    env: str | None = None,
    change: Callable[[list[dict[str, Any]]], None] | None = None,
) -> Callable[[Path], None]:
    r"""The candidate's edit: the reference re-recorded at ``new``, its
    file named ``name`` (``new``), the old file removed unless
    ``keep_old``, every citation of the old SHA outside ``docs/`` moved to
    ``new`` but those in ``skip``, ``reference-env.txt`` naming ``env``
    when given, and ``change`` applied to the new recording's lines."""

    def edit(repo: Path) -> None:
        old = old_sha(repo)
        source = repo / REFERENCE / f'{old}.jsonl'
        lines = [
            json.loads(line.replace(old, new))
            for line in source.read_text().splitlines()
        ]
        if change is not None:
            change(lines)
        if not keep_old:
            source.unlink()
        (repo / REFERENCE / f'{name or new}.jsonl').write_text(
            ''.join(json.dumps(line) + '\n' for line in lines)
        )
        for path in citing(repo, old):
            if path in skip:
                continue
            target = repo / path
            target.write_text(target.read_text().replace(old, new))
        if env is not None:
            target = repo / ENV
            target.write_text(
                re.sub(
                    r'(?m)^reference: [0-9a-f]{40}$',
                    f'reference: {env}',
                    target.read_text(),
                )
            )

    return edit


def run(repo: Path, checkpoint: str) -> Any:
    r"""The ratchet's report on HEAD, no observations, no trailer (the
    protected paths are then notes, for the landing to name)."""
    return ratchet.check(repo, checkpoint, [], None)


def rules(problems: list[str]) -> set[str]:
    return {problem.split(':', 1)[0] for problem in problems}


def approved_checkpoint(repo: Path, *rows: str) -> str:
    r"""The checkpoint: a landing adding ``rows`` (the owner's approval of
    ``NEW`` when none are given) to ``phase-2/decisions.md``."""
    return land(repo, 'T2.0d', decisions(*(rows or (row(approval(NEW)),))))


def twin_passes(repo: Path, checkpoint: str) -> None:
    r"""The unseeded twin: ``NEW`` re-recorded as the row approves passes,
    then the worktree is back at the checkpoint for the seed."""
    commit(repo, 'T2.0c: the reference re-recorded', rerecord(NEW))
    found = run(repo, checkpoint).problems
    assert found == [], found
    git(repo, 'reset', '-q', '--hard', checkpoint)


def refused_by_name(found: list[str]) -> bool:
    return any(p.startswith(f'{RULE}:') and REFUSAL in p for p in found)


# --- exit 1: the owner's row admits a new reference --------------------------------


def test_without_a_row_the_refusal_stays_by_name(repo: Path) -> None:
    r"""Control (exit 1, "without such a row, the current refusal stays,
    by name"): a whole, consistent re-recording with no row is refused as
    today. Passes on the base."""
    checkpoint = git(repo, 'rev-parse', 'HEAD')
    commit(repo, 'T2.0c: the reference re-recorded', rerecord(NEW))
    assert refused_by_name(run(repo, checkpoint).problems)


def test_an_owner_row_naming_the_sha_admits_the_new_reference(
    repo: Path,
) -> None:
    r"""Exit 1: the owner's row naming the full new SHA and "reference",
    the new file named for it, ``reference-env.txt`` and every vector file
    citing it, the old file gone: no problem."""
    checkpoint = approved_checkpoint(repo)
    commit(repo, 'T2.0c: the reference re-recorded', rerecord(NEW))
    report = run(repo, checkpoint)
    assert report.problems == [], report.problems
    assert [p.name for p in (repo / REFERENCE).iterdir()] == [f'{NEW}.jsonl']


def test_the_old_sha_is_not_a_new_reference(repo: Path) -> None:
    r"""Exit 1, "the new SHA is not the old one": a row approving the old
    SHA waives nothing; a changed line of the recording it names is still
    judged case by case. Control: passes on the base."""
    old = old_sha(repo)
    checkpoint = approved_checkpoint(repo, row(approval(old)))
    twin = rerecord(old, change=lambda lines: None)
    commit(repo, 'T2.0c: identity only', twin)
    assert run(repo, checkpoint).problems == []
    git(repo, 'reset', '-q', '--hard', checkpoint)

    def moved(lines: list[dict[str, Any]]) -> None:
        line = next(x for x in lines if x['case'] == CASE)
        line['outcome']['stage'] = 'materialize'

    commit(repo, 'T2.0c: a line changed', rerecord(old, change=moved))
    found = run(repo, checkpoint).problems
    assert any(
        p.startswith(f'{RULE}: {CASE}: its line changed') for p in found
    ), found


@pytest.mark.parametrize(
    ('seed', 'named'),
    [
        ('reference-env.txt not moved', ENV),
        ('reference-env.txt names another SHA', ENV),
        ('a vector file not moved', VECTORS),
        ('the file named for another SHA', OTHER),
    ],
)
def test_every_citation_agrees_or_is_named(
    repo: Path, seed: str, named: str
) -> None:
    r"""Exit 1: the file's name, ``reference-env.txt`` and every vector
    file name the approved SHA; a disagreement is refused, naming the file
    (or the SHA the file is named for)."""
    checkpoint = approved_checkpoint(repo)
    twin_passes(repo, checkpoint)
    edit = {
        'reference-env.txt not moved': rerecord(NEW, skip=(ENV,)),
        'reference-env.txt names another SHA': rerecord(NEW, env=OTHER),
        'a vector file not moved': rerecord(NEW, skip=(VECTORS,)),
        'the file named for another SHA': rerecord(NEW, name=OTHER),
    }[seed]
    commit(repo, f'T2.0c: {seed}', edit)
    found = run(repo, checkpoint).problems
    assert any(named in p for p in found), found


def test_a_case_the_new_recording_drops_is_named(repo: Path) -> None:
    r"""Exit 1, "nothing silently disappears": the new recording carries
    every case the old one did; a dropped case is refused by its ID."""
    checkpoint = approved_checkpoint(repo)
    twin_passes(repo, checkpoint)

    def drop(lines: list[dict[str, Any]]) -> None:
        lines.remove(next(x for x in lines if x['case'] == CASE))

    commit(repo, 'T2.0c: a case dropped', rerecord(NEW, change=drop))
    found = run(repo, checkpoint).problems
    assert any(CASE in p for p in found), found


def test_a_case_whose_recording_differs_is_listed_for_the_reviewers(
    repo: Path,
) -> None:
    r"""Exit 1: across a reference change the per-case comparison does not
    apply, but every case whose recording differs from the old one (other
    than its identity) is named in the ratchet's report, so the reviewers
    see it; a case recorded alike is not."""
    checkpoint = approved_checkpoint(repo)

    def moved(lines: list[dict[str, Any]]) -> None:
        line = next(x for x in lines if x['case'] == CASE)
        line['outcome']['stage'] = 'materialize'

    commit(repo, 'T2.0c: one answer moved', rerecord(NEW, change=moved))
    report = run(repo, checkpoint)
    said = report.problems + report.notes
    assert any(CASE in p for p in said), said
    # Not vacuous: the identity-only cases are not listed as differing.
    unchanged = next(
        line['case']
        for line in map(
            json.loads,
            (repo / REFERENCE / f'{NEW}.jsonl').read_text().splitlines(),
        )
        if line['case'] != CASE
    )
    assert not any(
        re.search(rf'\b{re.escape(unchanged)}\b', p) for p in said
    ), said


def then(*edits: Callable[[Path], None]) -> Callable[[Path], None]:
    def edit(repo: Path) -> None:
        for each in edits:
            each(repo)

    return edit


def test_a_duplicate_line_naming_another_reference_is_refused(
    repo: Path,
) -> None:
    r"""Exit 1 (review round 1): every line of the new recording names the
    approved reference. A copy of a case's line whose identity names
    another reference commit, put first with the valid original later in
    the file, is refused, naming the file and the line (line 1)."""
    checkpoint = approved_checkpoint(repo)
    twin_passes(repo, checkpoint)

    def duplicate(r: Path) -> None:
        path = r / REFERENCE / f'{NEW}.jsonl'
        lines = path.read_text().splitlines(keepends=True)
        copied = json.loads(lines[0])
        copied['identity']['reference']['commit']['sha'] = OTHER
        path.write_text(json.dumps(copied) + '\n' + ''.join(lines))

    commit(repo, 'T2.0c: a stale line first', then(rerecord(NEW), duplicate))
    found = run(repo, checkpoint).problems
    assert any(
        f'{NEW}.jsonl' in p and re.search(r'line 1\b|:1\b', p) for p in found
    ), found


@pytest.mark.parametrize(
    'seed', ['its reference field names another SHA', 'it is deleted']
)
def test_a_vector_file_s_reference_field_must_name_the_sha(
    repo: Path, seed: str
) -> None:
    r"""Exit 1 (review round 1): the check reads the vector file's
    reference field, the one its consistency test reads, not any mention
    of the SHA: the field naming another SHA, with the approved one in a
    note, is refused naming the file; so is a vector file the consistency
    tests require, deleted by the candidate."""
    checkpoint = approved_checkpoint(repo)
    twin_passes(repo, checkpoint)

    def seeded(r: Path) -> None:
        path = r / VECTORS
        if seed == 'it is deleted':
            path.unlink()
            return
        data = json.loads(path.read_text())
        data['reference'] = OTHER
        data['note'] = f'to be re-recorded at {NEW}'
        path.write_text(json.dumps(data, indent=2) + '\n')

    commit(repo, f'T2.0c: {seed}', then(rerecord(NEW), seeded))
    found = run(repo, checkpoint).problems
    assert any(VECTORS in p for p in found), found


@pytest.mark.parametrize(
    'name',
    [f'{NEW}.txt', NEW, f'{NEW}.jsonl.txt', f'{NEW}/{NEW}.jsonl'],
)
def test_the_new_recording_is_exactly_the_sha_s_jsonl(
    repo: Path, name: str
) -> None:
    r"""Exit 1 (review round 1): the replacement is exactly
    ``contract/reference/<the approved full SHA>.jsonl``, the file
    ``gate.sh`` reads; under another extension, none, or another path, it
    is refused by the reference rule."""
    checkpoint = approved_checkpoint(repo)
    twin_passes(repo, checkpoint)

    def moved(r: Path) -> None:
        source = r / REFERENCE / f'{NEW}.jsonl'
        target = r / REFERENCE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        source.unlink()
        target.write_bytes(data)

    commit(repo, f'T2.0c: recorded as {name}', then(rerecord(NEW), moved))
    assert not (repo / REFERENCE / f'{NEW}.jsonl').exists()
    found = run(repo, checkpoint).problems
    assert RULE in rules(found), found


#: The t132g golden, a JSON-lines file citing the reference on every line
#: (``identity.reference.commit.sha``); frozen from T2.0f (it keeps the
#: SHA it was recorded at), so the later-line test builds its own copy.
GOLDEN = 'tests/golden/t132g-fixture-cases/python-ref.jsonl'
#: The later-line test's JSON-lines citing file: the golden's lines under
#: a path the test adds to the citation table, with the field every line
#: holds.
LINES = 'contract/t20e-json-lines.jsonl'
LINES_FIELD = ('identity', 'reference', 'commit', 'sha')


@pytest.mark.parametrize(
    'seed',
    [
        'empty',
        'whitespace only',
        'malformed JSON naming the new SHA',
        'malformed JSON without it',
    ],
)
def test_a_vector_file_that_is_no_longer_json_is_refused(
    repo: Path, seed: str
) -> None:
    r"""Exit 1 (review round 2): a citing vector file that was JSON at the
    checkpoint and is not after an approved re-record (truncated, blank,
    or malformed, whether or not its text names the new SHA) is refused,
    naming the file: its reference field can't be read, so it can't
    agree."""
    checkpoint = approved_checkpoint(repo)
    twin_passes(repo, checkpoint)
    text = {
        'empty': '',
        'whitespace only': '  \n\t\n',
        'malformed JSON naming the new SHA': (
            f'{{"reference": "{OTHER}", "note": "moved to {NEW}", }}\n'
        ),
        'malformed JSON without it': f'{{"reference": "{OTHER}",\n',
    }[seed]

    def broken(r: Path) -> None:
        (r / VECTORS).write_text(text)

    commit(repo, f'T2.0c: {seed}', then(rerecord(NEW), broken))
    found = run(repo, checkpoint).problems
    assert any(VECTORS in p for p in found), found


@pytest.mark.parametrize(
    ('field', 'left'),
    [
        ('identity.reference.commit.sha', 'the old SHA'),
        ('identity.reference.commit.sha', 'another SHA'),
        ('recorded.reference, on line 2 only', 'the old SHA'),
    ],
)
def test_a_later_line_of_a_json_lines_file_is_checked(
    repo: Path, field: str, left: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""Exit 1 (review round 2): a JSON-lines citing file is judged line by
    line, after an approved re-record that moved its first line. Refused,
    naming the file: line 2's ``identity.reference.commit.sha`` (the field
    every line holds) still naming the old SHA or naming another; and a
    citation only line 2 holds (``recorded.reference``, added at the
    checkpoint) still naming the old SHA.

    The file is the test's own (T2.0f, once the t132g goldens were
    frozen): :data:`GOLDEN`'s lines at :data:`LINES`, landed at the
    checkpoint and added to the ratchet's citation table with the field
    every line holds, as a JSON-lines file of the table is."""
    old = old_sha(repo)
    value = old if left == 'the old SHA' else OTHER
    only_later = field.startswith('recorded.reference')
    monkeypatch.setitem(ratchet.CITATIONS, LINES, LINES_FIELD)

    def add_lines(r: Path) -> None:
        lines = [json.loads(x) for x in (r / GOLDEN).read_text().splitlines()]
        assert len(lines) > 1
        assert all(
            x['identity']['reference']['commit']['sha'] == old for x in lines
        )
        if only_later:
            lines[1]['recorded'] = {'reference': old}
        (r / LINES).write_text(''.join(json.dumps(x) + '\n' for x in lines))

    land(repo, 'T2.0b', add_lines)
    checkpoint = approved_checkpoint(repo)
    twin_passes(repo, checkpoint)

    def later(r: Path) -> None:
        path = r / LINES
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        assert len(lines) > 1
        assert lines[0]['identity']['reference']['commit']['sha'] == NEW
        if only_later:
            assert 'recorded' not in lines[0]
            lines[1]['recorded']['reference'] = value
        else:
            lines[1]['identity']['reference']['commit']['sha'] = value
        path.write_text(''.join(json.dumps(x) + '\n' for x in lines))

    commit(repo, f'T2.0c: line 2 left at {left}', then(rerecord(NEW), later))
    found = run(repo, checkpoint).problems
    assert any(LINES in p for p in found), found


def test_a_citing_file_with_dots_and_brackets_in_its_keys_passes(
    repo: Path,
) -> None:
    r"""Exit 1 (review round 2), a legitimate case: a citing file whose
    JSON keys hold dots or brackets (``"reference.note"``,
    ``"cases[0]"``) and cite the old SHA in their text, every citation
    moved to the new SHA by the re-record, passes with no problem."""
    old = old_sha(repo)

    def keys(r: Path) -> None:
        path = r / VECTORS
        data = json.loads(path.read_text())
        data['reference.note'] = f'Recorded at {old}'
        data['cases[0]'] = f'Recorded at {old}'
        path.write_text(json.dumps(data, indent=2) + '\n')

    land(repo, 'T2.0b', keys)
    checkpoint = approved_checkpoint(repo)
    commit(repo, 'T2.0c: the reference re-recorded', rerecord(NEW))
    data = json.loads((repo / VECTORS).read_text())
    assert data['reference.note'] == data['cases[0]'] == f'Recorded at {NEW}'
    found = run(repo, checkpoint).problems
    assert found == [], found


# --- exit 2: the approval can't be forged ------------------------------------------

#: Each seeded row, and the rows it stands for.
FORGED = {
    'a row without the SHA': row(
        'The reference moves to the new recording of rust-core-reference'
    ),
    'a row naming another SHA': row(approval(OTHER)),
    'an abbreviated SHA (8)': row(approval(NEW[:8])),
    'an abbreviated SHA (12)': row(approval(NEW[:12])),
    'an abbreviated SHA (39)': row(approval(NEW[:39])),
    'a row without the word reference': row(f'The recording moves to `{NEW}`'),
    'a proposed row': row(approval(NEW), status='proposed'),
    "a reviewer's approval": row(
        approval(NEW), status='approved by reviewer 2026-10-16'
    ),
    "an orchestrator's row, owner to confirm": row(
        approval(NEW), status='orchestrator, owner to confirm'
    ),
    'not approved by the owner': row(
        approval(NEW), status='not approved by owner 2026-10-16'
    ),
    'an approval withdrawn': row(
        approval(NEW), status='approved by owner 2026-10-16; withdrawn'
    ),
    'the owner approving in the text, not the status': row(
        approval(NEW) + ', approved by owner 2026-10-16', status='proposed'
    ),
}


@pytest.mark.parametrize('seed', sorted(FORGED))
def test_a_row_that_is_not_the_owner_s_approval_admits_nothing(
    repo: Path, seed: str
) -> None:
    r"""Exit 2's seeded faults: with only such a row, the replacement is
    refused by today's name. The twin, the owner's row naming the SHA,
    passes on the same history."""
    good = approved_checkpoint(repo)
    twin_passes(repo, good)
    git(repo, 'reset', '-q', '--hard', 'HEAD~2')
    checkpoint = approved_checkpoint(repo, FORGED[seed])
    commit(repo, 'T2.0c: the reference re-recorded', rerecord(NEW))
    found = run(repo, checkpoint).problems
    assert refused_by_name(found), found


def test_a_row_outside_a_phase_decisions_file_admits_nothing(
    repo: Path,
) -> None:
    r"""Exit 2: the owner's row counts only in a ``phase-*/decisions.md``
    (where ``ratchet.py`` reads every owner approval, T2.0); the same row
    in another file of the design admits nothing."""
    good = approved_checkpoint(repo)
    twin_passes(repo, good)
    git(repo, 'reset', '-q', '--hard', 'HEAD~2')
    checkpoint = land(
        repo,
        'T2.0d',
        decisions(
            row(approval(NEW)),
            path='docs/design/rust-core/phase-2/notes.md',
        ),
    )
    commit(repo, 'T2.0c: the reference re-recorded', rerecord(NEW))
    assert refused_by_name(run(repo, checkpoint).problems)


def test_a_row_the_candidate_writes_itself_admits_nothing(repo: Path) -> None:
    r"""Exit 2 (orchestrator's decision: the approval row is read only at
    the checkpoint, never from the candidate): a candidate that replaces
    the reference and adds its own owner-approval row naming the new SHA
    to ``phase-2/decisions.md`` is refused by today's name, as with no
    row. The twin, the same row landed at the checkpoint, passes first, so
    the refusal is the row's placement."""
    start = git(repo, 'rev-parse', 'HEAD')
    good = approved_checkpoint(repo)
    twin_passes(repo, good)
    git(repo, 'reset', '-q', '--hard', start)

    def both(r: Path) -> None:
        rerecord(NEW)(r)
        decisions(row(approval(NEW)))(r)

    commit(repo, 'T2.0c: the reference re-recorded, approved by itself', both)
    assert row(approval(NEW)) in (repo / DECISIONS).read_text()
    found = run(repo, start).problems
    assert refused_by_name(found), found


def test_two_reference_files_left_behind_are_refused(repo: Path) -> None:
    r"""Exit 2's seeded fault: the new recording added beside the old one
    (two reference files) is refused, naming the old file."""
    checkpoint = approved_checkpoint(repo)
    old = old_sha(repo)
    twin_passes(repo, checkpoint)
    commit(repo, 'T2.0c: both recordings', rerecord(NEW, keep_old=True))
    assert len(list((repo / REFERENCE).glob('*.jsonl'))) == 2
    found = run(repo, checkpoint).problems
    assert any(old in p and RULE in rules([p]) for p in found), found


# --- exit 5: the post-landing state ------------------------------------------------


@pytest.fixture
def landed(repo: Path) -> tuple[Path, str]:
    r"""The phase branch after the change landed: the owner's row, then
    the re-recording, each landed (the second the checkpoint)."""
    approved_checkpoint(repo)
    return repo, land(repo, 'T2.0c', rerecord(NEW))


def test_the_next_package_passes_after_the_landing(
    landed: tuple[Path, str],
) -> None:
    r"""Exit 5: the phase branch at the landing judges the next package
    (no reference change) with no problem; the landing itself, judged
    against its own landed state, has none either. Control: passes on
    the base (the landing is committed here, not judged), and must keep
    passing."""
    repo, checkpoint = landed
    assert run(repo, checkpoint).problems == []
    commit(repo, 'T2.1: the next package')
    assert run(repo, checkpoint).problems == []


@pytest.mark.parametrize('seed', ['a line changed', 'a line removed'])
def test_the_per_case_rules_hold_the_new_recording(
    landed: tuple[Path, str], seed: str
) -> None:
    r"""Exit 5: after the landing, the new recording is the one the
    per-case rules hold: a line changed or removed is refused by case.
    Control: passes on the base, and must keep passing."""
    repo, checkpoint = landed

    def edit(r: Path) -> None:
        path = r / REFERENCE / f'{NEW}.jsonl'
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        line = next(x for x in lines if x['case'] == CASE)
        if seed == 'a line changed':
            line['outcome']['stage'] = 'materialize'
        else:
            lines.remove(line)
        path.write_text(''.join(json.dumps(x) + '\n' for x in lines))

    commit(repo, f'T2.1: {seed}', edit)
    found = run(repo, checkpoint).problems
    assert any(p.startswith(f'{RULE}: {CASE}: its line') for p in found), found


def test_a_spent_approval_admits_no_further_reference(
    landed: tuple[Path, str],
) -> None:
    r"""Exit 5 and exit 2: the landed row approves ``NEW`` only; a further
    replacement to another SHA, every citation moved, is refused by name,
    and passes once the owner's row names that SHA."""
    repo, checkpoint = landed
    commit(repo, 'T2.1: another reference', rerecord(THIRD))
    assert refused_by_name(run(repo, checkpoint).problems)
    git(repo, 'reset', '-q', '--hard', checkpoint)
    again = land(
        repo, 'T2.0f', decisions(row(approval(THIRD), ident='P2E-902'))
    )
    commit(repo, 'T2.1: another reference', rerecord(THIRD))
    found = run(repo, again).problems
    assert found == [], found


def test_the_copied_history_cites_the_old_sha_where_the_real_one_does(
    origin: tuple[Path, str],
) -> None:
    r"""Not vacuous: the temporary repository cites the reference SHA in
    every file this checkout does (outside ``docs/``), the vector file
    the seeds name and ``reference-env.txt`` among them, and in the
    ratchet's frozen recordings, which name it as at the move they were
    frozen across; and the seeded SHAs are none of the real one."""
    source, _ = origin
    old = old_sha(source)
    assert citing(source, old) == sorted(
        {*citing(ROOT, old), *getattr(ratchet, 'FROZEN', {})}
    )
    assert {VECTORS, ENV} <= set(citing(source, old))
    assert old not in (NEW, OTHER, THIRD)
