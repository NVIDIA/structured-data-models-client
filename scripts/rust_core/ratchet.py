#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The ratchet: a candidate judged against its checkpoint (design,
"Reference observations and the gate" and "Register v2"; decisions H2, H3,
H19 and A6). ``gate.sh`` runs the checkpoint's copy of this file.

    ratchet.py checkpoint [--repo DIR] (--base REV | --phase-push BEFORE)
        [--epic] [--epic-ref REF]
    ratchet.py check [--repo DIR] --checkpoint SHA [--epic]
        [--protected-paths TEXT]
        [--observations FILE]... [--effective-out FILE]
    ratchet.py accept [--repo DIR] [--checkpoint SHA] --observations FILE...
    ratchet.py effective [--repo DIR] (--base REV | --phase-push BEFORE)
        [--epic] [--epic-ref REF] --out FILE
    ratchet.py where [--repo DIR]

``checkpoint`` resolves the checkpoint of the candidate, ``--repo``'s
``HEAD``: ``git merge-base CANDIDATE BASE``, never read from evidence.
``--base`` is the phase-branch tip at package start or at run time; when
the candidate is a landing commit (``Land <ID>: ...``, made by ``land.py``),
its recorded ``Base-SHA`` is the base instead, so a landing is judged
against the head it landed on even once the phase branch holds it. When
``--base`` moved while the run went on, and the merge base is no landing
commit but a package the candidate is stacked on, landed on a head that
is a landing commit, the checkpoint is the merge base with that head, so
every job of a run judges against the same commit (decision E131).
``--phase-push`` is a push to the phase branch, ``BEFORE`` its previous
tip: the push must land exactly one package, the candidate being that
landing commit with ``Base-SHA`` ``BEFORE``, as a fast-forward. It refuses
a candidate that is its own checkpoint (self-comparison: the base already
holds the candidate) and a checkpoint that is not a landing commit, so
advancing the phase branch to an unreviewed commit cannot make that commit
a checkpoint. It prints the checkpoint and whether its harness judges: a
checkpoint without this file is the bootstrap, before T1.14's landing, where
the candidate's harness runs and no acceptance claim is made.

One merge base that is no landing commit is accepted (T2.0): a phase's
first checkpoint, the commit the phase branch was cut at from the epic
branch ``rust-core``, its fork point ``git merge-base PHASE EPIC`` (PHASE
``origin/<phase branch>``, else the local branch, the phase branch as
``phases.py`` names it for CI's branch or the branch checked out; EPIC
``--epic-ref``, else ``origin/rust-core``, else ``rust-core``). The owner put
that commit there, by merging the previous phase and committing to the
epic branch, which is judged against the merged phase's last landing
(``where`` below). Any other commit that is no landing commit, a phase's
own product commit, an older epic commit or an unreviewed commit pushed to
the phase branch, is still refused. A fork point whose own ``ratchet.py``
predates this (Phase 2's) cannot resolve itself; ``gate.sh`` then judges
with Phase 1's last landing's harness, under the conditions it pins and
reads from git itself (the bootstrap T2.0 records).

``check`` holds the candidate, ``--repo``'s committed ``HEAD``, to the
checkpoint's files, read from git at ``--checkpoint``:

- ``protected``: each changed path ``contract/protected-paths.txt`` (the
  checkpoint's) protects is named in the landing commit's ``Reviewed-by`` or
  ``Protected-paths`` trailer, or before landing in ``--protected-paths``
  (the text the orchestrator gives ``land.py``); with neither, the paths are
  listed for the landing to name;
- ``reference``: reference immutability: each recorded reference line is
  still there, its semantic projection unchanged (:func:`semantic`); a line
  may be added; a changed or removed line passes only under an approved
  register entry naming its case on ``python-ref``, or with its case's
  approved removal; a new reference commit is the owner's: a reference
  file added is refused, naming it, unless a row of the checkpoint's
  decisions approves its name (T2.0e, below);
- ``cases``: a case removed, demoted from acceptance to diagnostic, or with
  its file changed passes only under an approved entry naming the case (for
  a removal or demotion, each baselined runner's exact old outcome and
  ``removed`` or ``diagnostic``); ``bases``: a base changed or removed passes
  only under an approved entry naming a case built on it, and one added
  passes; in both, only an entry this change adds or changes against the
  checkpoint's register authorizes (:func:`in_this_change`, T1.29): an entry
  approved by an earlier landing authorized that landing's change, not
  every later one to the same case or base;
- ``epic``: with ``--epic`` (the epic branch's head, as ``where`` names it
  for a push to ``rust-core``), every protected path, by the checkpoint's
  list or the head's, has the same entry in the head's tree as in the
  checkpoint's: a protected change reaches the epic branch only through a
  reviewed phase landing the owner merges (:func:`check_epic`, T2.0; its
  threat model is honest drift);
- ``register``: ``register.py validate``, with the checkpoint's register for
  never-reused IDs; entries added or changed are listed, ``approved_by:
  owner`` ones first, for the owner's check-ins;
- ``policy`` and ``availability``: a relaxation of ``compare-policy.yaml`` or
  ``availability.yaml`` passes only when the candidate's file lists it under
  ``relaxations`` with a row of the checkpoint's decisions.md approved by the
  owner (a row the package writes itself never authorizes);
- ``known-differing``: the candidate's ``KNOWN_DIFFERING`` file a subset of
  the checkpoint's;
- with observations, ``baselines``: each checkpoint baseline
  (``contract/baselines/<cell>/<runner>.jsonl``) of a given runner equals the
  observation's projection (``compare-policy.yaml``'s ``baseline`` fields, the
  checkpoint's), or an approved entry states that case, cell, runner and
  exact old and new outcome, an owner-approved one for a new refusal;
  ``INTERNAL_ERROR`` never passes, held or proposed, but python-old's equal
  to the checkpoint's reference line less host warnings and their origins
  (:func:`old_client_crash`), which is noted as the old client's crash; the
  candidate's proposed baselines equal the observations and hold every
  acceptance case; and ``availability``: each
  observation has every field its runner has available and none it has not.

A baseline is the checkpoint's, so a branch-side edit of one changes
nothing the candidate is judged by; the identity envelope is outside every
projection, so an identity-only change passes.

``check --effective-out`` also writes each baselined case's effective
expected outcome by runner (:func:`effective`: the checkpoint's baseline, or
an approved entry's exact transition from it; on a first admission, an
approved entry's transition from ``none``, from the reference's outcome),
which ``gate.sh`` hands the comparator as ``--expect``, so an approved
change passes its outcome rule while every other field is still compared
with the reference.

``accept`` writes the proposed baselines from the observations of every
runner given, each runner judged alone: each case it meets, its effective
expectation where there is one (``--checkpoint``'s baseline, or on a first
admission the reference's outcome, either moved by an approved entry), and
the committed reference's projection otherwise (the first baselines, T1.14,
had only the latter), none where its line is an ``INTERNAL_ERROR`` the
baselines rule refuses; it is the package's to commit and the reviewer's to
review (H3).

``effective`` writes the effective expectations ``check --effective-out``
writes, for the checkpoint ``checkpoint`` resolves, with no other rule run:
the comparisons outside the gate (``contract/tests``, T1.18) hand them to
``compare.py --expect`` as the gate does, so an approved outcome change
passes there too and an unapproved one fails. It fails on a register
``register.py validate`` refuses, since only a valid register's approved
entries count; a checkpoint before the ratchet (the bootstrap) has no
baselines, and so no expectations.

``where`` prints the gate's ``--base`` or ``--phase-push`` for the CI run
it is in, from GitHub's environment, or nothing outside the phase branches
and the epic branch (:func:`ci_base`): on a push to a phase branch
(``rust-core-phase-N``, any N) ``--phase-push``, on a branch under one
``--base origin/<that phase branch>``, and on a push to ``rust-core``
``--base`` the merged phase's last landing and ``--epic``, so the epic
branch is judged like a moved landing of that phase's head, with rule
``epic`` (T2.0). The landing is the second parent of a phase merge on the
head's first-parent history, never a landing-shaped commit elsewhere, and
its record must carry Reviewed-by (:func:`epic_merge`). ``--epic`` takes
the base as given, never the head's own trailers, and refuses any base but
that landing; a harness that predates it refuses the flag, so ``rust-core``
holding T2.0 but judged by an older landing fails closed.

A new reference commit (T2.0e): a candidate that adds a reference file
passes only when a row of the checkpoint's decisions approves its name
(:func:`approved_references`): its status is the owner's approval as
:func:`owner_approved` reads it, and its Decision cell, the second, holds
the word "reference" and exactly one full 40-hex SHA in backticks, ``NEW``
(the form the Phase 2 decisions file's header states; KO2-1 begins
``Reference: `<NEW>```). Only the checkpoint's rows count, never one the
candidate adds, as for a relaxation: the owner's row lands before the
package that re-records. Then, each named when it fails: the file is exactly
``contract/reference/<NEW>.jsonl``, which the gate reads; ``NEW`` is no
recorded reference's name; it is the only file added, and the only one left
(the old one removed); ``contract/reference-env.txt`` names ``NEW`` in its
one ``reference:`` line; no file outside ``docs/`` (history) and
``contract/reference/`` cites the old SHA in full; each file that cited it
at the checkpoint is still there, and each of its fields that held the old
SHA, by path (on every line of a JSON-lines file), holds ``NEW`` (the fields
the reference SHA consistency tests read: ``reference``,
``recorded.reference``, ``identity.reference.commit.sha``), a text field
``NEW`` and not the old SHA; every line of the new file, read as written
before any is indexed, names a case, names it once (a second line of a case
is refused by file and line) and records ``NEW`` as its reference commit;
and the new recording carries every case the old one did, but one whose
removal is approved. The per-case comparison does not apply across the
change: instead every case whose semantic projection differs from its
recorded line is listed in the notes, by case and with the fields that
differ (named, never their values), for the reviewers, with the move itself
and the row approving it; a case recorded alike is not listed. The landed
file is then the one the per-case rules hold, and the row admits no further
file: another commit needs its own row.

The decisions the rules read (owner approvals of relaxations and of a new
reference commit) are every phase's: each
``docs/design/rust-core/phase-N/decisions.md`` of the checkpoint, Phase 1's
rows in force and each later phase's added; a file missing, unreadable or
malformed fails the ``decisions`` rule by name
(``capture_check.phase_decisions``, T2.0).

Exit status 1 on any problem. Each problem names its rule and its case;
input values are never printed.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
POLICY = 'contract/compare-policy.yaml'
AVAILABILITY = 'contract/availability.yaml'
PROTECTED = 'contract/protected-paths.txt'
BASELINES = 'contract/baselines'
REFERENCE = 'contract/reference'
#: The reference environment, naming the reference commit (T1.13).
ENV = 'contract/reference-env.txt'
CASES = 'contract/cases'
BASES = 'contract/bases.yaml'
REGISTER = 'docs/design/rust-core/register.yaml'
#: Where every phase's ``decisions.md`` is (T2.0).
DECISIONS_DIR = 'docs/design/rust-core'
KNOWN_DIFFERING = 'rust/crates/sdm-core/tests/pql/known-differing.json'
#: The cell the baselines were taken in: the reference machine's (design,
#: "Runtime cells"). Their projection is platform-independent, so every cell
#: is held to them.
CELL = 'ref-pandas3'
RUNNERS = ('python-ref', 'python-core', 'python-old', 'rust-host')
PYTHON = ('python-ref', 'python-core', 'python-old')
#: The core operations that read data, whose case has a driver record on
#: rust-host.
READS = ('describe_source', 'materialize', 'predict')


def _phases() -> ModuleType:
    r"""This harness's ``phases.py``: the branch names and package IDs."""
    name = 'sdm_ratchet_phases'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / 'phases.py')
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # Registered first: a dataclass reads its module from sys.modules.
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


PHASES = _phases()
#: A landing commit's subject, as land.py writes it.
LANDING = re.compile(rf'Land ({PHASES.PACKAGE_ID.pattern}): ')
#: The epic branch's ref the fork point is read from when ``--epic-ref`` names
#: none: the remote's, else a local branch.
EPIC_REFS = (f'refs/remotes/origin/{PHASES.EPIC}', f'refs/heads/{PHASES.EPIC}')
#: An owner's approval, as a decisions row's status (its last column) reads
#: it: these words at its start and a date, followed by nothing, a
#: parenthesis, or a note after "; " or ", " (the forms decisions.md uses).
OWNER = re.compile(r'approved by owner \d{4}-\d{2}-\d{2}(?:(?: \(|[;,] ).*)?')
#: A note that withdraws or qualifies the approval voids it: fail closed.
WITHDRAWN = re.compile(
    r'not approved|unapproved|reject|revoke|withdraw|supersede|to confirm',
    re.IGNORECASE,
)
PANIC = 'INTERNAL_ERROR'
#: An outcome's fields, as a projection and an entry's text name them.
OUTCOME_FIELDS = (
    'outcome.result',
    'outcome.stage',
    'outcome.error',
    'outcome.code',
)


def owner_approved(status: str) -> bool:
    r"""The status is the owner's approval, read from its start, never a
    substring of another status ("not approved by owner ..."); shared with
    ``coverage.py``'s rule 8 (T1.15)."""
    status = status.strip()
    return bool(OWNER.fullmatch(status)) and not WITHDRAWN.search(status)


class Unreadable(Exception):
    r"""A file the ratchet cannot read, named."""


def _module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.cache
def register_module() -> ModuleType:
    r"""This harness's ``register.py``."""
    return _module('sdm_register', HERE / 'register.py')


@functools.cache
def capture_check() -> ModuleType:
    r"""This harness's ``capture_check.py``: the E35 mask and the decisions
    rows."""
    return _module('capture_check', REPO / 'contract/runners/capture_check.py')


# --- git ---------------------------------------------------------------------


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ['git', '-C', str(repo), *args],
        capture_output=True,
        encoding='utf-8',
        check=False,
    )
    if done.returncode != 0:
        raise Unreadable(f'git {args[0]}: {done.stderr.strip()}')
    return done.stdout


class Tree:
    r"""A commit's files, read from git through one ``git cat-file --batch``
    process."""

    def __init__(self, repo: Path, sha: str) -> None:
        self.repo, self.sha = repo, sha
        self._cache: dict[str, str | None] = {}
        self._batch: subprocess.Popen[bytes] | None = None

    def read(self, path: str) -> str | None:
        if path in self._cache:
            return self._cache[path]
        if self._batch is None:
            self._batch = subprocess.Popen(
                ['git', '-C', str(self.repo), 'cat-file', '--batch'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
        assert self._batch.stdin is not None and self._batch.stdout is not None
        self._batch.stdin.write(f'{self.sha}:{path}\n'.encode())
        self._batch.stdin.flush()
        header = self._batch.stdout.readline().split()
        text: str | None = None
        if header[-1:] == [b'missing']:
            pass
        elif len(header) == 3 and header[1] == b'blob':
            data = self._batch.stdout.read(int(header[2]))
            self._batch.stdout.read(1)
            try:
                text = data.decode('utf-8')
            except UnicodeDecodeError:
                raise Unreadable(f'{path}: not UTF-8') from None
        elif len(header) == 3:
            # Not a file (a tree): its contents are skipped.
            self._batch.stdout.read(int(header[2]) + 1)
        self._cache[path] = text
        return text

    def files(self, prefix: str) -> list[str]:
        # NUL-separated: under git's default core.quotePath a path with a
        # byte past ASCII is printed quoted and escaped, never its name.
        out = git(
            self.repo, 'ls-tree', '-r', '-z', '--name-only', self.sha, prefix
        )
        return sorted(name for name in out.split('\0') if name)

    def __del__(self) -> None:
        if self._batch is not None:
            self._batch.kill()
            self._batch.wait()


def trailers(repo: Path, sha: str) -> dict[str, str]:
    r"""A commit's trailers, by key; a repeated key's values joined."""
    found: dict[str, str] = {}
    out = git(repo, 'log', '-1', '--format=%(trailers:unfold)', sha)
    for line in out.splitlines():
        key, sep, value = line.partition(':')
        if sep:
            key = key.strip()
            found[key] = (found.get(key, '') + ' ' + value.strip()).strip()
    return found


def subject(repo: Path, sha: str) -> str:
    return git(repo, 'log', '-1', '--format=%s', sha).strip()


def is_landing(repo: Path, sha: str) -> bool:
    r"""``sha`` is a landing commit as land.py makes them: its subject, and
    the base and checkpoint it records."""
    found = trailers(repo, sha)
    return bool(
        LANDING.match(subject(repo, sha))
        and found.get('Base-SHA')
        and found.get('Checkpoint-SHA')
    )


# --- the checkpoint ------------------------------------------------------------


def landed_on(repo: Path, sha: str, base: str) -> str | None:
    r"""The phase head ``sha`` was landed on, or None (decision E131): the
    ``Base-SHA`` of the landing commit ``base`` holds that landed ``sha``,
    one on the ancestry path from ``sha`` to ``base`` whose recorded
    ``Base-SHA`` is a proper ancestor of ``sha``, when that ``Base-SHA`` is
    itself a landing commit.

    A run on a lane or topic branch reads the phase branch's tip when each
    job starts. When a package the candidate is stacked on lands mid-run,
    the merge base with the tip becomes that package's last commit, which
    is no landing commit; the merge base with the head that package landed
    on is what the run's earlier jobs found. A merge base pushed to the
    phase branch with no landing after it, or landed on a head that is no
    landing commit, gives None, and stays refused."""
    out = git(
        repo, 'log', '--ancestry-path', '--format=%H%x00%s', f'{sha}..{base}'
    )
    for line in out.splitlines():
        commit, _, title = line.partition('\0')
        if not (LANDING.match(title) and is_landing(repo, commit)):
            continue
        recorded = git(
            repo,
            'rev-parse',
            f'{trailers(repo, commit)["Base-SHA"]}^{{commit}}',
        ).strip()
        # ``sha`` in the range it landed: after its Base-SHA, not at it.
        if (
            recorded != sha
            and git(repo, 'merge-base', recorded, sha).strip() == recorded
        ):
            return recorded if is_landing(repo, recorded) else None
    return None


def epic_merge(repo: Path, sha: str) -> tuple[str, str] | None:
    r"""The newest phase merge ``sha``'s first-parent history holds (T2.0),
    ``sha`` included: a merge whose second parent is a landing commit, and
    that landing, the merged phase's last; None when there is none. Only a
    merge's second parent counts, never a landing-shaped commit elsewhere
    in the history, and the landing's record must carry Reviewed-by, as
    every landing ``land.py`` makes does: one without it fails here, by
    name, never giving way to an older merge. A landing commit on the
    first-parent history itself, before any such merge, is the epic branch
    fast-forwarded to a landing rather than merged: refused, by name, since
    the owner's merge of a phase must be ``--no-ff``."""
    out = git(repo, 'rev-list', '--first-parent', '--parents', sha)
    for line in out.splitlines():
        merge, *parents = line.split()
        if len(parents) < 2 and is_landing(repo, merge):
            raise Unreadable(
                f"{merge[:12]}, a landing commit, is on the epic branch's "
                'first-parent history itself: the epic branch was '
                "fast-forwarded to a landing, and the owner's merge of a "
                'phase into it must be a --no-ff merge'
            )
        if len(parents) >= 2 and is_landing(repo, parents[1]):
            if not trailers(repo, parents[1]).get('Reviewed-by'):
                raise Unreadable(
                    f'the phase merge {merge[:12]} brings in {parents[1][:12]}, '
                    'a landing record with no Reviewed-by: no reviewed '
                    'landing to judge the epic branch against'
                )
            return merge, parents[1]
    return None


def epic_checkpoint(repo: Path, sha: str) -> str | None:
    r"""The merged phase's last landing as ``sha`` holds it
    (:func:`epic_merge`); None when there is none."""
    found = epic_merge(repo, sha)
    return found[1] if found else None


def epic_tip(repo: Path, epic: str | None) -> tuple[str, str] | None:
    r"""The epic branch's tip and the ref it was read from: ``epic`` when
    given, else the first of :data:`EPIC_REFS` the repository holds."""
    for ref in (epic,) if epic else EPIC_REFS:
        try:
            return git(repo, 'rev-parse', f'{ref}^{{commit}}').strip(), ref
        except Unreadable:
            continue
    return None


def phase_name(repo: Path, environ: dict[str, str] | None = None) -> str:
    r"""The phase branch the run is on: ``phases.phase_branch`` of CI's
    branch (``GITHUB_REF_NAME``) or the branch checked out, after
    ``SDM_PHASE_BRANCH``."""
    environ = dict(os.environ) if environ is None else environ
    try:
        return str(
            PHASES.phase_branch(
                branch=environ.get('GITHUB_REF_NAME') or _checked_out(repo),
                environ=environ,
            )
        )
    except PHASES.PhaseError as error:
        raise Unreadable(str(error)) from None


def fork_point(
    repo: Path, phase: str | None, epic: str | None
) -> tuple[str | None, str]:
    r"""The phase branch's fork point from the epic branch, ``git
    merge-base PHASE EPIC`` (``origin/<phase>``, else the local branch;
    the epic ``epic``, else :data:`EPIC_REFS`), and how it was read."""
    phase = phase or phase_name(repo)
    tip = epic_tip(repo, epic)
    if tip is None:
        return None, f'no epic branch {epic or PHASES.EPIC} here'
    for ref in (f'refs/remotes/origin/{phase}', f'refs/heads/{phase}'):
        try:
            head = git(repo, 'rev-parse', f'{ref}^{{commit}}').strip()
        except Unreadable:
            continue
        try:
            fork = git(repo, 'merge-base', head, tip[0]).strip()
        except Unreadable:
            return None, f'{ref} and {tip[1]} have no merge base'
        return fork, f'git merge-base {ref} {tip[1]}'
    return None, f'no phase branch {phase} here'


@dataclass
class Checkpoint:
    sha: str
    base: str
    candidate: str
    bootstrap: bool
    notes: list[str] = field(default_factory=list)


def resolve(
    repo: Path,
    base: str | None = None,
    phase_push: str | None = None,
    epic: str | None = None,
    phase: str | None = None,
    epic_mode: bool = False,
) -> tuple[Checkpoint | None, list[str]]:
    r"""The candidate's checkpoint, or the problems that leave it none;
    ``epic`` the epic branch's ref and ``phase`` the phase branch
    (:func:`phase_name` by default) the fork point is read from.
    ``epic_mode`` (``--epic``, which ``where`` prints for a push to the
    epic branch): the candidate is the epic branch's head, judged against
    ``base``, which must be the merged phase's last landing the head holds
    (:func:`epic_merge`), whatever the head's own subject or trailers say."""
    candidate = git(repo, 'rev-parse', 'HEAD^{commit}').strip()
    notes: list[str] = []
    if epic_mode:
        if base is None or phase_push is not None:
            return None, ['--epic judges the epic branch against --base']
        given = git(repo, 'rev-parse', f'{base}^{{commit}}').strip()
        found_merge = epic_merge(repo, candidate)
        if found_merge is None or found_merge[1] != given:
            return None, [
                f"--epic: the base {given[:12]} is not the merged phase's "
                'last landing the head holds ('
                + (found_merge[1][:12] if found_merge else 'none')
                + ')'
            ]
        notes.append(
            f"the epic branch, judged against the merged phase's last "
            f'landing {given[:12]}'
        )
        bootstrap = (
            Tree(repo, given).read('scripts/rust_core/ratchet.py') is None
        )
        return Checkpoint(given, given, candidate, bootstrap, notes), []
    # A base the run read from the live phase branch, which may move while
    # it runs, rather than one recorded in a commit (E131).
    live = phase_push is None
    if phase_push is not None:
        try:
            before = git(repo, 'rev-parse', f'{phase_push}^{{commit}}').strip()
        except Unreadable:
            # A previous tip the push's history does not hold, as after a
            # forced push: named, not git's unknown revision.
            return None, [
                'the push is not a fast-forward of the previous tip: '
                f'{phase_push[:12]} is not in the history pushed'
            ]
        landings = [
            sha
            for sha in git(repo, 'rev-list', f'{before}..{candidate}').split()
            if LANDING.match(subject(repo, sha))
        ]
        problems = []
        if (
            git(repo, 'merge-base', before, candidate).strip() != before
            or before == candidate
        ):
            problems.append(
                'the push is not a fast-forward of the previous tip'
            )
        if landings != [candidate]:
            problems.append(
                'a push to the phase branch lands exactly one package, as '
                f'its tip: it holds {len(landings)} landing commits'
                + ('' if candidate in landings else ', and its tip is none')
            )
        elif trailers(repo, candidate).get('Base-SHA') != before:
            problems.append(
                "the landing's Base-SHA is not the phase branch's previous tip"
            )
        if problems:
            return None, problems
        base = before
    elif is_landing(repo, candidate):
        recorded = trailers(repo, candidate)['Base-SHA']
        notes.append(
            f'the candidate is a landing commit: its recorded Base-SHA '
            f'{recorded[:12]} is the base'
        )
        base = recorded
        live = False
    if base is None:
        return None, ['no base: give --base or --phase-push']
    base = git(repo, 'rev-parse', f'{base}^{{commit}}').strip()
    found = git(repo, 'merge-base', candidate, base).strip()
    landed = (
        landed_on(repo, found, base)
        if live and found != candidate and not is_landing(repo, found)
        else None
    )
    if landed is not None:
        # The merge base with the head the stacked package landed on: the
        # same for every job of a run, whenever the phase branch moves.
        earlier = git(repo, 'merge-base', candidate, landed).strip()
        notes.append(
            f'the merge base {found[:12]} with the phase branch is no '
            'landing commit: a package the candidate is stacked on landed '
            f'on {landed[:12]} after it started, so the merge base with '
            f'that head, {earlier[:12]}, is the checkpoint'
        )
        found = earlier
    problems = []
    if found == candidate:
        problems.append(
            'self-comparison: the base already holds the candidate, so it '
            'would be its own checkpoint; judge it against the base recorded '
            'before the phase branch advanced'
        )
    if found != candidate and not is_landing(repo, found):
        # A phase's first checkpoint (T2.0): its fork point from the epic
        # branch, the one commit the phase branch holds that no package
        # landed.
        fork, read = fork_point(repo, phase, epic)
        if fork != found:
            problems.append(
                f'the checkpoint {found[:12]} is not a landing commit, so not '
                'a reviewed one: the phase branch holds an unlanded commit; '
                + (
                    f'nor is it the fork point of the phase branch ({read})'
                    if fork is None
                    else f'nor the fork point of the phase branch from the '
                    f'epic branch, {fork[:12]} ({read})'
                )
            )
        else:
            notes.append(
                f'the checkpoint {found[:12]} is no landing commit but the '
                f'first of the phase branch: its fork point ({read})'
            )
    if problems:
        return None, problems
    if found != base:
        notes.append(
            f'the phase branch moved past the checkpoint to {base[:12]}: '
            'rebase onto it and gate again before landing'
        )
    bootstrap = Tree(repo, found).read('scripts/rust_core/ratchet.py') is None
    return Checkpoint(found, base, candidate, bootstrap, notes), []


# --- reading -------------------------------------------------------------------


def _yaml(text: str | None, name: str) -> Any:
    if text is None:
        return None
    try:
        return register_module().read_yaml(text, name)
    except register_module().Unreadable as error:
        raise Unreadable(str(error)) from None


def _json(text: str, name: str) -> Any:
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        raise Unreadable(f'{name}: not JSON it reads') from None


def jsonl(text: str | None, name: str) -> list[dict[str, Any]]:
    r"""The objects of a JSON-lines text, each line read or refused by
    name."""
    found = []
    for number, line in enumerate((text or '').splitlines(), 1):
        item = _json(line, f'{name}:{number}')
        if not isinstance(item, dict):
            raise Unreadable(f'{name}:{number}: not an object')
        found.append(item)
    return found


def cases(tree: Tree) -> dict[str, tuple[dict[str, Any], str]]:
    r"""A tree's cases by ID, each with its file's text."""
    found = {}
    for path in tree.files(CASES):
        if not path.endswith('.json'):
            continue
        text = tree.read(path) or ''
        case = _json(text, path)
        if isinstance(case, dict) and isinstance(case.get('id'), str):
            found[case['id']] = (case, text)
    return found


def baselines(tree: Tree) -> dict[str, dict[str, dict[str, Any]]]:
    r"""A tree's baselines in :data:`CELL`, by runner and case, each line's
    digest checked against its projection."""
    found: dict[str, dict[str, dict[str, Any]]] = {}
    for runner in RUNNERS:
        path = f'{BASELINES}/{CELL}/{runner}.jsonl'
        lines = jsonl(tree.read(path), path)
        by_case = {}
        for line in lines:
            if (
                set(line) != {'case', 'runner', 'cell', 'projection', 'digest'}
                or not isinstance(line['case'], str)
                or line['runner'] != runner
                or line['cell'] != CELL
                or not isinstance(line['projection'], dict)
            ):
                raise Unreadable(f'{path}: a line is not a baseline')
            if line['digest'] != digest(line['projection']):
                raise Unreadable(f'{path}: {line["case"]}: digest differs')
            if line['case'] in by_case:
                raise Unreadable(f'{path}: {line["case"]} twice')
            by_case[line['case']] = line
        if by_case:
            found[runner] = by_case
    return found


def observations(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    r"""Observations by runner and case, from their files."""
    found: dict[str, dict[str, Any]] = {}
    for path in paths:
        lines = jsonl(path.read_text(encoding='utf-8'), path.name)
        if not lines:
            raise Unreadable(f'{path.name}: no observations')
        for line in lines:
            identity = line.get('identity')
            runner = (
                identity.get('runner') if isinstance(identity, dict) else None
            )
            if runner not in RUNNERS or not isinstance(line.get('case'), str):
                raise Unreadable(f'{path.name}: an observation of no runner')
            by_case = found.setdefault(runner, {})
            if line['case'] in by_case:
                raise Unreadable(f'{path.name}: {line["case"]} twice')
            by_case[line['case']] = line
    return found


# --- projections ---------------------------------------------------------------


def at(document: Any, dotted: str) -> Any:
    r"""The value at a dotted path, or None where there is none."""
    for part in dotted.split('.'):
        if not isinstance(document, dict):
            return None
        document = document.get(part)
    return document


def reference_lines(tree: Tree) -> dict[str, dict[str, Any]]:
    r"""The committed ``python-ref`` lines, by case."""
    found: dict[str, dict[str, Any]] = {}
    for path in tree.files(REFERENCE):
        found.update(
            {line['case']: line for line in jsonl(tree.read(path), path)}
        )
    return found


def projection(
    observation: dict[str, Any], fields: list[str]
) -> dict[str, Any]:
    return {name: at(observation, name) for name in fields}


def digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def outcome_text(projected: dict[str, Any] | None) -> str:
    r"""A projected outcome as an entry states it (the schema's form)."""
    if projected is None:
        return 'none'
    result = projected.get('outcome.result')
    text = (
        'success'
        if result == 'success'
        else f'refusal {projected.get("outcome.code") or "-"}'
    )
    if projected.get('outcome.stage'):
        text += f' at {projected["outcome.stage"]}'
    if projected.get('outcome.error'):
        text += f' ({projected["outcome.error"]})'
    return text


def semantic(
    observation: dict[str, Any], ordered: dict[str, Any] | None = None
) -> dict[str, Any]:
    r"""A reference line's semantic projection: the line without its identity
    (checked apart, ``tests/test_contract_reference.py``) and without what
    ``compare.py`` never gates: each statement's text and typed parameters
    and the relations bound (H9's diagnostics), each raw result's and
    payload's SHA-256 (integrity only), the value of ``metadata.request_id``
    (E35), and a statement's in-order digest where its rows compare as a
    multiset (``ordered`` names the others)."""
    line = json.loads(json.dumps(observation))
    line.pop('identity', None)
    driver = line.get('driver') or {}
    capture = driver.get('capture') or {}
    capture.pop('bindings', None)
    places = (ordered or {}).get(line.get('case'), {})
    counters: dict[str, int] = {}
    for statement in capture.get('statements') or []:
        stage = statement.get('stage')
        position = counters.get(stage, 0)
        counters[stage] = position + 1
        statement.pop('sql', None)
        statement.pop('parameters', None)
        for read in statement.get('results') or []:
            read.pop('digest', None)
            if position not in places.get(stage, []):
                (read.get('values') or {}).pop('ordered', None)
    public = line.get('public') or {}
    for payload in public.get('payloads') or []:
        payload.pop('sha256', None)
        payload['parsed'] = capture_check().masked(payload.get('parsed'))
    return line


def old_client_crash(
    runner: str,
    observation: dict[str, Any] | None,
    reference: dict[str, Any] | None,
    ordered: dict[str, Any] | None,
) -> bool:
    r"""Whether ``observation`` is the old client's own crash: python-old's
    ``INTERNAL_ERROR`` equal to the case's reference line at the checkpoint
    in its semantic projection, less the host's warnings and their origins
    (runtime noise: python-old records origins and python-ref none, and
    each cell's libraries warn their own way), the reference's own
    ``StructuredDataError``, held with the reference result as its baseline
    (E110, the owner's). Every other ``INTERNAL_ERROR`` is a panic, which no
    baseline holds."""

    def exact(line: dict[str, Any]) -> dict[str, Any]:
        found = semantic(line, ordered)
        if isinstance(found.get('host'), dict):
            found['host'].pop('warnings', None)
            found['host'].pop('origins', None)
        return found

    return (
        runner == 'python-old'
        and observation is not None
        and reference is not None
        and at(observation, 'outcome.code') == PANIC
        and exact(observation) == exact(reference)
    )


# --- entries -------------------------------------------------------------------


def approved(register: Any) -> list[dict[str, Any]]:
    r"""The register's approved entries; an unreadable register has none."""
    if not isinstance(register, dict) or not isinstance(
        register.get('entries'), list
    ):
        return []
    return [
        entry
        for entry in register['entries']
        if isinstance(entry, dict) and entry.get('status') == 'approved'
    ]


def _this_change(
    report: Report,
    entries: list[dict[str, Any]],
    checkpoint: Tree,
    bootstrap: bool,
) -> list[dict[str, Any]]:
    r""":func:`in_this_change` against the checkpoint's register. None
    authorizes when that register cannot be read, which the register rule
    fails, or is absent outside the bootstrap, which fails the register
    rule here (T1.29 part 1's review): a checkpoint missing its register
    would otherwise make every approved entry new. At the bootstrap, a
    checkpoint before the ratchet, no register is held, so every entry
    counts."""
    try:
        text = checkpoint.read(REGISTER)
        if text is None:
            if bootstrap:
                return in_this_change(entries, None)
            report.fail(
                'register',
                'the checkpoint has no register: no entry authorizes a case '
                'or base change against it',
            )
            return []
        theirs = _yaml(text, REGISTER)
    except Unreadable:
        return []
    return in_this_change(entries, theirs)


def in_this_change(
    entries: list[dict[str, Any]], checkpoint_register: Any
) -> list[dict[str, Any]]:
    r"""The approved ``entries`` this change adds or changes: those the
    checkpoint's register does not hold as they are now, by ID (an entry
    approved here that was proposed there is changed). ``None``, a
    checkpoint with no register, holds none, so every entry counts: only
    :func:`_this_change` passes it, and only at the bootstrap."""
    before: dict[Any, Any] = {}
    if isinstance(checkpoint_register, dict) and isinstance(
        checkpoint_register.get('entries'), list
    ):
        before = {
            entry.get('id'): entry
            for entry in checkpoint_register['entries']
            if isinstance(entry, dict)
        }
    return [entry for entry in entries if before.get(entry.get('id')) != entry]


def authorized(
    entries: list[dict[str, Any]],
    case: str,
    runner: str | None,
    old: str | None,
    new: str | None,
    owner: bool = False,
) -> bool:
    r"""An approved entry states this case's change, by the owner where
    ``owner``; a None part matches any."""
    for entry in entries:
        if owner and entry.get('approved_by') != 'owner':
            continue
        for item in entry.get('case_outcomes') or []:
            if (
                isinstance(item, dict)
                and item.get('case') == case
                and item.get('cell') == CELL
                and runner in (None, item.get('runner'))
                and old in (None, item.get('old'))
                and new in (None, item.get('new'))
                and PANIC not in str(item.get('new'))
            ):
                return True
    return False


def outcome_projection(
    text: Any, like: dict[str, Any]
) -> dict[str, Any] | None:
    r"""An entry's outcome text as a projection shaped as ``like``, or None
    for one that is no outcome (``none``, ``removed``, ``diagnostic``) or
    is ``INTERNAL_ERROR``."""
    parts = register_module().parse_outcome(str(text))
    if parts is None or parts['code'] == PANIC:
        return None
    found = {
        'outcome.result': parts['result'],
        'outcome.code': None if parts['code'] in (None, '-') else parts['code'],
        'outcome.stage': parts['stage'],
        'outcome.error': parts['error'],
    }
    return {name: found.get(name, like[name]) for name in like}


def moved(
    entries: list[dict[str, Any]],
    case: str,
    runner: str,
    old: str,
    want: dict[str, Any],
) -> dict[str, Any] | None:
    r"""The new outcome of the first approved entry stating the exact
    transition of ``case`` on ``runner`` from ``old`` (``none`` where no
    baseline holds it), shaped as ``want``, the outcome it moves from: an
    owner's entry for a new refusal, never ``INTERNAL_ERROR``; or None."""
    for entry in entries:
        for item in entry.get('case_outcomes') or []:
            if not (
                isinstance(item, dict)
                and item.get('case') == case
                and item.get('cell') == CELL
                and item.get('runner') == runner
                and item.get('old') == old
            ):
                continue
            new = outcome_projection(item.get('new'), want)
            refusal = (
                new is not None
                and new.get('outcome.result') == 'refusal'
                and want.get('outcome.result') == 'success'
            )
            if new is not None and (
                not refusal or entry.get('approved_by') == 'owner'
            ):
                return new
    return None


def effective(
    held: dict[str, dict[str, dict[str, Any]]],
    entries: list[dict[str, Any]],
    reference: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    r"""Each baselined case's effective expected projection, by case and
    runner: the checkpoint's baseline, or, where an approved entry states
    the exact transition from it (case, cell, runner and old outcome), its
    new outcome, an owner's entry for a new refusal and never
    ``INTERNAL_ERROR``; and on a first admission, a runner no baseline holds
    the case for, the reference's outcome (``reference``, the committed
    python-ref lines) moved the same way by an entry from ``none`` (E110),
    with no expectation where none does, so the reference holds it. The
    reference plus approved deviations (design, "Reference observations
    and the gate"), for the comparator's outcome rule and for
    :func:`accept`."""
    found: dict[str, dict[str, dict[str, Any]]] = {}
    for runner, by_case in held.items():
        for case, line in by_case.items():
            want = dict(line['projection'])
            found.setdefault(case, {})[runner] = (
                moved(entries, case, runner, outcome_text(want), want) or want
            )
    for entry in entries:
        for item in entry.get('case_outcomes') or []:
            if not isinstance(item, dict) or item.get('old') != 'none':
                continue
            case, runner = item.get('case'), item.get('runner')
            if not isinstance(case, str) or not isinstance(runner, str):
                raise Unreadable(
                    f'{REGISTER}: {entry.get("id")}: a case outcome whose '
                    'case or runner is not a name'
                )
            ref = reference.get(case)
            if (
                ref is None
                or runner not in RUNNERS[1:]
                or case in held.get(runner, {})
            ):
                continue
            want = projection(ref, list(OUTCOME_FIELDS))
            new = moved(entries, case, runner, 'none', want)
            if new is not None:
                found.setdefault(case, {})[runner] = new
    return found


#: The key of the comparator's ``--expect`` that names, by case, the
#: runners an approved entry moved past the reference's stop
#: (:func:`moves`; E112); no case id starts with ``@``.
MOVED = '@moved'
#: The public calls in order, as an outcome's stage names them.
STAGES = ('from_source', 'materialize', 'predict')
#: The outcome fields a runner does not observe (``availability.yaml``):
#: an entry's old outcome is not held to the reference's there.
UNOBSERVED = {'rust-host': ('outcome.stage', 'outcome.error')}


def _stated(text: Any, like: dict[str, Any]) -> dict[str, Any] | None:
    r"""An entry's outcome text as a projection shaped as ``like``,
    ``INTERNAL_ERROR`` included (the reference's own outcome may be one);
    None for one that is no outcome."""
    parts = register_module().parse_outcome(str(text))
    if parts is None:
        return None
    found = {
        'outcome.result': parts['result'],
        'outcome.code': None if parts['code'] in (None, '-') else parts['code'],
        'outcome.stage': parts['stage'],
        'outcome.error': parts['error'],
    }
    return {name: found.get(name, like[name]) for name in like}


def _later(new: dict[str, Any], stop: dict[str, Any]) -> bool:
    r"""Whether ``new`` goes past the reference's refusal ``stop``: a
    success (past every stage; ``rust-host`` observes none), or a refusal
    at a later stage."""
    if new.get('outcome.result') == 'success':
        return True
    stage = new.get('outcome.stage')
    return (
        new.get('outcome.result') == 'refusal'
        and stage in STAGES
        and STAGES.index(stage) > STAGES.index(stop['outcome.stage'])
    )


def approved_move(
    entries: list[dict[str, Any]],
    case: str,
    runner: str,
    want: dict[str, Any],
    stop: dict[str, Any],
) -> bool:
    r"""Whether one of ``entries`` (approved) moves ``case`` on ``runner``
    in the reference's cell away from the reference's refusal ``stop``
    to ``want``, the runner's effective expectation: its ``new`` is
    ``want`` and past ``stop`` (:func:`_later`), and its ``old`` is
    ``none`` (a first admission) or the reference's own outcome, on the
    fields the runner observes. ``INTERNAL_ERROR`` is never a ``new``
    (``outcome_projection`` reads it as none)."""
    for entry in entries:
        for item in entry.get('case_outcomes') or []:
            if not (
                isinstance(item, dict)
                and item.get('case') == case
                and item.get('cell') == CELL
                and item.get('runner') == runner
            ):
                continue
            new = outcome_projection(item.get('new'), want)
            if new != want or not _later(want, stop):
                continue
            if item.get('old') == 'none':
                return True
            old = _stated(item.get('old'), stop)
            if old is not None and all(
                old[name] == stop[name]
                for name in stop
                if name not in UNOBSERVED.get(runner, ())
            ):
                return True
    return False


def moves(
    projected: dict[str, dict[str, dict[str, Any]]],
    entries: list[dict[str, Any]],
    this_change: list[dict[str, Any]],
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    ordered: dict[str, Any] | None,
) -> dict[str, list[str]]:
    r"""By case, the runners an approved entry moved past the reference's
    stop (E112, the orchestrator's rule): the checkpoint's reference line
    (``before``) refuses at a stage, and an entry moves the runner from
    there to its effective expectation (:func:`approved_move`); where the
    case's reference line changes in this change (``after``, in semantic
    projection), the new line's refusal and outcome are the ones judged
    (fix 2, the comparator relaxing from it) and only the entries this
    change adds or changes count (``this_change``); a case with no line at
    either side relaxes nothing. A move made now and one held since in a
    baseline are held to the same rule."""
    found: dict[str, list[str]] = {}
    for case, by_runner in sorted(projected.items()):
        line, now = before.get(case), after.get(case)
        if line is None or now is None:
            continue
        changed = semantic(line, ordered) != semantic(now, ordered)
        # Where this change re-records the line, its stop and outcome are
        # the new line's, the one the comparator relaxes from (fix 2).
        stop = projection(now if changed else line, list(OUTCOME_FIELDS))
        if (
            stop['outcome.result'] != 'refusal'
            or stop['outcome.stage'] not in STAGES
        ):
            continue
        counted = this_change if changed else entries
        runners = sorted(
            runner
            for runner, want in by_runner.items()
            if approved_move(counted, case, runner, want, stop)
        )
        if runners:
            found[case] = runners
    return found


def outcomes(
    projected: dict[str, dict[str, dict[str, Any]]],
    moved_past: dict[str, list[str]] | None = None,
) -> Any:
    r"""Effective projections as the comparator's ``--expect`` reads them:
    outcome fields by their names in an observation's ``outcome``; and
    under :data:`MOVED` the runners, by case, an approved entry moved past
    the reference's stop (:func:`moves`)."""
    found: dict[str, Any] = {
        case: {
            runner: {
                name.removeprefix('outcome.'): value
                for name, value in want.items()
                if name.startswith('outcome.')
            }
            for runner, want in by_runner.items()
        }
        for case, by_runner in projected.items()
    }
    found[MOVED] = moved_past or {}
    return found


def effective_document(
    checkpoint: Tree,
    candidate: Tree,
    entries: list[dict[str, Any]],
    this_change: list[dict[str, Any]],
) -> Any:
    r"""What ``--effective-out`` writes: :func:`effective`'s expectations
    and :func:`moves`' runners, by :func:`outcomes`."""
    reference = reference_lines(candidate)
    projected = effective(baselines(checkpoint), entries, reference)
    policy = _yaml(checkpoint.read(POLICY), POLICY) or {}
    ordered = policy.get('ordered') if isinstance(policy, dict) else None
    return outcomes(
        projected,
        moves(
            projected,
            entries,
            this_change,
            reference_lines(checkpoint),
            reference,
            ordered,
        ),
    )


def proposal(case: str, runner: str, old: str, new: str) -> str:
    r"""The ``case_outcomes`` item an entry would need, as generated text."""
    return (
        f'{{case: {case}, cell: {CELL}, runner: {runner}, '
        f"old: '{old}', new: '{new}', summary: ...}}"
    )


# --- the rules -----------------------------------------------------------------


@dataclass
class Report:
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def fail(self, rule: str, text: str) -> None:
        self.problems.append(f'{rule}: {text}')

    def note(self, rule: str, text: str) -> None:
        self.notes.append(f'{rule}: {text}')


def patterns(text: str | None) -> list[tuple[bool, re.Pattern[str]]]:
    r"""The protected-path patterns, each with whether it takes paths out."""
    found = []
    for raw in (text or '').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        negate = line.startswith('!')
        pattern = line[1:] if negate else line
        regex = ''
        index = 0
        while index < len(pattern):
            if pattern.startswith('**', index):
                regex += '.*'
                index += 2
            elif pattern[index] == '*':
                regex += '[^/]*'
                index += 1
            else:
                regex += re.escape(pattern[index])
                index += 1
        # DOTALL: ``**`` spans a name holding a newline too.
        found.append((negate, re.compile(regex, re.DOTALL)))
    return found


def protects(rules: list[tuple[bool, re.Pattern[str]]], path: str) -> bool:
    hit = False
    for negate, regex in rules:
        if regex.fullmatch(path):
            hit = not negate
    return hit


def named(path: str, text: str) -> bool:
    r"""``text`` names ``path``: the path itself, or a directory holding it
    written with a trailing ``/`` or ``/**``."""
    # Word characters past ASCII too, as a path may hold them (T1.18).
    for token in re.findall(r'[\w.*/-]+', text):
        token = token.rstrip('.')
        if token == path:
            return True
        for suffix in ('/**', '/'):
            if token.endswith(suffix) and path.startswith(
                token[: -len(suffix)] + '/'
            ):
                return True
    return False


def check_protected(
    report: Report,
    checkpoint: Tree,
    candidate: Tree,
    changed: list[str],
    trailer: str | None,
    bootstrap: bool,
) -> None:
    text = checkpoint.read(PROTECTED)
    if text is None and bootstrap:
        text = candidate.read(PROTECTED)
        report.note('protected', "bootstrap: the candidate's list is read")
    rules = patterns(text)
    if not rules:
        report.fail('protected', f'no {PROTECTED} at the checkpoint')
        return
    touched = [path for path in changed if protects(rules, path)]
    if trailer is None:
        for path in touched:
            report.note(
                'protected', f'changed, for the landing to name: {path}'
            )
        return
    for path in touched:
        if not named(path, trailer):
            report.fail(
                'protected',
                f'{path}: changed, and the Reviewed-by and Protected-paths '
                'trailers do not name it',
            )


#: A full commit SHA in backticks, as a reference approval names it.
APPROVED_SHA = re.compile(r'`([0-9a-f]{40})`')
#: The word a reference approval's Decision cell holds.
REFERENCE_WORD = re.compile(r'\breference\b', re.IGNORECASE)
#: ``reference-env.txt``'s line naming the reference commit.
ENV_LINE = re.compile(r'(?m)^reference: (.*)$')
#: Where a citation of the reference SHA is history, never moved with it.
HISTORY = ('docs/', f'{REFERENCE}/')


def approved_references(decisions: str) -> dict[str, str]:
    r"""The new reference commits the owner approved, each with its row's
    ID (T2.0e): a row of a phase's decisions file whose status is the
    owner's approval (:func:`owner_approved`) and whose Decision cell, its
    second, holds the word "reference" and exactly one full SHA in backticks
    (the form the Phase 2 decisions file's header states; KO2-1 begins
    ``Reference: `<40 hex>```). An abbreviated SHA, a row naming two, or any
    other status approves nothing."""
    found: dict[str, str] = {}
    for ident, (line, status) in (
        capture_check().decision_rows(decisions).items()
    ):
        cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
        if len(cells) < 3 or not owner_approved(status):
            continue
        shas = set(APPROVED_SHA.findall(cells[1]))
        if len(shas) == 1 and REFERENCE_WORD.search(cells[1]):
            found.setdefault(shas.pop(), ident)
    return found


def citing(tree: Tree, sha: str) -> list[str]:
    r"""The files of ``tree`` that cite ``sha`` in full, but those under
    :data:`HISTORY`."""
    done = subprocess.run(
        ['git', '-C', str(tree.repo), 'grep', '-l', '-z', '-F', '-e', sha,
         tree.sha, '--', '.', *(f':(exclude){p}' for p in HISTORY)],
        capture_output=True,
        encoding='utf-8',
        check=False,
    )  # fmt: skip
    if done.returncode not in (0, 1):
        raise Unreadable(f'git grep: {done.stderr.strip()}')
    prefix = f'{tree.sha}:'
    return sorted(
        name.removeprefix(prefix) for name in done.stdout.split('\0') if name
    )


def listed(names: list[str], most: int = 10) -> str:
    return ', '.join(names[:most]) + (
        f' and {len(names) - most} more' if len(names) > most else ''
    )


def _citations(value: Any, sha: str, at_path: str = '') -> list[str]:
    r"""The paths of ``value`` (a JSON document) whose string holds
    ``sha``, dotted, a list's items by index (``reference``,
    ``recorded.reference``, ``identity.reference.commit.sha``)."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found += _citations(item, sha, f'{at_path}.{key}'.lstrip('.'))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found += _citations(item, sha, f'{at_path}[{index}]')
    elif isinstance(value, str) and sha in value:
        found.append(at_path)
    return found


def _at_path(value: Any, dotted: str) -> Any:
    r"""The value at a path :func:`_citations` names, or a sentinel
    :class:`KeyError` instance when there is none."""
    for part in re.findall(r'[^.\[\]]+|\[\d+\]', dotted):
        if part.startswith('['):
            index = int(part[1:-1])
            if not isinstance(value, list) or index >= len(value):
                return KeyError(dotted)
            value = value[index]
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return KeyError(dotted)
    return value


def _documents(text: str) -> list[Any] | None:
    r"""A JSON file's document, or a JSON-lines file's documents (each
    line its own), or None when it is neither."""
    try:
        return [json.loads(text)]
    except (ValueError, RecursionError):
        pass
    try:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    except (ValueError, RecursionError):
        return None


def check_citation(
    report: Report,
    name: str,
    before: str,
    after: str | None,
    old: str,
    new: str,
) -> None:
    r"""A file that cited the recorded reference ``old`` at the checkpoint
    moves with it: it is still there, and each field that held ``old``
    (by its path, per document of a JSON-lines file) holds ``new`` in its
    place: ``new`` itself where it was ``old`` itself, else a text holding
    ``new`` and not ``old``. A file that is no JSON is held as text."""
    rule = 'reference'
    if after is None:
        report.fail(
            rule,
            f'{name}: removed; it cited the recorded reference {old[:12]}, and '
            f'every such file moves to the new one {new} (the reference SHA '
            'consistency tests read it)',
        )
        return
    then, now = _documents(before), _documents(after)
    if then is None or now is None:
        if old in after or new not in after:
            report.fail(rule, f'{name}: does not cite the new reference {new}')
        return
    lines = len(then) > 1 or len(now) > 1
    # A JSON-lines file's fields are those of its first line citing the
    # reference, judged on every line it has now.
    paths = next(
        (
            (document, found)
            for document in then
            if (found := _citations(document, old))
        ),
        None,
    )
    if paths is None:
        return
    document, found = paths
    for path in found:
        exact = _at_path(document, path) == old
        for index, item in enumerate(now if lines else now[:1]):
            value = _at_path(item, path)
            if (
                value == new
                if exact
                else isinstance(value, str)
                and new in value
                and old not in value
            ):
                continue
            where = f'{name}:{index + 1}' if lines else name
            report.fail(
                rule,
                f'{where}: {path or "the document"} '
                + (
                    'is missing'
                    if isinstance(value, KeyError)
                    else 'does not name the new reference'
                )
                + f' {new} (it named {old[:12]})',
            )


def _differing(old: Any, new: Any, at_path: str = '') -> list[str]:
    r"""The paths at which two semantic projections differ, by name only
    (values are never printed)."""
    if isinstance(old, dict) and isinstance(new, dict):
        found: list[str] = []
        for key in sorted(set(old) | set(new), key=str):
            found += _differing(
                old.get(key), new.get(key), f'{at_path}.{key}'.lstrip('.')
            )
        return found
    return [] if old == new else [at_path or 'the line']


def check_new_reference(
    report: Report,
    checkpoint: Tree,
    candidate: Tree,
    added: list[str],
    removed: set[str],
    decisions: str,
) -> None:
    r"""A candidate that adds a reference file: a new reference commit,
    which only the owner's approval at the checkpoint admits (T2.0e)."""
    before = checkpoint.files(REFERENCE)
    after = candidate.files(REFERENCE)
    olds = sorted(Path(path).stem for path in before)
    approved = approved_references(decisions)
    refused = [
        path
        for path in added
        if Path(path).stem not in approved or Path(path).stem in olds
    ]
    for path in refused:
        report.fail(
            'reference',
            f"a new reference commit replaces the recorded one: the owner's; "
            f"{path} is named for a commit no row of the checkpoint's "
            'decisions approved by the owner names as the reference '
            '(Reference: `<its full SHA>`)',
        )
    if refused:
        return
    misnamed = [
        path for path in added if path != f'{REFERENCE}/{Path(path).stem}.jsonl'
    ]
    for path in misnamed:
        report.fail(
            'reference',
            f'{path}: a new reference recording is exactly '
            f'{REFERENCE}/<the approved full SHA>.jsonl, which the gate reads',
        )
    if misnamed:
        return
    if len(added) != 1:
        report.fail(
            'reference',
            f'{len(added)} new reference files ({listed(added)}): a new '
            'reference commit is one recording',
        )
        return
    (path,) = added
    new = Path(path).stem
    report.note(
        'reference',
        f'the reference moves from {listed(olds)} to {new}, approved by '
        f'{approved[new]} at the checkpoint; each case is compared with its '
        'recorded line below, and a case recorded differently is listed, '
        'with the fields that differ, for the reviewers',
    )
    for left in after:
        if left != path:
            report.fail(
                'reference',
                f'{left}: left beside the new reference {path}; a new '
                'reference replaces the recorded one, one reference file '
                'remains',
            )
    env = candidate.read(ENV)
    named_env = ENV_LINE.findall(env or '')
    if named_env != [new]:
        report.fail(
            'reference',
            f'{ENV}: names {listed(named_env) or "no reference"}, not the new '
            f'reference {new}',
        )
    for old in olds:
        for name in citing(candidate, old):
            report.fail(
                'reference',
                f'{name}: still cites the recorded reference {old[:12]}; every '
                f'file citing the reference outside docs/ names the new one '
                f'{new}',
            )
        for name in citing(checkpoint, old):
            if name != ENV:
                check_citation(
                    report,
                    name,
                    checkpoint.read(name) or '',
                    candidate.read(name),
                    old,
                    new,
                )
    # Every line read and judged as written, before any is indexed: a
    # case on two lines is refused by file and line, never one of them
    # silently kept.
    lines: dict[Any, dict[str, Any]] = {}
    first: dict[Any, int] = {}
    for number, line in enumerate(jsonl(candidate.read(path), path), 1):
        case = line.get('case')
        if not isinstance(case, str) or not case:
            report.fail('reference', f'{path}:{number}: names no case')
            continue
        if at(line, 'identity.reference.commit.sha') != new:
            report.fail(
                'reference',
                f'{path}:{number}: {case}: records another reference commit '
                f"than {new}, the file's name",
            )
        if case in first:
            report.fail(
                'reference',
                f'{path}:{number}: {case}: a second line of the case (its '
                f'first is line {first[case]}); a recording holds each case '
                'once',
            )
            continue
        first[case] = number
        lines[case] = line
    policy = _yaml(checkpoint.read(POLICY), POLICY) or {}
    ordered = policy.get('ordered') if isinstance(policy, dict) else None
    recorded = reference_lines(checkpoint)
    for case, line in sorted(recorded.items(), key=lambda item: str(item[0])):
        if case not in lines:
            if case not in removed:
                report.fail(
                    'reference',
                    f'{case}: its line removed: the new recording carries '
                    'every case the recorded one did',
                )
            continue
        then, now = semantic(line, ordered), semantic(lines[case], ordered)
        if then != now:
            report.note(
                'reference',
                f'{case}: recorded differently at the new reference {new[:12]}'
                f', in {listed(_differing(then, now), 6)}: for the reviewers',
            )
    fresh = len(set(lines) - set(recorded))
    if fresh:
        report.note('reference', f'{fresh} case(s) recorded for the first time')


def check_reference(
    report: Report,
    checkpoint: Tree,
    candidate: Tree,
    entries: list[dict[str, Any]],
    removed: set[str],
    decisions: str = '',
) -> None:
    before = checkpoint.files(REFERENCE)
    if not before:
        return
    after = candidate.files(REFERENCE)
    added = [p for p in after if p not in before]
    if added:
        check_new_reference(
            report, checkpoint, candidate, added, removed, decisions
        )
        return
    policy = _yaml(checkpoint.read(POLICY), POLICY) or {}
    ordered = policy.get('ordered') if isinstance(policy, dict) else None
    for path in before:
        old = {
            line.get('case'): line
            for line in jsonl(checkpoint.read(path), path)
        }
        new = {
            line.get('case'): line for line in jsonl(candidate.read(path), path)
        }
        for case, line in old.items():
            if new.get(case) == line:
                continue
            if case not in new:
                if case not in removed:
                    report.fail('reference', f'{case}: its line removed')
            elif semantic(line, ordered) != semantic(new[case], ordered):
                if not authorized(entries, case, 'python-ref', None, None):
                    report.fail(
                        'reference',
                        f'{case}: its line changed, and no approved entry '
                        'names the case on python-ref',
                    )
            else:
                report.note('reference', f'{case}: re-recorded, identity only')


def check_cases(
    report: Report,
    checkpoint: Tree,
    candidate: Tree,
    entries: list[dict[str, Any]],
    held: dict[str, dict[str, dict[str, Any]]],
) -> tuple[set[str], set[str]]:
    r"""Removed, demoted and changed cases, and changed bases; returns the
    cases removed or demoted, and those whose removal is approved."""
    before, after = cases(checkpoint), cases(candidate)
    gone, removed = set(), set()
    for case, (old, text) in sorted(before.items()):
        if case in after:
            mine, changed = after[case][0], after[case][1] != text
            if not (
                old.get('status') == 'acceptance'
                and mine.get('status') != 'acceptance'
            ):
                if changed and not authorized(entries, case, None, None, None):
                    report.fail(
                        'cases',
                        f'{case}: its input changed, and no approved entry '
                        'names it',
                    )
                continue
            new = 'diagnostic'
        else:
            new = 'removed'
        gone.add(case)
        lines = [
            (runner, outcome_text(by_case[case]['projection']))
            for runner, by_case in held.items()
            if case in by_case
        ] or [(None, 'none')]
        missing = [
            (runner, old_text)
            for runner, old_text in lines
            if not authorized(entries, case, runner, old_text, new)
        ]
        for runner, old_text in missing:
            report.fail(
                'cases',
                f'{case}: {new}, and no approved entry states it on '
                f'{runner or "any runner"}: '
                + proposal(case, runner or '<runner>', old_text, new),
            )
        if not missing and new == 'removed':
            removed.add(case)
    old_bases = _yaml(checkpoint.read(BASES), BASES)
    new_bases = _yaml(candidate.read(BASES), BASES)
    if isinstance(old_bases, dict) and isinstance(old_bases.get('bases'), dict):
        mine = new_bases.get('bases') if isinstance(new_bases, dict) else None
        mine = mine if isinstance(mine, dict) else {}
        for name, statements in old_bases['bases'].items():
            if mine.get(name) == statements:
                continue
            users = [
                case
                for case, (each, _) in before.items()
                if (each.get('input') or {}).get('base') == name
            ]
            if not any(
                authorized(entries, case, None, None, None) for case in users
            ):
                report.fail(
                    'cases',
                    f'base {name}: changed or removed, and no approved entry '
                    'names a case built on it',
                )
    return gone, removed


def policy_relaxations(old: Any, new: Any) -> list[str]:
    r"""What the candidate's ``compare-policy.yaml`` relaxes."""
    old = old if isinstance(old, dict) else {}
    new = new if isinstance(new, dict) else {}
    found = []
    mine = new.get('rules') or {}
    for rule, cls in (old.get('rules') or {}).items():
        if rule not in mine:
            found.append(f'rule {rule} removed')
        elif cls == 'gate' and mine[rule] != 'gate':
            found.append(f'rule {rule}: gate to {mine[rule]}')
    theirs = {
        e.get('id'): e
        for e in old.get('exclusions') or []
        if isinstance(e, dict)
    }
    for item in new.get('exclusions') or []:
        key = item.get('id') if isinstance(item, dict) else None
        if key not in theirs:
            found.append(f'exclusion {key} added')
        elif theirs[key] != item:
            found.append(f'exclusion {key} changed')
    kept = new.get('ordered') or {}
    for case, stages in (old.get('ordered') or {}).items():
        for stage, positions in (stages or {}).items():
            for position in positions or []:
                if position not in ((kept.get(case) or {}).get(stage) or []):
                    found.append(f'ordered {case} {stage} {position} removed')
    reach = new.get('reach') or {}
    for case, stage in (old.get('reach') or {}).items():
        if case not in reach:
            found.append(f'reach {case} removed')
        elif reach[case] != stage:
            found.append(f'reach {case}: {stage} to {reach[case]}')
    fields = new.get('baseline') or []
    for name in old.get('baseline') or []:
        if name not in fields:
            found.append(f'baseline field {name} removed')
    return found


def availability_relaxations(old: Any, new: Any) -> list[str]:
    r"""What the candidate's ``availability.yaml`` relaxes."""
    theirs = (old or {}).get('runners') or {} if isinstance(old, dict) else {}
    mine = (new or {}).get('runners') or {} if isinstance(new, dict) else {}
    found = []
    for runner, fields in theirs.items():
        for name, value in (fields or {}).items():
            now = (mine.get(runner) or {}).get(name)
            if value == 'available' and now != 'available':
                found.append(f'{runner} {name}: available to {now or "absent"}')
    for runner, fields in mine.items():
        for name, value in (fields or {}).items():
            if value != 'available' and name not in (theirs.get(runner) or {}):
                found.append(f'{runner} {name}: added as {value}')
    return found


def relaxations(read: Any, old: Any, new: Any, path: str) -> list[str]:
    r"""``read``'s relaxations of ``path``, a file of another shape refused
    by name."""
    try:
        found: list[str] = read(old, new)
    except (AttributeError, TypeError):
        raise Unreadable(f'{path}: not a file of its format') from None
    return found


def check_relaxations(
    report: Report,
    rule: str,
    path: str,
    found: list[str],
    allowed: Any,
    decisions: str | None,
) -> None:
    rows = capture_check().decision_rows(decisions or '')
    listed = allowed if isinstance(allowed, list) else []
    name = Path(path).name
    for change in found:
        decision = next(
            (
                item.get('decision')
                for item in listed
                if isinstance(item, dict) and item.get('change') == change
            ),
            None,
        )
        row = rows.get(decision) if isinstance(decision, str) else None
        if (
            row is None
            or not owner_approved(row[1])
            or f'`{name}`' not in row[0]
        ):
            report.fail(
                rule,
                f"{change}: a relaxation, with no row of the checkpoint's "
                f'decisions.md approved by the owner and naming `{name}`',
            )
        else:
            report.note(rule, f'{change}: approved by {decision}')


def tree_entries(repo: Path, rev: str) -> dict[str, str]:
    r"""Every entry of ``rev``'s tree, by path: its mode, type and object ID
    (``git ls-tree -r -z --full-tree``), a submodule's commit included. No
    configuration changes what it lists (``diff.ignoreSubmodules``, a
    ``.gitmodules`` ``ignore``), and names are read NUL-separated, never
    git's quoting."""
    out = git(repo, 'ls-tree', '-r', '-z', '--full-tree', rev)
    found = {}
    for record in out.split('\0'):
        meta, tab, path = record.partition('\t')
        if tab:
            found[path] = meta
    return found


def changed_paths(repo: Path, before: str, after: str) -> list[str]:
    r"""The paths whose entry (presence, mode, type, object ID) differs
    between ``before``'s tree and ``after``'s (:func:`tree_entries`)."""
    old, new = tree_entries(repo, before), tree_entries(repo, after)
    return sorted(
        p for p in old.keys() | new.keys() if old.get(p) != new.get(p)
    )


def epic_head(repo: Path, head: str) -> bool:
    r"""``head`` is the epic branch's: the tip of :data:`EPIC_REFS`' first
    the repository holds, or the branch checked out is it; a run that
    judges it without ``--epic`` says rule ``epic`` did not run."""
    tip = epic_tip(repo, None)
    return (tip is not None and tip[0] == head) or (
        _checked_out(repo) == PHASES.EPIC
    )


def check_epic(
    report: Report, repo: Path, checkpoint: Tree, candidate: Tree, head: str
) -> None:
    r"""Rule ``epic`` (T2.0), run on the epic branch's head (``--epic``,
    which ``where`` prints for a push to ``rust-core``) judged against the
    merged phase's last landing L, the checkpoint: every protected path has
    the same entry in the head's tree as in L's, its presence, mode and
    object ID (:func:`tree_entries`, which no git configuration changes), the
    protected paths being those L's ``contract/protected-paths.txt``
    matches and those the head's own list matches, so a head can't shrink
    the list to hide a change. Any difference fails, naming the path.

    No commit is walked and no row approves anything: a protected change
    reaches the epic branch only through a reviewed phase landing the owner
    merges, which then is the checkpoint. (Fix 7 replaced an attribution
    of changes to commits, which missed a git edge case in each of five
    review rounds: renames, merges, conflicts, modes.)

    Threat model (as decision E56(b) states it for the L0 guards): honest
    drift. CI on ``rust-core`` runs the pushed head's own ``ci.yml`` and
    ``gate.sh``, so a push crafted to evade the rule (changing either) can;
    only the owner pushes ``rust-core``, and this rule catches an honest
    protected change no reviewed landing brought."""
    lists = (
        patterns(checkpoint.read(PROTECTED)),
        patterns(candidate.read(PROTECTED)),
    )
    for path in sorted(changed_paths(repo, checkpoint.sha, head)):
        if any(protects(rules, path) for rules in lists):
            report.fail(
                'epic',
                f'{path}: differs on the epic branch from the merged '
                f'landing {checkpoint.sha[:12]}; a protected change reaches '
                'rust-core only through a reviewed phase landing',
            )


def checkpoint_decisions(report: Report, checkpoint: Tree) -> str:
    r"""Every phase's decisions at the checkpoint, as one text
    (``capture_check.phase_decisions``, T2.0); a file missing, unreadable
    or malformed fails the ``decisions`` rule by name, and then no row
    approves anything."""
    module = capture_check()
    try:
        return module.phase_decisions(
            checkpoint.files(DECISIONS_DIR), checkpoint.read
        )
    except module.DecisionsError as error:
        report.fail('decisions', f"the checkpoint's {error}")
        return ''


def check_known_differing(
    report: Report, checkpoint: Tree, candidate: Tree
) -> None:
    def listed(tree: Tree, which: str) -> dict[str, set[str]] | None:
        r"""The file's cases and their parts, or None, the failure named:
        a file missing, without its ``cases`` list, or with an entry not
        an ID and a list of parts reads as no list, never as an empty one."""
        text = tree.read(KNOWN_DIFFERING)
        if text is None:
            report.fail(
                'known-differing', f'no {KNOWN_DIFFERING} at the {which}'
            )
            return None
        document = _json(text, KNOWN_DIFFERING)
        items = document.get('cases') if isinstance(document, dict) else None
        if not isinstance(items, list) or not all(
            isinstance(item, dict)
            and isinstance(item.get('id'), str)
            and isinstance(item.get('parts'), list)
            and all(isinstance(part, str) for part in item['parts'])
            for item in items
        ):
            report.fail(
                'known-differing',
                f"the {which}'s {KNOWN_DIFFERING} is not a list of cases, "
                'each an ID and its parts',
            )
            return None
        return {item['id']: set(item['parts']) for item in items}

    before = listed(checkpoint, 'checkpoint')
    after = listed(candidate, 'candidate')
    if before is None or after is None:
        return
    for case, parts in sorted(after.items()):
        if case not in before:
            report.fail('known-differing', f'{case}: added')
        elif not parts <= before[case]:
            report.fail('known-differing', f'{case}: parts added')
    for case in sorted(set(before) - set(after)):
        report.note('known-differing', f'{case}: removed')


def check_register(
    report: Report, checkpoint: Tree, candidate: Tree, root: Path
) -> Any:
    module = register_module()
    try:
        mine = _yaml(candidate.read(REGISTER), REGISTER)
        theirs = _yaml(checkpoint.read(REGISTER), REGISTER)
        for problem in module.problems(mine, root, theirs):
            report.fail('register', problem)
    except (Unreadable, module.runner().RunnerFailure) as error:
        report.fail('register', str(error))
        return None
    before = {
        e.get('id'): e
        for e in (theirs or {}).get('entries') or []
        if isinstance(e, dict)
    }
    for entry in sorted(
        (mine or {}).get('entries') or [],
        key=lambda e: (e.get('approved_by') != 'owner', e.get('id')),
    ):
        if before.get(entry.get('id')) != entry:
            what = 'added' if entry.get('id') not in before else 'changed'
            report.note(
                'register',
                f'{entry.get("id")} {what}: {entry.get("status")}, approved_by '
                f'{entry.get("approved_by")}',
            )
    return mine


def check_availability(
    report: Report,
    fields: Any,
    observed: dict[str, dict[str, Any]],
    corpus: dict[str, tuple[dict[str, Any], str]],
) -> None:
    runners = (
        (fields or {}).get('runners') if isinstance(fields, dict) else None
    )
    if not isinstance(runners, dict):
        report.fail('availability', f'no {AVAILABILITY}')
        return
    for runner, by_case in observed.items():
        marked = runners.get(runner)
        if not isinstance(marked, dict):
            report.fail('availability', f'{runner}: not in {AVAILABILITY}')
            continue
        for case, observation in sorted(by_case.items()):
            spec = corpus.get(case, ({}, ''))[0]
            for name, value in marked.items():
                found = at(observation, name)
                if value == 'unavailable':
                    if found is not None:
                        report.fail(
                            'availability',
                            f'{case}: {runner}: {name} recorded, and it is '
                            'unavailable',
                        )
                    continue
                if (
                    name == 'outcome.error'
                    and at(observation, 'outcome.result') != 'refusal'
                ):
                    continue
                if (
                    name == 'driver.capture'
                    and runner == 'rust-host'
                    and not any(op in (spec.get('core') or {}) for op in READS)
                ):
                    continue
                if found is None:
                    report.fail(
                        'availability',
                        f'{case}: {runner}: {name} available, and missing',
                    )


def check_baselines(
    report: Report,
    held: dict[str, dict[str, dict[str, Any]]],
    proposed: dict[str, dict[str, dict[str, Any]]],
    old_fields: list[str],
    new_fields: list[str],
    observed: dict[str, dict[str, Any]],
    corpus: dict[str, tuple[dict[str, Any], str]],
    entries: list[dict[str, Any]],
    gone: set[str],
    reference: dict[str, dict[str, Any]],
    ordered: dict[str, Any] | None,
) -> None:
    for runner, by_case in sorted(observed.items()):
        crashes = {
            case
            for case, observation in by_case.items()
            if old_client_crash(
                runner, observation, reference.get(case), ordered
            )
        }
        for case, line in sorted(held.get(runner, {}).items()):
            if case in gone:
                continue
            observation = by_case.get(case)
            if observation is None:
                report.fail(
                    'baselines',
                    f'{case}: {runner}: baselined, and not observed',
                )
                continue
            now = projection(observation, old_fields)
            if now.get('outcome.code') == PANIC and case not in crashes:
                report.fail(
                    'baselines',
                    f'{case}: {runner}: {PANIC}, a panic, which no entry accepts',
                )
                continue
            if now == line['projection']:
                continue
            old, new = outcome_text(line['projection']), outcome_text(now)
            refusal = (
                now.get('outcome.result') == 'refusal'
                and line['projection'].get('outcome.result') == 'success'
            )
            if not authorized(entries, case, runner, old, new, owner=refusal):
                report.fail(
                    'baselines',
                    f'{case}: {runner}: {old} became {new}, with no approved '
                    + ('owner ' if refusal else '')
                    + 'entry stating it: '
                    + proposal(case, runner, old, new),
                )
            else:
                report.note(
                    'baselines', f'{case}: {runner}: {old} to {new}, approved'
                )
        mine = proposed.get(runner, {})
        for case, line in sorted(mine.items()):
            observation = by_case.get(case)
            if (
                line['projection'].get('outcome.code') == PANIC
                and case not in crashes
            ):
                report.fail(
                    'baselines',
                    f'{case}: {runner}: the proposed baseline is {PANIC}, a '
                    'panic, which no entry accepts',
                )
                continue
            if (
                observation is None
                or projection(observation, new_fields) != line['projection']
            ):
                report.fail(
                    'baselines',
                    f'{case}: {runner}: the proposed baseline is not the '
                    'observation (ratchet.py accept writes it)',
                )
        for case, (spec, _) in sorted(corpus.items()):
            if (
                spec.get('status') == 'acceptance'
                and case in by_case
                and case not in mine
            ):
                report.fail(
                    'baselines',
                    f'{case}: {runner}: an acceptance case with no proposed baseline',
                )
        for case in sorted(crashes & (set(mine) | set(held.get(runner, {})))):
            report.note(
                'baselines',
                f"{case}: {runner}: {PANIC}, the old client's crash, held as "
                'the reference recorded it',
            )


def check(
    repo: Path,
    checkpoint_sha: str,
    observation_files: list[Path],
    protected_text: str | None = None,
    bootstrap: bool = False,
    effective_out: Path | None = None,
    epic: bool = False,
) -> Report:
    r"""Every rule's problems; ``effective_out`` receives the effective
    expectations (:func:`effective`) for the comparator."""
    report = Report()
    checkpoint = Tree(repo, checkpoint_sha)
    head = git(repo, 'rev-parse', 'HEAD^{commit}').strip()
    candidate = Tree(repo, head)
    if git(repo, 'status', '--porcelain', '--untracked-files=no').strip():
        report.fail(
            'tree',
            'the worktree has uncommitted changes: the gate judges a commit',
        )
        return report
    trailer = protected_text
    if is_landing(repo, head):
        found = trailers(repo, head)
        if not found.get('Reviewed-by'):
            report.fail(
                'protected', 'the landing commit has no Reviewed-by trailer'
            )
        trailer = ' '.join(
            found.get(key, '') for key in ('Reviewed-by', 'Protected-paths')
        )
    # Without rename detection, a move lists both its paths, so a protected
    # file moved out of (or onto) a protected path is seen at both; and
    # NUL-separated, so a path past ASCII is its name, not git's quoting.
    changed = changed_paths(repo, checkpoint_sha, head)
    try:
        mine = check_register(report, checkpoint, candidate, repo)
        entries = approved(mine)
        held = baselines(checkpoint)
        if effective_out is not None:
            effective_out.write_text(
                json.dumps(
                    effective_document(
                        checkpoint,
                        candidate,
                        entries,
                        _this_change(Report(), entries, checkpoint, bootstrap),
                    ),
                    sort_keys=True,
                ),
                encoding='utf-8',
            )
        # A case file or base changes only under an entry of this change
        # (T1.29): an approved entry authorizes the landing that adds it.
        gone, removed = check_cases(
            report,
            checkpoint,
            candidate,
            _this_change(report, entries, checkpoint, bootstrap),
            held,
        )
        check_protected(
            report, checkpoint, candidate, changed, trailer, bootstrap
        )
        decisions = checkpoint_decisions(report, checkpoint)
        check_reference(
            report, checkpoint, candidate, entries, removed, decisions
        )
        if epic:
            found_merge = epic_merge(repo, head)
            if (
                head == checkpoint.sha
                or found_merge is None
                or (found_merge[1] != checkpoint.sha)
            ):
                report.fail(
                    'epic',
                    f'--epic: the checkpoint {checkpoint.sha[:12]} is not the '
                    "merged phase's last landing the head holds ("
                    + (found_merge[1][:12] if found_merge else 'none')
                    + ')',
                )
            else:
                check_epic(report, repo, checkpoint, candidate, head)
        elif epic_head(repo, head):
            report.note(
                'epic',
                "not run: this is the epic branch's head; judge it as a "
                'push to rust-core is judged, with --epic (gate.sh $(ratchet.py '
                'where) on the push, or --base <the merged landing> --epic)',
            )
        old_policy = _yaml(checkpoint.read(POLICY), POLICY)
        new_policy = _yaml(candidate.read(POLICY), POLICY)
        check_relaxations(
            report,
            'policy',
            POLICY,
            relaxations(policy_relaxations, old_policy, new_policy, POLICY),
            (new_policy or {}).get('relaxations')
            if isinstance(new_policy, dict)
            else None,
            decisions,
        )
        old_marks = _yaml(checkpoint.read(AVAILABILITY), AVAILABILITY)
        new_marks = _yaml(candidate.read(AVAILABILITY), AVAILABILITY)
        if old_marks is None and bootstrap:
            report.note('availability', 'bootstrap: the file is introduced')
        else:
            check_relaxations(
                report,
                'availability',
                AVAILABILITY,
                relaxations(
                    availability_relaxations, old_marks, new_marks, AVAILABILITY
                ),
                (new_marks or {}).get('relaxations')
                if isinstance(new_marks, dict)
                else None,
                decisions,
            )
        check_known_differing(report, checkpoint, candidate)
        if observation_files:
            observed = observations(observation_files)
            corpus = cases(candidate)
            check_availability(
                report,
                old_marks if old_marks is not None else new_marks,
                observed,
                corpus,
            )

            def fields(policy: Any) -> list[str]:
                found = (
                    policy.get('baseline') if isinstance(policy, dict) else None
                )
                return (
                    [str(f) for f in found] if isinstance(found, list) else []
                )

            for runner in sorted(observed):
                if runner not in held and not bootstrap:
                    report.fail(
                        'baselines',
                        f'the checkpoint holds no baselines of {runner}',
                    )
            check_baselines(
                report,
                held,
                baselines(candidate),
                fields(old_policy) or fields(new_policy),
                fields(new_policy),
                observed,
                corpus,
                entries,
                gone,
                reference_lines(checkpoint),
                (new_policy or {}).get('ordered')
                if isinstance(new_policy, dict)
                else None,
            )
    except Unreadable as error:
        report.fail('read', str(error))
    except (AttributeError, TypeError, KeyError) as error:
        # A file of another shape than its format's, whichever rule met it.
        report.fail(
            'read',
            f'a file not of its format ({type(error).__name__}: {error})',
        )
    return report


# --- accept --------------------------------------------------------------------


def accept(
    repo: Path, observation_files: list[Path], checkpoint: str | None = None
) -> dict[str, list[dict[str, Any]]]:
    r"""The proposed baselines: per runner given, a line for each case the
    runner observes and meets: its effective expectation where
    :func:`effective` gives one (``checkpoint``'s baseline or, on a first
    admission, the reference's outcome, either moved by the candidate's
    approved entries), otherwise the committed reference's projection,
    fields a runner cannot observe left out of that comparison. Each runner
    is judged alone, so one off its expectation gets no line and the
    others keep theirs; ``check`` then names the acceptance case it lacks.
    Without a checkpoint (the first baselines) only reference equality
    admits a case. A line that would be an ``INTERNAL_ERROR`` other than
    the old client's crash (:func:`old_client_crash`), which ``check``
    refuses, is left out."""
    tree = Tree(repo, git(repo, 'rev-parse', 'HEAD^{commit}').strip())
    reference = reference_lines(tree)
    recorded = (
        reference_lines(Tree(repo, checkpoint)) if checkpoint else reference
    )
    expected = (
        effective(
            baselines(Tree(repo, checkpoint)),
            approved(_yaml(tree.read(REGISTER), REGISTER)),
            reference,
        )
        if checkpoint
        else {}
    )
    policy = _yaml(tree.read(POLICY), POLICY)
    fields = [str(f) for f in policy['baseline']]
    marks = (_yaml(tree.read(AVAILABILITY), AVAILABILITY) or {})['runners']
    observed = observations(observation_files)
    found: dict[str, list[dict[str, Any]]] = {runner: [] for runner in observed}
    for case in sorted(cases(tree)):
        ref = reference.get(case)
        for runner, by_case in observed.items():
            observation = by_case.get(case)
            if observation is None:
                continue
            want = expected.get(case, {}).get(runner)
            if want is None and ref is not None:
                want = projection(ref, fields)
            # What the runner cannot observe is left out (availability.yaml),
            # so one entry text serves every runner.
            meets = want is not None and all(
                at(observation, name) == want.get(name)
                for name in fields
                if (marks.get(runner) or {}).get(name) != 'unavailable'
            )
            if not meets or (
                at(observation, 'outcome.code') == PANIC
                and not old_client_crash(
                    runner,
                    observation,
                    recorded.get(case),
                    policy.get('ordered'),
                )
            ):
                continue
            projected = projection(observation, fields)
            found[runner].append(
                {
                    'case': case,
                    'runner': runner,
                    'cell': CELL,
                    'projection': projected,
                    'digest': digest(projected),
                }
            )
    return found


def write_effective(repo: Path, checkpoint: str, out: Path) -> list[str]:
    r"""Writes :func:`effective`'s expectations for the candidate, ``repo``'s
    ``HEAD``, to ``out`` as :func:`check` does; returns the register's
    problems, writing nothing when there are any."""
    candidate = Tree(repo, git(repo, 'rev-parse', 'HEAD^{commit}').strip())
    before = Tree(repo, checkpoint)
    module = register_module()
    mine = _yaml(candidate.read(REGISTER), REGISTER)
    try:
        problems = list(
            module.problems(mine, repo, _yaml(before.read(REGISTER), REGISTER))
        )
    except module.runner().RunnerFailure as error:
        problems = [str(error)]
    if problems:
        return problems
    entries = approved(mine)
    out.write_text(
        json.dumps(
            effective_document(
                before,
                candidate,
                entries,
                _this_change(Report(), entries, before, False),
            ),
            sort_keys=True,
        ),
        encoding='utf-8',
    )
    return []


def ci_base(environ: dict[str, str], repo: Path = Path('.')) -> str:
    r"""The gate's base for a CI run, from GitHub's environment: on a push
    to a phase branch (``rust-core-phase-N``, any N), ``--phase-push`` and
    the push's previous tip (from the event file); on a push to a lane or
    another branch under one, ``--base`` that phase branch's remote tip; on
    a push to the epic branch, ``--base`` the merged phase's last landing
    ``repo``'s ``HEAD`` holds and ``--epic`` (:func:`epic_merge`; T2.0), so the epic
    branch is judged like a moved landing of that phase's head, and with
    none it fails; with no CI environment (a local replay), ``--base`` the
    remote tip of the phase branch ``phases.phase_branch`` names for the
    branch checked out; elsewhere nothing, since there is no ratchet to
    run."""
    ref = environ.get('GITHUB_REF_NAME')
    if ref is None:
        try:
            phase = PHASES.phase_branch(
                branch=_checked_out(repo), environ=environ
            )
        except PHASES.PhaseError as error:
            raise Unreadable(str(error)) from None
        return f'--base origin/{phase}'
    if environ.get('GITHUB_EVENT_NAME') != 'push':
        return ''
    if ref == PHASES.EPIC:
        head = git(repo, 'rev-parse', 'HEAD^{commit}').strip()
        landing = epic_checkpoint(repo, head)
        if landing is None:
            raise Unreadable(
                f'{PHASES.EPIC} holds no merged phase: no merge on its '
                'first-parent history has a landing commit as second parent'
            )
        return f'--base {landing} --epic'
    phase = PHASES.phase_of(ref)
    if phase is None:
        return ''
    if ref == phase:
        event = _json(
            Path(environ['GITHUB_EVENT_PATH']).read_text(encoding='utf-8'),
            'the event',
        )
        before = event.get('before') if isinstance(event, dict) else None
        if not isinstance(before, str) or not re.fullmatch(
            r'[0-9a-f]{40}', before
        ):
            raise Unreadable('the push event names no previous tip')
        return f'--phase-push {before}'
    return f'--base origin/{phase}'


def _checked_out(repo: Path) -> str | None:
    try:
        return (
            git(repo, 'symbolic-ref', '--short', '-q', 'HEAD').strip() or None
        )
    except Unreadable:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    point = commands.add_parser('checkpoint')
    point.add_argument('--repo', type=Path, default=Path('.'))
    where = point.add_mutually_exclusive_group(required=True)
    where.add_argument('--base')
    where.add_argument('--phase-push')
    point.add_argument('--epic-ref')
    point.add_argument('--epic', action='store_true')
    judge = commands.add_parser('check')
    judge.add_argument('--repo', type=Path, default=Path('.'))
    judge.add_argument('--checkpoint', required=True)
    judge.add_argument('--protected-paths')
    judge.add_argument('--bootstrap', action='store_true')
    judge.add_argument('--observations', type=Path, action='append', default=[])
    judge.add_argument('--effective-out', type=Path)
    judge.add_argument('--epic', action='store_true')
    spot = commands.add_parser('where')
    spot.add_argument('--repo', type=Path, default=Path('.'))
    expect = commands.add_parser('effective')
    expect.add_argument('--repo', type=Path, default=Path('.'))
    among = expect.add_mutually_exclusive_group(required=True)
    among.add_argument('--base')
    among.add_argument('--phase-push')
    expect.add_argument('--epic-ref')
    expect.add_argument('--epic', action='store_true')
    expect.add_argument('--out', type=Path, required=True)
    take = commands.add_parser('accept')
    take.add_argument('--repo', type=Path, default=Path('.'))
    take.add_argument('--checkpoint')
    take.add_argument(
        '--observations', type=Path, action='append', required=True
    )
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == 'where':
            print(ci_base(dict(os.environ), arguments.repo))
            return 0
        if arguments.command == 'checkpoint':
            found, problems = resolve(
                arguments.repo,
                arguments.base,
                arguments.phase_push,
                arguments.epic_ref,
                epic_mode=arguments.epic,
            )
            for problem in problems:
                print(f'checkpoint: {problem}', file=sys.stderr)
            if found is None:
                return 1
            for note in found.notes:
                print(f'checkpoint: {note}', file=sys.stderr)
            print(
                found.sha,
                found.base,
                'bootstrap' if found.bootstrap else 'harness',
            )
            return 0
        if arguments.command == 'effective':
            found, problems = resolve(
                arguments.repo,
                arguments.base,
                arguments.phase_push,
                arguments.epic_ref,
                epic_mode=arguments.epic,
            )
            if found is not None:
                problems = write_effective(
                    arguments.repo, found.sha, arguments.out
                )
            for problem in problems:
                print(f'effective: {problem}', file=sys.stderr)
            return 1 if problems else 0
        if arguments.command == 'accept':
            for runner, lines in accept(
                arguments.repo, arguments.observations, arguments.checkpoint
            ).items():
                path = arguments.repo / BASELINES / CELL / f'{runner}.jsonl'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    ''.join(
                        json.dumps(line, sort_keys=True) + '\n'
                        for line in lines
                    ),
                    encoding='utf-8',
                )
                print(f'accept: {len(lines)} baselines of {runner} in {path}')
            return 0
        report = check(
            arguments.repo,
            arguments.checkpoint,
            arguments.observations,
            arguments.protected_paths,
            arguments.bootstrap,
            arguments.effective_out,
            arguments.epic,
        )
    except Unreadable as error:
        print(f'ratchet: {error}\nratchet: FAIL')
        return 1
    for note in report.notes:
        print(note)
    for problem in report.problems:
        print(f'FAILS {problem}')
    print(f'ratchet: {"FAIL" if report.problems else "ok"}')
    return 1 if report.problems else 0


if __name__ == '__main__':
    sys.exit(main())
