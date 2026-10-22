#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lands a reviewed package on the phase branch, so no one lands by hand.

Decision E43 (b) and the design's "One gate run per work package": the
only commit after approval is an empty landing commit carrying the
trailers (E25), and the phase branch fast-forwards to it at once when its
tree is the candidate's and the candidate's own exact-SHA CI run
concluded success. CI skips the landing commit's own runs (decision E131):
its tree was tested on the candidate's exact SHA, and ``land.py verify``
stands on that run.

    land.py --candidate SHA --topic-branch NAME --review FILE --review FILE \\
        --title 'T1.12: the S0 case corpus' --protected-paths TEXT \\
        [--checkpoint SHA] [--phase-branch rust-core-phase-N] \\
        [--reviewer TEXT] [--summary TEXT] [--dry-run]
        [--moved-from SHA]
    land.py verify LANDING_SHA [--phase-branch rust-core-phase-N]

The phase branch is ``--phase-branch``, else ``SDM_PHASE_BRANCH``, else
(landing) the phase the topic branch is under, else the phase in progress
(``phases.py``, T2.0); ``verify`` of an earlier phase's landing names its
phase. Before it changes anything it refuses, naming the reason, unless:

- the phase branch is a phase branch ``rust-core-phase-N`` and the topic
  branch a package branch under it (``rust-core-phase-N-<name>``, a lane
  ``rust-core-phase-N-lane-<x>`` included), never ``main``, ``rust-core``
  or anything else; a name under ``rust-core-phase-N-lane-`` must be a
  lane, ``<x>`` lower-case letters and digits, and an archive branch
  (``rust-core-phase-N-archive-...``) is refused;
- on a lane, the lane is the branch checked out (not another branch, not
  a detached HEAD), the package ``--title`` names (``Land <ID>: ...``) is
  the one ``lane.py start`` recorded for this worktree (``git config
  --worktree sdm.package``), and every commit the candidate adds to the
  phase head is owned by that package: its subject starts with the ID
  followed by a colon or a space (``T1.11e: ...``, ``T1.11e tracker:
  ...``) and names no other package ID. A mention in the body or a trailer
  (``Refs: T1.11e``) is not ownership, so a commit whose subject is another
  package's, or names no package, never lands under this title. The one
  other commit owned is an earlier landing commit of the same package, left
  on the lane by a red ``--wait-for-landing-run`` (obsolete since E131): subject ``Land <ID>:
  ...``, one parent, no file changed, and a ``Base-SHA`` in the phase
  branch. A fix builds on it, and the new landing commit's parent is that
  fix and its ``Base-SHA`` the current phase head, so the phase branch
  still fast-forwards. A package that brings ``main`` into the phase
  branch (T2.0e; T2.0c is the first) does so from its lane, by one merge
  commit whose subject it owns, with two parents, the second an ancestor
  of ``main`` as the remote holds it at landing time (``git ls-remote``,
  then fetched; never a local ``main`` or a remote-tracking ref as it
  stands): the commits that merge brings in are ``main``'s and exempt from
  the subject rule (``main_merge``). A second merge commit, a merge of
  anything else, or a ``main`` commit rewritten exempts nothing, and is
  refused naming the commits;
- the worktree is clean, HEAD is the candidate, and the candidate descends
  from the phase branch's head on the remote without being it;
- ``--checkpoint``, when given, names the checkpoint land.py computes:
  ``git merge-base CANDIDATE BASE``, the base being the phase head (the
  design's "Reference observations and the gate", H2). land.py always
  computes it and never takes it from the caller; the flag is only a
  cross-check, so a design-checkpoint review's SHA passed in its place, as
  every landing from T1.11a to T1.11e carried, is refused;
- there are at least two review files, distinct, each whose first line is
  exactly ``APPROVE`` and which names the candidate by its full SHA or a
  prefix of at least seven characters that git resolves to the candidate
  and to nothing else;
- the newest CI run ``gh`` reports on the candidate's exact SHA completed
  with conclusion success; a cancelled, failed or unfinished run does not
  count (T1.8b's candidate run was cancelled). Only a push run on the
  phase branch or a topic branch under it (an archive branch never) counts
  as evidence, here and in ``verify``: a run of another event (a pull
  request's) skips the checkpoint's gate, which CI runs only on such
  pushes (T1.20, Codex P2). A candidate whose run is not green is rerun or
  waited on, then landed.

It then commits the landing commit (``git commit --allow-empty -s``) with
``Reviewed-by``, ``Protected-paths``, ``Base-SHA`` (the phase head) and
``Checkpoint-SHA`` (the merge base computed above) trailers, as earlier
landing commits do, checks that the
commit's tree is the candidate's and its parent the candidate (so a
landing whose tree differs is never made), and pushes
it to the topic branch and the phase branch in one atomic push, which the
remote refuses if the phase branch moved. On a lane it then reads both
branches back from the remote and reports, exiting 1, if the lane is not
the phase head, since the lane's next package starts from there. It
prints the landing SHA; it does not wait for the landing commit's runs,
which CI skips.

The landing commit's message names the CI evidence it lands on: the
candidate's green run, by id and SHA.

``--wait-for-landing-run``, which pushed the landing commit to the topic
branch and moved the phase branch only once that commit's own run was
green, is refused as obsolete (E131): CI skips that run, so it would never
come.

A run CI left out tested nothing: wherever a run is evidence (the
candidate's, the moved-from commit's, and in ``verify``), an empty landing
commit (one parent, its parent's tree, a subject starting ``Land ``) is
followed down to the first commit below it that is not one, whose run
stands for it (E131).

``verify`` (decision E131) checks that the landing commit changes no file,
carries its trailers and is in the phase branch on the remote, and stands
on the run it landed on: the newest
gating run on the candidate (the landing commit's parent), or, for a moved
landing, on its ``Moved-from`` commit, whose tree must be the landing's.
That run must be green; a red one fails, and a cancelled or unfinished one
asks for a rerun. The landing commit's own runs, which CI skips, are
reported and need not exist; one that ran and failed fails. A moved
landing's are the exception: CI runs them (E131 (c)), judging it against
its own Base-SHA, which the moved-from commit's run never did, so one of
them must be green, and without one (none, skipped, cancelled,
unfinished, or a pull request's only) ``verify`` is incomplete (exit 3).
On a failure it prints how to revert the fast-forward and changes
nothing: it never reverts by itself.

``--moved-from X`` (decision E130, the owner's of 2026-10-14) lands a
candidate C on X's reviews and X's green run, with no reviewer re-run and
no new CI, when C is X moved over landing commits only. Every check above
still applies to C (the branches, the lane and its commits' ownership,
the worktree and HEAD, the descent from the phase head H, the checkpoint,
and, at landing, the protected paths, the atomic push and the read-back),
but the review and CI checks are made for X in C's place: two reviews
naming X and the newest gating run on X's exact SHA green, checked
first. Then, with B the merge base of X and H, it refuses, naming the
condition, unless:

- same code: C's tree is X's;
- same patches: the commits of ``B..X`` (L_X), oldest first, pair one to
  one, in order, with the commits of ``B..H`` that are not landing commits
  (L_H) followed by those of ``H..C`` (L_C), each pair with the same tree;
  as every side starts from B's tree and a landing commit keeps its
  parent's, each of C's commits then makes exactly the change its partner
  in X makes, byte for byte. No commit of L_X or L_C is empty;
- only landing commits added: B is not H (C was moved over something),
  and each commit of ``B..H`` is a landing commit (one parent, its
  parent's tree, a subject starting ``Land ``) or, in order, one of X's own
  commits landed before it as a moved copy (L_H: for a single move it is
  empty; for a stack of packages, the earlier packages' moved commits);
- no merge in ``B..X``, ``B..H`` or ``H..C``.

X that is C itself, or is already in the phase branch, is refused. The
landing commit names X, X's green run by id and SHA, and that C was moved
over landing commits only, and adds the trailers ``Moved-from`` (X) and
``Moved-from-run`` (the run's id); ``verify`` stands on X's newest gating
run, which must be green, when X's tree is the landing's.

``--dry-run`` makes every check, prints the landing commit it would make,
and commits and pushes nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import commit_msg  # noqa: E402
import judging  # noqa: E402
from phases import (  # noqa: E402
    ENVIRONMENT,
    PACKAGE_ID,
    PHASE_BRANCH,
    Phase,
    PhaseError,
    phase_branch,
    phase_of,
)

# The names under a phase branch (``phases.Phase``): its topic branches,
# a lane (``<phase>-lane-a``) or a task (``<phase>-t1.12``), its lanes
# (decision E43, CI2), and its archive branches, where the tip of an
# abandoned lane package is kept, by hand, on the owner's decision, never
# a topic branch (CI2). A package (``phases.PACKAGE_ID``): a task
# (``T1.11e``, ``T2.0a``) or a process package (``CI3``).
#: The package that owns a commit: its subject starts with the ID, then a
#: colon or a space (``T1.11e: ...``, ``T1.11e fix 1: ...``,
#: ``T1.11e tracker: ...``).
SUBJECT_OWNER = re.compile(rf'({PACKAGE_ID.pattern})(?=[: ])')
#: Any package ID in a subject, even glued to letters before it
#: (``xT1.11f``), so a subject naming a second package is refused.
SUBJECT_ID = re.compile(rf'(?<![0-9.])({PACKAGE_ID.pattern})(?![A-Za-z0-9])')
#: An earlier landing commit's subject, left on the lane when its landing
#: run was red (the former ``--wait-for-landing-run``); owned only when it is what
#: land.py makes (see ``landing_problem``).
LANDING_SUBJECT = re.compile(rf'Land ({PACKAGE_ID.pattern}): ')
#: Where ``lane.py start`` records a lane worktree's package.
PACKAGE_KEY = 'sdm.package'
WORKFLOW = 'ci.yml'
APPROVE = 'APPROVE'
MIN_PREFIX = 7
RUN_FIELDS = (
    'databaseId,headSha,headBranch,status,conclusion,event,url,createdAt'
)


class Refusal(Exception):
    """A check failed; nothing was committed or pushed after it."""


@dataclass
class Completed:
    returncode: int
    stdout: str = ''
    stderr: str = ''


Run = Callable[[Sequence[str]], Completed]


def run_command(argv: Sequence[str]) -> Completed:
    proc = subprocess.run(list(argv), capture_output=True, text=True)
    return Completed(proc.returncode, proc.stdout, proc.stderr)


@dataclass
class Tools:
    """git and gh, through one function, so the tests can replace both."""

    run: Run = run_command
    repo: str | None = None
    sleep: Callable[[float], None] = time.sleep
    calls: list[tuple[str, ...]] = field(default_factory=list)
    #: The phase landed on or verified, which ``main`` resolves.
    phase: Phase | None = None

    def _call(self, argv: Sequence[str], check: bool) -> str:
        self.calls.append(tuple(argv))
        done = self.run(argv)
        if check and done.returncode != 0:
            raise Refusal(
                f'`{" ".join(argv)}` failed ({done.returncode}): '
                f'{(done.stderr or done.stdout).strip()}'
            )
        return done.stdout

    def git(self, *args: str, check: bool = True) -> str:
        return self._call(('git', *args), check)

    def attempt(self, *argv: str) -> Completed:
        """Runs a command whose failure is an answer, not an error."""
        self.calls.append(tuple(argv))
        return self.run(argv)

    def git_ok(self, *args: str) -> bool:
        return self.attempt('git', *args).returncode == 0

    def gh(self, *args: str) -> str:
        repo = ('--repo', self.repo) if self.repo else ()
        return self._call(('gh', *args, *repo), True)


# --- the checks ---------------------------------------------------------------


def resolve(tools: Tools, rev: str, what: str) -> str:
    done = tools.attempt(
        'git', 'rev-parse', '--verify', '--quiet', rev + '^{commit}'
    )
    sha = done.stdout.strip()
    if done.returncode != 0 or not re.fullmatch(r'[0-9a-f]{40}', sha):
        raise Refusal(f'{what} {rev!r} is not a commit here')
    return sha


def check_worktree(tools: Tools, candidate: str) -> None:
    status = tools.git('status', '--porcelain', '--untracked-files=all')
    if status.strip():
        raise Refusal(
            'the worktree is not clean; commit or remove:\n' + status.rstrip()
        )
    head = tools.git('rev-parse', 'HEAD').strip()
    if head != candidate:
        raise Refusal(
            f'HEAD is {head[:12]}, not the candidate {candidate[:12]}; check '
            'the candidate out first'
        )


def phase_head(tools: Tools, remote: str, phase: str) -> str:
    tools.git('fetch', '--quiet', remote, f'refs/heads/{phase}')
    return resolve(tools, f'refs/remotes/{remote}/{phase}', 'the phase head')


def check_descent(tools: Tools, candidate: str, base: str, phase: str) -> None:
    if candidate == base:
        raise Refusal(
            f'the candidate is the head of {phase} itself; there is nothing '
            'to land'
        )
    if not tools.git_ok('merge-base', '--is-ancestor', base, candidate):
        raise Refusal(
            f'the candidate {candidate[:12]} does not descend from the head of '
            f'{phase} ({base[:12]}); rebase it onto that head and gate it again'
        )


def checkpoint_of(
    tools: Tools, candidate: str, base: str, given: str | None
) -> str:
    """The checkpoint, ``git merge-base CANDIDATE BASE`` (the design's
    "Reference observations and the gate", H2), computed here and never
    taken from the caller. A ``--checkpoint`` that resolves to another
    commit is refused: until T1.11f the caller's value was only checked to
    be an ancestor, and seven landings carried the design-checkpoint
    review's SHA in its place (checkpoint 3)."""
    merged = tools.git('merge-base', candidate, base).strip()
    if not re.fullmatch(r'[0-9a-f]{40}', merged):
        raise Refusal(
            f'git merge-base of the candidate {candidate[:12]} and the phase '
            f'head {base[:12]} gave {merged!r}, not a commit'
        )
    if given is not None:
        named = resolve(tools, given, 'the checkpoint')
        if named != merged:
            raise Refusal(
                f'the checkpoint {named[:12]} is not the merge base of the '
                f'candidate and the phase head, {merged[:12]}; the checkpoint '
                'is `git merge-base CANDIDATE BASE` (design, H2), not a '
                'design-checkpoint review; omit --checkpoint and land.py '
                'computes it'
            )
    return merged


def phase_named(name: str) -> Phase:
    """The phase branch ``name`` names, or a refusal saying it is none."""
    try:
        return Phase(name)
    except PhaseError as error:
        raise Refusal(
            f'the phase branch is {name!r}; land.py lands only on a phase '
            'branch, rust-core-phase-N'
        ) from error


def check_branches(phase_name: str, topic: str) -> None:
    """Refuses a destination other than the phase branch and a package
    branch under it, before anything is fetched, committed or pushed."""
    phase = phase_named(phase_name)
    name = phase.branch
    if topic in ('main', 'master', 'rust-core', name):
        raise Refusal(
            f'the topic branch is {topic!r}; a landing pushes a package branch '
            f'({name}-<name>, a lane {name}-lane-<x>) and the '
            'phase branch, never main, rust-core or the phase branch as topic'
        )
    if not phase.topic.fullmatch(topic) or topic.endswith(('.', '.lock')):
        raise Refusal(
            f'the topic branch {topic!r} is not a package branch '
            f'({name}-<name> or {name}-lane-<x>)'
        )
    if topic.startswith(phase.archive_prefix):
        raise Refusal(
            f'the topic branch {topic!r} is an archive branch, which keeps an '
            'abandoned package and never lands'
        )
    if topic.startswith(phase.lane_prefix) and phase.lane_of(topic) is None:
        raise Refusal(
            f'the topic branch {topic!r} is under {phase.lane_prefix} but is '
            f'not a lane; a lane is {phase.lane_prefix}<x>, <x> lower-case '
            'letters and digits'
        )


def lane_of(branch: str) -> str | None:
    """The lane's name when ``branch`` is a lane of its phase branch
    (``rust-core-phase-N-lane-<x>``), else None."""
    phase = phase_of(branch)
    return Phase(phase).lane_of(branch) if phase else None


def owner_of(subject: str) -> str | None:
    """The package a commit's subject names as its owner, if any."""
    match = SUBJECT_OWNER.match(subject)
    return match.group(1) if match else None


def ownership_problem(subject: str, package: str) -> str | None:
    """None when ``package`` owns a commit with this subject; else why
    not. Only the subject counts: the body and trailers may cite other
    packages, and a mention there owns nothing."""
    owner = owner_of(subject)
    if owner is None:
        return (
            f'the subject does not start with {package} '
            f"('{package}: ...' or '{package} <part>: ...')"
        )
    if owner != package:
        return f"the subject is {owner}'s"
    others = sorted({m for m in SUBJECT_ID.findall(subject) if m != package})
    if others:
        return (
            f'the subject also names {", ".join(others)}; cite another '
            'package in the body'
        )
    return None


def recorded_package(tools: Tools, path: str | None = None) -> str | None:
    """The package ``lane.py start`` recorded for the worktree at ``path``
    (the current one by default): ``sdm.package`` in that worktree's own
    configuration. Without ``extensions.worktreeConfig`` git has no
    per-worktree configuration, so nothing counts as recorded."""
    where = ('-C', path) if path else ()
    ext = tools.attempt(
        'git', *where, 'config', '--bool', '--get', 'extensions.worktreeConfig'
    )
    if ext.returncode != 0 or ext.stdout.strip() != 'true':
        return None
    done = tools.attempt(
        'git', *where, 'config', '--worktree', '--get', PACKAGE_KEY
    )
    value = done.stdout.strip()
    return value if done.returncode == 0 and value else None


def landing_problem(
    tools: Tools, sha: str, package: str, base: str, *where: str
) -> str | None:
    """None when ``sha`` is a landing commit of ``package`` as land.py
    makes one: subject ``Land <package>: ...``, one parent, the parent's
    tree (it changes no file), and a ``Base-SHA`` trailer naming a commit
    of the phase branch (an ancestor of ``base``, the phase head checked
    against). Such a commit stayed on the lane when its landing run was
    red under the former ``--wait-for-landing-run`` (obsolete since E131);
    the package's fixes build on it and land again."""
    match = LANDING_SUBJECT.match(
        tools.git(*where, 'log', '-1', '--format=%s', sha).strip()
    )
    if not match or match.group(1) != package:
        return f"the subject is not 'Land {package}: ...'"
    second = tools.attempt(
        'git', *where, 'rev-parse', '--verify', '--quiet', f'{sha}^2'
    )
    if second.returncode == 0:
        return 'a landing commit has one parent'
    tree = tools.git(*where, 'rev-parse', f'{sha}^{{tree}}').strip()
    parent = tools.git(*where, 'rev-parse', f'{sha}^^{{tree}}').strip()
    if tree != parent:
        return 'a landing commit changes no file, and this one does'
    base_sha = trailers_of(tools, sha, *where).get('Base-SHA', '')
    if not re.fullmatch(r'[0-9a-f]{40}', base_sha):
        return 'a landing commit has a Base-SHA trailer, and this one has none'
    if not tools.git_ok(*where, 'merge-base', '--is-ancestor', base_sha, base):
        return (
            f'its Base-SHA {base_sha[:12]} is not in the phase branch, so it '
            'was not landed from this lane onto the phase head'
        )
    return None


#: The product branch a package may merge into its phase branch (T2.0e),
#: read from the remote at check time, never from a local ref.
MAIN = 'main'


def remote_main(tools: Tools, remote: str, *where: str) -> str:
    """``main`` as the remote holds it now: its SHA as ``git ls-remote``
    reads it, its commits fetched (which refreshes
    ``refs/remotes/<remote>/main``). A local ``main``, or a remote-tracking
    ref moved by hand, never stands in for it."""
    ref = f'refs/heads/{MAIN}'
    out = tools.git(*where, 'ls-remote', remote, ref)
    tips = [
        tip
        for tip, _, name in (line.partition('\t') for line in out.splitlines())
        if name == ref
    ]
    if len(tips) != 1 or not re.fullmatch(r'[0-9a-f]{40}', tips[0]):
        raise Refusal(
            f'the lane merges a commit, and the remote {remote} has no {MAIN} '
            f'to judge it against; a package merges only {MAIN}'
        )
    tools.git(*where, 'fetch', '--quiet', remote, ref)
    return tips[0]


def main_merge(
    tools: Tools, remote: str, base: str, head: str, package: str, *where: str
) -> tuple[set[str], list[tuple[str, str, str]]]:
    """The commits of ``base..head`` a merge of ``main`` exempts from the
    subject rule, and the merge commits refused, each as (SHA, subject,
    why). A package may bring ``main`` in by one merge commit (T2.0e): two
    parents, a subject ``package`` owns, its second parent an ancestor of
    the remote's ``main`` (:func:`remote_main`). The commits it brings in,
    reachable from that parent but not from its first parent or ``base``,
    are then ``main``'s, and exempt; every other commit still needs the
    package's ID. Only the merges ``main`` does not hold are the
    package's: ``main``'s own merge commits (its pull requests) come in
    with it and are exempt like its other commits. Without a merge of the
    package's nothing is exempt; with two or more, each is refused and
    nothing is exempt; a merge whose subject the package does not own
    exempts nothing (the subject rule names it)."""
    merges = tools.git(
        *where, 'rev-list', '--merges', f'{base}..{head}'
    ).split()
    if not merges:
        return set(), []

    def title(sha: str) -> str:
        return tools.git(*where, 'log', '-1', '--format=%s', sha).strip()

    main = remote_main(tools, remote, *where)
    merges = [
        sha
        for sha in merges
        if not tools.git_ok(*where, 'merge-base', '--is-ancestor', sha, main)
    ]
    if not merges:
        return set(), []
    if len(merges) > 1:
        return set(), [
            (
                sha,
                title(sha),
                f'one of {len(merges)} merge commits; a package brings {MAIN} '
                'in by one merge commit and merges nothing else',
            )
            for sha in merges
        ]
    (merge,) = merges
    subject = title(merge)
    if ownership_problem(subject, package) is not None:
        return set(), []
    parents = tools.git(
        *where, 'rev-list', '--parents', '-n', '1', merge
    ).split()[1:]
    if len(parents) != 2:
        return set(), [
            (
                merge,
                subject,
                f'a merge of {len(parents)} parents; a merge of {MAIN} has two',
            )
        ]
    if not tools.git_ok(
        *where, 'merge-base', '--is-ancestor', parents[1], main
    ):
        return set(), [
            (
                merge,
                subject,
                f'it merges {parents[1][:12]}, which is not in {MAIN} as '
                f'{remote} holds it ({main[:12]}); a package merges only '
                f'{MAIN}, never another branch, a local {MAIN} or a rewritten '
                'commit',
            )
        ]
    brought = tools.git(
        *where, 'rev-list', parents[1], '--not', base, parents[0]
    ).split()
    return set(brought), []


def commits_not_owned(
    tools: Tools,
    base: str,
    head: str,
    package: str,
    *where: str,
    remote: str = 'origin',
    stacked: bool = False,
) -> list[tuple[str, str, str]]:
    """The commits in ``base..head`` that ``package`` does not own, each as
    (SHA, subject, why). An earlier landing commit of ``package`` that
    ``landing_problem`` accepts is owned, and the commits of ``main`` one
    merge commit of the package brings in are exempt (:func:`main_merge`,
    T2.0e). With ``stacked`` (T3.0), the package may sit on a stack of
    other packages' landing commits (``land.py --stack``): the stack below
    its newest landing is checked as a stack (:func:`stack_chain`), each
    landing owning the commits below it, and only the commits above are
    the package's."""
    if stacked:
        chain, problems = stack_chain(tools, base, head, *where, remote=remote)
        if any(link.package != package for link in chain):
            floor = chain[-1].sha
            return [
                (floor, 'the stack below', why) for why in problems
            ] + commits_not_owned(
                tools, floor, head, package, *where, remote=remote
            )
    exempt, found = main_merge(tools, remote, base, head, package, *where)
    refused = {sha for sha, _, _ in found}
    out = tools.git(*where, 'log', '--format=%H%x00%s%x1e', f'{base}..{head}')
    for entry in out.split('\x1e'):
        sha, sep, text = entry.strip('\n').partition('\x00')
        subject = text.split('\n', 1)[0]
        if not sep or sha in exempt or sha in refused:
            continue
        if LANDING_SUBJECT.match(subject):
            why = landing_problem(tools, sha, package, base, *where)
        else:
            why = ownership_problem(subject, package)
        if why:
            found.append((sha, subject, why))
    return found


def check_lane_branch(tools: Tools, topic: str) -> None:
    """On a lane, the branch checked out is the lane itself: land.py pushes
    with --no-verify, so it makes the pre-push hook's branch check here,
    and the package it compares is the lane worktree's own."""
    done = tools.attempt('git', 'symbolic-ref', '--quiet', '--short', 'HEAD')
    current = done.stdout.strip() if done.returncode == 0 else ''
    if current != topic:
        raise Refusal(
            f'{topic} is a lane, and this worktree has '
            f'{current or "a detached HEAD"} checked out; land a lane from '
            'its own worktree, on its own branch'
        )


def check_lane_package(tools: Tools, topic: str, package: str) -> None:
    """On a lane, the package the title names is the worktree's."""
    if not PACKAGE_ID.fullmatch(package):
        raise Refusal(
            f'--title names {package!r}, not a package ID such as T1.11e or '
            f"CI3; a lane's landing title is 'Land <ID>: <summary>'"
        )
    recorded = recorded_package(tools)
    if recorded is None:
        raise Refusal(
            f'{topic} is a lane, and this worktree records no package '
            f'(`git config --worktree {PACKAGE_KEY}`); start the lane with '
            '`scripts/rust_core/lane.py start`'
        )
    if recorded != package:
        raise Refusal(
            f'the title lands {package}, but this worktree of {topic} records '
            f'{recorded} (`git config --worktree {PACKAGE_KEY}`); a lane '
            'carries one package at a time'
        )


def check_lane_commits(
    tools: Tools, base: str, candidate: str, package: str, remote: str
) -> None:
    stray = commits_not_owned(tools, base, candidate, package, remote=remote)
    if stray:
        raise Refusal(
            f'commits on the lane are not owned by {package} (a subject '
            f"'{package}: ...' or '{package} <part>: ...', naming no other "
            "package), so they may be another package's:\n- "
            + '\n- '.join(
                f'{sha[:12]} {subject} ({why})' for sha, subject, why in stray
            )
        )


def confirm_lane(
    tools: Tools, remote: str, topic: str, phase: str, sha: str
) -> str | None:
    """None when the remote has the lane and the phase branch both at
    ``sha``; else what it has."""
    out = tools.git(
        'ls-remote', remote, f'refs/heads/{topic}', f'refs/heads/{phase}'
    )
    tips = {
        ref.removeprefix('refs/heads/'): tip
        for tip, _, ref in (line.partition('\t') for line in out.splitlines())
    }
    if tips.get(topic) == sha and tips.get(phase) == sha:
        return None
    return ', '.join(
        f'{branch} at {(tips.get(branch) or "nothing")[:12]}'
        for branch in (topic, phase)
    )


#: A hex word of seven to forty digits, standing on its own.
HEX_WORD = re.compile(r'(?<![0-9a-fA-F])[0-9a-fA-F]{7,40}(?![0-9a-fA-F])')


def names_candidate(
    tools: Tools, text: str, candidate: str, what: str = 'the candidate'
) -> str | None:
    """None when the text names the candidate, by its full SHA or by an
    abbreviation git resolves to the candidate and to nothing else; else
    why not. An abbreviation that is the candidate's prefix but that git
    cannot resolve uniquely names no commit, so it is refused. ``what``
    names the commit in the answer (the moved-from commit, under
    ``--moved-from``)."""
    ambiguous: list[str] = []
    for token in HEX_WORD.findall(text):
        token = token.lower()
        if not candidate.startswith(token):
            continue
        if len(token) == 40:
            return None
        if resolve_quiet(tools, token) == candidate:
            return None
        ambiguous.append(token)
    if ambiguous:
        return (
            f'names {what} only by {", ".join(ambiguous)}, which git '
            f'does not resolve to {what} alone (ambiguous); use the '
            'full SHA'
        )
    return (
        f'does not name {what} {candidate[:12]} (its full SHA or an '
        f'unambiguous prefix of at least {MIN_PREFIX} characters)'
    )


def resolve_quiet(tools: Tools, rev: str) -> str | None:
    done = tools.attempt(
        'git', 'rev-parse', '--verify', '--quiet', rev + '^{commit}'
    )
    sha = done.stdout.strip()
    return sha if done.returncode == 0 and len(sha) == 40 else None


def check_reviews(
    tools: Tools,
    paths: Sequence[str],
    candidate: str,
    what: str = 'the candidate',
) -> list[Path]:
    if len(paths) < 2:
        raise Refusal(
            f'{len(paths)} review file(s); a landing needs both reviews, '
            'given as --review twice'
        )
    seen: dict[Path, str] = {}
    contents: dict[str, Path] = {}
    problems: list[str] = []
    for raw in paths:
        path = Path(raw)
        try:
            text = path.read_text()
        except OSError as error:
            problems.append(f'{raw}: unreadable ({error.strerror})')
            continue
        key = path.resolve()
        if key in seen:
            problems.append(f'{raw}: given twice')
            continue
        seen[key] = text
        if text in contents:
            problems.append(f'{raw}: the same text as {contents[text]}')
            continue
        contents[text] = path
        first = text.split('\n', 1)[0]
        if first != APPROVE:
            problems.append(
                f'{raw}: the first line is {first[:60]!r}, not exactly '
                f'{APPROVE!r}'
            )
        why = names_candidate(tools, text, candidate, what)
        if why:
            problems.append(f'{raw}: {why}')
    if problems:
        raise Refusal(
            f'the reviews do not approve {what}:\n- ' + '\n- '.join(problems)
        )
    return [Path(p) for p in paths]


def gating(run: dict[str, object], phase: Phase) -> bool:
    """Whether a run is evidence: a push to the phase branch or a topic
    branch under it, never an archive branch, so CI ran the checkpoint's
    gate in it; a pull request's run on the same SHA skips the gate, and
    a run under another phase was judged against another checkpoint."""
    branch = str(run.get('headBranch'))
    return run.get('event') == 'push' and phase.holds(branch)


def phase_of_tools(tools: Tools) -> Phase:
    """The phase ``main`` resolved, or, for a caller that set none, the
    one ``phases.phase_branch`` gives."""
    return tools.phase or Phase(phase_branch())


def runs_on(tools: Tools, sha: str) -> list[dict[str, object]]:
    out = tools.gh(
        'run',
        'list',
        '--commit',
        sha,
        '--workflow',
        WORKFLOW,
        '--limit',
        '50',
        '--json',
        RUN_FIELDS,
    )
    runs = [
        r
        for r in json.loads(out or '[]')
        if r.get('headSha') == sha and gating(r, phase_of_tools(tools))
    ]
    return sorted(runs, key=lambda r: str(r.get('createdAt', '')), reverse=True)


def describe(run: dict[str, object]) -> str:
    state = run.get('conclusion') or run.get('status')
    return f'run {run.get("databaseId")} on {run.get("headBranch")}: {state}'


def skip_shaped(tools: Tools, sha: str) -> bool:
    """Whether CI leaves out ``sha``'s jobs (decision E131): one parent,
    its parent's tree, a subject starting ``Land ``. Its run, green with
    every job but the first skipped, tested nothing."""
    if not tools.git('log', '-1', '--format=%s', sha).startswith('Land '):
        return False
    parent = tools.attempt('git', 'rev-parse', '--verify', '--quiet', f'{sha}^')
    second = tools.attempt(
        'git', 'rev-parse', '--verify', '--quiet', f'{sha}^2'
    )
    if parent.returncode != 0 or second.returncode == 0:
        return False
    return (
        tools.git('rev-parse', f'{sha}^{{tree}}').strip()
        == tools.git('rev-parse', f'{sha}^^{{tree}}').strip()
    )


def tested_commit(tools: Tools, sha: str) -> str:
    """``sha``, or, when CI leaves out its jobs, the first commit below it
    whose jobs CI runs: the same tree, so its run is the evidence."""
    while skip_shaped(tools, sha):
        sha = tools.git('rev-parse', f'{sha}^').strip()
    return sha


def below(sha: str, tested: str) -> str:
    """How a refusal names the commit whose run stands for ``sha``."""
    if tested == sha:
        return ''
    return (
        f' (the first commit below {sha[:12]} that is no empty landing '
        'commit, whose runs CI leaves out)'
    )


def check_candidate_ci(
    tools: Tools, candidate: str, what: str = 'the candidate'
) -> dict[str, object]:
    """The newest gating run on ``candidate``'s exact SHA, when it is
    green; ``what`` names the commit in a refusal (the moved-from commit,
    under ``--moved-from``). An empty landing commit's run tested nothing
    (E131), so for one the first commit below it that is none stands."""
    tested = tested_commit(tools, candidate)
    what = what + below(candidate, tested)
    candidate = tested
    runs = runs_on(tools, candidate)
    if not runs:
        raise Refusal(
            f'no CI run on {what} {candidate[:12]} itself that is a '
            'push to the phase branch or a topic branch under it (a pull '
            "request's run skips the gate and does not count); push it and "
            'let its run finish'
        )
    newest = runs[0]
    if (
        newest.get('status') != 'completed'
        or newest.get('conclusion') != 'success'
    ):
        raise Refusal(
            f'the newest CI run on {what} is not green ({describe(newest)}); '
            'a cancelled, failed or unfinished run does not count. Wait for '
            'or rerun it'
        )
    return newest


# --- a candidate moved over landing commits (--moved-from) ------------------


@dataclass
class Commit:
    sha: str
    parents: int
    subject: str


def range_commits(tools: Tools, rng: str) -> list[Commit]:
    """The commits in ``rng``, oldest first."""
    out = tools.git('log', '--topo-order', '--format=%H%x00%P%x00%s%x1e', rng)
    found: list[Commit] = []
    for entry in out.split('\x1e'):
        fields = entry.strip('\n').split('\x00')
        if len(fields) == 3:
            sha, parents, subject = fields
            found.append(Commit(sha, len(parents.split()), subject))
    return found[::-1]


def tree_of(tools: Tools, sha: str) -> str:
    return tools.git('rev-parse', f'{sha}^{{tree}}').strip()


def empty_commit(tools: Tools, commit: Commit) -> bool:
    """One parent and its parent's tree: the commit changes no file."""
    return commit.parents == 1 and tree_of(tools, commit.sha) == tree_of(
        tools, f'{commit.sha}^'
    )


def landing_why_not(tools: Tools, commit: Commit) -> str | None:
    """None when the commit is a landing commit: one parent, its parent's
    tree, and a subject starting ``Land ``; else why not."""
    if not commit.subject.startswith('Land '):
        return "the subject does not start with 'Land '"
    if commit.parents != 1:
        return 'it has more than one parent'
    if not empty_commit(tools, commit):
        return "its tree is not its parent's: it changes files"
    return None


def check_moved_from(
    tools: Tools, moved: str, candidate: str, base: str, phase: str
) -> None:
    """Refuses a ``--moved-from`` that cannot name a move: the candidate
    itself, or a commit already in the phase branch."""
    label = f'--moved-from {moved[:12]}'
    if moved == candidate:
        raise Refusal(
            f'{label} is the candidate itself; land it without --moved-from'
        )
    if tools.git_ok('merge-base', '--is-ancestor', moved, base):
        raise Refusal(
            f'{label} is already in {phase} (an ancestor of its head '
            f'{base[:12]}); there is nothing of it to land'
        )


@dataclass
class Move:
    """What ``check_moved`` found: the landing commits moved over, X's
    commits already in the phase branch as moved copies, and the
    candidate's own commits."""

    landings: int
    landed: int
    commits: int


def check_moved(
    tools: Tools, moved: str, candidate: str, base: str, phase: str
) -> Move:
    """Refuses unless the candidate is ``moved`` (X) moved over landing
    commits only (decision E130). With B the merge base of X and the phase
    head H, the commits of ``B..X`` (L_X), oldest first, pair one to one,
    in order, with those of ``B..H`` that are not landing commits (L_H)
    followed by those of ``H..C`` (L_C), and each pair has the same tree.
    Since a landing commit keeps its parent's tree and every side starts
    from B's, each commit of L_H and L_C then makes exactly the change its
    partner in X makes, byte for byte. L_H is empty for a single move, and
    holds X's own earlier commits, landed as moved copies, for a stack.
    No merge on any side, and no empty commit in L_X or L_C. The same tree
    for C and X is checked first. X's reviews and CI are checked by the
    caller, before this."""
    label = f'--moved-from {moved[:12]}'
    # 2. Same code.
    tree_x, tree_c = tree_of(tools, moved), tree_of(tools, candidate)
    if tree_x != tree_c:
        raise Refusal(
            f"{label}: not the same code: the candidate's tree "
            f"{tree_c[:12]} is not the moved-from commit's {tree_x[:12]}; "
            'a candidate whose code differs needs its own reviews and CI'
        )
    done = tools.attempt('git', 'merge-base', moved, base)
    common = done.stdout.strip()
    if done.returncode != 0 or not re.fullmatch(r'[0-9a-f]{40}', common):
        raise Refusal(
            f'{label} shares no history with the head of {phase} '
            f'{base[:12]}; it was not moved from there'
        )
    if common == base:
        raise Refusal(
            f'{label}: not moved over landing commits: the head of {phase} '
            f"{base[:12]} is already in the moved-from commit's history, so "
            'the candidate was moved over nothing; land the moved-from '
            'commit itself'
        )
    l_x = range_commits(tools, f'{common}..{moved}')
    b_h = range_commits(tools, f'{common}..{base}')
    l_c = range_commits(tools, f'{base}..{candidate}')
    # No merge anywhere: it has no change of its own to pair, and is no
    # landing commit.
    merges = [
        f'{c.sha[:12]} {c.subject} (in {where})'
        for where, commits in (
            (f"{common[:12]}..X, the moved-from commit's", l_x),
            (f'{common[:12]}..{base[:12]}, the phase branch', b_h),
            ('the candidate, since the phase head', l_c),
        )
        for c in commits
        if c.parents > 1
    ]
    if merges:
        raise Refusal(
            f'{label}: a merge commit, which has no patch of its own to '
            'compare and is not a landing commit (a landing commit has one '
            'parent):\n- ' + '\n- '.join(merges)
        )
    # 3. No empty commit in X's commits or the candidate's.
    empty = [
        f"{c.sha[:12]} {c.subject} (the {side}'s)"
        for side, commits in (('moved-from commit', l_x), ('candidate', l_c))
        for c in commits
        if empty_commit(tools, c)
    ]
    if empty:
        raise Refusal(
            f'{label}: not the same patches: an empty commit, with no patch '
            'of its own:\n- ' + '\n- '.join(empty)
        )
    # 4. Between B and H, landing commits, and otherwise X's own commits
    # in order (a stack's earlier packages, landed as moved copies).
    l_h = [c for c in b_h if landing_why_not(tools, c) is not None]
    stray: list[str] = []
    for n, c in enumerate(l_h):
        why = landing_why_not(tools, c)
        if n >= len(l_x):
            stray.append(
                f'{c.sha[:12]} {c.subject} ({why}; and X has no commit '
                'left to be its moved copy)'
            )
        elif tree_of(tools, c.sha) != tree_of(tools, l_x[n].sha):
            stray.append(
                f'{c.sha[:12]} {c.subject} ({why}; and its tree is not that '
                f"of X's commit {n + 1}, {l_x[n].sha[:12]} "
                f'{l_x[n].subject}, so it is not that commit moved)'
            )
    if stray:
        raise Refusal(
            f'{label}: not only landing commits between the merge base '
            f'{common[:12]} of the moved-from commit and the head of {phase}: '
            'each commit there is a landing commit (one parent, its '
            "parent's tree, a subject starting 'Land ') or, in order, one "
            "of X's own commits moved (the same tree):\n- " + '\n- '.join(stray)
        )
    # 3. The candidate's commits are the rest of X's, one to one.
    rest = l_x[len(l_h) :]
    if len(rest) != len(l_c):
        raise Refusal(
            f'{label}: not the same patches: the candidate adds {len(l_c)} '
            f'commit(s) to the head of {phase}, and the moved-from commit '
            f'has {len(rest)} not yet in it'
        )
    differ = [
        f'commit {n}: {c.sha[:12]} {c.subject} against {x.sha[:12]} {x.subject}'
        for n, (c, x) in enumerate(zip(l_c, rest, strict=True), start=1)
        if tree_of(tools, c.sha) != tree_of(tools, x.sha)
    ]
    if differ:
        raise Refusal(
            f"{label}: not the same patches: in order, each of the candidate's "
            "commits has its partner's tree in X, and these do not:\n- "
            + '\n- '.join(differ)
        )
    return Move(len(b_h) - len(l_h), len(l_h), len(l_c))


# --- the landing commit -------------------------------------------------------


@dataclass
class Landing:
    title: str
    reviewer: str
    protected_paths: str
    base: str
    checkpoint: str
    summary: str
    evidence: str
    #: ``Moved-from`` and ``Moved-from-run`` for a moved candidate (E130);
    #: ``Stack-base``, ``Topic-branch`` and ``Train`` for a stacked one.
    moved: dict[str, str] = field(default_factory=dict)

    def message(self) -> str:
        subject = (
            self.title
            if self.title.startswith('Land ')
            else f'Land {self.title}'
        )
        trailers = [
            f'Reviewed-by: {self.reviewer}',
            f'Protected-paths: {self.protected_paths}',
            f'Base-SHA: {self.base}',
            f'Checkpoint-SHA: {self.checkpoint}',
            *(f'{key}: {value}' for key, value in self.moved.items()),
        ]
        # Wrapped as the earlier landing commits are, paragraph by paragraph.
        summary = '\n\n'.join(
            textwrap.fill(' '.join(part.split()), 72)
            for part in (*self.summary.split('\n\n'), self.evidence)
            if part.strip()
        )
        return f'{subject}\n\n{summary}\n\n' + '\n'.join(trailers) + '\n'

    def expected_trailers(self) -> dict[str, str]:
        return {
            'Reviewed-by': self.reviewer,
            'Protected-paths': self.protected_paths,
            'Base-SHA': self.base,
            'Checkpoint-SHA': self.checkpoint,
            **self.moved,
        }


def one_line(text: str, what: str) -> str:
    text = ' '.join(text.split())
    if not text:
        raise Refusal(f'{what} is empty')
    return text


def package_of(title: str) -> str:
    return title.removeprefix('Land ').split(':', 1)[0].strip()


def trailers_of(tools: Tools, sha: str, *where: str) -> dict[str, str]:
    """The commit's trailers, as git reads them; the last value of a key."""
    text = tools.git(
        *where, 'log', '-1', '--format=%(trailers:only,unfold)', sha
    )
    trailers: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(': ')
        if sep:
            trailers[key.strip()] = value.strip()
    return trailers


def make_landing(tools: Tools, landing: Landing, candidate: str) -> str:
    with tempfile.NamedTemporaryFile(
        'w', suffix='.txt', delete=False
    ) as handle:
        handle.write(landing.message())
        message = handle.name
    try:
        tools.git('commit', '--allow-empty', '--quiet', '-s', '-F', message)
    finally:
        Path(message).unlink(missing_ok=True)
    sha = tools.git('rev-parse', 'HEAD').strip()
    parent = tools.git('rev-parse', 'HEAD^').strip()
    tree = tools.git('rev-parse', 'HEAD^{tree}').strip()
    expected = tools.git('rev-parse', candidate + '^{tree}').strip()
    if parent != candidate or tree != expected:
        raise Refusal(
            f'the landing commit {sha[:12]} is not the candidate plus trailers '
            f'(parent {parent[:12]}, tree {tree[:12]} against {expected[:12]}); '
            f'nothing was pushed. Undo it with `git reset --soft {candidate}`'
        )
    trailers = trailers_of(tools, sha)
    missing = {
        k: v
        for k, v in landing.expected_trailers().items()
        if trailers.get(k) != v
    }
    if missing or 'Signed-off-by' not in trailers:
        raise Refusal(
            f'the landing commit {sha[:12]} lacks trailers {sorted(missing)}'
            f'{" and Signed-off-by" if "Signed-off-by" not in trailers else ""}; '
            f'nothing was pushed. Undo it with `git reset --soft {candidate}`'
        )
    return sha


def watch(tools: Tools, run_id: object) -> dict[str, object]:
    """Waits for a run to finish and returns its final state."""
    repo = ('--repo', tools.repo) if tools.repo else ()
    # Its exit status is the run's; the state read next is what counts.
    tools.attempt('gh', 'run', 'watch', str(run_id), '--interval', '30', *repo)
    out = tools.gh('run', 'view', str(run_id), '--json', RUN_FIELDS)
    return json.loads(out)


# --- stacks and trains (T3.0, items 1 and 2) --------------------------------

#: How many packages one CI run may cover: the owner's "every N packages
#: (e.g. 4 packages) before CI is mandatory" (2026-10-21, evening).
STACK_MAX = 4
STACK_MAX_ENV = 'SDM_STACK_MAX'
#: The trailer naming the phase head a stack was built on.
STACK_BASE = 'Stack-base'
#: The trailer a train's landing carries.
TRAIN = 'Train'
#: The epic branch, whose merge base with the phase head is the phase's start.
EPIC = 'rust-core'


def stack_max(given: int | None) -> int:
    """The most packages one stack may hold: ``--max-stack``, else
    ``SDM_STACK_MAX``, else 4."""
    if given is not None:
        value = given
    else:
        raw = os.environ.get(STACK_MAX_ENV, '')
        try:
            value = int(raw) if raw else STACK_MAX
        except ValueError:
            raise Refusal(f'{STACK_MAX_ENV}={raw!r} is not a number') from None
    if value < 1:
        raise Refusal(f'the stack limit is {value}; it is at least 1')
    if value > STACK_MAX:
        # CI's ratchet holds a phase push to 4 landings (ratchet.py
        # stack_limit, with no SDM_STACK_MAX in CI): more would land and
        # turn the phase branch red (Claude r1 P3-7).
        raise Refusal(
            f'the stack limit is {value}; CI judges at most {STACK_MAX} '
            'landings in one push'
        )
    return value


@dataclass
class Stacked:
    """A landing commit of a stack: the package it lands, and the commit
    below its own commits (the landing before it, or the phase head)."""

    sha: str
    package: str
    below: str
    trailers: dict[str, str]

    @property
    def train(self) -> bool:
        return self.trailers.get(TRAIN, '').lower() == 'yes'


def landing_shaped(tools: Tools, sha: str, *where: str) -> bool:
    """One parent, its parent's tree, a subject ``Land <ID>: ...``."""
    subject = tools.git(*where, 'log', '-1', '--format=%s', sha).strip()
    if not LANDING_SUBJECT.match(subject):
        return False
    if (
        tools.attempt(
            'git', *where, 'rev-parse', '--verify', '--quiet', f'{sha}^2'
        ).returncode
        == 0
    ):
        return False
    return tools.git(*where, 'rev-parse', f'{sha}^{{tree}}').strip() == (
        tools.git(*where, 'rev-parse', f'{sha}^^{{tree}}').strip()
    )


def stack_chain(
    tools: Tools, base: str, tip: str, *where: str, remote: str = 'origin'
) -> tuple[list[Stacked], list[str]]:
    """The landing commits between the phase head ``base`` and ``tip``,
    oldest first, and what is wrong with them as a stack: each one empty,
    with ``Reviewed-by`` and ``Protected-paths``, its ``Base-SHA`` and
    ``Checkpoint-SHA`` the landing before it (the phase head for the
    first), and every commit between them owned by its package."""
    out = tools.git(*where, 'log', '--format=%H%x00%s%x1e', f'{base}..{tip}')
    listed = []
    for entry in out.split('\x1e'):
        sha, sep, text = entry.strip('\n').partition('\x00')
        match = LANDING_SUBJECT.match(text.split('\n', 1)[0])
        if sep and match:
            listed.append((sha, match))
    chain: list[Stacked] = []
    problems: list[str] = []
    below_sha = base
    for sha, match in reversed(listed):
        subject = tools.git(*where, 'log', '-1', '--format=%s', sha).strip()
        trailers = trailers_of(tools, sha, *where)
        recorded = trailers.get('Base-SHA', '')
        package = match.group(1)
        label = f'{sha[:12]} {subject}'
        if not landing_shaped(tools, sha, *where):
            problems.append(
                f'{label}: a landing commit has one parent and changes no file'
            )
        if recorded != below_sha:
            problems.append(
                f'{label}: its Base-SHA {recorded[:12] or "(none)"} is not '
                f'the landing below it, {below_sha[:12]}; restack it '
                '(land.py restack)'
            )
        if trailers.get('Checkpoint-SHA') != below_sha:
            problems.append(
                f'{label}: its Checkpoint-SHA is not the landing below it, '
                f'{below_sha[:12]}'
            )
        for key in ('Reviewed-by', 'Protected-paths'):
            if not trailers.get(key):
                problems.append(f'{label}: no {key} trailer')
        problems.extend(
            f'{label}: {stray[:12]} {stray_subject} ({why})'
            for stray, stray_subject, why in commits_not_owned(
                tools, below_sha, f'{sha}^', package, *where, remote=remote
            )
        )
        chain.append(Stacked(sha, package, below_sha, trailers))
        below_sha = sha
    return chain, problems


def changed_paths(tools: Tools, old: str, new: str, *where: str) -> list[str]:
    out = tools.git(*where, 'diff', '--name-only', '-z', old, new)
    return [p for p in out.split('\0') if p]


def judging_rules(tools: Tools, base: str, *where: str) -> judging.Rules | None:
    """The judging paths as the phase head lists them, or None before the
    list exists (T3.0 lands it)."""
    done = tools.attempt('git', *where, 'show', f'{base}:{judging.FILE}')
    if done.returncode != 0:
        return None
    return judging.patterns(done.stdout)


def phase_first_landing(
    tools: Tools, remote: str, base: str, *where: str
) -> str | None:
    """The phase's first landing (up to ``base``), as ``<sha> <subject>``,
    or None while the phase holds none. The phase starts at its merge base
    with the epic branch as the remote holds it."""
    if (
        tools.attempt(
            'git', *where, 'fetch', '--quiet', remote, f'refs/heads/{EPIC}'
        ).returncode
        == 0
    ):
        epic = f'refs/remotes/{remote}/{EPIC}'
        start = tools.attempt('git', *where, 'merge-base', base, epic)
    else:
        start = Completed(1)
    rng = f'{start.stdout.strip()}..{base}' if start.returncode == 0 else base
    out = tools.git(
        *where, 'log', '--first-parent', '--reverse', '--format=%H%x00%s', rng
    )
    for line in out.splitlines():
        sha, _, subject = line.partition('\0')
        if LANDING_SUBJECT.match(subject):
            return f'{sha[:12]} {subject}'
    return None


def judging_problem(
    paths: Sequence[str],
    rules: judging.Rules | None,
    package: str,
    train: bool,
    first: bool,
    earlier: str | None,
) -> str | None:
    """Why a package changing ``paths`` may not land where it is (T3.0,
    item 1), or None. ``first``: it is the first package of its stack, on
    the phase head; ``earlier``: a landing of the phase before it, so it is
    not the phase's first package."""
    if rules is None:
        return None
    found = judging.judging(rules, paths)
    if train:
        if not found:
            return (
                f'{package} is landed as a train but changes no judging path; '
                'land it without --train'
            )
        if not first:
            return (
                f'{package} is a train, and a train lands at the front of a '
                'stack, on the phase head: restack it first (land.py restack '
                '--train-first) or land it before the others'
            )
        others = judging.not_train(rules, paths)
        if others:
            return (
                f'{package} is a train, which changes judging paths, their '
                'tests and docs only; it also changes '
                + ', '.join(others[:8])
                + (' ...' if len(others) > 8 else '')
                + '; move those into a product package'
            )
        return None
    if not found or earlier is None:
        return None
    return (
        f'{package} changes judging paths ('
        + ', '.join(found[:8])
        + (' ...' if len(found) > 8 else '')
        + f") and is not the phase's first package ({earlier} landed "
        'before it). Judging changes found mid-phase ride one tooling train: '
        'move them '
        'into the train package and land it at the front of the next stack '
        'with `land.py --stack --train --candidate <train> ...`, then stack '
        'this package on it'
    )


def rule_problems(
    tools: Tools, old: str, new: str, rules: judging.Rules | None, *where: str
) -> list[str]:
    """Each commit of ``old..new`` refused by the commit-msg rule: a product
    fix without its ``Rule:`` line (T3.0, item 4). None before the judging
    list exists (the rule starts with T3.0's landing)."""
    if rules is None:
        return []
    out = tools.git(
        *where, 'log', '--no-merges', '--format=%H%x00%B%x1e', f'{old}..{new}'
    )
    found = []
    for entry in out.split('\x1e'):
        sha, sep, message = entry.strip('\n').partition('\x00')
        if not sep:
            continue
        files = tools.git(
            *where,
            'diff-tree',
            '--no-commit-id',
            '--name-only',
            '-r',
            '-z',
            '--root',
            sha,
        )
        paths = [p for p in files.split('\0') if p]
        why = commit_msg.rule_problem(message, paths, rules)
        if why:
            found.append(f'{sha[:12]} {message.splitlines()[0]}: {why}')
    return found


#: Where each CI job looks, by its name's start, for a failed job whose
#: log names no file the stack changed: the stack's commits under these
#: paths are implicated instead.
JOB_AREAS = (
    (
        r'(rust-|native-|binding|wheel-smoke|build|engine-wheels|installed-wheel)',
        'rust/**',
    ),
    (
        r'(contract-python|cross-host|l4-goldens|l5-conversion)',
        'contract/**\nrust/**\npackages/**',
    ),
    (r'(test|engine|typecheck|runtime-install)', 'packages/**\ntests/**'),
    (r'(tooling|timing guard)', 'scripts/**\ncontract/**\ntests/**'),
    (r'(lint|hygiene)', '**'),
)


def implicated(
    tools: Tools, run: dict[str, object], base: str, top: str
) -> list[tuple[str, list[tuple[str, str]]]]:
    """For each failed job of ``run``, the commits of ``base..top`` that
    touch a path its log names, else a path in the job's area
    (:data:`JOB_AREAS`), as (job, [(sha, subject)])."""
    repo = ('--repo', tools.repo) if tools.repo else ()
    run_id = str(run.get('databaseId'))
    jobs = json.loads(
        tools.gh('run', 'view', run_id, '--json', 'jobs') or '{}'
    ).get('jobs', [])
    failed = [
        j
        for j in jobs
        if j.get('conclusion') in ('failure', 'timed_out')
    ]
    commits = []
    for line in tools.git(
        'log', '--no-merges', '--format=%H%x00%s', f'{base}..{top}'
    ).splitlines():
        sha, _, subject = line.partition('\0')
        if LANDING_SUBJECT.match(subject):
            continue
        files = tools.git(
            'diff-tree', '--no-commit-id', '--name-only', '-r', '-z', sha
        )
        commits.append((sha, subject, [p for p in files.split('\0') if p]))
    report = []
    for job in failed:
        log = tools.attempt(
            'gh',
            'run',
            'view',
            run_id,
            '--log-failed',
            '--job',
            str(job.get('databaseId')),
            *repo,
        ).stdout
        named = set(re.findall(r'[A-Za-z0-9_][\w./-]*\.[A-Za-z]{1,5}\b', log))
        hit = [
            (sha, subject)
            for sha, subject, files in commits
            if any(
                path in named or any(n.endswith('/' + path) for n in named)
                for path in files
            )
        ]
        if not hit:
            name = str(job.get('name'))
            area = next(
                (
                    judging.patterns(paths)
                    for start, paths in JOB_AREAS
                    if re.match(start, name)
                ),
                [],
            )
            hit = [
                (sha, subject)
                for sha, subject, files in commits
                if any(judging.matches(area, path) for path in files)
            ]
        report.append((str(job.get('name')), hit))
    return report


# --- the commands -------------------------------------------------------------


def land(
    args: argparse.Namespace, tools: Tools, out: Callable[[str], None]
) -> int:
    if args.wait_for_landing_run:
        raise Refusal(
            '--wait-for-landing-run is obsolete (decision E131): CI skips an '
            "empty landing commit's own runs, so that run would never come. "
            "A landing stands on the candidate's green exact-SHA run: wait "
            "for or rerun the candidate's run, then land without the flag"
        )
    check_branches(args.phase_branch, args.topic_branch)
    if args.stack and args.moved_from is not None:
        raise Refusal(
            '--stack and --moved-from: a stacked package stands on the '
            "stack top's run, not a moved-from commit's; restack a moved "
            'stack with land.py restack'
        )
    title = one_line(args.title, '--title')
    package = package_of(title)
    on_lane = lane_of(args.topic_branch) is not None
    if on_lane:
        check_lane_branch(tools, args.topic_branch)
        check_lane_package(tools, args.topic_branch, package)
    candidate = resolve(tools, args.candidate, 'the candidate')
    check_worktree(tools, candidate)
    base = phase_head(tools, args.remote, args.phase_branch)
    check_descent(tools, candidate, base, args.phase_branch)
    chain, chain_problems = stack_chain(
        tools, base, candidate, remote=args.remote
    )
    foreign = [link for link in chain if link.package != package]
    tail = base
    if args.stack:
        if chain_problems:
            raise Refusal(
                'the stack below the candidate is not sound:\n- '
                + '\n- '.join(chain_problems)
            )
        limit = stack_max(args.max_stack)
        if len(chain) >= limit:
            raise Refusal(
                f'the stack already holds {len(chain)} packages, the most one '
                f'CI run covers ({limit}; --max-stack or {STACK_MAX_ENV}); '
                'land it with land.py stack-land first'
            )
        tail = chain[-1].sha if chain else base
        if args.checkpoint is not None and (
            resolve(tools, args.checkpoint, 'the checkpoint') != tail
        ):
            raise Refusal(
                f'the checkpoint of a stacked package is the landing below '
                f'it, {tail[:12]}; omit --checkpoint'
            )
        checkpoint = tail
        stray = commits_not_owned(
            tools, tail, candidate, package, remote=args.remote
        )
        if stray:
            raise Refusal(
                f'commits above the stack are not owned by {package}:\n- '
                + '\n- '.join(f'{c[:12]} {t} ({w})' for c, t, w in stray)
            )
    elif foreign:
        raise Refusal(
            f"the candidate sits on a stack of other packages' landing "
            f'commits ({", ".join(link.package for link in foreign)}); add it '
            'with `land.py --stack ...`, then land the whole stack with '
            '`land.py stack-land --tip <its landing commit>`'
        )
    else:
        checkpoint = checkpoint_of(tools, candidate, base, args.checkpoint)
        if on_lane:
            check_lane_commits(tools, base, candidate, package, args.remote)
    # The judging-path rule (T3.0, item 1) and the fix rule (item 4), from
    # the phase head's list on (none before T3.0 lands it).
    rules = judging_rules(tools, base)
    paths = changed_paths(tools, tail, candidate) if rules is not None else []
    if args.train and rules is None:
        raise Refusal(
            f'the head of {args.phase_branch} has no {judging.FILE}, so no '
            'train: land without --train'
        )
    earlier = None
    if rules is not None and judging.judging(rules, paths) and not args.train:
        earlier = phase_first_landing(tools, args.remote, base) or next(
            (f'{link.sha[:12]} Land {link.package}' for link in chain), None
        )
    why = judging_problem(
        paths, rules, package, args.train, tail == base, earlier
    )
    if why:
        raise Refusal(why)
    unruled = rule_problems(tools, tail, candidate, rules)
    if unruled:
        raise Refusal(
            "fix commits that change product code without a 'Rule:' line "
            "(the old client's rule they port, one sentence):\n- "
            + '\n- '.join(unruled)
        )
    moved: str | None = None
    if args.moved_from is not None:
        moved = resolve(tools, args.moved_from, 'the moved-from commit')
        check_moved_from(tools, moved, candidate, base, args.phase_branch)
    # The evidence: the candidate's own, or, moved, the moved-from commit's.
    evidenced = moved or candidate
    what = 'the moved-from commit' if moved else 'the candidate'
    check_reviews(tools, args.review, evidenced, what)
    if args.stack:
        return stack_one(
            args, tools, out, title, package, candidate, base, tail, chain
        )
    if moved:
        # E130's order: X's evidence (its reviews, then its run), then the
        # same tree, the same patches and only landing commits in X..H.
        green = check_candidate_ci(tools, moved, what)
        move = check_moved(tools, moved, candidate, base, args.phase_branch)
    out(f'ok   worktree clean, HEAD is the candidate {candidate}')
    out(f'ok   the candidate descends from {args.phase_branch} at {base}')
    out(f'ok   the checkpoint is their merge base, {checkpoint}')
    if on_lane:
        out(
            f'ok   the lane is checked out here and carries {package}, the '
            'package this worktree records, which owns every commit'
        )
    if moved:
        out(
            f'ok   the candidate is {moved[:12]} moved over landing commits '
            f'only ({move.landings} landing commit(s)'
            + (
                f', {move.landed} of its own commits already landed as '
                'moved copies'
                if move.landed
                else ''
            )
            + f'): the same tree, commit by commit, for its {move.commits} '
            'commit(s)'
        )
    out(f'ok   {len(args.review)} reviews approve {evidenced[:12]}')
    if moved:
        out(f'ok   {describe(green)}, on {moved[:12]} itself')
        evidence = (
            f'Moved from {moved}: its run {green.get("databaseId")} is green '
            f'at {green.get("headSha")} and both reviews approve it. The '
            f'candidate {candidate} is that commit moved over landing commits '
            "only (each commit with its partner's tree in order, only "
            'landing commits added), so it lands on that evidence (E130); '
            'this commit changes no file, so the phase branch moves at once '
            'and CI skips its own runs (E43 (b), E131).'
        )
    else:
        green = check_candidate_ci(tools, candidate)
        out(f'ok   {describe(green)}, on {candidate[:12]} itself')
        evidence = (
            f'Candidate run {green.get("databaseId")} green at '
            f'{green.get("headSha") or candidate}; '
            'this commit changes no file, so the phase branch moves at once '
            'and CI skips its own runs (E43 (b), E131).'
        )
    reviewer = one_line(
        args.reviewer
        or f'independent reviewer (fresh agent) and Codex, {package}',
        '--reviewer',
    )
    summary = args.summary or (
        f'Both reviews approve {moved[:7]}, which the candidate '
        f'{candidate[:7]} is, moved over landing commits only.'
        if moved
        else f'Both reviews approve the candidate {candidate[:7]}.'
    )
    landing = Landing(
        title=title,
        reviewer=reviewer,
        protected_paths=one_line(args.protected_paths, '--protected-paths'),
        base=base,
        checkpoint=checkpoint,
        summary=summary.strip(),
        evidence=evidence,
        moved=(
            {
                'Moved-from': moved,
                'Moved-from-run': str(green.get('databaseId')),
            }
            if moved
            else {}
        ),
    )
    if args.dry_run:
        out('dry run: every check passed; the landing commit would read:\n')
        out(landing.message())
        out('dry run: nothing committed, nothing pushed')
        return 0
    sha = make_landing(tools, landing, candidate)
    out(f"ok   landing commit {sha}: empty, tree equal to the candidate's")
    topic_ref = f'{sha}:refs/heads/{args.topic_branch}'
    phase_ref = f'{sha}:refs/heads/{args.phase_branch}'
    undo = (
        f'the landing commit {sha[:12]} is made but was not pushed; before '
        f'running land.py again, undo it with `git reset --soft {candidate}`'
    )
    # One atomic push: the remote refuses both if the phase branch has
    # moved since the fetch above, since this is not a forced update.
    push(tools, undo, '--atomic', args.remote, topic_ref, phase_ref)
    out(
        f'ok   pushed {args.topic_branch} and fast-forwarded '
        f'{args.phase_branch} to {sha} (decision E43 (b))'
    )
    out(f'LANDED {sha}')
    status = 0
    if on_lane:
        mismatch = confirm_lane(
            tools, args.remote, args.topic_branch, args.phase_branch, sha
        )
        if mismatch:
            out(
                f'LANE MISMATCH: the remote has {mismatch}, not both at {sha}; '
                "the lane's next package starts from the phase head, so "
                'look before `lane.py start`'
            )
            status = 1
        else:
            out(f'ok   {args.topic_branch} equals {args.phase_branch} at {sha}')
    # One look, no wait: CI skips the landing commit's runs (E131), so they
    # are listed for their links only.
    listed: dict[str, dict[str, object]] = {}
    for run in runs_on(tools, sha):
        listed.setdefault(str(run.get('headBranch')), run)
    for branch in (args.topic_branch, args.phase_branch):
        run = listed.get(branch)
        out(
            f'run  {branch}: {run.get("databaseId")} {run.get("url")}'
            if run
            else f'run  {branch}: not yet listed (CI skips it, E131)'
        )
    out(f'next: land.py verify {sha}')
    return status


def stack_one(
    args: argparse.Namespace,
    tools: Tools,
    out: Callable[[str], None],
    title: str,
    package: str,
    candidate: str,
    base: str,
    tail: str,
    chain: list[Stacked],
) -> int:
    """``land.py --stack``: the package's landing commit, made on the
    candidate locally with the stack's tail as its Base-SHA and
    Checkpoint-SHA, pushing nothing (T3.0, item 2). Its CI evidence is the
    stack top's run, which ``stack-land`` checks."""
    out(f'ok   worktree clean, HEAD is the candidate {candidate}')
    out(
        f'ok   the candidate descends from {args.phase_branch} at {base}'
        + (
            f', on a stack of {len(chain)} '
            f'({", ".join(link.package for link in chain)})'
            if chain
            else ''
        )
    )
    out(f'ok   {len(args.review)} reviews approve {candidate[:12]}')
    reviewer = one_line(
        args.reviewer
        or f'independent reviewer (fresh agent) and Codex, {package}',
        '--reviewer',
    )
    extra = {STACK_BASE: base, 'Topic-branch': args.topic_branch}
    if args.train:
        extra[TRAIN] = 'yes'
    landing = Landing(
        title=title,
        reviewer=reviewer,
        protected_paths=one_line(args.protected_paths, '--protected-paths'),
        base=tail,
        checkpoint=tail,
        summary=(
            args.summary
            or f'Both reviews approve the candidate {candidate[:7]}.'
        ).strip(),
        evidence=(
            f'Stacked on {tail} ({len(chain)} package(s) below it, on the phase '
            f'head {base}): it lands with its stack, on the green run of the '
            "stack's top (land.py stack-land, T3.0), whose tree holds this "
            "one's; this commit changes no file."
        ),
        moved=extra,
    )
    if args.dry_run:
        out(
            'dry run: every check passed; the stacked landing commit would read:\n'
        )
        out(landing.message())
        out('dry run: nothing committed')
        return 0
    sha = make_landing(tools, landing, candidate)
    out(f'STACKED {sha}')
    out(
        f'next: build the next package on {sha[:12]}, or land the stack with '
        f'`land.py stack-land --tip {sha}` once the run on its top is green'
    )
    return 0


def stack_land(
    args: argparse.Namespace, tools: Tools, out: Callable[[str], None]
) -> int:
    """``land.py stack-land --tip SHA``: the stack from the phase head up to
    TIP, a landing commit made by ``--stack``, lands in one atomic push on
    one green run: the newest gating run on TIP's tested commit, whose tree
    holds every package below (T3.0, item 2). A red run is reported by the
    commits its failed jobs implicate, with the green prefix to land."""
    if args.topic_branch:
        check_branches(args.phase_branch, args.topic_branch)
    tip = resolve(tools, args.tip, 'the stack tip')
    base = phase_head(tools, args.remote, args.phase_branch)
    check_descent(tools, tip, base, args.phase_branch)
    chain, problems = stack_chain(tools, base, tip, remote=args.remote)
    if not chain or chain[-1].sha != tip:
        raise Refusal(
            f'{tip[:12]} is not the top landing commit of a stack on the '
            f'head of {args.phase_branch} ({base[:12]}); stack each package '
            'with `land.py --stack ...` and name the last landing commit'
        )
    if problems:
        raise Refusal('the stack is not sound:\n- ' + '\n- '.join(problems))
    limit = stack_max(args.max_stack)
    if len(chain) > limit:
        raise Refusal(
            f'the stack holds {len(chain)} packages, more than one CI run '
            f'covers ({limit}); land a prefix (--tip a lower landing)'
        )
    rules = judging_rules(tools, base)
    earlier = phase_first_landing(tools, args.remote, base) if rules else None
    refused: list[str] = []
    for n, link in enumerate(chain):
        paths = changed_paths(tools, link.below, link.sha)
        why = judging_problem(
            paths, rules, link.package, link.train, n == 0, earlier
        )
        if why:
            refused.append(why)
        refused.extend(rule_problems(tools, link.below, link.sha, rules))
        earlier = earlier or f'{link.sha[:12]} Land {link.package}'
    if refused:
        raise Refusal(
            'the stack breaks a landing rule:\n- ' + '\n- '.join(refused)
        )
    tested = tested_commit(tools, tip)
    runs = runs_on(tools, tested)
    if not runs:
        raise Refusal(
            f"no CI run on the stack top {tested[:12]} (the last package's "
            'top) that is a push under the phase branch; push it, for '
            f'example `git push {args.remote} {tested}:refs/heads/'
            f'{args.phase_branch}-{chain[-1].package.lower()}-stack`, and '
            'let its run finish'
        )
    newest = runs[0]
    if newest.get('status') != 'completed':
        raise Refusal(
            f'the run on the stack top is not finished ({describe(newest)}); '
            'wait for it'
        )
    if newest.get('conclusion') == 'cancelled':
        # Cancelled says nothing about the tree (Claude r1 P2-6): rerun.
        raise Refusal(
            f'the run on the stack top was cancelled ({describe(newest)}); '
            f'nothing failed: rerun it (`gh run rerun {newest.get("databaseId")}`) '
            'and land again'
        )
    if newest.get('conclusion') != 'success':
        out(
            f'RED  {describe(newest)} {newest.get("url")}, on the stack top {tested[:12]}'
        )
        report = implicated(tools, newest, base, tested)
        first_bad: int | None = None
        for job, hits in report:
            if not hits:
                out(
                    f'IMPLICATED {job}: no stack commit by path or area; read '
                    f'`gh run view {newest.get("databaseId")} --log-failed`'
                )
            for sha, subject in hits:
                out(f'IMPLICATED {job}: {sha[:12]} {subject}')
                for n, link in enumerate(chain):
                    if tools.git_ok(
                        'merge-base', '--is-ancestor', sha, link.sha
                    ):
                        first_bad = (
                            n if first_bad is None else min(first_bad, n)
                        )
                        break
        if first_bad is None:
            out(
                'no package is implicated by path; if the failure is drift or a '
                f'flake, rerun it (`gh run rerun {newest.get("databaseId")} '
                '--failed`) and land again'
            )
        elif first_bad == 0:
            out(
                f'the first package, {chain[0].package}, is implicated: no green prefix'
            )
        else:
            prefix = chain[first_bad - 1]
            ptested = tested_commit(tools, prefix.sha)
            out(
                f'PREFIX RUN: git push {args.remote} {ptested}:refs/heads/'
                f'{args.phase_branch}-{prefix.package.lower()}-prefix'
            )
            out(f'GREEN PREFIX: land.py stack-land --tip {prefix.sha}')
        return 1
    out(
        f'ok   the stack of {len(chain)} on {args.phase_branch} at {base[:12]} is sound'
    )
    for link in chain:
        out(f'ok   {link.sha[:12]} Land {link.package} on {link.below[:12]}')
    out(f'ok   {describe(newest)}, on the stack top {tested[:12]}')
    if args.dry_run:
        out('dry run: nothing pushed')
        return 0
    refs = [f'{tip}:refs/heads/{args.phase_branch}']
    if args.topic_branch:
        refs.append(f'{tip}:refs/heads/{args.topic_branch}')
    push(
        tools,
        'nothing was committed; the push changed nothing',
        '--atomic',
        args.remote,
        *refs,
    )
    out(f'ok   fast-forwarded {args.phase_branch} to {tip}')
    for link in chain:
        out(f'landed {link.sha} Land {link.package}')
    out(f'LANDED {tip}')
    status = 0
    if args.topic_branch and lane_of(args.topic_branch) is not None:
        mismatch = confirm_lane(
            tools, args.remote, args.topic_branch, args.phase_branch, tip
        )
        if mismatch:
            out(f'LANE MISMATCH: the remote has {mismatch}, not both at {tip}')
            status = 1
    out('next: land.py verify <each landing above>')
    return status


def replace_trailers(message: str, values: dict[str, str]) -> str:
    """``message`` with each trailer of ``values`` set to its value."""
    lines = message.rstrip('\n').split('\n')
    seen = set()
    for n, line in enumerate(lines):
        key = line.partition(': ')[0]
        if key in values and line.startswith(key + ': '):
            lines[n] = f'{key}: {values[key]}'
            seen.add(key)
    missing = [f'{k}: {v}' for k, v in values.items() if k not in seen]
    return '\n'.join(lines + missing) + '\n'


def restack(
    args: argparse.Namespace, tools: Tools, out: Callable[[str], None]
) -> int:
    """``land.py restack --tip SHA``: moves a stack onto the phase head as
    it is now, keeping its reviews when every patch is the same (T3.0,
    item 2; move_confirm.py's rule, by patch-id). Each package's commits
    are cherry-picked in a scratch worktree, in order (the train first with
    ``--train-first``), each new commit's patch-id must be its old one's,
    and each landing commit is made again with the new Base-SHA,
    Checkpoint-SHA and Stack-base. A conflict or a changed patch refuses and
    changes nothing: that package needs a review round. With
    ``--move-branches`` each package's Topic-branch, where the remote holds
    the package's old top, moves to its new top as ``auto_land.py`` moves a
    rebased lane: the old tip kept under an archive branch, the branch
    pushed with a lease on the old tip. The new top needs its own CI run."""
    tip = resolve(tools, args.tip, 'the stack tip')
    base = phase_head(tools, args.remote, args.phase_branch)
    old_base = tools.git('merge-base', tip, base).strip()
    if old_base == base:
        raise Refusal(
            f'the stack is already on the head of {args.phase_branch}'
        )
    chain, problems = stack_chain(tools, old_base, tip, remote=args.remote)
    if not chain or chain[-1].sha != tip:
        raise Refusal(f'{tip[:12]} is not the top landing commit of a stack')
    if problems:
        raise Refusal('the stack is not sound:\n- ' + '\n- '.join(problems))
    order = list(chain)
    if args.train_first:
        trains = [link for link in order if link.train]
        order = trains + [link for link in order if not link.train]
    scratch = Path(tempfile.mkdtemp(prefix='sdm-restack-'))
    where = str(scratch / 'tree')
    tools.git('worktree', 'add', '--quiet', '--detach', where, base)
    moved: list[tuple[Stacked, str, str]] = []
    try:
        current = base
        for link in order:
            olds = tools.git(
                'rev-list',
                '--reverse',
                '--no-merges',
                f'{link.below}..{link.sha}^',
            ).split()
            for old in olds:
                done = tools.attempt('git', '-C', where, 'cherry-pick', old)
                if done.returncode != 0:
                    tools.attempt('git', '-C', where, 'cherry-pick', '--abort')
                    raise Refusal(
                        f'{link.package}: {old[:12]} does not apply on the new '
                        'head without a conflict; it needs a normal move and '
                        'review round'
                    )
            news = tools.git(
                '-C', where, 'rev-list', '--reverse', f'{current}..HEAD'
            ).split()

            def ids(shas: list[str]) -> list[str]:
                found = []
                for sha in shas:
                    patch = tools.git('show', '--format=', sha)
                    done = subprocess.run(
                        ['git', 'patch-id', '--stable'],
                        input=patch,
                        capture_output=True,
                        text=True,
                    )
                    found.append((done.stdout.split() or [''])[0])
                return found

            if len(news) != len(olds) or ids(news) != ids(olds):
                raise Refusal(
                    f'{link.package}: the move changed a patch; it needs a '
                    'review round on the new SHA'
                )
            top = tools.git('-C', where, 'rev-parse', 'HEAD').strip()
            message = replace_trailers(
                tools.git('log', '-1', '--format=%B', link.sha),
                {
                    'Base-SHA': current,
                    'Checkpoint-SHA': current,
                    STACK_BASE: base,
                    # The audit trail (Claude r1 P3-9): the landing this one
                    # replaces, whose reviews carry over by patch-id.
                    'Restacked-from': link.sha,
                },
            )
            note = Path(scratch / 'message.txt')
            note.write_text(message)
            tools.git(
                '-C',
                where,
                'commit',
                '--allow-empty',
                '--quiet',
                '-F',
                str(note),
            )
            current = tools.git('-C', where, 'rev-parse', 'HEAD').strip()
            moved.append((link, top, current))
    finally:
        tools.attempt('git', 'worktree', 'remove', where)
    for link, top, landing in moved:
        out(
            f'moved {link.package}: {link.sha[:12]} -> {landing} (top {top[:12]})'
        )
        branch = link.trailers.get('Topic-branch', '')
        if not (args.move_branches and branch):
            continue
        old_top = tools.git('rev-parse', f'{link.sha}^').strip()
        held = tools.git(
            'ls-remote', args.remote, f'refs/heads/{branch}'
        ).split()
        if not held or held[0] not in (old_top, link.sha):
            out(
                f"left {branch}: the remote does not hold {link.package}'s old top"
            )
            continue
        archive = phase_named(args.phase_branch).archive_prefix + (
            f'{branch.rsplit("-", 1)[-1]}-{link.package}-{held[0][:8]}'
        )
        push(
            tools,
            f'{branch} was not moved',
            '--atomic',
            f'--force-with-lease=refs/heads/{branch}:{held[0]}',
            args.remote,
            f'{held[0]}:refs/heads/{archive}',
            f'{top}:refs/heads/{branch}',
        )
        out(f'ok   {branch} moved to {top[:12]}; the old tip kept as {archive}')
    new_tip = moved[-1][2]
    out(f'RESTACKED {new_tip}')
    out(
        f'next: push the new top {moved[-1][1][:12]} for its CI run, then '
        f'`land.py stack-land --tip {new_tip}`'
    )
    return 0


def push(tools: Tools, undo: str, *args: str) -> None:
    # --no-verify: the pre-push hook would run prepush.sh's whole cheap tier
    # again. On a lane, land.py has made the hook's lane checks itself
    # before this push: the lane is the branch checked out
    # (check_lane_branch), the worktree records the title's package
    # (check_lane_package), and that package owns every commit since the
    # phase head (check_lane_commits). The hook's other checks (cargo,
    # clippy, the scans) are the candidate's own CI run, which land.py
    # requires green or waits for.
    try:
        tools.git('push', '--quiet', '--no-verify', *args)
    except Refusal as why:
        raise Refusal(f'{why}\n{undo}') from None


def revert_instructions(
    sha: str, base: str | None, phase: str, remote: str
) -> str:
    lines = [
        'Nothing was reverted. Decide first whether the failure is the change '
        'or drift outside it (a runner image, an unpinned tool, a registry):',
        '  gh run view <run> --log-failed',
        '  gh run rerun <run> --failed      # drift or a flake: rerun, then verify again',
        'If the change is at fault and the phase branch is still at the landing,',
        'put it back to the previous head:',
        f'  git push --force-with-lease=refs/heads/{phase}:{sha} {remote} '
        f'{base or "<Base-SHA>"}:refs/heads/{phase}',
        'If another package has landed on top since, revert the range on a topic',
        'branch instead and land that:',
        f'  git revert --no-edit {base or "<Base-SHA>"}..{sha}',
    ]
    return '\n'.join(lines)


def evidence_problem(
    tools: Tools, runs: list[dict[str, object]], sha: str, what: str
) -> tuple[str | None, dict[str, object] | None]:
    """Why ``sha``'s newest gating run is no evidence, or None and that run.

    Only a push run on the phase branch or a topic branch under it counts
    (:func:`gating`); a cancelled or unfinished one is reported as such, so
    the caller can ask for a rerun rather than a revert."""
    if not runs:
        return (
            f'no CI run on {what} {sha[:12]} itself that is a push to the '
            "phase branch or a topic branch under it (a pull request's run "
            'skips the gate and does not count)',
            None,
        )
    newest = runs[0]
    if (
        newest.get('status') == 'completed'
        and newest.get('conclusion') == 'success'
    ):
        return None, newest
    return f'the newest CI run on {what} is {describe(newest)}', newest


def verify(
    args: argparse.Namespace, tools: Tools, out: Callable[[str], None]
) -> int:
    """A landing commit verified on the run it landed on (decision E131).

    CI skips an empty landing commit's own runs: its tree is the
    candidate's, which ran on its exact SHA. So the evidence is the
    candidate's newest gating run (the landing commit's parent), or, for a
    moved landing (E130), the newest gating run on its ``Moved-from``
    commit, whose tree must be the landing's. A run on the landing commit
    itself is reported and need not exist; one that ran and failed still
    fails, since CI should have skipped it."""
    phase_named(args.phase_branch)
    sha = resolve(tools, args.landing, 'the landing commit')
    trailers = trailers_of(tools, sha)
    base = trailers.get('Base-SHA')
    tree = tools.git('rev-parse', sha + '^{tree}').strip()
    parent = tools.git('rev-parse', sha + '^').strip()
    parent_tree = tools.git('rev-parse', sha + '^^{tree}').strip()
    problems: list[str] = []
    if tree != parent_tree:
        problems.append(
            f'{sha[:12]} changes files; a landing commit is empty (E25)'
        )
    for key in ('Reviewed-by', 'Protected-paths', 'Base-SHA', 'Checkpoint-SHA'):
        if key not in trailers:
            problems.append(f'{sha[:12]} has no {key} trailer')
    # The evidence: the candidate's run, or the moved-from commit's.
    evidenced, what = parent, 'the candidate'
    if 'Moved-from' in trailers:
        moved = resolve_quiet(tools, trailers['Moved-from'])
        if moved is None:
            problems.append(
                f'its Moved-from commit {trailers["Moved-from"][:12]} is not '
                'in this repository; fetch it'
            )
        elif tools.git('rev-parse', moved + '^{tree}').strip() != tree:
            problems.append(
                f"{sha[:12]}'s tree is not its Moved-from commit "
                f"{moved[:12]}'s, so that commit's run is no evidence for it"
            )
        evidenced, what = moved or '', 'the moved-from commit'
    if evidenced:
        # An empty landing commit's run tested nothing (E131).
        tested = tested_commit(tools, evidenced)
        what, evidenced = what + below(evidenced, tested), tested
    # The landing is on the phase branch: verify stands for what landed.
    head = phase_head(tools, args.remote, args.phase_branch)
    if not tools.git_ok('merge-base', '--is-ancestor', sha, head):
        problems.append(
            f'{sha[:12]} is not in {args.phase_branch} (at {head[:12]}); '
            'verify checks a landing the phase branch holds'
        )
    if evidenced and not problems and STACK_BASE in trailers:
        # A stacked landing (T3.0) stands on its stack top's run when its
        # own candidate has none green: the first later landing of the same
        # stack whose tested commit's run is green, whose tree holds this
        # one's (it descends from it).
        own, _ = evidence_problem(
            tools, runs_on(tools, evidenced), evidenced, what
        )
        if own is not None:
            later = tools.git(
                'log',
                '--first-parent',
                '--reverse',
                '--format=%H%x00%s',
                f'{sha}..{head}',
            ).splitlines()
            for line in later:
                above, _, subject = line.partition('\0')
                if not LANDING_SUBJECT.match(subject):
                    continue
                if (
                    trailers_of(tools, above).get(STACK_BASE)
                    != trailers[STACK_BASE]
                ):
                    continue
                top = tested_commit(tools, above)
                why_top, _ = evidence_problem(
                    tools, runs_on(tools, top), top, 'the stack top'
                )
                if why_top is None:
                    evidenced = top
                    what = (
                        f'the stack top (landed with {above[:12]}), whose tree '
                        f"holds {sha[:12]}'s, at"
                    )
                    break
    unresolved: list[dict[str, object]] = []
    green: dict[str, object] | None = None
    if evidenced and not problems:
        why, run = evidence_problem(
            tools, runs_on(tools, evidenced), evidenced, what
        )
        if why is None and run is not None:
            green = run
            out(
                f'ok   {describe(run)} {run.get("url")}, on {what} '
                f'{evidenced[:12]}, whose tree the landing commit has (E131)'
            )
        elif run is not None and (
            run.get('status') != 'completed'
            or run.get('conclusion') == 'cancelled'
        ):
            out(f'CANC {why}')
            unresolved.append(run)
        else:
            problems.append(why or f'no evidence on {what}')
    # The landing commit's own runs. CI skips an ordinary landing's, so
    # they are reported, never required; one that ran and failed is real,
    # and fails. A moved landing's are not skipped (E131 (c)): one of them,
    # green, is its judgement against its own Base-SHA, which the
    # moved-from commit's run never made, so without one it is incomplete.
    moved_landing = 'Moved-from' in trailers
    own_green: list[dict[str, object]] = []
    for run in runs_on(tools, sha):
        final = run
        if run.get('status') != 'completed':
            out(f'wait {describe(run)} {run.get("url")}')
            final = watch(tools, run.get('databaseId'))
        conclusion = final.get('conclusion')
        if final.get('status') != 'completed':
            out(
                f'WAIT {describe(final)} {final.get("url")} (the landing commit)'
            )
        elif conclusion == 'success':
            own_green.append(final)
            out(
                f'ok   {describe(final)} {final.get("url")} (the landing commit)'
            )
        elif conclusion == 'cancelled':
            # Cancelled says nothing about the tree; the run it landed on
            # does.
            out(
                f'CANC {describe(final)} {final.get("url")} (the landing '
                'commit)'
                + (
                    f'; the same tree is green in run {green.get("databaseId")}'
                    f' at {evidenced[:12]}'
                    if green
                    else ''
                )
            )
        elif conclusion in ('skipped', 'neutral'):
            out(
                f'ok   {describe(final)} {final.get("url")} (the landing commit)'
            )
        else:
            out(f'FAIL {describe(final)} {final.get("url")}')
            problems.append(f'{describe(final)}, on the landing commit itself')
    if problems:
        out('VERIFY FAILED:\n- ' + '\n- '.join(problems))
        out(revert_instructions(sha, base, args.phase_branch, args.remote))
        return 1
    if unresolved:
        out(
            f'VERIFY INCOMPLETE: the newest run on {what} was cancelled or '
            'has not finished; nothing failed, so rerun or wait rather than '
            'revert:'
        )
        for run in unresolved:
            out(f'  gh run rerun {run.get("databaseId")}   # then verify again')
        return 3
    if moved_landing and not own_green:
        out(
            f'VERIFY INCOMPLETE: {sha[:12]} is a moved landing (Moved-from), '
            'which CI runs to judge it against its own Base-SHA, and no run '
            'of it on a push is green yet; nothing failed, so wait for its '
            'runs, or rerun a cancelled one, and verify again'
        )
        return 3
    out(
        f'VERIFIED {sha}: the run it landed on is green, and it changes no file'
    )
    return 0


def repo_of(tools: Tools, remote: str) -> str | None:
    done = tools.attempt('git', 'remote', 'get-url', remote)
    match = re.search(
        r'github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$', done.stdout.strip()
    )
    return match.group(1) if match else None


def parser_for(command: str) -> argparse.ArgumentParser:
    if command == 'verify':
        parser = argparse.ArgumentParser(
            prog='land.py verify',
            description='Waits for the runs on a landing commit and reports them.',
        )
        parser.add_argument('landing', metavar='LANDING_SHA')
        # Accepted for older command lines: verify no longer waits for the
        # landing commit's runs to appear, since CI skips them (E131).
        parser.add_argument(
            '--appear-timeout', type=float, default=0, help=argparse.SUPPRESS
        )
    elif command in ('stack-land', 'restack'):
        parser = argparse.ArgumentParser(
            prog=f'land.py {command}',
            description=(
                stack_land if command == 'stack-land' else restack
            ).__doc__.split('\n\n', 1)[0],
        )
        parser.add_argument(
            '--tip', required=True, help='the top landing commit of the stack'
        )
        if command == 'stack-land':
            parser.add_argument(
                '--topic-branch',
                help="also fast-forward this branch (the stack top's) to the tip",
            )
            parser.add_argument('--max-stack', type=int)
            parser.add_argument('--dry-run', action='store_true')
        else:
            parser.add_argument(
                '--train-first',
                action='store_true',
                help='move the train package to the front of the stack',
            )
            parser.add_argument(
                '--move-branches',
                action='store_true',
                help="move each package's Topic-branch to its new top",
            )
    else:
        parser = argparse.ArgumentParser(
            prog='land.py', description=__doc__.split('\n', 1)[0]
        )
        parser.add_argument('--candidate', required=True)
        parser.add_argument('--topic-branch', required=True)
        parser.add_argument(
            '--review', action='append', default=[], required=True
        )
        parser.add_argument('--title', required=True)
        parser.add_argument('--protected-paths', required=True)
        parser.add_argument(
            '--checkpoint',
            help='optional cross-check; land.py computes the merge base itself',
        )
        parser.add_argument('--reviewer')
        parser.add_argument('--summary')
        parser.add_argument('--dry-run', action='store_true')
        # Refused as obsolete (E131), with the reason, rather than dropped,
        # so an old command line learns why.
        parser.add_argument(
            '--wait-for-landing-run',
            action='store_true',
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            '--moved-from',
            metavar='SHA',
            help="land on this commit's reviews and green run: the "
            'candidate is it moved over landing commits only (E130)',
        )
        parser.add_argument(
            '--stack',
            action='store_true',
            help='add the package to the stack: make its landing commit on '
            'the stack tail locally, needing no CI run and pushing nothing; '
            'land.py stack-land lands the stack (T3.0)',
        )
        parser.add_argument(
            '--train',
            action='store_true',
            help='the package is the tooling train: judging paths, their '
            'tests and docs only, first in its stack (T3.0)',
        )
        parser.add_argument(
            '--max-stack',
            type=int,
            help=f'the most packages one stack holds (default {STACK_MAX}, '
            f'or {STACK_MAX_ENV})',
        )
    # Resolved in main: the flag, SDM_PHASE_BRANCH, the topic branch's
    # phase (landing), else the phase in progress (phases.py).
    parser.add_argument('--phase-branch')
    parser.add_argument('--remote', default='origin')
    parser.add_argument(
        '--repo', help='OWNER/NAME for gh; read from the remote'
    )
    return parser


def landing_phase(tools: Tools, remote: str, landing: str) -> str:
    """The phase a landing commit landed on, for ``verify`` given no
    phase: the lowest-numbered phase branch on the remote that holds it
    (a later phase holds an earlier one's landings through the epic
    branch's merges). None holding it is a refusal, by name."""
    sha = resolve(tools, landing, 'the landing commit')
    out = tools.git('ls-remote', remote, 'refs/heads/rust-core-phase-*')
    tips = {}
    for line in out.splitlines():
        tip, _, ref = line.partition('\t')
        name = ref.removeprefix('refs/heads/')
        if PHASE_BRANCH.fullmatch(name):
            tips[name] = tip.strip()
    for name in sorted(tips, key=lambda n: Phase(n).number):
        tools.git('fetch', '--quiet', remote, f'refs/heads/{name}')
        if tools.git_ok('merge-base', '--is-ancestor', sha, tips[name]):
            return name
    raise Refusal(
        f'no phase branch on {remote} holds {sha[:12]}: name its phase with '
        '--phase-branch'
    )


def main(
    argv: Sequence[str] | None = None,
    tools: Tools | None = None,
    out: Callable[[str], None] = print,
) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = (
        argv[0]
        if argv[:1] in (['verify'], ['stack-land'], ['restack'])
        else 'land'
    )
    args = parser_for(command).parse_args(
        argv if command == 'land' else argv[1:]
    )
    tools = tools or Tools()
    tools.repo = args.repo or tools.repo or repo_of(tools, args.remote)
    try:
        try:
            if (
                command == 'verify'
                and args.phase_branch is None
                and not os.environ.get(ENVIRONMENT)
            ):
                # The landing's own phase, never the phase in progress.
                args.phase_branch = landing_phase(
                    tools, args.remote, args.landing
                )
            args.phase_branch = phase_branch(
                args.phase_branch,
                branch=getattr(args, 'topic_branch', None),
            )
        except PhaseError as error:
            raise Refusal(
                f'{error}; land.py lands only on a phase branch'
            ) from error
        tools.phase = phase_named(args.phase_branch)
        commands = {
            'verify': verify,
            'land': land,
            'stack-land': stack_land,
            'restack': restack,
        }
        return commands[command](args, tools, out)
    except Refusal as why:
        out(f'REFUSED: {why}')
        return 2


if __name__ == '__main__':
    sys.exit(main())
