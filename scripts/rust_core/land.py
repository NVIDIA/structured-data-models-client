#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lands a reviewed package on the phase branch, so no one lands by hand.

Decision E43 (b) and the design's "One gate run per work package": the
only commit after approval is an empty landing commit carrying the
trailers (E25), and the phase branch fast-forwards to it at once when its
tree is the candidate's and the candidate's own exact-SHA CI run
concluded success. The landing commit's runs are recorded afterwards
(``land.py verify``), not waited on.

    land.py --candidate SHA --topic-branch NAME --review FILE --review FILE \\
        --title 'T1.12: the S0 case corpus' --protected-paths TEXT \\
        [--checkpoint SHA] [--phase-branch rust-core-phase-1] \\
        [--reviewer TEXT] [--summary TEXT] [--dry-run]
        [--wait-for-landing-run | --moved-from SHA]
    land.py verify LANDING_SHA [--phase-branch rust-core-phase-1]

Before it changes anything it refuses, naming the reason, unless:

- the phase branch is exactly ``rust-core-phase-1`` and the topic branch a
  package branch under it (``rust-core-phase-1-<name>``, a lane
  ``rust-core-phase-1-lane-<x>`` included), never ``main``, ``rust-core``
  or anything else; a name under ``rust-core-phase-1-lane-`` must be a
  lane, ``<x>`` lower-case letters and digits, and an archive branch
  (``rust-core-phase-1-archive-...``) is refused;
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
  on the lane by a red ``--wait-for-landing-run``: subject ``Land <ID>:
  ...``, one parent, no file changed, and a ``Base-SHA`` in the phase
  branch. A fix builds on it, and the new landing commit's parent is that
  fix and its ``Base-SHA`` the current phase head, so the phase branch
  still fast-forwards;
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
  as evidence, here, for the landing runs and in ``verify``: a run of
  another event (a pull request's) skips the checkpoint's gate, which CI
  runs only on such pushes (T1.20, Codex P2).

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
prints the landing SHA and the landing runs once they appear.

The landing commit's message names the CI evidence it lands on: the
candidate's green run, by id and SHA, or, on the other path, the
candidate's run that was not green and the landing run that must be.

``--wait-for-landing-run`` is the decision's other path, for a candidate
whose run is not green: the CI check is reported rather than required, the
landing commit is pushed to the topic branch only, and the phase branch is
fast-forwarded only after the landing commit's own run concludes success.

``verify`` waits for the runs on a landing SHA (the topic branch's and the
phase branch's) and reports each. A cancelled run is reported as cancelled,
not failed: it passes when another run on the same SHA, or on the
candidate, whose tree the landing commit shares, is green, and otherwise
asks for a rerun. On a failure it prints how to revert the fast-forward and
changes nothing: it never reverts by itself.

``--moved-from X`` (decision E120, the owner's of 2026-10-14) lands a
candidate C on X's reviews and X's green run, with no reviewer re-run and
no new CI, when C is X moved over landing commits only. Every check above
still applies to C (the branches, the lane and its commits' ownership,
the worktree and HEAD, the descent from the phase head H, the checkpoint,
and, at landing, the protected paths, the atomic push and the read-back),
but the review and CI checks are made for X in C's place: two reviews
naming X and the newest gating run on X's exact SHA green. It further
refuses, naming the condition, unless:

- same code: C's tree is X's;
- same patches: the commits in ``H..C`` have, in order, the same ``git
  patch-id --stable`` values as those in ``H..X``: the same count, the same
  order, and no commit without a patch of its own (an empty commit or a
  merge) on either side;
- only landing commits between: ``X..H`` is not empty, and each commit in
  it has one parent and its parent's tree and a subject starting
  ``Land ``.

X that is C itself, or is already in the phase branch, is refused, and so
is ``--moved-from`` with ``--wait-for-landing-run``, which lands without a
green run. The landing commit names X, X's green run by id and SHA, and
that C was moved over landing commits only; its trailers are unchanged.

``--dry-run`` makes every check, prints the landing commit it would make,
and commits and pushes nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

PHASE_BRANCH = 'rust-core-phase-1'
#: A package's topic branch: a branch under the phase branch, a lane
#: (``rust-core-phase-1-lane-a``) or a task (``rust-core-phase-1-t1.12``).
TOPIC_BRANCH = re.compile(
    re.escape(PHASE_BRANCH) + r'-[A-Za-z0-9](?:[A-Za-z0-9_-]|\.(?!\.))*'
)
#: A lane branch is this prefix and a lane name (decision E43, CI2).
LANE_PREFIX = PHASE_BRANCH + '-lane-'
LANE_NAME = re.compile(r'[a-z0-9]{1,16}')
#: Where the tip of an abandoned lane package is kept, by hand, on the
#: owner's decision; never a topic branch (CI2).
ARCHIVE_PREFIX = PHASE_BRANCH + '-archive-'
#: A package: a task (``T1.11e``) or a process package (``CI3``).
PACKAGE_ID = re.compile(r'T[0-9]+\.[0-9]+[a-z]?|CI[0-9]+')
#: The package that owns a commit: its subject starts with the ID, then a
#: colon or a space (``T1.11e: ...``, ``T1.11e fix 1: ...``,
#: ``T1.11e tracker: ...``).
SUBJECT_OWNER = re.compile(rf'({PACKAGE_ID.pattern})(?=[: ])')
#: Any package ID in a subject, even glued to letters before it
#: (``xT1.11f``), so a subject naming a second package is refused.
SUBJECT_ID = re.compile(rf'(?<![0-9.])({PACKAGE_ID.pattern})(?![A-Za-z0-9])')
#: An earlier landing commit's subject, left on the lane when its landing
#: run was red (``--wait-for-landing-run``); owned only when it is what
#: land.py makes (see ``landing_problem``).
LANDING_SUBJECT = re.compile(rf'Land ({PACKAGE_ID.pattern}): ')
#: Where ``lane.py start`` records a lane worktree's package.
PACKAGE_KEY = 'sdm.package'
WORKFLOW = 'ci.yml'
APPROVE = 'APPROVE'
MIN_PREFIX = 7
#: How long to wait for a pushed commit's runs to appear, in seconds.
APPEAR_TIMEOUT = 180
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
    # surrogateescape: a byte that is not text (a patch's, for
    # ``--moved-from``) is kept, not a crash, and goes back out unchanged.
    proc = subprocess.run(
        list(argv), capture_output=True, text=True, errors='surrogateescape'
    )
    return Completed(proc.returncode, proc.stdout, proc.stderr)


def run_command_input(argv: Sequence[str], text: str) -> Completed:
    proc = subprocess.run(
        list(argv),
        input=text,
        capture_output=True,
        text=True,
        errors='surrogateescape',
    )
    return Completed(proc.returncode, proc.stdout, proc.stderr)


@dataclass
class Tools:
    """git and gh, through one function, so the tests can replace both."""

    run: Run = run_command
    #: A command reading its input, ``git patch-id`` (``--moved-from``).
    run_input: Callable[[Sequence[str], str], Completed] = run_command_input
    repo: str | None = None
    sleep: Callable[[float], None] = time.sleep
    calls: list[tuple[str, ...]] = field(default_factory=list)

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

    def git_input(self, text: str, *args: str) -> str:
        """Runs git with ``text`` as its input; a failure is refused."""
        argv = ('git', *args)
        self.calls.append(argv)
        done = self.run_input(argv, text)
        if done.returncode != 0:
            raise Refusal(
                f'`{" ".join(argv)}` failed ({done.returncode}): '
                f'{(done.stderr or done.stdout).strip()}'
            )
        return done.stdout

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


def check_branches(phase: str, topic: str) -> None:
    """Refuses a destination other than the phase branch and a package
    branch under it, before anything is fetched, committed or pushed."""
    if phase != PHASE_BRANCH:
        raise Refusal(
            f'the phase branch is {phase!r}; land.py lands only on '
            f'{PHASE_BRANCH!r}'
        )
    if topic in ('main', 'master', 'rust-core', PHASE_BRANCH):
        raise Refusal(
            f'the topic branch is {topic!r}; a landing pushes a package branch '
            f'({PHASE_BRANCH}-<name>, a lane {PHASE_BRANCH}-lane-<x>) and the '
            'phase branch, never main, rust-core or the phase branch as topic'
        )
    if not TOPIC_BRANCH.fullmatch(topic) or topic.endswith(('.', '.lock')):
        raise Refusal(
            f'the topic branch {topic!r} is not a package branch '
            f'({PHASE_BRANCH}-<name> or {PHASE_BRANCH}-lane-<x>)'
        )
    if topic.startswith(ARCHIVE_PREFIX):
        raise Refusal(
            f'the topic branch {topic!r} is an archive branch, which keeps an '
            'abandoned package and never lands'
        )
    if topic.startswith(LANE_PREFIX) and lane_of(topic) is None:
        raise Refusal(
            f'the topic branch {topic!r} is under {LANE_PREFIX} but is not a '
            f'lane; a lane is {LANE_PREFIX}<x>, <x> lower-case letters and '
            'digits'
        )


def lane_of(branch: str) -> str | None:
    """The lane's name when ``branch`` is a lane branch, else None."""
    if not branch.startswith(LANE_PREFIX):
        return None
    name = branch.removeprefix(LANE_PREFIX)
    return name if LANE_NAME.fullmatch(name) else None


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
    against). Such a commit stays on the lane when its landing run was
    red; the package's fixes build on it and land again."""
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


def commits_not_owned(
    tools: Tools, base: str, head: str, package: str, *where: str
) -> list[tuple[str, str, str]]:
    """The commits in ``base..head`` that ``package`` does not own, each as
    (SHA, subject, why). An earlier landing commit of ``package`` that
    ``landing_problem`` accepts is owned."""
    out = tools.git(*where, 'log', '--format=%H%x00%s%x1e', f'{base}..{head}')
    found: list[tuple[str, str, str]] = []
    for entry in out.split('\x1e'):
        sha, sep, text = entry.strip('\n').partition('\x00')
        subject = text.split('\n', 1)[0]
        if not sep:
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
    tools: Tools, base: str, candidate: str, package: str
) -> None:
    stray = commits_not_owned(tools, base, candidate, package)
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


def gating(run: dict[str, object]) -> bool:
    """Whether a run is evidence: a push to the phase branch or a topic
    branch under it, never an archive branch, so CI ran the checkpoint's
    gate in it; a pull request's run on the same SHA skips the gate."""
    branch = str(run.get('headBranch'))
    under = branch == PHASE_BRANCH or (
        TOPIC_BRANCH.fullmatch(branch) is not None
        and not branch.startswith(ARCHIVE_PREFIX)
    )
    return run.get('event') == 'push' and under


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
        if r.get('headSha') == sha and gating(r)
    ]
    return sorted(runs, key=lambda r: str(r.get('createdAt', '')), reverse=True)


def describe(run: dict[str, object]) -> str:
    state = run.get('conclusion') or run.get('status')
    return f'run {run.get("databaseId")} on {run.get("headBranch")}: {state}'


def check_candidate_ci(
    tools: Tools, candidate: str, what: str = 'the candidate'
) -> dict[str, object]:
    """The newest gating run on ``candidate``'s exact SHA, when it is
    green; ``what`` names the commit in a refusal (the moved-from commit,
    under ``--moved-from``)."""
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
        advice = (
            'Wait for or rerun it, or land with --wait-for-landing-run'
            if what == 'the candidate'
            else 'Wait for or rerun it'
        )
        raise Refusal(
            f'the newest CI run on {what} is not green ({describe(newest)}); '
            f'a cancelled, failed or unfinished run does not count. {advice}'
        )
    return newest


# --- a candidate moved over landing commits (--moved-from) ------------------


def range_commits(tools: Tools, rng: str) -> list[tuple[str, str]]:
    """The commits in ``rng``, oldest first, each as (SHA, subject)."""
    out = tools.git('log', '--topo-order', '--format=%H%x00%s%x1e', rng)
    found: list[tuple[str, str]] = []
    for entry in out.split('\x1e'):
        sha, sep, text = entry.strip('\n').partition('\x00')
        if sep:
            found.append((sha, text.split('\n', 1)[0]))
    return found[::-1]


def patch_id(tools: Tools, sha: str) -> str | None:
    """``git patch-id --stable`` of the commit's own patch; None for a
    commit with no patch of its own (an empty commit, or a merge, which
    ``diff-tree`` shows no patch for)."""
    diff = tools.git('diff-tree', '--patch', '--binary', '--root', sha)
    out = tools.git_input(diff, 'patch-id', '--stable').split()
    return out[0] if out else None


def between_problem(tools: Tools, sha: str, subject: str) -> str | None:
    """None when ``sha`` is a landing commit: one parent, its parent's
    tree, and a subject starting with ``Land ``; else why not."""
    if not subject.startswith('Land '):
        return "the subject does not start with 'Land '"
    second = tools.attempt(
        'git', 'rev-parse', '--verify', '--quiet', f'{sha}^2'
    )
    if second.returncode == 0:
        return 'it has more than one parent'
    tree = tools.git('rev-parse', f'{sha}^{{tree}}').strip()
    parent = tools.git('rev-parse', f'{sha}^^{{tree}}').strip()
    if tree != parent:
        return "its tree is not its parent's: it changes files"
    return None


def check_moved(
    tools: Tools, moved: str, candidate: str, base: str, phase: str
) -> tuple[int, int]:
    """Refuses unless the candidate is ``moved`` (X) moved over landing
    commits only (decision E120): the same tree, the same patches in the
    same order since the phase head, and only landing commits between X
    and the phase head. Returns the number of patches and of landing
    commits between. X's reviews and CI are checked by the caller."""
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
    # 2. Same code.
    tree_x = tools.git('rev-parse', f'{moved}^{{tree}}').strip()
    tree_c = tools.git('rev-parse', f'{candidate}^{{tree}}').strip()
    if tree_x != tree_c:
        raise Refusal(
            f"{label}: not the same code: the candidate's tree "
            f"{tree_c[:12]} is not the moved-from commit's {tree_x[:12]}; "
            'a candidate whose code differs needs its own reviews and CI'
        )
    # 4. Only landing commits between X and the phase head.
    between = range_commits(tools, f'{moved}..{base}')
    if not between:
        raise Refusal(
            f'{label}: not moved over landing commits: the head of {phase} '
            f"{base[:12]} is already in the moved-from commit's history, so "
            'the candidate was moved over nothing; land the moved-from '
            'commit itself'
        )
    stray = [
        (sha, subject, why)
        for sha, subject in between
        if (why := between_problem(tools, sha, subject))
    ]
    if stray:
        raise Refusal(
            f'{label}: not only landing commits between the moved-from commit '
            f'and the head of {phase} (a landing commit has one parent, its '
            "parent's tree, and a subject starting 'Land '):\n- "
            + '\n- '.join(
                f'{sha[:12]} {subject} ({why})' for sha, subject, why in stray
            )
        )
    # 3. Same patches, in order, with no extra or empty commit.
    sides = {
        'candidate': range_commits(tools, f'{base}..{candidate}'),
        'moved-from commit': range_commits(tools, f'{base}..{moved}'),
    }
    ids: dict[str, list[str | None]] = {}
    empty: list[str] = []
    for side, commits in sides.items():
        ids[side] = [patch_id(tools, sha) for sha, _ in commits]
        empty += [
            f"{sha[:12]} {subject} (the {side}'s)"
            for (sha, subject), pid in zip(commits, ids[side], strict=True)
            if pid is None
        ]
    if empty:
        raise Refusal(
            f'{label}: not the same patches: a commit with no patch of its own '
            '(an empty commit or a merge) since the phase head:\n- '
            + '\n- '.join(empty)
        )
    ours, theirs = sides['candidate'], sides['moved-from commit']
    if len(ours) != len(theirs):
        raise Refusal(
            f'{label}: not the same patches: the candidate adds {len(ours)} '
            f'commit(s) to the head of {phase}, the moved-from commit '
            f'{len(theirs)}'
        )
    differ = [
        f'commit {n}: {c[0][:12]} {c[1]} against {x[0][:12]} {x[1]}'
        for n, (c, x, pc, px) in enumerate(
            zip(
                ours,
                theirs,
                ids['candidate'],
                ids['moved-from commit'],
                strict=True,
            ),
            start=1,
        )
        if pc != px
    ]
    if differ:
        raise Refusal(
            f"{label}: not the same patches: in order, these commits' "
            '`git patch-id --stable` differ:\n- ' + '\n- '.join(differ)
        )
    return len(ours), len(between)


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


def wait_for_runs(
    tools: Tools,
    sha: str,
    branches: Sequence[str],
    timeout: float = APPEAR_TIMEOUT,
) -> dict[str, dict[str, object]]:
    """The newest run on ``sha`` for each branch, waiting for them to
    appear; a branch whose run has not appeared in time is left out."""
    waited = 0.0
    while True:
        found: dict[str, dict[str, object]] = {}
        for run in runs_on(tools, sha):
            branch = str(run.get('headBranch'))
            if branch in branches and branch not in found:
                found[branch] = run
        if len(found) == len(branches) or waited >= timeout:
            return found
        tools.sleep(10)
        waited += 10


def watch(tools: Tools, run_id: object) -> dict[str, object]:
    """Waits for a run to finish and returns its final state."""
    repo = ('--repo', tools.repo) if tools.repo else ()
    # Its exit status is the run's; the state read next is what counts.
    tools.attempt('gh', 'run', 'watch', str(run_id), '--interval', '30', *repo)
    out = tools.gh('run', 'view', str(run_id), '--json', RUN_FIELDS)
    return json.loads(out)


# --- the commands -------------------------------------------------------------


def land(
    args: argparse.Namespace, tools: Tools, out: Callable[[str], None]
) -> int:
    check_branches(args.phase_branch, args.topic_branch)
    if args.moved_from is not None and args.wait_for_landing_run:
        raise Refusal(
            "--moved-from lands on the moved-from commit's green run, and "
            '--wait-for-landing-run lands without a green run; give one'
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
    checkpoint = checkpoint_of(tools, candidate, base, args.checkpoint)
    if on_lane:
        check_lane_commits(tools, base, candidate, package)
    moved: str | None = None
    if args.moved_from is not None:
        moved = resolve(tools, args.moved_from, 'the moved-from commit')
        patches, between = check_moved(
            tools, moved, candidate, base, args.phase_branch
        )
    # The evidence: the candidate's own, or, moved, the moved-from commit's.
    evidenced = moved or candidate
    what = 'the moved-from commit' if moved else 'the candidate'
    check_reviews(tools, args.review, evidenced, what)
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
            f'ok   the candidate is {moved[:12]} moved over {between} landing '
            f'commit(s) only: the same tree and the same {patches} patch(es) '
            'in order'
        )
    out(f'ok   {len(args.review)} reviews approve {evidenced[:12]}')
    if args.wait_for_landing_run:
        newest = (runs_on(tools, candidate) or [None])[0]
        state = (
            'no run'
            if newest is None
            else f'run {newest.get("databaseId")}: '
            f'{newest.get("conclusion") or newest.get("status")}'
        )
        out(
            f"note the candidate's CI ({state}) is not relied on; landing waits"
        )
        evidence = (
            f'Candidate CI not relied on ({state} at {candidate}); the phase '
            "branch moves only when this commit's own exact-SHA run is "
            'green (E43 (b), --wait-for-landing-run).'
        )
    elif moved:
        green = check_candidate_ci(tools, moved, what)
        out(f'ok   {describe(green)}, on {moved[:12]} itself')
        evidence = (
            f'Moved from {moved}: its run {green.get("databaseId")} is green '
            f'at {green.get("headSha")} and both reviews approve it. The '
            f'candidate {candidate} is that commit moved over landing commits '
            'only (the same tree, the same patches in order, only landing '
            'commits between), so it lands on that evidence (E120); this '
            'commit changes no file, so the phase branch moves at once and '
            'its own runs are recorded afterwards (E43 (b)).'
        )
    else:
        green = check_candidate_ci(tools, candidate)
        out(f'ok   {describe(green)}, on {candidate[:12]} itself')
        evidence = (
            f'Candidate run {green.get("databaseId")} green at {candidate}; '
            'this commit changes no file, so the phase branch moves at once '
            'and its own runs are recorded afterwards (E43 (b)).'
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
    if args.wait_for_landing_run:
        push(tools, undo, args.remote, topic_ref)
        out(f'ok   pushed {args.topic_branch}; waiting for the landing run')
        runs = wait_for_runs(tools, sha, [args.topic_branch])
        if args.topic_branch not in runs:
            raise Refusal(
                f'no landing run appeared on {sha[:12]}; {args.phase_branch} '
                'was not moved'
            )
        final = watch(tools, runs[args.topic_branch]['databaseId'])
        if final.get('conclusion') != 'success':
            raise Refusal(
                f'the landing run is not green ({describe(final)}); '
                f'{args.phase_branch} was not moved'
            )
        tools.git('push', '--quiet', '--no-verify', args.remote, phase_ref)
        out(
            f'ok   {describe(final)}; {args.phase_branch} fast-forwarded to {sha}'
        )
    else:
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
    runs = wait_for_runs(tools, sha, [args.topic_branch, args.phase_branch])
    for branch in (args.topic_branch, args.phase_branch):
        run = runs.get(branch)
        out(
            f'run  {branch}: {run.get("databaseId")} {run.get("url")}'
            if run
            else f'run  {branch}: not yet listed; `land.py verify {sha}` finds it'
        )
    out(f'next: land.py verify {sha}')
    return status


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


def verify(
    args: argparse.Namespace, tools: Tools, out: Callable[[str], None]
) -> int:
    if args.phase_branch != PHASE_BRANCH:
        raise Refusal(
            f'the phase branch is {args.phase_branch!r}, not {PHASE_BRANCH!r}'
        )
    sha = resolve(tools, args.landing, 'the landing commit')
    trailers = trailers_of(tools, sha)
    base = trailers.get('Base-SHA')
    tree = tools.git('rev-parse', sha + '^{tree}').strip()
    parent_tree = tools.git('rev-parse', sha + '^^{tree}').strip()
    problems: list[str] = []
    if tree != parent_tree:
        problems.append(
            f'{sha[:12]} changes files; a landing commit is empty (E25)'
        )
    for key in ('Reviewed-by', 'Protected-paths', 'Base-SHA', 'Checkpoint-SHA'):
        if key not in trailers:
            problems.append(f'{sha[:12]} has no {key} trailer')
    runs = wait_for_runs(
        tools, sha, [args.phase_branch], timeout=args.appear_timeout
    )
    every = {str(r.get('headBranch')): r for r in reversed(runs_on(tools, sha))}
    every.update(runs)
    if args.phase_branch not in every:
        problems.append(
            f'no CI run on {sha[:12]} on {args.phase_branch}; is the phase '
            'branch at the landing?'
        )
    cancelled: list[dict[str, object]] = []
    finals: list[dict[str, object]] = []
    for _branch, run in sorted(every.items()):
        final = run
        if run.get('status') != 'completed':
            out(f'wait {describe(run)} {run.get("url")}')
            final = watch(tools, run.get('databaseId'))
        finals.append(final)
        conclusion = final.get('conclusion')
        mark = {'success': 'ok  ', 'cancelled': 'CANC'}.get(
            str(conclusion), 'FAIL'
        )
        out(f'{mark} {describe(final)} {final.get("url")}')
        if conclusion == 'cancelled':
            cancelled.append(final)
        elif conclusion != 'success':
            problems.append(f'{describe(final)}')
    unresolved: list[dict[str, object]] = []
    if cancelled:
        # A cancelled run says nothing about the tree. Another green run on
        # the same SHA, or on the candidate, whose tree an empty landing
        # commit shares, is the evidence instead.
        green = [f for f in finals if f.get('conclusion') == 'success']
        if not green and tree == parent_tree:
            parent = tools.git('rev-parse', sha + '^').strip()
            green = [
                r
                for r in runs_on(tools, parent)
                if r.get('status') == 'completed'
                and r.get('conclusion') == 'success'
            ][:1]
        for run in cancelled:
            if green:
                out(
                    f'note {describe(run)} was cancelled; the same tree is green '
                    f'in run {green[0].get("databaseId")} at '
                    f'{str(green[0].get("headSha"))[:12]}'
                )
            else:
                unresolved.append(run)
    if problems:
        out('VERIFY FAILED:\n- ' + '\n- '.join(problems))
        out(revert_instructions(sha, base, args.phase_branch, args.remote))
        return 1
    if unresolved:
        out(
            'VERIFY INCOMPLETE: cancelled, with no green run on the same tree; '
            'nothing failed, so rerun rather than revert:'
        )
        for run in unresolved:
            out(f'  gh run rerun {run.get("databaseId")}   # then verify again')
        return 3
    out(f'VERIFIED {sha}: every run on it is green or stands for a green tree')
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
        parser.add_argument(
            '--appear-timeout', type=float, default=APPEAR_TIMEOUT
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
        parser.add_argument('--wait-for-landing-run', action='store_true')
        parser.add_argument(
            '--moved-from',
            metavar='SHA',
            help="land on this commit's reviews and green run: the "
            'candidate is it moved over landing commits only (E120)',
        )
    parser.add_argument('--phase-branch', default=PHASE_BRANCH)
    parser.add_argument('--remote', default='origin')
    parser.add_argument(
        '--repo', help='OWNER/NAME for gh; read from the remote'
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    tools: Tools | None = None,
    out: Callable[[str], None] = print,
) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = 'verify' if argv[:1] == ['verify'] else 'land'
    args = parser_for(command).parse_args(
        argv[1:] if command == 'verify' else argv
    )
    tools = tools or Tools()
    tools.repo = args.repo or tools.repo or repo_of(tools, args.remote)
    try:
        return (verify if command == 'verify' else land)(args, tools, out)
    except Refusal as why:
        out(f'REFUSED: {why}')
        return 2


if __name__ == '__main__':
    sys.exit(main())
