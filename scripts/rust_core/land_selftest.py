#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The self-tests of land.py: every check passes on a sound landing and
refuses each seeded fault, with git and gh replaced by a scripted fake.

Nothing here runs git or gh or reaches the network: ``Fake`` answers each
command land.py issues from a small model of a repository (commits with
parents and trees, ancestry, the remote phase head) and of the CI runs on
each SHA, and records what was committed and pushed, so a refusal can be
held to having committed and pushed nothing. The CI3 cases
(``--moved-from``) and CI4's verify cases run git for real, in temporary repositories, since a
patch-id or a tree cannot be faked without fixing land.py's git commands;
gh stays the fake. A case listed in ``land_selftest-pending.json`` must
still fail, as when its test author wrote it, until the implementer takes
it out. Run it as

    python3 scripts/rust_core/land_selftest.py

Decision E43; the checks themselves are documented in land.py.
"""

from __future__ import annotations

import contextlib
import fnmatch
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import land  # noqa: E402

# The cases set the phase themselves (T2.0): one named in the caller's
# environment would override the topic branch's.
os.environ.pop('SDM_PHASE_BRANCH', None)

CAND = 'c' * 8 + '1' * 32
BASE = 'b' * 8 + '2' * 32
CHECK = 'a' * 8 + '3' * 32
OTHER = 'd' * 8 + '4' * 32
LAND = 'e' * 8 + '5' * 32
#: Shares the candidate's first eight digits, so that prefix is ambiguous.
TWIN = 'c' * 8 + '6' * 32
TOPIC = 'rust-core-phase-1-lane-a'
PHASE = 'rust-core-phase-1'
SIGNOFF = 'Signed-off-by: Test Person <test@example.com>'


@dataclass
class Result:
    name: str
    ok: bool
    detail: str


def ci_run(
    sha: str,
    branch: str,
    status: str = 'completed',
    conclusion: str = 'success',
    run_id: int = 1,
    created: str = '2026-10-09T00:00:00Z',
    event: str = 'push',
) -> dict[str, Any]:
    return {
        'databaseId': run_id,
        'headSha': sha,
        'headBranch': branch,
        'status': status,
        'conclusion': conclusion if status == 'completed' else '',
        'event': event,
        'url': f'https://example.invalid/runs/{run_id}',
        'createdAt': created,
    }


@dataclass
class Fake:
    """A scripted git and gh."""

    head: str = CAND
    status: str = ''
    remote_phase: str = BASE
    commits: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            CAND: {
                'parent': BASE,
                'tree': 'tree-cand',
                'message': 'T1.99: a test package\n\nThe work.\n',
            },
            BASE: {'parent': CHECK, 'tree': 'tree-base', 'message': ''},
            CHECK: {'parent': '', 'tree': 'tree-check', 'message': ''},
            OTHER: {'parent': '', 'tree': 'tree-other', 'message': ''},
        }
    )
    ancestors: set[tuple[str, str]] = field(
        default_factory=lambda: {(BASE, CAND), (CHECK, CAND), (CHECK, BASE)}
    )
    runs: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {CAND: [ci_run(CAND, TOPIC)]}
    )
    #: Seeded faults in the landing commit git makes.
    landing_tree: str | None = None
    drop_trailer: str | None = None
    #: Pushes the remote refuses, by a ref they name.
    refuse_push: str | None = None
    #: The conclusion the landing commit's runs reach when watched.
    landing_conclusion: dict[str, str] = field(default_factory=dict)
    pushes: list[tuple[str, ...]] = field(default_factory=list)
    committed: list[str] = field(default_factory=list)
    #: The worktree's configuration: whether it has its own, and the
    #: package recorded there (``lane.py start``).
    worktree_config: bool = True
    package: str | None = 'T1.99'
    #: The branch checked out; None is a detached HEAD.
    branch: str | None = TOPIC
    #: The branches on the remote, as pushes leave them; a seeded value
    #: stands for a push by someone else after the landing.
    remote_refs: dict[str, str] = field(default_factory=dict)
    moved_after_push: dict[str, str] = field(default_factory=dict)
    #: Runs a watch leaves as they are: still unfinished when it returns.
    stuck: set[int] = field(default_factory=set)

    def sha(self, rev: str) -> str | None:
        rev = rev.removesuffix('^{commit}')
        if rev == 'HEAD':
            return self.head
        if re.fullmatch(r'refs/remotes/origin/rust-core-phase-\d+', rev):
            return self.remote_phase
        found = [s for s in self.commits if s.startswith(rev.lower())]
        return found[0] if len(found) == 1 and len(rev) >= 4 else None

    def rev(self, rev: str) -> str | None:
        if rev.endswith('^{tree}'):
            sha = self.rev(rev.removesuffix('^{tree}'))
            return self.commits[sha]['tree'] if sha else None
        if rev.endswith('^'):
            sha = self.rev(rev[:-1])
            return self.commits[sha]['parent'] if sha else None
        return self.sha(rev)

    def merge_base(self, a: str | None, b: str | None) -> land.Completed:
        """The common ancestor no other common ancestor descends from."""

        def below(x: str, y: str | None) -> bool:
            return x == y or (x, y) in self.ancestors

        common = [c for c in self.commits if below(c, a) and below(c, b)]
        best = [
            c
            for c in common
            if not any(d != c and (c, d) in self.ancestors for d in common)
        ]
        if len(best) != 1:
            return land.Completed(1)
        return land.Completed(0, best[0] + '\n')

    def __call__(self, argv: Sequence[str]) -> land.Completed:
        args = list(argv)
        if args[0] == 'git':
            return self.git(args[1:])
        return self.gh(args[1:])

    def git(self, args: list[str]) -> land.Completed:
        ok = land.Completed(0)
        if args[0] == 'status':
            return land.Completed(0, self.status)
        if args[0] == 'show':
            # No judging-paths list at the phase head: the scripted phase
            # predates T3.0, whose rules the T3.0 tests hold.
            return land.Completed(1)
        if args[0] == 'fetch':
            return ok
        if args[:2] == ['rev-list', '--merges']:
            # The scripted history is linear: no merge of main (T2.0e).
            return ok
        if args[0] == 'remote':
            return land.Completed(
                0, 'git@github.com:NVIDIA/structured-data-models-client.git\n'
            )
        if args[0] == 'rev-parse':
            value = self.rev(args[-1])
            return (
                land.Completed(0, value + '\n') if value else land.Completed(1)
            )
        if args[0] == 'merge-base' and args[1] == '--is-ancestor':
            a, b = self.sha(args[2]), self.sha(args[3])
            return land.Completed(
                0 if a == b or (a, b) in self.ancestors else 1
            )
        if args[0] == 'merge-base':
            return self.merge_base(self.sha(args[1]), self.sha(args[2]))
        if args[0] == 'commit':
            message = Path(args[args.index('-F') + 1]).read_text()
            if self.drop_trailer:
                message = '\n'.join(
                    line
                    for line in message.splitlines()
                    if not line.startswith(self.drop_trailer)
                )
            message = message.rstrip('\n') + '\n' + SIGNOFF + '\n'
            tree = self.landing_tree or self.commits[self.head]['tree']
            self.commits[LAND] = {
                'parent': self.head,
                'tree': tree,
                'message': message,
            }
            self.head = LAND
            self.committed.append(message)
            return ok
        if args[0] == 'symbolic-ref':
            if self.branch is None:
                return land.Completed(1)
            return land.Completed(0, self.branch + '\n')
        if args[0] == 'log' and '--format=%s' in args:
            message = self.commits[self.sha(args[-1]) or '']['message']
            return land.Completed(0, message.split('\n', 1)[0] + '\n')
        if args[0] == 'config':
            if args[-1] == 'extensions.worktreeConfig':
                return land.Completed(
                    0 if self.worktree_config else 1,
                    'true\n' if self.worktree_config else '',
                )
            if args[1:] == ['--worktree', '--get', land.PACKAGE_KEY]:
                if self.package is None:
                    return land.Completed(1)
                return land.Completed(0, self.package + '\n')
            raise AssertionError(f'unexpected git {args}')
        if args[0] == 'ls-remote':
            refs = {**self.remote_refs, **self.moved_after_push}
            lines = [
                f'{refs[ref]}\t{ref}'
                for pattern in args[2:]
                for ref in sorted(refs)
                if fnmatch.fnmatchcase(ref, pattern)
            ]
            return land.Completed(0, ''.join(line + '\n' for line in lines))
        if args[0] == 'log' and '..' in args[-1]:
            base, head = (self.sha(r) for r in args[-1].split('..'))
            entries = []
            while head and head != base:
                entries.append(f'{head}\x00{self.commits[head]["message"]}')
                head = self.commits[head]['parent']
            return land.Completed(0, '\x1e\n'.join(entries) + '\x1e\n')
        if args[0] == 'log':
            message = self.commits[self.sha(args[-1]) or '']['message']
            block = message.strip().split('\n\n')[-1]
            return land.Completed(0, block + '\n')
        if args[0] == 'push':
            self.pushes.append(tuple(args[1:]))
            if self.refuse_push and any(self.refuse_push in a for a in args):
                return land.Completed(1, '', '! [rejected] (fetch first)')
            for ref in args[1:]:
                if ':refs/heads/' in ref:
                    sha, branch = ref.split(':refs/heads/')
                    if re.fullmatch(r'rust-core-phase-\d+', branch):
                        self.remote_phase = sha
                    self.remote_refs[f'refs/heads/{branch}'] = sha
                    self.runs.setdefault(sha, []).append(
                        ci_run(
                            sha,
                            branch,
                            status='in_progress',
                            run_id=100 + len(self.pushes) * 10 + len(branch),
                            created=f'2026-10-09T01:00:{len(self.pushes):02d}Z',
                        )
                    )
            return ok
        raise AssertionError(f'unexpected git {args}')

    def gh(self, args: list[str]) -> land.Completed:
        if args[:2] == ['run', 'list']:
            sha = args[args.index('--commit') + 1]
            listed = self.runs.get(sha, [])
            for flag, key in (('--event', 'event'), ('--branch', 'headBranch')):
                if flag in args:
                    wanted = args[args.index(flag) + 1]
                    listed = [r for r in listed if r[key] == wanted]
            return land.Completed(0, json.dumps(listed))
        if args[:2] in (['run', 'watch'], ['run', 'view']):
            run_id = int(args[2])
            for runs in self.runs.values():
                for run in runs:
                    if run['databaseId'] == run_id:
                        if args[1] == 'watch' and run_id in self.stuck:
                            return land.Completed(1, '', 'still running')
                        if args[1] == 'watch':
                            # A finished run stays as it finished.
                            if run['status'] != 'completed':
                                run['status'] = 'completed'
                                run['conclusion'] = self.landing_conclusion.get(
                                    run['headBranch'], 'success'
                                )
                            return land.Completed(
                                0 if run['conclusion'] == 'success' else 1
                            )
                        return land.Completed(0, json.dumps(run))
            return land.Completed(1, '', 'no such run')
        raise AssertionError(f'unexpected gh {args}')


class Case:
    """One land.py invocation against a fake, with its review files."""

    def __init__(self, tmp: Path, fake: Fake | None = None) -> None:
        self.tmp = tmp
        self.fake = fake or Fake()
        self.lines: list[str] = []
        self.code: int | None = None
        self.calls: list[tuple[str, ...]] = []
        self.reviews = [
            self.review(
                'a.txt', f'APPROVE\n\nReviewed {CAND[:7]}; no findings.\n'
            ),
            self.review('b.txt', f'APPROVE\nCandidate {CAND}, both rounds.\n'),
        ]

    def review(self, name: str, text: str) -> str:
        path = self.tmp / name
        path.write_text(text)
        return str(path)

    def land(self, *extra: str, reviews: Sequence[str] | None = None) -> Case:
        argv = [
            '--candidate', CAND[:10], '--topic-branch', TOPIC,
            '--title', 'T1.99: a test package',
            '--protected-paths', '.github/workflows/ci.yml (no gate removed)',
        ]  # fmt: skip
        for path in self.reviews if reviews is None else reviews:
            argv += ['--review', path]
        return self.call([*argv, *extra])

    def verify(self, sha: str = LAND) -> Case:
        # A Phase 1 landing, verified by naming its phase (T2.0).
        return self.call(
            ['verify', sha, '--appear-timeout', '0', '--phase-branch', PHASE]
        )

    def call(self, argv: list[str]) -> Case:
        tools = land.Tools(run=self.fake, sleep=lambda _: None, repo='o/r')
        self.code = land.main(argv, tools, self.lines.append)
        self.calls = tools.calls
        return self

    @property
    def text(self) -> str:
        return '\n'.join(self.lines)

    def untouched(self) -> bool:
        return not self.fake.committed and not self.fake.pushes


Check = Callable[[Path], tuple[bool, str]]
CASES: list[tuple[str, Check]] = []


def case(name: str) -> Callable[[Check], Check]:
    def register(check: Check) -> Check:
        CASES.append((name, check))
        return check

    return register


def refused(c: Case, fragment: str) -> tuple[bool, str]:
    ok = c.code == 2 and fragment in c.text and c.untouched()
    return ok, (
        'refused, nothing committed or pushed'
        if ok
        else f'exit {c.code}, committed {len(c.fake.committed)}, pushes '
        f'{c.fake.pushes}: {c.text[-600:]}'
    )


# --- a sound landing ----------------------------------------------------------


@case('lands: one empty commit, then one atomic push of both branches')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp).land()
    f = c.fake
    ok = (
        c.code == 0
        and f.commits[LAND]['parent'] == CAND
        and f.commits[LAND]['tree'] == f.commits[CAND]['tree']
        and f.pushes
        == [
            (
                '--quiet',
                '--no-verify',
                '--atomic',
                'origin',
                f'{LAND}:refs/heads/{TOPIC}',
                f'{LAND}:refs/heads/{PHASE}',
            )
        ]
        and f'LANDED {LAND}' in c.text
        and f'next: land.py verify {LAND}' in c.text
    )
    return ok, 'landed' if ok else f'exit {c.code}, pushes {f.pushes}: {c.text}'


@case('lands: the landing runs on both branches are printed')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp).land()
    ok = c.code == 0 and all(
        re.search(rf'run  {branch}: \d+ https://', c.text)
        for branch in (TOPIC, PHASE)
    )
    return ok, 'both printed' if ok else c.text


@case("lands: the message has the earlier landing commits' form")
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp).land()
    message = c.fake.committed[0] if c.fake.committed else ''
    expected = (
        r'Land T1\.99: a test package\n\n'
        r'Both reviews approve the candidate c{7}\.\n\n'
        rf'Candidate run 1 green at {CAND}; (?:[^\n]{{1,72}}\n)+\n'
        r'Reviewed-by: independent reviewer \(fresh agent\) and Codex, T1\.99\n'
        r'Protected-paths: \.github/workflows/ci\.yml \(no gate removed\)\n'
        rf'Base-SHA: {BASE}\n'
        rf'Checkpoint-SHA: {BASE}\n'
        rf'{re.escape(SIGNOFF)}\n'
    )
    ok = re.fullmatch(expected, message) is not None
    return ok, 'subject, summary, four trailers, sign-off' if ok else message


@case('lands: --reviewer and --summary are used as given')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp).land(
        '--reviewer', 'reviewer X and Codex, T1.99', '--summary', 'Custom.'
    )
    message = c.fake.committed[0] if c.fake.committed else ''
    ok = '\n\nCustom.\n\n' in message and (
        'Reviewed-by: reviewer X and Codex, T1.99\n' in message
    )
    return ok, 'used' if ok else message


@case('dry run: every check, nothing committed or pushed')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp).land('--dry-run')
    ok = (
        c.code == 0
        and c.untouched()
        and 'Base-SHA: ' + BASE in c.text
        and 'nothing committed, nothing pushed' in c.text
    )
    return ok, 'checked only' if ok else c.text


@case('dry run: still refuses a failing check')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp, Fake(status=' M README.md\n')).land('--dry-run')
    return refused(c, 'not clean')


# --- the worktree and ancestry ------------------------------------------------


@case('refused: a dirty worktree')
def _(tmp: Path) -> tuple[bool, str]:
    return refused(Case(tmp, Fake(status='?? stray.txt\n')).land(), 'not clean')


@case('refused: HEAD is not the candidate')
def _(tmp: Path) -> tuple[bool, str]:
    return refused(Case(tmp, Fake(head=BASE)).land(), 'not the candidate')


@case('refused: a candidate that is not a commit')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp)
    c.call(['--candidate', 'f' * 12, '--topic-branch', TOPIC, '--title', 'T1.99: t',
            '--protected-paths', 'p', '--checkpoint', CHECK,
            '--review', c.reviews[0], '--review', c.reviews[1]])  # fmt: skip
    return refused(c, 'is not a commit')


@case('refused: the candidate does not descend from the phase head')
def _(tmp: Path) -> tuple[bool, str]:
    fake = Fake(remote_phase=OTHER)
    fake.ancestors.add((CHECK, OTHER))
    return refused(Case(tmp, fake).land(), 'does not descend')


@case('refused: the candidate is the phase head')
def _(tmp: Path) -> tuple[bool, str]:
    return refused(Case(tmp, Fake(remote_phase=CAND)).land(), 'nothing to land')


# --- the checkpoint (design, H2: the merge base of candidate and base) ---------


@case('lands: without --checkpoint, the trailer is the computed merge base')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp).land()
    message = c.fake.committed[0] if c.fake.committed else ''
    ok = (
        c.code == 0
        and f'Checkpoint-SHA: {BASE}\n' in message
        and ('git', 'merge-base', CAND, BASE) in c.calls
        and f'the checkpoint is their merge base, {BASE}' in c.text
    )
    return ok, 'computed' if ok else f'exit {c.code}: {c.text} | {message}'


@case('lands: a --checkpoint naming the merge base, by a prefix')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp).land('--checkpoint', BASE[:10])
    message = c.fake.committed[0] if c.fake.committed else ''
    ok = c.code == 0 and f'Checkpoint-SHA: {BASE}\n' in message
    return ok, 'accepted' if ok else f'exit {c.code}: {c.text}'


@case('lands: the merge base, not the phase head, when they differ')
def _(tmp: Path) -> tuple[bool, str]:
    # A fake whose merge-base answers an older commit than the phase head,
    # so the trailer is shown to be git's answer, not a copy of Base-SHA.
    fake = Fake()
    fake.merge_base = lambda a, b: land.Completed(0, CHECK + '\n')  # type: ignore[method-assign]
    c = Case(tmp, fake).land()
    message = fake.committed[0] if fake.committed else ''
    ok = (
        c.code == 0
        and f'Checkpoint-SHA: {CHECK}\n' in message
        and f'Base-SHA: {BASE}\n' in message
    )
    return ok, "git's merge base" if ok else f'exit {c.code}: {message}'


@case('refused: a --checkpoint that is an older ancestor, not the merge base')
def _(tmp: Path) -> tuple[bool, str]:
    # The mislabel checkpoint 3 found: the design-checkpoint review's SHA,
    # an ancestor of both, passed in place of the merge base.
    return refused(
        Case(tmp).land('--checkpoint', CHECK[:12]),
        f'the checkpoint {CHECK[:12]} is not the merge base',
    )


@case('refused: a --checkpoint that is not an ancestor of the candidate')
def _(tmp: Path) -> tuple[bool, str]:
    return refused(
        Case(tmp).land('--checkpoint', OTHER), 'is not the merge base'
    )


@case('refused: a --checkpoint that is not a commit')
def _(tmp: Path) -> tuple[bool, str]:
    return refused(Case(tmp).land('--checkpoint', 'f' * 12), 'the checkpoint')


@case('refused: no merge base to compute')
def _(tmp: Path) -> tuple[bool, str]:
    fake = Fake()
    fake.merge_base = lambda a, b: land.Completed(1)  # type: ignore[method-assign]
    return refused(Case(tmp, fake).land(), 'merge-base')


# --- the reviews --------------------------------------------------------------


@case('refused: one review')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp)
    return refused(c.land(reviews=c.reviews[:1]), 'needs both reviews')


@case('refused: the same review file twice')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp)
    return refused(c.land(reviews=[c.reviews[0]] * 2), 'given twice')


@case('refused: two files with the same review')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp)
    copy = c.review('copy.txt', Path(c.reviews[0]).read_text())
    return refused(c.land(reviews=[c.reviews[0], copy]), 'the same text')


@case('refused: an unreadable review file')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp)
    return refused(
        c.land(reviews=[c.reviews[0], str(tmp / 'missing.txt')]), 'unreadable'
    )


for verdict in (
    'Approve',
    'APPROVE.',
    ' APPROVE',
    'APPROVE ',
    'APPROVE with nits',
    'REQUEST CHANGES',
    '',
):

    @case(f'refused: a review whose first line is {verdict!r}')
    def _(tmp: Path, verdict: str = verdict) -> tuple[bool, str]:
        c = Case(tmp)
        bad = c.review('bad.txt', f'{verdict}\nAPPROVE {CAND}\n')
        return refused(c.land(reviews=[c.reviews[0], bad]), 'not exactly')


@case('refused: a review that names no SHA')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp)
    bad = c.review('bad.txt', 'APPROVE\nLooks good.\n')
    return refused(
        c.land(reviews=[c.reviews[0], bad]), 'does not name the candidate'
    )


@case('refused: a review naming the candidate by six characters')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp)
    bad = c.review('bad.txt', f'APPROVE\nReviewed {CAND[:6]}.\n')
    return refused(
        c.land(reviews=[c.reviews[0], bad]), 'does not name the candidate'
    )


@case('refused: a review naming another commit')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp)
    bad = c.review('bad.txt', f'APPROVE\nReviewed {BASE[:10]} and {OTHER}.\n')
    return refused(
        c.land(reviews=[c.reviews[0], bad]), 'does not name the candidate'
    )


@case('refused: a review naming a longer hex word that only starts the same')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp)
    bad = c.review('bad.txt', f'APPROVE\nblob {CAND[:7]}ff00\n')
    longer = CAND[:7] + 'ff00'
    ok_prefix = not CAND.startswith(longer)
    ok, detail = refused(
        c.land(reviews=[c.reviews[0], bad]), 'does not name the candidate'
    )
    return ok and ok_prefix, detail


@case('lands: a review naming the candidate by seven characters, upper case')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp)
    good = c.review('good.txt', f'APPROVE\nReviewed ({CAND[:7].upper()}).\n')
    c.land(reviews=[c.reviews[0], good])
    return c.code == 0, 'landed' if c.code == 0 else c.text


@case('names_candidate: full, prefix, embedded and too short')
def _(tmp: Path) -> tuple[bool, str]:
    tools = land.Tools(run=Fake())
    checks = {
        f'at {CAND}.': True,
        f'at {CAND[:7]};': True,
        f'x{CAND[:7]}': True,  # a non-hex letter before it ends the word
        f'0{CAND[:7]}': False,
        f'{CAND[:6]}': False,
        f'{CAND[:7]}0': False,
    }
    wrong = {
        text: expect
        for text, expect in checks.items()
        if (land.names_candidate(tools, text, CAND) is None) != expect
    }
    return not wrong, 'as expected' if not wrong else f'wrong: {wrong}'


def with_twin() -> Fake:
    fake = Fake()
    fake.commits[TWIN] = {'parent': BASE, 'tree': 'tree-twin', 'message': ''}
    return fake


@case('refused: a review naming the candidate by an ambiguous prefix')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp, with_twin())
    bad = c.review('bad.txt', f'APPROVE\nReviewed {CAND[:8]}.\n')
    return refused(c.land(reviews=[c.reviews[0], bad]), 'ambiguous')


@case('refused: a review naming the twin, whose prefix the candidate shares')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp, with_twin())
    bad = c.review('bad.txt', f'APPROVE\nReviewed {TWIN[:12]}.\n')
    return refused(
        c.land(reviews=[c.reviews[0], bad]), 'does not name the candidate'
    )


@case('lands: a longer prefix that resolves to the candidate alone')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp, with_twin())
    good = c.review('good.txt', f'APPROVE\nReviewed {CAND[:9]}.\n')
    c.land(reviews=[c.reviews[1], good])
    return c.code == 0, 'landed' if c.code == 0 else c.text


# --- the branches -------------------------------------------------------------


for topic in (
    'main',
    'master',
    'rust-core',
    PHASE,
    'rust-core-phase-2-lane-a',
    'feature-x',
    f'{PHASE}-',
    f'{PHASE}-a..b',
    f'{PHASE}-x.lock',
    f'refs/heads/{PHASE}-lane-a',
    f'{PHASE}-lane-',
    f'{PHASE}-lane-A',
    f'{PHASE}-lane-1.b',
    f'{PHASE}-lane-a_b',
    f'{PHASE}-archive-1b-T1.11e-abc1234',
):

    @case(f'refused: --topic-branch {topic}')
    def _(tmp: Path, topic: str = topic) -> tuple[bool, str]:
        c = Case(tmp).land('--topic-branch', topic, '--phase-branch', PHASE)
        ok, detail = refused(c, 'topic branch')
        # Refused before anything was asked of git or gh.
        return ok and not c.calls, detail


for phase in (
    'main',
    'rust-core',
    f'{PHASE}-lane-a',
    'rust-core-phase-01',
    'rust-core-phase-x',
):

    @case(f'refused: --phase-branch {phase}')
    def _(tmp: Path, phase: str = phase) -> tuple[bool, str]:
        c = Case(tmp).land('--phase-branch', phase)
        ok, detail = refused(c, 'phase branch')
        return ok and not c.calls, detail


# --- the phase is one setting (T2.0) ------------------------------------------
#
# The phase branch is --phase-branch, else SDM_PHASE_BRANCH, else the phase
# the topic branch is under, else phases.CURRENT; every name under it (the
# topic, lane and archive branches, and which runs are evidence) follows.

PHASE_2 = 'rust-core-phase-2'
TOPIC_2 = f'{PHASE_2}-lane-a'


@contextlib.contextmanager
def phase_environment(value: str | None) -> Iterator[None]:
    old = os.environ.pop('SDM_PHASE_BRANCH', None)
    if value is not None:
        os.environ['SDM_PHASE_BRANCH'] = value
    try:
        yield
    finally:
        os.environ.pop('SDM_PHASE_BRANCH', None)
        if old is not None:
            os.environ['SDM_PHASE_BRANCH'] = old


def phase_2_fake() -> Fake:
    return Fake(branch=TOPIC_2, runs={CAND: [ci_run(CAND, TOPIC_2)]})


def pushed_to(c: Case, phase: str, topic: str) -> bool:
    return c.code == 0 and c.fake.pushes == [
        (
            '--quiet',
            '--no-verify',
            '--atomic',
            'origin',
            f'{LAND}:refs/heads/{topic}',
            f'{LAND}:refs/heads/{phase}',
        )
    ]


@case("phase: a Phase 2 lane lands on rust-core-phase-2, the topic's phase")
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp, phase_2_fake()).land('--topic-branch', TOPIC_2)
    ok = pushed_to(c, PHASE_2, TOPIC_2)
    return ok, 'landed on Phase 2' if ok else f'{c.fake.pushes}: {c.text}'


@case('phase: --phase-branch takes effect over SDM_PHASE_BRANCH')
def _(tmp: Path) -> tuple[bool, str]:
    with phase_environment(PHASE):
        c = Case(tmp, phase_2_fake()).land(
            '--topic-branch', TOPIC_2, '--phase-branch', PHASE_2
        )
    ok = pushed_to(c, PHASE_2, TOPIC_2)
    return ok, "landed on the flag's phase" if ok else c.text


@case('phase: SDM_PHASE_BRANCH takes effect: a lane of another phase refused')
def _(tmp: Path) -> tuple[bool, str]:
    with phase_environment(PHASE):
        c = Case(tmp, phase_2_fake()).land('--topic-branch', TOPIC_2)
    ok, detail = refused(c, f'not a package branch ({PHASE}-<name>')
    return ok and not c.calls, detail


@case('phase: --phase-branch takes effect: a lane of another phase refused')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp).land('--phase-branch', PHASE_2)
    ok, detail = refused(c, f'not a package branch ({PHASE_2}-<name>')
    return ok and not c.calls, detail


@case('phase: SDM_PHASE_BRANCH that is no phase branch is refused, by name')
def _(tmp: Path) -> tuple[bool, str]:
    with phase_environment('rust-core-phase-two'):
        c = Case(tmp).land()
    ok, detail = refused(c, "'rust-core-phase-two' is not a phase branch")
    return ok and not c.calls, detail


@case("phase: only a run under the landing's phase is evidence")
def _(tmp: Path) -> tuple[bool, str]:
    # The candidate's only green run is on a Phase 1 lane: judged against
    # Phase 1's checkpoint, so no evidence for a Phase 2 landing.
    fake = Fake(branch=TOPIC_2, runs={CAND: [ci_run(CAND, TOPIC)]})
    c = Case(tmp, fake).land('--topic-branch', TOPIC_2)
    return refused_for_push_evidence(c)


for archive in (f'{PHASE_2}-archive-a-T2.1-abc1234',):

    @case(f'phase: a Phase 2 archive branch is refused: {archive}')
    def _(tmp: Path, archive: str = archive) -> tuple[bool, str]:
        c = Case(tmp, phase_2_fake()).land('--topic-branch', archive)
        ok, detail = refused(c, 'archive branch')
        return ok and not c.calls, detail


for package in ('T2.0', 'T2.0a', 'T2.15', 'CI5'):

    @case(f'phase: the package ID {package} lands on a Phase 2 lane')
    def _(tmp: Path, package: str = package) -> tuple[bool, str]:
        fake = phase_2_fake()
        fake.package = package
        fake.commits[CAND]['message'] = f'{package}: a test package\n'
        c = Case(tmp, fake).land(
            '--topic-branch', TOPIC_2, '--title', f'{package}: a package'
        )
        ok = pushed_to(c, PHASE_2, TOPIC_2) and (
            c.fake.committed[0].startswith(f'Land {package}: a package\n')
        )
        return ok, f'landed {package}' if ok else c.text


@case("phase: verify with no --phase-branch infers the landing's phase")
def _(tmp: Path) -> tuple[bool, str]:
    c = landed(tmp)
    # Phase 2's tip holds no Phase 1 landing here; Phase 1's does.
    c.fake.remote_refs[f'refs/heads/{PHASE_2}'] = OTHER
    c.call(['verify', LAND, '--appear-timeout', '0'])
    ok = c.code == 0 and f'VERIFIED {LAND}' in c.text and PHASE_2 not in c.text
    return ok, 'verified on Phase 1, inferred' if ok else c.text


@case('phase: verify with no --phase-branch, no phase holding it, refused')
def _(tmp: Path) -> tuple[bool, str]:
    c = landed(tmp)
    c.fake.remote_refs = {f'refs/heads/{PHASE_2}': OTHER}
    c.call(['verify', LAND, '--appear-timeout', '0'])
    ok = c.code == 2 and 'no phase branch on origin holds' in c.text
    return ok, 'refused by name' if ok else c.text


@case('lands: a task-named topic branch under the phase branch')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp).land('--topic-branch', f'{PHASE}-t1.12')
    ok = c.code == 0 and any(
        f'refs/heads/{PHASE}-t1.12' in ref
        for push in c.fake.pushes
        for ref in push
    )
    return ok, 'landed' if ok else c.text


# --- the candidate's CI run ---------------------------------------------------


@case('refused: no CI run on the candidate')
def _(tmp: Path) -> tuple[bool, str]:
    return refused(Case(tmp, Fake(runs={})).land(), 'no CI run')


for state, conclusion in (
    ('completed', 'cancelled'),
    ('completed', 'failure'),
    ('completed', 'skipped'),
    ('in_progress', ''),
    ('queued', ''),
):

    @case(f'refused: the candidate run is {conclusion or state}')
    def _(
        tmp: Path, state: str = state, conclusion: str = conclusion
    ) -> tuple[bool, str]:
        runs = {CAND: [ci_run(CAND, TOPIC, state, conclusion)]}
        return refused(Case(tmp, Fake(runs=runs)).land(), 'not green')


@case('refused: a green run, then a newer cancelled one on the same SHA')
def _(tmp: Path) -> tuple[bool, str]:
    runs = {
        CAND: [
            ci_run(CAND, TOPIC, run_id=1, created='2026-10-09T00:00:00Z'),
            ci_run(
                CAND,
                TOPIC,
                conclusion='cancelled',
                run_id=2,
                created='2026-10-09T00:05:00Z',
            ),
        ]
    }
    return refused(Case(tmp, Fake(runs=runs)).land(), 'cancelled')


@case('refused: a green run listed for another SHA does not count')
def _(tmp: Path) -> tuple[bool, str]:
    return refused(
        Case(tmp, Fake(runs={CAND: [ci_run(OTHER, TOPIC)]})).land(),
        'no CI run',
    )


@case('lands: the newest run green after an older failure')
def _(tmp: Path) -> tuple[bool, str]:
    runs = {
        CAND: [
            ci_run(CAND, TOPIC, conclusion='failure', run_id=1),
            ci_run(CAND, TOPIC, run_id=2, created='2026-10-09T00:09:00Z'),
        ]
    }
    c = Case(tmp, Fake(runs=runs)).land()
    return c.code == 0, 'landed' if c.code == 0 else c.text


# --- only a push run under the phase branch is evidence (T1.20 fix 1) --------
#
# CI's gate steps exit early on a pull_request run (ratchet.py names no base
# there), so a green one never ran the checkpoint gate. Only a push run on
# the phase branch or a branch under it (a lane, a task or an old topic
# branch; never an archive branch) counts, for the candidate, the landing
# run and verify.

#: An archive branch: where an abandoned lane's tip is kept, never a topic.
ARCHIVED = 'rust-core-phase-1-archive-lane-a-t1.99'
#: A pull request's ref, as a run on it may name its branch.
PR_REF = 'refs/pull/7/merge'
#: Names the missing evidence as a push run.
PUSH_RUN = re.compile(r'\bpush\b[^\n]*\brun|\brun[^\n]*\bpush\b', re.I)


def refused_for_push_evidence(c: Case) -> tuple[bool, str]:
    """Refused, nothing committed or pushed, naming a missing push run."""
    ok = c.code == 2 and bool(PUSH_RUN.search(c.text)) and c.untouched()
    return ok, (
        'refused for want of a push run, nothing committed or pushed'
        if ok
        else f'exit {c.code}, committed {len(c.fake.committed)}, pushes '
        f'{c.fake.pushes}: {c.text[-600:]}'
    )


for branch in (TOPIC, PR_REF):

    @case(
        f'refused: the only green run is a pull_request run (on {branch}), '
        'naming the missing push run'
    )
    def _(tmp: Path, branch: str = branch) -> tuple[bool, str]:
        runs = {CAND: [ci_run(CAND, branch, event='pull_request')]}
        return refused_for_push_evidence(Case(tmp, Fake(runs=runs)).land())


@case('refused: a green pull_request run newer than a failed push run')
def _(tmp: Path) -> tuple[bool, str]:
    runs = {
        CAND: [
            ci_run(CAND, TOPIC, conclusion='failure', run_id=1),
            ci_run(
                CAND,
                TOPIC,
                run_id=2,
                created='2026-10-09T00:09:00Z',
                event='pull_request',
            ),
        ]
    }
    return refused(Case(tmp, Fake(runs=runs)).land(), 'not green')


@case('refused: the only green push run is on an archive branch')
def _(tmp: Path) -> tuple[bool, str]:
    runs = {CAND: [ci_run(CAND, ARCHIVED)]}
    return refused_for_push_evidence(Case(tmp, Fake(runs=runs)).land())


@case('refused: a green archive-branch push run newer than a failed lane run')
def _(tmp: Path) -> tuple[bool, str]:
    runs = {
        CAND: [
            ci_run(CAND, TOPIC, conclusion='failure', run_id=1),
            ci_run(CAND, ARCHIVED, run_id=2, created='2026-10-09T00:09:00Z'),
        ]
    }
    return refused(Case(tmp, Fake(runs=runs)).land(), 'not green')


@case('refused: the only green push run is on a branch outside the phase')
def _(tmp: Path) -> tuple[bool, str]:
    runs = {CAND: [ci_run(CAND, 'main')]}
    return refused_for_push_evidence(Case(tmp, Fake(runs=runs)).land())


@case('lands: a green push run beside a newer red pull_request run')
def _(tmp: Path) -> tuple[bool, str]:
    # A pull_request run is no evidence either way.
    runs = {
        CAND: [
            ci_run(CAND, TOPIC, run_id=1),
            ci_run(
                CAND,
                TOPIC,
                conclusion='failure',
                run_id=2,
                created='2026-10-09T00:09:00Z',
                event='pull_request',
            ),
        ]
    }
    c = Case(tmp, Fake(runs=runs)).land()
    ok = c.code == 0 and 'Candidate run 1 green' in ' '.join(
        (c.fake.committed or [''])[0].split()
    )
    return ok, 'landed on the push run' if ok else c.text


for branch in (PHASE, 'rust-core-phase-1-t1.19', 'rust-core-phase-1-lane-b'):

    @case(f'lands: a green push run on {branch} counts')
    def _(tmp: Path, branch: str = branch) -> tuple[bool, str]:
        c = Case(tmp, Fake(runs={CAND: [ci_run(CAND, branch)]})).land()
        return c.code == 0, 'landed' if c.code == 0 else c.text


# --- the landing commit and the push ------------------------------------------


@case("refused: a landing commit whose tree is not the candidate's")
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp, Fake(landing_tree='tree-changed')).land()
    ok = (
        c.code == 2
        and 'nothing was pushed' in c.text
        and not c.fake.pushes
        and f'git reset --soft {CAND}' in c.text
    )
    return ok, 'refused before the push' if ok else c.text


@case('refused: a landing commit that lost a trailer')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp, Fake(drop_trailer='Checkpoint-SHA')).land()
    ok = c.code == 2 and 'Checkpoint-SHA' in c.text and not c.fake.pushes
    return ok, 'refused before the push' if ok else c.text


@case('refused: the remote refuses the push (the phase branch moved)')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp, Fake(refuse_push=PHASE)).land()
    ok = (
        c.code == 2
        and 'rejected' in c.text
        and 'LANDED' not in c.text
        and f'git reset --soft {CAND}' in c.text
    )
    return ok, 'reported with the undo, not landed' if ok else c.text


# --- verify -------------------------------------------------------------------


def landed(tmp: Path, **faults: Any) -> Case:
    c = Case(tmp, Fake(**faults)).land()
    c.lines.clear()
    c.fake.pushes.clear()
    return c


@case('verify: both runs green')
def _(tmp: Path) -> tuple[bool, str]:
    c = landed(tmp).verify()
    ok = (
        c.code == 0
        and f'VERIFIED {LAND}' in c.text
        and c.text.count('ok  ') >= 2
    )
    return ok, 'verified' if ok else c.text


@case('verify: a red phase run prints the revert and reverts nothing')
def _(tmp: Path) -> tuple[bool, str]:
    c = landed(tmp, landing_conclusion={PHASE: 'failure'}).verify()
    ok = (
        c.code == 1
        and 'VERIFY FAILED' in c.text
        and 'Nothing was reverted' in c.text
        and f'--force-with-lease=refs/heads/{PHASE}:{LAND}' in c.text
        and f'{BASE}:refs/heads/{PHASE}' in c.text
        and f'git revert --no-edit {BASE}..{LAND}' in c.text
        and not c.fake.pushes
        and not c.fake.committed[1:]
    )
    return ok, 'instructions only' if ok else c.text


@case('verify: a landing commit that changes files fails')
def _(tmp: Path) -> tuple[bool, str]:
    c = landed(tmp)
    c.fake.commits[LAND]['tree'] = 'tree-changed'
    c.verify()
    ok = c.code == 1 and 'changes files' in c.text
    return ok, 'reported' if ok else c.text


@case('verify: a cancelled phase run beside a green topic run passes')
def _(tmp: Path) -> tuple[bool, str]:
    c = landed(tmp, landing_conclusion={PHASE: 'cancelled'}).verify()
    ok = (
        c.code == 0
        and 'CANC' in c.text
        and 'same tree is green' in c.text
        and 'VERIFY FAILED' not in c.text
    )
    return ok, 'reported cancelled, passed' if ok else c.text


@case('verify: both cancelled, the candidate green: passes on the tree')
def _(tmp: Path) -> tuple[bool, str]:
    c = landed(
        tmp, landing_conclusion={PHASE: 'cancelled', TOPIC: 'cancelled'}
    ).verify()
    ok = c.code == 0 and f'at {CAND[:12]}' in c.text
    return ok, "the candidate's run stands" if ok else c.text


@case('verify: cancelled with no green run asks for a rerun, not a revert')
def _(tmp: Path) -> tuple[bool, str]:
    c = landed(tmp, landing_conclusion={PHASE: 'cancelled', TOPIC: 'cancelled'})
    for run in c.fake.runs[CAND]:
        run['conclusion'] = 'cancelled'
    c.verify()
    ok = (
        c.code == 3
        and 'VERIFY INCOMPLETE' in c.text
        and 'gh run rerun' in c.text
        and 'Nothing was reverted' not in c.text
        and not c.fake.pushes
    )
    return ok, 'rerun asked' if ok else c.text


@case('verify: a red pull_request run on the landing commit is no evidence')
def _(tmp: Path) -> tuple[bool, str]:
    c = landed(tmp)
    c.fake.runs[LAND].append(
        ci_run(
            LAND,
            TOPIC,
            conclusion='failure',
            run_id=98,
            created='2026-10-09T02:00:00Z',
            event='pull_request',
        )
    )
    c.verify()
    ok = c.code == 0 and f'VERIFIED {LAND}' in c.text
    return ok, 'verified on the push runs' if ok else c.text


@case('verify: a green pull_request run does not stand for a cancelled one')
def _(tmp: Path) -> tuple[bool, str]:
    c = landed(tmp, landing_conclusion={PHASE: 'cancelled', TOPIC: 'cancelled'})
    for run in c.fake.runs[CAND]:
        run['conclusion'] = 'cancelled'
    c.fake.runs[CAND].append(
        ci_run(
            CAND,
            TOPIC,
            run_id=97,
            created='2026-10-09T02:00:00Z',
            event='pull_request',
        )
    )
    c.fake.runs[LAND].append(
        ci_run(
            LAND,
            TOPIC,
            run_id=96,
            created='2026-10-09T02:00:00Z',
            event='pull_request',
        )
    )
    c.verify()
    ok = (
        c.code == 3
        and 'VERIFY INCOMPLETE' in c.text
        and 'same tree is green' not in c.text
        and not c.fake.pushes
    )
    return ok, 'rerun asked' if ok else c.text


@case('verify: a green archive-branch run does not stand for a cancelled one')
def _(tmp: Path) -> tuple[bool, str]:
    c = landed(tmp, landing_conclusion={PHASE: 'cancelled', TOPIC: 'cancelled'})
    for run in c.fake.runs[CAND]:
        run['conclusion'] = 'cancelled'
    c.fake.runs[CAND].append(
        ci_run(CAND, ARCHIVED, run_id=95, created='2026-10-09T02:00:00Z')
    )
    c.verify()
    ok = c.code == 3 and 'same tree is green' not in c.text
    return ok, 'rerun asked' if ok else c.text


# --- a lane carries one package -----------------------------------------------


def package_calls(c: Case) -> list[tuple[str, ...]]:
    return [call for call in c.calls if call[:2] == ('git', 'config')]


@case('lands: a lane whose worktree records the package the title names')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp).land('--title', 'Land T1.99: a test package')
    ok = (
        c.code == 0
        and bool(package_calls(c))
        and f'ok   {TOPIC} equals {PHASE} at {LAND}' in c.text
    )
    return ok, 'checked and landed' if ok else c.text


@case('refused: a lane whose worktree records another package')
def _(tmp: Path) -> tuple[bool, str]:
    return refused(Case(tmp, Fake(package='T1.98')).land(), 'records T1.98')


@case('refused: a lane whose worktree records no package')
def _(tmp: Path) -> tuple[bool, str]:
    return refused(Case(tmp, Fake(package=None)).land(), 'records no package')


@case('refused: a lane in a clone without per-worktree configuration')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp, Fake(worktree_config=False)).land()
    return refused(c, 'records no package')


for title in ('a test package', 'Land t1.99: lower case', 'Land T1.99a1: x'):

    @case(f'refused: a lane landed under the title {title!r}')
    def _(tmp: Path, title: str = title) -> tuple[bool, str]:
        c = Case(tmp).land('--title', title)
        return refused(c, 'not a package ID')


@case("refused: a lane commit that does not name the title's package")
def _(tmp: Path) -> tuple[bool, str]:
    fake = Fake()
    fake.commits[CAND]['message'] = 'T1.98: the previous package\n'
    return refused(Case(tmp, fake).land(), 'T1.98: the previous package')


@case('refused: a lane commit naming only a longer ID (T1.99a, T1.990)')
def _(tmp: Path) -> tuple[bool, str]:
    fake = Fake()
    fake.commits[CAND]['message'] = 'T1.99a and T1.990 and xT1.99\n'
    return refused(Case(tmp, fake).land(), 'not owned by T1.99')


for message, why in (
    ('T1.98: the next package, which builds on T1.99\n', "T1.98's"),
    ('Unrelated change\n\nRefs: T1.99\n', 'does not start with T1.99'),
    ('T1.98: next package\n\nDepends-on: T1.99\n', "T1.98's"),
    ('T1.99: port the helper T1.98 added\n', 'also names T1.98'),
):

    @case(f'refused: a lane commit {message.splitlines()[0]!r}')
    def _(
        tmp: Path, message: str = message, why: str = why
    ) -> tuple[bool, str]:
        fake = Fake()
        fake.commits[CAND]['message'] = message
        return refused(Case(tmp, fake).land(), why)


@case('lands: a lane commit citing another package in its body only')
def _(tmp: Path) -> tuple[bool, str]:
    fake = Fake()
    fake.commits[CAND]['message'] = (
        'T1.99 tracker: the previous package landed\n\n'
        'T1.98 landed at abc1234.\n\nRefs: T1.98\n'
    )
    c = Case(tmp, fake).land()
    return c.code == 0, 'landed' if c.code == 0 else c.text


@case('lands: a task branch is not held to a recorded package')
def _(tmp: Path) -> tuple[bool, str]:
    fake = Fake(package=None)
    fake.commits[CAND]['message'] = 'no package named\n'
    c = Case(tmp, fake).land('--topic-branch', f'{PHASE}-t1.12')
    ok = c.code == 0 and not package_calls(c) and 'equals' not in c.text
    return ok, 'landed, nothing read' if ok else c.text


@case('reported: the lane is not the phase head after the push')
def _(tmp: Path) -> tuple[bool, str]:
    fake = Fake(moved_after_push={f'refs/heads/{TOPIC}': OTHER})
    c = Case(tmp, fake).land()
    ok = (
        c.code == 1
        and f'LANDED {LAND}' in c.text
        and 'LANE MISMATCH' in c.text
        and f'{TOPIC} at {OTHER[:12]}' in c.text
    )
    return ok, 'landed, the mismatch reported' if ok else c.text


@case('refused: a lane landed from another branch checked out')
def _(tmp: Path) -> tuple[bool, str]:
    return refused(
        Case(tmp, Fake(branch='scratch')).land(), 'scratch checked out'
    )


@case('refused: a lane landed from a detached HEAD')
def _(tmp: Path) -> tuple[bool, str]:
    return refused(Case(tmp, Fake(branch=None)).land(), 'a detached HEAD')


@case("refused: a lane landed from another lane's worktree")
def _(tmp: Path) -> tuple[bool, str]:
    fake = Fake(branch=f'{PHASE}-lane-b')
    return refused(Case(tmp, fake).land(), f'{PHASE}-lane-b checked out')


#: An earlier landing of T1.99 whose run was red: the work (WORK), its
#: empty landing commit (EARLIER), and the fix on top, the new candidate.
WORK = '1' * 8 + '7' * 32
EARLIER = '2' * 8 + '8' * 32


def relanding(
    subject: str = 'Land T1.99: a test package',
    tree: str = 'tree-work',
    base: str | None = BASE,
) -> Fake:
    fake = Fake()
    trailers = f'Base-SHA: {base}\n' if base else ''
    fake.commits[WORK] = {
        'parent': BASE, 'tree': 'tree-work',
        'message': 'T1.99: a test package\n',
    }  # fmt: skip
    fake.commits[EARLIER] = {
        'parent': WORK, 'tree': tree,
        'message': f'{subject}\n\nThe landing run was red.\n\n'
        f'Reviewed-by: r\n{trailers}{SIGNOFF}\n',
    }  # fmt: skip
    fake.commits[CAND] = {
        'parent': EARLIER, 'tree': 'tree-cand',
        'message': 'T1.99 fix 1: what the red run found\n',
    }  # fmt: skip
    return fake


@case('lands: a fix on an earlier landing commit whose run was red')
def _(tmp: Path) -> tuple[bool, str]:
    c = Case(tmp, relanding()).land()
    landing = c.fake.commits.get(LAND, {})
    ok = (
        c.code == 0
        and landing.get('parent') == CAND
        and f'Base-SHA: {BASE}' in landing.get('message', '')
        and c.fake.remote_phase == LAND
    )
    return ok, 'landed on the fix, Base-SHA the phase head' if ok else c.text


for label, fake_of, why in (
    ('changes a file', lambda: relanding(tree='tree-changed'), 'changes no file'),
    ('has no Base-SHA', lambda: relanding(base=None), 'has none'),
    ('has a Base-SHA outside the phase', lambda: relanding(base=OTHER), 'not in the phase branch'),
    # Since T3.0 another package's landing commit below is a stack, which
    # plain land.py refuses, naming stack-land.
    ("is another package's", lambda: relanding(subject='Land T1.98: other'), 'stack-land'),
):  # fmt: skip

    @case(f'refused: an earlier landing commit on the lane that {label}')
    def _(
        tmp: Path, fake_of: Callable[[], Fake] = fake_of, why: str = why
    ) -> tuple[bool, str]:
        return refused(Case(tmp, fake_of()).land(), why)


@case('ownership_problem: only the subject owns, and names no other ID')
def _(tmp: Path) -> tuple[bool, str]:
    checks = {
        'T1.11e: the work': True,
        'T1.11e tracker: the previous package landed': True,
        'T1.11e fix 2: review findings': True,
        'T1.11e:no space': True,
        'T1.11f: the next package, which builds on T1.11e': False,
        'T1.11e: port the helper T1.11d added': False,
        'Unrelated change': False,
        'Fix (T1.11e).': False,
        'Land T1.11e: the work': False,
        'T1.11: a shorter ID': False,
        'T1.11ee: a longer ID': False,
        't1.11e: lower case': False,
        'T1.11e-ish: no separator': False,
        'T1.11e: xT1.11f glued to a letter': False,
        'T1.11e: the T1.11f work': False,
    }
    wrong = {
        subject: expect
        for subject, expect in checks.items()
        if (land.ownership_problem(subject, 'T1.11e') is None) != expect
    }
    return not wrong, 'as expected' if not wrong else f'wrong: {wrong}'


# --- CI3: a candidate moved over landing commits lands on its source's evidence
#
# ``--moved-from X``: the candidate C is X moved over landing commits only,
# so C lands on X's two reviews and X's green CI run (owner's decision,
# 2026-10-14). These cases run land.py against real temporary git
# repositories (``git patch-id`` and tree comparison cannot be faked without
# dictating land.py's git commands); gh is still the scripted ``Fake``.
#
# Interface the cases hold land.py to:
# - ``--moved-from REV`` names X; without it nothing changes.
# - The checks are made in the order the decision lists them: X's evidence,
#   the same tree, then the patches; each refusal names its condition: the
#   reviews' refusal X's SHA (its first twelve digits), X's CI refusal X's
#   run id or X's SHA, the tree refusal the word "tree", the patch refusal
#   "patch" (or, for an empty commit, "empty").
# - The patches are the generalized rule's (the chained cases below): a
#   commit on H's side that is not a landing commit is an extra patch, so
#   its refusal names the landing commits, the patches or the trees, and an
#   empty commit, 'Landed' or a merge also its SHA; a commit whose tree
#   parts from its pair's is named, with "tree" or "patch".
# - The landing commit's message names X by its full SHA, X's green run by
#   its id, and says C was moved over landing commits only (the words
#   "moved" and "landing commits only"), with the trailers ``Moved-from: X``
#   (the full SHA) and ``Moved-from-run: <id>``.
#
# Cases listed in land_selftest-pending.json fail until land.py does what
# they hold it to; ``main`` holds each listed case to failing.

MOVED_TOPIC = f'{PHASE}-t1.99'
#: X's green run.
X_RUN = 4242
GIT_ENV = {
    'GIT_CONFIG_GLOBAL': os.devnull,
    'GIT_CONFIG_NOSYSTEM': '1',
    'GIT_AUTHOR_NAME': 'Test Person',
    'GIT_AUTHOR_EMAIL': 'test@example.com',
    'GIT_COMMITTER_NAME': 'Test Person',
    'GIT_COMMITTER_EMAIL': 'test@example.com',
    'GIT_TERMINAL_PROMPT': '0',
}


def git_env() -> dict[str, str]:
    # Drop GIT_DIR, GIT_INDEX_FILE and the like a hook running this sets.
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_ENV)
    return env


class Repo:
    """A bare ``origin`` and a clone of it to land from."""

    def __init__(self, tmp: Path) -> None:
        self.origin = tmp / 'origin.git'
        self.work = tmp / 'work'
        self.env = git_env()
        self.run_git(tmp, 'init', '-q', '--bare', str(self.origin))
        self.run_git(tmp, 'clone', '-q', str(self.origin), str(self.work))

    def run_git(self, where: Path, *args: str) -> str:
        proc = subprocess.run(
            ['git', *args],
            cwd=where,
            env=self.env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise AssertionError(f'git {args}: {proc.stderr.strip()}')
        return proc.stdout

    def git(self, *args: str) -> str:
        return self.run_git(self.work, *args).strip()

    def commit(
        self, subject: str, files: dict[str, str | None] | None = None
    ) -> str:
        """A commit on HEAD: each file written, or removed when None; no
        files is an empty commit."""
        for name, text in (files or {}).items():
            path = self.work / name
            if text is None:
                path.unlink()
            else:
                path.write_text(text)
            self.git('add', '-A', name)
        self.git('commit', '-q', '--allow-empty', '-m', subject)
        return self.git('rev-parse', 'HEAD')

    def remote(self, branch: str) -> str | None:
        proc = subprocess.run(
            ['git', 'rev-parse', '--verify', '--quiet', f'refs/heads/{branch}'],
            cwd=self.origin,
            env=self.env,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip() if proc.returncode == 0 else None


Commits = Sequence[tuple[str, dict[str, str | None] | None]]
LAND_98: Commits = (('Land T1.98: the earlier package', None),)
WORK_99: Commits = (('T1.99: a test package', {'a.txt': 'a\n'}),)


@dataclass
class Moved:
    """X, built on the earlier package's candidate A; the phase head H,
    A's landing commit (or what ``between`` seeds); and C, X moved onto H
    (X's commits cherry-picked, or what ``onto`` seeds)."""

    repo: Repo
    a: str
    x: str
    h: str
    c: str
    between: list[str]


def moved(
    tmp: Path,
    work: Commits = WORK_99,
    between: Commits = LAND_98,
    onto: Commits | None = None,
) -> Moved:
    r = Repo(tmp)
    r.git('checkout', '-q', '-b', 'start')
    r.commit('Land T1.97: the phase so far', {'base.txt': 'base\n'})
    a = r.commit('T1.98: the earlier package', {'b98.txt': '98\n'})
    r.git('checkout', '-q', '-b', 'x', a)
    for subject, files in work:
        x = r.commit(subject, files)
    r.git('checkout', '-q', '-b', 'phase', a)
    seeded = [r.commit(subject, files) for subject, files in between]
    h = seeded[-1] if seeded else a
    r.git('push', '-q', 'origin', f'{h}:refs/heads/{PHASE}')
    r.git('checkout', '-q', '-b', MOVED_TOPIC, h)
    if onto is None:
        r.git('cherry-pick', '--allow-empty', f'{a}..{x}')
    else:
        for subject, files in onto:
            r.commit(subject, files)
    r.git('fetch', '-q', 'origin')
    return Moved(r, a, x, h, r.git('rev-parse', 'HEAD'), seeded)


class Real:
    """land.py's git against a real repository; gh against ``Fake``."""

    def __init__(self, repo: Repo, runs: dict[str, list[dict[str, Any]]]):
        self.repo = repo
        self.fake = Fake(runs=runs)

    def __call__(
        self, argv: Sequence[str], *args: Any, **kw: Any
    ) -> land.Completed:
        if argv[0] == 'gh':
            return self.fake.gh(list(argv[1:]))
        # land.py's own runner, so its handling of the output (text mode,
        # a lone carriage return) is what is tested; ``in_repo`` puts it in
        # the repository with the test's git environment.
        return land.run_command(argv, *args, **kw)


@contextlib.contextmanager
def in_repo(repo: Repo) -> Iterator[None]:
    """Runs land.py in the repository's worktree with its git
    environment, restoring the directory and environment after."""
    cwd = os.getcwd()
    saved = dict(os.environ)
    try:
        os.chdir(repo.work)
        os.environ.clear()
        os.environ.update(repo.env)
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
        os.chdir(cwd)


class MovedCase:
    """One land.py invocation on a moved candidate."""

    def __init__(
        self,
        tmp: Path,
        m: Moved,
        runs: dict[str, list[dict[str, Any]]] | None = None,
        name: str | None = None,
    ) -> None:
        self.tmp = tmp
        self.m = m
        self.runs = (
            {m.x: [ci_run(m.x, MOVED_TOPIC, run_id=X_RUN)]}
            if runs is None
            else runs
        )
        named = m.x if name is None else name
        self.reviews = [
            self.review('a.txt', f'APPROVE\n\nReviewed {named[:7]}.\n'),
            self.review('b.txt', f'APPROVE\nCandidate {named}.\n'),
        ]
        self.lines: list[str] = []
        self.code: int | str | None = None

    def review(self, name: str, text: str) -> str:
        path = self.tmp / name
        path.write_text(text)
        return str(path)

    def land(self, *extra: str, moved_from: str | None = '') -> MovedCase:
        argv = [
            '--candidate', self.m.c[:12], '--topic-branch', MOVED_TOPIC,
            '--title', 'T1.99: a test package',
            '--protected-paths', '.github/workflows/ci.yml (no gate removed)',
        ]  # fmt: skip
        for path in self.reviews:
            argv += ['--review', path]
        if moved_from is not None:
            argv += ['--moved-from', moved_from or self.m.x[:12]]
        tools = land.Tools(
            run=Real(self.m.repo, self.runs), sleep=lambda _: None, repo='o/r'
        )
        try:
            with (
                in_repo(self.m.repo),
                contextlib.redirect_stderr(io.StringIO()) as err,
            ):
                self.code = land.main([*argv, *extra], tools, self.lines.append)
        except SystemExit as error:  # argparse: an unknown option
            self.code = f'SystemExit({error.code})'
            self.lines.append(err.getvalue().strip().splitlines()[-1])
        return self

    @property
    def text(self) -> str:
        return '\n'.join(self.lines)

    @property
    def refusal(self) -> str:
        return self.text.partition('REFUSED:')[2]

    def untouched(self) -> bool:
        r = self.m.repo
        return (
            r.git('rev-parse', 'HEAD') == self.m.c
            and r.remote(PHASE) == self.m.h
            and r.remote(MOVED_TOPIC) is None
        )

    def message(self) -> str:
        return self.m.repo.git('log', '-1', '--format=%B', 'HEAD')

    def trailers(self) -> dict[str, str]:
        """The landing commit's trailers, as git reads them."""
        text = self.m.repo.git(
            'log', '-1', '--format=%(trailers:only,unfold)', 'HEAD'
        )
        return {
            key: value
            for key, sep, value in (
                line.partition(': ') for line in text.splitlines()
            )
            if sep
        }


def refused_naming(
    c: MovedCase,
    *every: str,
    one_of: Sequence[str] = (),
    and_one_of: Sequence[str] = (),
) -> tuple[bool, str]:
    why = c.refusal
    ok = (
        c.code == 2
        and bool(why)
        and all(f in why for f in every)
        and all(
            not group or any(f in why for f in group)
            for group in (one_of, and_one_of)
        )
        and c.untouched()
    )
    return ok, (
        'refused naming the condition, nothing committed or pushed'
        if ok
        else f'exit {c.code}, wanted {list(every)} {list(one_of)} '
        f'{list(and_one_of)}: {c.text[-600:]}'
    )


@case("moved: lands C on X's reviews and green run over a landing commit")
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp)
    c = MovedCase(tmp, m).land()
    r = m.repo
    sha = r.git('rev-parse', 'HEAD')
    ok = (
        c.code == 0
        and f'LANDED {sha}' in c.text
        and r.git('rev-parse', 'HEAD^') == m.c
        and r.git('rev-parse', 'HEAD^{tree}')
        == r.git('rev-parse', f'{m.c}^{{tree}}')
        and r.remote(PHASE) == sha
        and r.remote(MOVED_TOPIC) == sha
        and f'Base-SHA: {m.h}' in c.message()
    )
    return ok, 'landed on C' if ok else f'exit {c.code}: {c.text[-600:]}'


@case('moved: lands over several landing commits')
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(
        tmp,
        between=(
            ('Land T1.98: the earlier package', None),
            ('Land T1.96: another lane', None),
            ('Land T1.95: a third', None),
        ),
    )
    c = MovedCase(tmp, m).land()
    ok = c.code == 0 and m.repo.remote(PHASE) == m.repo.git('rev-parse', 'HEAD')
    return ok, 'landed' if ok else f'exit {c.code}: {c.text[-600:]}'


@case('moved: lands a package of several commits, moved in order')
def _(tmp: Path) -> tuple[bool, str]:
    work: Commits = (
        ('T1.99: a test package', {'a.txt': 'a\n'}),
        ('T1.99 fix 1: review findings', {'a.txt': 'a2\n', 'c.txt': 'c\n'}),
    )
    m = moved(tmp, work=work)
    c = MovedCase(tmp, m).land()
    return c.code == 0, 'landed' if c.code == 0 else c.text[-600:]


@case(
    'moved: the landing message names X, its green run, and the move, '
    'with Moved-from and Moved-from-run trailers'
)
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp)
    c = MovedCase(tmp, m).land()
    message = c.message() if c.code == 0 else ''
    flat = ' '.join(message.split())
    trailers = c.trailers() if c.code == 0 else {}
    ok = (
        m.x in flat
        and re.search(rf'(?<!\d){X_RUN}(?!\d)', flat) is not None
        and re.search(r'\bmoved\b', flat) is not None
        and 'landing commits only' in flat
        and f'Base-SHA: {m.h}' in message
        and f'Checkpoint-SHA: {m.h}' in message
        and trailers.get('Moved-from') == m.x
        and trailers.get('Moved-from-run') == str(X_RUN)
    )
    return (
        ok,
        'names X, its run, the move' if ok else f'{c.text[-300:]} | {message}',
    )


@case('moved: --dry-run checks and changes nothing')
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp)
    c = MovedCase(tmp, m).land('--dry-run')
    ok = (
        c.code == 0
        and c.untouched()
        and 'nothing committed, nothing pushed' in c.text
        and m.x in ' '.join(c.text.split())
    )
    return ok, 'checked only' if ok else f'exit {c.code}: {c.text[-600:]}'


# Condition 1: X's evidence, as today's for a candidate.


@case('moved: refused, the reviews name C instead of X')
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp)
    runs = {
        m.x: [ci_run(m.x, MOVED_TOPIC, run_id=X_RUN)],
        m.c: [ci_run(m.c, MOVED_TOPIC, run_id=X_RUN + 1)],
    }
    c = MovedCase(tmp, m, runs=runs, name=m.c).land()
    return refused_naming(c, 'review', m.x[:12])


@case('moved: refused, the reviews name neither C nor X')
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp)
    return refused_naming(
        MovedCase(tmp, m, name=m.a).land(), 'review', m.x[:12]
    )


@case('moved: refused, one review names X and the other C')
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp)
    c = MovedCase(tmp, m)
    c.reviews[1] = c.review('b.txt', f'APPROVE\nCandidate {m.c}.\n')
    return refused_naming(c.land(), 'review', m.x[:12])


@case("moved: refused, X's newest run failed after a green one")
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp)
    runs = {
        m.x: [
            ci_run(m.x, MOVED_TOPIC, run_id=X_RUN),
            ci_run(m.x, MOVED_TOPIC, conclusion='failure', run_id=X_RUN + 7,
                   created='2026-10-09T05:00:00Z'),
        ],
        m.c: [ci_run(m.c, MOVED_TOPIC, run_id=X_RUN + 1)],
    }  # fmt: skip
    c = MovedCase(tmp, m, runs=runs).land()
    return refused_naming(c, 'not green', str(X_RUN + 7))


@case("moved: refused, X's run was cancelled")
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp)
    runs = {
        m.x: [ci_run(m.x, MOVED_TOPIC, conclusion='cancelled', run_id=X_RUN)],
        m.c: [ci_run(m.c, MOVED_TOPIC, run_id=X_RUN + 1)],
    }
    c = MovedCase(tmp, m, runs=runs).land()
    return refused_naming(c, 'not green', 'cancelled', str(X_RUN))


@case("moved: refused, X's run is unfinished")
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp)
    runs = {m.x: [ci_run(m.x, MOVED_TOPIC, status='in_progress', run_id=X_RUN)]}
    c = MovedCase(tmp, m, runs=runs).land()
    return refused_naming(c, 'not green', str(X_RUN))


@case('moved: refused, X has no CI run, though C has a green one')
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp)
    runs = {m.c: [ci_run(m.c, MOVED_TOPIC, run_id=X_RUN + 1)]}
    c = MovedCase(tmp, m, runs=runs).land()
    return refused_naming(c, 'no CI run', m.x[:12])


@case("moved: refused, X's only green run is a pull request's")
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp)
    runs = {m.x: [ci_run(m.x, MOVED_TOPIC, run_id=X_RUN, event='pull_request')]}
    c = MovedCase(tmp, m, runs=runs).land()
    return refused_naming(c, 'no CI run', m.x[:12])


# Condition 2: the same tree.


@case('moved: refused, C changes a file X does not')
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp, onto=(('T1.99: a test package', {'a.txt': 'other\n'}),))
    return refused_naming(MovedCase(tmp, m).land(), 'tree')


@case('moved: refused, C adds a commit that changes a file')
def _(tmp: Path) -> tuple[bool, str]:
    onto: Commits = (*WORK_99, ('T1.99 fix 1: one more', {'d.txt': 'd\n'}))
    m = moved(tmp, onto=onto)
    return refused_naming(MovedCase(tmp, m).land(), 'tree')


# Condition 3: the same patches, in order, none empty or extra.

TWO: Commits = (
    ('T1.99: a test package', {'a.txt': 'a\n'}),
    ('T1.99 fix 1: review findings', {'c.txt': 'c\n'}),
)


@case("moved: refused, C squashes X's two commits into one")
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(
        tmp,
        work=TWO,
        onto=(('T1.99: a test package', {'a.txt': 'a\n', 'c.txt': 'c\n'}),),
    )
    return refused_naming(MovedCase(tmp, m).land(), one_of=('patch', 'Patch'))


@case("moved: refused, C splits X's one commit into two")
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(
        tmp,
        work=(('T1.99: a test package', {'a.txt': 'a\n', 'c.txt': 'c\n'}),),
        onto=TWO,
    )
    return refused_naming(MovedCase(tmp, m).land(), one_of=('patch', 'Patch'))


@case("moved: refused, C has X's commits in another order")
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp, work=TWO, onto=(TWO[1], TWO[0]))
    return refused_naming(MovedCase(tmp, m).land(), one_of=('patch', 'Patch'))


@case('moved: refused, C adds an empty commit')
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp, onto=(*WORK_99, ('T1.99 note: nothing changed', None)))
    return refused_naming(
        MovedCase(tmp, m).land(), one_of=('patch', 'Patch', 'empty')
    )


@case('moved: refused, X has an empty commit C dropped')
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(
        tmp,
        work=(*WORK_99, ('T1.99 note: nothing changed', None)),
        onto=WORK_99,
    )
    return refused_naming(
        MovedCase(tmp, m).land(), one_of=('patch', 'Patch', 'empty')
    )


# Condition 4: only landing commits in X..H. Each seeded commit that changes
# files is undone by another, so C's tree and patches are X's. Under the
# generalized rule (X's patches are H's non-landing commits' then C's) such
# a commit is an extra patch on H's side, so the refusal names the landing
# commits, the patches or the trees; an empty commit, 'Landed' or a merge
# is named by its SHA.

#: The words a refusal of H's side may name its condition by.
H_SIDE = ('landing commit', 'patch', 'Patch', 'tree', 'empty', 'merge')

ADD_Z = {'z.txt': 'z\n'}
DROP_Z: dict[str, str | None] = {'z.txt': None}


@case('moved: refused, X..H has a commit that is not a landing commit')
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(
        tmp,
        between=(
            ('T1.97 fix 1: a stray change', ADD_Z),
            ('Land T1.98: the earlier package', None),
            ('T1.97 fix 2: undone', DROP_Z),
        ),
    )
    return refused_naming(MovedCase(tmp, m).land(), one_of=H_SIDE)


@case("moved: refused, X..H has a commit titled 'Land' that changes files")
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(
        tmp,
        between=(
            ('Land T1.98: the earlier package', ADD_Z),
            ('Land T1.96: another lane', DROP_Z),
        ),
    )
    return refused_naming(MovedCase(tmp, m).land(), one_of=H_SIDE)


@case("moved: refused, X..H has an empty commit not titled 'Land '")
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(
        tmp,
        between=(
            ('Land T1.98: the earlier package', None),
            ('T1.98 tracker: the earlier package landed', None),
        ),
    )
    return refused_naming(
        MovedCase(tmp, m).land(), m.between[1][:12], one_of=H_SIDE
    )


@case("moved: refused, X..H has an empty commit titled 'Landed'")
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp, between=(('Landed T1.98: the earlier package', None),))
    return refused_naming(
        MovedCase(tmp, m).land(), m.between[0][:12], one_of=H_SIDE
    )


@case('moved: refused, X..H has a merge commit')
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp)
    r = m.repo
    # A merge of the landing commit and an empty side commit, both trees
    # A's, made in place of the plain phase head.
    r.git('checkout', '-q', '-b', 'side', m.a)
    side = r.commit('Land T1.96: a side landing')
    r.git('checkout', '-q', 'phase')
    r.git('merge', '-q', '--no-ff', '-m', 'Land T1.98: merged', side)
    h = r.git('rev-parse', 'HEAD')
    r.git('push', '-q', 'origin', f'{h}:refs/heads/{PHASE}')
    r.git('checkout', '-q', '-B', MOVED_TOPIC, h)
    r.git('cherry-pick', f'{m.a}..{m.x}')
    r.git('fetch', '-q', 'origin')
    m = Moved(r, m.a, m.x, h, r.git('rev-parse', 'HEAD'), [h])
    return refused_naming(MovedCase(tmp, m).land(), h[:12], one_of=H_SIDE)


# Chained moves (the coordinator's generalized rule, replacing conditions 3
# and 4): a stack H0 -> P1 (X1) -> P2 (X2), each package reviewed and green
# on its original commits. P1 lands moved: its copies on H, then its landing
# commit. P2, moved onto H, lands with --moved-from X2, whose history holds
# P1's original commits while H holds P1's moved copies and landing commits.
# With B = merge-base(X, H): patch-ids(B..X) == patch-ids(B..H without its
# landing commits) ++ patch-ids(H..C), no merge, no empty commit in B..X or
# H..C, and, commit by commit, the trees of B..X equal those of the commits
# paired with them. A refusal names the patches or the trees.

#: The words a refusal of a chained move may name its condition by.
CHAIN_SIDE = ('patch', 'Patch', 'tree')


def package(n: int, *parts: str) -> Commits:
    """Package T1.9<n>: one commit per part, each adding its own file."""
    return tuple(
        (f'T1.9{n}{" " + part if part else ""}: package {n}',
         {f'p{n}{part.replace(" ", "")}.txt': f'p{n} {part}\n'})
        for part in (parts or ('',))
    )  # fmt: skip


def stack(
    tmp: Path,
    packages: Sequence[Commits],
    landed: dict[int, Commits] | None = None,
    onto: Commits | None = None,
) -> Moved:
    """X: the packages stacked on H0, each on the one before. H: H0, an
    earlier lane's landing commit, then each package but the last moved
    there (its commits cherry-picked, or ``landed[i]`` in their place) and
    its landing commit. C: the last package moved onto H (cherry-picked, or
    ``onto``)."""
    r = Repo(tmp)
    r.git('checkout', '-q', '-b', 'start')
    h0 = r.commit('Land T1.89: the phase so far', {'base.txt': 'base\n'})
    r.git('checkout', '-q', '-b', 'x', h0)
    tips = [h0]
    for commits in packages:
        for subject, files in commits:
            r.commit(subject, files)
        tips.append(r.git('rev-parse', 'HEAD'))
    r.git('checkout', '-q', '-b', 'phase', h0)
    seeded = [r.commit('Land T1.90: an earlier lane', None)]
    for i in range(len(packages) - 1):
        if landed and i in landed:
            for subject, files in landed[i]:
                seeded.append(r.commit(subject, files))
        else:
            r.git('cherry-pick', f'{tips[i]}..{tips[i + 1]}')
        seeded.append(r.commit(f'Land T1.9{i + 1}: package {i + 1}', None))
    h = r.git('rev-parse', 'HEAD')
    r.git('push', '-q', 'origin', f'{h}:refs/heads/{PHASE}')
    r.git('checkout', '-q', '-b', MOVED_TOPIC, h)
    if onto is None:
        r.git('cherry-pick', f'{tips[-2]}..{tips[-1]}')
    else:
        for subject, files in onto:
            r.commit(subject, files)
    r.git('fetch', '-q', 'origin')
    return Moved(r, h0, tips[-1], h, r.git('rev-parse', 'HEAD'), seeded)


LAST = (('T1.99: a test package', {'a.txt': 'a\n'}),)

for depth in (2, 3):

    @case(f'chained: a stack of {depth} packages, each moved, lands the last')
    def _(tmp: Path, depth: int = depth) -> tuple[bool, str]:
        m = stack(tmp, [*(package(n) for n in range(1, depth)), LAST])
        c = MovedCase(tmp, m).land()
        r = m.repo
        sha = r.git('rev-parse', 'HEAD')
        ok = (
            c.code == 0
            and r.git('rev-parse', 'HEAD^') == m.c
            and r.remote(PHASE) == sha
            and f'Base-SHA: {m.h}' in c.message()
        )
        return ok, 'landed' if ok else f'exit {c.code}: {c.text[-600:]}'


@case('chained: a package of two commits, moved, then the next lands')
def _(tmp: Path) -> tuple[bool, str]:
    m = stack(tmp, [package(1, 'a', 'b'), package(2), LAST])
    c = MovedCase(tmp, m).land()
    return c.code == 0, 'landed' if c.code == 0 else c.text[-600:]


@case("chained: refused, H's copy of P1 differs from P1, C fixes the tree")
def _(tmp: Path) -> tuple[bool, str]:
    changed = (('T1.91: package 1', {'p1.txt': 'p1 changed\n'}),)
    fixup = ('T1.99 fix 1: P1 as reviewed', {'p1.txt': 'p1 \n'})
    m = stack(tmp, [package(1), LAST], landed={0: changed}, onto=(*LAST, fixup))
    return refused_naming(MovedCase(tmp, m).land(), one_of=CHAIN_SIDE)


@case('chained: refused, H has a non-landing commit X lacks, C undoes it')
def _(tmp: Path) -> tuple[bool, str]:
    stray = ('T1.91 fix 1: not reviewed in X', ADD_Z)
    m = stack(
        tmp,
        [package(1), LAST],
        landed={0: (*package(1), stray)},
        onto=(*LAST, ('T1.99 fix 1: drop z', DROP_Z)),
    )
    return refused_naming(MovedCase(tmp, m).land(), one_of=CHAIN_SIDE)


@case("chained: refused, X has a commit of P1's that H lacks, C carries it")
def _(tmp: Path) -> tuple[bool, str]:
    p1 = package(1, 'a', 'b')
    m = stack(
        tmp,
        [p1, LAST],
        landed={0: p1[:1]},
        onto=(
            ('T1.99: a test package', {'a.txt': 'a\n', 'p1b.txt': 'p1 b\n'}),
        ),
    )
    return refused_naming(MovedCase(tmp, m).land(), one_of=CHAIN_SIDE)


@case("chained: refused, H has P1's commits in another order")
def _(tmp: Path) -> tuple[bool, str]:
    p1 = package(1, 'a', 'b')
    m = stack(tmp, [p1, LAST], landed={0: (p1[1], p1[0])})
    return refused_naming(MovedCase(tmp, m).land(), one_of=CHAIN_SIDE)


@case("chained: refused, C has the last package's commits in another order")
def _(tmp: Path) -> tuple[bool, str]:
    last = package(9, 'a', 'b')
    m = stack(tmp, [package(1), last], onto=(last[1], last[0]))
    return refused_naming(MovedCase(tmp, m).land(), one_of=CHAIN_SIDE)


# Patch-ids that agree where the code does not (the subagent review): the
# intermediate trees differ, the final trees are equal, and the move is
# refused, naming the trees and the commit, X's or C's, where they part.


@case("moved: refused, C's first commit differs after a lone carriage return")
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(
        tmp,
        work=(
            ('T1.99: a test package', {'f.txt': 'a\rX\n'}),
            ('T1.99 fix 1: settle f', {'f.txt': 'b\n'}),
        ),
        onto=(
            ('T1.99: a test package', {'f.txt': 'a\rY\n'}),
            ('T1.99 fix 1: settle f', {'f.txt': 'b\n'}),
        ),
    )
    x1 = m.repo.git('rev-parse', f'{m.x}^')
    c1 = m.repo.git('rev-parse', f'{m.c}^')
    return refused_naming(
        MovedCase(tmp, m).land(),
        one_of=CHAIN_SIDE,
        and_one_of=(x1[:12], c1[:12]),
    )


@case("moved: refused, C's intermediate commit differs in whitespace only")
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(
        tmp,
        work=(
            ('T1.99: a test package', {'f.txt': 'x = 1\n'}),
            ('T1.99 fix 1: settle f', {'f.txt': 'y\n'}),
        ),
        onto=(
            ('T1.99: a test package', {'f.txt': 'x  =  1 \n'}),
            ('T1.99 fix 1: settle f', {'f.txt': 'y\n'}),
        ),
    )
    x1 = m.repo.git('rev-parse', f'{m.x}^')
    c1 = m.repo.git('rev-parse', f'{m.c}^')
    return refused_naming(
        MovedCase(tmp, m).land(),
        one_of=CHAIN_SIDE,
        and_one_of=(x1[:12], c1[:12]),
    )


# Condition 5: the rest of land.py's checks still hold C.


@case('moved: refused, C is X itself, not moved onto the phase head')
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp)
    m.repo.git('checkout', '-q', '-B', MOVED_TOPIC, m.x)
    m = Moved(m.repo, m.a, m.x, m.h, m.x, m.between)
    return refused_naming(MovedCase(tmp, m).land(), 'does not descend')


@case('moved: refused, a dirty worktree')
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp)
    (m.repo.work / 'stray.txt').write_text('x\n')
    return refused_naming(MovedCase(tmp, m).land(), 'not clean')


@case('moved: refused, --moved-from is not a commit')
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp)
    return refused_naming(
        MovedCase(tmp, m).land(moved_from='f' * 12), 'is not a commit'
    )


# Without --moved-from nothing changes.


@case("unmoved: X's reviews and run do not land C without --moved-from")
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp)
    c = MovedCase(tmp, m).land(moved_from=None)
    return refused_naming(c, f'does not name the candidate {m.c[:12]}')


@case("unmoved: C's own reviews and run land it as before")
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp)
    runs = {m.c: [ci_run(m.c, MOVED_TOPIC, run_id=X_RUN + 1)]}
    c = MovedCase(tmp, m, runs=runs, name=m.c).land(moved_from=None)
    message = c.message() if c.code == 0 else ''
    ok = (
        c.code == 0
        and f'Candidate run {X_RUN + 1} green at {m.c}'
        in ' '.join(message.split())
        and m.x not in message
        and 'landing commits only' not in ' '.join(message.split())
        and m.repo.remote(PHASE) == m.repo.git('rev-parse', 'HEAD')
    )
    return ok, 'landed as before' if ok else f'{c.text[-300:]} | {message}'


@case("unmoved: C with no run of its own is refused though X's is green")
def _(tmp: Path) -> tuple[bool, str]:
    m = moved(tmp)
    c = MovedCase(tmp, m, name=m.c).land(moved_from=None)
    return refused_naming(c, 'no CI run', m.c[:12])


# --- CI4: a landing is verified on its candidate's run (decision 30) -----------
#
# The owner's decision 30 (2026-10-14): ci.yml skips the empty landing
# commits on the lane and phase branches, since a landing commit (one
# parent, no file changed, subject 'Land ', its candidate's tree) has its
# code tested by the candidate's green exact-SHA run. So land.py stops
# requiring the landing commit's own runs to be green.
#
# Interface the cases hold land.py to:
# - ``land.py verify L`` passes (exit 0, ``VERIFIED <L>``) when L changes no
#   file and its parent, the candidate, has a green exact-SHA push run as
#   its newest gating run, whether L's own runs are absent or skipped; for
#   a moved landing (``Moved-from: X``), when L's tree is X's, X's newest
#   gating run is green, and, since round 4, L's own run is green (see
#   there). The output names the run it stands on, by id.
# - It fails (nonzero, no ``VERIFIED``) when L changes files (naming that,
#   ``changes files`` or ``empty``), when a moved landing's tree is not X's
#   (naming ``tree``), when the candidate's or X's newest gating run is red
#   (exit 1, ``VERIFY FAILED``, the run by id and the revert instructions)
#   or cancelled (exit 1 or 3, the run by id), and when its only green run
#   is a pull request's (naming the missing push run and the commit). It
#   never complains that L has no run of its own.
# - A run on L that ran and failed still fails verify (exit 1, the run by
#   id, the revert instructions): ci.yml should skip it, so one that ran
#   and failed is real.
# - ``--wait-for-landing-run`` is refused as obsolete (exit 2, ``REFUSED``
#   naming the flag and the candidate's run, nothing committed or pushed),
#   whatever the candidate's run. It used to push L to the topic branch and
#   move the phase branch only once L's own run was green; ci.yml now skips
#   that run, so the flag could never land, and repurposing it to wait on
#   the candidate's run would keep a name that says the opposite. A
#   candidate whose run is not yet green is rerun or waited on, then landed
#   as usual. Refusing it, rather than dropping it from argparse, tells a
#   runbook's old command line why, not just 'unrecognized arguments'. The
#   refusal of a candidate whose run is not green no longer suggests it.
#
# The verify cases run git for real, as the moved cases do; gh is the fake.

#: The candidate's green run, and a red or cancelled one newer than it.
CAND_RUN = 3131
NEWER_RUN = 5151
#: Runs on the landing commit itself.
LAND_TOPIC_RUN = 7001
LAND_PHASE_RUN = 7002


def landing_on(
    r: Repo,
    base: str,
    files: dict[str, str] | None = None,
    moved_from: tuple[str, int] | None = None,
    to_phase: bool = True,
) -> str:
    """A landing commit on HEAD, with land.py's trailers (``Moved-from``
    and ``Moved-from-run`` when moved), pushed to the topic branch and,
    unless ``to_phase`` is false, the phase branch; empty unless ``files``
    are written."""
    for name, text in (files or {}).items():
        (r.work / name).write_text(text)
        r.git('add', '-A', name)
    moved = (
        f'Moved-from: {moved_from[0]}\nMoved-from-run: {moved_from[1]}\n'
        if moved_from
        else ''
    )
    message = (
        'Land T1.99: a test package\n\n'
        'Both reviews approve the candidate.\n\n'
        'Reviewed-by: independent reviewer (fresh agent) and Codex, T1.99\n'
        'Protected-paths: .github/workflows/ci.yml (no gate removed)\n'
        f'Base-SHA: {base}\n'
        f'Checkpoint-SHA: {base}\n'
        f'{moved}'
        f'{SIGNOFF}\n'
    )
    r.git('commit', '-q', '--allow-empty', '-m', message)
    sha = r.git('rev-parse', 'HEAD')
    refs = [f'{sha}:refs/heads/{MOVED_TOPIC}']
    if to_phase:
        refs.append(f'{sha}:refs/heads/{PHASE}')
    r.git('push', '-q', 'origin', *refs)
    return sha


@dataclass
class Landed:
    """A landing commit ``sha`` on its candidate ``cand``; for a moved
    landing, ``x`` is the commit it was moved from."""

    repo: Repo
    cand: str
    sha: str
    x: str | None = None


def plain_landed(
    tmp: Path,
    files: dict[str, str] | None = None,
    wrapped: bool = False,
    to_phase: bool = True,
) -> Landed:
    """A landing on a candidate; with ``wrapped``, the candidate is an
    empty ``Land`` commit on the package's untested commit, and ``cand``
    that commit below it."""
    r = Repo(tmp)
    r.git('checkout', '-q', '-b', 'start')
    base = r.commit('Land T1.97: the phase so far', {'base.txt': 'base\n'})
    r.git('push', '-q', 'origin', f'{base}:refs/heads/{PHASE}')
    r.git('checkout', '-q', '-b', MOVED_TOPIC)
    cand = r.commit('T1.99: a test package', {'a.txt': 'a\n'})
    if wrapped:
        r.commit('Land T1.99: an earlier landing, left on the branch')
    return Landed(r, cand, landing_on(r, base, files, to_phase=to_phase))


def moved_landed(tmp: Path, onto: Commits | None = None) -> Landed:
    m = moved(tmp, onto=onto)
    sha = landing_on(m.repo, m.h, moved_from=(m.x, X_RUN))
    return Landed(m.repo, m.c, sha, m.x)


def skipped_on(sha: str) -> list[dict[str, Any]]:
    """The landing commit's runs as ci.yml now leaves them: skipped."""
    return [
        ci_run(sha, MOVED_TOPIC, conclusion='skipped', run_id=LAND_TOPIC_RUN),
        ci_run(
            sha,
            PHASE,
            conclusion='skipped',
            run_id=LAND_PHASE_RUN,
            created='2026-10-09T00:00:01Z',
        ),
    ]


class Verified:
    """``land.py verify`` on a real landing commit, gh the fake."""

    def __init__(
        self,
        landed: Landed,
        runs: dict[str, list[dict[str, Any]]],
        stuck: Sequence[int] = (),
    ):
        self.landed = landed
        self.lines: list[str] = []
        self.code: int | str | None = None
        real = Real(landed.repo, runs)
        real.fake.stuck = set(stuck)
        tools = land.Tools(run=real, sleep=lambda _: None, repo='o/r')
        try:
            with (
                in_repo(landed.repo),
                contextlib.redirect_stderr(io.StringIO()) as err,
            ):
                self.code = land.main(
                    ['verify', landed.sha, '--phase-branch', PHASE],
                    tools,
                    self.lines.append,
                )
        except SystemExit as error:  # argparse
            self.code = f'SystemExit({error.code})'
            self.lines.append(err.getvalue().strip())

    @property
    def text(self) -> str:
        return '\n'.join(self.lines)


def verified_on(v: Verified, run_id: int) -> tuple[bool, str]:
    ok = (
        v.code == 0
        and f'VERIFIED {v.landed.sha}' in v.text
        and str(run_id) in v.text
    )
    return ok, (
        f'verified on run {run_id}'
        if ok
        else f'exit {v.code}, wanted VERIFIED naming run {run_id}: '
        f'{v.text[-600:]}'
    )


def failed_for(
    v: Verified,
    *every: str,
    one_of: Sequence[str] = (),
    codes: Sequence[int] = (1,),
    revert: bool = False,
) -> tuple[bool, str]:
    """Failed, naming each of ``every`` and one of ``one_of``, and not for
    want of the landing commit's own runs."""
    text = v.text
    ok = (
        v.code in codes
        and 'VERIFIED' not in text
        and all(f in text for f in every)
        and (not one_of or any(f in text for f in one_of))
        and f'no CI run on {v.landed.sha[:12]}' not in text
        and (not revert or 'Nothing was reverted' in text)
    )
    return ok, (
        'failed naming the reason'
        if ok
        else f'exit {v.code}, wanted {list(every)} {list(one_of)} '
        f'(codes {list(codes)}): {text[-600:]}'
    )


def cand_runs(landed: Landed, *conclusions: str) -> list[dict[str, Any]]:
    """The candidate's runs, oldest first: green, then ``conclusions``."""
    runs = [ci_run(landed.cand, MOVED_TOPIC, run_id=CAND_RUN)]
    for n, conclusion in enumerate(conclusions):
        runs.append(
            ci_run(
                landed.cand,
                MOVED_TOPIC,
                conclusion=conclusion,
                run_id=NEWER_RUN + n,
                created=f'2026-10-09T00:0{n + 1}:00Z',
            )
        )
    return runs


@case('CI4 verify: a landing with no runs of its own passes on the candidate')
def _(tmp: Path) -> tuple[bool, str]:
    landed = plain_landed(tmp)
    v = Verified(landed, {landed.cand: cand_runs(landed)})
    return verified_on(v, CAND_RUN)


@case(
    'CI4 verify: a landing whose own runs were skipped passes on the candidate'
)
def _(tmp: Path) -> tuple[bool, str]:
    landed = plain_landed(tmp)
    runs = {landed.cand: cand_runs(landed), landed.sha: skipped_on(landed.sha)}
    return verified_on(Verified(landed, runs), CAND_RUN)


@case('CI4 verify: fails, the landing commit changes a file')
def _(tmp: Path) -> tuple[bool, str]:
    landed = plain_landed(tmp, files={'a.txt': 'changed at landing\n'})
    v = Verified(landed, {landed.cand: cand_runs(landed)})
    return failed_for(v, one_of=('changes files', 'empty'))


@case("CI4 verify: fails, a moved landing's tree is not X's, X green")
def _(tmp: Path) -> tuple[bool, str]:
    landed = moved_landed(
        tmp, onto=(('T1.99: a test package', {'a.txt': 'not X\n'}),)
    )
    assert landed.x
    runs = {
        landed.x: [ci_run(landed.x, MOVED_TOPIC, run_id=X_RUN)],
        landed.sha: own_green(landed.sha),
    }
    return failed_for(Verified(landed, runs), 'tree')


@case("CI4 verify: fails, the candidate's newest run is red after a green one")
def _(tmp: Path) -> tuple[bool, str]:
    landed = plain_landed(tmp)
    v = Verified(landed, {landed.cand: cand_runs(landed, 'failure')})
    return failed_for(v, 'VERIFY FAILED', f'run {NEWER_RUN}', revert=True)


@case("CI4 verify: fails, the candidate's newest run was cancelled")
def _(tmp: Path) -> tuple[bool, str]:
    landed = plain_landed(tmp)
    v = Verified(landed, {landed.cand: cand_runs(landed, 'cancelled')})
    return failed_for(v, f'run {NEWER_RUN}', codes=(1, 3))


@case("CI4 verify: fails, X's newest run is red after the green one landed on")
def _(tmp: Path) -> tuple[bool, str]:
    landed = moved_landed(tmp)
    assert landed.x
    runs = {
        landed.x: [
            ci_run(landed.x, MOVED_TOPIC, run_id=X_RUN),
            ci_run(
                landed.x,
                MOVED_TOPIC,
                conclusion='failure',
                run_id=NEWER_RUN,
                created='2026-10-09T00:01:00Z',
            ),
        ],
        landed.sha: own_green(landed.sha),
    }
    v = Verified(landed, runs)
    return failed_for(v, 'VERIFY FAILED', f'run {NEWER_RUN}', revert=True)


@case("CI4 verify: fails, the candidate's only green run is a pull request's")
def _(tmp: Path) -> tuple[bool, str]:
    landed = plain_landed(tmp)
    runs = {
        landed.cand: [
            ci_run(
                landed.cand, MOVED_TOPIC, run_id=CAND_RUN, event='pull_request'
            )
        ]
    }
    v = Verified(landed, runs)
    ok, detail = failed_for(v, landed.cand[:12], codes=(1, 2, 3))
    named = bool(PUSH_RUN.search(v.text))
    return ok and named, detail if not ok else (
        detail if named else f'the push run is not named: {v.text[-600:]}'
    )


@case('CI4 verify: fails, a run on the landing commit ran and failed')
def _(tmp: Path) -> tuple[bool, str]:
    landed = plain_landed(tmp)
    own = skipped_on(landed.sha)
    own[1].update(conclusion='failure')
    runs = {landed.cand: cand_runs(landed), landed.sha: own}
    v = Verified(landed, runs)
    return failed_for(v, 'VERIFY FAILED', f'run {LAND_PHASE_RUN}', revert=True)


# --- CI4 round 4: a moved landing stands on its own run; evidence follows
# down; verify checks what the phase branch holds ---------------------------
#
# Codex's P1 on cb46bc5a: ci.yml runs a moved landing's jobs (E131 (c)):
# in a chained move X was judged against an older checkpoint's harness and
# baselines, so the landing's own run is the current checkpoint's
# judgement. ``land.py verify`` of a moved landing requires a green gating
# run on the landing commit itself (exit 0, naming it by id): none, only a
# skipped one, only a pull request's, a cancelled one or one still
# unfinished after the wait is incomplete (exit 3, ``VERIFY INCOMPLETE``,
# naming the landing commit, no revert instructions); a red one fails
# (exit 1, ``VERIFY FAILED``, the run by id, the revert instructions). Its
# tree is still X's, and a red newest run of X still fails. A plain
# landing, whose runs ci.yml skips, keeps candidate-only verification (the
# cases above).
#
# The subagent's P3-2: fix round 1 made a candidate that is itself an
# empty ``Land`` commit stand on the first commit below it that is none,
# in land.py's CI check and in verify, and made verify refuse a landing the
# phase branch on the remote does not hold. A wrapper's own run is never
# evidence.


def own_green(sha: str) -> list[dict[str, Any]]:
    """The moved landing's own run on the phase branch, green."""
    return [ci_run(sha, PHASE, run_id=LAND_PHASE_RUN)]


def x_green(landed: Landed) -> list[dict[str, Any]]:
    assert landed.x
    return [ci_run(landed.x, MOVED_TOPIC, run_id=X_RUN)]


def incomplete_for(v: Verified) -> tuple[bool, str]:
    text = v.text
    ok = (
        v.code == 3
        and 'VERIFY INCOMPLETE' in text
        and 'VERIFIED' not in text
        and v.landed.sha[:12] in text
        and 'Nothing was reverted' not in text
    )
    return ok, (
        'incomplete, naming the landing commit'
        if ok
        else f'exit {v.code}, wanted 3 and VERIFY INCOMPLETE naming '
        f'{v.landed.sha[:12]}: {text[-600:]}'
    )


@case('CI4 verify: a moved landing passes on its own green run')
def _(tmp: Path) -> tuple[bool, str]:
    landed = moved_landed(tmp)
    runs = {landed.x: x_green(landed), landed.sha: own_green(landed.sha)}
    return verified_on(Verified(landed, runs), LAND_PHASE_RUN)


@case('CI4 verify: a moved landing with no run of its own is incomplete')
def _(tmp: Path) -> tuple[bool, str]:
    landed = moved_landed(tmp)
    return incomplete_for(Verified(landed, {landed.x: x_green(landed)}))


@case('CI4 verify: a moved landing whose own runs were skipped is incomplete')
def _(tmp: Path) -> tuple[bool, str]:
    landed = moved_landed(tmp)
    runs = {landed.x: x_green(landed), landed.sha: skipped_on(landed.sha)}
    return incomplete_for(Verified(landed, runs))


@case('CI4 verify: a moved landing whose own run was cancelled is incomplete')
def _(tmp: Path) -> tuple[bool, str]:
    landed = moved_landed(tmp)
    own = own_green(landed.sha)
    own[0].update(conclusion='cancelled')
    runs = {landed.x: x_green(landed), landed.sha: own}
    return incomplete_for(Verified(landed, runs))


@case('CI4 verify: a moved landing whose own run never finishes is incomplete')
def _(tmp: Path) -> tuple[bool, str]:
    landed = moved_landed(tmp)
    own = [
        ci_run(landed.sha, PHASE, status='in_progress', run_id=LAND_PHASE_RUN)
    ]
    runs = {landed.x: x_green(landed), landed.sha: own}
    return incomplete_for(Verified(landed, runs, stuck=[LAND_PHASE_RUN]))


@case(
    "CI4 verify: a moved landing whose own green run is a pull request's is incomplete"
)
def _(tmp: Path) -> tuple[bool, str]:
    landed = moved_landed(tmp)
    own = [
        ci_run(landed.sha, PHASE, run_id=LAND_PHASE_RUN, event='pull_request')
    ]
    runs = {landed.x: x_green(landed), landed.sha: own}
    return incomplete_for(Verified(landed, runs))


@case('CI4 verify: fails, a moved landing whose own run is red')
def _(tmp: Path) -> tuple[bool, str]:
    landed = moved_landed(tmp)
    own = own_green(landed.sha)
    own[0].update(conclusion='failure')
    runs = {landed.x: x_green(landed), landed.sha: own}
    v = Verified(landed, runs)
    return failed_for(v, 'VERIFY FAILED', f'run {LAND_PHASE_RUN}', revert=True)


@case(
    'CI4 verify: a landing on an empty Land commit stands on the commit below'
)
def _(tmp: Path) -> tuple[bool, str]:
    landed = plain_landed(tmp, wrapped=True)
    return verified_on(
        Verified(landed, {landed.cand: cand_runs(landed)}), CAND_RUN
    )


@case(
    "CI4 verify: fails, a landing on an empty Land commit, only the wrapper's run green"
)
def _(tmp: Path) -> tuple[bool, str]:
    landed = plain_landed(tmp, wrapped=True)
    wrapper = landed.repo.git('rev-parse', f'{landed.sha}^')
    runs = {wrapper: [ci_run(wrapper, MOVED_TOPIC, run_id=CAND_RUN)]}
    v = Verified(landed, runs)
    return failed_for(v, landed.cand[:12], codes=(1, 3))


@case('CI4 verify: refuses a landing the phase branch on the remote lacks')
def _(tmp: Path) -> tuple[bool, str]:
    landed = plain_landed(tmp, to_phase=False)
    v = Verified(landed, {landed.cand: cand_runs(landed)})
    ok = (
        v.code not in (0, None)
        and 'VERIFIED' not in v.text
        and landed.sha[:12] in v.text
        and f'not in {PHASE}' in v.text
    )
    return ok, (
        'refused, naming the phase branch'
        if ok
        else f'exit {v.code}: {v.text[-600:]}'
    )


def wrapped_candidate(tmp: Path) -> tuple[Moved, str]:
    """A candidate that is an empty ``Land`` commit on the package's
    untested commit U, on the phase head; and U."""
    r = Repo(tmp)
    r.git('checkout', '-q', '-b', 'start')
    h = r.commit('Land T1.97: the phase so far', {'base.txt': 'base\n'})
    r.git('push', '-q', 'origin', f'{h}:refs/heads/{PHASE}')
    r.git('checkout', '-q', '-b', MOVED_TOPIC)
    u = r.commit('T1.99: a test package', {'a.txt': 'a\n'})
    e = r.commit('Land T1.99: an earlier landing, left on the branch')
    r.git('fetch', '-q', 'origin')
    return Moved(r, h, u, h, e, []), u


@case(
    'CI4 land: a candidate that is an empty Land commit on an untested commit is refused'
)
def _(tmp: Path) -> tuple[bool, str]:
    m, u = wrapped_candidate(tmp)
    c = MovedCase(tmp, m, runs={}, name=m.c).land(moved_from=None)
    return refused_naming(c, u[:12])


@case(
    'CI4 land: a candidate that is an empty Land commit, only its own run green, is refused'
)
def _(tmp: Path) -> tuple[bool, str]:
    m, u = wrapped_candidate(tmp)
    runs = {m.c: [ci_run(m.c, MOVED_TOPIC, run_id=CAND_RUN)]}
    c = MovedCase(tmp, m, runs=runs, name=m.c).land(moved_from=None)
    return refused_naming(c, u[:12])


@case(
    'CI4 land: a candidate that is an empty Land commit lands on the green run below it'
)
def _(tmp: Path) -> tuple[bool, str]:
    m, u = wrapped_candidate(tmp)
    runs = {u: [ci_run(u, MOVED_TOPIC, run_id=CAND_RUN)]}
    c = MovedCase(tmp, m, runs=runs, name=m.c).land(moved_from=None)
    sha = m.repo.git('rev-parse', 'HEAD')
    ok = (
        c.code == 0
        and f'LANDED {sha}' in c.text
        and m.repo.git('rev-parse', 'HEAD^') == m.c
        and m.repo.remote(PHASE) == sha
        and str(CAND_RUN) in c.message()
    )
    return (
        ok,
        'landed on the run below' if ok else f'exit {c.code}: {c.text[-600:]}',
    )


for conclusion in ('success', 'cancelled'):

    @case(
        f'CI4 land: --wait-for-landing-run is refused as obsolete '
        f'(candidate run {conclusion})'
    )
    def _(tmp: Path, conclusion: str = conclusion) -> tuple[bool, str]:
        runs = {CAND: [ci_run(CAND, TOPIC, conclusion=conclusion)]}
        c = Case(tmp, Fake(runs=runs))
        try:
            c.land('--wait-for-landing-run')
        except SystemExit as error:  # dropped from argparse
            return False, f'SystemExit({error.code}), not a refusal'
        why = c.text.partition('REFUSED:')[2]
        ok = (
            c.code == 2
            and 'wait-for-landing-run' in why
            and 'candidate' in why
            and c.untouched()
            and 'LANDED' not in c.text
        )
        return ok, (
            'refused, nothing committed or pushed'
            if ok
            else f'exit {c.code}, committed {len(c.fake.committed)}, pushes '
            f'{c.fake.pushes}: {c.text[-600:]}'
        )


@case('CI4 land: a cancelled candidate run is refused without the old advice')
def _(tmp: Path) -> tuple[bool, str]:
    runs = {CAND: [ci_run(CAND, TOPIC, conclusion='cancelled')]}
    c = Case(tmp, Fake(runs=runs)).land()
    ok, detail = refused(c, 'not green')
    if ok and 'wait-for-landing-run' in c.text:
        return False, f'still suggests --wait-for-landing-run: {c.text[-600:]}'
    return ok, detail


# --- T3.0: stacks of packages on one CI run, and the tooling train ------------
#
# The interfaces, as T3.0's implementer gives them:
#
# - ``land.py --stack <the usual land arguments>`` makes the package's empty
#   ``Land <ID>: ...`` commit locally, its Base-SHA and Checkpoint-SHA the
#   stack's tail (the previous stacked landing, or the phase head) and a
#   ``Stack-base: <phase head>`` trailer, and prints ``STACKED <sha>``. It
#   refuses past N packages (default 4). The next package is built on that
#   landing commit.
# - ``land.py stack-land --tip SHA [--topic-branch B]`` lands the chain in one
#   atomic push on one green CI run on the tip's tested commit, and prints
#   ``LANDED <tip>``. On a red run it exits 1 and prints ``IMPLICATED <job>:
#   <sha> <subject>``, ``PREFIX RUN: ...`` and ``GREEN PREFIX: land.py
#   stack-land --tip <Land Pk>``.
# - ``ratchet.py checkpoint --phase-push`` accepts a stack pushed at once, as
#   a Base-SHA chain; a plain ``land.py`` on a candidate stacked on unlanded
#   landings is refused, naming stack-land.
# - The train: ``scripts/rust_core/judging-paths.txt`` (protected-paths
#   syntax). A package changing a judging path is refused, naming ``land.py
#   --stack --train``, once the phase is past its first package. With
#   ``--train`` every changed file is a judging path and the package is
#   first in its stack; its landing carries ``Train: yes``.
# - ``land.py restack --tip SHA`` moves a stack onto a newer phase head,
#   keeping its reviews when every patch is the same, and prints
#   ``RESTACKED <new tip>``; only the new top needs a CI run.
# - ``land.py verify`` of a stacked landing stands on the run of a later
#   landing of the same stack (the same Stack-base) when its own candidate
#   has none.
# - A ``<ID> fix <n>:`` commit changing product code (any path that is no
#   judging path and not under tests/, contract/tests/, docs/, scripts/ or
#   .claude/) needs a ``Rule:`` line of at least three words, in the
#   commit-msg hook and in every range land.py lands.
#
# They run git for real, in temporary repositories (``Repo``), with gh the
# fake below. Each phase starts at a commit pushed to ``rust-core`` (the
# epic branch), and the phase head is a landing commit with its trailers, as
# the ratchet reads them. Listed in land_selftest-pending.json until land.py
# does what they hold it to.

T30_ROOT = Path(__file__).resolve().parents[2]
#: The top's green run, and the prefix's.
TOP_RUN = 9301
PREFIX_RUN = 9302
#: The phase's earlier landing in the train cases: a product change and a
#: judging-path change, so the phase is past its first package however
#: "the first package" is read (its first landing, or its first product
#: landing).
T30_EARLIER = {
    'base.txt': 'base 2\n',
    'scripts/rust_core/compare.py': '# the comparator, earlier\n',
}
#: A path no judging tool reads, and one every judging-path list names.
PRODUCT_FILE = 'notes/p{k}.txt'
JUDGING_FILE = 'scripts/rust_core/compare.py'
RUST_FILE = 'rust/crates/demo/src/lib.rs'


def t30_jobs(run_id: int, failing: Sequence[str]) -> list[dict[str, Any]]:
    """The jobs of a run: those named fail, the rest pass."""
    names = sorted({*failing, 'lint', 'rust-test', 'contract-python'})
    return [
        {
            'databaseId': run_id * 100 + n,
            'id': run_id * 100 + n,
            'name': name,
            'status': 'completed',
            'conclusion': 'failure' if name in failing else 'success',
            'steps': [
                {
                    'name': 'the step',
                    'number': 1,
                    'status': 'completed',
                    'conclusion': 'failure' if name in failing else 'success',
                }
            ],
        }
        for n, name in enumerate(names)
    ]


class StackGh:
    """gh for the stack cases: the runs on each SHA, and each run's jobs,
    through ``gh run list``, ``gh run view`` and ``gh api``."""

    def __init__(self, runs: dict[str, list[dict[str, Any]]]) -> None:
        self.runs = runs
        #: The failing jobs of each run, by id, and each run's failed log
        #: (``gh run view ID --log-failed``).
        self.failing: dict[int, list[str]] = {}
        self.logs: dict[int, str] = {}

    def by_id(self, run_id: int) -> dict[str, Any] | None:
        for runs in self.runs.values():
            for run in runs:
                if run['databaseId'] == run_id:
                    return run
        return None

    def __call__(self, args: list[str]) -> land.Completed:
        if args[:2] == ['run', 'list']:
            return Fake(runs=self.runs).gh(args)
        numbers = [int(n) for a in args for n in re.findall(r'\b(\d{4,})\b', a)]
        run = next((r for n in numbers if (r := self.by_id(n))), None)
        if run is None:
            return land.Completed(1, '', f'no such run: {args}')
        jobs = t30_jobs(
            run['databaseId'], self.failing.get(run['databaseId'], [])
        )
        if '--log-failed' in args:
            return land.Completed(0, self.logs.get(run['databaseId'], ''))
        if any('jobs' in a for a in args):
            return land.Completed(
                0, json.dumps({**run, 'total_count': len(jobs), 'jobs': jobs})
            )
        if args[:2] == ['run', 'view']:
            return land.Completed(0, json.dumps(run))
        if args[:1] == ['api']:
            return land.Completed(
                0,
                json.dumps(
                    {
                        'id': run['databaseId'],
                        'head_sha': run['headSha'],
                        'head_branch': run['headBranch'],
                        'event': run['event'],
                        'status': run['status'],
                        'conclusion': run['conclusion'] or None,
                        'html_url': run['url'],
                        'created_at': run['createdAt'],
                    }
                ),
            )
        return land.Completed(1, '', f'unexpected gh {args}')


class StackReal:
    """land.py's git against a real repository; gh against ``StackGh``."""

    def __init__(self, gh: StackGh) -> None:
        self.gh = gh

    def __call__(
        self, argv: Sequence[str], *args: Any, **kw: Any
    ) -> land.Completed:
        if argv[0] == 'gh':
            return self.gh(list(argv[1:]))
        return land.run_command(argv, *args, **kw)


def t30_topic(k: int) -> str:
    return f'{PHASE}-t1.9{k}'


def t30_package(k: int) -> str:
    return f'T1.9{k}'


class Stack:
    """A phase with landings, in a real repository: the epic branch's
    commit, an earlier package (its change ``earlier``) and its landing
    commit, the phase head ``h``. Packages are built on it one at a time,
    each on its own topic branch, and stacked or landed with land.py."""

    def __init__(
        self, tmp: Path, earlier: dict[str, str | None] | None = None
    ) -> None:
        self.tmp = tmp
        self.r = Repo(tmp)
        self.log = tmp / 'origin-updates.log'
        hook = self.r.origin / 'hooks' / 'update'
        hook.write_text(f'#!/bin/sh\necho "$1 $2 $3" >> "{self.log}"\n')
        hook.chmod(0o755)
        self.gh = StackGh({})
        self.lines: list[str] = []
        self.codes: list[int | str | None] = []
        self.r.git('checkout', '-q', '-b', 'start')
        judging = T30_ROOT / 'scripts/rust_core/judging-paths.txt'
        listed = (
            judging.read_text(encoding='utf-8')
            if judging.is_file()
            else 'scripts/rust_core/**\n'
        )
        epic = self.write_commit(
            'Merge Phase 0 into rust-core',
            {
                'base.txt': 'base\n',
                'scripts/rust_core/judging-paths.txt': listed,
                JUDGING_FILE: '# the comparator\n',
            },
        )
        self.r.git('push', '-q', 'origin', f'{epic}:refs/heads/rust-core')
        before = self.write_commit(
            'T1.89: the earlier package', earlier or {'base.txt': 'base 2\n'}
        )
        self.h = self.landing('T1.89', 'the earlier package', epic, before)
        self.r.git('push', '-q', 'origin', f'{self.h}:refs/heads/{PHASE}')
        self.r.git('fetch', '-q', 'origin')
        #: The stack's tail: the last landing made, the phase head at first.
        self.tail = self.h
        self.candidates: dict[int, str] = {}
        self.landings: dict[int, str] = {}

    def write_commit(self, subject: str, files: dict[str, str | None]) -> str:
        for name, text in files.items():
            path = self.r.work / name
            if text is None:
                path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            self.r.git('add', '-A', name)
        self.r.git('commit', '-q', '--allow-empty', '-m', subject)
        return self.r.git('rev-parse', 'HEAD')

    def landing(self, package: str, summary: str, base: str, on: str) -> str:
        """A landing commit as land.py makes one, on ``on``."""
        self.r.git('checkout', '-q', '--detach', on)
        self.r.git(
            'commit', '-q', '--allow-empty', '-m',
            f'Land {package}: {summary}\n\nBoth reviews approve it.\n\n'
            f'Reviewed-by: two reviewers, {package}\n'
            'Protected-paths: none (a test)\n'
            f'Base-SHA: {base}\nCheckpoint-SHA: {base}',
        )  # fmt: skip
        return self.r.git('rev-parse', 'HEAD')

    def build(self, k: int, files: dict[str, str | None] | None = None) -> str:
        """Package k's candidate on the stack's tail, on its topic branch,
        pushed there."""
        self.r.git('checkout', '-q', '-b', t30_topic(k), self.tail)
        sha = self.write_commit(
            f'{t30_package(k)}: package {k}',
            files or {PRODUCT_FILE.format(k=k): f'package {k}\n'},
        )
        self.r.git('push', '-q', 'origin', f'{sha}:refs/heads/{t30_topic(k)}')
        self.candidates[k] = sha
        return sha

    def reviews(self, k: int) -> list[str]:
        sha = self.candidates[k]
        out = []
        for name, text in (
            ('a', f'APPROVE\n\nReviewed {sha[:7]}; no findings.\n'),
            ('b', f'APPROVE\nCandidate {sha}.\n'),
        ):
            path = self.tmp / f'review-{t30_package(k)}-{name}.md'
            path.write_text(text)
            out.append(str(path))
        return out

    def call(self, argv: list[str]) -> int | str | None:
        tools = land.Tools(
            run=StackReal(self.gh), sleep=lambda _: None, repo='o/r'
        )
        lines: list[str] = []
        try:
            with (
                in_repo(self.r),
                contextlib.redirect_stderr(io.StringIO()) as err,
            ):
                code: int | str | None = land.main(argv, tools, lines.append)
        except SystemExit as error:  # argparse: an option it lacks
            code = f'SystemExit({error.code})'
            lines.append(err.getvalue().strip()[-400:])
        self.lines = lines
        self.codes.append(code)
        return code

    def land_args(self, k: int, *extra: str) -> list[str]:
        argv = [
            '--candidate', self.candidates[k],
            '--topic-branch', t30_topic(k),
            '--title', f'{t30_package(k)}: package {k}',
            '--protected-paths', 'none (a test)',
            '--phase-branch', PHASE,
        ]  # fmt: skip
        for path in self.reviews(k):
            argv += ['--review', path]
        return [*argv, *extra]

    def stack(self, k: int, *extra: str) -> int | str | None:
        """``land.py --stack`` for package k, from its topic branch; on
        ``STACKED`` the landing commit becomes the stack's tail."""
        code = self.call(['--stack', *self.land_args(k, *extra)])
        head = self.r.git('rev-parse', 'HEAD')
        if code == 0 and f'STACKED {head}' in self.text:
            self.landings[k] = head
            self.tail = head
        return code

    def stack_land(self, k: int, *extra: str) -> int | str | None:
        self.r.git('checkout', '-q', t30_topic(k))
        return self.call(
            [
                'stack-land', '--tip', self.landings[k],
                '--topic-branch', t30_topic(k),
                '--phase-branch', PHASE, *extra,
            ]
        )  # fmt: skip

    def green(self, k: int, run_id: int = TOP_RUN, **kw: Any) -> None:
        sha = self.candidates[k]
        self.gh.runs.setdefault(sha, []).insert(
            0, ci_run(sha, t30_topic(k), run_id=run_id, **kw)
        )

    @property
    def text(self) -> str:
        return '\n'.join(self.lines)

    def remote_phase(self) -> str | None:
        return self.r.remote(PHASE)

    def updates(self) -> list[tuple[str, str]]:
        """The phase branch's updates on the remote, (old, new), in order."""
        if not self.log.exists():
            return []
        out = []
        for line in self.log.read_text().splitlines():
            ref, old, new = line.split()
            if ref == f'refs/heads/{PHASE}':
                out.append((old, new))
        return out

    def trailers(self, sha: str) -> dict[str, str]:
        text = self.r.git('log', '-1', '--format=%(trailers:only,unfold)', sha)
        return {
            key.strip(): value.strip()
            for key, sep, value in (
                line.partition(': ') for line in text.splitlines()
            )
            if sep
        }

    def tree(self, rev: str) -> str:
        return self.r.git('rev-parse', f'{rev}^{{tree}}')

    def ratchet_checkpoint(self, tip: str, before: str) -> tuple[int, str]:
        """``ratchet.py checkpoint --phase-push BEFORE`` on ``tip``, as CI's
        phase check runs it, in a clone of the remote."""
        where = self.tmp / f'check-{tip[:8]}'
        self.r.run_git(self.tmp, 'clone', '-q', str(self.r.origin), str(where))
        self.r.run_git(where, 'checkout', '-q', '--detach', tip)
        proc = subprocess.run(
            [
                sys.executable,
                str(T30_ROOT / 'scripts/rust_core/ratchet.py'),
                'checkpoint', '--repo', str(where), '--phase-push', before,
            ],
            env={**self.r.env, 'SDM_PHASE_BRANCH': PHASE},
            capture_output=True,
            text=True,
            timeout=300,
        )  # fmt: skip
        return proc.returncode, proc.stdout + proc.stderr


def t30_stacked(tmp: Path, n: int = 4, files: Any = None) -> Stack:
    """A stack of n packages, each stacked with ``land.py --stack``."""
    s = Stack(tmp)
    for k in range(1, n + 1):
        s.build(k, (files or {}).get(k))
        code = s.stack(k)
        if k not in s.landings:
            raise AssertionError(
                f'--stack of {t30_package(k)}: exit {code}: {s.text[-600:]}'
            )
    return s


def t30_detail(s: Stack, ok: bool, good: str) -> tuple[bool, str]:
    return ok, good if ok else f'exits {s.codes}: {s.text[-700:]}'


@case(
    'T3.0 stack: four packages land on one green run on the top, a landing '
    'commit each, in order, in one push'
)
def _(tmp: Path) -> tuple[bool, str]:
    s = t30_stacked(tmp)
    s.green(4)
    before = len(s.updates())
    code = s.stack_land(4)
    tip = s.landings[4]
    if code != 0 or f'LANDED {tip}' not in s.text:
        return t30_detail(s, False, '')
    lands = [
        line.split('\x00')
        for line in s.r.git(
            'log', '--first-parent', '--reverse', '--format=%H%x00%s',
            f'{s.h}..{tip}',
        ).splitlines()
    ]  # fmt: skip
    landed = [
        (sha, subject) for sha, subject in lands if subject.startswith('Land ')
    ]
    problems = []
    if [subject.split(':')[0] for _, subject in landed] != [
        f'Land {t30_package(k)}' for k in range(1, 5)
    ]:
        problems.append(f'landing commits {[t for _, t in landed]}')
    previous = s.h
    for k, (sha, _) in enumerate(landed, start=1):
        trailers = s.trailers(sha)
        if s.tree(sha) != s.tree(f'{sha}^') or s.tree(sha) != s.tree(
            s.candidates.get(k, '')
        ):
            problems.append(
                f'Land {t30_package(k)} is not its candidate, empty'
            )
        if (
            trailers.get('Base-SHA') != previous
            or trailers.get('Checkpoint-SHA') != previous
            or trailers.get('Stack-base') != s.h
        ):
            problems.append(f'Land {t30_package(k)} trailers {trailers}')
        previous = sha
    if s.remote_phase() != tip:
        problems.append(f'the phase branch is at {s.remote_phase()}')
    if s.updates()[before:] != [(s.h, tip)]:
        problems.append(f'phase updates {s.updates()[before:]}, not one push')
    if problems:
        return False, '; '.join(problems) + ' | ' + s.text[-300:]
    return True, 'four landings in order, one push'


@case(
    "T3.0 stack: the ratchet's phase check accepts the stack's push, each "
    'landing checkpointed on the one before it'
)
def _(tmp: Path) -> tuple[bool, str]:
    s = t30_stacked(tmp)
    s.green(4)
    before = len(s.updates())
    if s.stack_land(4) != 0:
        return t30_detail(s, False, '')
    problems = []
    for old, new in s.updates()[before:]:
        code, out = s.ratchet_checkpoint(new, old)
        wanted = s.trailers(new).get('Checkpoint-SHA', '')
        first = out.split()[0] if code == 0 and out.split() else ''
        if code != 0 or first != wanted:
            problems.append(
                f'{old[:8]}..{new[:8]}: exit {code}, checkpoint {first[:12]} '
                f'not {wanted[:12]}: {out[-300:]}'
            )
    landed = [s.landings[k] for k in range(1, 5)]
    for k, sha in enumerate(landed):
        wanted = landed[k - 1] if k else s.h
        if s.trailers(sha).get('Checkpoint-SHA') != wanted:
            problems.append(f'Land {t30_package(k + 1)} not on the one before')
    if not s.updates()[before:]:
        problems.append('nothing pushed')
    return (not problems), '; '.join(
        problems
    ) or 'each checkpoint the landing before'


@case('T3.0 stack: a fifth package is refused by default, nothing made')
def _(tmp: Path) -> tuple[bool, str]:
    s = t30_stacked(tmp)
    c5 = s.build(5)
    code = s.stack(5)
    ok = (
        code == 2
        and 'REFUSED' in s.text
        and 5 not in s.landings
        and 'STACKED' not in s.text
        and '4' in s.text
        and s.r.git('rev-parse', 'HEAD') == c5
        and s.remote_phase() == s.h
    )
    return t30_detail(s, ok, 'refused past 4')


@case(
    'T3.0 stack: plain land.py on a candidate stacked on an unlanded landing '
    'is refused, naming stack-land'
)
def _(tmp: Path) -> tuple[bool, str]:
    s = t30_stacked(tmp, n=1)
    s.build(2)
    s.green(2)
    code = s.call(s.land_args(2))
    ok = (
        code == 2
        and 'stack-land' in s.text.partition('REFUSED:')[2]
        and s.remote_phase() == s.h
        and s.r.git('rev-parse', 'HEAD') == s.candidates[2]
    )
    return t30_detail(s, ok, 'refused naming stack-land')


@case(
    'T3.0 stack: a red top names the implicated commit and the green '
    'prefix, which then lands on its own run'
)
def _(tmp: Path) -> tuple[bool, str]:
    s = t30_stacked(tmp, files={3: {RUST_FILE: 'pub fn demo() {}\n'}})
    s.green(4, conclusion='failure')
    s.gh.failing[TOP_RUN] = ['rust-test']
    s.gh.logs[TOP_RUN] = (
        'rust-test\tcargo test\t2026-10-22T01:02:03Z error[E0425]: cannot '
        f'find value `x` in this scope\n --> {RUST_FILE}:1:16\n'
    )
    code = s.stack_land(4)
    lines = s.lines
    implicated = [
        line for line in lines if line.startswith('IMPLICATED rust-test:')
    ]
    prefix = [
        line.split('--tip', 1)[1].split()[0]
        for line in lines
        if line.startswith('GREEN PREFIX:') and '--tip' in line
    ]
    problems = []
    if code != 1:
        problems.append(f'exit {code}')
    if not any(
        'T1.93' in line or s.candidates[3][:7] in line for line in implicated
    ):
        problems.append(f'T1.93 not implicated by rust-test: {implicated}')
    if not prefix or not s.landings[2].startswith(prefix[0]):
        problems.append(f'green prefix {prefix}, not Land T1.92')
    if not any(line.startswith('PREFIX RUN:') for line in lines):
        problems.append('no PREFIX RUN line')
    if s.remote_phase() != s.h:
        problems.append('something landed on a red run')
    if problems:
        return False, '; '.join(problems) + ' | ' + s.text[-500:]
    s.green(2, run_id=PREFIX_RUN)
    code = s.stack_land(2)
    ok = (
        code == 0
        and f'LANDED {s.landings[2]}' in s.text
        and s.remote_phase() == s.landings[2]
    )
    return t30_detail(s, ok, 'implicated T1.93; the prefix landed')


@case(
    'T3.0 train: a judging-path change outside the train is refused, naming '
    'land.py --stack --train'
)
def _(tmp: Path) -> tuple[bool, str]:
    s = Stack(tmp, earlier=T30_EARLIER)
    s.build(1, {JUDGING_FILE: '# the comparator, changed\n'})
    s.green(1)
    code = s.call(s.land_args(1))
    why = s.text.partition('REFUSED:')[2]
    ok = (
        code == 2
        and 'land.py --stack --train' in ' '.join(why.split())
        and s.remote_phase() == s.h
        and s.r.git('rev-parse', 'HEAD') == s.candidates[1]
    )
    return t30_detail(s, ok, 'refused, naming the train')


@case(
    'T3.0 train: a judging-only package first in its stack lands as the '
    'train, with Train: yes, before the product package'
)
def _(tmp: Path) -> tuple[bool, str]:
    s = Stack(tmp, earlier=T30_EARLIER)
    s.build(1, {JUDGING_FILE: '# the comparator, changed\n'})
    if s.stack(1, '--train') != 0 or 1 not in s.landings:
        return t30_detail(s, False, '')
    s.build(2)
    if s.stack(2) != 0 or 2 not in s.landings:
        return t30_detail(s, False, '')
    s.green(2)
    code = s.stack_land(2)
    ok = (
        code == 0
        and s.remote_phase() == s.landings[2]
        and s.trailers(s.landings[1]).get('Train') == 'yes'
        and 'Train' not in s.trailers(s.landings[2])
        and s.trailers(s.landings[2]).get('Base-SHA') == s.landings[1]
    )
    return t30_detail(s, ok, 'the train landed first')


@case('T3.0 train: a --train package not first in its stack is refused')
def _(tmp: Path) -> tuple[bool, str]:
    s = Stack(tmp, earlier=T30_EARLIER)
    s.build(1)
    if s.stack(1) != 0 or 1 not in s.landings:
        return t30_detail(s, False, '')
    s.build(2, {JUDGING_FILE: '# the comparator, changed\n'})
    code = s.stack(2, '--train')
    ok = (
        code == 2
        and 'REFUSED' in s.text
        and 2 not in s.landings
        and s.r.git('rev-parse', 'HEAD') == s.candidates[2]
    )
    return t30_detail(s, ok, 'refused')


@case('T3.0 train: a --train package changing a product file is refused')
def _(tmp: Path) -> tuple[bool, str]:
    s = Stack(tmp, earlier=T30_EARLIER)
    s.build(
        1,
        {JUDGING_FILE: '# the comparator, changed\n', 'notes/p1.txt': 'p1\n'},
    )
    code = s.stack(1, '--train')
    ok = (
        code == 2
        and 'REFUSED' in s.text
        and 1 not in s.landings
        and s.r.git('rev-parse', 'HEAD') == s.candidates[1]
    )
    return t30_detail(s, ok, 'refused')


def t30_moved_on(s: Stack, files: dict[str, str | None]) -> str:
    """Another package lands on the phase branch while the stack waits:
    its commit and its landing commit on the phase head, pushed."""
    s.r.git('checkout', '-q', '--detach', s.h)
    work = s.write_commit('T1.88: another lane', files)
    head = s.landing('T1.88', 'another lane', s.h, work)
    s.r.git('push', '-q', 'origin', f'{head}:refs/heads/{PHASE}')
    s.r.git('fetch', '-q', 'origin')
    return head


@case(
    'T3.0 restack: a stack moved onto a newer phase head keeps its reviews, '
    'and lands on a new run on its top alone'
)
def _(tmp: Path) -> tuple[bool, str]:
    s = t30_stacked(tmp, n=2)
    s.green(2)
    newer = t30_moved_on(s, {'notes/other.txt': 'another lane\n'})
    code = s.call(['restack', '--tip', s.landings[2], '--phase-branch', PHASE])
    moved = [
        line.split()[1] for line in s.lines if line.startswith('RESTACKED ')
    ]
    if code != 0 or not moved:
        return t30_detail(s, False, '')
    tip = s.r.git('rev-parse', moved[0])
    first = s.r.git('rev-parse', f'{tip}^')
    lands = [
        sha
        for sha, _, subject in (
            line.partition('\x00')
            for line in s.r.git(
                'log', '--first-parent', '--reverse', '--format=%H%x00%s',
                f'{newer}..{tip}',
            ).splitlines()
        )
        if subject.startswith('Land ')
    ]  # fmt: skip
    problems = []
    if len(lands) != 2:
        problems.append(f'{len(lands)} landing commits on the new head')
    previous = newer
    for k, sha in enumerate(lands, start=1):
        trailers, old = s.trailers(sha), s.trailers(s.landings[k])
        if trailers.get('Reviewed-by') != old.get('Reviewed-by'):
            problems.append(f'Land T1.9{k} lost its reviews: {trailers}')
        if (
            trailers.get('Base-SHA') != previous
            or trailers.get('Stack-base') != newer
        ):
            problems.append(f'Land T1.9{k} trailers {trailers}')
        previous = sha
    # The old top's green run is no evidence for the moved stack.
    code = s.call(['stack-land', '--tip', tip, '--phase-branch', PHASE])
    if code == 0 or s.remote_phase() != newer:
        problems.append(f'landed on the old run: exit {code}')
    s.gh.runs[first] = [ci_run(first, t30_topic(2), run_id=PREFIX_RUN)]
    code = s.call(['stack-land', '--tip', tip, '--phase-branch', PHASE])
    if code != 0 or s.remote_phase() != tip:
        problems.append(f'not landed on the new top run: exit {code}')
    if problems:
        return False, '; '.join(problems) + ' | ' + s.text[-400:]
    return True, 'moved with its reviews; landed on the new top run'


@case('T3.0 restack: a stack whose patch changes on the new head is refused')
def _(tmp: Path) -> tuple[bool, str]:
    s = t30_stacked(tmp, n=2)
    newer = t30_moved_on(s, {'notes/p1.txt': 'another lane wrote it first\n'})
    code = s.call(['restack', '--tip', s.landings[2], '--phase-branch', PHASE])
    ok = (
        code == 2
        and 'REFUSED' in s.text
        and 'RESTACKED' not in s.text
        and s.remote_phase() == newer
    )
    return t30_detail(s, ok, 'refused, nothing moved')


@case(
    "T3.0 verify: a stacked landing stands on its stack top's green run, "
    'and fails when that run is red'
)
def _(tmp: Path) -> tuple[bool, str]:
    s = t30_stacked(tmp)
    s.green(4)
    if s.stack_land(4) != 0:
        return t30_detail(s, False, '')
    lower = s.landings[2]
    code = s.call(['verify', lower, '--phase-branch', PHASE])
    if code != 0 or f'VERIFIED {lower}' not in s.text:
        return t30_detail(s, False, '')
    s.green(4, run_id=TOP_RUN + 10, conclusion='failure')
    code = s.call(['verify', lower, '--phase-branch', PHASE])
    ok = code == 1 and 'VERIFIED' not in s.text
    return t30_detail(s, ok, 'verified on the top run; red fails')


def t30_fix(s: Stack, k: int, message: str) -> str:
    """A fix commit of package k on its topic branch, changing its product
    file, pushed."""
    sha = s.write_commit(message, {PRODUCT_FILE.format(k=k): f'fixed {k}\n'})
    s.r.git('push', '-q', 'origin', f'{sha}:refs/heads/{t30_topic(k)}')
    s.candidates[k] = sha
    return sha


@case(
    'T3.0 rule: land.py refuses a product fix without its Rule: line, and '
    'stacks it with one'
)
def _(tmp: Path) -> tuple[bool, str]:
    s = Stack(tmp)
    s.build(1)
    t30_fix(s, 1, 'T1.91 fix 1: keep the order')
    s.green(1)
    code = s.stack(1)
    why = s.text.partition('REFUSED:')[2]
    if code != 2 or 'Rule:' not in why or 1 in s.landings:
        return t30_detail(s, False, '')
    plain = s.call(s.land_args(1))
    if plain != 2 or 'Rule:' not in s.text or s.remote_phase() != s.h:
        return t30_detail(s, False, '')
    # The same fix with its rule, on a stack of its own, stacks.
    (tmp / 'with-rule').mkdir()
    s = Stack(tmp / 'with-rule')
    s.build(1)
    t30_fix(
        s,
        1,
        'T1.91 fix 1: keep the order\n\nRule: the old client keeps the '
        "caller's column order.",
    )
    code = s.stack(1)
    return t30_detail(s, code == 0 and 1 in s.landings, 'refused, then stacked')


# --- the pending list -----------------------------------------------------------

PENDING = Path(__file__).with_name('land_selftest-pending.json')


def pending_list() -> tuple[set[str], list[str]]:
    """The cases listed pending, and what breaks the list's rules: pending
    within authored, every name a case."""
    if not PENDING.exists():
        return set(), []
    loaded = json.loads(PENDING.read_text(encoding='utf-8'))
    pending = set(loaded['pending']['cases'])
    authored = set(loaded['authored']['cases'])
    names = {name for name, _ in CASES}
    problems = [
        f'{PENDING.name}: {name!r} is pending but was never authored'
        for name in sorted(pending - authored)
    ] + [
        f'{PENDING.name}: {name!r} is not a self-test'
        for name in sorted(authored - names)
    ]
    return pending, problems


def main() -> int:
    pending, problems = pending_list()
    results: list[Result] = []
    for name, check in CASES:
        with tempfile.TemporaryDirectory(prefix='sdm-land-') as tmp:
            try:
                ok, detail = check(Path(tmp))
            except Exception as error:  # a crash is a failed self-test
                ok, detail = False, f'raised {type(error).__name__}: {error}'
        detail = detail.replace('\n', ' | ')[:400]
        if name in pending:
            # Pending: it must still fail, as when it was written.
            mark = 'pend' if not ok else 'FAIL'
            if ok:
                detail = f'passes now: take it out of {PENDING.name} pending'
            ok = not ok
        else:
            mark = 'ok  ' if ok else 'FAIL'
        results.append(Result(name, ok, f'{mark} {name}: {detail}'))
    for result in results:
        print(result.detail)
    for problem in problems:
        print(f'FAIL {problem}')
    failed = [r for r in results if not r.ok]
    held = sum(1 for r in results if r.name in pending and r.ok)
    print(
        f'{len(results) - len(failed)} of {len(results)} self-tests as expected'
        f' ({held} pending, failing as listed in {PENDING.name})'
    )
    return 1 if failed or problems else 0


if __name__ == '__main__':
    sys.exit(main())
