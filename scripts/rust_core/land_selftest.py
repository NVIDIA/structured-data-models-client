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
held to having committed and pushed nothing. Run it as

    python3 scripts/rust_core/land_selftest.py

Decision E43; the checks themselves are documented in land.py.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import land  # noqa: E402

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

    def sha(self, rev: str) -> str | None:
        rev = rev.removesuffix('^{commit}')
        if rev == 'HEAD':
            return self.head
        if rev == f'refs/remotes/origin/{PHASE}':
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
        if args[0] == 'fetch':
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
            lines = [f'{refs[ref]}\t{ref}' for ref in args[2:] if ref in refs]
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
                    if branch == PHASE:
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
        return self.call(['verify', sha, '--appear-timeout', '0'])

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
        c = Case(tmp).land('--topic-branch', topic)
        ok, detail = refused(c, 'topic branch')
        # Refused before anything was asked of git or gh.
        return ok and not c.calls, detail


for phase in ('main', 'rust-core', f'{PHASE}-lane-a', 'rust-core-phase-2'):

    @case(f'refused: --phase-branch {phase}')
    def _(tmp: Path, phase: str = phase) -> tuple[bool, str]:
        c = Case(tmp).land('--phase-branch', phase)
        ok, detail = refused(c, 'phase branch')
        return ok and not c.calls, detail


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


@case('waits: a cancelled candidate run lands only after the landing run')
def _(tmp: Path) -> tuple[bool, str]:
    runs = {CAND: [ci_run(CAND, TOPIC, conclusion='cancelled')]}
    c = Case(tmp, Fake(runs=runs)).land('--wait-for-landing-run')
    pushes = c.fake.pushes
    ok = (
        c.code == 0
        and pushes
        == [
            ('--quiet', '--no-verify', 'origin', f'{LAND}:refs/heads/{TOPIC}'),
            ('--quiet', '--no-verify', 'origin', f'{LAND}:refs/heads/{PHASE}'),
        ]
        and 'fast-forwarded' in c.text
    )
    message = ' '.join((c.fake.committed or [''])[0].split())
    named = (
        f'Candidate CI not relied on (run 1: cancelled at {CAND}); the phase '
        "branch moves only when this commit's own exact-SHA run is green"
    ) in message and 'green at' not in message
    return ok and named, (
        'topic, watched, then phase; the message names the evidence'
        if ok and named
        else f'{pushes}: {message} | {c.text}'
    )


@case('waits: a red landing run leaves the phase branch where it was')
def _(tmp: Path) -> tuple[bool, str]:
    runs = {CAND: [ci_run(CAND, TOPIC, conclusion='cancelled')]}
    fake = Fake(runs=runs, landing_conclusion={TOPIC: 'failure'})
    c = Case(tmp, fake).land('--wait-for-landing-run')
    ok = (
        c.code == 2
        and fake.pushes
        == [('--quiet', '--no-verify', 'origin', f'{LAND}:refs/heads/{TOPIC}')]
        and fake.remote_phase == BASE
        and 'was not moved' in c.text
    )
    return ok, 'phase untouched' if ok else f'{fake.pushes}: {c.text}'


@case('waits: a green pull_request run on the landing commit is not its run')
def _(tmp: Path) -> tuple[bool, str]:
    # The push run on the topic branch goes red; a newer green pull_request
    # run on the same SHA and branch name does not stand for it.
    runs = {
        CAND: [ci_run(CAND, TOPIC, conclusion='cancelled')],
        LAND: [
            ci_run(
                LAND,
                TOPIC,
                run_id=99,
                created='2026-10-09T02:00:00Z',
                event='pull_request',
            )
        ],
    }
    fake = Fake(runs=runs, landing_conclusion={TOPIC: 'failure'})
    c = Case(tmp, fake).land('--wait-for-landing-run')
    ok = (
        c.code == 2
        and fake.pushes
        == [('--quiet', '--no-verify', 'origin', f'{LAND}:refs/heads/{TOPIC}')]
        and fake.remote_phase == BASE
        and 'was not moved' in c.text
    )
    return ok, 'phase untouched' if ok else f'{fake.pushes}: {c.text}'


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
        and c.text.count('ok  ') == 2
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


@case('verify: no run on the phase branch fails')
def _(tmp: Path) -> tuple[bool, str]:
    c = landed(tmp)
    c.fake.runs[LAND] = [
        r for r in c.fake.runs[LAND] if r['headBranch'] != PHASE
    ]
    c.verify()
    ok = c.code == 1 and f'no CI run on {LAND[:12]} on {PHASE}' in c.text
    return ok, 'reported' if ok else c.text


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


@case('waits: on a lane, the lane is confirmed after the phase push')
def _(tmp: Path) -> tuple[bool, str]:
    runs = {CAND: [ci_run(CAND, TOPIC, conclusion='cancelled')]}
    c = Case(tmp, Fake(runs=runs)).land('--wait-for-landing-run')
    ok = c.code == 0 and f'ok   {TOPIC} equals {PHASE} at {LAND}' in c.text
    return ok, 'confirmed' if ok else c.text


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
    ("is another package's", lambda: relanding(subject='Land T1.98: other'), "is not 'Land T1.99"),
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


def main() -> int:
    results: list[Result] = []
    for name, check in CASES:
        with tempfile.TemporaryDirectory(prefix='sdm-land-') as tmp:
            try:
                ok, detail = check(Path(tmp))
            except Exception as error:  # a crash is a failed self-test
                ok, detail = False, f'raised {type(error).__name__}: {error}'
        results.append(Result(name, ok, detail.replace('\n', ' | ')[:400]))
    for result in results:
        print(
            f'{"ok  " if result.ok else "FAIL"} {result.name}: {result.detail}'
        )
    failed = [r for r in results if not r.ok]
    print(
        f'{len(results) - len(failed)} of {len(results)} self-tests as expected'
    )
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
