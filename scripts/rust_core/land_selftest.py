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
) -> dict[str, Any]:
    return {
        'databaseId': run_id,
        'headSha': sha,
        'headBranch': branch,
        'status': status,
        'conclusion': conclusion if status == 'completed' else '',
        'event': 'push',
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
        if args[0] == 'merge-base':
            a, b = self.sha(args[2]), self.sha(args[3])
            return land.Completed(
                0 if a == b or (a, b) in self.ancestors else 1
            )
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
            return land.Completed(0, json.dumps(self.runs.get(sha, [])))
        if args[:2] in (['run', 'watch'], ['run', 'view']):
            run_id = int(args[2])
            for runs in self.runs.values():
                for run in runs:
                    if run['databaseId'] == run_id:
                        if args[1] == 'watch':
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
            '--checkpoint', CHECK[:12],
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
        rf'Checkpoint-SHA: {CHECK}\n'
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


@case('refused: the checkpoint is not an ancestor of the candidate')
def _(tmp: Path) -> tuple[bool, str]:
    fake = Fake()
    fake.ancestors.discard((CHECK, CAND))
    return refused(Case(tmp, fake).land(), 'checkpoint')


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
            ('--quiet', 'origin', f'{LAND}:refs/heads/{TOPIC}'),
            ('--quiet', 'origin', f'{LAND}:refs/heads/{PHASE}'),
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
        and fake.pushes == [('--quiet', 'origin', f'{LAND}:refs/heads/{TOPIC}')]
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
    return refused(Case(tmp, fake).land(), 'do not name T1.99')


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


def names(text: str, package: str) -> bool:
    return land.names_package(text, package)


@case('names_package: a word of its own, case and suffix exact')
def _(tmp: Path) -> tuple[bool, str]:
    checks = {
        ('T1.11e: the work', 'T1.11e'): True,
        ('Fix (T1.11e).', 'T1.11e'): True,
        ('body\n\nPart of CI3.', 'CI3'): True,
        ('T1.11e', 'T1.11'): False,
        ('CI30', 'CI3'): False,
        ('t1.11e', 'T1.11e'): False,
        ('xT1.11e', 'T1.11e'): False,
        ('1.T1.11e', 'T1.11e'): False,
    }
    wrong = {k: v for k, v in checks.items() if names(*k) != v}
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
