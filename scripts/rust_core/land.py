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
        --checkpoint SHA [--phase-branch rust-core-phase-1] [--reviewer TEXT] \\
        [--summary TEXT] [--dry-run] [--wait-for-landing-run]
    land.py verify LANDING_SHA [--phase-branch rust-core-phase-1]

Before it changes anything it refuses, naming the reason, unless:

- the phase branch is exactly ``rust-core-phase-1`` and the topic branch a
  package branch under it (``rust-core-phase-1-<name>``, a lane
  ``rust-core-phase-1-lane-<x>`` included), never ``main``, ``rust-core``
  or anything else;
- the worktree is clean, HEAD is the candidate, and the candidate descends
  from the phase branch's head on the remote without being it;
- the checkpoint is a commit the candidate descends from;
- there are at least two review files, distinct, each whose first line is
  exactly ``APPROVE`` and which names the candidate by its full SHA or a
  prefix of at least seven characters that git resolves to the candidate
  and to nothing else;
- the newest CI run ``gh`` reports on the candidate's exact SHA completed
  with conclusion success; a cancelled, failed or unfinished run does not
  count (T1.8b's candidate run was cancelled).

It then commits the landing commit (``git commit --allow-empty -s``) with
``Reviewed-by``, ``Protected-paths``, ``Base-SHA`` (the phase head) and
``Checkpoint-SHA`` trailers, as earlier landing commits do, checks that the
commit's tree is the candidate's and its parent the candidate (so a
landing whose tree differs is never made), and pushes
it to the topic branch and the phase branch in one atomic push, which the
remote refuses if the phase branch moved. It prints the landing SHA and
the landing runs once they appear.

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
    proc = subprocess.run(list(argv), capture_output=True, text=True)
    return Completed(proc.returncode, proc.stdout, proc.stderr)


@dataclass
class Tools:
    """git and gh, through one function, so the tests can replace both."""

    run: Run = run_command
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


def check_descent(
    tools: Tools, candidate: str, base: str, checkpoint: str, phase: str
) -> None:
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
    if not tools.git_ok('merge-base', '--is-ancestor', checkpoint, candidate):
        raise Refusal(
            f'the checkpoint {checkpoint[:12]} is not an ancestor of the '
            f'candidate {candidate[:12]}'
        )


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


#: A hex word of seven to forty digits, standing on its own.
HEX_WORD = re.compile(r'(?<![0-9a-fA-F])[0-9a-fA-F]{7,40}(?![0-9a-fA-F])')


def names_candidate(tools: Tools, text: str, candidate: str) -> str | None:
    """None when the text names the candidate, by its full SHA or by an
    abbreviation git resolves to the candidate and to nothing else; else
    why not. An abbreviation that is the candidate's prefix but that git
    cannot resolve uniquely names no commit, so it is refused."""
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
            f'names the candidate only by {", ".join(ambiguous)}, which git '
            'does not resolve to the candidate alone (ambiguous); use the '
            'full SHA'
        )
    return (
        f'does not name the candidate {candidate[:12]} (its full SHA or an '
        f'unambiguous prefix of at least {MIN_PREFIX} characters)'
    )


def resolve_quiet(tools: Tools, rev: str) -> str | None:
    done = tools.attempt(
        'git', 'rev-parse', '--verify', '--quiet', rev + '^{commit}'
    )
    sha = done.stdout.strip()
    return sha if done.returncode == 0 and len(sha) == 40 else None


def check_reviews(
    tools: Tools, paths: Sequence[str], candidate: str
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
        why = names_candidate(tools, text, candidate)
        if why:
            problems.append(f'{raw}: {why}')
    if problems:
        raise Refusal(
            'the reviews do not approve the candidate:\n- '
            + '\n- '.join(problems)
        )
    return [Path(p) for p in paths]


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
    runs = [r for r in json.loads(out or '[]') if r.get('headSha') == sha]
    return sorted(runs, key=lambda r: str(r.get('createdAt', '')), reverse=True)


def describe(run: dict[str, object]) -> str:
    state = run.get('conclusion') or run.get('status')
    return f'run {run.get("databaseId")} on {run.get("headBranch")}: {state}'


def check_candidate_ci(tools: Tools, candidate: str) -> dict[str, object]:
    runs = runs_on(tools, candidate)
    if not runs:
        raise Refusal(
            f'no CI run on the candidate {candidate[:12]} itself; push it and '
            'let its run finish'
        )
    newest = runs[0]
    if (
        newest.get('status') != 'completed'
        or newest.get('conclusion') != 'success'
    ):
        raise Refusal(
            f'the newest CI run on the candidate is not green ({describe(newest)}); '
            'a cancelled, failed or unfinished run does not count. Wait for or '
            'rerun it, or land with --wait-for-landing-run'
        )
    return newest


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


def trailers_of(tools: Tools, sha: str) -> dict[str, str]:
    """The commit's trailers, as git reads them; the last value of a key."""
    text = tools.git('log', '-1', '--format=%(trailers:only,unfold)', sha)
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
    candidate = resolve(tools, args.candidate, 'the candidate')
    check_worktree(tools, candidate)
    base = phase_head(tools, args.remote, args.phase_branch)
    checkpoint = resolve(tools, args.checkpoint, 'the checkpoint')
    check_descent(tools, candidate, base, checkpoint, args.phase_branch)
    check_reviews(tools, args.review, candidate)
    out(f'ok   worktree clean, HEAD is the candidate {candidate}')
    out(f'ok   the candidate descends from {args.phase_branch} at {base}')
    out(f'ok   {len(args.review)} reviews approve {candidate[:12]}')
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
    else:
        green = check_candidate_ci(tools, candidate)
        out(f'ok   {describe(green)}, on {candidate[:12]} itself')
        evidence = (
            f'Candidate run {green.get("databaseId")} green at {candidate}; '
            'this commit changes no file, so the phase branch moves at once '
            'and its own runs are recorded afterwards (E43 (b)).'
        )
    title = one_line(args.title, '--title')
    package = package_of(title)
    reviewer = one_line(
        args.reviewer
        or f'independent reviewer (fresh agent) and Codex, {package}',
        '--reviewer',
    )
    summary = args.summary or (
        f'Both reviews approve the candidate {candidate[:7]}.'
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
        tools.git('push', '--quiet', args.remote, phase_ref)
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
    runs = wait_for_runs(tools, sha, [args.topic_branch, args.phase_branch])
    for branch in (args.topic_branch, args.phase_branch):
        run = runs.get(branch)
        out(
            f'run  {branch}: {run.get("databaseId")} {run.get("url")}'
            if run
            else f'run  {branch}: not yet listed; `land.py verify {sha}` finds it'
        )
    out(f'next: land.py verify {sha}')
    return 0


def push(tools: Tools, undo: str, *args: str) -> None:
    try:
        tools.git('push', '--quiet', *args)
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
        parser.add_argument('--checkpoint', required=True)
        parser.add_argument('--reviewer')
        parser.add_argument('--summary')
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--wait-for-landing-run', action='store_true')
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
