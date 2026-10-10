#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Starts a package on a lane branch, shows the lanes, and checks that a
lane carries one package (decision E43, confirmed by the owner 2026-10-09;
package CI2).

A lane is a branch ``rust-core-phase-1-lane-<x>`` that a parallel line of
packages reuses as its topic branch, so a package's first CI run restores
the builds the lane's previous package saved. A lane carries one package
at a time: it starts at the phase head, takes that package's commits, and
returns to the phase head when ``land.py`` lands them.

    lane.py start --lane X --package ID [--worktree PATH] [--wait] [--dry-run]
    lane.py status [--lane X]
    lane.py check

``start`` prepares the lane for a new package. ``X`` is lower-case
letters and digits (``1b``), the package an ID such as ``T1.11e`` or
``CI3``. It reads the remote and refuses, changing nothing, when:

- the lane holds commits the phase branch lacks: a package that has not
  landed. It names the commits and the package they name. Abandoning that
  package is the owner's decision, so it only prints the manual steps
  (keep the tip under an archive branch with a normal push, then move the
  lane with ``--force-with-lease``), and never takes them;
- a CI run on the lane is queued or in progress (``--wait`` watches it
  instead, then looks again);
- the local lane branch holds commits the phase branch lacks, the worktree
  holding it has uncommitted changes, or the worktree path is taken.

Otherwise it creates the remote lane at the phase head, or fast-forwards
it there, each with a normal push (none when it is there already); makes
or reuses a local worktree on the lane at the phase head; records the
package in that worktree's own configuration (``git config --worktree
sdm.package ID``), first enabling ``extensions.worktreeConfig`` when the
clone lacks it, which it undoes and refuses if that would change any other
worktree's configuration; and prints the next steps. ``--dry-run`` makes
every check, prints what it would do, and pushes, creates and configures
nothing (it fetches objects only, updating no ref).

``status`` shows each lane on the remote: its tip against the phase head
(equal, behind, or ahead with its unlanded commits and their package),
the package its local worktree records, and its latest CI run.

``check`` is what ``prepush.sh`` runs: on a lane branch, every commit since
the phase head (``origin/rust-core-phase-1`` as last fetched) must name the
worktree's recorded package in its subject or body, and a lane branch with
no recorded package is refused. On any other branch it does nothing.

It never force-pushes, never deletes a branch or tag, and pushes nothing
but the lane: ``main``, ``rust-core`` and the phase branch are only read.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from land import (  # noqa: E402
    LANE_NAME,
    LANE_PREFIX,
    PACKAGE_ID,
    PACKAGE_KEY,
    PHASE_BRANCH,
    WORKFLOW,
    Refusal,
    Tools,
    commits_not_naming,
    lane_of,
    recorded_package,
    repo_of,
)

#: A package ID standing as a word in a commit message.
PACKAGE_WORD = re.compile(
    rf'(?<![A-Za-z0-9.])({PACKAGE_ID.pattern})(?![A-Za-z0-9])'
)
#: A run in one of these states has not finished.
ACTIVE = ('queued', 'in_progress', 'waiting', 'requested', 'pending')
RUN_FIELDS = 'databaseId,headSha,headBranch,status,conclusion,url,createdAt'
#: How many times ``--wait`` watches runs before giving up.
WAIT_ROUNDS = 10

Out = Callable[[str], None]


# --- names --------------------------------------------------------------------


def lane_branch(name: str) -> str:
    if not LANE_NAME.fullmatch(name):
        raise Refusal(
            f'the lane {name!r} is not a lane name: one to sixteen lower-case '
            'letters and digits, such as 1b'
        )
    return LANE_PREFIX + name


def package_id(package: str) -> str:
    if not PACKAGE_ID.fullmatch(package):
        raise Refusal(
            f'the package {package!r} is not a package ID such as T1.11e or CI3'
        )
    return package


# --- the remote ---------------------------------------------------------------


def remote_tips(tools: Tools, remote: str, *patterns: str) -> dict[str, str]:
    """Branch to SHA on the remote, for refs matching ``patterns``."""
    out = tools.git('ls-remote', '--heads', remote, *patterns)
    tips: dict[str, str] = {}
    for line in out.splitlines():
        sha, _, ref = line.partition('\t')
        if ref.startswith('refs/heads/'):
            tips[ref.removeprefix('refs/heads/')] = sha
    return tips


def fetch_objects(tools: Tools, remote: str, *branches: str) -> None:
    """Fetches the branches' commits without updating any ref."""
    refs = [f'refs/heads/{b}' for b in branches]
    tools.git(
        'fetch', '--quiet', '--refmap=', '--no-write-fetch-head', remote, *refs
    )


def is_ancestor(tools: Tools, a: str, b: str) -> bool:
    return tools.git_ok('merge-base', '--is-ancestor', a, b)


def commits(tools: Tools, base: str, head: str) -> list[tuple[str, str]]:
    """``base..head`` as (SHA, message), newest first."""
    out = tools.git('log', '--format=%H%x00%B%x1e', f'{base}..{head}')
    found: list[tuple[str, str]] = []
    for entry in out.split('\x1e'):
        sha, sep, body = entry.strip('\n').partition('\x00')
        if sep:
            found.append((sha, body))
    return found


def packages_named(found: Sequence[tuple[str, str]]) -> str:
    """The packages the commits name, the commonest first, as text."""
    counts: Counter[str] = Counter()
    for _, body in found:
        match = PACKAGE_WORD.search(body)
        counts[match.group(1) if match else '(none)'] += 1
    return ', '.join(
        f'{package} ({n} of {len(found)})'
        for package, n in counts.most_common()
    )


def first_package(found: Sequence[tuple[str, str]]) -> str | None:
    counts = Counter(
        m.group(1) for _, body in found if (m := PACKAGE_WORD.search(body))
    )
    return counts.most_common(1)[0][0] if counts else None


def lane_runs(
    tools: Tools, branch: str, limit: int = 20
) -> list[dict[str, object]]:
    out = tools.gh(
        'run', 'list', '--branch', branch, '--workflow', WORKFLOW,
        '--limit', str(limit), '--json', RUN_FIELDS,
    )  # fmt: skip
    runs = [r for r in json.loads(out or '[]') if r.get('headBranch') == branch]
    return sorted(runs, key=lambda r: str(r.get('createdAt', '')), reverse=True)


def describe_run(run: dict[str, object]) -> str:
    state = run.get('conclusion') or run.get('status')
    return (
        f'run {run.get("databaseId")}: {state} at '
        f'{str(run.get("headSha"))[:12]} {run.get("url") or ""}'
    ).rstrip()


def push_lane(tools: Tools, remote: str, sha: str, branch: str) -> None:
    """The only push lane.py makes: a normal, unforced push of ``sha`` to a
    lane branch, which the remote refuses unless it is a fast-forward."""
    if lane_of(branch) is None or not re.fullmatch(r'[0-9a-f]{40}', sha):
        raise Refusal(f'refusing to push {sha!r} to {branch!r}: not a lane')
    tools.git('push', '--quiet', remote, f'{sha}:refs/heads/{branch}')


# --- worktrees ----------------------------------------------------------------


@dataclass
class Worktree:
    path: str
    head: str = ''
    branch: str = ''
    bare: bool = False
    prunable: bool = False


def worktrees(tools: Tools) -> list[Worktree]:
    found: list[Worktree] = []
    for block in tools.git('worktree', 'list', '--porcelain').split('\n\n'):
        fields = dict(
            (line.partition(' ')[0], line.partition(' ')[2])
            for line in block.splitlines()
            if line
        )
        if 'worktree' not in fields:
            continue
        found.append(
            Worktree(
                path=fields['worktree'],
                head=fields.get('HEAD', ''),
                branch=fields.get('branch', ''),
                bare='bare' in fields,
                prunable='prunable' in fields,
            )
        )
    return found


def config_snapshot(tools: Tools, trees: Sequence[Worktree]) -> dict[str, str]:
    """Each live worktree's whole configuration as git reads it there."""
    snapshot: dict[str, str] = {}
    for tree in trees:
        if tree.bare or tree.prunable or not Path(tree.path).is_dir():
            continue
        done = tools.attempt('git', '-C', tree.path, 'config', '--list')
        lines = sorted(
            line
            for line in done.stdout.splitlines()
            if line.lower() != 'extensions.worktreeconfig=true'
        )
        snapshot[tree.path] = '\n'.join(lines)
    return snapshot


def worktree_config_on(tools: Tools) -> bool:
    done = tools.attempt(
        'git', 'config', '--bool', '--get', 'extensions.worktreeConfig'
    )
    return done.returncode == 0 and done.stdout.strip() == 'true'


def check_worktree_config_enablable(tools: Tools) -> None:
    """Git's documented precondition: a shared ``core.bare = true`` or
    ``core.worktree`` would apply to every worktree once the extension is
    on, so it must first move to the main worktree's own file."""
    common = tools.git('rev-parse', '--git-common-dir').strip()
    config = str(Path(common) / 'config')
    bare = tools.attempt(
        'git', 'config', '--file', config, '--get', 'core.bare'
    )
    tree = tools.attempt(
        'git', 'config', '--file', config, '--get', 'core.worktree'
    )
    if bare.stdout.strip() == 'true' or tree.stdout.strip():
        raise Refusal(
            f'{config} sets core.bare = true or core.worktree, which every '
            'worktree would read once extensions.worktreeConfig is on; move '
            "them to the main worktree's config.worktree first "
            '(git help worktree, CONFIGURATION FILE)'
        )


def dormant_worktree_configs(
    tools: Tools, trees: Sequence[Worktree]
) -> list[str]:
    """The worktrees' own ``config.worktree`` files that git ignores while
    the extension is off, and would read once it is on."""
    found: list[str] = []
    for tree in trees:
        if tree.bare or tree.prunable or not Path(tree.path).is_dir():
            continue
        where = tools.git(
            '-C', tree.path, 'rev-parse', '--path-format=absolute',
            '--git-path', 'config.worktree',
        ).strip()  # fmt: skip
        if Path(where).is_file() and Path(where).read_text().strip():
            found.append(where)
    return found


def enable_worktree_config(tools: Tools, out: Out) -> None:
    trees = worktrees(tools)
    before = config_snapshot(tools, trees)
    tools.git('config', 'extensions.worktreeConfig', 'true')
    after = config_snapshot(tools, trees)
    changed = sorted(p for p in before if before[p] != after.get(p))
    if changed:
        tools.git('config', '--unset', 'extensions.worktreeConfig')
        raise Refusal(
            'enabling extensions.worktreeConfig would change the '
            'configuration of ' + ', '.join(changed) + ' (a config.worktree '
            'file of their own), so it was turned off again; look at those '
            'files first'
        )
    out(
        'ok   enabled extensions.worktreeConfig; the configuration git '
        f'reads in each of the {len(before)} worktrees is unchanged'
    )


# --- start --------------------------------------------------------------------


@dataclass
class Plan:
    lane: str
    branch: str
    package: str
    phase: str
    remote_lane: str | None
    path: str
    holder: Worktree | None
    local: str | None
    enable_config: bool

    @property
    def push(self) -> str | None:
        if self.remote_lane is None:
            return 'create'
        if self.remote_lane != self.phase:
            return 'fast-forward'
        return None


def refuse_unlanded(
    remote: str,
    branch: str,
    lane: str,
    tip: str,
    phase: str,
    found: Sequence[tuple[str, str]],
) -> Refusal:
    package = first_package(found) or 'unknown'
    listed = '\n'.join(
        f'  {sha[:12]} {body.splitlines()[0] if body else ""}'
        for sha, body in found
    )
    archive = f'{PHASE_BRANCH}-archive-{lane}-{package}-{tip[:7]}'
    return Refusal(
        f'{branch} holds {len(found)} commit(s) {PHASE_BRANCH} lacks, naming '
        f'{packages_named(found)}: package {package} has not landed.\n'
        f'{listed}\n'
        f'Land {package} first (land.py), or start the new package on another '
        f"lane. Abandoning {package} is the owner's decision, and lane.py "
        'never does it. If the owner decides so, by hand: keep the tip under '
        'an archive branch (a normal push), then move the lane with a lease '
        'on that tip, and run lane.py start again:\n'
        f'  git push {remote} {tip}:refs/heads/{archive}\n'
        f'  git push --force-with-lease=refs/heads/{branch}:{tip} {remote} '
        f'{phase}:refs/heads/{branch}'
    )


def default_path(trees: Sequence[Worktree], lane: str) -> str:
    main = Path(trees[0].path)
    return str(main.parent / f'{main.name}-lane-{lane}')


def plan_start(args: argparse.Namespace, tools: Tools, out: Out) -> Plan | None:
    """Every check; the plan, or None when ``--wait`` watched runs and the
    remote must be read again."""
    branch = lane_branch(args.lane)
    package = package_id(args.package)
    tips = {
        b: s
        for b, s in remote_tips(
            tools, args.remote, PHASE_BRANCH, branch
        ).items()
        if b in (PHASE_BRANCH, branch)
    }
    if PHASE_BRANCH not in tips:
        raise Refusal(f'{args.remote} has no {PHASE_BRANCH}')
    phase = tips[PHASE_BRANCH]
    remote_lane = tips.get(branch)
    fetch_objects(tools, args.remote, *tips)
    out(f'ok   {PHASE_BRANCH} is at {phase}')

    if remote_lane is not None:
        active = [
            r for r in lane_runs(tools, branch) if r.get('status') in ACTIVE
        ]
        if active:
            if not args.wait:
                raise Refusal(
                    f'CI on {branch} has not finished; start the lane when it '
                    'has, or pass --wait:\n'
                    + '\n'.join(f'  {describe_run(r)}' for r in active)
                )
            for run in active:
                out(f'wait {describe_run(run)}')
                tools.attempt(
                    'gh', 'run', 'watch', str(run.get('databaseId')),
                    '--interval', '30',
                    *(('--repo', tools.repo) if tools.repo else ()),
                )  # fmt: skip
            return None
        if remote_lane != phase and not is_ancestor(tools, remote_lane, phase):
            found = commits(tools, phase, remote_lane)
            raise refuse_unlanded(
                args.remote, branch, args.lane, remote_lane, phase, found
            )
        out(
            f'ok   {branch} is at {remote_lane[:12]}, '
            + (
                'the phase head'
                if remote_lane == phase
                else 'behind the phase head, with nothing unlanded'
            )
        )
    else:
        out(f'ok   {branch} is not on {args.remote} yet')

    trees = worktrees(tools)
    holder = next(
        (t for t in trees if t.branch == f'refs/heads/{branch}'), None
    )
    path = str(
        Path(
            args.worktree
            or (holder.path if holder else default_path(trees, args.lane))
        )
        .expanduser()
        .resolve()
    )
    if holder and Path(holder.path).resolve() != Path(path):
        raise Refusal(
            f'{branch} is checked out at {holder.path}; use that worktree '
            f'(--worktree {holder.path}) or remove it first'
        )
    local = (
        tools.attempt(
            'git', 'rev-parse', '--verify', '--quiet', f'refs/heads/{branch}'
        ).stdout.strip()
        or None
    )
    if local and not is_ancestor(tools, local, phase):
        found = commits(tools, phase, local)
        raise Refusal(
            f'the local {branch} holds {len(found)} commit(s) {PHASE_BRANCH} '
            f'lacks, naming {packages_named(found)}:\n'
            + '\n'.join(
                f'  {sha[:12]} {body.splitlines()[0] if body else ""}'
                for sha, body in found
            )
            + '\npush and land them, or ask the owner; lane.py moves no '
            'branch that holds unlanded work'
        )
    if holder:
        dirty = tools.git(
            '-C', holder.path, 'status', '--porcelain', '--untracked-files=no'
        )
        if dirty.strip():
            raise Refusal(
                f'the worktree {holder.path} has uncommitted changes:\n'
                + dirty.rstrip()
            )
    elif Path(path).exists() and any(Path(path).iterdir()):
        raise Refusal(f'{path} exists and is not a worktree of {branch}')
    enable = not worktree_config_on(tools)
    if enable:
        check_worktree_config_enablable(tools)
        dormant = dormant_worktree_configs(tools, trees)
        if dormant:
            raise Refusal(
                'extensions.worktreeConfig is off, and enabling it would make '
                'git read ' + ', '.join(dormant) + ', changing those '
                "worktrees' configuration; look at those files first"
            )
    return Plan(
        lane=args.lane,
        branch=branch,
        package=package,
        phase=phase,
        remote_lane=remote_lane,
        path=path,
        holder=holder,
        local=local,
        enable_config=enable,
    )


def describe_plan(plan: Plan, remote: str) -> list[str]:
    steps: list[str] = []
    if plan.push == 'create':
        steps.append(
            f'push {plan.phase[:12]} to a new {plan.branch} on {remote}'
        )
    elif plan.push == 'fast-forward':
        steps.append(
            f'fast-forward {plan.branch} on {remote} from '
            f'{str(plan.remote_lane)[:12]} to {plan.phase[:12]} (a normal push)'
        )
    if plan.holder:
        if plan.holder.head != plan.phase:
            steps.append(
                f'fast-forward the worktree {plan.path} to {plan.phase[:12]}'
            )
    elif plan.local:
        steps.append(
            f'add a worktree {plan.path} on the local {plan.branch}, '
            f'fast-forwarded to {plan.phase[:12]}'
        )
    else:
        steps.append(
            f'add a worktree {plan.path} on a new local {plan.branch} at '
            f'{plan.phase[:12]}'
        )
    if plan.enable_config:
        steps.append(
            'enable extensions.worktreeConfig, if no other worktree changes'
        )
    steps.append(
        f'record {plan.package} in that worktree: git config --worktree '
        f'{PACKAGE_KEY} {plan.package}'
    )
    return steps


def execute(
    plan: Plan, args: argparse.Namespace, tools: Tools, out: Out
) -> None:
    remote = args.remote
    if plan.push:
        push_lane(tools, remote, plan.phase, plan.branch)
        now = remote_tips(tools, remote, plan.branch).get(plan.branch)
        if now != plan.phase:
            raise Refusal(
                f'after the push {remote} has {plan.branch} at {now}, not '
                f'{plan.phase}'
            )
        out(
            f'ok   {plan.branch} on {remote} is at the phase head ({plan.push})'
        )
    # The remote-tracking ref, for the local branch's upstream.
    tools.git('fetch', '--quiet', remote, f'refs/heads/{plan.branch}')
    if plan.holder is None:
        if plan.local:
            tools.git('worktree', 'add', '--quiet', plan.path, plan.branch)
        else:
            tools.git(
                'worktree', 'add', '--quiet', '--no-track', '-b', plan.branch,
                plan.path, plan.phase,
            )  # fmt: skip
        out(f'ok   worktree {plan.path} on {plan.branch}')
    head = tools.git('-C', plan.path, 'rev-parse', 'HEAD').strip()
    if head != plan.phase:
        tools.git('-C', plan.path, 'merge', '--quiet', '--ff-only', plan.phase)
        out(f'ok   fast-forwarded {plan.path} to {plan.phase[:12]}')
    tools.git(
        '-C', plan.path, 'branch', '--quiet',
        f'--set-upstream-to={remote}/{plan.branch}', plan.branch,
    )  # fmt: skip
    if plan.enable_config:
        enable_worktree_config(tools, out)
    previous = recorded_package(tools, plan.path)
    tools.git(
        '-C', plan.path, 'config', '--worktree', PACKAGE_KEY, plan.package
    )
    # Read back, as land.py and prepush.sh will read it.
    head = tools.git('-C', plan.path, 'rev-parse', 'HEAD').strip()
    on = tools.git('-C', plan.path, 'symbolic-ref', '--short', 'HEAD').strip()
    recorded = recorded_package(tools, plan.path)
    if (head, on, recorded) != (plan.phase, plan.branch, plan.package):
        raise Refusal(
            f'the worktree reads back {on} at {head[:12]} recording '
            f'{recorded}, not {plan.branch} at {plan.phase[:12]} recording '
            f'{plan.package}'
        )
    out(
        f'ok   {plan.path} records {plan.package}'
        + (
            f' (was {previous})'
            if previous and previous != plan.package
            else ''
        )
    )


def next_steps(plan: Plan, remote: str) -> list[str]:
    lines = [
        f'next: cd {plan.path}',
        f'  name {plan.package} in every commit (`git commit -s -m '
        f"'{plan.package}: ...'`); prepush.sh refuses a commit that does not",
        '  scripts/rust_core/prepush.sh before each push, --full before the '
        'last',
        f'  git push {remote} HEAD:{plan.branch}',
        '  land with scripts/rust_core/land.py --topic-branch '
        f"{plan.branch} --title 'Land {plan.package}: ...' (it checks the "
        'package), then land.py verify and prune_caches.py --delete',
        f"  the lane's next package: lane.py start --lane {plan.lane} "
        '--package <ID>',
    ]
    if plan.push:
        lines.insert(
            1,
            f'  the push starts a CI run on {plan.branch} at the phase head; '
            "it saves the lane's caches, and the package's first push waits "
            'for it',
        )
    return lines


def start(args: argparse.Namespace, tools: Tools, out: Out) -> int:
    plan = None
    for _ in range(WAIT_ROUNDS):
        plan = plan_start(args, tools, out)
        if plan is not None:
            break
    if plan is None:
        raise Refusal(
            f'runs on the lane kept starting for {WAIT_ROUNDS} rounds'
        )
    steps = describe_plan(plan, args.remote)
    if args.dry_run:
        out('dry run: every check passed; lane.py would:')
        out('\n'.join(f'  - {step}' for step in steps))
        out('dry run: nothing pushed, no worktree made, no configuration set')
        return 0
    execute(plan, args, tools, out)
    out(f'STARTED {plan.package} on {plan.branch} at {plan.phase}')
    out('\n'.join(next_steps(plan, args.remote)))
    return 0


# --- status -------------------------------------------------------------------


def status(args: argparse.Namespace, tools: Tools, out: Out) -> int:
    pattern = lane_branch(args.lane) if args.lane else LANE_PREFIX + '*'
    tips = remote_tips(tools, args.remote, PHASE_BRANCH, pattern)
    if PHASE_BRANCH not in tips:
        raise Refusal(f'{args.remote} has no {PHASE_BRANCH}')
    phase = tips.pop(PHASE_BRANCH)
    lanes = {b: s for b, s in sorted(tips.items()) if lane_of(b)}
    out(f'{PHASE_BRANCH} at {phase}')
    if not lanes:
        out(f'no lane branches on {args.remote}')
        return 0
    fetch_objects(tools, args.remote, PHASE_BRANCH, *lanes)
    holders = {
        t.branch.removeprefix('refs/heads/'): t for t in worktrees(tools)
    }
    for branch, tip in lanes.items():
        out(f'{branch}')
        if tip == phase:
            relation = 'equals the phase head'
        elif is_ancestor(tools, tip, phase):
            behind = len(commits(tools, tip, phase))
            relation = f'behind the phase head by {behind}, nothing unlanded'
        else:
            found = commits(tools, phase, tip)
            behind = len(commits(tools, tip, phase))
            relation = (
                f'AHEAD by {len(found)} unlanded commit(s), naming '
                f'{packages_named(found)}'
                + (f'; behind by {behind}' if behind else '')
            )
        out(f'  tip      {tip[:12]} {relation}')
        holder = holders.get(branch)
        if holder:
            recorded = recorded_package(tools, holder.path) or 'none recorded'
            out(f'  package  {recorded} (worktree {holder.path})')
        else:
            out('  package  no local worktree')
        runs = lane_runs(tools, branch, limit=1)
        out(f'  ci       {describe_run(runs[0]) if runs else "no run"}')
    return 0


# --- check (prepush.sh) -------------------------------------------------------


def check(args: argparse.Namespace, tools: Tools, out: Out) -> int:
    done = tools.attempt('git', 'symbolic-ref', '--quiet', '--short', 'HEAD')
    branch = done.stdout.strip()
    if not branch.startswith(LANE_PREFIX):
        out(
            f'not a lane branch ({branch or "detached HEAD"}); nothing to check'
        )
        return 0
    if lane_of(branch) is None:
        raise Refusal(
            f'{branch} is under {LANE_PREFIX} but is not a lane name: '
            'lower-case letters and digits'
        )
    package = recorded_package(tools)
    if package is None:
        raise Refusal(
            f'{branch} is a lane, and this worktree records no package '
            f'({PACKAGE_KEY} is unset; `git config --worktree {PACKAGE_KEY}`). '
            'Start the lane with scripts/rust_core/lane.py start'
        )
    package_id(package)
    base_ref = f'refs/remotes/{args.remote}/{PHASE_BRANCH}'
    base = tools.attempt(
        'git', 'rev-parse', '--verify', '--quiet', base_ref
    ).stdout.strip()
    if not base:
        raise Refusal(f'no {base_ref}; fetch {args.remote} first')
    missing = commits_not_naming(tools, base, 'HEAD', package)
    if missing:
        raise Refusal(
            f'{branch} carries {package}, and these commits since '
            f'{args.remote}/{PHASE_BRANCH} ({base[:12]}) do not name it, so '
            "they may be another package's (if the phase branch has moved, "
            'fetch and look again):\n'
            + '\n'.join(f'  {sha[:12]} {subject}' for sha, subject in missing)
        )
    count = len(commits(tools, base, 'HEAD'))
    out(f'ok   {branch} carries {package}: {count} commit(s), each naming it')
    return 0


# --- main ---------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(
        prog='lane.py', description=__doc__.split('\n\n', 1)[0]
    )
    top.add_argument('--remote', default='origin')
    top.add_argument('--repo', help='OWNER/NAME for gh; read from the remote')
    sub = top.add_subparsers(dest='command', required=True)
    begin = sub.add_parser('start', help='prepare a lane for a new package')
    begin.add_argument('--lane', required=True)
    begin.add_argument('--package', required=True)
    begin.add_argument('--worktree')
    begin.add_argument('--wait', action='store_true')
    begin.add_argument('--dry-run', action='store_true')
    show = sub.add_parser('status', help='show the lanes')
    show.add_argument('--lane')
    sub.add_parser('check', help="prepush.sh's check of a lane's commits")
    return top


def main(
    argv: Sequence[str] | None = None,
    tools: Tools | None = None,
    out: Out = print,
) -> int:
    args = parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    tools = tools or Tools()
    command = {'start': start, 'status': status, 'check': check}[args.command]
    try:
        # Names first, before git or gh is asked anything.
        if args.command == 'start':
            lane_branch(args.lane)
            package_id(args.package)
        elif args.command == 'status' and args.lane:
            lane_branch(args.lane)
        if args.command != 'check':
            tools.repo = args.repo or tools.repo or repo_of(tools, args.remote)
        return command(args, tools, out)
    except Refusal as why:
        out(f'REFUSED: {why}')
        return 2


if __name__ == '__main__':
    sys.exit(main())
