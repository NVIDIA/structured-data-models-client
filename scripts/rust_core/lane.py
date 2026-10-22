#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Starts a package on a lane branch, shows the lanes, and checks that a
lane carries one package (decision E43, confirmed by the owner 2026-10-09;
package CI2).

A lane is a branch ``rust-core-phase-N-lane-<x>`` that a parallel line of
packages reuses as its topic branch, so a package's first CI run restores
the builds the lane's previous package saved. A lane carries one package
at a time: it starts at the phase head, takes that package's commits, and
returns to the phase head when ``land.py`` lands them.

    lane.py [--phase-branch rust-core-phase-N] start --lane X --package ID
        --worktree PATH [--wait] [--dry-run]
    lane.py [--phase-branch rust-core-phase-N] status [--lane X]
    lane.py [--phase-branch rust-core-phase-N] pause --lane X [--resume]
    lane.py check [--pushed-refs FILE]

The phase branch is ``--phase-branch``, else ``SDM_PHASE_BRANCH``, else the
phase the branch checked out is under, else the phase in progress
(``phases.py``, T2.0); ``check`` measures a lane against its own phase.

A package owns a commit when the commit's subject starts with the
package's ID followed by a colon or a space and names no other package
ID: ``T1.11e: the payload``, ``T1.11e fix 1: review findings``,
``T1.11e tracker: the previous package landed``. Another package is
cited in the body (``Refs: T1.11d``), which owns nothing. The one
exemption is a merge of ``main`` (T2.0e): the commits one merge commit
the package owns brings in from ``main``, as the remote holds it when
``check`` runs (``git ls-remote``, then fetched, never a local ref), are
``main``'s and need no ID; a second merge, a merge of another branch or
of a rewritten ``main`` commit exempts nothing (``land.main_merge``). A
merge of ``main`` lands from the package's lane, like any package.

``start`` prepares the lane for a new package. ``X`` is lower-case
letters and digits (``1b``), the package an ID such as ``T1.11e`` or
``CI3``. It reads the remote and refuses, changing nothing, when:

- the lane holds commits the phase branch lacks: a package that has not
  landed. It names the commits and the package owning them. Abandoning that
  package is the owner's decision, so it only prints the manual steps
  (keep the tip under an archive branch with a normal push, then move the
  lane with ``--force-with-lease``), and never takes them;
- a CI run on the lane is queued or in progress (``--wait`` watches it
  instead, then looks again);
- the local lane branch holds commits the phase branch lacks, the worktree
  holding it has uncommitted changes or untracked files that are not
  ignored (they would carry into the next package), or the worktree path
  is taken;
- ``MAX_LANES`` lanes are active already (the owner's rule, 2026-10-22: at
  most three packages run at once on this machine; ``MAX_LANES`` in
  ``orchestration/heavy.conf``, or ``$SDM_HEAVY_CONFIG``, default 3). An
  active lane is a worktree on another lane of the phase whose recorded
  package has no landing (``Land <ID>: ...``) in the phase head's history
  and that is not paused. The refusal names them.

``pause --lane X`` marks the lane's worktree paused (``sdm.paused`` in its
own configuration, beside ``sdm.package``), which frees its slot;
``--resume`` takes the mark away. ``start`` clears it for the next package.

Otherwise it creates the remote lane at the phase head, or fast-forwards
it there, each with a normal push (none when it is there already); makes
or reuses a local worktree on the lane at the phase head; records the
package in that worktree's own configuration (``git config --worktree
sdm.package ID``), first enabling ``extensions.worktreeConfig`` when the
clone lacks it, which it undoes and refuses if that would change any other
worktree's configuration; refreshes the remote-tracking refs of the phase
branch and the lane (never a local phase branch), which ``check`` reads;
registers the worktree with ``prepush.sh``'s pre-push hook
(``prepush.sh --install-hook``); and prints the next steps. ``--dry-run`` makes
every check, prints what it would do, and pushes, creates and configures
nothing (it fetches objects only, updating no ref).

``status`` shows each lane on the remote: its tip against the phase head
(equal, behind, or ahead with its unlanded commits and their package),
the package its local worktree records, and its latest CI run.

``check`` is what ``prepush.sh`` runs: on a lane branch, the recorded
package must own every commit since the phase head
(``origin/rust-core-phase-N`` as last fetched), and a lane branch with no
recorded package is refused. On any other branch it does nothing. With
``--pushed-refs FILE`` (the pre-push hook's standard input, which the
hook passes on) it checks what is pushed instead: every lane ref pushed,
from whichever branch or detached HEAD, must be this worktree's lane,
hold only commits its recorded package owns, and not be a deletion.
The hook runs only in the worktrees ``prepush.sh --install-hook``
registered, which ``start`` does for each lane worktree; a push to a lane
from anywhere else, or with ``--no-verify``, is not checked here, and
``land.py`` is the backstop: it lands a lane only from the lane's own
worktree, on the lane's branch, with every commit owned by its package.

It never force-pushes, never deletes a branch or tag, and pushes nothing
but the lane: ``main``, ``rust-core`` and the phase branch are only read.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from land import (  # noqa: E402
    LANDING_SUBJECT,
    PACKAGE_ID,
    PACKAGE_KEY,
    WORKFLOW,
    Completed,
    Refusal,
    Tools,
    commits_not_owned,
    lane_of,
    owner_of,
    phase_named,
    phase_of_tools,
    recorded_package,
    repo_of,
)
from phases import (  # noqa: E402
    EPIC,
    LANE_NAME,
    Phase,
    PhaseError,
    phase_branch,
    phase_of,
)

#: prepush.sh, whose ``--install-hook`` start runs in the lane's worktree.
PREPUSH = Path(__file__).resolve().parent / 'prepush.sh'
#: A run in one of these states has not finished.
ACTIVE = ('queued', 'in_progress', 'waiting', 'requested', 'pending')
RUN_FIELDS = 'databaseId,headSha,headBranch,status,conclusion,url,createdAt'
#: How many times ``--wait`` watches runs before giving up, and the first
#: pause after ``gh run watch`` fails, doubled each round up to the cap.
WAIT_ROUNDS = 10
WAIT_SECONDS = 30
WAIT_CAP = 600
ZERO_SHA = '0' * 40
#: The owner's rule (2026-10-22): at most this many packages run at once on
#: this machine. heavy.conf's ``MAX_LANES`` sets it (T3.2 fix 4).
HEAVY_CONF = Path(__file__).resolve().parent / 'orchestration' / 'heavy.conf'
MAX_LANES = 3
#: A lane's worktree marked paused frees its slot (``lane.py pause``).
PAUSED_KEY = 'sdm.paused'

Out = Callable[[str], None]


# --- names --------------------------------------------------------------------


def lane_branch(name: str, phase: Phase) -> str:
    if not LANE_NAME.fullmatch(name):
        raise Refusal(
            f'the lane {name!r} is not a lane name: one to sixteen lower-case '
            'letters and digits, such as 1b'
        )
    return phase.lane_prefix + name


#: A package ID from Phase 3 on: an integer within its phase, no letter
#: (T3.1's rule, learnings.md: few big packages; T2.14a1 broke landings).
WHOLE_PACKAGE_ID = re.compile(r'T[0-9]+\.[0-9]+|CI[0-9]+')
#: The first phase whose package IDs are whole and whose briefs are checked
#: before a lane starts (T3.2).
WHOLE_FROM = 3


def package_id(package: str, phase: int | None = None) -> str:
    if not PACKAGE_ID.fullmatch(package):
        raise Refusal(
            f'the package {package!r} is not a package ID such as T1.11e or CI3'
        )
    if (
        phase is not None
        and phase >= WHOLE_FROM
        and not WHOLE_PACKAGE_ID.fullmatch(package)
    ):
        raise Refusal(
            f'the package {package!r} has a letter: from Phase {WHOLE_FROM} '
            'on a package ID is a whole number within its phase, such as '
            'T3.4 (fewer, bigger packages)'
        )
    return package


def brief_problems(brief: str | None) -> list[str]:
    r"""Why the package's brief may not start a lane (T3.2; learnings.md,
    lesson 16): ``brief_check.py`` finds a heading of the brief template
    missing from it, each named."""
    if brief is None:
        return []
    import brief_check

    try:
        headings = brief_check.required(brief_check.TEMPLATE)
        lacking = brief_check.missing(Path(brief), headings)
    except brief_check.Unreadable as error:
        return [f'the brief cannot be read: {error}']
    return [
        f'the brief {brief} lacks the heading {heading[3:]} '
        '(briefs/package.md; brief_check.py)'
        for heading in lacking
    ]


# --- the number of lanes ------------------------------------------------------


def max_lanes(environ: dict[str, str] | None = None) -> int:
    r"""``MAX_LANES`` from heavy.conf (``$SDM_HEAVY_CONFIG``, else the one
    beside ``heavy``), else :data:`MAX_LANES`."""
    environ = dict(os.environ) if environ is None else environ
    path = Path(environ.get('SDM_HEAVY_CONFIG') or HEAVY_CONF)
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return MAX_LANES
    for line in text.splitlines():
        key, sep, value = line.partition('=')
        if sep and key.strip() == 'MAX_LANES':
            if not value.strip().isdigit() or int(value) < 1:
                raise Refusal(
                    f'{path}: MAX_LANES={value.strip()!r} is not a whole number '
                    'of at least 1'
                )
            return int(value)
    return MAX_LANES


def paused(tools: Tools, path: str) -> bool:
    done = tools.attempt(
        'git', '-C', path, 'config', '--worktree', '--bool', '--get',
        PAUSED_KEY,
    )  # fmt: skip
    return done.returncode == 0 and done.stdout.strip() == 'true'


def active_lanes(
    tools: Tools, phase: Phase, head: str, starting: str
) -> list[tuple[str, str]]:
    r"""The lanes of ``phase`` other than ``starting`` whose worktree records
    a package not landed by ``head`` and is not paused, as (lane, package)."""
    landed = tools.git('log', '--first-parent', '--format=%s', head)
    subjects = landed.splitlines()
    found = []
    for tree in worktrees(tools):
        branch = tree.branch.removeprefix('refs/heads/')
        name = phase.lane_of(branch)
        if name is None or branch == starting or tree.prunable:
            continue
        package = recorded_package(tools, tree.path)
        if package is None or paused(tools, tree.path):
            continue
        if any(
            s.startswith((f'Land {package}:', f'Land {package} '))
            for s in subjects
        ):
            continue
        found.append((name, package))
    return found


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


def subject_package(subject: str) -> str | None:
    """The package a subject names first: its owner, or the package an
    earlier landing commit (``Land <ID>: ...``) lands."""
    landing = LANDING_SUBJECT.match(subject)
    return landing.group(1) if landing else owner_of(subject)


def owners(found: Sequence[tuple[str, str]]) -> Counter[str]:
    """How many of the commits each package owns, by their subjects."""
    return Counter(
        subject_package(body.split('\n', 1)[0]) or '(no package)'
        for _, body in found
    )


def packages_named(found: Sequence[tuple[str, str]]) -> str:
    """The packages owning the commits, the commonest first, as text."""
    return ', '.join(
        f'{package} ({n} of {len(found)})'
        for package, n in owners(found).most_common()
    )


def first_package(found: Sequence[tuple[str, str]]) -> str | None:
    counts = owners(found)
    counts.pop('(no package)', None)
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
    # --no-verify: it pushes the phase head, holding no package's commits,
    # from wherever lane.py runs, which the pre-push hook would refuse.
    tools.git(
        'push', '--quiet', '--no-verify', remote, f'{sha}:refs/heads/{branch}'
    )


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
    phase_branch: str
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
    phase_name: str,
) -> Refusal:
    package = first_package(found) or 'unnamed'
    listed = '\n'.join(
        f'  {sha[:12]} {body.splitlines()[0] if body else ""}'
        for sha, body in found
    )
    archive = f'{phase_name}-archive-{lane}-{package}-{tip[:7]}'
    return Refusal(
        f'{branch} holds {len(found)} commit(s) {phase_name} lacks, owned by '
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


def plan_start(
    args: argparse.Namespace, tools: Tools, out: Out, round_: int = 0
) -> Plan | None:
    """Every check; the plan, or None when ``--wait`` watched runs and the
    remote must be read again."""
    phase_name = phase_of_tools(tools).branch
    branch = lane_branch(args.lane, phase_of_tools(tools))
    package = package_id(args.package)
    tips = {
        b: s
        for b, s in remote_tips(tools, args.remote, phase_name, branch).items()
        if b in (phase_name, branch)
    }
    if phase_name not in tips:
        raise Refusal(f'{args.remote} has no {phase_name}')
    phase = tips[phase_name]
    remote_lane = tips.get(branch)
    fetch_objects(tools, args.remote, *tips)
    out(f'ok   {phase_name} is at {phase}')
    # From Phase 3 on (T3.2): whole package IDs, and the package's brief
    # holding every heading of the template.
    number = phase_of_tools(tools).number
    package_id(package, number)
    problems = brief_problems(getattr(args, 'brief', None))
    if problems:
        raise Refusal('\n'.join(problems))
    # A phase the epic branch holds is finished: no lane starts on it
    # (T3.2; T3.8's run made a Phase 2 lane from the rust-core checkout).
    # A phase just cut from the epic, its head the epic's own tip and no
    # landing of its own, is not.
    epic = remote_tips(tools, args.remote, EPIC).get(EPIC)
    if epic is not None:
        fetch_objects(tools, args.remote, EPIC)
        subject = tools.git('log', '-1', '--format=%s', phase).strip()
        own = subject.startswith(f'Land T{number}.') or subject.startswith(
            'Land CI'
        )
        if is_ancestor(tools, phase, epic) and (epic != phase or own):
            raise Refusal(
                f'{phase_name} is finished: {EPIC} already holds its head '
                f'{phase[:12]}; start the lane on the phase in progress '
                '(--phase-branch or SDM_PHASE_BRANCH)'
            )

    # The owner's rule (2026-10-22): at most MAX_LANES packages at once.
    limit = max_lanes()
    busy = active_lanes(tools, phase_of_tools(tools), phase, branch)
    if len(busy) >= limit:
        raise Refusal(
            f'{len(busy)} lane(s) are active, and at most {limit} packages run '
            'at once on this machine (the owner, 2026-10-22; MAX_LANES in '
            'heavy.conf): '
            + ', '.join(f'lane {name} ({pkg})' for name, pkg in busy)
            + '; land one, or free a slot with `lane.py pause --lane <x>`'
        )

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
                done = tools.attempt(
                    'gh', 'run', 'watch', str(run.get('databaseId')),
                    '--interval', '30',
                    *(('--repo', tools.repo) if tools.repo else ()),
                )  # fmt: skip
                if done.returncode != 0:
                    # Watching failed (gh, the network); pause, doubling
                    # each round, rather than spend the rounds at once.
                    pause = min(args.wait_seconds * 2**round_, WAIT_CAP)
                    why = (done.stderr or done.stdout).strip()[:200]
                    out(f'note gh run watch failed ({why}); again in {pause}s')
                    tools.sleep(pause)
            return None
        if remote_lane != phase and not is_ancestor(tools, remote_lane, phase):
            found = commits(tools, phase, remote_lane)
            raise refuse_unlanded(
                args.remote,
                branch,
                args.lane,
                remote_lane,
                phase,
                found,
                phase_name,
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
    path = str(Path(args.worktree).expanduser().resolve())
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
            f'the local {branch} holds {len(found)} commit(s) {phase_name} '
            f'lacks, owned by {packages_named(found)}:\n'
            + '\n'.join(
                f'  {sha[:12]} {body.splitlines()[0] if body else ""}'
                for sha, body in found
            )
            + '\npush and land them, or ask the owner; lane.py moves no '
            'branch that holds unlanded work'
        )
    if holder:
        # Untracked files that are not ignored would carry into the next
        # package; ignored build and cache files are kept for reuse.
        dirty = tools.git(
            '-C', holder.path, 'status', '--porcelain',
            '--untracked-files=normal',
        )  # fmt: skip
        if dirty.strip():
            raise Refusal(
                f'the worktree {holder.path} has uncommitted changes or '
                'untracked files, which would carry into the next package; '
                'commit, remove or ignore them:\n' + dirty.rstrip()
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
        phase_branch=phase_name,
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
    steps.append(
        f"refresh refs/remotes/{remote}/{plan.phase_branch} and the lane's "
        'tracking ref (no local phase branch)'
    )
    steps.append(
        "install prepush.sh's pre-push hook for that worktree "
        '(prepush.sh --install-hook)'
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
    # The remote-tracking refs: the lane's, for the local branch's
    # upstream, and the phase branch's, which `check` measures the lane's
    # commits from. Only refs/remotes/ is written; no local phase branch.
    tools.git(
        'fetch', '--quiet', remote,
        f'+refs/heads/{plan.phase_branch}'
        f':refs/remotes/{remote}/{plan.phase_branch}',
        f'+refs/heads/{plan.branch}:refs/remotes/{remote}/{plan.branch}',
    )  # fmt: skip
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
    if paused(tools, plan.path):
        # A new package runs: the pause was the previous one's.
        tools.git(
            '-C', plan.path, 'config', '--worktree', '--unset', PAUSED_KEY
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
    hooks_path = tools.attempt(
        'git', '-C', plan.path, 'config', '--get', 'core.hooksPath'
    ).stdout.strip()
    if hooks_path:
        # prepush.sh installs into the clone's hooks directory, which git
        # does not read while core.hooksPath points elsewhere.
        hook = Completed(
            1, '', f'core.hooksPath is {hooks_path}, so git would not run it'
        )
    else:
        hook = tools.attempt(
            'bash', '-c', 'cd "$1" && exec bash "$2" --install-hook',
            'lane.py', plan.path, str(PREPUSH),
        )  # fmt: skip
    if hook.returncode == 0:
        out(f"ok   prepush.sh's pre-push hook checks pushes from {plan.path}")
    else:
        out(
            'note the pre-push hook was not installed '
            f'({(hook.stderr or hook.stdout).strip()[:200]}); run prepush.sh '
            'by hand before each push; land.py checks the lane again'
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
        f'  start every commit subject with {plan.package}: '
        f"'{plan.package}: ...' or '{plan.package} tracker: ...' "
        '(prepush.sh and land.py refuse any other); squash fixup! and '
        'squash! commits (git rebase --autosquash) before pushing',
        '  scripts/rust_core/handoff.py run --package '
        f'{plan.package} before reporting ready (reviews refuse a SHA '
        'without its stamp); scripts/rust_core/prepush.sh (the cheap tier, '
        'never --full) before each push',
        f'  git push {remote} HEAD:{plan.branch}',
        '  land with scripts/rust_core/land.py --stack --topic-branch '
        f"{plan.branch} --title 'Land {plan.package}: ...' (it checks the "
        'package), then land.py stack-land --tip <landing>, land.py verify '
        'and prune_caches.py --delete',
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
    for round_ in range(WAIT_ROUNDS):
        plan = plan_start(args, tools, out, round_)
        if plan is not None:
            break
    if plan is None:
        raise Refusal(
            f'CI on the lane had not finished after {WAIT_ROUNDS} rounds of '
            'watching; look at it, then start again'
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


# --- pause --------------------------------------------------------------------


def pause(args: argparse.Namespace, tools: Tools, out: Out) -> int:
    assert tools.phase is not None
    branch = lane_branch(args.lane, tools.phase)
    holder = next(
        (t for t in worktrees(tools) if t.branch == f'refs/heads/{branch}'),
        None,
    )
    if holder is None or recorded_package(tools, holder.path) is None:
        raise Refusal(f'no worktree here holds {branch} with a package')
    if args.resume:
        if paused(tools, holder.path):
            tools.git(
                '-C', holder.path, 'config', '--worktree', '--unset',
                PAUSED_KEY,
            )  # fmt: skip
        out(f'RESUMED lane {args.lane} ({holder.path})')
    else:
        tools.git('-C', holder.path, 'config', '--worktree', PAUSED_KEY, 'true')
        out(f'PAUSED lane {args.lane} ({holder.path}): its slot is free')
    return 0


# --- status -------------------------------------------------------------------


def status(args: argparse.Namespace, tools: Tools, out: Out) -> int:
    current = phase_of_tools(tools)
    name = current.branch
    pattern = (
        lane_branch(args.lane, current)
        if args.lane
        else current.lane_prefix + '*'
    )
    tips = remote_tips(tools, args.remote, name, pattern)
    if name not in tips:
        raise Refusal(f'{args.remote} has no {name}')
    phase = tips.pop(name)
    lanes = {b: s for b, s in sorted(tips.items()) if current.lane_of(b)}
    out(f'{name} at {phase}')
    if not lanes:
        out(f'no lane branches on {args.remote}')
        return 0
    fetch_objects(tools, args.remote, name, *lanes)
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


def check_lane_range(
    tools: Tools, remote: str, branch: str, head: str, package: str
) -> int:
    """Refuses a commit since the head of the phase ``branch`` is a lane
    of that ``package`` does not own; the number of commits checked."""
    phase_name = phase_of(branch)
    base_ref = f'refs/remotes/{remote}/{phase_name}'
    base = tools.attempt(
        'git', 'rev-parse', '--verify', '--quiet', base_ref
    ).stdout.strip()
    if not base:
        raise Refusal(f'no {base_ref}; fetch {remote} first')
    # A package stacked on approved packages' landing commits (land.py
    # --stack, T3.0) carries the stack below it: checked as a stack.
    stray = commits_not_owned(
        tools, base, head, package, remote=remote, stacked=True
    )
    if stray:
        raise Refusal(
            f'{branch} carries {package}, and these commits since '
            f'{remote}/{phase_name} ({base[:12]}) are not owned by it (a '
            f"subject '{package}: ...' or '{package} <part>: ...', naming no "
            "other package), so they may be another package's; if the phase "
            'branch has moved, fetch and look again:\n'
            + '\n'.join(
                f'  {sha[:12]} {subject} ({why})' for sha, subject, why in stray
            )
        )
    return len(commits(tools, base, head))


def worktree_package(tools: Tools, branch: str) -> str:
    package = recorded_package(tools)
    if package is None:
        raise Refusal(
            f'{branch} is a lane, and this worktree records no package '
            f'({PACKAGE_KEY} is unset; `git config --worktree {PACKAGE_KEY}`). '
            'Start the lane with scripts/rust_core/lane.py start'
        )
    return package_id(package)


def check(args: argparse.Namespace, tools: Tools, out: Out) -> int:
    done = tools.attempt('git', 'symbolic-ref', '--quiet', '--short', 'HEAD')
    current = done.stdout.strip()
    if args.pushed_refs:
        return check_pushed(args, tools, out, current)
    under = phase_of(current)
    prefix = Phase(under).lane_prefix if under else None
    if prefix is None or not current.startswith(prefix):
        out(
            f'not a lane branch ({current or "detached HEAD"}); '
            'nothing to check'
        )
        return 0
    if lane_of(current) is None:
        raise Refusal(
            f'{current} is under {prefix} but is not a lane name: '
            'lower-case letters and digits'
        )
    package = worktree_package(tools, current)
    count = check_lane_range(tools, args.remote, current, 'HEAD', package)
    out(f'ok   {current} carries {package}: {count} commit(s), each its own')
    return 0


#: The branches only the owner's merge pushes (the owner, Q14, 2026-10-22).
PROTECTED_REFS = {
    f'refs/heads/{name}'
    for name in ('rust-core', 'main', 'rust-core-reference')
}


def protected_push(branch: str) -> None:
    """Refuses a push that updates or deletes a protected branch, unless
    ``SDM_APPROVED`` names an owner-answers section (guard_cmd.py's tokens),
    which is logged."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'orchestration'))
    import guard_cmd  # noqa: PLC0415

    token = os.environ.get(guard_cmd.TOKEN_VAR, '')
    if token and token in guard_cmd.valid_tokens():
        guard_cmd.log('approved-commands.log', f'pre-push {token} {branch}')
        return
    raise Refusal(
        f"the push updates {branch}, which only the owner's merge pushes; "
        f'run it as {guard_cmd.TOKEN_VAR}=<owner-answers id> git push ... '
        'when the owner approved it'
    )


def check_pushed(
    args: argparse.Namespace, tools: Tools, out: Out, current: str
) -> int:
    """The pre-push hook's refs: ``<local ref> <local sha> <remote ref>
    <remote sha>`` a line. Each lane ref pushed is checked, whatever is
    checked out."""
    pushed = 0
    for line in Path(args.pushed_refs).read_text().splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue
        _, sha, remote_ref, _ = fields
        branch = remote_ref.removeprefix('refs/heads/')
        if remote_ref in PROTECTED_REFS:
            protected_push(branch)
        under = (
            phase_of(branch) if remote_ref.startswith('refs/heads/') else None
        )
        if under is None or not branch.startswith(Phase(under).lane_prefix):
            continue
        if lane_of(branch) is None:
            raise Refusal(f'the push names {branch}, which is not a lane')
        if sha == ZERO_SHA:
            raise Refusal(
                f'the push deletes {branch}; a lane is never deleted (E43)'
            )
        if current != branch:
            raise Refusal(
                f'the push sends {sha[:12]} to {branch} from '
                f'{current or "a detached HEAD"}; push a lane only from its '
                'own worktree, on its own branch'
            )
        package = worktree_package(tools, branch)
        count = check_lane_range(tools, args.remote, branch, sha, package)
        out(f'ok   {branch} carries {package}: {count} commit(s), each its own')
        pushed += 1
    if not pushed:
        out('no lane branch in this push; nothing to check')
    return 0


# --- main ---------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(
        prog='lane.py', description=__doc__.split('\n\n', 1)[0]
    )
    top.add_argument('--remote', default='origin')
    top.add_argument('--repo', help='OWNER/NAME for gh; read from the remote')
    top.add_argument(
        '--phase-branch',
        help='default: SDM_PHASE_BRANCH, else the phase of the branch '
        'checked out, else the phase in progress (phases.py)',
    )
    sub = top.add_subparsers(dest='command', required=True)
    begin = sub.add_parser('start', help='prepare a lane for a new package')
    begin.add_argument('--lane', required=True)
    begin.add_argument('--package', required=True)
    begin.add_argument(
        '--worktree', required=True, help="the lane's worktree directory"
    )
    begin.add_argument('--wait', action='store_true')
    begin.add_argument(
        '--brief',
        help="the package's brief, which brief_check.py must pass",
    )
    begin.add_argument(
        '--wait-seconds', type=float, default=WAIT_SECONDS,
        help=argparse.SUPPRESS,
    )  # fmt: skip
    begin.add_argument('--dry-run', action='store_true')
    hold = sub.add_parser(
        'pause', help="mark a lane's worktree paused, freeing its slot"
    )
    hold.add_argument('--lane', required=True)
    hold.add_argument(
        '--resume', action='store_true', help='take the mark away'
    )
    show = sub.add_parser('status', help='show the lanes')
    show.add_argument('--lane')
    inspect = sub.add_parser(
        'check', help="prepush.sh's check of a lane's commits"
    )
    inspect.add_argument(
        '--pushed-refs', metavar='FILE',
        help="the pre-push hook's standard input: check the refs pushed",
    )  # fmt: skip
    return top


def main(
    argv: Sequence[str] | None = None,
    tools: Tools | None = None,
    out: Out = print,
) -> int:
    args = parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    tools = tools or Tools()
    command = {
        'start': start,
        'status': status,
        'check': check,
        'pause': pause,
    }[args.command]
    try:
        # Names first, before gh is asked anything: the phase branch (the
        # branch checked out read from git), then the lane and package.
        here = tools.attempt(
            'git', 'symbolic-ref', '--quiet', '--short', 'HEAD'
        )
        try:
            tools.phase = phase_named(
                phase_branch(
                    args.phase_branch,
                    branch=here.stdout.strip()
                    if here.returncode == 0
                    else None,
                )
            )
        except PhaseError as error:
            raise Refusal(
                f'{error}; lane.py works on a phase branch'
            ) from error
        if args.command == 'start':
            lane_branch(args.lane, tools.phase)
            package_id(args.package)
        elif args.command in ('status', 'pause') and args.lane:
            lane_branch(args.lane, tools.phase)
        if args.command not in ('check', 'pause'):
            tools.repo = args.repo or tools.repo or repo_of(tools, args.remote)
        return command(args, tools, out)
    except Refusal as why:
        out(f'REFUSED: {why}')
        return 2


if __name__ == '__main__':
    sys.exit(main())
