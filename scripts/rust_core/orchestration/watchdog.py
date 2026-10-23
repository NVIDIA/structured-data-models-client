#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What needs the orchestrator now (T3.2 fix 4, the owner's speed change 4).

    watchdog.py [--phase BRANCH] [--repo PATH] [--forecast FILE]
        [--idle-minutes 45] [--slip-hours 2] [--now TIME] [--dispatch]

The heartbeat runs it every 15 minutes. It prints one line per problem and
nothing when all is well; it exits 0 either way (2 on a usage error), so the
heartbeat reads its output, not its status:

- ``IDLE <worktree> (<branch>): no commit for N min and no process``: a lane
  worktree (on ``<phase>-lane-<x>``) whose newest commit is older than
  ``--idle-minutes`` and in which no running process has its working
  directory. A lane at the phase head with no recorded package
  (``sdm.package``) has nothing to do and is not idle;
- ``RED CI <branch> run <id>: <jobs>``: the newest finished CI run of each
  branch under the phase (the phase, its lanes and topics) failed; the jobs
  named are the ones that failed. A ``gh`` failure prints one
  ``WATCHDOG gh failed: ...`` line instead;
- ``NIGHTLY <phase> run <id>: <jobs>``: the newest finished nightly run
  (scheduled or dispatched) of CI on the phase head failed; its failed jobs
  are named, and the run becomes one fix item, once, in
  ``$SDM_ORCH_STATE/phase-<N>/fix-items.md`` (``- [ ] nightly run <id> on
  <phase> <sha12>: <jobs>``). Nothing waits on the nightly (the owner's
  "2026-10-22 evening: CI off the landing path"). With ``--dispatch``, a
  phase head with no nightly run started in the last 24 hours gets one
  (``gh workflow run ci.yml --ref <phase> -f fresh=true``), printed as
  ``NIGHTLY <phase>: dispatched``: GitHub runs the workflow's schedule from
  the default branch's copy, which holds no nightly yet;
- ``SLIP <package>: expected by <time>, not landed; <h> h behind``: a
  package of the forecast (``docs/design/rust-core/phase-3/forecast.yaml``
  by default; ``phase_stats.py`` reads it the same way) past its forecast
  landing by more than ``--slip-hours``, with no ``Land <package>`` commit
  on the phase branch.

The phase is ``--phase``, else ``SDM_PHASE_BRANCH``; the repository
``--repo``, else ``SDM_REPO``, else this script's checkout. ``gh`` runs
under a timeout with ``GH_PROMPT_DISABLED=1``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase_stats import StatsError, read_forecast  # noqa: E402

ROOT = HERE.parents[2]
STATE = Path(
    os.environ.get('SDM_ORCH_STATE')
    or Path.home() / '.local/share/sdm-orchestration'
)
FORECAST = 'docs/design/rust-core/phase-3/forecast.yaml'
LANE = re.compile(r'[a-z0-9]{1,16}')
GH_TIMEOUT = 120
RUN_FIELDS = (
    'databaseId,headBranch,headSha,conclusion,status,createdAt,url,event'
)
#: The events of a nightly run: the schedule's, or its dispatch.
NIGHTLY_EVENTS = ('schedule', 'workflow_dispatch')
#: A run's conclusion that is red.
RED = ('failure', 'timed_out', 'startup_failure')

#: ``gh ARGS`` -> its standard output; raises RuntimeError on a failure.
Gh = Callable[[Sequence[str]], str]


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['git', '-C', str(repo), *args],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )


def run_gh(args: Sequence[str]) -> str:
    try:
        done = subprocess.run(
            ['gh', *args],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=GH_TIMEOUT,
            env={**os.environ, 'GH_PROMPT_DISABLED': '1'},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f'gh {" ".join(args)}: {error}') from None
    if done.returncode != 0:
        raise RuntimeError(
            f'gh {" ".join(args)}: {(done.stderr or done.stdout).strip()[:200]}'
        )
    return done.stdout


def process_cwds() -> list[str]:
    """The working directory of every process this user can read."""
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            found.append(os.readlink(entry / 'cwd'))
        except OSError:
            continue
    return found


# --- idle lanes -------------------------------------------------------------


def worktrees(repo: Path) -> list[tuple[str, str, str]]:
    """(path, branch, head) of each worktree on a branch."""
    done = git(repo, 'worktree', 'list', '--porcelain')
    found = []
    for block in done.stdout.split('\n\n'):
        fields = dict(
            line.partition(' ')[::2] for line in block.splitlines() if line
        )
        branch = fields.get('branch', '').removeprefix('refs/heads/')
        if 'worktree' in fields and branch:
            found.append((fields['worktree'], branch, fields.get('HEAD', '')))
    return found


def phase_head(repo: Path, phase: str) -> str | None:
    for ref in (f'refs/remotes/origin/{phase}', f'refs/heads/{phase}'):
        done = git(repo, 'rev-parse', '--verify', '--quiet', ref)
        if done.returncode == 0:
            return done.stdout.strip()
    return None


def inside(cwd: str, path: str) -> bool:
    return cwd == path or cwd.startswith(path.rstrip('/') + '/')


def idle_lanes(
    repo: Path,
    phase: str,
    now: dt.datetime,
    minutes: float,
    cwds: Iterable[str],
) -> list[str]:
    cwds = list(cwds)
    head = phase_head(repo, phase)
    prefix = f'{phase}-lane-'
    lines = []
    for path, branch, sha in worktrees(repo):
        if not branch.startswith(prefix) or not LANE.fullmatch(
            branch[len(prefix) :]
        ):
            continue
        package = git(
            Path(path), 'config', '--worktree', '--get', 'sdm.package'
        )
        if sha == head and (
            package.returncode != 0 or not package.stdout.strip()
        ):
            continue
        done = git(Path(path), 'log', '-1', '--format=%cI', 'HEAD')
        if done.returncode != 0:
            continue
        at = dt.datetime.fromisoformat(done.stdout.strip())
        age = (now - at).total_seconds() / 60
        if age <= minutes or any(inside(c, path) for c in cwds):
            continue
        lines.append(
            f'IDLE {path} ({branch}): no commit for {int(age)} min and no process'
        )
    return lines


# --- red CI -----------------------------------------------------------------


def red_ci(phase: str, gh: Gh) -> list[str]:
    try:
        runs = json.loads(
            gh(['run', 'list', '--limit', '50', '--json', RUN_FIELDS])
        )
    except (RuntimeError, ValueError) as error:
        return [f'WATCHDOG gh failed: {error}']
    newest: dict[str, dict] = {}
    for run in runs:
        branch = str(run.get('headBranch') or '')
        if not branch.startswith(phase) or run.get('status') != 'completed':
            continue
        if run.get('event') in NIGHTLY_EVENTS:
            continue  # the nightly's, which nightly() reads
        known = newest.get(branch)
        if known is None or str(run.get('createdAt')) > str(
            known.get('createdAt')
        ):
            newest[branch] = run
    lines = []
    for branch in sorted(newest):
        run = newest[branch]
        if run.get('conclusion') not in RED:
            continue
        number = str(run.get('databaseId'))
        try:
            jobs = json.loads(gh(['run', 'view', number, '--json', 'jobs']))
        except (RuntimeError, ValueError) as error:
            lines.append(f'WATCHDOG gh failed: {error}')
            continue
        failed = [
            str(job.get('name'))
            for job in jobs.get('jobs', [])
            if job.get('conclusion') in RED
        ]
        lines.append(
            f'RED CI {branch} run {number}: '
            + (', '.join(failed) or f'no failed job named ({run.get("url")})')
        )
    return lines


# --- the nightly ------------------------------------------------------------


def fix_items_path(phase: str, state: Path = STATE) -> Path:
    match = re.fullmatch(r'rust-core-phase-([0-9]+)', phase)
    folder = f'phase-{match.group(1)}' if match else phase
    return state / folder / 'fix-items.md'


def nightly(
    phase: str,
    gh: Gh,
    items: Path,
    now: dt.datetime,
    dispatch: bool = False,
) -> list[str]:
    r"""The newest finished nightly run on the phase head: red, a line and
    one fix item per run; with ``dispatch``, a run started when none began
    in the last 24 hours."""
    try:
        runs = json.loads(
            gh(['run', 'list', '--workflow', 'ci.yml', '--branch', phase,
                '--limit', '20', '--json', RUN_FIELDS])
        )  # fmt: skip
    except (RuntimeError, ValueError) as error:
        return [f'WATCHDOG gh failed: {error}']
    ours = [r for r in runs if r.get('event') in NIGHTLY_EVENTS]
    lines = []
    if dispatch and not any(
        now - _time(str(r.get('createdAt'))) < dt.timedelta(hours=24)
        for r in ours
    ):
        try:
            gh(
                [
                    'workflow',
                    'run',
                    'ci.yml',
                    '--ref',
                    phase,
                    '-f',
                    'fresh=true',
                ]
            )
            lines.append(f'NIGHTLY {phase}: dispatched (none in 24 h)')
        except RuntimeError as error:
            lines.append(f'WATCHDOG gh failed: {error}')
    done = [r for r in ours if r.get('status') == 'completed']
    # A nightly that did not run or report is a fix item too (the owner,
    # "2026-10-23 early: nightly forced-route run").
    if not any(
        now - _time(str(r.get('createdAt'))) < dt.timedelta(hours=MISSING_HOURS)
        for r in done
    ):
        lines.append(f'NIGHTLY {phase}: none finished in {MISSING_HOURS} h')
        _add_item(
            items,
            f'- [ ] nightly on {phase} missing {now:%Y-%m-%d}: none finished '
            f'in {MISSING_HOURS} h',
            f'nightly on {phase} missing {now:%Y-%m-%d}',
            lines,
        )
    if not done:
        return lines
    run = max(done, key=lambda r: str(r.get('createdAt')))
    if run.get('conclusion') not in RED:
        return lines
    number = str(run.get('databaseId'))
    try:
        jobs = json.loads(gh(['run', 'view', number, '--json', 'jobs']))
    except (RuntimeError, ValueError) as error:
        return [*lines, f'WATCHDOG gh failed: {error}']
    failed = (
        ', '.join(
            str(job.get('name'))
            for job in jobs.get('jobs', [])
            if job.get('conclusion') in RED
        )
        or f'no failed job named ({run.get("url")})'
    )
    tests = failing_tests(gh, jobs)
    named = f'{failed}; failing tests: {", ".join(tests)}' if tests else failed
    lines.append(f'NIGHTLY {phase} run {number}: {named}')
    _add_item(
        items,
        f'- [ ] nightly run {number} on {phase} '
        f'{str(run.get("headSha"))[:12]}: {named}',
        f'nightly run {number} ',
        lines,
    )
    return lines


#: Hours with no finished nightly before the watchdog calls it missing.
MISSING_HOURS = 26
FAILED_TEST = re.compile(r'\bFAILED (\S+::\S+)')


def failing_tests(gh: Gh, jobs: dict[str, object], most: int = 20) -> list[str]:
    """The failing pytest node IDs the red jobs' logs name."""
    found: list[str] = []
    for job in jobs.get('jobs', []) or []:  # type: ignore[union-attr]
        if not isinstance(job, dict) or job.get('conclusion') not in RED:
            continue
        try:
            log = gh(
                [
                    'run',
                    'view',
                    '--job',
                    str(job.get('databaseId')),
                    '--log-failed',
                ]
            )
        except RuntimeError:
            continue
        for match in FAILED_TEST.finditer(log):
            if match.group(1) not in found:
                found.append(match.group(1))
    return found[:most]


def _add_item(items: Path, item: str, key: str, lines: list[str]) -> None:
    """``item`` appended to the fix items once, ``key`` naming it."""
    try:
        known = items.read_text(encoding='utf-8') if items.is_file() else ''
        if key not in known:
            items.parent.mkdir(parents=True, exist_ok=True)
            with open(items, 'a', encoding='utf-8') as handle:
                handle.write(item + '\n')
    except OSError as error:
        lines.append(f'WATCHDOG fix items unwritable: {error}')


def _time(text: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(text.replace('Z', '+00:00'))
    except ValueError:
        return dt.datetime.min.replace(tzinfo=dt.timezone.utc)


# --- slip -------------------------------------------------------------------


def landed(repo: Path, phase: str) -> set[str]:
    head = phase_head(repo, phase)
    if head is None:
        return set()
    done = git(repo, 'log', '--first-parent', '--format=%s', head)
    found = set()
    for subject in done.stdout.splitlines():
        match = re.match(r'Land ([A-Z][A-Za-z0-9.]*)[: ]', subject)
        if match:
            found.add(match.group(1))
    return found


def slips(
    forecast: Path,
    done: set[str],
    now: dt.datetime,
    threshold: float,
) -> list[str]:
    _, planned = read_forecast(forecast, dt.timezone.utc)
    lines = []
    for package, _, due in planned:
        behind = (now - due).total_seconds() / 3600
        if package in done or behind <= threshold:
            continue
        lines.append(
            f'SLIP {package}: expected by {due:%Y-%m-%d %H:%M}Z, not landed; '
            f'{behind:.1f} h behind'
        )
    return lines


# --- main -------------------------------------------------------------------


def watch(
    repo: Path,
    phase: str,
    forecast: Path | None,
    now: dt.datetime,
    idle_minutes: float,
    slip_hours: float,
    gh: Gh,
    cwds: Callable[[], Iterable[str]],
    items: Path | None = None,
    dispatch: bool = False,
) -> list[str]:
    lines = idle_lanes(repo, phase, now, idle_minutes, cwds())
    lines += red_ci(phase, gh)
    lines += nightly(phase, gh, items or fix_items_path(phase), now, dispatch)
    if forecast is not None:
        try:
            lines += slips(forecast, landed(repo, phase), now, slip_hours)
        except (OSError, StatsError) as error:
            lines.append(f'WATCHDOG forecast unreadable: {error}')
    # One gh outage reads as one line, not one per check.
    return list(dict.fromkeys(lines))


def main(
    argv: Sequence[str] | None = None,
    gh: Gh = run_gh,
    cwds: Callable[[], Iterable[str]] = process_cwds,
) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--phase', default=os.environ.get('SDM_PHASE_BRANCH'))
    parser.add_argument('--repo', default=os.environ.get('SDM_REPO'))
    parser.add_argument('--forecast')
    parser.add_argument('--idle-minutes', type=float, default=45)
    parser.add_argument('--slip-hours', type=float, default=2)
    parser.add_argument('--now', help='an ISO time (default: now)')
    parser.add_argument(
        '--dispatch',
        action='store_true',
        help='start the nightly on the phase head when none began in 24 h',
    )
    parser.add_argument('--fix-items', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.phase:
        print('watchdog.py: give --phase or set SDM_PHASE_BRANCH')
        return 2
    repo = Path(args.repo) if args.repo else ROOT
    forecast = Path(args.forecast) if args.forecast else repo / FORECAST
    now = (
        dt.datetime.fromisoformat(args.now.replace('Z', '+00:00'))
        if args.now
        else dt.datetime.now(dt.timezone.utc)
    )
    # The default forecast may be absent (a phase without one); a named one
    # must be read.
    for line in watch(
        repo,
        args.phase,
        forecast if args.forecast or forecast.is_file() else None,
        now,
        args.idle_minutes,
        args.slip_hours,
        gh,
        cwds,
        Path(args.fix_items) if args.fix_items else None,
        args.dispatch,
    ):
        print(line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
