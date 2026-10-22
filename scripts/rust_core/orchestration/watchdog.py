#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What needs the orchestrator now (T3.2 fix 4, the owner's speed change 4).

    watchdog.py [--phase BRANCH] [--repo PATH] [--forecast FILE]
        [--idle-minutes 45] [--slip-hours 2] [--now TIME]

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
FORECAST = 'docs/design/rust-core/phase-3/forecast.yaml'
LANE = re.compile(r'[a-z0-9]{1,16}')
GH_TIMEOUT = 120
RUN_FIELDS = 'databaseId,headBranch,headSha,conclusion,status,createdAt,url'
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
) -> list[str]:
    lines = idle_lanes(repo, phase, now, idle_minutes, cwds())
    lines += red_ci(phase, gh)
    if forecast is not None:
        try:
            lines += slips(forecast, landed(repo, phase), now, slip_hours)
        except (OSError, StatsError) as error:
            lines.append(f'WATCHDOG forecast unreadable: {error}')
    return lines


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
    ):
        print(line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
