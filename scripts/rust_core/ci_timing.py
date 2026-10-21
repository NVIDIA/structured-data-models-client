#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""CI's timing guard (T3.0, item 3): a job close to its time limit is
named while it still passes, not found when it is cut.

Phase 2's CI grew from 35 to 93 minutes a run and hit a job's limit eight
times; each time the limit was the first signal. ``guard`` runs as CI's
last job, ``timing guard``, on every run that runs jobs:

- it records every job's duration in this run (``job-times.json`` in
  ``--out``, uploaded as the run's ``ci-timing`` artifact);
- it reads each job's earlier runs on the Rust core's branches
  (``rust-core`` and every ``rust-core-phase-*``) from the GitHub API,
  newest first, and fails, naming the job, when the job used more than
  ``--threshold`` (80%) of its ``timeout-minutes`` (read from the
  workflow) on two of its last three runs, this one included; a run the
  job was cut in by its limit counts as over, one cancelled before it
  (a newer push) or skipped is no run of the job;
- for each such job it writes a fix item (``fix-items.json`` and
  ``fix-item-<job>.md`` in ``--out``), which ``pull`` brings to the
  orchestrator. No GitHub issue is opened.

The job is ``continue-on-error``: its failure marks the job red and names
the slow job, but does not fail the run, so a candidate is never refused
for a time limit (the owner: no hard budgets). The cure is a faster job,
not a higher limit.

``pull`` runs on the orchestration machine (the heartbeat runs it): it
reads the newest ``ci-timing`` artifact on the Rust core's branches and
writes ``<state>/fix-items/ci-timing.md`` (``state`` is ``--state``, else
``$SDM_ORCH_STATE``): the jobs it names, each with its runs, or "none".

    ci_timing.py guard --run ID [--workflow FILE] [--out DIR]
        [--threshold 0.8] [--history N]
    ci_timing.py pull [--state DIR] [--limit N]

Both call ``gh api`` (``GH_TOKEN`` in CI) for the repository ``gh``
resolves (``GITHUB_REPOSITORY`` in CI); a call that fails is retried, and
an API that does not answer passes the guard with a warning: the guard
never fails for want of data.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / '.github/workflows/ci.yml'
#: The guard's own job, which it never judges.
GUARD = 'timing guard'
ARTIFACT = 'ci-timing'
THRESHOLD = 0.8
#: The runs looked at: this one and the job's last ``LAST - 1`` before it.
LAST = 3
OVER = 2
#: The newest completed runs read for history, at most.
HISTORY = 40
#: A job cut at this share of its limit or more was cut by the limit.
CUT = 0.98
EXPRESSION = re.compile(r'\$\{\{.*?\}\}')


class Unanswered(Exception):
    r"""The GitHub API did not answer."""


def api(path: str, tries: int = 4) -> Any:
    r"""``gh api PATH`` as JSON, retried with a growing pause."""
    error = ''
    for attempt in range(tries):
        done = subprocess.run(
            ['gh', 'api', path],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=120,
            check=False,
        )
        if done.returncode == 0:
            try:
                return json.loads(done.stdout)
            except ValueError as bad:
                error = f'not JSON: {bad}'
        else:
            error = (done.stderr or done.stdout).strip()
        time.sleep(min(30, 2 * 2**attempt))
    raise Unanswered(f'gh api {path}: {error}')


def repository() -> str:
    return os.environ.get('GITHUB_REPOSITORY') or '{owner}/{repo}'


# --- the workflow's limits -------------------------------------------------------


@dataclass(frozen=True)
class Limit:
    job_id: str
    pattern: re.Pattern[str]
    minutes: int


def limits(workflow: Path = WORKFLOW) -> list[Limit]:
    r"""Each job's display name as a pattern (a matrix value or an
    expression in its name matches anything; a job with no name is its id,
    or its id with the matrix values in parentheses) and its limit."""
    import yaml  # CI installs it for this step; the module loads without.

    jobs = yaml.safe_load(workflow.read_text(encoding='utf-8'))['jobs']
    found = []
    for job_id, job in jobs.items():
        minutes = job.get('timeout-minutes')
        if not isinstance(minutes, int):
            continue
        name = job.get('name')
        if name:
            parts = EXPRESSION.split(str(name))
            pattern = '.+'.join(re.escape(part) for part in parts)
        else:
            pattern = re.escape(job_id) + r'(?: \(.+\))?'
        found.append(Limit(job_id, re.compile(pattern), minutes))
    return found


def limit_of(name: str, known: list[Limit]) -> Limit | None:
    for limit in known:
        if limit.pattern.fullmatch(name):
            return limit
    return None


# --- runs and jobs -----------------------------------------------------------------


def _minutes(job: dict[str, Any]) -> float | None:
    started, completed = job.get('started_at'), job.get('completed_at')
    if not started or not completed:
        return None
    seconds = (
        datetime.fromisoformat(completed.replace('Z', '+00:00'))
        - datetime.fromisoformat(started.replace('Z', '+00:00'))
    ).total_seconds()
    return seconds / 60


@dataclass
class Sample:
    run: int
    branch: str
    minutes: float
    conclusion: str
    url: str


def sample(
    job: dict[str, Any], run: dict[str, Any], limit: Limit
) -> Sample | None:
    r"""The job's time in a run, or None when the job did not run there to
    its end or its limit (skipped, or cancelled before its limit)."""
    minutes = _minutes(job)
    conclusion = job.get('conclusion') or ''
    if minutes is None or conclusion in ('skipped', 'neutral', ''):
        return None
    if conclusion == 'cancelled' and minutes < CUT * limit.minutes:
        return None
    return Sample(
        int(run.get('id') or job.get('run_id') or 0),
        str(run.get('head_branch') or ''),
        minutes,
        conclusion,
        str(job.get('html_url') or ''),
    )


def jobs_of(run_id: int) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    page = 1
    while True:
        answer = api(
            f'repos/{repository()}/actions/runs/{run_id}/jobs'
            f'?filter=latest&per_page=100&page={page}'
        )
        found.extend(answer.get('jobs', []))
        if len(found) >= answer.get('total_count', 0) or not answer.get('jobs'):
            return found
        page += 1


def ours(branch: str) -> bool:
    r"""A branch whose runs use this workflow's jobs and limits."""
    return branch == 'rust-core' or branch.startswith('rust-core-phase-')


def earlier_runs(run_id: int, most: int) -> list[dict[str, Any]]:
    r"""The newest completed runs of the workflow on the Rust core's
    branches, this one left out, newest first."""
    found: list[dict[str, Any]] = []
    page = 1
    while len(found) < most and page <= 10:
        answer = api(
            f'repos/{repository()}/actions/workflows/ci.yml/runs'
            f'?status=completed&per_page=100&page={page}'
        )
        runs = answer.get('workflow_runs', [])
        if not runs:
            break
        found.extend(
            run
            for run in runs
            if run.get('id') != run_id
            and ours(str(run.get('head_branch') or ''))
        )
        page += 1
    return found[:most]


# --- the guard ---------------------------------------------------------------------


@dataclass
class Verdict:
    name: str
    limit: int
    samples: list[Sample] = field(default_factory=list)

    @property
    def over(self) -> list[Sample]:
        return [s for s in self.samples if s.minutes > THRESHOLD * self.limit]

    @property
    def hot(self) -> bool:
        return len(self.over) >= OVER


def judge(
    current: list[dict[str, Any]],
    run: dict[str, Any],
    history: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    known: list[Limit],
) -> tuple[dict[str, float], list[Verdict]]:
    r"""This run's job times, and each of its jobs judged over its last
    :data:`LAST` runs, this one first. ``history`` is the earlier runs,
    newest first, each with its jobs."""
    times: dict[str, float] = {}
    verdicts: list[Verdict] = []
    for job in current:
        name = str(job.get('name') or '')
        limit = limit_of(name, known)
        if name == GUARD or limit is None:
            continue
        here = sample(job, run, limit)
        if here is None:
            continue
        times[name] = round(here.minutes, 1)
        verdict = Verdict(name, limit.minutes, [here])
        for earlier, jobs in history:
            if len(verdict.samples) >= LAST:
                break
            for other in jobs:
                if other.get('name') == name:
                    found = sample(other, earlier, limit)
                    if found is not None:
                        verdict.samples.append(found)
                    break
        verdicts.append(verdict)
    return times, verdicts


def fix_item(verdict: Verdict) -> str:
    runs = '\n'.join(
        f'- run {s.run} on {s.branch}: {s.minutes:.1f} of {verdict.limit} minutes '
        f'({s.minutes / verdict.limit:.0%}, {s.conclusion}) {s.url}'
        for s in verdict.samples
    )
    return (
        f'# CI fix item: {verdict.name} is close to its time limit\n\n'
        f'The job used more than {THRESHOLD:.0%} of its timeout-minutes '
        f'({verdict.limit}) on {len(verdict.over)} of its last '
        f'{len(verdict.samples)} runs:\n\n{runs}\n\n'
        'Cut its time at the cause (a slow test file, a duplicated step, a '
        'job that should be split); raise the limit only with that reason '
        'in ci.yml. Found by scripts/rust_core/ci_timing.py (T3.0).\n'
    )


def slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


def guard(arguments: argparse.Namespace) -> int:
    global THRESHOLD
    THRESHOLD = arguments.threshold
    out: Path = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    known = limits(arguments.workflow)
    try:
        run = api(f'repos/{repository()}/actions/runs/{arguments.run}')
        current = jobs_of(arguments.run)
        wanted = {
            str(job.get('name'))
            for job in current
            if limit_of(str(job.get('name')), known) is not None
        }
        history: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        seen: dict[str, int] = {}
        for earlier in earlier_runs(arguments.run, arguments.history):
            if wanted and all(seen.get(n, 0) >= LAST - 1 for n in wanted):
                break
            jobs = jobs_of(int(earlier['id']))
            history.append((earlier, jobs))
            for job in jobs:
                if job.get('conclusion') not in (None, 'skipped'):
                    seen[str(job.get('name'))] = (
                        seen.get(str(job.get('name')), 0) + 1
                    )
    except Unanswered as error:
        print(
            f'timing guard: the GitHub API did not answer, nothing judged: {error}'
        )
        return 0
    times, verdicts = judge(current, run, history, known)
    (out / 'job-times.json').write_text(
        json.dumps(
            {'run': arguments.run, 'minutes': times}, indent=2, sort_keys=True
        )
        + '\n',
        encoding='utf-8',
    )
    hot = [v for v in verdicts if v.hot]
    items = []
    for verdict in verdicts:
        share = (
            max((s.minutes for s in verdict.samples), default=0) / verdict.limit
        )
        print(
            f'{verdict.name}: {verdict.samples[0].minutes:.1f} of {verdict.limit} min; '
            f'over {THRESHOLD:.0%} on {len(verdict.over)} of its last '
            f'{len(verdict.samples)} runs (most {share:.0%})'
        )
    for verdict in hot:
        text = fix_item(verdict)
        (out / f'fix-item-{slug(verdict.name)}.md').write_text(
            text, encoding='utf-8'
        )
        items.append(
            {
                'job': verdict.name,
                'limit': verdict.limit,
                'runs': [vars(s) for s in verdict.samples],
            }
        )
        print(f'::error title=timing guard::{verdict.name} used more than '
              f'{THRESHOLD:.0%} of its {verdict.limit}-minute limit on '
              f'{len(verdict.over)} of its last {len(verdict.samples)} runs')  # fmt: skip
    (out / 'fix-items.json').write_text(
        json.dumps({'run': arguments.run, 'items': items}, indent=2) + '\n',
        encoding='utf-8',
    )
    return 1 if hot else 0


# --- pull --------------------------------------------------------------------------


def pull(arguments: argparse.Namespace) -> int:
    state = arguments.state or os.environ.get('SDM_ORCH_STATE')
    if not state:
        print(
            'ci_timing pull: give --state or set SDM_ORCH_STATE',
            file=sys.stderr,
        )
        return 2
    target = Path(state) / 'fix-items' / 'ci-timing.md'
    try:
        answer = api(
            f'repos/{repository()}/actions/artifacts?name={ARTIFACT}'
            f'&per_page={arguments.limit}'
        )
        artifact = next(
            (
                a
                for a in answer.get('artifacts', [])
                if not a.get('expired')
                and ours(
                    str((a.get('workflow_run') or {}).get('head_branch') or '')
                )
            ),
            None,
        )
        if artifact is None:
            print(
                'ci_timing pull: no ci-timing artifact on the Rust core branches yet'
            )
            return 0
        run = (artifact.get('workflow_run') or {}).get('id')
        with tempfile.TemporaryDirectory() as scratch:
            done = subprocess.run(
                [
                    'gh',
                    'run',
                    'download',
                    str(run),
                    '--name',
                    ARTIFACT,
                    '--dir',
                    scratch,
                ],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=300,
                check=False,
            )
            if done.returncode != 0:
                raise Unanswered(done.stderr.strip())
            items = json.loads((Path(scratch) / 'fix-items.json').read_text())[
                'items'
            ]
            texts = sorted(Path(scratch).glob('fix-item-*.md'))
            body = '\n'.join(t.read_text(encoding='utf-8') for t in texts)
    except (Unanswered, OSError, ValueError, KeyError) as error:
        print(f'ci_timing pull: {error}', file=sys.stderr)
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    head = f'<!-- written by scripts/rust_core/ci_timing.py pull from run {run} -->\n'
    if items:
        target.write_text(head + body, encoding='utf-8')
        print(
            f'ci_timing pull: {len(items)} fix item(s) from run {run} in {target}'
        )
    else:
        target.write_text(
            head + f'# CI timing: no job near its limit (run {run})\n',
            encoding='utf-8',
        )
        print(f'ci_timing pull: none in run {run}; {target} says so')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('guard')
    run.add_argument('--run', type=int, required=True)
    run.add_argument('--workflow', type=Path, default=WORKFLOW)
    run.add_argument('--out', type=Path, default=Path(ARTIFACT))
    run.add_argument('--threshold', type=float, default=THRESHOLD)
    run.add_argument('--history', type=int, default=HISTORY)
    get = commands.add_parser('pull')
    get.add_argument('--state')
    get.add_argument('--limit', type=int, default=30)
    arguments = parser.parse_args(argv)
    return guard(arguments) if arguments.command == 'guard' else pull(arguments)


if __name__ == '__main__':
    sys.exit(main())
