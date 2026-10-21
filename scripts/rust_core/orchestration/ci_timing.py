#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""CI's timing guard (T3.0, item 3): a job close to its time limit is
named while it still passes, not found when it is cut.

Phase 2's CI grew from 35 to 93 minutes a run and hit a job's limit eight
times; each time the limit was the first signal. CI runs this as its last
job, ``timing guard``, on every run that runs jobs:

    ci_timing.py --run ID [--workflow FILE] [--out DIR]
    ci_timing.py --durations FILE [--workflow FILE] [--out DIR]
    ci_timing.py --pull [--state DIR]

It judges each job of the newest run over its last three runs: a job
that used more than ``--threshold`` (80%) of its ``timeout-minutes``
(read from ``--workflow``, by default this repository's ci.yml) on two of
them is named, and the exit status is 1. A run the job was cut in by its
limit counts as over; a run it was cancelled in before its limit (a newer
push) or skipped is no run of the job. For each named job it writes a fix
item, ``ci-timing-<job>.md``, and ``fix-items.json`` into ``--out``, else
``$SDM_ORCH_STATE/fix-items``, else ``ci-timing``.

- ``--run ID`` reads the run's jobs and, newest first, the earlier
  completed runs on the Rust core's branches (``rust-core`` and every
  ``rust-core-phase-*``) from the GitHub API (``gh api``; ``GH_TOKEN`` in
  CI), and also records this run's job times as ``job-times.json``, in the
  durations form below; CI uploads ``--out`` as the run's ``ci-timing``
  artifact. An API that does not answer judges nothing (exit 0, said).
- ``--durations FILE`` judges given times instead: a JSON list of runs,
  newest first, each ``{"run": ID, "jobs": [{"name", "minutes"}, ...]}``,
  a job optionally with its ``conclusion``.
- ``--pull`` (the heartbeat runs it on the orchestration machine) reads
  the newest ``ci-timing`` artifact on the Rust core's branches and writes
  ``<state>/fix-items/ci-timing.md``: the jobs it names, or "none".

CI's step reports a named job as an error annotation and in the fix item
and does not fail the run for it: no candidate is refused for a time
limit (the owner's rule: no hard budgets). The cure is a faster job, not
a higher limit. No GitHub issue is opened. Exit status 2 is the tool's
own failure.
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

ROOT = Path(__file__).resolve().parents[3]
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


# --- runs and jobs -----------------------------------------------------------

#: A run in the durations form: ``{"run": ID, "branch": NAME, "jobs":
#: [{"name", "minutes", "conclusion", "url"}]}``.
Run = dict[str, Any]


def _minutes(job: dict[str, Any]) -> float | None:
    started, completed = job.get('started_at'), job.get('completed_at')
    if not started or not completed:
        return None
    seconds = (
        datetime.fromisoformat(completed.replace('Z', '+00:00'))
        - datetime.fromisoformat(started.replace('Z', '+00:00'))
    ).total_seconds()
    return seconds / 60


def as_run(run: dict[str, Any], jobs: list[dict[str, Any]]) -> Run:
    r"""A run and its jobs from the API, in the durations form; a job that
    has not finished is left out."""
    found = []
    for job in jobs:
        minutes = _minutes(job)
        if minutes is None:
            continue
        found.append(
            {
                'name': str(job.get('name') or ''),
                'minutes': round(minutes, 2),
                'conclusion': str(job.get('conclusion') or ''),
                'url': str(job.get('html_url') or ''),
            }
        )
    return {
        'run': int(run.get('id') or 0),
        'branch': str(run.get('head_branch') or ''),
        'jobs': found,
    }


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


def from_api(run_id: int, most: int, known: list[Limit]) -> list[Run]:
    r"""The run, then the newest completed runs on the Rust core's
    branches before it, until each of its jobs has :data:`LAST` runs or
    ``most`` runs were read."""
    newest = as_run(
        api(f'repos/{repository()}/actions/runs/{run_id}'), jobs_of(run_id)
    )
    wanted = {j['name'] for j in newest['jobs'] if limit_of(j['name'], known)}
    runs = [newest]
    seen: dict[str, int] = {}
    page = 1
    while len(runs) <= most and page <= 10:
        answer = api(
            f'repos/{repository()}/actions/workflows/ci.yml/runs'
            f'?status=completed&per_page=100&page={page}'
        )
        listed = answer.get('workflow_runs', [])
        if not listed:
            break
        for run in listed:
            if run.get('id') == run_id or not ours(
                str(run.get('head_branch') or '')
            ):
                continue
            if all(seen.get(name, 0) >= LAST - 1 for name in wanted):
                return runs
            earlier = as_run(run, jobs_of(int(run['id'])))
            runs.append(earlier)
            for job in earlier['jobs']:
                if job['conclusion'] not in ('skipped', ''):
                    seen[job['name']] = seen.get(job['name'], 0) + 1
            if len(runs) > most:
                break
        page += 1
    return runs


# --- the guard --------------------------------------------------------------------


@dataclass
class Sample:
    run: int
    branch: str
    minutes: float
    conclusion: str
    url: str


def sample(job: dict[str, Any], run: Run, limit: Limit) -> Sample | None:
    r"""The job's time in a run, or None when the job did not run there to
    its end or its limit (skipped, or cancelled before its limit)."""
    conclusion = str(job.get('conclusion') or 'success')
    minutes = float(job['minutes'])
    if conclusion in ('skipped', 'neutral'):
        return None
    if conclusion == 'cancelled' and minutes < CUT * limit.minutes:
        return None
    return Sample(
        int(run.get('run') or 0),
        str(run.get('branch') or ''),
        minutes,
        conclusion,
        str(job.get('url') or ''),
    )


@dataclass
class Verdict:
    name: str
    limit: int
    threshold: float
    samples: list[Sample] = field(default_factory=list)

    @property
    def over(self) -> list[Sample]:
        return [
            s for s in self.samples if s.minutes > self.threshold * self.limit
        ]

    @property
    def hot(self) -> bool:
        return len(self.over) >= OVER


def judge(
    runs: list[Run], known: list[Limit], threshold: float
) -> list[Verdict]:
    r"""Each job of the newest run, judged over its last :data:`LAST` runs,
    the newest first."""
    if not runs:
        return []
    verdicts = []
    for job in runs[0]['jobs']:
        name = str(job['name'])
        limit = limit_of(name, known)
        if name == GUARD or limit is None:
            continue
        here = sample(job, runs[0], limit)
        if here is None:
            continue
        verdict = Verdict(name, limit.minutes, threshold, [here])
        for earlier in runs[1:]:
            if len(verdict.samples) >= LAST:
                break
            for other in earlier['jobs']:
                if other['name'] == name:
                    found = sample(other, earlier, limit)
                    if found is not None:
                        verdict.samples.append(found)
                    break
        verdicts.append(verdict)
    return verdicts


def fix_item(verdict: Verdict) -> str:
    runs = '\n'.join(
        f'- run {s.run}{" on " + s.branch if s.branch else ""}: {s.minutes:.1f} '
        f'of {verdict.limit} minutes ({s.minutes / verdict.limit:.0%}, '
        f'{s.conclusion}){" " + s.url if s.url else ""}'
        for s in verdict.samples
    )
    return (
        f'# CI fix item: {verdict.name} is close to its time limit\n\n'
        f'The job used more than {verdict.threshold:.0%} of its timeout-minutes '
        f'({verdict.limit}) on {len(verdict.over)} of its last '
        f'{len(verdict.samples)} runs:\n\n{runs}\n\n'
        'Cut its time at the cause (a slow test file, a duplicated step, a '
        'job that should be split); raise the limit only with that reason '
        'in ci.yml. Found by scripts/rust_core/orchestration/ci_timing.py '
        '(T3.0).\n'
    )


def slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


def out_dir(given: Path | None) -> Path:
    if given is not None:
        return given
    state = os.environ.get('SDM_ORCH_STATE')
    return Path(state) / 'fix-items' if state else Path(ARTIFACT)


def guard(arguments: argparse.Namespace) -> int:
    known = limits(arguments.workflow)
    if arguments.durations is not None:
        runs = json.loads(arguments.durations.read_text(encoding='utf-8'))
        record = False
    else:
        try:
            runs = from_api(arguments.run, arguments.history, known)
        except Unanswered as error:
            print(
                f'timing guard: the GitHub API did not answer, nothing judged: {error}'
            )
            return 0
        record = True
    verdicts = judge(runs, known, arguments.threshold)
    hot = [v for v in verdicts if v.hot]
    out = out_dir(arguments.out)
    if record or hot:
        out.mkdir(parents=True, exist_ok=True)
    if record:
        (out / 'job-times.json').write_text(
            json.dumps(runs[:1], indent=1) + '\n', encoding='utf-8'
        )
    items = []
    for verdict in hot:
        path = out / f'ci-timing-{slug(verdict.name)}.md'
        path.write_text(fix_item(verdict), encoding='utf-8')
        items.append(
            {
                'job': verdict.name,
                'limit': verdict.limit,
                'runs': [vars(s) for s in verdict.samples],
            }
        )
        print(
            f'::error title=timing guard::{verdict.name} used more than '
            f'{verdict.threshold:.0%} of its {verdict.limit}-minute limit on '
            f'{len(verdict.over)} of its last {len(verdict.samples)} runs; '
            f'fix item {path}'
        )
    if record or hot:
        (out / 'fix-items.json').write_text(
            json.dumps(
                {'run': runs[0]['run'] if runs else None, 'items': items},
                indent=1,
            )
            + '\n',
            encoding='utf-8',
        )
    print(
        f'timing guard: {len(verdicts)} jobs judged over their last {LAST} runs; '
        + (
            f'{len(hot)} near their limit'
            if hot
            else f'none over {arguments.threshold:.0%} of its limit twice'
        )
    )
    return 1 if hot else 0


# --- pull --------------------------------------------------------------------------


def pull(arguments: argparse.Namespace) -> int:
    state = arguments.state or os.environ.get('SDM_ORCH_STATE')
    if not state:
        print(
            'ci_timing --pull: give --state or set SDM_ORCH_STATE',
            file=sys.stderr,
        )
        return 2
    target = Path(state) / 'fix-items' / 'ci-timing.md'
    try:
        answer = api(
            f'repos/{repository()}/actions/artifacts?name={ARTIFACT}&per_page=30'
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
                'ci_timing --pull: no ci-timing artifact on the Rust core branches yet'
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
            body = '\n'.join(
                t.read_text(encoding='utf-8')
                for t in sorted(Path(scratch).glob('ci-timing-*.md'))
            )
    except (Unanswered, OSError, ValueError, KeyError) as error:
        print(f'ci_timing --pull: {error}', file=sys.stderr)
        return 2
    target.parent.mkdir(parents=True, exist_ok=True)
    head = f'<!-- written by ci_timing.py --pull from run {run} -->\n'
    if items:
        target.write_text(head + body, encoding='utf-8')
        print(
            f'ci_timing --pull: {len(items)} fix item(s) from run {run} in {target}'
        )
    else:
        target.write_text(
            head + f'# CI timing: no job near its limit (run {run})\n',
            encoding='utf-8',
        )
        print(f'ci_timing --pull: none in run {run}; {target} says so')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run', type=int)
    source.add_argument('--durations', type=Path)
    source.add_argument('--pull', action='store_true')
    parser.add_argument('--workflow', type=Path, default=WORKFLOW)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--threshold', type=float, default=THRESHOLD)
    parser.add_argument('--history', type=int, default=HISTORY)
    parser.add_argument('--state')
    arguments = parser.parse_args(argv)
    try:
        return pull(arguments) if arguments.pull else guard(arguments)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'ci_timing: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
