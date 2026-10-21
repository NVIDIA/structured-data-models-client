#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retrospective numbers for a phase, read from git, GitHub and reviews.

    phase_stats.py --phase-branch REF [--base REF] [--repo PATH]
        [--package-re RE] [--utc-offset HOURS]
        [--reviews DIR_OR_FILE]... [--reviewer NAME=REGEX]...
        [--no-ci] [--no-jobs] [--gh-repo OWNER/NAME] [--workflow NAME]
        [--since YYYY-MM-DD] [--limit N] [--max-job-fetch N]
        [--json OUT]
    phase_stats.py --phase-branch REF --forecast FILE [--now TIME] [--repo PATH]
        [--package-re RE] [--utc-offset HOURS]

Read-only: it runs ``git log``, ``gh run list`` and ``gh api`` GETs, and
reads the review files. It prints:

- Landings per day: the phase branch's landing commits (subject
  ``Land <package>...``), by committer date.
- Packages: per package named in a landing subject (``--package-re``,
  default ``T<n>.<n>[a-z]`` or ``CI<n>``, as in ``Land T1.12: ...``) the
  number of landings, commits and test commits (subject ``<word> tests``,
  as in ``T1.12 tests: ...``), and its span from the first commit to the
  last landing.
- Stage times per landing, from the commits since the previous landing:
  tests written ahead of the implementation, implementation (first
  non-test commit to the last commit that is not a test or a fix, as in
  ``T1.12 fix 3: ...``, a rebase or a renumbering), review
  and fixes (from there to the last commit), and the tail to the landing.
- CI per day: runs on the phase branch and on branches named
  ``<phase>-...`` (lanes, topic branches), green, red, cancelled, other,
  reruns and the median minutes of a finished run.
- CI failures split into infrastructure and real: for each failed or rerun
  run attempt, the failed jobs are read, and an attempt is infra-only when
  every failed job failed in setup (no failed step, "Set up job",
  "Initialize containers", a download-artifact step or a lane-image pull)
  or is a ``wheel-smoke`` job; otherwise it failed on a real step.
- Review verdicts: every file in ``--reviews`` whose first line is a
  verdict (``APPROVE``, ``CHANGES REQUIRED`` and the like) and whose second
  line is a commit SHA. Verdicts are tallied per reviewer (``--reviewer
  NAME=REGEX`` matched against the file name, first match wins; default
  ``codex`` and ``subagent``), and, for each SHA both of the first two
  reviewers reported on, the pair is counted as both approve, one blocks,
  or both block (each reviewer's latest report on that SHA).

With ``--forecast FILE`` it prints only the slip report (T3.0, item 1;
the heartbeat runs it): the design's forecast, a JSON or YAML mapping
``{"start": TIME, "packages": [{"id": ID, "hours": H}, ...]}``, the
packages in landing order with the hours each holds the landing queue,
against the landings by ``--now`` (default: now). Package i is forecast to
land at ``start`` plus the hours of packages 1 to i. The slip is how late
the newest forecast package that landed was, or, when a package due by
``--now`` has not landed, how long past due it is, whichever is more. Its
main cause is the package with the most hours over its own forecast (from
the landing before it to its own), or the landings the forecast does not
hold (judging and other unplanned packages), when they took more. It
never blocks: the exit status is 0 whatever the slip.

Commits between two landings belong to the second (the phase branch only
fast-forwards). ``--base`` limits the commits to ``BASE..PHASE``; without
it the commits before the first landing are left out. ``--reviews`` takes
directories (their ``*.md`` files), files or quoted globs.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import os
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_PACKAGE_RE = r'T\d+\.\d+[a-z]?|CI\d+'
INFRA_STEP_RE = (
    r'Set up job|Initialize containers|download-artifact|Pull the lane'
)
INFRA_JOB_RE = r'^wheel-smoke'
FIX_RE = r'\bfix ?\d|\bfix:|\brebase|\brenumber'
BOOKKEEPING = re.compile(r'tracker:|docs:', re.I)
SHA_RE = re.compile(r'^[0-9a-f]{7,40}$')


class StatsError(Exception):
    pass


def run(cmd, cwd=None):
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=300,
        )
    except subprocess.TimeoutExpired as e:
        raise StatsError(f'{" ".join(cmd)} timed out') from e
    if proc.returncode != 0:
        raise StatsError(f'{" ".join(cmd)} failed: {proc.stderr.strip()}')
    return proc.stdout


def parse_time(s, tz):
    t = dt.datetime.fromisoformat(s.replace('Z', '+00:00'))
    return t.astimezone(tz)


def hours(a, b):
    return round((b - a).total_seconds() / 3600, 2)


# --- commits ---------------------------------------------------------------


def commits(repo, phase, base, tz):
    rng = f'{base}..{phase}' if base else phase
    out = run(
        [
            'git',
            'log',
            '--reverse',
            '--first-parent',
            '--format=%H%x1f%aI%x1f%cI%x1f%s',
            rng,
        ],
        cwd=repo,
    )
    rows = []
    for line in out.splitlines():
        sha, a, c, subj = line.split('\x1f', 3)
        rows.append(
            {
                'sha': sha,
                'authored': parse_time(a, tz),
                'committed': parse_time(c, tz),
                'subject': subj,
            }
        )
    return rows


def classify_commits(rows, package_re):
    land = re.compile(rf'^Land ({package_re})\b')
    tests = re.compile(r'^\S+ tests\b')
    for r in rows:
        m = land.match(r['subject'])
        r['land'] = m.group(1) if m else None
        r['tests'] = not r['land'] and bool(tests.match(r['subject']))
    return rows


def landing_groups(rows, has_base):
    """(landing, commits since the previous landing) in order.

    The phase branch only fast-forwards, so every commit between two
    landings belongs to the second. Without a base the commits before the
    first landing are the whole earlier history, so that group is dropped.
    """
    out, group, first = [], [], True
    for r in rows:
        if r['land']:
            if group and (has_base or not first):
                out.append((r, group))
            group, first = [], False
        else:
            group.append(r)
    return out


def landings_per_day(rows):
    days = collections.Counter(
        r['committed'].date().isoformat() for r in rows if r['land']
    )
    return dict(sorted(days.items()))


def packages(rows, groups):
    pk = collections.defaultdict(
        lambda: {'first': None, 'lands': [], 'commits': 0, 'tests': 0}
    )
    for r in rows:
        if r['land']:
            pk[r['land']]['lands'].append(r['committed'])
    for land, group in groups:
        d = pk[land['land']]
        first = min(r['authored'] for r in group)
        if d['first'] is None or first < d['first']:
            d['first'] = first
        d['commits'] += len(group)
        d['tests'] += sum(r['tests'] for r in group)
    out = []
    for name, d in pk.items():
        first = d['first'] or min(d['lands'])
        out.append(
            {
                'package': name,
                'first_landing': min(d['lands']).isoformat(),
                'landings': len(d['lands']),
                'commits': d['commits'],
                'test_commits': d['tests'],
                'span_h': hours(first, max(d['lands'])),
            }
        )
    out.sort(key=lambda x: x['first_landing'])
    return out


def stages(groups):
    fix = re.compile(FIX_RE, re.I)
    return [stage_row(land, group, fix) for land, group in groups]


def stage_row(land, group, fix):
    tests = [r for r in group if r['tests']]
    fixes = [r for r in group if not r['tests'] and fix.search(r['subject'])]
    impl = [
        r
        for r in group
        if not r['tests'] and not BOOKKEEPING.search(r['subject'])
    ]
    first = min(r['authored'] for r in group)
    t_first = min((r['authored'] for r in tests), default=None)
    i_first = min((r['authored'] for r in impl), default=first)
    nonfix = [r for r in impl if r not in fixes]
    cand = max((r['authored'] for r in nonfix), default=i_first)
    last = max(r['committed'] for r in group)
    lead = hours(t_first, i_first) if t_first and t_first < i_first else 0.0
    return {
        'landing': land['subject'],
        'landed': land['committed'].isoformat(),
        'tests_lead_h': lead,
        'impl_h': max(hours(i_first, cand), 0.0),
        'review_fix_h': max(hours(cand, last), 0.0),
        'tail_h': max(hours(last, land['committed']), 0.0),
        'total_h': hours(first, land['committed']),
        'fix_commits': len(fixes),
        'test_commits': len(tests),
    }


# --- CI --------------------------------------------------------------------


def gh_slug(repo, remote):
    url = run(['git', 'remote', 'get-url', remote], cwd=repo).strip()
    m = re.search(r'github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$', url)
    if not m:
        raise StatsError(f'remote {remote} is not on GitHub; pass --gh-repo')
    return m.group(1)


def ci_runs(slug, branch, since, limit, workflow):
    fields = (
        'databaseId,workflowName,headBranch,headSha,status,conclusion,'
        'createdAt,startedAt,updatedAt,attempt,displayTitle,event'
    )
    cmd = [
        'gh',
        'run',
        'list',
        '-R',
        slug,
        '--limit',
        str(limit),
        '--json',
        fields,
        '--created',
        f'>={since}',
    ]
    if workflow:
        cmd += ['--workflow', workflow]
    runs = json.loads(run(cmd) or '[]')
    return [
        x
        for x in runs
        if x['headBranch'] == branch or x['headBranch'].startswith(branch + '-')
    ]


def ci_daily(runs, tz):
    by = collections.defaultdict(list)
    for x in runs:
        by[parse_time(x['createdAt'], tz).date().isoformat()].append(x)
    out = {}
    for day in sorted(by):
        xs = by[day]
        c = collections.Counter(x['conclusion'] or x['status'] for x in xs)
        durs = [
            (
                parse_time(x['updatedAt'], tz) - parse_time(x['startedAt'], tz)
            ).total_seconds()
            / 60
            for x in xs
            if x['status'] == 'completed'
            and x['conclusion'] in ('success', 'failure')
        ]
        out[day] = {
            'runs': len(xs),
            'green': c['success'],
            'red': c['failure'],
            'cancelled': c['cancelled'],
            'other': len(xs) - c['success'] - c['failure'] - c['cancelled'],
            'reruns': sum(1 for x in xs if x['attempt'] > 1),
            'median_min': round(statistics.median(durs), 1) if durs else None,
        }
    return out


def fetch_jobs(slug, runs, cap):
    """{(run id, attempt): [job]} for failed or rerun runs, at most cap."""
    jobs, todo = {}, []
    for x in runs:
        if x['conclusion'] != 'failure' and x['attempt'] == 1:
            continue
        for a in range(1, x['attempt'] + 1):
            todo.append((x['databaseId'], a))
    skipped = max(len(todo) - cap, 0)

    def one(key):
        rid, a = key
        url = f'repos/{slug}/actions/runs/{rid}/attempts/{a}/jobs?per_page=100'
        try:
            data = json.loads(run(['gh', 'api', url]))
        except (StatsError, json.JSONDecodeError):
            return key, None
        return key, [
            {
                'name': j['name'],
                'conclusion': j['conclusion'],
                'started': j.get('started_at'),
                'completed': j.get('completed_at'),
                'failed_steps': [
                    s['name']
                    for s in j.get('steps') or []
                    if s.get('conclusion') == 'failure'
                ],
            }
            for j in data.get('jobs', [])
        ]

    with ThreadPoolExecutor(max_workers=8) as pool:
        for key, js in pool.map(one, todo[:cap]):
            if js is not None:
                jobs[key] = js
    return jobs, skipped


def is_infra(job, step_re, job_re):
    steps = job['failed_steps']
    return (
        not steps
        or any(step_re.search(s) for s in steps)
        or bool(job_re.search(job['name']))
    )


def classify_failures(jobs, step_re, job_re, tz):
    counts = collections.Counter()
    hrs = collections.Counter()
    real_groups = collections.Counter()
    for (_rid, _a), js in jobs.items():
        failed = [j for j in js if j['conclusion'] == 'failure']
        if not failed:
            continue
        real = [j for j in failed if not is_infra(j, step_re, job_re)]
        kind = 'real-step' if real else 'infra-only'
        counts[kind] += 1
        st = [parse_time(j['started'], tz) for j in js if j['started']]
        en = [parse_time(j['completed'], tz) for j in js if j['completed']]
        if st and en:
            hrs[kind] += (max(en) - min(st)).total_seconds() / 3600
        if real:
            g = ' + '.join(sorted({j['name'].split(' ')[0] for j in real}))
            real_groups[g] += 1
    return {
        'attempts': dict(counts),
        'hours': {k: round(v, 1) for k, v in hrs.items()},
        'real_by_job': dict(real_groups.most_common()),
    }


# --- reviews ---------------------------------------------------------------


def verdict(line):
    up = line.strip().upper()
    if up.startswith('APPROVE'):
        return 'approve'
    if up.startswith(('CHANGES', 'NOT ACCEPTED', 'REJECT', 'BLOCK')):
        return 'block'
    return None


def review_files(paths):
    for p in paths:
        p = os.path.expanduser(p)
        if glob.has_magic(p):
            yield from (Path(f) for f in sorted(glob.glob(p)))
            continue
        path = Path(p)
        if path.is_dir():
            yield from sorted(
                f for f in path.iterdir() if f.is_file() and f.suffix == '.md'
            )
        elif path.is_file():
            yield path


def resolve_sha(repo, sha):
    proc = subprocess.run(
        ['git', 'rev-parse', '--verify', '--quiet', sha + '^{commit}'],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() or sha


def reviews(repo, paths, reviewers):
    rows, ignored = [], 0
    for f in review_files(paths):
        try:
            lines = f.read_text(errors='replace').splitlines()
        except OSError:
            ignored += 1
            continue
        v = verdict(lines[0]) if lines else None
        sha = lines[1].strip().lower() if len(lines) > 1 else ''
        if v is None or not SHA_RE.match(sha):
            ignored += 1
            continue
        who = next((n for n, rx in reviewers if rx.search(f.name)), 'unknown')
        rows.append(
            {
                'file': f.name,
                'reviewer': who,
                'verdict': v,
                'sha': resolve_sha(repo, sha),
                'mtime': f.stat().st_mtime,
            }
        )
    tally = collections.Counter((r['reviewer'], r['verdict']) for r in rows)
    per = collections.defaultdict(dict)
    for r in sorted(rows, key=lambda r: r['mtime']):
        per[r['sha']][r['reviewer']] = r['verdict']
    pairs = collections.Counter()
    single = 0
    pair = [n for n, _ in reviewers[:2]]
    for by in per.values():
        if not all(n in by for n in pair) or len(pair) < 2:
            single += 1
            continue
        blocks = sum(by[n] == 'block' for n in pair)
        pairs[('both approve', 'one blocks', 'both block')[blocks]] += 1
    return {
        'reports': len(rows),
        'ignored_files': ignored,
        'by_reviewer': {
            f'{who} {v}': n for (who, v), n in sorted(tally.items())
        },
        'shas': len(per),
        'pair': pair,
        'same_sha_pairs': dict(pairs),
        'shas_not_paired': single,
    }


# --- output ----------------------------------------------------------------


def report(res):
    print(f'phase {res["phase"]} ({res["commits"]} commits read)')
    print('\nLandings per day')
    for day, n in res['landings_per_day'].items():
        print(f'  {day}  {n}')
    print(f'  total {sum(res["landings_per_day"].values())}')

    print('\nPackages (first landing, landings, commits, tests, span h)')
    for p in res['packages']:
        print(
            f'  {p["first_landing"][:16]}  {p["package"]:<8} '
            f'{p["landings"]:>3} {p["commits"]:>4} {p["test_commits"]:>4} '
            f'{p["span_h"]:>7.1f}'
        )
    print('\nStage hours per landing')
    hdr = ('tests', 'impl', 'review', 'tail', 'total', 'fix', 'tst')
    print('  ' + f'{"landing":<56}' + ''.join(f'{h:>8}' for h in hdr))
    tot = collections.Counter()
    for s in res['stages']:
        vals = (
            s['tests_lead_h'],
            s['impl_h'],
            s['review_fix_h'],
            s['tail_h'],
            s['total_h'],
        )
        for k, v in zip(hdr, vals, strict=False):
            tot[k] += v
        print(
            f'  {s["landing"][:56]:<56}'
            + ''.join(f'{v:>8.1f}' for v in vals)
            + f'{s["fix_commits"]:>8}{s["test_commits"]:>8}'
        )
    print('  totals: ' + ', '.join(f'{k} {tot[k]:.1f}h' for k in hdr[:5]))

    ci = res.get('ci')
    if ci:
        print(
            f'\nCI per day ({ci["runs"]} runs on {res["branch"]} and '
            f'{res["branch"]}-*)'
        )
        print('  day         runs green  red cancel other rerun median_min')
        for day, d in ci['daily'].items():
            med = '' if d['median_min'] is None else f'{d["median_min"]:.1f}'
            print(
                f'  {day}  {d["runs"]:>4} {d["green"]:>5} {d["red"]:>4} '
                f'{d["cancelled"]:>6} {d["other"]:>5} {d["reruns"]:>5} '
                f'{med:>10}'
            )
        f = ci.get('failures')
        if f:
            print(
                '\nFailed run attempts: '
                + ', '.join(f'{k} {v}' for k, v in f['attempts'].items())
                + ' (hours: '
                + ', '.join(f'{k} {v}' for k, v in f['hours'].items())
                + ')'
            )
            if ci.get('jobs_not_fetched'):
                print(
                    f'  {ci["jobs_not_fetched"]} attempt(s) not fetched '
                    '(--max-job-fetch)'
                )
            for g, n in f['real_by_job'].items():
                print(f'  real-step {n:>4}  {g}')
    elif res.get('ci_error'):
        print(f'\nCI: skipped ({res["ci_error"]})')

    rv = res.get('reviews')
    if rv:
        print(
            f'\nReviews: {rv["reports"]} verdict reports on {rv["shas"]} '
            f'SHAs ({rv["ignored_files"]} other files ignored)'
        )
        for k, n in rv['by_reviewer'].items():
            print(f'  {k:<24} {n}')
        print(
            f'  same SHA, {" and ".join(rv["pair"])}: '
            + ', '.join(
                f'{k} {v}' for k, v in sorted(rv['same_sha_pairs'].items())
            )
            + f'; SHAs not reviewed by both {rv["shas_not_paired"]}'
        )


# --- the slip report (T3.0) -------------------------------------------------


def read_forecast(path, tz):
    text = Path(path).read_text(encoding='utf-8')
    try:
        document = json.loads(text)
    except ValueError:
        import yaml

        document = yaml.safe_load(text)
    if (
        not isinstance(document, dict)
        or 'start' not in document
        or not isinstance(document.get('packages'), list)
    ):
        raise StatsError(
            f'{path}: a forecast is {{start, packages: [{{id, hours}}]}}'
        )
    start = parse_time(str(document['start']), tz)
    planned, at = [], start
    for item in document['packages']:
        if (
            not isinstance(item, dict)
            or 'id' not in item
            or 'hours' not in item
        ):
            raise StatsError(f'{path}: each package has an id and hours')
        at = at + dt.timedelta(hours=float(item['hours']))
        planned.append((str(item['id']), float(item['hours']), at))
    return start, planned


def slip_report(rows, start, planned, now):
    r"""The slip at ``now`` and its main cause, in plain words, as lines."""
    landed = {}
    order = []
    for r in rows:
        if r['land'] and r['committed'] <= now and r['land'] not in landed:
            landed[r['land']] = r['committed']
            order.append(r['land'])
    ids = {p for p, _, _ in planned}
    late, due = None, None
    for package, _, forecast in planned:
        if package in landed:
            late = (package, forecast, landed[package])
        elif forecast <= now and due is None:
            due = (package, forecast)
    slip, why = 0.0, 'on the forecast'
    if late is not None and hours(late[1], late[2]) > 0:
        slip = hours(late[1], late[2])
        why = (
            f'{late[0]} was forecast to land by {late[1]:%Y-%m-%d %H:%M} '
            f'and landed at {late[2]:%Y-%m-%d %H:%M}'
        )
    if due is not None and hours(due[1], now) > slip:
        slip = hours(due[1], now)
        why = f'{due[0]} was due by {due[1]:%Y-%m-%d %H:%M} and has not landed'
    # Each landing's own hours on the queue: from the landing before it
    # (or the start) to its own.
    over, unplanned, previous = [], [], start
    forecast_hours = {p: h for p, h, _ in planned}
    for package in order:
        at = landed[package]
        took = hours(max(previous, start), at) if at > start else 0.0
        previous = max(previous, at)
        if package in forecast_hours:
            over.append(
                (round(took - forecast_hours[package], 2), package, took)
            )
        elif took > 0:
            unplanned.append((package, took))
    # The next forecast package, not landed, holds the queue since then.
    pending = next((p for p, _, _ in planned if p not in landed), None)
    if pending is not None and now > previous:
        took = hours(max(previous, start), now)
        over.append(
            (
                round(took - forecast_hours[pending], 2),
                f'{pending} (not landed)',
                took,
            )
        )
        forecast_hours[f'{pending} (not landed)'] = forecast_hours[pending]
    lines = [
        f'Slip at {now:%Y-%m-%d %H:%M}: {slip:.1f} hours behind the forecast ({why}).',
        f'{sum(p in landed for p in ids)} of {len(planned)} forecast packages landed; '
        f'{len(unplanned)} landings the forecast does not hold.',
    ]
    over.sort(reverse=True)
    extra = round(sum(t for _, t in unplanned), 2)
    if over and over[0][0] > 0 and over[0][0] >= extra:
        first = over[0]
        lines.append(
            f'Main cause: {first[1]} held the queue {first[2]:.1f} hours against '
            f'{forecast_hours[first[1]]:g} forecast, {first[0]:.1f} over.'
        )
        rest = [f'{p} {o:.1f} over' for o, p, _ in over[1:] if o > 0]
        if rest:
            lines.append('Then: ' + ', '.join(rest) + '.')
    elif extra > 0:
        lines.append(
            f'Main cause: {extra:.1f} hours of landings the forecast does not hold: '
            + ', '.join(f'{p} {t:.1f}' for p, t in unplanned)
            + '.'
        )
    elif slip > 0:
        lines.append(f'Main cause: {why}.')
    return slip, lines


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('\n\n', 1)[1],
    )
    ap.add_argument(
        '--phase-branch', required=True, help='ref to read, e.g. origin/<phase>'
    )
    ap.add_argument('--base', help='read only BASE..PHASE')
    ap.add_argument('--repo', default='.')
    ap.add_argument('--remote', default='origin')
    ap.add_argument('--package-re', default=DEFAULT_PACKAGE_RE)
    ap.add_argument(
        '--utc-offset',
        type=float,
        help='hours east of UTC for days (default: local)',
    )
    ap.add_argument(
        '--reviews',
        action='append',
        default=[],
        help='review report directory or file (repeatable)',
    )
    ap.add_argument(
        '--reviewer',
        action='append',
        default=[],
        help='NAME=REGEX on the file name (repeatable)',
    )
    ap.add_argument('--no-ci', action='store_true')
    ap.add_argument(
        '--no-jobs',
        action='store_true',
        help='skip the per-job infra/real split',
    )
    ap.add_argument('--gh-repo', help='OWNER/NAME (default: from --remote)')
    ap.add_argument('--workflow', help='only this workflow name')
    ap.add_argument('--since', help='first CI day (default: first commit)')
    ap.add_argument('--limit', type=int, default=5000)
    ap.add_argument('--max-job-fetch', type=int, default=500)
    ap.add_argument('--infra-step-re', default=INFRA_STEP_RE)
    ap.add_argument('--infra-job-re', default=INFRA_JOB_RE)
    ap.add_argument('--json', help='also write everything as JSON')
    ap.add_argument(
        '--forecast', help="the design's forecast: print the slip report alone"
    )
    ap.add_argument('--now', help='the slip report at TIME (default: now)')
    args = ap.parse_args(argv)

    tz = (
        dt.timezone(dt.timedelta(hours=args.utc_offset))
        if args.utc_offset is not None
        else dt.datetime.now().astimezone().tzinfo
    )
    reviewers = []
    for spec in args.reviewer or ['codex=codex', 'subagent=subagent']:
        name, sep, rx = spec.partition('=')
        if not sep:
            ap.error(f'--reviewer wants NAME=REGEX, got {spec!r}')
        reviewers.append((name, re.compile(rx, re.I)))
    try:
        rows = classify_commits(
            commits(args.repo, args.phase_branch, args.base, tz),
            args.package_re,
        )
        if args.forecast:
            start, planned = read_forecast(args.forecast, tz)
            now = (
                parse_time(args.now, tz)
                if args.now
                else dt.datetime.now(dt.timezone.utc).astimezone(tz)
            )
            _, lines = slip_report(rows, start, planned, now)
            print('\n'.join(lines))
            return 0
        groups = landing_groups(rows, bool(args.base))
        branch = re.sub(
            rf'^(refs/remotes/)?{re.escape(args.remote)}/|^refs/heads/',
            '',
            args.phase_branch,
        )
        res = {
            'phase': args.phase_branch,
            'branch': branch,
            'commits': len(rows),
            'landings_per_day': landings_per_day(rows),
            'packages': packages(rows, groups),
            'stages': stages(groups),
        }
        if not args.no_ci:
            try:
                slug = args.gh_repo or gh_slug(args.repo, args.remote)
                firsts = [g[0]['authored'] for _, g in groups]
                since = args.since or (
                    min(firsts).date().isoformat() if firsts else '1970-01-01'
                )
                runs = ci_runs(slug, branch, since, args.limit, args.workflow)
                ci = {
                    'runs': len(runs),
                    'since': since,
                    'daily': ci_daily(runs, tz),
                }
                if not args.no_jobs:
                    jobs, skipped = fetch_jobs(slug, runs, args.max_job_fetch)
                    ci['failures'] = classify_failures(
                        jobs,
                        re.compile(args.infra_step_re),
                        re.compile(args.infra_job_re),
                        tz,
                    )
                    ci['jobs_not_fetched'] = skipped
                res['ci'] = ci
            except StatsError as e:
                res['ci_error'] = str(e)
        if args.reviews:
            res['reviews'] = reviews(args.repo, args.reviews, reviewers)
    except StatsError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    report(res)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
