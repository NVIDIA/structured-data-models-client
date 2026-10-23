#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Codex reviews in herdr panes: one pane per package, kept alive.

Every Codex review step of the Rust core migration goes through this helper,
so the herdr and Codex details are written once:

- ``open``: create the package's tab in the ``codex-reviews`` workspace, a
  review worktree for it, and start Codex there with the owner's ``cx``
  alias (``codex --yolo``) on gpt-6-astra at high reasoning effort.
- ``review``: check the worktree out at the exact SHA and send the round's
  prompt, which points at a brief file and names a fresh report file.
  It needs no handoff stamp: the stamp is required only by landing.
  At most ``SDM_CODEX_CAP`` (default 4) reviews run at once; ``--queue``
  waits for a free slot, otherwise the command exits 3.
- ``wait``: block until the round settles and print one line:
  ``REPORT``, ``BLOCKED``, ``IDLE-NO-REPORT``, ``CAPACITY``, ``GONE`` or
  ``TIMEOUT``. A capacity error is retried with backoff first. Run it in
  the background; it is the waiter the handbook asks for.
- ``list``: the registry, with each pane's live state.

There is no close command: panes stay alive after their package lands.
State lives in ``$SDM_ORCH_STATE/codex-panes.json``. Nothing here prints
herdr's raw JSON, which carries terminal titles with user and host names.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

STATE = Path(
    os.environ.get(
        'SDM_ORCH_STATE', Path.home() / '.local/share/sdm-orchestration'
    )
)


REPO = Path(os.environ.get('SDM_REPO', Path.home() / 'code/sdmc-rust-core'))
HERDR = os.environ.get('SDM_HERDR', 'herdr')
CAP = int(os.environ.get('SDM_CODEX_CAP', '4'))
MODEL = os.environ.get('SDM_CODEX_MODEL', 'gpt-6-astra')
EFFORT = os.environ.get('SDM_CODEX_EFFORT', 'high')
WORKSPACE = 'codex-reviews'
REGISTRY = STATE / 'codex-panes.json'
WORKTREES = STATE / 'review-worktrees'
REPORTS = STATE / 'codex-reports'
POLL = float(os.environ.get('SDM_CODEX_POLL', '15'))
GRACE = float(os.environ.get('SDM_CODEX_GRACE', '60'))
OPEN_TIMEOUT = float(os.environ.get('SDM_CODEX_OPEN_TIMEOUT', '90'))
BACKOFF = [
    float(s)
    for s in os.environ.get('SDM_CODEX_BACKOFF', '60,120,240').split(',')
]
PACKAGE = re.compile(r'(T[0-9]+\.[0-9]+[a-z]?|CI[0-9]+)(-[a-z0-9]+)?')
VERDICTS = ('APPROVE', 'CHANGES REQUIRED')
CAPACITY = re.compile(
    r'capacity|rate.?limit|\b429\b|overloaded|too many requests|try again later|'
    r'high demand|stream disconnected|usage limit',
    re.IGNORECASE,
)
SETTLED = ('idle', 'done')


class Failure(Exception):
    """A refusal with a message for the orchestrator."""


def herdr(*args: str, json_out: bool = True) -> dict | str:
    proc = subprocess.run(
        [HERDR, *args],
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip()
        with contextlib.suppress(ValueError):
            error = json.loads(detail).get('error', {})
            detail = f'{error.get("code", "")}: {error.get("message", "")}'
        raise Failure(
            f'herdr {args[0]} {args[1] if len(args) > 1 else ""} failed: {detail[:300]}'
        )
    if not json_out:
        return proc.stdout
    if not proc.stdout.strip():
        return {}
    try:
        return json.loads(proc.stdout).get('result', {})
    except ValueError as error:
        raise Failure(
            f'herdr {args[0]} {args[1] if len(args) > 1 else ""} gave output that is not JSON'
        ) from error


def git(*args: str, cwd: Path = REPO) -> str:
    proc = subprocess.run(
        ['git', '-C', str(cwd), *args],
        capture_output=True,
        text=True,
        timeout=600,
        stdin=subprocess.DEVNULL,
        env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'},
    )
    if proc.returncode != 0:
        raise Failure(
            f'git {" ".join(args)} failed: {proc.stderr.strip()[:300]}'
        )
    return proc.stdout.strip()


def agent_name(package: str) -> str:
    if not PACKAGE.fullmatch(package):
        raise Failure(
            f'{package!r} is not a package ID (T<n>.<n>[a-z] or CI<n>, optional -<suffix>)'
        )
    return 'cx-' + package.lower().replace('.', '-')


@contextlib.contextmanager
def registry():
    """The registry, locked for the duration, written back on exit."""
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / '.codex-panes.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
        yield data
        tmp = REGISTRY.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
        tmp.replace(REGISTRY)


def live_agents() -> dict[str, dict]:
    agents = herdr('agent', 'list').get('agents', [])
    return {a['name']: a for a in agents if a.get('name')}


def agent_status(name: str) -> str | None:
    agent = live_agents().get(name)
    return agent.get('agent_status') if agent else None


def pane_text(name: str, lines: int = 60) -> str:
    return herdr(
        'agent',
        'read',
        name,
        '--source',
        'recent-unwrapped',
        '--lines',
        str(lines),
        json_out=False,
    )


def workspace_id() -> str:
    for ws in herdr('workspace', 'list').get('workspaces', []):
        if ws.get('label') == WORKSPACE:
            return ws['workspace_id']
    created = herdr(
        'workspace',
        'create',
        '--label',
        WORKSPACE,
        '--cwd',
        str(REPO),
        '--no-focus',
    )
    return created['workspace']['workspace_id']


def report_verdict(report: Path, sha: str) -> str | None:
    """The report's verdict if it is complete enough for land.py, else None."""
    if not report.is_file():
        return None
    text = report.read_text(errors='replace')
    first = text.split('\n', 1)[0].strip()
    if first in VERDICTS and sha in text:
        return first
    return None


# --- open ---------------------------------------------------------------


def cmd_open(args: argparse.Namespace) -> int:
    name = agent_name(args.package)
    with registry() as reg:
        entry = reg.get(args.package)
        if entry and name in live_agents():
            print(
                f'OPEN {args.package} {name} (already running) worktree={entry["worktree"]}'
            )
            return 0
        worktree = WORKTREES / args.package
        if not worktree.exists():
            WORKTREES.mkdir(parents=True, exist_ok=True)
            git('worktree', 'add', '--detach', str(worktree), args.base)
        REPORTS.mkdir(parents=True, exist_ok=True)
        ws = workspace_id()
        leftover = [
            t
            for t in herdr('tab', 'list', '--workspace', ws).get('tabs', [])
            if t.get('label') == args.package
        ]
        codex = [
            a
            for a in herdr('agent', 'list').get('agents', [])
            if a.get('agent') == 'codex'
            and a.get('tab_id') in {t['tab_id'] for t in leftover}
        ]
        if codex:
            # A half-finished open left Codex running in the package's tab: adopt it.
            tab = {'tab': {'tab_id': codex[0]['tab_id']}}
            pane = codex[0]['pane_id']
        else:
            if leftover:
                raise Failure(
                    f'tab {args.package} exists with no Codex in it; inspect it before opening again'
                )
            tab = herdr(
                'tab',
                'create',
                '--workspace',
                ws,
                '--cwd',
                str(worktree),
                '--label',
                args.package,
                '--no-focus',
            )
            pane = tab['root_pane']['pane_id']
            herdr('pane', 'rename', pane, name)
            command = f'cx -m {MODEL} -c model_reasoning_effort={EFFORT} -C {worktree}'
            herdr('pane', 'run', pane, command)
        deadline = time.monotonic() + OPEN_TIMEOUT
        while time.monotonic() < deadline:
            occupant = [
                a
                for a in herdr('agent', 'list').get('agents', [])
                if a.get('pane_id') == pane and a.get('agent') == 'codex'
            ]
            if occupant and occupant[0].get('agent_status') in SETTLED:
                break
            time.sleep(2)
        else:
            raise Failure(
                f'Codex did not come up in pane {pane} within {OPEN_TIMEOUT:.0f} s; inspect it with herdr pane read {pane}'
            )
        herdr('agent', 'rename', pane, name)
        reg[args.package] = {
            'name': name,
            'pane': pane,
            'tab': tab['tab']['tab_id'],
            'worktree': str(worktree),
            'rounds': (entry or {}).get('rounds', []),
            'status': 'open',
        }
    print(f'OPEN {args.package} {name} pane={pane} worktree={worktree}')
    return 0


# --- review -------------------------------------------------------------


def refresh(reg: dict) -> None:
    """Settle rounds whose report is in and whose pane is no longer working."""
    agents = None
    for entry in reg.values():
        if entry.get('status') != 'reviewing':
            continue
        current = entry['rounds'][-1]
        verdict = report_verdict(Path(current['report']), current['sha'])
        if verdict is None:
            continue
        agents = agents if agents is not None else live_agents()
        state = (agents.get(entry['name']) or {}).get('agent_status')
        if state != 'working':
            current['verdict'] = verdict
            entry['status'] = 'open'


def reviewing(reg: dict) -> list[str]:
    return [
        package
        for package, entry in reg.items()
        if entry.get('status') == 'reviewing'
    ]


def prompt_text(
    package: str, rnd: int, sha: str, brief: Path, report: Path, worktree: str
) -> str:
    return (
        f'Review round {rnd} of package {package}, at commit {sha}. '
        f'The worktree {worktree} is checked out at exactly that commit; review it there and change nothing in it, '
        f'in the repository or on GitHub. Read the brief {brief} in full and follow it. '
        f'Write your report to {report}: line 1 exactly APPROVE or CHANGES REQUIRED, line 2 the full SHA {sha}, '
        f'then the findings as the brief says. Write the report last, when the review is complete.'
    )


def cmd_review(args: argparse.Namespace) -> int:
    name = agent_name(args.package)
    brief = Path(args.brief).resolve()
    if not brief.is_file():
        raise Failure(f'no brief at {brief}')
    report = Path(args.report).resolve() if args.report else None
    while True:
        with registry() as reg:
            entry = reg.get(args.package)
            if not entry:
                raise Failure(f'no pane for {args.package}; run open first')
            refresh(reg)
            if entry.get('status') == 'reviewing':
                raise Failure(
                    f'{args.package} is still reviewing round {entry["rounds"][-1]["round"]}; wait for it first'
                )
            busy = reviewing(reg)
            if len(busy) < CAP:
                state = agent_status(name)
                if state is None:
                    raise Failure(f'{name} is not running; run open again')
                if state not in SETTLED:
                    raise Failure(
                        f'{name} is {state}, not idle; inspect it with herdr agent read {name}'
                    )
                rnd = len(entry['rounds']) + 1
                report = report or REPORTS / f'{args.package}-r{rnd}.md'
                if report.exists():
                    raise Failure(
                        f'{report} already exists; each round needs a fresh report file'
                    )
                worktree = Path(entry['worktree'])
                git('fetch', '--quiet', 'origin')
                sha = git('rev-parse', '--verify', f'{args.sha}^{{commit}}')
                if git(
                    'status',
                    '--porcelain',
                    '--untracked-files=no',
                    cwd=worktree,
                ):
                    raise Failure(
                        f'{worktree} has tracked changes; the reviewer changed files, look before reusing it'
                    )
                git('checkout', '--quiet', '--detach', sha, cwd=worktree)
                if git('rev-parse', 'HEAD', cwd=worktree) != sha:
                    raise Failure(f'{worktree} is not at {sha} after checkout')
                text = prompt_text(
                    args.package, rnd, sha, brief, report, str(worktree)
                )
                if args.extra:
                    text += ' ' + Path(args.extra).read_text().strip()
                herdr('agent', 'prompt', name, text)
                entry['rounds'].append(
                    {
                        'round': rnd,
                        'sha': sha,
                        'brief': str(brief),
                        'report': str(report),
                        'prompt': text,
                        'sent': time.time(),
                        'retries': 0,
                        'verdict': None,
                    }
                )
                entry['status'] = 'reviewing'
                print(f'SENT {args.package} r{rnd} {sha[:12]} report={report}')
                return 0
        if not args.queue:
            print(f'AT-CAP {len(busy)}/{CAP} reviewing: {" ".join(busy)}')
            return 3
        time.sleep(POLL)


# --- wait ---------------------------------------------------------------


def settle(package: str, verdict: str | None, outcome: str) -> None:
    with registry() as reg:
        entry = reg[package]
        entry['rounds'][-1]['verdict'] = verdict
        entry['rounds'][-1]['outcome'] = outcome
        entry['status'] = 'open'


def cmd_wait(args: argparse.Namespace) -> int:
    with registry() as reg:
        entry = reg.get(args.package)
        if not entry or not entry.get('rounds'):
            raise Failure(f'no review round for {args.package}')
        name, current = entry['name'], dict(entry['rounds'][-1])
    report, sha = Path(current['report']), current['sha']
    tag = f'{args.package} r{current["round"]}'
    deadline = time.time() + args.timeout_min * 60
    idle_since = None
    while True:
        verdict = report_verdict(report, sha)
        state = agent_status(name)
        if verdict and state != 'working':
            settle(args.package, verdict, 'report')
            print(f'REPORT {tag} {verdict} {report}')
            return 0
        if state is None:
            settle(args.package, verdict, 'gone')
            print(
                f'GONE {tag}: {name} is no longer running; report {"present" if verdict else "missing"}'
            )
            return 6
        if state == 'blocked':
            print(
                f'BLOCKED {tag}: {name} waits at a dialog; inspect with herdr agent read {name}'
            )
            return 4
        if (
            state in SETTLED
            and time.time() - current['sent'] > GRACE
            and not verdict
        ):
            idle_since = idle_since or time.time()
            if time.time() - idle_since >= 2 * POLL:
                tail = pane_text(name)
                if CAPACITY.search(tail[-3000:]) and current['retries'] < len(
                    BACKOFF
                ):
                    time.sleep(BACKOFF[current['retries']])
                    current['retries'] += 1
                    herdr(
                        'agent',
                        'prompt',
                        name,
                        'The last attempt stopped on a capacity or rate-limit error. Continue the same review: '
                        + current['prompt'],
                    )
                    with registry() as reg:
                        reg[args.package]['rounds'][-1]['retries'] = current[
                            'retries'
                        ]
                    current['sent'] = time.time()
                    idle_since = None
                    continue
                if CAPACITY.search(tail[-3000:]):
                    settle(args.package, None, 'capacity')
                    print(
                        f'CAPACITY {tag}: still failing after {current["retries"]} retries'
                    )
                    return 5
                settle(args.package, None, 'idle-no-report')
                print(
                    f'IDLE-NO-REPORT {tag}: {name} stopped without a valid report at {report}'
                )
                return 5
        else:
            idle_since = None
        if time.time() > deadline:
            print(
                f'TIMEOUT {tag}: still {state} after {args.timeout_min} min; the round stays open'
            )
            return 7
        time.sleep(POLL)


# --- list ---------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    with registry() as reg:
        refresh(reg)
        agents = live_agents()
        for package, entry in sorted(reg.items()):
            state = (agents.get(entry['name']) or {}).get(
                'agent_status', 'gone'
            )
            last = entry['rounds'][-1] if entry['rounds'] else {}
            print(
                f'{package:<10} {entry["name"]:<14} {state:<8} {entry["status"]:<10} '
                f'r{last.get("round", 0)} {last.get("sha", "")[:12]:<12} {last.get("verdict") or "-"}'
            )
        print(f'reviewing {len(reviewing(reg))}/{CAP}')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('open', help='create the package pane and start Codex')
    p.add_argument('--package', required=True)
    p.add_argument(
        '--base',
        default='origin/rust-core',
        help='commit for a new review worktree',
    )
    p.set_defaults(func=cmd_open)
    p = sub.add_parser('review', help='send a review round')
    p.add_argument('--package', required=True)
    p.add_argument('--sha', required=True)
    p.add_argument(
        '--brief',
        required=True,
        help='the round brief, a file outside the review worktree',
    )
    p.add_argument(
        '--report',
        help='default: $SDM_ORCH_STATE/codex-reports/<package>-r<n>.md',
    )
    p.add_argument(
        '--extra', help='a file whose text is appended to the prompt'
    )
    p.add_argument(
        '--queue',
        action='store_true',
        help='wait for a free slot instead of exiting 3',
    )
    p.set_defaults(func=cmd_review)
    p = sub.add_parser(
        'wait', help='wait for the round to settle; one line out'
    )
    p.add_argument('--package', required=True)
    p.add_argument('--timeout-min', type=float, default=90)
    p.set_defaults(func=cmd_wait)
    p = sub.add_parser('list', help='every pane and its last round')
    p.set_defaults(func=cmd_list)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (Failure, subprocess.TimeoutExpired) as error:
        print(
            f'FAILED {getattr(args, "package", "")}: {error}', file=sys.stderr
        )
        return 2


if __name__ == '__main__':
    sys.exit(main())
