#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report a lane that no running agent has worked in recently.

    lane_idle.py [--phase BRANCH] [TASKS_DIR] [MINUTES]

A lane is busy if some agent transcript made a tool call mentioning its worktree
(<repo>-lane-<x>, e.g. ~/code/myclone-lane-2a for the repo ~/code/myclone) within the
last MINUTES (default 25). Lanes listed in the done file, <state>/lanes_done (one lane
name per line, e.g. "2a"), are skipped. Prints "IDLE lane-<x>: ..." per idle lane.

Lanes are discovered, not hard-coded: every origin/<phase>-lane-<name> branch and
every <repo>-lane-<name> worktree directory, where <name> matches lane.py's
[a-z0-9]{1,16} (so -tests*, -next and -t132e branches/worktrees are not lanes).

TASKS_DIR holds the agents' a*.output transcripts; it defaults to $SDM_TASKS_DIR and one
of the two is required. Environment:
  SDM_REPO                  the repo (default: the git toplevel of this script)
  SDM_LANE_WORKTREE_PREFIX  the lane worktrees' path prefix (default <repo>-lane-)
  SDM_PHASE_BRANCH          the phase branch, e.g. rust-core-phase-3 (required unless --phase
                            gives it; exits 2 when neither does)
  SDM_ORCH_STATE            state directory (default ~/.local/share/sdm-orchestration)
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import subprocess
import time

LANE = re.compile(r'[a-z0-9]{1,16}')
HERE = pathlib.Path(__file__).resolve().parent
STATE = pathlib.Path(
    os.environ.get(
        'SDM_ORCH_STATE',
        pathlib.Path.home() / '.local/share/sdm-orchestration',
    )
).expanduser()


def repo_dir():
    if os.environ.get('SDM_REPO'):
        return pathlib.Path(os.environ['SDM_REPO']).expanduser()
    top = subprocess.run(
        ['git', '-C', str(HERE), 'rev-parse', '--show-toplevel'],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return pathlib.Path(top)


def lanes(repo, phase, prefix):
    found = set()
    out = subprocess.run(
        [
            'git',
            '-C',
            str(repo),
            'branch',
            '-r',
            '--list',
            f'origin/{phase}-lane-*',
            '--format=%(refname:lstrip=3)',
        ],
        capture_output=True,
        text=True,
    ).stdout
    for b in out.split():
        name = b.removeprefix(f'{phase}-lane-')
        if LANE.fullmatch(name):
            found.add(name)
    for d in prefix.parent.glob(prefix.name + '*'):
        name = d.name.removeprefix(prefix.name)
        if d.is_dir() and LANE.fullmatch(name):
            found.add(name)
    return sorted(found)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        '--phase',
        default=os.environ.get('SDM_PHASE_BRANCH') or '',
        help='the phase branch, e.g. rust-core-phase-3 (default $SDM_PHASE_BRANCH; one is required)',
    )
    ap.add_argument(
        'tasks_dir', nargs='?', default=os.environ.get('SDM_TASKS_DIR')
    )
    ap.add_argument('minutes', nargs='?', type=float, default=25)
    a = ap.parse_args()
    if not a.phase:
        ap.error(
            'set SDM_PHASE_BRANCH or pass --phase: the phase branch, e.g. rust-core-phase-3'
        )
    if not a.tasks_dir:
        ap.error('give TASKS_DIR or set SDM_TASKS_DIR')
    tasks = pathlib.Path(a.tasks_dir)
    window_min = a.minutes
    repo = repo_dir()
    prefix = pathlib.Path(
        os.environ.get('SDM_LANE_WORKTREE_PREFIX') or f'{repo}-lane-'
    ).expanduser()
    # The name a transcript mentions a lane worktree by (e.g. "myclone-lane-").
    mention = prefix.name
    done_file = STATE / 'lanes_done'
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    now = time.time()
    busy = set()
    all_lanes = lanes(repo, a.phase, prefix)
    for f in tasks.glob('a*.output'):
        try:
            if now - f.stat().st_mtime > window_min * 60:
                continue
            for line in f.open():
                if mention not in line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                ts = d.get('timestamp')
                if not ts:
                    continue
                t = datetime.datetime.fromisoformat(
                    ts.replace('Z', '+00:00')
                ).timestamp()
                if now - t > window_min * 60:
                    continue
                for lane in all_lanes:
                    if re.search(
                        rf'{re.escape(mention)}{lane}(?![a-z0-9])', line
                    ):
                        busy.add(lane)
        except Exception:
            pass
    for lane in all_lanes:
        if lane not in busy and lane not in done:
            print(
                f'IDLE lane-{lane}: no agent has worked in it for {window_min:.0f} min'
            )


if __name__ == '__main__':
    main()
