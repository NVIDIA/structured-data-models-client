#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report subagents whose last tool call has had no result for too long.

    stuck_agents.py [TASKS_DIR] [MINUTES]

A tool call waiting on the owner's approval never gets a result, so it shows
up here. Long waits that are expected (gh run watch, until/sleep loops,
herdr agent wait) are labelled as such rather than flagged. Prints one line
per call pending for MINUTES (default 15) or more:
  POSSIBLY-BLOCKED|expected-wait <agent> <age>min <tool>: <input>

TASKS_DIR holds the agents' a*.output transcripts; it defaults to
$SDM_TASKS_DIR and one of the two is required.
"""

import argparse
import datetime
import json
import os
import pathlib
import time

EXPECTED = (
    'gh run watch',
    'until ',
    'herdr',
    'sleep ',
    'prepush.sh --full',
    'gate.sh',
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        'tasks_dir', nargs='?', default=os.environ.get('SDM_TASKS_DIR')
    )
    ap.add_argument('minutes', nargs='?', type=float, default=15)
    a = ap.parse_args()
    if not a.tasks_dir:
        ap.error('give TASKS_DIR or set SDM_TASKS_DIR')
    tasks = pathlib.Path(a.tasks_dir)
    limit_min = a.minutes

    now = time.time()
    for f in sorted(tasks.glob('a*.output')):
        try:
            if now - f.stat().st_mtime > 6 * 3600:
                continue
            pending, last_ts = {}, None
            for line in f.open():
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                last_ts = d.get('timestamp', last_ts)
                m = d.get('message', {})
                if not isinstance(m, dict) or not isinstance(
                    m.get('content'), list
                ):
                    continue
                for c in m['content']:
                    if c.get('type') == 'tool_use':
                        pending[c['id']] = (
                            d.get('timestamp'),
                            c.get('name'),
                            json.dumps(c.get('input')),
                        )
                    elif c.get('type') == 'tool_result':
                        pending.pop(c.get('tool_use_id'), None)
            for ts, name, inp in pending.values():
                t = datetime.datetime.fromisoformat(
                    ts.replace('Z', '+00:00')
                ).timestamp()
                age = (now - t) / 60
                if age < limit_min:
                    continue
                risky = any(
                    r in inp for r in ('rm -rf', 'rm -r ', '--force', 'sudo ')
                )
                kind = (
                    'expected-wait'
                    if any(e in inp for e in EXPECTED) and not risky
                    else 'POSSIBLY-BLOCKED'
                )
                print(f'{kind} {f.stem} {age:.0f}min {name}: {inp[:240]}')
        except Exception as e:
            print(f'error {f.name}: {e}')


if __name__ == '__main__':
    main()
