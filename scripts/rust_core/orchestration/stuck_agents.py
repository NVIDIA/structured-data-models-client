#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report subagents whose last tool call has had no result for too long.

    stuck_agents.py [TASKS_DIR] [MINUTES]

A tool call waiting on the owner's approval never gets a result, so it shows
up here. Long waits that are expected (gh run watch, until/sleep loops,
herdr agent wait, gate.sh) are labelled as such rather than flagged, until
they pass a ceiling: an expected wait pending for longer than
$SDM_EXPECTED_WAIT_CEILING_MIN minutes (default 90) is reported as
POSSIBLY-BLOCKED too, since a hung loop or gate run looks the same. Prints one
line per call pending for MINUTES (default 15) or more:
  POSSIBLY-BLOCKED|expected-wait <agent> <age>min <tool>[ (note)]: <input>

TASKS_DIR holds the agents' a*.output transcripts; it defaults to
$SDM_TASKS_DIR and one of the two is required. A watchdog that cannot see
fails loudly (transcripts.py): a missing tasks directory, an unreadable
transcript or a line in one that is not JSON (a partial last line, still
being written, excepted) prints
  WATCHDOG-ERROR stuck_agents.py: <what>
and exits 1; the loop's grep keeps those lines.
"""

import argparse
import datetime
import json
import os
import time

import transcripts

EXPECTED = (
    'gh run watch',
    'until ',
    'herdr',
    'sleep ',
    'prepush.sh --full',
    'gate.sh',
)
CEILING_MIN = 90  # an expected wait older than this is flagged anyway


def main(problems):
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
    try:
        ceiling_min = float(
            os.environ.get('SDM_EXPECTED_WAIT_CEILING_MIN') or CEILING_MIN
        )
    except ValueError:
        ap.error('SDM_EXPECTED_WAIT_CEILING_MIN must be a number of minutes')
    tasks = transcripts.tasks_dir(a.tasks_dir, problems)
    if tasks is None:
        return 1
    limit_min = a.minutes

    now = time.time()
    for f in sorted(tasks.glob('a*.output')):
        try:
            if now - f.stat().st_mtime > 6 * 3600:
                continue
            pending = {}
            for d in transcripts.records(f, problems):
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
                expected = any(e in inp for e in EXPECTED) and not risky
                kind, note = 'POSSIBLY-BLOCKED', ''
                if expected and age > ceiling_min:
                    # Past the ceiling: a hung `until` loop or gate.sh run, or an approval wait.
                    note = f' (expected wait past the {ceiling_min:.0f}min ceiling)'
                elif expected:
                    kind = 'expected-wait'
                print(f'{kind} {f.stem} {age:.0f}min {name}{note}: {inp[:240]}')
        except Exception as e:  # noqa: BLE001 - a record of another shape
            problems.report(f'{f.name}: {type(e).__name__}: {e}')
    return problems.exit_status()


if __name__ == '__main__':
    transcripts.guarded('stuck_agents.py', main)
