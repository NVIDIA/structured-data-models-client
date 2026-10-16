#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flag process-rule violations in running subagents' transcripts.

    rule_watch.py [TASKS_DIR] [MINUTES]

Prints one line per violation in a tool call made in the last MINUTES (default 10):
  VIOLATION <agent> <rule>: <command>
Rules (owner, 2026-10-12): no local `prepush.sh --full`; implementers don't
`gh run watch`; no `rm -rf`/`rm -r`, `--force` removals or `sudo`.

TASKS_DIR holds the agents' a*.output transcripts; it defaults to $SDM_TASKS_DIR and one
of the two is required. Violations already acknowledged are listed in the ignore file,
$SDM_ORCH_STATE/rule_watch.ignore (state default ~/.local/share/sdm-orchestration): one
pattern per line, a Python regular expression searched in the VIOLATION line (an agent
id alone drops every violation of that agent); blank lines and lines starting with #
are skipped.
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import time

STATE = pathlib.Path(
    os.environ.get(
        'SDM_ORCH_STATE',
        pathlib.Path.home() / '.local/share/sdm-orchestration',
    )
).expanduser()
IGNORE = STATE / 'rule_watch.ignore'
RULES = [
    ('no-local-full', re.compile(r'prepush\.sh\b[^\n|;&]*--full')),
    ('no-ci-watch', re.compile(r'\bgh run watch\b')),
    ('no-rm-r', re.compile(r'\brm\s+-[a-zA-Z]*r')),
    (
        'no-force-remove',
        re.compile(r'worktree remove[^\n]*--force|branch -D\b'),
    ),
    ('no-sudo', re.compile(r'\bsudo\b')),
]


def ignore_patterns(path):
    pats = []
    if not path.exists():
        return pats
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            pats.append(re.compile(line))
        except re.error:
            pats.append(re.compile(re.escape(line)))
    return pats


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        'tasks_dir', nargs='?', default=os.environ.get('SDM_TASKS_DIR')
    )
    ap.add_argument('minutes', nargs='?', type=float, default=10)
    a = ap.parse_args()
    if not a.tasks_dir:
        ap.error('give TASKS_DIR or set SDM_TASKS_DIR')
    tasks = pathlib.Path(a.tasks_dir)
    window_min = a.minutes
    ignored = ignore_patterns(IGNORE)
    now = time.time()
    for f in sorted(tasks.glob('a*.output')):
        if now - f.stat().st_mtime > window_min * 60:
            continue
        try:
            for line in f.open():
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
                m = d.get('message', {})
                if not isinstance(m, dict) or not isinstance(
                    m.get('content'), list
                ):
                    continue
                for c in m['content']:
                    if c.get('type') == 'tool_use' and c.get('name') == 'Bash':
                        cmd = c.get('input', {}).get('command', '')
                        # ignore heredoc bodies (report text)
                        cmd = re.split(r'<<-?\s*[\'"]?[A-Z]+', cmd)[0]
                        for name, rx in RULES:
                            if name == 'no-rm-r' and (
                                'scratchpad' in cmd or '$S/' in cmd
                            ):
                                continue  # deleting inside the session scratchpad needs no approval
                            if name == 'no-local-full' and 'grep' in cmd:
                                continue
                            if rx.search(cmd):
                                v = f'VIOLATION {f.stem} {name}: {cmd[:200]}'
                                if not any(p.search(v) for p in ignored):
                                    print(v)
        except Exception as e:
            print(f'error {f.name}: {e}')


if __name__ == '__main__':
    main()
