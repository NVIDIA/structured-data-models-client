# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading agent transcripts for the watchdogs, failing loudly.

``stuck_agents.py`` and ``rule_watch.py`` run in a silent loop that keeps
only the lines its ``grep`` matches (prompts.md, "Watchdog texts"). A
watchdog that cannot see the transcripts must therefore say so on a line
that loop keeps, never on stderr or under another prefix: every problem is
printed to stdout as ``WATCHDOG-ERROR <tool>: <what>`` and the tool exits 1.

What counts as a problem:

- the tasks directory does not exist or is not a directory;
- a transcript that cannot be read (permissions, not UTF-8);
- a line that is not a JSON object, except the file's last line when it
  has no trailing newline: the agent is still writing that line, so a
  partial last line is normal and is skipped, not flagged.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections.abc import Iterator

PREFIX = 'WATCHDOG-ERROR'


class Problems:
    """Collects the run's problems, each printed at once."""

    def __init__(self, tool: str) -> None:
        self.tool = tool
        self.count = 0

    def report(self, what: str) -> None:
        self.count += 1
        print(f'{PREFIX} {self.tool}: {what}', flush=True)

    def exit_status(self) -> int:
        return 1 if self.count else 0


def tasks_dir(path: str, problems: Problems) -> pathlib.Path | None:
    """The tasks directory, or None (reported) when there is none."""
    tasks = pathlib.Path(path)
    if not tasks.exists():
        problems.report(f'the tasks directory {tasks} does not exist')
        return None
    if not tasks.is_dir():
        problems.report(f'the tasks directory {tasks} is not a directory')
        return None
    return tasks


def records(path: pathlib.Path, problems: Problems) -> Iterator[dict]:
    """The transcript's records, in order; problems reported, never raised.

    An unreadable file yields nothing; bad lines are skipped and reported
    once per file, after the good ones; a partial last line is skipped
    silently."""
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as error:
        problems.report(f'{path.name}: cannot be read ({error})')
        return
    lines = text.split('\n')
    # The last element follows the last newline: empty when every line is
    # complete, else the line the agent is still writing, never judged.
    lines.pop()
    bad = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            record = None
        if not isinstance(record, dict):
            bad.append(number)
            continue
        yield record
    if bad:
        more = f' (and {len(bad) - 1} more)' if len(bad) > 1 else ''
        problems.report(
            f'{path.name}: line {bad[0]}{more} is not a JSON object'
        )


def guarded(tool: str, main) -> None:
    """Runs ``main(problems)`` and exits with its status; an unexpected
    exception is a WATCHDOG-ERROR line too, never a traceback alone on
    stderr, which the loop drops."""
    problems = Problems(tool)
    try:
        status = main(problems)
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001 - every failure must be seen
        problems.report(f'crashed: {type(error).__name__}: {error}')
        status = 1
    sys.exit(status or problems.exit_status())
