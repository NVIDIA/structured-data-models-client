#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The commit-msg hook: a product fix names the old client's rule (T3.0,
item 4).

    commit_msg.py MESSAGE_FILE

A fix commit (subject ``<ID> fix <n>: ...``) that changes a product path
(anything but the judging paths, tests, docs and the scripts; see
``judging.product``) carries a line

    Rule: <the old client's rule, in one sentence>

so the fix ports the reference's rule rather than making a reviewer's
example pass (Phase 2's RCA: 105 findings came from such fixes). Without it
the commit is refused, exit 1, naming the rule. Any other commit passes.
The paths are the staged ones (``git diff --cached``), as the hook sees
them; ``land.py`` makes the same check on every commit it lands
(:func:`rule_problem`), so a commit made without the hook is still caught.

``prepush.sh --install-hook`` installs it as the clone's commit-msg hook,
which runs only in the worktrees registered there (``lane.py start``
registers each lane worktree).
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judging  # noqa: E402
from phases import PACKAGE_ID  # noqa: E402

#: A fix commit's subject: ``T3.1 fix 2: ...``.
FIX_SUBJECT = re.compile(rf'({PACKAGE_ID.pattern}) fix [0-9]+: ')
#: The rule line: ``Rule:`` and at least three words.
RULE_LINE = re.compile(r'^Rule:\s*(\S+\s+){2,}\S+', re.MULTILINE)


def rule_problem(message: str, paths: Iterable[str], rules: judging.Rules) -> str | None:
    """None when the commit with ``message`` changing ``paths`` needs no
    rule line or has one; else why it is refused."""
    lines = [
        line for line in message.splitlines() if not line.startswith('#')
    ]
    subject = lines[0] if lines else ''
    if not FIX_SUBJECT.match(subject):
        return None
    touched = judging.product(rules, paths)
    if not touched:
        return None
    if RULE_LINE.search('\n'.join(lines[1:])):
        return None
    return (
        'a fix commit that changes product code names the old client\'s rule '
        "it ports, in a line 'Rule: <one sentence>' (product paths: "
        + ', '.join(touched[:5])
        + (' ...' if len(touched) > 5 else '')
        + ')'
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print('usage: commit_msg.py MESSAGE_FILE', file=sys.stderr)
        return 2
    message = Path(argv[0]).read_text(encoding='utf-8', errors='replace')
    done = subprocess.run(
        ['git', 'diff', '--cached', '--name-only', '-z'],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    paths = [p for p in done.stdout.split('\0') if p]
    top = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    ).stdout.strip()
    listed = Path(top or '.') / judging.FILE
    rules = judging.patterns(
        listed.read_text(encoding='utf-8') if listed.is_file() else ''
    )
    why = rule_problem(message, paths, rules)
    if why:
        print(f'commit-msg: REFUSED: {why}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
