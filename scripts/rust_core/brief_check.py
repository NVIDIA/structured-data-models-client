#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A package brief answers the hard questions before anyone writes code
(learnings.md, lesson 16; T3.1).

    brief_check.py BRIEF [BRIEF ...] [--template FILE]

The required headings are the ``## `` lines inside the fenced Markdown block
of the package brief template,
``docs/design/rust-core/orchestration/briefs/package.md`` (``--template``
names another). A brief passes when it holds each of them as a line of its
own, outside any fenced block, in any order; a heading may be followed by
"none" and a reason, but not left out. Exit 0 when every brief passes, 1
naming each missing heading of each brief that does not, 2 when the
template or a brief cannot be read or the template holds no heading.

The orchestrator runs it on every brief before it starts the brief's test
author or implementer; T3.2's tooling train makes ``lane.py start`` refuse
a brief it fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / 'docs/design/rust-core/orchestration/briefs/package.md'


class Unreadable(Exception):
    r"""A template or brief that cannot be used."""


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as error:
        raise Unreadable(f'{path}: {error}') from None


def _fenced(text: str) -> tuple[list[str], list[str]]:
    r"""The lines inside fenced blocks, and the lines outside them."""
    inside, outside, fence = [], [], False
    for line in text.splitlines():
        if line.lstrip().startswith('```'):
            fence = not fence
            continue
        (inside if fence else outside).append(line.rstrip())
    return inside, outside


def required(template: Path) -> list[str]:
    r"""The ``## `` headings of the template's fenced brief, in order."""
    inside, _ = _fenced(_read(template))
    headings = [line for line in inside if line.startswith('## ')]
    if not headings:
        raise Unreadable(f'{template}: no "## " heading inside a fenced block')
    return headings


def missing(brief: Path, headings: list[str]) -> list[str]:
    r"""The required headings ``brief`` does not hold."""
    _, outside = _fenced(_read(brief))
    present = set(outside)
    return [heading for heading in headings if heading not in present]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('briefs', nargs='+', type=Path)
    parser.add_argument('--template', type=Path, default=TEMPLATE)
    arguments = parser.parse_args(argv)
    try:
        headings = required(arguments.template)
        problems = {
            str(brief): missing(brief, headings) for brief in arguments.briefs
        }
    except Unreadable as error:
        print(f'brief_check.py: {error}', file=sys.stderr)
        return 2
    failed = False
    for brief, lacking in problems.items():
        if lacking:
            failed = True
            names = ', '.join(heading[3:] for heading in lacking)
            print(
                f'{brief}: missing the heading(s) {names} (template: briefs/package.md)'
            )
        else:
            print(f'{brief}: ok')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
