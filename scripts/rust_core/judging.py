# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The judging paths (T3.0, item 1).

``judging-paths.txt`` lists the paths that decide whether a candidate
passes, in ``contract/protected-paths.txt``'s pattern syntax. ``land.py``
reads it from the phase head (``judging_rules``), so a package cannot take
its own change out of the list. A package may change judging paths and
product code together and land normally (the owner, 2026-10-23: the train
rule is gone, T3.24).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

#: The list, as a path in the tree.
FILE = 'scripts/rust_core/judging-paths.txt'
#: Changes that make a landing a product landing: anything outside these.
NOT_PRODUCT = ('docs/**', 'tests/**', 'contract/tests/**', '.github/**')

Rules = list[tuple[bool, 're.Pattern[str]']]


def patterns(text: str | None) -> Rules:
    r"""The patterns of ``text``, each with whether it takes paths out."""
    found: Rules = []
    for raw in (text or '').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        negate = line.startswith('!')
        pattern = line[1:] if negate else line
        regex, index = '', 0
        while index < len(pattern):
            if pattern.startswith('**', index):
                regex += '.*'
                index += 2
            elif pattern[index] == '*':
                regex += '[^/]*'
                index += 1
            else:
                regex += re.escape(pattern[index])
                index += 1
        found.append((negate, re.compile(regex, re.DOTALL)))
    return found


def matches(rules: Rules, path: str) -> bool:
    hit = False
    for negate, regex in rules:
        if regex.fullmatch(path):
            hit = not negate
    return hit


def judging(rules: Rules, paths: Iterable[str]) -> list[str]:
    """The judging paths among ``paths``."""
    return sorted(p for p in paths if matches(rules, p))


def product(rules: Rules, paths: Iterable[str]) -> list[str]:
    """The product paths among ``paths``: neither judging paths, nor tests,
    nor docs, nor the orchestration scripts."""
    other = patterns('\n'.join((*NOT_PRODUCT, 'scripts/**', '.claude/**')))
    # The root's files (pyproject.toml, uv.lock, the Makefile) are the
    # repository's, not a product's (Claude r1 P3-13).
    return sorted(
        p
        for p in paths
        if '/' in p and not matches(rules, p) and not matches(other, p)
    )
