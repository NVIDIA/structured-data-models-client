# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The judging paths and the tooling train's rule (T3.0, item 1).

``judging-paths.txt`` lists the paths that decide whether a candidate
passes, in ``contract/protected-paths.txt``'s pattern syntax. ``land.py``
reads it from the phase head (:func:`rules_at`), so a package cannot take
its own change out of the list.

A package that changes a judging path lands either as the phase's first
package (T3.0 in Phase 3) or as a train: a package of
judging-path changes only, with their tests and docs (:data:`TRAIN_ALSO`),
at the front of a stack (``land.py --stack --train``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from typing import Any

#: The list, as a path in the tree.
FILE = 'scripts/rust_core/judging-paths.txt'
#: What a train may change besides judging paths: their tests, the docs,
#: and the other checking tools.
TRAIN_ALSO = (
    'tests/**',
    'contract/tests/**',
    'docs/**',
    # The checking tools the list leaves out because product packages also
    # edit their data (bridge_freeze.py, wheel_smoke.py, the recorders): a
    # fix to their checking logic may ride the train (Codex r2).
    'scripts/rust_core/**',
    'contract/runners/**',
    'contract/tools/**',
    # The test kit's crate tests of the case loader and observation writer
    # (T3.2), which judge the judging paths case.rs and observation.rs.
    'rust/crates/sdm-testkit/tests/**',
    # The CI workflow, whose triggers the owner moved off the landing path
    # ("2026-10-22 evening: CI off the landing path"; T3.2 fix 4): it
    # decides no pass or fail of a landing any more.
    '.github/workflows/ci.yml',
)
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


#: The checkers left off the judging list because product packages edit
#: their data in the same file: their checking logic is train-worthy on its
#: own (Codex r3), though a change to them does not have to ride a train.
CHECKERS = (
    'scripts/rust_core/bridge_freeze.py',
    'scripts/rust_core/wheel_smoke.py',
    'scripts/rust_core/selftest.py',
)


def train_worthy(rules: Rules, paths: Iterable[str]) -> list[str]:
    """The paths among ``paths`` a train exists for: judging paths and the
    checkers."""
    checkers = patterns('\n'.join(CHECKERS))
    return sorted(p for p in paths if matches(rules, p) or matches(checkers, p))


#: The old client's recorded answers, which a train may change only by
#: adding fields to lines already there (the owner, 2026-10-22: option 1 of
#: the coordinator's question on T3.2's three text digests).
REFERENCE = 'contract/reference/**'


def not_train(
    rules: Rules,
    paths: Iterable[str],
    read: Callable[[str], tuple[str, str]] | None = None,
) -> list[str]:
    """The paths among ``paths`` a train may not change. A reference file
    rides only when ``read`` gives its text before and after and
    :func:`additions_only` finds nothing else changed."""
    also = patterns('\n'.join(TRAIN_ALSO))
    reference = patterns(REFERENCE)
    found = []
    for p in paths:
        if matches(rules, p) or matches(also, p):
            continue
        if read is not None and matches(reference, p):
            if additions_only(*read(p)) is None:
                continue
        found.append(p)
    return sorted(found)


def additions_only(old: str, new: str) -> str | None:
    """None when ``new`` is ``old`` with fields added to its JSON lines
    and nothing else: no line added, removed or reordered, no field
    removed or changed; else the first change that is more."""
    before = [line for line in old.splitlines() if line.strip()]
    after = [line for line in new.splitlines() if line.strip()]
    if len(before) != len(after):
        return f'{len(after) - len(before):+d} line(s): a line added or removed'
    for n, (a, b) in enumerate(zip(before, after, strict=True), 1):
        try:
            mine, theirs = json.loads(b), json.loads(a)
        except ValueError:
            return f'line {n}: not JSON'
        if not isinstance(mine, dict) or not isinstance(theirs, dict):
            return f'line {n}: not a JSON object'
        problem = _added_only(theirs, mine, f'line {n}')
        if problem is not None:
            return problem
    return None


def _added_only(old: Any, new: Any, where: str) -> str | None:
    if isinstance(old, dict) and isinstance(new, dict):
        for key, value in old.items():
            if key not in new:
                return f'{where}: {key} removed'
            problem = _added_only(value, new[key], f'{where}: {key}')
            if problem is not None:
                return problem
        return None
    if old != new or type(old) is not type(new):
        return f'{where}: changed'
    return None


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
