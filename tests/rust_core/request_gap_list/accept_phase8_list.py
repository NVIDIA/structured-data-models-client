# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P87.phase8-gaps: Phase 8's ``tests/parity/COVERAGE_GAPS.md`` no longer
says a group A branch is never taken (``CONTRACT.md``, "Phase 8's list").

A row of a group A branch that is still there holds none of the claims that
the branch cannot be taken, and names a host test under ``tests/rust_core``
that holds the route. Whether the corpus misses each listed branch is
P8.coverage's check, which P87.p86-regress runs.
"""

from __future__ import annotations

import re

from _gaplist import GROUP_A, PHASE8_GAPS, squeeze

BRANCH = re.compile(r'(-?\d+)\s*->\s*(-?\d+)')
#: Claims that a branch cannot be taken, which the verifier's probes showed
#: false for group A (case-insensitive).
CANNOT = (
    r'\bnever\s+(taken|returns|sends|settled|reached|asked)\b',
    r'\bunreachable\b',
    r'\bonly when settling raises\b',
    r'\bcallers always pass the pinned dtypes\b',
)
HOST_TEST = re.compile(r'tests/rust_core/[\w./-]+\.py')


def _rows() -> dict[tuple[str, tuple[int, int]], str]:
    rows: dict[tuple[str, tuple[int, int]], str] = {}
    for line in PHASE8_GAPS.read_text('utf-8').splitlines():
        cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
        if len(cells) < 3:
            continue
        module = cells[0].strip('`')
        found = BRANCH.fullmatch(cells[1].strip('`'))
        if not module.endswith('.py') or found is None:
            continue
        rows[(module, (int(found[1]), int(found[2])))] = '|'.join(cells[2:])
    return rows


def test_the_phase_8_list_is_read() -> None:
    assert len(_rows()) >= 100, len(_rows())


def test_no_group_a_row_says_its_branch_cannot_be_taken() -> None:
    rows = _rows()
    problems = []
    for entry in GROUP_A:
        reason = rows.get((entry.module, entry.branch))
        if reason is None:
            continue
        for pattern in CANNOT:
            if re.search(pattern, squeeze(reason)):
                problems.append(
                    f'{entry.name}: COVERAGE_GAPS.md says it cannot be taken '
                    f'({pattern!r}), yet a public call takes it '
                    f'({entry.what})'
                )
        if HOST_TEST.search(reason) is None:
            problems.append(
                f'{entry.name}: COVERAGE_GAPS.md names no host test under '
                f'tests/rust_core holding the route ({entry.what})'
            )
    assert not problems, '\n'.join(problems)
