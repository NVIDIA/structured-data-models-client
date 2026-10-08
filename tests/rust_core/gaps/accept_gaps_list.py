# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P77.gaps: the sampling coverage gap list no longer lists the three
branches the scenarios take, and the coverage report shows each taken
(``CONTRACT.md``, "Gaps").

Phase 7.6's ``accept_link_derived.py``, run beside this file, holds the
report to what ``coverage.py`` writes; Phase 7.5's
``accept_driver_coverage.py`` holds every listed entry to a ported branch
left untaken.
"""

from __future__ import annotations

import json

from _gaps import (
    BRANCHES,
    COVERAGE_JSON,
    GAPS,
    check_lines,
    coverage_modules,
    gap_entries,
)


def test_the_branches_are_where_the_checks_say() -> None:
    check_lines()


def test_none_of_the_three_is_listed() -> None:
    check_lines()
    entries = gap_entries()
    listed = [
        f'{branch} ({name})'
        for name, branch in BRANCHES.items()
        if branch.pair in entries.get(branch.module, set())
    ]
    assert not listed, (
        f'{GAPS.name} still lists branches a scenario takes:\n'
        + '\n'.join(listed)
    )


def test_the_report_shows_them_taken() -> None:
    check_lines()
    modules = coverage_modules(json.loads(COVERAGE_JSON.read_text('utf-8')))
    problems = []
    for name, branch in BRANCHES.items():
        body = modules.get(branch.module)
        if body is None:
            problems.append(f'{branch.module} is not in the report')
            continue
        missing = {tuple(b) for b in body.get('missing_branches', [])}
        if branch.pair in missing:
            problems.append(f'{branch} ({name}) is untaken in the report')
    assert not problems, '\n'.join(problems)
