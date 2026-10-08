# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P86.cases: each branch Phase 8.5's verifier found ordinary public calls
reach is taken by a recorded case the core is held to (``CONTRACT.md``,
"Held cases").

A held case is a stored transcript, or a request fixture case that
``scripts/parity/request_fixture_exclusions.json`` does not exclude: Phase
8.5's checks, which P86.core runs again, hold the core to every one. Today's
Python runs them under branch coverage (``_held.py``), and every public
branch, and every branch the verifier named that the gap list no longer
lists, must be among those taken. No fixture case may be newly excluded.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from _reqgaps import (
    ENTRIES,
    PUBLIC,
    base,
    exclusions,
    held_measure,
    listed,
)


@cache
def _held() -> dict[str, Any]:
    return held_measure()


def _taken(entry: Any) -> bool:
    return entry.branch in _held()['executed'].get(entry.module, set())


def test_the_held_cases_ran() -> None:
    found = _held()
    assert len(found['transcripts']) >= 137, len(found['transcripts'])
    assert len(found['fixture_cases']) >= 553, len(found['fixture_cases'])
    assert not found['failed_transcripts'], found['failed_transcripts']


def test_each_public_branch_is_taken_by_a_held_case() -> None:
    untaken = [
        f'{entry.name} ({entry.what})'
        for entry in ENTRIES
        if entry.group == PUBLIC and not _taken(entry)
    ]
    assert not untaken, (
        f'{len(untaken)} branches ordinary public calls reach are taken by no '
        f'stored transcript and no request fixture case left unexcluded:\n'
        + '\n'.join(untaken)
    )


def test_each_named_branch_dropped_from_the_list_is_taken() -> None:
    gone = set(_named()) - set(listed())
    untaken = [
        f'{entry.name} ({entry.group}: {entry.what})'
        for entry in ENTRIES
        if (entry.module, entry.branch) in gone and not _taken(entry)
    ]
    assert not untaken, (
        'these branches left request_coverage_gaps.json, yet no held case '
        'takes them:\n' + '\n'.join(untaken)
    )


def test_no_fixture_case_is_newly_excluded() -> None:
    added = sorted(
        set(exclusions()) - set(base()['request_fixture_exclusions'])
    )
    assert not added, (
        f'request_fixture_exclusions.json excludes cases it did not at the '
        f'phase base, so the core is not held to them: {added}'
    )


def _named() -> dict[tuple[str, tuple[int, int]], Any]:
    return {(entry.module, entry.branch): entry for entry in ENTRIES}
