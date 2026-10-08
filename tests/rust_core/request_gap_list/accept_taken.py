# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P87.taken: every group A branch, one a public call reaches and nothing
held at the base, is taken by a recorded case the core is held to, or by a
host test the gap list cites (``CONTRACT.md``, "Group A").

- A group A entry gone from ``scripts/parity/request_coverage_gaps.json`` is
  taken by a held case: a stored transcript, or a request fixture case not
  excluded. Phase 8.5's checks, which P87.p86-core runs again, hold the core
  to every held case.
- A group A entry still listed says "Reachable only", holds none of the
  clauses the verifier showed false, and cites by node id at least one host
  test; each it cites passes, calls into ``sdm_core._native``, and takes the
  branch in today's Python.
- ``request_fixture_exclusions.json`` excludes no case it did not at the
  base.
"""

from __future__ import annotations

from _gaplist import (
    GROUP_A,
    base,
    cited,
    corpus_measure,
    exclusions,
    false_clauses,
    host_measure,
    host_problems,
    listed,
    listed_at_base,
    says_reachable_only,
)

#: What the verifier ran at the base: 137 transcripts and 572 held fixture
#: cases.
BASE_TRANSCRIPTS = 137
BASE_HELD_CASES = 572


def test_every_group_a_entry_was_listed_at_the_base() -> None:
    at_base = listed_at_base()
    missing = [e.name for e in GROUP_A if (e.module, e.branch) not in at_base]
    assert len(GROUP_A) == 30, len(GROUP_A)
    assert not missing, missing


def test_the_recorded_cases_ran() -> None:
    found = corpus_measure()
    assert len(found['transcripts']) >= BASE_TRANSCRIPTS, found['transcripts']
    assert len(found['held_cases']) >= BASE_HELD_CASES, len(found['held_cases'])
    assert not found['failed_transcripts'], found['failed_transcripts']


def test_no_fixture_case_is_newly_excluded() -> None:
    added = sorted(
        set(exclusions()) - set(base()['request_fixture_exclusions'])
    )
    assert not added, (
        f'request_fixture_exclusions.json excludes cases it did not at the '
        f'phase base, so the core is not held to them: {added}'
    )


def test_each_group_a_branch_dropped_from_the_list_is_held() -> None:
    now = listed()
    gone = [e for e in GROUP_A if (e.module, e.branch) not in now]
    if not gone:
        return
    held = corpus_measure()['held']
    untaken = [
        f'{e.name} ({e.what})'
        for e in gone
        if e.branch not in held.get(e.module, set())
    ]
    assert not untaken, (
        'these group A branches left request_coverage_gaps.json, yet no '
        'stored transcript and no request fixture case left unexcluded takes '
        'them:\n' + '\n'.join(untaken)
    )


def test_each_group_a_branch_still_listed_is_held_by_a_host_test() -> None:
    now = listed()
    still = [
        (e, now[(e.module, e.branch)])
        for e in GROUP_A
        if (e.module, e.branch) in now
    ]
    problems = []
    for entry, reason in still:
        if not says_reachable_only(reason):
            problems.append(
                f'{entry.name}: a public call reaches it ({entry.what}), so '
                f"its reason must say 'Reachable only' and cite the host test "
                f'that holds the route'
            )
        for clause in false_clauses(entry, reason):
            problems.append(
                f'{entry.name}: the reason still says {clause!r}, which the '
                f'verifier showed false'
            )
        if not cited(reason):
            problems.append(
                f'{entry.name}: cites no host test by pytest node id '
                f'(tests/rust_core/<path>.py::<test>[<param id>])'
            )
    if any(cited(reason) for _, reason in still):
        hosts = host_measure()
        for entry, reason in still:
            for node in cited(reason):
                problems += host_problems(
                    entry.name, entry.branch, entry.module, node, hosts
                )
    assert not problems, '\n'.join(problems)
