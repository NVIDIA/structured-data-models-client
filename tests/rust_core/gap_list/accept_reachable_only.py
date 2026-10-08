# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P78.reachable-only: every entry of the sampling gap list whose reason
says "reachable only" cites tests that exist, pass and, at least one of
them, take its branch; no other entry's branch is taken by a transcript, a
fixture case or a test (``CONTRACT.md``, "Reachable only").

The host suites run once, under branch coverage with a context per test
(``_gap_list.host_suites``), as the Phase 7.7 verifier measured them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import coverage  # noqa: F401 - the real one, before any path change

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _gap_list import (  # noqa: E402
    NAIVE_UTC,
    REPO,
    Measured,
    check_lines,
    citations,
    corpus_missing,
    entries,
    host_suites,
    measure_tests,
    says_reachable_only,
)

CONTROL = HERE / 'control_gap_list.py'
CONTROL_TAKES = f'{CONTROL.relative_to(REPO)}::test_takes_the_naive_utc_branch_and_calls_the_core'
CONTROL_IDLE = (
    f'{CONTROL.relative_to(REPO)}::test_takes_nothing_and_calls_nothing'
)

#: Fewer tests than this in the measured run means it did not run the
#: suites (the gate's P77.suite-rest ran 4405 at the base).
MIN_TESTS = 4000


def test_the_measure_credits_a_known_test() -> None:
    r"""The measure, on two known tests: one takes ``base/utils.py`` 65->66
    and calls the binding, the other neither."""
    check_lines({'naive-utc': NAIVE_UTC})
    measured = measure_tests(
        [str(CONTROL)], watch_core=[CONTROL_TAKES, CONTROL_IDLE]
    )
    assert measured.outcomes == {
        CONTROL_TAKES: 'passed',
        CONTROL_IDLE: 'passed',
    }, measured.tail
    assert NAIVE_UTC.pair in measured.taken_by(CONTROL_TAKES).get(
        NAIVE_UTC.module, set()
    )
    assert not any(measured.taken_by(CONTROL_IDLE).values())
    assert measured.core_calls == {CONTROL_TAKES}


def test_every_listed_entry_is_untaken_by_the_corpus() -> None:
    r"""No transcript, fixture case or PQL case takes a listed branch
    (``tests/parity/coverage.json``, which P77.gaps holds to what
    ``scripts/parity/coverage.py`` writes)."""
    missing = corpus_missing()
    taken = [
        str(e) for e in entries() if e.pair not in missing.get(e.module, set())
    ]
    assert not taken, 'listed, but taken by the corpus:\n' + '\n'.join(taken)


def test_every_reachable_only_entry_cites_a_test() -> None:
    bare = [
        str(e)
        for e in entries()
        if says_reachable_only(e.reason) and not citations(e.reason)
    ]
    assert not bare, (
        'says "reachable only" but cites no test by its node id, in '
        'backticks:\n' + '\n'.join(bare)
    )


def _measured_suites() -> Measured:
    measured = host_suites()
    assert len(measured.outcomes) >= MIN_TESTS, (
        f'the measured run ran {len(measured.outcomes)} tests:\n{measured.tail}'
    )
    taken = measured.taken()
    unmeasured = sorted(
        {e.module for e in entries() if not taken.get(e.module)}
    )
    assert not unmeasured, f'the measured run took no branch of {unmeasured}'
    return measured


def test_every_cited_test_passes_and_one_takes_the_branch() -> None:
    measured = _measured_suites()
    problems = []
    for entry in entries():
        if not says_reachable_only(entry.reason):
            continue
        cited = citations(entry.reason)
        for test in cited:
            outcome = measured.outcomes.get(test)
            if outcome != 'passed':
                problems.append(
                    f'{entry}: {test} '
                    + ('was not run' if outcome is None else outcome)
                )
        taking = [
            t
            for t in cited
            if entry.pair in measured.taken_by(t).get(entry.module, set())
        ]
        if not taking:
            problems.append(f'{entry}: none of {cited} takes the branch')
    assert not problems, '\n'.join(problems)


def test_no_other_entry_is_taken_by_a_test() -> None:
    measured = _measured_suites()
    taken = measured.taken()
    found = [
        str(e)
        for e in entries()
        if not says_reachable_only(e.reason)
        and e.pair in taken.get(e.module, set())
    ]
    assert not found, (
        'taken by the host suites, but the reason does not say "reachable '
        'only" and cite the tests:\n' + '\n'.join(found)
    )
