# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P87.reasons: every group B entry, a branch still unreachable whose reason
at the base was false or named the wrong guard, has its reason rewritten
(``CONTRACT.md``, "Group B").

A group B entry still listed has a reason other than its base one (runs of
whitespace as one space), holding none of the clauses the verifier showed
false and naming the guard the verifier found. A group B entry gone from the
list is taken by a held case.
"""

from __future__ import annotations

from _gaplist import (
    GROUP_B,
    corpus_measure,
    false_clauses,
    listed,
    listed_at_base,
    squeeze,
    unnamed,
)


def test_every_group_b_entry_was_listed_at_the_base() -> None:
    at_base = listed_at_base()
    missing = [e.name for e in GROUP_B if (e.module, e.branch) not in at_base]
    assert len(GROUP_B) == 10, len(GROUP_B)
    assert not missing, missing


def test_each_group_b_reason_is_rewritten() -> None:
    now, at_base = listed(), listed_at_base()
    problems = []
    gone = []
    for entry in GROUP_B:
        key = (entry.module, entry.branch)
        if key not in now:
            gone.append(entry)
            continue
        reason = now[key]
        if squeeze(reason) == squeeze(at_base[key]):
            problems.append(f'{entry.name}: the reason is the base reason')
        for clause in false_clauses(entry, reason):
            problems.append(
                f'{entry.name}: the reason still says {clause!r}, which the '
                f'verifier showed false'
            )
        for pattern in unnamed(entry, reason):
            problems.append(
                f'{entry.name}: the reason does not name the guard the '
                f'verifier found ({entry.what}): nothing matches {pattern!r}'
            )
    if gone:
        held = corpus_measure()['held']
        problems += [
            f'{entry.name}: left the list, yet no held case takes it'
            for entry in gone
            if entry.branch not in held.get(entry.module, set())
        ]
    assert not problems, '\n'.join(problems)
