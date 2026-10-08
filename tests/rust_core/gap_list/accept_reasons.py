# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P78.reasons: the false clause of ``base/utils.py`` 65->66 is gone, as
the Phase 7.7 verifier's file quotes it, and none of the false clauses the
verifier found is left in any reason (``CONTRACT.md``, "Reasons")."""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _gap_list import (  # noqa: E402
    FALSE_CLAUSES,
    NAIVE_UTC,
    VERIFIER,
    check_lines,
    corpus_missing,
    entries,
    entry_of,
)


def _quoted_clause() -> str:
    r"""What the verifier's section C says the reason says."""
    assert VERIFIER.is_file(), f'{VERIFIER} is missing'
    text = VERIFIER.read_text('utf-8')
    found = re.search(
        r'^### C\..*?`base/utils\.py` 65->66.*?The reason says "([^"]+)"',
        text,
        re.S | re.M,
    )
    assert found, f'{VERIFIER} quotes no reason for base/utils.py 65->66'
    return found[1]


def test_the_clause_is_the_one_the_verifier_quotes() -> None:
    quoted = _quoted_clause()
    (pinned,) = FALSE_CLAUSES['naive-utc']
    assert quoted.startswith(pinned), (quoted, pinned)


def test_the_naive_utc_reason_drops_the_false_clause() -> None:
    check_lines({'naive-utc': NAIVE_UTC})
    entry = entry_of(NAIVE_UTC)
    assert entry is not None, (
        f'{NAIVE_UTC} is no longer listed; it is still untaken'
        if NAIVE_UTC.pair in corpus_missing().get(NAIVE_UTC.module, set())
        else f'{NAIVE_UTC} is no longer listed'
    )
    reason = entry.reason.lower()
    quoted = _quoted_clause()
    for clause in {quoted, quoted.split(' (')[0], *FALSE_CLAUSES['naive-utc']}:
        assert clause.lower() not in reason, (
            f'{NAIVE_UTC} still says {clause!r}'
        )
    assert 'tasktable' in reason or 'inference/task.py' in reason, (
        f'{NAIVE_UTC}: the reason does not name the TaskTable time column '
        'setter (inference/task.py), whose Series reach to_naive_utc too'
    )


def test_no_reason_keeps_a_false_clause() -> None:
    clauses = {c.lower() for found in FALSE_CLAUSES.values() for c in found}
    kept = [
        f'{e}: {c!r}'
        for e in entries()
        for c in sorted(clauses)
        if c in e.reason.lower()
    ]
    assert not kept, (
        'reasons keeping a clause the verifier showed false:\n'
        + '\n'.join(kept)
    )
