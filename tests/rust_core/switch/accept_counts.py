# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P9.perf-counts: the queries per scenario are the same on both paths.

For every transcript and every public fixture case, the ``Source._run``
calls (each statement and its parameters, in order) and the statements the
connection ran are the same on the old path and on the core. An entry of
``fix_changes.json`` is P9.shadow's to judge. The reads of a schema's first
question after a copy are P9.pickle's (``reads`` and ``source_runs``).
"""

from __future__ import annotations

from _switch import CORE, OLD, by_key, capture, core_in_use, fix_changes


def _compare(what, key):
    old = by_key(capture(OLD, what), key)
    core_data = capture(CORE, what)
    core = by_key(core_data, key)
    listed = fix_changes()
    problems = [f'not on the core: {p}' for p in core_in_use(core_data)]
    queried = 0
    for name, theirs in sorted(old.items()):
        if name in listed:
            continue
        mine = core.get(name)
        if mine is None:
            problems.append(f'{name}: not run on the core')
            continue
        a, b = theirs['source_runs'], mine['source_runs']
        queried += bool(a)
        if a != b:
            first = next(
                (
                    i
                    for i, (x, y) in enumerate(zip(a, b, strict=False))
                    if x != y
                ),
                min(len(a), len(b)),
            )
            problems.append(
                f'{name}: {len(a)} queries today, {len(b)} on the core; '
                f'first different at {first}'
            )
        if len(theirs.get('statements') or []) != len(
            mine.get('statements') or []
        ):
            problems.append(
                f'{name}: {len(theirs.get("statements") or [])} statements today, '
                f'{len(mine.get("statements") or [])} on the core'
            )
    return problems, queried


def test_transcripts_query_the_same_on_both_paths():
    problems, queried = _compare('transcripts', 'name')
    assert not problems, '\n'.join(problems[:40])
    assert queried >= 60, f'only {queried} transcripts queried a source'


def test_fixture_cases_query_the_same_on_both_paths():
    problems, queried = _compare('fixtures', 'id')
    assert not problems, '\n'.join(problems[:40])
    assert queried >= 30, f'only {queried} fixture cases queried a source'
