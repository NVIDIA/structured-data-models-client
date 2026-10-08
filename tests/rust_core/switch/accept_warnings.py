# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Brief task 1: on the core, each warning lands on the caller's line.

Every warning a transcript or a public fixture case gives on the core path
is attributed (``warnings.WarningMessage.filename``) to the code that made
the public call: the replay harness or the fixture runner, never a file of a
package, of the binding or of the old implementation. That is what
``caller_stacklevel()`` gives, and it keeps Python's once-per-location
filter working per call site. The warnings' categories, texts and order are
P9.shadow's.
"""

from __future__ import annotations

from _switch import CORE, REPO, capture, core_in_use

CALLERS = (str(REPO / 'tests') + '/', str(REPO / 'scripts') + '/')


def _misplaced(what, key):
    data = capture(CORE, what)
    problems = [f'not on the core: {p}' for p in core_in_use(data)]
    count = 0
    for rec in data['records']:
        for w in rec.get('warnings') or []:
            count += 1
            if not w['filename'].startswith(CALLERS):
                problems.append(
                    f'{rec[key]}: {w["category"]} "{w["message"][:80]}" '
                    f'lands in {w["filename"]}'
                )
    return problems, count


def test_transcript_warnings_land_on_the_callers_line():
    problems, count = _misplaced('transcripts', 'name')
    assert not problems, '\n'.join(problems[:40])
    assert count >= 100, f'only {count} warnings'


def test_fixture_warnings_land_on_the_callers_line():
    problems, count = _misplaced('fixtures', 'id')
    assert not problems, '\n'.join(problems[:40])
    assert count >= 50, f'only {count} warnings'
