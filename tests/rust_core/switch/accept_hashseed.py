# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P9.hashseed: on the core, what every transcript and public fixture case
does and gives is the same under ``PYTHONHASHSEED`` 1 and 4242, and every
transcript replays as stored under both."""

from __future__ import annotations

import pytest
from _switch import CORE, capture, core_in_use, first_difference


@pytest.mark.parametrize(
    ('what', 'key'), [('transcripts', 'name'), ('fixtures', 'id')]
)
def test_the_same_under_two_hash_seeds(what, key):
    one = capture(CORE, what, '1')
    other = capture(CORE, what, '4242')
    problems = [f'not on the core: {p}' for p in core_in_use(one)]
    a = {r[key]: r for r in one['records']}
    b = {r[key]: r for r in other['records']}
    assert a.keys() == b.keys()
    for name in sorted(a):
        found = first_difference(a[name], b[name])
        if found:
            problems.append(f'{name}: {found}')
        if what == 'transcripts':
            for seed, rec in (('1', a[name]), ('4242', b[name])):
                if rec['check'] is not None:
                    problems.append(f'{name} under {seed}: {rec["check"]}')
    assert not problems, '\n'.join(problems[:40])
