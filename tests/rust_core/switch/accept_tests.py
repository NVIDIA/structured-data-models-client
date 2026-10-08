# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P9.tests: no fewer tests than BASE, none quietly skipped, no fewer
assertions, and every changed test file listed.

``base.json`` holds BASE's facts, taken when these checks were locked: the
tests the suite-rest command collects, those that passed and were skipped
in the phase venv, and the assertions in its test code. The implementer's
shadow tests (``tests/rust_core/shadow/``) count for none of these.
``changed_tests.tsv`` lists every test-code file changed since BASE, each
with what changed and where any moved assertion went.
"""

from __future__ import annotations

import json

from _switch import (
    BASE_FACTS,
    CHANGED_TESTS,
    assertion_count,
    changed_test_files,
    collected,
    suite_report,
    suite_test_files,
)


def _base():
    return json.loads(BASE_FACTS.read_text())


def test_no_fewer_tests_are_collected():
    base = _base()
    ids = collected()
    gone = sorted(set(base['collected_ids']) - set(ids))
    assert len(ids) >= base['collected'], (
        f'{len(ids)} tests collected, BASE collected {base["collected"]}; '
        f'gone since BASE (first 40): {gone[:40]}'
    )


def test_no_fewer_pass_and_no_more_are_skipped():
    base = _base()
    outcomes = suite_report()['outcomes']
    assert outcomes['passed'] >= base['passed'], (outcomes, base['passed'])
    assert outcomes['skipped'] <= base['skipped'], (outcomes, base['skipped'])


def test_no_fewer_assertions():
    base = _base()
    count = assertion_count(suite_test_files())
    assert count >= base['assertions'], (count, base['assertions'])


def test_every_changed_test_file_is_listed():
    changed = set(changed_test_files())
    listed = {}
    problems = []
    if CHANGED_TESTS.is_file():
        for n, line in enumerate(CHANGED_TESTS.read_text().splitlines(), 1):
            if not line.strip() or line.startswith('#'):
                continue
            path, _, why = line.partition('\t')
            if path in listed:
                problems.append(f'line {n}: {path} listed twice')
            listed[path] = why
            if len(why.split()) < 5:
                problems.append(
                    f'line {n}: {path}: say what changed, in five words or more'
                )
    for path in sorted(changed - set(listed)):
        problems.append(f'{path} changed since BASE and is not listed')
    for path in sorted(set(listed) - changed):
        problems.append(f'{path} is listed and did not change since BASE')
    assert not problems, '\n'.join(problems)
