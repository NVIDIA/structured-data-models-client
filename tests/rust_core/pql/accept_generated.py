# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P5.generated: 50,000 queries generated from the grammar, valid and
invalid, against the corpus schemas, under every profile and both modes,
give the same answer from today's Python and from ``pql_validate``.

Valid ones: equal trees, by value and JSON type, and equal warnings. Invalid
ones: equal codes and spans, and equal messages except for
``PQL_PARSE_SYNTAX``, whose new text must not carry ANTLR's or pydantic's
wording; equal warnings too, except where the query is refused for its
syntax. The generator is ``_generate.queries``, seeded with ``SEED``.
"""

from __future__ import annotations

import collections

from _generate import queries
from _pql import PARSE_SYNTAX, mismatch, python_result, rust_result, schemas

SEED = 20261002
COUNT = 50_000


def _kind(result: dict) -> str:
    if 'tree' in result:
        return 'valid'
    if any(e['code'] == PARSE_SYNTAX for e in result['errors']):
        return 'syntax'
    return 'invalid'


def test_generated_queries_match_todays_python():
    kinds: collections.Counter[str] = collections.Counter()
    codes: collections.Counter[str] = collections.Counter()
    failures, raised, compared = [], [], 0
    # Generated until COUNT have been compared; a query today's code raises
    # on instead of answering does not count.
    for case in queries(SEED, COUNT + 10_000, schemas()):
        if compared == COUNT:
            break
        args = (case['query'], case['schema'], case['profile'], case['mode'])
        want = python_result(*args)
        try:
            got = rust_result(*args)
        except Exception as error:  # noqa: BLE001 - reported per case
            failures.append(f'{case["query"]!r} {args[2:]}: raised {error!r}')
            continue
        if 'exception' in want:
            # Today's code raised rather than answered: nothing to compare,
            # but the core must still answer.
            raised.append(case['query'])
            continue
        compared += 1
        kind = _kind(want)
        kinds[kind] += 1
        codes.update(e['code'] for e in want.get('errors', []))
        if kind == 'syntax' and 'errors' in got:
            got = dict(got, warnings=want['warnings'])
        problem = mismatch(want, got)
        if problem:
            failures.append(f'{case["query"]!r} {args[2:]}: {problem}')
    summary = (
        f'seed {SEED}: {compared} compared ({dict(kinds)}), '
        f'{len(raised)} raised today, {len(codes)} distinct codes'
    )
    print(summary)
    assert not failures, (
        f"{summary}\n{len(failures)} differ from today's Python:\n"
        + '\n'.join(failures[:60])
    )
    assert compared == COUNT, summary
    # The mix this seed gives today; a generator that drifted to all one
    # kind would compare far less than it claims.
    assert kinds['valid'] >= 8_000, summary
    assert kinds['invalid'] >= 15_000, summary
    assert kinds['syntax'] >= 8_000, summary
    assert len(codes) >= 55, summary
