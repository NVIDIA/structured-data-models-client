# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P5.valid and P5.invalid, through the binding: every PQL corpus case runs
through today's Python and through ``sdm_core._native.pql_validate``, and
both must give what the corpus recorded, under ``test_pql_corpus.py``'s
comparison, with new text allowed only for ``PQL_PARSE_SYNTAX`` messages.

Also: the corpus keeps every case it held when Phase 5 started, and the
binding takes ``mode``, and refuses input it cannot read with an exception,
never a panic.
"""

from __future__ import annotations

import json

import pytest
from _pql import (
    base_cases,
    corpus_cases,
    expected,
    mismatch,
    native,
    python_result,
    rust_result,
    schemas,
)


def _check(valid: bool) -> None:
    cases = [c for c in corpus_cases() if ('expected_tree' in c) == valid]
    floor = len(base_cases('valid.json' if valid else 'invalid.json'))
    assert len(cases) >= floor, f'{len(cases)} cases, fewer than {floor}'
    failures = []
    for case in cases:
        profile = case.get('profile', 'RELATIONAL')
        want = expected(case)
        today = python_result(case['query'], case['schema'], profile)
        # Today's Python must still give exactly the recorded answer; only
        # the Rust side gets new PQL_PARSE_SYNTAX text.
        if today != want:
            failures.append(f"{case['id']}: today's Python: {today}")
            continue
        try:
            got = rust_result(case['query'], case['schema'], profile)
        except Exception as error:  # noqa: BLE001 - reported per case
            failures.append(f'{case["id"]}: raised {error!r}')
            continue
        problem = mismatch(want, got)
        if problem:
            failures.append(f'{case["id"]}: {problem}')
    assert not failures, (
        f'{len(failures)} of {len(cases)} cases differ:\n'
        + '\n'.join(failures[:60])
    )


def test_valid_cases_match_the_corpus():
    _check(valid=True)


def test_invalid_cases_match_the_corpus():
    _check(valid=False)


def test_the_corpus_keeps_every_case_it_started_with():
    now = {case['id']: case for case in corpus_cases()}
    for name in ('valid.json', 'invalid.json'):
        for case in base_cases(name):
            assert case['id'] in now, f'{name}: {case["id"]} was removed'
            assert now[case['id']] == case, f'{name}: {case["id"]} changed'


def test_mode_replaces_the_profiles_mode():
    shop = schemas()[0]
    queries = [
        'PREDICT COUNT(orders.*, 0, 30, days) FOR users.user_id = 1',
        'PREDICT COUNT(orders.*, 0, 30, days) FOR users.user_id IN (1, 2)',
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
        'PREDICT users.age FOR users.user_id = 1',
    ]
    differs = False
    for query in queries:
        for profile in ('RELATIONAL', 'FINE_TUNING', 'TABULAR'):
            answers = {}
            for mode in (None, 'TRAINING', 'INFERENCE'):
                want = python_result(query, shop, profile, mode)
                got = rust_result(query, shop, profile, mode)
                problem = mismatch(want, got)
                assert problem is None, f'{query} {profile} {mode}: {problem}'
                answers[mode] = json.dumps(want, sort_keys=True)
            differs = differs or answers['TRAINING'] != answers['INFERENCE']
    assert differs, 'no query here answers differently in the two modes'
    # Omitted, mode is the profile's own.
    one = native().pql_validate(queries[0], json.dumps(shop), 'FINE_TUNING')
    assert json.loads(one) == rust_result(
        queries[0], shop, 'FINE_TUNING', 'TRAINING'
    )


@pytest.mark.parametrize(
    ('args', 'error'),
    [
        (
            ('PREDICT x.y FOR EACH x.z', '{not json', 'RELATIONAL', None),
            ValueError,
        ),
        (('PREDICT x.y FOR EACH x.z', '[]', 'RELATIONAL', None), ValueError),
        (
            ('PREDICT x.y FOR EACH x.z', '{"tables": 3}', 'RELATIONAL', None),
            ValueError,
        ),
        (
            ('PREDICT x.y FOR EACH x.z', 'SHOP', 'NOT_A_PROFILE', None),
            ValueError,
        ),
        (('PREDICT x.y FOR EACH x.z', 'SHOP', 'relational', None), ValueError),
        (
            ('PREDICT x.y FOR EACH x.z', 'SHOP', 'RELATIONAL', 'TESTING'),
            ValueError,
        ),
        ((None, 'SHOP', 'RELATIONAL', None), TypeError),
        ((3, 'SHOP', 'RELATIONAL', None), TypeError),
    ],
)
def test_the_binding_refuses_what_it_cannot_read(args, error):
    query, schema, profile, mode = args
    if schema == 'SHOP':
        schema = json.dumps(schemas()[0])
    with pytest.raises(error):
        native().pql_validate(query, schema, profile, mode)


@pytest.mark.parametrize(
    'query',
    [
        '',
        '\x00',
        '(' * 5000,
        'PREDICT ' + 'NOT ' * 5000 + 'users.age > 1',
        '\ud800',
    ],
)
def test_the_binding_answers_odd_text_without_a_panic(query):
    shop = json.dumps(schemas()[0])
    try:
        text = native().pql_validate(query, shop, 'RELATIONAL', None)
    except (ValueError, UnicodeError):
        # A lone surrogate is not text Rust can hold; refusing it is fine.
        assert query == '\ud800'
        return
    got = json.loads(text)
    assert 'errors' in got and got['errors'], got


def test_deep_nesting_today_takes_is_taken():
    shop = schemas()[0]
    query = (
        'PREDICT '
        + '(' * 300
        + 'users.age > 1'
        + ')' * 300
        + ' FOR EACH users.user_id'
    )
    want = python_result(query, shop, 'RELATIONAL')
    assert 'tree' in want, want
    assert mismatch(want, rust_result(query, shop, 'RELATIONAL')) is None
