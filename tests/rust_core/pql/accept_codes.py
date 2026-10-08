# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P5.codes: every diagnostic code in ``schema/codes.py`` except the
``PQL_REWRITE`` ones is exercised by an invalid corpus case, or is listed as
unreachable, with a reason, in ``rust/crates/pql/UNREACHABLE_CODES.md``.

A listed code must really be unexercised, and neither the corpus, the quirk
table nor the 50,000 generated queries may produce it today: if one does, it
is reachable. Every one of the codes, ``PQL_REWRITE`` included, is spelled
in the crate's source, since the crate carries all of them.

Whether each reason holds is for the reviewer (``CONTRACT.md``).
"""

from __future__ import annotations

import json
import re

from _generate import queries
from _pql import PQL_CRATE, corpus_cases, python_result, schemas
from accept_generated import COUNT, SEED
from quirks import TABLE

UNREACHABLE = PQL_CRATE / 'UNREACHABLE_CODES.md'
ENTRY = re.compile(r'^\s*[-*]\s+`(PQL_[A-Z0-9_]+)`\s*[:–-]\s*(\S.*)$', re.M)


def _all_codes() -> set[str]:
    from predictive_query.schema import codes

    return set(codes.ALL)


def _wanted() -> set[str]:
    return {c for c in _all_codes() if not c.startswith('PQL_REWRITE_')}


def _exercised() -> set[str]:
    return {
        error['code']
        for case in corpus_cases()
        for error in case.get('expected_errors', [])
    }


def _listed() -> dict[str, str]:
    if not UNREACHABLE.exists():
        return {}
    found = ENTRY.findall(UNREACHABLE.read_text(encoding='utf-8'))
    codes = [code for code, _ in found]
    assert len(codes) == len(set(codes)), f'a code is listed twice: {codes}'
    return dict(found)


def test_there_are_96_codes():
    assert len(_all_codes()) == 96


def test_every_code_is_exercised_or_listed_unreachable():
    missing = sorted(_wanted() - _exercised() - set(_listed()))
    assert not missing, (
        f'{len(missing)} codes neither exercised by an invalid corpus case '
        f'nor listed in {UNREACHABLE.name}: {missing}'
    )


def test_each_listed_code_is_real_unexercised_and_has_a_reason():
    listed = _listed()
    for code, reason in listed.items():
        assert code in _wanted(), f'{code} is not a non-REWRITE code'
        assert code not in _exercised(), (
            f'{code} is listed unreachable but a corpus case exercises it'
        )
        assert len(reason.split()) >= 5, f'{code}: no real reason: {reason!r}'


def test_no_listed_code_is_produced_today():
    listed = set(_listed())
    if not listed:
        return
    produced: dict[str, str] = {}

    def note(result: dict, query: str) -> None:
        for error in result.get('errors', []):
            produced.setdefault(error['code'], query)

    table = json.loads(TABLE.read_text(encoding='utf-8'))
    for row in table['queries']:
        note(row['python'], row['query'])
    for case in queries(SEED, COUNT, schemas()):
        args = (case['query'], case['schema'], case['profile'], case['mode'])
        note(python_result(*args), case['query'])
    reached = {code: produced[code] for code in listed if code in produced}
    assert not reached, f'listed unreachable, yet produced today by: {reached}'


def test_the_crate_spells_every_code():
    source = '\n'.join(
        path.read_text(encoding='utf-8')
        for path in sorted((PQL_CRATE / 'src').rglob('*.rs'))
    )
    absent = sorted(c for c in _all_codes() if f'"{c}"' not in source)
    assert not absent, f'codes not in rust/crates/pql/src: {absent}'
