# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P8.nan-order (task 6): a fixture records today's order for a ranking in
which a candidate's score is NaN.

Some fixture case's ``expected`` output holds a ranking: a list of at least
three ``{"candidate", "score", ...}`` objects in which one score is NaN
(``{"$float": "nan"}``, or a JSON ``NaN``) and at least two are numbers. The
case passes against today's Python, so the order it records is today's. The
port decides later whether to keep that order or refuse NaN; nothing here
says which.
"""

from __future__ import annotations

import math
from typing import Any

from _net import check_import_location, fixture_cases, fixture_ops, run_case


def _nan(score: Any) -> bool:
    return (isinstance(score, float) and math.isnan(score)) or score == {
        '$float': 'nan'
    }


def _number(score: Any) -> bool:
    return (
        isinstance(score, (int, float))
        and not isinstance(score, bool)
        and not math.isnan(score)
    )


def _rankings(value: Any) -> list[list[dict[str, Any]]]:
    out = []
    if isinstance(value, list):
        if len(value) >= 3 and all(
            isinstance(v, dict) and {'candidate', 'score'} <= v.keys()
            for v in value
        ):
            out.append(value)
        for item in value:
            out += _rankings(item)
    elif isinstance(value, dict):
        for item in value.values():
            out += _rankings(item)
    return out


def _records_a_nan_ranking(case: dict[str, Any]) -> bool:
    for ranking in _rankings(case.get('expected')):
        scores = [entry['score'] for entry in ranking]
        if any(_nan(s) for s in scores) and sum(map(_number, scores)) >= 2:
            return True
    return False


def test_a_fixture_records_the_order_of_a_nan_score() -> None:
    check_import_location()
    found = [c for c in fixture_cases() if _records_a_nan_ranking(c)]
    assert found, (
        'no fixture case records a ranking of three or more candidates in '
        'which one score is NaN'
    )
    ops = fixture_ops()
    outcomes = [run_case(ops, case) for case in found]
    passing = [o for o in outcomes if not o.problems and o.error is None]
    assert passing, '\n'.join(p for o in outcomes for p in o.problems)
