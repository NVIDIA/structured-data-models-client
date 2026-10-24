# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A date literal at pandas' bounds answers alike on both routes (T1.20 fix 1).

The old validator holds a PQL date literal as a pandas ``Timestamp``. pandas
2 holds only ``Timestamp.min`` to ``Timestamp.max`` (1677-09-21
00:12:43.145224193 to 2262-04-11 23:47:16.854775807) and refuses a literal
outside as a query (``INVALID_REQUEST``); pandas 3 holds any year. The host
declares pandas 2's range to the core (``date_literal_range``), which then
refuses such a literal as an unreadable one, so S0 answers it on the core
route as on the old route, and never as ``INTERNAL_ERROR`` (the shadow
check's disagreement before the fix).

This suite runs on pandas 2 in CI's ``test`` job on Python 3.10, which
pandas 3 does not support, and on pandas 3 on every later Python.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
import pytest

from sdm_client import (
    PredictiveQuery,
    Schema,
    StructuredDataError,
    _routing,
)
from sdm_client._rust import s0
from sdm_client.query import RELATIONAL

CORE = ','.join(_routing.S0)

#: pandas 2 holds only 1677 to 2262; pandas 3 holds any year.
PANDAS_3 = int(pd.__version__.split('.')[0]) >= 3

#: Each boundary literal and whether pandas 2 holds it (the reference's
#: answers at b413e9a on pandas 2.2.3).
BOUNDARY = [
    ('0001-01-01', False),
    ('1600-01-01', False),
    ('1677-09-20', False),
    ('1677-09-21', False),
    ('1677-09-21 00:12:43', False),
    ('1677-09-21 00:12:44', True),
    ('1677-09-22', True),
    ('2262-04-10', True),
    ('2262-04-11', True),
    ('2262-04-11 23:47:16', True),
    ('2262-04-11 23:47:17', False),
    ('2262-04-12', False),
    ('2263-01-01', False),
    # Month-first, year 1.
    ('1/2/0001', False),
]


@pytest.fixture(autouse=True)
def _old(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_routing.ENVIRONMENT, '')


@pytest.fixture
def validated_on_core(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Each query the core validated."""
    validated: list[str] = []
    real = s0.validate

    def spy(text: str, definition: Any, profile: Any) -> Any:
        validated.append(text)
        return real(text, definition, profile)

    monkeypatch.setattr(s0, 'validate', spy)
    return validated


def _orders_schema() -> Any:
    # The users/orders schema of test_the_routing_seam.py.
    users = pd.DataFrame({'user_id': [1, 2, 3]})
    orders = pd.DataFrame(
        {
            'order_id': [1, 2, 3],
            'user_id': [1, 2, 2],
            'ts': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data({'users': users, 'orders': orders})


def _filtered(condition: str) -> str:
    return (
        f'PREDICT COUNT(orders.* WHERE {condition}, 0, 30, days) '
        'FOR EACH users.user_id'
    )


def _answers(
    text: str, monkeypatch: pytest.MonkeyPatch
) -> dict[str, str | None]:
    r"""Each route's answer to S0's validation of ``text``: ``None`` when
    accepted, else the code it refuses with."""
    schema = _orders_schema()
    answers: dict[str, str | None] = {}
    for route in ('old', 'core'):
        if route == 'core':
            monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
        else:
            monkeypatch.setenv(_routing.ENVIRONMENT, '')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                PredictiveQuery(text, schema)._resolve(RELATIONAL)
                answers[route] = None
            except StructuredDataError as refused:
                # Refused as a query, never by a named rule.
                assert 'rule' not in refused.details, refused.details
                answers[route] = refused.code
    monkeypatch.setenv(_routing.ENVIRONMENT, '')
    return answers


@pytest.mark.parametrize(('literal', 'held'), BOUNDARY)
def test_a_boundary_literal_answers_alike_on_both_routes(
    validated_on_core: list[str],
    monkeypatch: pytest.MonkeyPatch,
    literal: str,
    held: bool,
) -> None:
    text = _filtered(f'orders.ts > {literal}')
    answers = _answers(text, monkeypatch)
    assert validated_on_core == [text]
    assert 'INTERNAL_ERROR' not in answers.values(), answers
    want = None if PANDAS_3 or held else 'INVALID_REQUEST'
    assert answers == {'old': want, 'core': want}


@pytest.mark.parametrize(
    ('array', 'held'),
    [
        ('1677-09-22, 2262-04-11', True),
        ('2024-01-01, 1600-01-01', False),
        ('2263-01-01, 2024-01-01', False),
        ('2262-04-12, 1600-01-01', False),
    ],
)
def test_an_array_answers_alike_on_both_routes(
    validated_on_core: list[str],
    monkeypatch: pytest.MonkeyPatch,
    array: str,
    held: bool,
) -> None:
    text = _filtered(f'orders.ts IN ({array})')
    answers = _answers(text, monkeypatch)
    assert validated_on_core == [text]
    assert 'INTERNAL_ERROR' not in answers.values(), answers
    want = None if PANDAS_3 or held else 'INVALID_REQUEST'
    assert answers == {'old': want, 'core': want}


def test_an_out_of_range_literal_with_an_unknown_column_answers_alike(
    validated_on_core: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Refused either way: on pandas 2 by the literal, on pandas 3 by the
    # column; the same code on both routes.
    text = _filtered('orders.nope > 3 AND orders.ts > 1600-01-01')
    answers = _answers(text, monkeypatch)
    assert validated_on_core == [text]
    assert answers == {'old': 'INVALID_REQUEST', 'core': 'INVALID_REQUEST'}
