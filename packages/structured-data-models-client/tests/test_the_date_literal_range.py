# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PQL date bounds follow the declared pandas compatibility profile.
Fixed boundary expectations preserve the historical pandas 2/3 distinction.
The native validator must report query errors, never internal errors.
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
)
from sdm_client._rust import s0
from sdm_client.query import RELATIONAL

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


def _answer(text: str) -> str | None:
    """The current validator accepts the query or returns its public code."""
    schema = _orders_schema()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            PredictiveQuery(text, schema)._resolve(RELATIONAL)
        except StructuredDataError as refused:
            assert 'rule' not in refused.details, refused.details
            return refused.code
    return None


@pytest.mark.parametrize(('literal', 'held'), BOUNDARY)
def test_boundary_literal_respects_host_date_range(
    validated_on_core: list[str],
    literal: str,
    held: bool,
) -> None:
    text = _filtered(f'orders.ts > {literal}')
    answer = _answer(text)
    assert validated_on_core == [text]
    assert answer != 'INTERNAL_ERROR'
    want = None if PANDAS_3 or held else 'INVALID_REQUEST'
    assert answer == want


@pytest.mark.parametrize(
    ('array', 'held'),
    [
        ('1677-09-22, 2262-04-11', True),
        ('2024-01-01, 1600-01-01', False),
        ('2263-01-01, 2024-01-01', False),
        ('2262-04-12, 1600-01-01', False),
    ],
)
def test_date_array_respects_host_date_range(
    validated_on_core: list[str],
    array: str,
    held: bool,
) -> None:
    text = _filtered(f'orders.ts IN ({array})')
    answer = _answer(text)
    assert validated_on_core == [text]
    assert answer != 'INTERNAL_ERROR'
    want = None if PANDAS_3 or held else 'INVALID_REQUEST'
    assert answer == want


def test_unknown_column_with_out_of_range_date_is_a_query_error(
    validated_on_core: list[str],
) -> None:
    # Refused either way: on pandas 2 by the literal, on pandas 3 by the
    # column; both cases produce a query error.
    text = _filtered('orders.nope > 3 AND orders.ts > 1600-01-01')
    answer = _answer(text)
    assert validated_on_core == [text]
    assert answer == 'INVALID_REQUEST'
