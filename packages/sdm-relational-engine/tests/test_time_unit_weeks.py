# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from predictive_query.ast.date_offset_range import (
    DateOffsetRange,
)
from predictive_query.schema.typing import TimeUnit
from sdm_relational_engine.rfm import Graph
from sdm_relational_engine.rfm.backend.local import LocalTable
from sdm_relational_engine.rfm.query_parser import parse_query_locally


@pytest.fixture
def graph_definition() -> object:
    users = LocalTable(
        df=pd.DataFrame({'user_id': [1, 2], 'age': [20, 30]}),
        name='users',
        primary_key='user_id',
    )
    orders = LocalTable(
        df=pd.DataFrame(
            {
                'order_id': [1, 2],
                'user_id': [1, 2],
                'price': [1.0, 2.0],
                'ts': pd.to_datetime(['2024-01-01', '2024-02-01']),
            }
        ),
        name='orders',
        primary_key='order_id',
        time_column='ts',
    )
    graph = Graph([users, orders])
    graph.link('orders', 'user_id', 'users')
    return graph._to_api_graph_definition()


@pytest.mark.parametrize('unit', ['weeks', 'WEEKS'])
def test_a_query_can_be_written_in_weeks(graph_definition, unit) -> None:
    query = parse_query_locally(
        f'PREDICT COUNT(orders.*, 0, 4, {unit}) FOR EACH users.user_id',
        graph_definition,
    )

    offset = query.get_all_target_aggregations()[0].date_offset_range
    assert offset.unit == TimeUnit.WEEKS
    assert offset.end == 4


def test_weeks_reduce_to_days_when_units_are_combined() -> None:
    r"""A week is seven days everywhere, so unlike a month this conversion is
    exact and needs no warning.
    """
    weeks = DateOffsetRange(start=0, end=4, unit=TimeUnit.WEEKS)
    days = DateOffsetRange(start=0, end=30, unit=TimeUnit.DAYS)

    first, second = DateOffsetRange.move_to_same_unit(weeks, days)

    assert first.unit == second.unit == TimeUnit.DAYS
    assert first.end == 28
    assert second.end == 30


def test_four_weeks_is_twenty_eight_days_on_the_calendar() -> None:
    offset = DateOffsetRange._to_date_offset(TimeUnit.WEEKS, 4)

    assert pd.Timestamp('2024-01-01') + offset == pd.Timestamp('2024-01-29')


@pytest.mark.parametrize(
    ('weeks', 'days'), [(1, 7), (2, 14), (4, 28), (13, 91)]
)
def test_a_week_window_is_seven_days_of_seconds(weeks, days) -> None:
    r"""The sampler bounds its window in seconds, and a unit it does not
    recognise contributes nothing: weeks silently became a zero-length window,
    so every prediction came back empty rather than failing.
    """
    from sdm_relational_engine.rfm.backend.local.sampler import (
        date_offset_to_seconds,
    )

    assert date_offset_to_seconds(
        pd.DateOffset(weeks=weeks)
    ) == date_offset_to_seconds(pd.DateOffset(days=days))


def test_every_unit_the_grammar_accepts_converts_to_seconds() -> None:
    r"""Pins the whole set together, so adding a unit to the grammar without
    teaching the sampler about it fails here rather than in someone's
    predictions.
    """
    from sdm_relational_engine.rfm.backend.local.sampler import (
        date_offset_to_seconds,
    )

    for unit in (
        TimeUnit.MINUTES,
        TimeUnit.HOURS,
        TimeUnit.DAYS,
        TimeUnit.WEEKS,
        TimeUnit.MONTHS,
    ):
        offset = DateOffsetRange._to_date_offset(unit, 3)
        assert date_offset_to_seconds(offset) > 0, unit


def test_the_diagnostics_agree_with_the_grammar() -> None:
    r"""Two lists describe which units exist, and a unit the grammar accepts
    while the diagnostics call it rejected produces an error message that
    contradicts the parser.
    """
    from predictive_query.parser.error_translator import (
        _REJECTED_TIME_UNITS,
        _SUPPORTED_TIME_UNITS,
    )

    assert 'weeks' in _SUPPORTED_TIME_UNITS
    assert 'weeks' not in _REJECTED_TIME_UNITS
    # The singular stays rejected: the grammar only accepts plurals.
    assert 'week' in _REJECTED_TIME_UNITS
    assert not set(_SUPPORTED_TIME_UNITS) & _REJECTED_TIME_UNITS
