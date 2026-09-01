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
from sdm_relational_engine.rfm.backend.local.sampler import (
    date_offset_to_seconds,
)
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


@pytest.mark.parametrize('unit', ['seconds', 'SECONDS'])
def test_a_query_can_be_written_in_seconds(graph_definition, unit) -> None:
    query = parse_query_locally(
        f'PREDICT COUNT(orders.*, 0, 90, {unit}) FOR EACH users.user_id',
        graph_definition,
    )

    offset = query.get_all_target_aggregations()[0].date_offset_range
    assert offset.unit == TimeUnit.SECONDS
    assert offset.end == 90


def test_a_second_window_is_a_second_of_sampling_window() -> None:
    r"""The sampler bounds its window in seconds by walking a DateOffset's
    components, and a component it does not recognise contributes nothing.
    """
    assert date_offset_to_seconds(
        pd.DateOffset(seconds=60)
    ) == date_offset_to_seconds(pd.DateOffset(minutes=1))


@pytest.mark.parametrize(
    ('other', 'value', 'expected'),
    [
        (TimeUnit.MINUTES, 3, 180),
        (TimeUnit.HOURS, 1, 3600),
        (TimeUnit.DAYS, 1, 86_400),
        (TimeUnit.WEEKS, 1, 604_800),
    ],
)
def test_seconds_are_the_floor_of_the_cascade(other, value, expected) -> None:
    r"""Seconds are the smallest unit, so combining anything with them has to
    settle there rather than leaving the two sides in different units.
    """
    seconds = DateOffsetRange(start=0, end=30, unit=TimeUnit.SECONDS)
    coarser = DateOffsetRange(start=0, end=value, unit=other)

    first, second = DateOffsetRange.move_to_same_unit(seconds, coarser)

    assert first.unit == second.unit == TimeUnit.SECONDS
    assert first.end == 30
    assert second.end == expected


def test_the_diagnostics_agree_with_the_grammar() -> None:
    from predictive_query.parser.error_translator import (
        _REJECTED_TIME_UNITS,
        _SUPPORTED_TIME_UNITS,
    )

    assert 'seconds' in _SUPPORTED_TIME_UNITS
    assert 'seconds' not in _REJECTED_TIME_UNITS
    # The singular stays rejected: the grammar only accepts plurals.
    assert 'second' in _REJECTED_TIME_UNITS
    assert not set(_SUPPORTED_TIME_UNITS) & _REJECTED_TIME_UNITS


def test_every_unit_the_grammar_accepts_converts_to_seconds() -> None:
    for unit in TimeUnit:
        offset = DateOffsetRange._to_date_offset(unit, 3)
        assert date_offset_to_seconds(offset) > 0, unit
