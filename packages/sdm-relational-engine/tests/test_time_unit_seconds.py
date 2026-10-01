# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

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
from sdm_relational_engine.rfm.query.parser import parse_query_locally


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


# --- pooling two windows of different units --------------------------------


ANCHORS = [
    '2025-01-31',
    '2025-02-04',
    '2025-02-28',
    '2025-03-04',
    '2024-02-29',
    '2025-12-15',
    '2025-11-30',
    '2026-01-31',
    '2025-06-30',
]

PAIRS = [
    (pd.DateOffset(months=1), pd.DateOffset(months=1)),
    (pd.DateOffset(days=30), pd.DateOffset(days=30)),
    (pd.DateOffset(days=28), pd.DateOffset(months=1)),
    (pd.DateOffset(days=30), pd.DateOffset(months=1)),
    (pd.DateOffset(days=-30), pd.DateOffset(months=-1)),
    (pd.DateOffset(days=365), pd.DateOffset(years=1)),
    (pd.DateOffset(days=7), pd.DateOffset(days=30)),
    (pd.DateOffset(months=1), pd.DateOffset(months=3)),
    (pd.DateOffset(days=-30), pd.DateOffset(months=1)),
    (pd.DateOffset(hours=36), pd.DateOffset(days=2)),
]


@pytest.mark.parametrize('pair', PAIRS, ids=lambda p: str(p))
@pytest.mark.parametrize('anchor', ANCHORS)
def test_a_pooled_bound_brackets_at_every_anchor(pair, anchor):
    """No single anchor orders a month against thirty days.

    At the 4th of February a month lands earlier than thirty days and at the
    15th of December it lands later, so a bound picked by adding both to one
    fixed date is the wrong way round at every anchor that date does not
    describe, and a bound inside the calendar window drops rows no later
    step can recover.
    """
    from sdm_relational_engine.rfm.base.sampler import (
        max_date_offset,
        min_date_offset,
    )

    first, second = pair
    at = pd.Timestamp(anchor)

    assert at + min_date_offset(first, second) <= min(at + first, at + second)
    assert at + max_date_offset(first, second) >= max(at + first, at + second)


@pytest.mark.parametrize('pair', PAIRS, ids=lambda p: str(p))
def test_a_pooled_bound_brackets_in_seconds_too(pair):
    """The sampler converts it rather than adding it, so both readings hold."""
    from sdm_relational_engine.rfm.base.sampler import (
        max_date_offset,
        min_date_offset,
    )
    from sdm_relational_engine.rfm.base.utils import date_offset_to_seconds

    first, second = pair
    assert date_offset_to_seconds(
        min_date_offset(first, second), 'earliest'
    ) <= min(
        date_offset_to_seconds(first, 'earliest'),
        date_offset_to_seconds(second, 'earliest'),
    )
    assert date_offset_to_seconds(
        max_date_offset(first, second), 'latest'
    ) >= max(
        date_offset_to_seconds(first, 'latest'),
        date_offset_to_seconds(second, 'latest'),
    )


def test_a_calendar_window_pooled_with_itself_stays_exact():
    """Two aggregations over the same edge write the same window twice, and an
    offset does not bound an equal one by the seconds test: the most a month
    can span is longer than the least a month can span, so neither was found
    to bound the other and a calendar month pooled into a rigid 31 days.
    """
    from sdm_relational_engine.rfm.base.sampler import (
        max_date_offset,
        min_date_offset,
    )

    one, also_one = pd.DateOffset(months=1), pd.DateOffset(months=1)
    assert one is not also_one and one == also_one

    february = pd.Timestamp('2025-02-01')
    assert february + max_date_offset(one, also_one) == february + one
    assert february + min_date_offset(one, also_one) == february + one


def test_a_calendar_bound_stays_a_calendar_bound_where_one_of_them_bounds():
    """Only the pair no single offset bounds is turned into a rigid span."""
    from sdm_relational_engine.rfm.base.sampler import (
        max_date_offset,
        min_date_offset,
    )

    one, three = pd.DateOffset(months=1), pd.DateOffset(months=3)
    assert min_date_offset(one, three) is one
    assert max_date_offset(one, three) is three

    seven, thirty = pd.DateOffset(days=7), pd.DateOffset(days=30)
    assert min_date_offset(seven, thirty) is seven
    assert max_date_offset(seven, thirty) is thirty


def test_a_missing_bound_pools_to_nothing():
    from sdm_relational_engine.rfm.base.sampler import min_date_offset

    assert min_date_offset(None, pd.DateOffset(days=7)) is None
