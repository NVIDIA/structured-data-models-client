# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pandas as pd
import pytest
from sdm_relational_engine.rfm import Graph, LocalTable
from sdm_relational_engine.rfm.base.utils import Timedelta, Timestamp
from sdm_relational_engine.rfm.graph import (
    _infer_timedelta_from_timestamps,
    _timedelta_to_pquery,
)


def test_timedelta_to_pquery_days() -> None:
    assert _timedelta_to_pquery(Timedelta('1D')) == (1, 'days')
    assert _timedelta_to_pquery(Timedelta('7D')) == (7, 'days')
    assert _timedelta_to_pquery(Timedelta('30D')) == (30, 'days')


def test_timedelta_to_pquery_hours() -> None:
    assert _timedelta_to_pquery(Timedelta('1h')) == (1, 'hours')
    assert _timedelta_to_pquery(Timedelta('6h')) == (6, 'hours')
    assert _timedelta_to_pquery(Timedelta('23h')) == (23, 'hours')


def test_timedelta_to_pquery_prefers_days_over_hours() -> None:
    # 24h, 48h etc. are whole days, so this returns days, not hours
    assert _timedelta_to_pquery(Timedelta('24h')) == (1, 'days')
    assert _timedelta_to_pquery(Timedelta('48h')) == (2, 'days')


def test_timedelta_to_pquery_minutes() -> None:
    assert _timedelta_to_pquery(Timedelta('1min')) == (1, 'minutes')
    assert _timedelta_to_pquery(Timedelta('30min')) == (30, 'minutes')
    assert _timedelta_to_pquery(Timedelta('90min')) == (90, 'minutes')


def test_timedelta_to_pquery_prefers_hours_over_minutes() -> None:
    # 60min, 120min etc. are whole hours, so this returns hours, not minutes
    assert _timedelta_to_pquery(Timedelta('60min')) == (1, 'hours')
    assert _timedelta_to_pquery(Timedelta('120min')) == (2, 'hours')


def test_timedelta_to_pquery_unsupported_raises() -> None:
    with pytest.raises(ValueError, match='minutes, hours, or days'):
        _timedelta_to_pquery(Timedelta('45s'))
    with pytest.raises(ValueError, match='minutes, hours, or days'):
        _timedelta_to_pquery(Timedelta('90s'))


def test_timedelta_to_pquery_non_positive_raises() -> None:
    with pytest.raises(ValueError, match='positive'):
        _timedelta_to_pquery(Timedelta('0D'))
    with pytest.raises(ValueError, match='positive'):
        _timedelta_to_pquery(Timedelta('-1D'))


def test_infer_timedelta_uniform() -> None:
    series = pd.Series(
        [
            pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            pd.to_datetime(['2024-01-05', '2024-01-06', '2024-01-07']),
        ]
    )
    result = _infer_timedelta_from_timestamps(series)
    assert result == Timedelta('1D')


def test_infer_timedelta_warns_on_non_uniform() -> None:
    # Mix of 1-day and 2-day gaps
    series = pd.Series(
        [
            pd.to_datetime(
                ['2024-01-01', '2024-01-02', '2024-01-04']
            ),  # 1D, 2D
            pd.to_datetime(
                ['2024-01-01', '2024-01-02', '2024-01-03']
            ),  # 1D, 1D
        ]
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _infer_timedelta_from_timestamps(series)
    assert len(w) == 1
    assert 'not all equal' in str(w[0].message)


def test_infer_timedelta_no_warning_on_uniform() -> None:
    series = pd.Series(
        [
            pd.to_datetime(['2024-01-01', '2024-01-02']),
            pd.to_datetime(['2024-02-01', '2024-02-02']),
        ]
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _infer_timedelta_from_timestamps(series)
    assert len(w) == 0


def test_infer_timedelta_too_short_raises() -> None:
    # All series have only 1 element, so no deltas can be computed
    series = pd.Series(
        [
            pd.to_datetime(['2024-01-01']),
            pd.to_datetime(['2024-02-01']),
        ]
    )
    with pytest.raises(ValueError, match='fewer than 2 observations'):
        _infer_timedelta_from_timestamps(series)


@pytest.fixture
def daily_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'customer_id': [1, 2, 3],
            'region': ['US', 'EU', 'US'],
            'sales': [[10, 20, 15, 30], [5, 8, 12], [100, 95, 80, 60, 50]],
        }
    )


@pytest.fixture
def anchor() -> pd.Timestamp:
    return Timestamp('2024-01-10')


def test_basic_with_entity_col(
    daily_df: pd.DataFrame, anchor: pd.Timestamp
) -> None:
    graph, pquery = Graph.graph_and_pquery_from_timeseries(
        daily_df,
        timeseries_col='sales',
        entity_col='customer_id',
        time_delta=Timedelta('1D'),
        anchor_time=anchor,
        num_timeframes=4,
    )

    assert pquery == (
        'PREDICT MAX(target.value, 0, 1, days) '
        'FORECAST 4 TIMEFRAMES FOR EACH entity.customer_id'
    )

    assert set(graph.tables.keys()) == {'entity', 'target'}
    assert len(graph.edges) == 1
    src, fkey, dst = graph.edges[0]
    assert src == 'target'
    assert fkey == 'customer_id'
    assert dst == 'entity'


def test_entity_table_structure(
    daily_df: pd.DataFrame, anchor: pd.Timestamp
) -> None:
    graph, _ = Graph.graph_and_pquery_from_timeseries(
        daily_df,
        timeseries_col='sales',
        entity_col='customer_id',
        time_delta=Timedelta('1D'),
        anchor_time=anchor,
    )

    entity = graph.tables['entity']
    assert isinstance(entity, LocalTable)
    assert entity._primary_key == 'customer_id'
    assert set(entity._data.columns) == {'customer_id', 'region'}
    assert list(entity._data['customer_id']) == [1, 2, 3]
    # timeseries_col must be absent
    assert 'sales' not in entity._data.columns


def test_target_table_structure(
    daily_df: pd.DataFrame, anchor: pd.Timestamp
) -> None:
    graph, _ = Graph.graph_and_pquery_from_timeseries(
        daily_df,
        timeseries_col='sales',
        entity_col='customer_id',
        time_delta=Timedelta('1D'),
        anchor_time=anchor,
    )

    target = graph.tables['target']
    assert isinstance(target, LocalTable)
    assert target._time_column == 'timestamp'
    assert set(target._data.columns) == {'customer_id', 'timestamp', 'value'}
    # Total rows = 4 + 3 + 5
    assert len(target._data) == 12


def test_timestamps_go_backwards_from_anchor(
    daily_df: pd.DataFrame, anchor: pd.Timestamp
) -> None:
    graph, _ = Graph.graph_and_pquery_from_timeseries(
        daily_df,
        timeseries_col='sales',
        entity_col='customer_id',
        time_delta=Timedelta('1D'),
        anchor_time=anchor,
    )

    target_table = graph.tables['target']
    assert isinstance(target_table, LocalTable)
    target = target_table._data
    # Every entity's last observation must be exactly one step before anchor
    for cid in [1, 2, 3]:
        last_ts = target[target['customer_id'] == cid]['timestamp'].max()
        assert last_ts == anchor - Timedelta('1D')


def test_auto_entity_id(anchor: pd.Timestamp) -> None:
    df = pd.DataFrame({'sales': [[1, 2], [3, 4, 5]]})
    graph, pquery = Graph.graph_and_pquery_from_timeseries(
        df,
        timeseries_col='sales',
        time_delta=Timedelta('1D'),
        anchor_time=anchor,
        num_timeframes=3,
    )

    assert 'FOR EACH entity.entity_id' in pquery
    entity = graph.tables['entity']
    assert isinstance(entity, LocalTable)
    assert entity._primary_key == 'entity_id'
    assert list(entity._data['entity_id']) == [0, 1]


def test_hourly_delta(anchor: pd.Timestamp) -> None:
    df = pd.DataFrame({'id': [1], 'vals': [[10, 20, 30]]})
    _, pquery = Graph.graph_and_pquery_from_timeseries(
        df,
        timeseries_col='vals',
        entity_col='id',
        time_delta=Timedelta('6h'),
        anchor_time=anchor,
        num_timeframes=6,
    )
    assert '6, hours' in pquery


def test_with_timestamps_col() -> None:
    df = pd.DataFrame(
        {
            'id': [1, 2],
            'vals': [[10, 20, 30], [5, 15, 25, 35]],
            'times': [
                pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
                pd.to_datetime(
                    ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']
                ),
            ],
        }
    )
    graph, pquery = Graph.graph_and_pquery_from_timeseries(
        df,
        timeseries_col='vals',
        timestamps_col='times',
        entity_col='id',
        num_timeframes=5,
    )

    assert '1, days' in pquery
    target_table = graph.tables['target']
    assert isinstance(target_table, LocalTable)
    assert len(target_table._data) == 7
    # timestamps_col must not appear in entity table
    entity_table = graph.tables['entity']
    assert isinstance(entity_table, LocalTable)
    assert 'times' not in entity_table._data.columns


def test_timestamps_col_takes_priority_over_anchor(
    anchor: pd.Timestamp,
) -> None:
    """When timestamps_col is provided, anchor_time is unused."""
    df = pd.DataFrame(
        {
            'id': [1],
            'vals': [[1, 2]],
            'times': [pd.to_datetime(['2023-06-01', '2023-06-02'])],
        }
    )
    graph, _ = Graph.graph_and_pquery_from_timeseries(
        df,
        timeseries_col='vals',
        timestamps_col='times',
        entity_col='id',
        anchor_time=anchor,  # should be ignored
    )
    target_table = graph.tables['target']
    assert isinstance(target_table, LocalTable)
    assert list(target_table._data['timestamp']) == list(
        pd.to_datetime(['2023-06-01', '2023-06-02'])
    )


def test_error_missing_timeseries_col(anchor: pd.Timestamp) -> None:
    df = pd.DataFrame({'a': [[1, 2]]})
    with pytest.raises(ValueError, match='timeseries_col'):
        Graph.graph_and_pquery_from_timeseries(
            df,
            timeseries_col='missing',
            time_delta=Timedelta('1D'),
            anchor_time=anchor,
        )


def test_error_missing_timestamps_col(anchor: pd.Timestamp) -> None:
    df = pd.DataFrame({'vals': [[1, 2]]})
    with pytest.raises(ValueError, match='timestamps_col'):
        Graph.graph_and_pquery_from_timeseries(
            df,
            timeseries_col='vals',
            timestamps_col='missing',
            time_delta=Timedelta('1D'),
            anchor_time=anchor,
        )


def test_error_missing_entity_col(anchor: pd.Timestamp) -> None:
    df = pd.DataFrame({'vals': [[1, 2]]})
    with pytest.raises(ValueError, match='entity_col'):
        Graph.graph_and_pquery_from_timeseries(
            df,
            timeseries_col='vals',
            entity_col='missing',
            time_delta=Timedelta('1D'),
            anchor_time=anchor,
        )


def test_error_no_timestamps_and_no_time_delta(anchor: pd.Timestamp) -> None:
    df = pd.DataFrame({'vals': [[1, 2]]})
    with pytest.raises(ValueError, match='time_delta'):
        Graph.graph_and_pquery_from_timeseries(
            df, timeseries_col='vals', anchor_time=anchor
        )


def test_error_no_timestamps_and_no_anchor_time() -> None:
    df = pd.DataFrame({'vals': [[1, 2]]})
    with pytest.raises(ValueError, match='anchor_time'):
        Graph.graph_and_pquery_from_timeseries(
            df, timeseries_col='vals', time_delta=Timedelta('1D')
        )


def test_error_entity_id_conflicts_with_column(anchor: pd.Timestamp) -> None:
    df = pd.DataFrame({'entity_id': [1, 2], 'vals': [[1, 2], [3, 4]]})
    with pytest.raises(ValueError, match='conflicts'):
        Graph.graph_and_pquery_from_timeseries(
            df,
            timeseries_col='vals',
            time_delta=Timedelta('1D'),
            anchor_time=anchor,
        )


def test_error_unsupported_time_delta(anchor: pd.Timestamp) -> None:
    df = pd.DataFrame({'vals': [[1, 2]]})
    with pytest.raises(ValueError, match='minutes, hours, or days'):
        Graph.graph_and_pquery_from_timeseries(
            df,
            timeseries_col='vals',
            time_delta=Timedelta('45s'),
            anchor_time=anchor,
        )
