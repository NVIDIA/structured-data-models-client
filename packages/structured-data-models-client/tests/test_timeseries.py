# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import warnings
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from sdm_client import KumoRelational, StructuredDataError, build_timeseries

CASES = json.loads(
    (
        Path(__file__).resolve().parents[3]
        / 'tests/fixtures/timeseries/cases.json'
    ).read_text()
)


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['name'])
def test_shared_history_preparation(case):
    frame = pd.DataFrame(
        {'id': range(1, len(case['histories']) + 1), 'sales': case['histories']}
    )
    if case['timestamps'] is not None:
        frame['times'] = [
            [
                None if v is None else pd.Timestamp(v, unit='ns', tz='UTC')
                for v in row
            ]
            for row in case['timestamps']
        ]
    options = dict(
        entity_col='id',
        timestamps_col=None if case['timestamps'] is None else 'times',
        time_delta=None
        if case['step_minutes'] is None
        else pd.Timedelta(minutes=case['step_minutes']),
        anchor_time=None
        if case['anchor_ns'] is None
        else pd.Timestamp(case['anchor_ns'], unit='ns', tz='UTC'),
        num_timeframes=2,
    )
    before = frame.copy(deep=True)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        if case.get('error'):
            with pytest.raises(StructuredDataError) as error:
                build_timeseries(frame, 'sales', **options)
            assert error.value.code == 'INVALID_REQUEST'
            return
        prepared = build_timeseries(frame, 'sales', **options)
    pd.testing.assert_frame_equal(frame, before)
    rows = prepared.schema['target'].data
    assert rows['id'].tolist() == case['expected_ids']
    assert [v.value for v in rows['timestamp']] == case['expected_times']
    assert [None if pd.isna(v) else v for v in rows['value']] == case[
        'expected_values'
    ]
    assert prepared.schema['target'].primary_key is None
    assert prepared.schema['entity'].time_column is None
    assert len(prepared.schema['entity'].data) == len(frame)
    assert 'FORECAST 2 TIMEFRAMES' in prepared.query.text
    if case['name'] == 'explicit_irregular':
        assert any('median' in str(w.message) for w in emitted)


@pytest.mark.parametrize(
    'options',
    [
        {'entity_col': 'value'},
        {'entity_col': 'timestamp'},
        {'entity_col': 'sales'},
        {'timestamps_col': 'sales'},
        {'num_timeframes': True},
        {'num_timeframes': 0},
        {'time_delta': pd.Timedelta(seconds=60, microseconds=1)},
        {'anchor_time': pd.Timestamp.min, 'time_delta': pd.Timedelta('1D')},
    ],
)
def test_bad_options_are_structured(options):
    kwargs = dict(
        time_delta=pd.Timedelta('1D'), anchor_time=pd.Timestamp('2024-01-03')
    )
    kwargs.update(options)
    with pytest.raises(StructuredDataError):
        build_timeseries(pd.DataFrame({'sales': [[1, 2]]}), 'sales', **kwargs)


def test_arrow_and_pandas_generated_ids_and_ownership():
    table = pa.table({'sales': [[1.0, 2.0], [3.0]], 'label': ['a', 'b']})
    result = build_timeseries(
        table,
        'sales',
        time_delta=pd.Timedelta('1D'),
        anchor_time=pd.Timestamp('2024-01-03'),
    )
    assert table.to_pydict()['sales'] == [[1.0, 2.0], [3.0]]
    assert result.schema['entity'].data['entity_id'].tolist() == [0, 1]
    assert str(result.anchor_time.tz) == 'UTC'


@pytest.mark.parametrize('ids', [[1, 1], [1, None]])
def test_invalid_entity_keys(ids):
    with pytest.raises(StructuredDataError):
        build_timeseries(
            pd.DataFrame({'id': ids, 'sales': [[1], [2]]}),
            'sales',
            entity_col='id',
            time_delta=pd.Timedelta('1D'),
            anchor_time=pd.Timestamp('2024-01-03'),
        )


def test_mixed_timezones_and_duplicate_names_refused():
    frame = pd.DataFrame(
        {
            'sales': [[1, 2]],
            'times': [
                [
                    pd.Timestamp('2024-01-01'),
                    pd.Timestamp('2024-01-02', tz='UTC'),
                ]
            ],
        }
    )
    with pytest.raises(StructuredDataError):
        build_timeseries(frame, 'sales', timestamps_col='times')
    duplicate = pa.Table.from_arrays(
        [pa.array([[1]]), pa.array([[2]])], names=['sales', 'sales']
    )
    with pytest.raises(StructuredDataError):
        build_timeseries(
            duplicate,
            'sales',
            time_delta=pd.Timedelta('1D'),
            anchor_time=pd.Timestamp('2024-01-03'),
        )


def test_public_forecast_materializes_through_existing_route():
    frame = pd.DataFrame(
        {
            'id': range(20),
            'sales': [[float(i + j) for j in range(30)] for i in range(20)],
        }
    )
    made = build_timeseries(
        frame,
        'sales',
        entity_col='id',
        time_delta=pd.Timedelta('1D'),
        anchor_time=pd.Timestamp('2024-02-01'),
        num_timeframes=2,
    )
    request = made.query.materialize(
        model=KumoRelational(),
        indices=[1, 2],
        anchor_time=made.anchor_time,
        num_context_samples=8,
        random_seed=1,
    )
    assert request.sanitization_report.status == 'available'
