# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A forecast and an anomaly score asked in PQL, and answered in process.

The query names the series and the question; gathering picks the rows of that
series, puts them in time order, and hands the model the history it reads.
Small random-weight models stand in for the published ones.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sdm_client import (
    KumoAnomaly,
    KumoForecast,
    KumoRelational,
    PredictiveQuery,
    Request,
    SDMClient,
    StructuredDataError,
    UnsupportedRequestError,
)
from sdm_client.core.warnings import SDMWarning
from sdm_client.models import spec_for

FORECAST = (
    'PREDICT readings.flow FORECAST 6 TIMEFRAMES FOR EACH readings.sensor_id'
)
ANOMALY = 'PREDICT readings.* DETECT ANOMALY FOR EACH readings.sensor_id'
FAST = {'num_samples': 2, 'num_steps': 4}


def one_series(sensor: str, rows: int = 120, lift: float = 0.0) -> pd.DataFrame:
    step = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            'sensor_id': sensor,
            'at': pd.date_range('2026-01-01', periods=rows, freq='h'),
            'flow': np.sin(step / 9) + lift,
            'temp': np.cos(step / 7),
        }
    )


def readings() -> pd.DataFrame:
    """Two sensors' readings in one table, as a warehouse would hand them."""
    return pd.concat(
        [one_series('s-1'), one_series('s-2', lift=3.0)], ignore_index=True
    )


def gathered(text: str, frame: pd.DataFrame, **how) -> Request:
    return PredictiveQuery(text, {'readings': frame}).materialize(**how)


def refusal(text: str, frame: pd.DataFrame, **how) -> str:
    with pytest.raises(StructuredDataError) as caught:
        gathered(text, frame, **how)
    assert caught.value.code == 'INVALID_REQUEST'
    return str(caught.value)


def test_a_forecast_query_reads_the_series_it_names(forecast):
    """The entity is the series key and `indices` picks one. What the model is
    shown is that series alone, without the key that named it or the time
    that ordered it, so the answer is the one the frames themselves give."""
    request = gathered(
        FORECAST, readings(), indices=['s-2'], model=KumoForecast()
    )

    history = request.context.instance_table
    assert list(history.columns) == ['flow', 'temp']
    assert len(history) == 120
    assert request.horizon == 6
    assert request.query == FORECAST

    with SDMClient.in_process(forecast) as client:
        asked = client.predict(request, model=KumoForecast()).to_pandas()
        direct = client.predict(
            Request.forecast(
                one_series('s-2', lift=3.0)[['flow', 'temp']],
                target='flow',
                horizon=6,
            ),
            model=KumoForecast(),
        ).to_pandas()

    assert asked['prediction'].tolist() == pytest.approx(
        direct['prediction'].tolist()
    )


def test_an_anomaly_query_scores_every_row_of_the_series(anomaly):
    request = gathered(
        ANOMALY, readings(), indices=['s-2'], model=KumoAnomaly(), **FAST
    )

    scored = request.predict.instance_table
    assert list(scored.columns) == ['flow', 'temp', 'row']
    assert request.schema['readings'].primary_key == 'row'
    assert request.inference == FAST

    with SDMClient.in_process(anomaly) as client:
        answer = client.predict(request, model=KumoAnomaly()).to_pandas()

    assert answer['row_index'].tolist() == list(range(120))
    assert answer['id'].tolist() == list(range(120, 240))


def test_a_score_names_the_row_it_was_given_not_where_it_was_sorted_to(
    anomaly,
):
    """Gathering puts the rows in time order, so a score's place in the
    answer is not its row's place in the table. Each score names its row by
    where the caller had it, and the same reading scores the same either
    way round."""
    in_order = one_series('s-1')
    reversed_ = in_order.iloc[::-1].reset_index(drop=True)

    def scores(frame):
        request = gathered(ANOMALY, frame, model=KumoAnomaly(), **FAST)
        with SDMClient.in_process(anomaly) as client:
            return client.predict(request, model=KumoAnomaly()).to_pandas()

    with pytest.warns(SDMWarning, match='not in at order'):
        backwards = scores(reversed_)
    forwards = scores(in_order)

    assert forwards['id'].tolist() == list(range(120))
    assert backwards['id'].tolist() == list(range(119, -1, -1))
    assert backwards['prediction'].tolist() == pytest.approx(
        forwards['prediction'].tolist()
    )


def test_a_key_the_table_has_names_the_rows_when_the_time_names_the_series(
    anomaly,
):
    """The entity is the time, which is left out of what the model reads.
    The table's own key is still there to name each row by, and a key is
    never read as a reading."""
    from sdm_client import Schema

    frame = one_series('s-1').drop(columns='sensor_id')
    frame.insert(0, 'row_id', [f'r-{step:03d}' for step in range(len(frame))])
    schema = Schema.from_data(
        {'readings': frame}, primary_keys={'readings': 'row_id'}
    )
    text = 'PREDICT readings.* DETECT ANOMALY FOR EACH readings.at'

    request = PredictiveQuery(text, schema).materialize(
        model=KumoAnomaly(), **FAST
    )

    assert list(request.predict.instance_table.columns) == [
        'row_id',
        'flow',
        'temp',
    ]
    assert request.schema['readings'].primary_key == 'row_id'
    with SDMClient.in_process(anomaly) as client:
        answer = client.predict(request, model=KumoAnomaly()).to_pandas()
    assert answer['id'].tolist() == frame['row_id'].tolist()


def test_a_key_displaced_by_the_entity_is_not_read_as_a_reading():
    """The table says `row_id` is its key and the query names each row by
    its step. One of them names the row; the other is a second name for
    it, and a name is never a reading."""
    from sdm_client import Schema

    frame = one_series('s-1')[['flow', 'temp']]
    frame.insert(0, 'step', np.arange(len(frame)))
    frame.insert(0, 'row_id', [f'r-{step:03d}' for step in range(len(frame))])
    schema = Schema.from_data(
        {'readings': frame}, primary_keys={'readings': 'row_id'}
    )
    text = 'PREDICT readings.* DETECT ANOMALY FOR EACH readings.step'

    request = PredictiveQuery(text, schema).materialize(
        model=KumoAnomaly(), **FAST
    )

    assert list(request.predict.instance_table.columns) == [
        'step',
        'flow',
        'temp',
    ]
    assert request.schema['readings'].primary_key == 'step'


def test_a_series_column_with_nothing_in_it_names_no_series():
    frame = one_series('s-1').assign(sensor_id=None)

    said = refusal(FORECAST, frame, model=KumoForecast())

    assert "no row of 'readings' has a sensor_id" in said


def test_a_table_with_a_column_named_row_keeps_it_as_a_reading():
    frame = one_series('s-1').drop(columns='sensor_id').assign(row=1.5)
    text = 'PREDICT readings.* DETECT ANOMALY FOR EACH readings.at'

    request = gathered(text, frame, model=KumoAnomaly())

    scored = request.predict.instance_table
    assert list(scored.columns) == ['flow', 'temp', 'row', 'row_']
    assert request.schema['readings'].primary_key == 'row_'


def test_a_table_holding_one_series_needs_no_index():
    request = gathered(FORECAST, one_series('s-1'), model=KumoForecast())

    assert len(request.context.instance_table) == 120


def test_an_entity_that_names_the_row_makes_the_table_the_series():
    """A key with one value per row says nothing about which series, so
    there is one: the table."""
    frame = one_series('s-1').drop(columns='sensor_id')
    frame.insert(0, 'reading_id', np.arange(len(frame)))
    text = 'PREDICT readings.* DETECT ANOMALY FOR EACH readings.reading_id'

    request = gathered(text, frame, model=KumoAnomaly())

    scored = request.predict.instance_table
    assert list(scored.columns) == ['reading_id', 'flow', 'temp']
    assert request.schema['readings'].primary_key == 'reading_id'
    assert 'there is no series to choose' in refusal(
        text, frame, indices=[3], model=KumoAnomaly()
    )


def test_rows_already_in_order_are_read_without_a_word():
    with warnings.catch_warnings():
        warnings.simplefilter('error', SDMWarning)
        gathered(FORECAST, readings(), indices=['s-1'], model=KumoForecast())


def test_times_written_across_a_clock_change_are_still_times():
    """Local times carry a different offset either side of a clock change,
    and one writer may give seconds where another gives fractions. They are
    all moments, and moments order."""
    from sdm_client import Schema

    in_order = one_series('s-1').drop(columns='sensor_id').head(40)
    local = pd.date_range(
        '2026-10-15', periods=40, freq='D', tz='America/Los_Angeles'
    )
    spelled = [moment.isoformat() for moment in local]
    spelled[3] = local[3].isoformat(timespec='milliseconds')
    schema = Schema.from_data(
        {'readings': in_order.assign(at=spelled)},
        time_columns={'readings': 'at'},
    )
    text = 'PREDICT readings.flow FORECAST 4 TIMEFRAMES FOR EACH readings.at'

    request = PredictiveQuery(text, schema).materialize(model=KumoForecast())

    assert request.context.instance_table['flow'].tolist() == pytest.approx(
        in_order['flow'].tolist()
    )


@pytest.mark.parametrize('spelled', ['1/2/2026', '2 Jan 2026'])
def test_a_time_not_written_the_one_unambiguous_way_is_refused(spelled):
    """1/2/2026 is January to one writer and February to another. A guess
    would order the rows confidently and wrongly, so only ISO 8601 is read."""
    from sdm_client import Schema

    frame = one_series('s-1').drop(columns='sensor_id').astype({'at': str})
    frame.loc[3, 'at'] = spelled
    schema = Schema.from_data(
        {'readings': frame}, time_columns={'readings': 'at'}
    )
    text = 'PREDICT readings.flow FORECAST 4 TIMEFRAMES FOR EACH readings.at'

    with pytest.raises(StructuredDataError) as caught:
        PredictiveQuery(text, schema).materialize(model=KumoForecast())

    assert f'is not an ISO 8601 time, for example {spelled!r}' in str(
        caught.value
    )


def test_two_rows_at_one_time_are_not_a_series():
    """Two sensors read as one series would interleave their steps, which is
    what a repeated time is nearly always a sign of."""
    frame = readings().assign(sensor_id='s-1')

    said = refusal(FORECAST, frame, model=KumoForecast())

    assert '120 rows' in said
    assert 'one row per step' in said


def test_a_row_with_no_time_has_no_place_in_the_series():
    frame = one_series('s-1')
    frame.loc[5, 'at'] = pd.NaT

    assert 'have no at' in refusal(FORECAST, frame, model=KumoForecast())


def test_several_series_have_to_be_told_apart():
    said = refusal(FORECAST, readings(), model=KumoForecast())

    assert '2 series by sensor_id' in said
    assert "indices=['s-1']" in said


def test_one_request_reads_one_series():
    said = refusal(
        FORECAST, readings(), indices=['s-1', 's-2'], model=KumoForecast()
    )

    assert 'Materialize once for each sensor_id' in said


def test_a_series_that_is_not_there_is_said_to_be_missing():
    said = refusal(FORECAST, readings(), indices=['s-9'], model=KumoForecast())

    assert "no row of 'readings' has sensor_id 's-9'" in said


def test_a_series_named_in_the_query_is_refused_for_indices():
    """The language can name one entity with FOR, but a series is named by
    indices, so the query's own choice is refused rather than overruled."""
    text = (
        'PREDICT readings.flow FORECAST 6 TIMEFRAMES '
        "FOR readings.sensor_id = 's-2'"
    )

    said = refusal(text, readings(), indices=['s-1'], model=KumoForecast())

    assert 'Write FOR EACH readings.sensor_id' in said


def test_an_index_names_one_series_by_one_value():
    picks = [['s-1', 's-2'] * 120]

    said = refusal(FORECAST, readings(), indices=picks, model=KumoForecast())

    assert 'its one entry is a list' in said


def test_a_column_empty_in_the_series_keeps_the_type_the_table_gave_it():
    """`pressure` is measured on one sensor only. Read from that slice
    alone, the other sensor's empty column has no type to infer."""
    frame = readings()
    frame['pressure'] = pd.Series([None] * 120 + list(range(120)), dtype=object)

    request = gathered(FORECAST, frame, indices=['s-1'], model=KumoForecast())

    assert request.schema['readings']['pressure'].stype == 'numerical'


@pytest.mark.parametrize(
    ('model', 'text', 'form'),
    [
        (
            KumoForecast(),
            'PREDICT readings.flow FOR EACH readings.sensor_id',
            'FORECAST <n> TIMEFRAMES',
        ),
        (
            KumoAnomaly(),
            'PREDICT readings.flow FOR EACH readings.sensor_id',
            'DETECT ANOMALY',
        ),
    ],
)
def test_a_query_without_the_clause_asks_the_model_nothing(model, text, form):
    """Each profile accepts a bare column, which names no task. The clause
    is what asks, so a query without it is refused with the form to write."""
    said = refusal(text, readings(), indices=['s-1'], model=model)

    assert f'does not ask {model.name} anything' in said
    assert form in said


@pytest.mark.parametrize(
    'unread',
    [
        {'predict': pd.DataFrame({'flow': [1.0]})},
        {'num_context_samples': 10},
        {'candidates': [1]},
        {'task': 'regression'},
    ],
)
def test_an_argument_that_chooses_rows_is_refused_by_name(unread):
    said = refusal(FORECAST, readings(), model=KumoForecast(), **unread)

    assert str(sorted(unread)) in said


def test_a_second_table_is_not_part_of_a_series():
    query = PredictiveQuery(
        FORECAST,
        {
            'readings': readings(),
            'sensors': pd.DataFrame({'sensor_id': ['s-1', 's-2']}),
        },
    )

    with pytest.raises(StructuredDataError, match='one series from one table'):
        query.materialize(indices=['s-1'], model=KumoForecast())


def test_a_series_forecast_is_a_question_only_its_model_answers(forecast):
    """Its target is a bare column, so it is the static shape and no window
    is needed; what it does need is a model that continues a series."""
    request = gathered(
        FORECAST, readings(), indices=['s-1'], model=KumoForecast()
    )

    with SDMClient.in_process(forecast) as client:
        with pytest.raises(UnsupportedRequestError) as caught:
            client.predict(request, model=KumoRelational())
    assert 'series_forecasting' in str(caught.value)
    assert spec_for(KumoRelational()).model_id in str(caught.value)


def test_what_the_caller_said_a_column_means_is_what_the_model_is_told():
    """The series is a slice of the caller's table, and a slice can look
    different from the whole. What a column means was settled against the
    whole, so it is not inferred again from the slice."""
    from sdm_client import Schema

    schema = Schema.from_data({'readings': readings()})
    schema['readings']['temp'].stype = 'categorical'

    request = PredictiveQuery(FORECAST, schema).materialize(
        indices=['s-1'], model=KumoForecast()
    )

    assert request.schema['readings']['temp'].stype == 'categorical'
    assert request.schema['readings'].primary_key is None


def test_a_target_the_schema_calls_a_key_is_forecast_once_it_is_a_number():
    """A reading that never repeats and is named like a key is inferred as
    one. The schema is what says what a column means, so it is the schema
    that is corrected, and the query then reads it as the series."""
    from sdm_client import Schema

    frame = one_series('s-1')[['at', 'flow']].rename(columns={'flow': 'bid'})
    text = 'PREDICT readings.bid FORECAST 3 TIMEFRAMES FOR EACH readings.at'
    schema = Schema.from_data({'readings': frame})
    assert schema['readings'].primary_key == 'bid'

    schema['readings'].primary_key = None
    schema['readings']['bid'].stype = 'numerical'
    request = PredictiveQuery(text, schema).materialize(model=KumoForecast())

    assert request.schema['readings']['bid'].stype == 'numerical'
    assert request.schema['readings'].primary_key is None
