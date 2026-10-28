# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the time-series models may be asked, and what they refuse."""

from __future__ import annotations

import pytest

from predictive_query import (
    ANOMALY,
    RELATIONAL,
    TABULAR,
    TIMESERIES,
    SchemaDefinition,
    TableDefinition,
)
from predictive_query.api import validate
from predictive_query.parser.parser import PQLParser
from predictive_query.profiles import Mode
from predictive_query.schema import codes
from predictive_query.schema.definition import ColumnKey, ColumnKeyGroup
from predictive_query.schema.table import Column
from predictive_query.schema.typing import Dtype, ProblemType, Stype
from predictive_query.validator.capability import check_parsed_capabilities
from predictive_query.validator.problem_type_validator import (
    ProblemTypeValidator,
)


def _column(name: str, stype: Stype, dtype: Dtype) -> Column:
    return Column(name=name, stype=stype, dtype=dtype)


def _series_schema() -> SchemaDefinition:
    """A prepared series: one row per timestep, keyed by its sensor."""
    tables = {
        'sensors': TableDefinition(
            cols=[_column('sensor_id', Stype.ID, Dtype.int)],
            pkey='sensor_id',
        ),
        'readings': TableDefinition(
            cols=[
                _column('reading_id', Stype.ID, Dtype.int),
                _column('sensor_id', Stype.ID, Dtype.int),
                _column('ts', Stype.timestamp, Dtype.time),
                _column('value', Stype.numerical, Dtype.float),
                _column('temp', Stype.numerical, Dtype.float),
            ],
            pkey='reading_id',
            time_col='ts',
        ),
    }
    col_groups = [
        ColumnKeyGroup(
            columns=(
                ColumnKey('readings', 'sensor_id'),
                ColumnKey('sensors', 'sensor_id'),
            )
        )
    ]
    return SchemaDefinition(tables=tables, col_groups=col_groups)


def _events_schema() -> SchemaDefinition:
    """Raw events, which only an aggregate can forecast."""
    tables = {
        'users': TableDefinition(
            cols=[_column('user_id', Stype.ID, Dtype.int)],
            pkey='user_id',
        ),
        'orders': TableDefinition(
            cols=[
                _column('order_id', Stype.ID, Dtype.int),
                _column('user_id', Stype.ID, Dtype.int),
                _column('ts', Stype.timestamp, Dtype.time),
                _column('price', Stype.numerical, Dtype.float),
            ],
            pkey='order_id',
            time_col='ts',
        ),
    }
    col_groups = [
        ColumnKeyGroup(
            columns=(
                ColumnKey('orders', 'user_id'),
                ColumnKey('users', 'user_id'),
            )
        )
    ]
    return SchemaDefinition(tables=tables, col_groups=col_groups)


SERIES_FORECAST = (
    'PREDICT readings.value FORECAST 72 TIMEFRAMES FOR EACH sensors.sensor_id'
)
WINDOWED_FORECAST = (
    'PREDICT SUM(orders.price, 0, 30, days) FORECAST 12 TIMEFRAMES '
    'FOR EACH users.user_id'
)
ROW_ENTITY_FORECAST = (
    'PREDICT readings.value FORECAST 72 TIMEFRAMES FOR EACH readings.reading_id'
)
ANOMALY_QUERY = 'PREDICT readings.* DETECT ANOMALY FOR EACH sensors.sensor_id'
ROW_ENTITY_ANOMALY = (
    'PREDICT readings.* DETECT ANOMALY FOR EACH readings.reading_id'
)


def parse(text: str):
    return PQLParser(mode=Mode.INFERENCE).to_parsed_predictive_query(text)


def problem_codes(text: str) -> list[str]:
    response = ProblemTypeValidator().validate(parse(text), Mode.INFERENCE)
    return [error.code for error in response.errors]


def capability_codes(text: str, profile) -> list[str]:
    response = check_parsed_capabilities(parse(text), profile)
    return [error.code for error in response.errors]


def test_detect_anomaly_resolves_to_its_own_problem_type():
    """The query text alone decides the task, before a model is named.

    A bare column target is already a legal static query, so reusing one
    would make the task depend on who the request was sent to.
    """
    assert parse(ANOMALY_QUERY).problem_type == ProblemType.ANOMALY


@pytest.mark.parametrize('text', ['detect anomaly', 'DETECT ANOMALY'])
def test_the_clause_reads_in_either_case(text: str):
    query = f'PREDICT readings.* {text} FOR EACH sensors.sensor_id'
    assert parse(query).problem_type == ProblemType.ANOMALY


def test_forecast_takes_a_bare_column():
    """The model forecasts the series it is given, at its own cadence.

    It cannot resample, so requiring a window would mean either lying about
    what the window does or resampling in the client.
    """
    resolved, response = validate(
        ROW_ENTITY_FORECAST, _series_schema(), TIMESERIES
    )
    assert response.errors == []
    assert resolved is not None
    assert resolved.num_forecasts == 72


def test_an_anomaly_query_validates_through_the_public_entry_point():
    """`validate` is what a caller reaches, so it is what is asserted.

    The wildcard rule permits `table.*` only inside COUNT, and an anomaly
    target is a bare one, so every anomaly query was refused there while the
    checks below passed on their own.
    """
    resolved, response = validate(ROW_ENTITY_ANOMALY, _series_schema(), ANOMALY)
    assert response.errors == []
    assert resolved is not None
    assert resolved.problem_type == ProblemType.ANOMALY


def test_a_wildcard_is_still_refused_outside_an_anomaly_target():
    """The exception is the anomaly target, not wildcards generally."""
    _, response = validate(
        'PREDICT readings.* FOR EACH readings.reading_id',
        _series_schema(),
        RELATIONAL,
    )
    assert [error.code for error in response.errors] == [
        codes.SCHEMA_WILDCARD_POSITION
    ]


@pytest.mark.parametrize('text', [ROW_ENTITY_ANOMALY, ROW_ENTITY_FORECAST])
def test_a_query_survives_being_written_back_out(text: str):
    """Materializers persist `to_string()`, so it has to be valid PQL."""
    written = parse(text).to_string()
    assert parse(written).to_string() == written


def test_a_series_entity_does_not_resolve_a_bare_column_target():
    """Open: the entity a series forecast should name is unsettled.

    A sensor has many readings, so `readings.value` is not one value per
    sensor and resolution refuses it. The aggregation form is what used to
    collapse the many to one, and a bare column has nothing in its place.
    Naming the row resolves, but then the query no longer says which series
    is being forecast.
    """
    _, response = validate(SERIES_FORECAST, _series_schema(), TIMESERIES)
    assert [error.code for error in response.errors] == [
        codes.JOIN_AMBIGUOUS_TARGET_VALUE
    ]


def test_forecast_still_takes_a_windowed_aggregate():
    """The relational form is untouched by the bare-column one."""
    _, response = validate(WINDOWED_FORECAST, _events_schema(), RELATIONAL)
    assert response.errors == []


def test_an_anomaly_target_names_the_row_and_not_a_column():
    """The score is joint across the scored channels.

    Naming one column would promise a score specific to it, which the model
    does not produce.
    """
    assert problem_codes(ANOMALY_QUERY) == []
    assert codes.TASK_ANOMALY_REQUIRES_WILDCARD_TARGET in problem_codes(
        'PREDICT readings.value DETECT ANOMALY FOR EACH sensors.sensor_id'
    )


@pytest.mark.parametrize(
    'text',
    [
        'PREDICT readings.* DETECT ANOMALY FOR EACH sensors.sensor_id '
        'ASSUMING readings.temp > 5',
        'PREDICT SUM(readings.value, 0, 1, days) DETECT ANOMALY '
        'FOR EACH sensors.sensor_id',
    ],
)
def test_a_score_carries_none_of_the_clauses_a_label_does(text: str):
    """Each asks something of a score that has no meaning."""
    assert codes.TASK_ANOMALY_INCOMPATIBLE_CLAUSE in problem_codes(text)


@pytest.mark.parametrize(
    'text',
    [
        'PREDICT readings.* DETECT ANOMALY TOP 5 FOR EACH sensors.sensor_id',
        'PREDICT readings.* DETECT ANOMALY RANK FOR EACH sensors.sensor_id',
        'PREDICT readings.* DETECT ANOMALY FORECAST 3 TIMEFRAMES '
        'FOR EACH sensors.sensor_id',
    ],
)
def test_the_grammar_already_excludes_the_other_problem_types(text: str):
    """They are alternatives of one clause, so this never reaches a check."""
    with pytest.raises(ValueError):
        parse(text)


def test_only_the_anomaly_model_is_offered_an_anomaly_query():
    for profile in (RELATIONAL, TABULAR, TIMESERIES):
        assert codes.MODEL_NO_ANOMALY_DETECTION in capability_codes(
            ANOMALY_QUERY, profile
        )
    assert capability_codes(ANOMALY_QUERY, ANOMALY) == []


def test_only_the_time_series_model_is_offered_a_series_forecast():
    for profile in (RELATIONAL, TABULAR, ANOMALY):
        assert codes.MODEL_NO_SERIES_FORECASTING in capability_codes(
            SERIES_FORECAST, profile
        )
    assert capability_codes(SERIES_FORECAST, TIMESERIES) == []


def test_the_time_series_model_refuses_a_windowed_forecast():
    """It cannot resample, so the window is what it refuses."""
    assert codes.MODEL_NO_TEMPORAL_TARGET in capability_codes(
        WINDOWED_FORECAST, TIMESERIES
    )
    assert capability_codes(WINDOWED_FORECAST, RELATIONAL) == []


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        (
            'SUM(orders.price, 0, 7, days) FOR users.user_id = 0',
            'Query does not start with the PREDICT keyword',
        ),
        (
            'PREDICT FOR EACH users.user_id',
            'The target (PREDICT) clause in this query is empty or invalid',
        ),
    ],
)
def test_a_grammar_change_keeps_the_friendly_syntax_errors(
    text: str, expected: str
) -> None:
    """The translator keys off ANTLR state numbers, which a rule shifts.

    Adding `DETECT ANOMALY` moved both by two and turned these back into raw
    parser output. Nothing failed in this package when that happened, so the
    messages are pinned here.
    """
    with pytest.raises(ValueError) as excinfo:
        parse(text)
    assert expected in str(excinfo.value)
