# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What each model may be asked, as data."""

from __future__ import annotations

from dataclasses import fields

from predictive_query import ANOMALY, RELATIONAL, TABULAR, TIMESERIES
from predictive_query.profiles import Capabilities

FLAGS = tuple(f.name for f in fields(Capabilities) if f.name != 'model')

RELATIONAL_MATRIX = {
    'relationships': True,
    'multiple_tables': True,
    'link_prediction': True,
    'forecasting': True,
    'series_forecasting': False,
    'anomaly_detection': False,
    'static': True,
    'temporal': True,
    'entity_filters': True,
    'counterfactuals': True,
    'derived_targets': True,
    'key_columns_in_filters': True,
    'timestamp_columns_in_filters': True,
}

TABULAR_MATRIX = {
    'relationships': False,
    'multiple_tables': False,
    'link_prediction': False,
    'forecasting': False,
    'series_forecasting': False,
    'anomaly_detection': False,
    'static': True,
    'temporal': False,
    'entity_filters': False,
    'counterfactuals': False,
    'derived_targets': False,
    'key_columns_in_filters': True,
    'timestamp_columns_in_filters': True,
}

TIMESERIES_MATRIX = {
    'relationships': False,
    'multiple_tables': False,
    'link_prediction': False,
    'forecasting': True,
    'series_forecasting': True,
    'anomaly_detection': False,
    'static': True,
    'temporal': False,
    'entity_filters': False,
    'counterfactuals': False,
    'derived_targets': False,
    'key_columns_in_filters': False,
    'timestamp_columns_in_filters': False,
}

ANOMALY_MATRIX = {
    'relationships': False,
    'multiple_tables': False,
    'link_prediction': False,
    'forecasting': False,
    'series_forecasting': False,
    'anomaly_detection': True,
    'static': True,
    'temporal': False,
    'entity_filters': False,
    'counterfactuals': False,
    'derived_targets': False,
    'key_columns_in_filters': False,
    'timestamp_columns_in_filters': False,
}

PROFILES = (
    (RELATIONAL, RELATIONAL_MATRIX),
    (TABULAR, TABULAR_MATRIX),
    (TIMESERIES, TIMESERIES_MATRIX),
    (ANOMALY, ANOMALY_MATRIX),
)


def matrix(profile):
    return {name: getattr(profile.capabilities, name) for name in FLAGS}


def test_every_capability_is_asserted_here():
    """The matrices below name every flag, not the ones someone remembered.

    Reading the list off the dataclass is what makes that true. A flag added
    to `Capabilities` and not to a matrix fails here, so its permissive
    default cannot reach a profile unnoticed.
    """
    for _, expected in PROFILES:
        assert set(expected) == set(FLAGS)


def test_the_profiles_cover_the_models_we_serve_and_the_ones_we_plan_to():
    """Assert each profile's whole capability matrix, not a sample of it.

    A profile may only reject, so these flags are the entire statement of what
    a model will refuse before anything is sent. Checking two of them would let
    a silent widening through.
    """
    for profile, expected in PROFILES:
        assert matrix(profile) == expected


def test_only_the_anomaly_model_scores_without_a_label():
    """`DETECT ANOMALY` asks for a score the data does not contain."""
    assert ANOMALY.capabilities.anomaly_detection is True
    for profile in (RELATIONAL, TABULAR, TIMESERIES):
        assert profile.capabilities.anomaly_detection is False


def test_only_the_time_series_model_forecasts_a_bare_column():
    """A relational forecast is of an aggregate; a series forecast is not.

    `static` alone would let the relational model take a bare-column forecast,
    which it cannot answer, so the two are separate flags.
    """
    assert TIMESERIES.capabilities.series_forecasting is True
    for profile in (RELATIONAL, TABULAR, ANOMALY):
        assert profile.capabilities.series_forecasting is False
