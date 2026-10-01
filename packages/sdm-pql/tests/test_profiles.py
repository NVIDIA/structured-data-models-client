# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What each model may be asked, as data."""

from __future__ import annotations

FLAGS = (
    'relationships',
    'multiple_tables',
    'link_prediction',
    'forecasting',
    'static',
    'temporal',
    'entity_filters',
    'counterfactuals',
    'derived_targets',
)


def matrix(profile):
    return {name: getattr(profile.capabilities, name) for name in FLAGS}


def test_the_profiles_cover_the_models_we_serve_and_the_ones_we_plan_to():
    """Assert each profile's whole capability matrix, not a sample of it.

    A profile may only reject, so these flags are the entire statement of what
    a model will refuse before anything is sent. Checking two of them would let
    a silent widening through.
    """
    from predictive_query import ANOMALY, RELATIONAL, TABULAR, TIMESERIES

    assert matrix(RELATIONAL) == {
        'relationships': True,
        'multiple_tables': True,
        'link_prediction': True,
        'forecasting': True,
        'static': True,
        'temporal': True,
        'entity_filters': True,
        'counterfactuals': True,
        'derived_targets': True,
    }
    # One table, one column, as it stands. Everything the tabular
    # materializer would otherwise read past is refused here instead.
    assert matrix(TABULAR) == {
        'relationships': False,
        'multiple_tables': False,
        'link_prediction': False,
        'forecasting': False,
        'static': True,
        'temporal': False,
        'entity_filters': False,
        'counterfactuals': False,
        'derived_targets': False,
    }
    # Named here so a future model's arrival is a change to this test rather
    # than a surprise; see the architecture plan, "Adding a model".
    assert matrix(TIMESERIES) == {
        'relationships': False,
        'multiple_tables': False,
        'link_prediction': False,
        'forecasting': True,
        'static': False,
        'temporal': True,
        'entity_filters': True,
        'counterfactuals': True,
        'derived_targets': True,
    }
    assert matrix(ANOMALY) == {
        'relationships': False,
        'multiple_tables': False,
        'link_prediction': False,
        'forecasting': False,
        'static': True,
        'temporal': False,
        'entity_filters': True,
        'counterfactuals': True,
        'derived_targets': True,
    }
