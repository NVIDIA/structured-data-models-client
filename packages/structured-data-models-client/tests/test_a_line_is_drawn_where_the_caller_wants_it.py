# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deciding which scores are anomalous without asking the model again."""

from __future__ import annotations

import numpy as np
import pytest

from sdm_client import StructuredDataError, percentile_thresholds


def test_a_scattered_run_has_to_clear_a_higher_bar():
    """The upper threshold is widened by how scattered the scores are: at
    the same percentile a scattered run reaches it with ordinary rows."""
    rng = np.random.default_rng(0)
    tight = 1.0 + 0.01 * rng.normal(size=1000)
    scattered = 1.0 + 1.0 * rng.normal(size=1000)

    def widening(scores):
        _, upper = percentile_thresholds(scores)
        return upper / np.percentile(scores, 99.0)

    assert widening(scattered) > widening(tight)
    assert widening(tight) == pytest.approx(1.0, abs=0.02)


def test_the_quiet_line_is_never_below_zero():
    """A score is a reconstruction error and cannot be negative, so neither
    can the line beneath which one is ordinary."""
    lower, _ = percentile_thresholds(np.linspace(-5.0, 5.0, 100))

    assert lower == 0.0


def test_a_stricter_percentile_detects_no_more_than_a_looser_one():
    scores = np.concatenate([np.random.default_rng(0).normal(size=500), [40.0]])

    _, loose = percentile_thresholds(scores, upper_percentile=90.0)
    _, strict = percentile_thresholds(scores, upper_percentile=99.9)

    assert strict > loose


@pytest.mark.parametrize(
    ('kwargs', 'said'),
    [
        ({'lower_percentile': -1.0}, 'between 0 and 100'),
        ({'upper_percentile': 101.0}, 'between 0 and 100'),
        ({'lower_percentile': 99.0, 'upper_percentile': 1.0}, 'above'),
    ],
)
def test_percentiles_that_cannot_draw_a_line_are_refused(kwargs, said):
    with pytest.raises(StructuredDataError) as caught:
        percentile_thresholds(np.arange(10.0), **kwargs)

    assert said in str(caught.value)


@pytest.mark.parametrize(
    ('scores', 'said'),
    [
        ([], 'no scores'),
        ([1.0, np.nan, 3.0], 'finite'),
        ([1.0, np.inf], 'finite'),
    ],
)
def test_scores_a_line_cannot_be_found_in_are_refused(scores, said):
    """A threshold worked out from a NaN is NaN, and compares false against
    every score, so the run detects nothing and says nothing went wrong."""
    with pytest.raises(StructuredDataError) as caught:
        percentile_thresholds(scores)

    assert said in str(caught.value)
