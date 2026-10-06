# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A string column's meaning moved with the number of rows handed in.

`Schema.from_data` calls a string column `categorical` below a fixed count of
distinct values and `text` above it, and counts them in the frame it is given.
For a column whose values repeat, that is the same answer at any size. For one
whose values do not, the count measures the frame, so a request written against
`df.head(100)` describes the column one way and the full table another, with
nothing said in between.

The signal is not nearness to the threshold: 53 distinct in 100 rows is not
near 64 by any measure and still flips. It is whether the values ran out
before the rows did.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sdm_client import Schema
from sdm_client.schema import _warn_stype_was_decided_by_the_rows_read


def said_about(frame: pd.DataFrame) -> str:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        Schema.from_data({'b': frame})
    return ' '.join(str(w.message) for w in caught)


def test_a_column_whose_values_never_repeat_is_reported():
    said = said_about(pd.DataFrame({'city': [f'c{i}' for i in range(64)]}))

    assert 'holds 64 distinct values in the 64 rows that have one' in said
    assert 'the rows ran out before the values did' in said
    assert "schema['b']['city'].stype = 'categorical'" in said


def test_the_case_that_nearness_to_the_threshold_would_miss():
    """53 of 100 is not near 64, and reading more rows still moves it."""
    said = said_about(
        pd.DataFrame({'city': [f'c{i % 53}' for i in range(100)]})
    )

    assert 'holds 53 distinct values in the 100 rows that have one' in said


@pytest.mark.parametrize(
    ('label', 'distinct', 'rows'),
    [
        ('a state column', 27, 150_000),
        ('a repeating code', 30, 100),
        ('a two-valued label', 2, 4),
    ],
)
def test_a_column_whose_values_repeat_says_nothing(label, distinct, rows):
    """The count is a property of the column here, not of the frame, so it
    does not move and there is nothing to report."""
    values = np.resize([f'v{i}' for i in range(distinct)], rows)
    said = said_about(pd.DataFrame({'v': values}))

    assert 'rows ran out' not in said


def test_a_column_already_over_the_threshold_says_nothing():
    """It is text at this size and text at any larger one: reading more rows
    cannot bring a count back down."""
    said = said_about(pd.DataFrame({'city': [f'c{i}' for i in range(500)]}))

    assert 'rows ran out' not in said


def test_declaring_the_stype_settles_it():
    frame = pd.DataFrame({'city': [f'c{i}' for i in range(40)]})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'b': frame})
    schema['b']['city'].stype = 'categorical'

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _warn_stype_was_decided_by_the_rows_read(schema.tables)

    assert not [w for w in caught if 'rows ran out' in str(w.message)]


def test_a_numeric_column_is_not_reported():
    """The cardinality rule decides string columns only; a numeric column is
    numerical at every size and has nothing to flip."""
    said = said_about(pd.DataFrame({'v': [float(i) for i in range(64)]}))

    assert 'rows ran out' not in said


def test_a_boolean_column_is_not_reported():
    """A bool is categorical by its type rather than by a count, and says
    nothing. Two values never come within reach of the limit, so this does not
    reach the dtype precondition and does not claim to cover it."""
    said = said_about(pd.DataFrame({'flag': [i % 2 == 0 for i in range(64)]}))

    assert 'rows ran out' not in said


def test_nulls_do_not_dilute_the_measure():
    """The distinct count is taken over rows that have a value, so the share
    must be too. Forty values among a hundred and sixty nulls has saturated
    nothing, and dividing by the height of the frame said it had."""
    said = said_about(
        pd.DataFrame({'city': [f'c{i}' for i in range(40)] + [None] * 160})
    )

    assert 'holds 40 distinct values in the 40 rows that have one' in said


def test_an_all_null_column_is_not_reported():
    """A column with no values has none to have run out of. Held as behaviour
    rather than as a guard: no value count can put it within reach."""
    said = said_about(pd.DataFrame({'city': [None] * 100}))

    assert 'rows ran out' not in said


def test_an_empty_frame_is_not_reported():
    said = said_about(pd.DataFrame({'city': pd.Series([], dtype=object)}))

    assert 'rows ran out' not in said


@pytest.mark.parametrize('rows', [2, 4, 12])
def test_a_small_frame_of_few_values_says_nothing(rows):
    """Every value distinct in a handful of rows is unsaturated by the first
    measure and nowhere near the threshold by the second. Reporting it would
    put the warning on every fixture and demo frame in existence, which is how
    a true warning stops being read."""
    said = said_about(pd.DataFrame({'name': [f'n{i}' for i in range(rows)]}))

    assert 'rows ran out' not in said


def test_both_measures_are_required():
    """Neither alone is the signal: a saturated column near the threshold does
    not move, and an unsaturated one far below it has nowhere to move to."""
    near_but_saturated = np.resize([f'v{i}' for i in range(60)], 100_000)
    assert 'rows ran out' not in said_about(
        pd.DataFrame({'v': near_but_saturated})
    )

    unsaturated_but_far = [f'v{i}' for i in range(20)]
    assert 'rows ran out' not in said_about(
        pd.DataFrame({'v': unsaturated_but_far})
    )
