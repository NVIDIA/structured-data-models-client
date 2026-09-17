# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A two-valued target asked through a query is a classification.

`PREDICT users.churn FOR EACH users.user_id`, where churn is 0 or 1, was
gathered as a regression: 0 and 1 are integers, integers are numerical, and a
numerical target meant regression. The endpoint answered it -- with continuous
values near 0.5, which read like probabilities and are not. Nothing warned.

Zero-and-one is the most common way there is to write a binary label, so the
derivation is corrected for exactly that case, and only that case. Everything
wider stays a regression, because a target holding 0, 1 and 2 really might be
a count. What settles those is the caller, and `task=` now exists to say so.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from sdm_client import PredictiveQuery, Schema, SDMWarning
from sdm_client.errors import StructuredDataError

QUERY = 'PREDICT people.churn FOR EACH people.person_id'


def frame(target: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'person_id': range(len(target)),
            'age': [20.0 + i for i in range(len(target))],
            'churn': target,
        }
    )


def materialize(target: list[object], **kwargs: object):
    data = frame(target)
    scored = data.head(2).drop(columns=['churn'])
    return PredictiveQuery(QUERY, {'people': data}).materialize(
        predict=scored, **kwargs
    )


class TestTheDerivation:
    def test_a_zero_one_integer_target_is_a_classification(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SDMWarning)
            request = materialize([i % 2 for i in range(40)])
        assert request.task.kind == 'classification'

    def test_it_says_so_rather_than_deciding_quietly(self):
        with pytest.warns(SDMWarning, match='classification'):
            materialize([i % 2 for i in range(40)])

    def test_the_warning_names_the_other_reading_and_the_way_out(self):
        with pytest.warns(SDMWarning) as caught:
            materialize([i % 2 for i in range(40)])
        said = str(caught[0].message)
        assert 'regression' in said, (
            'the warning must name what it did not pick'
        )
        assert 'task=' in said, 'and how to say so'

    @pytest.mark.parametrize('pair', [[0, 1], [1, 2], [-1, 1], [7, 9]])
    def test_any_two_whole_numbers_read_the_same_way(self, pair):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SDMWarning)
            request = materialize([pair[i % 2] for i in range(40)])
        assert request.task.kind == 'classification'

    def test_a_two_valued_float_target_is_left_as_a_regression(self):
        """0.5 and 1.5 are two values and neither is a label."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SDMWarning)
            request = materialize([0.5 if i % 2 else 1.5 for i in range(40)])
        assert request.task.kind == 'regression'

    def test_a_continuous_target_is_still_a_regression(self):
        request = materialize([float(i) for i in range(40)])
        assert request.task.kind == 'regression'

    def test_three_whole_numbers_are_still_a_regression(self):
        """A target holding 0, 1 and 2 may be a count; the caller settles it."""
        request = materialize([i % 3 for i in range(40)])
        assert request.task.kind == 'regression'

    def test_a_categorical_target_is_unaffected(self):
        data = frame(['yes' if i % 2 else 'no' for i in range(40)])
        request = PredictiveQuery(QUERY, {'people': data}).materialize(
            predict=data.head(2).drop(columns=['churn'])
        )
        assert request.task.kind == 'classification'

    def test_nulls_do_not_count_as_a_third_value(self):
        target: list[object] = [i % 2 for i in range(40)]
        target[3] = None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SDMWarning)
            request = materialize(target)
        assert request.task.kind == 'classification'


class TestTheCallerHasTheLastWord:
    def test_task_overrides_a_derivation_that_would_classify(self):
        request = materialize([i % 2 for i in range(40)], task='regression')
        assert request.task.kind == 'regression'

    def test_task_overrides_a_derivation_that_would_regress(self):
        request = materialize(
            [float(i) for i in range(40)], task='classification'
        )
        assert request.task.kind == 'classification'

    def test_stating_it_silences_the_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', SDMWarning)
            materialize([i % 2 for i in range(40)], task='classification')

    def test_a_task_the_model_does_not_serve_is_refused_by_name(self):
        with pytest.raises(StructuredDataError) as raised:
            materialize([i % 2 for i in range(40)], task='forecasting')
        assert 'forecasting' in str(raised.value)


class TestTheWireStillSaysWhatItSaid:
    def test_the_target_dtype_is_untouched_by_the_task_it_implies(self):
        """Only the task kind moves. The column is described as it always was."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SDMWarning)
            request = materialize([i % 2 for i in range(40)])
        assert request.task.target.dtype == 'int64'

    def test_the_schema_still_calls_the_column_numerical(self):
        """The fix is about the question asked, not about retyping the data."""
        data = frame([i % 2 for i in range(40)])
        schema = Schema.from_data({'people': data})
        assert schema['people']['churn'].stype == 'numerical'
