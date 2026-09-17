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
from decimal import Decimal

import pandas as pd
import pytest

from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMWarning
from sdm_client.adapters.tabular import TabularAdapter
from sdm_client.errors import StructuredDataError
from sdm_client.materializers.tabular import _holds_two_whole_numbers

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

    def test_a_fractional_pair_is_left_as_a_regression(self):
        """0.5 and 1.5 are two values and neither is a label."""
        request = materialize([0.5 if i % 2 else 1.5 for i in range(40)])
        assert request.task.kind == 'regression'

    def test_whole_numbers_stored_as_floats_read_the_same_way(self):
        """The rule is about the values, not how pandas happens to hold them.

        A 0/1 column with one missing value arrives as ``float64``, and it is
        still a label. Reading 2.0 and 3.0 differently from 2 and 3 would be an
        accident of storage rather than a distinction anyone meant.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SDMWarning)
            request = materialize([2.0 if i % 2 else 3.0 for i in range(40)])
        assert request.task.kind == 'classification'

    def test_a_decimal_that_is_not_quite_whole_is_a_regression(self):
        """Going through a float would round this to 2 and call it a label."""
        request = materialize(
            [
                Decimal('2.0000000000000000001') if i % 2 else Decimal('3')
                for i in range(40)
            ]
        )
        assert request.task.kind == 'regression'

    def test_integers_too_large_to_be_floats_do_not_raise(self):
        """``float()`` overflows on these; the check must not go through one.

        Tested on the predicate rather than end to end, because pandas refuses
        the column first: ``pd.DataFrame`` raises ``OverflowError: int too
        large to convert to float`` while building it, long before any of this
        runs. The predicate is still written not to raise, so that what decides
        is the column's shape and not an accident of arithmetic.
        """
        huge = pd.Series([10**400, 10**401], dtype=object)
        assert _holds_two_whole_numbers(huge) is True

    def test_a_single_valued_target_is_left_alone(self):
        request = materialize([1] * 40)
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

    @pytest.mark.parametrize(
        'bad', [['regression'], {'kind': 'r'}, 42, object()]
    )
    def test_a_task_that_is_not_a_string_is_refused_as_one(self, bad):
        """An unhashable value used to raise TypeError out of a set lookup."""
        with pytest.raises(StructuredDataError):
            materialize([i % 2 for i in range(40)], task=bad)


class TestTheWireStillSaysWhatItSaid:
    @staticmethod
    def _payload(request):
        return next(
            iter(TabularAdapter().prediction_plan(request, ('prediction',)))
        )

    def test_the_target_dtype_is_untouched_by_the_task_it_implies(self):
        """Only the task kind moves. The column is described as it always was."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SDMWarning)
            request = materialize([i % 2 for i in range(40)])
        assert request.task.target.dtype == 'int64'

    def test_the_payload_declares_the_target_categorical_once_it_classifies(
        self,
    ):
        """The adapter already made this correction; it could not, before.

        Checked on the payload rather than the Request, because the Request
        carries the schema a caller may reuse for another question and this
        correction is the adapter's, not the schema's.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SDMWarning)
            request = materialize([i % 2 for i in range(40)])
        columns = self._payload(request)['schema']['instance_table']['columns']
        assert columns['churn'] == {'dtype': 'int64', 'stype': 'categorical'}

    def test_a_regression_target_is_still_declared_numerical(self):
        request = materialize([float(i) for i in range(40)])
        columns = self._payload(request)['schema']['instance_table']['columns']
        assert columns['churn']['stype'] == 'numerical'

    def test_nothing_but_the_target_column_moved(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SDMWarning)
            classified = self._payload(materialize([i % 2 for i in range(40)]))
        regressed = self._payload(
            materialize([i % 2 for i in range(40)], task='regression')
        )
        a = classified['schema']['instance_table']['columns']
        b = regressed['schema']['instance_table']['columns']
        assert {k: v for k, v in a.items() if k != 'churn'} == {
            k: v for k, v in b.items() if k != 'churn'
        }

    def test_the_schema_still_calls_the_column_numerical(self):
        """The fix is about the question asked, not about retyping the data."""
        data = frame([i % 2 for i in range(40)])
        schema = Schema.from_data({'people': data})
        assert schema['people']['churn'].stype == 'numerical'


class TestTheRelationalPathSaysSoRatherThanIgnoringIt:
    """`task=` settles a single-table question, and only a single-table one.

    A relational query does not have its kind derived here at all -- the engine
    resolves the query and states it in the payload. So there is nothing for an
    override to override, and the first version of this accepted the argument
    and dropped it, which is worse than not accepting it.
    """

    @staticmethod
    def graph():
        users = pd.DataFrame(
            {'uid': range(20), 'age': [20.0 + i for i in range(20)]}
        )
        buys = pd.DataFrame(
            {
                'bid': range(400),
                'uid': [i % 20 for i in range(400)],
                'amt': [1.0 + i for i in range(400)],
                'ts': pd.date_range('2026-01-01', periods=400, freq='6h'),
            }
        )
        return Schema.from_data({'users': users, 'buys': buys})

    @staticmethod
    def query(schema):
        return PredictiveQuery(
            'PREDICT COUNT(buys.*, 0, 30, days) FOR EACH users.uid', schema
        )

    def test_a_relational_query_still_materializes_without_one(self):
        pytest.importorskip('sdm_relational_engine')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SDMWarning)
            request = self.query(self.graph()).materialize(
                indices=[0, 1], model=KumoRelational()
            )
        assert request.task.kind == 'regression'

    def test_passing_one_is_refused_and_says_why(self):
        pytest.importorskip('sdm_relational_engine')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SDMWarning)
            with pytest.raises(StructuredDataError) as raised:
                self.query(self.graph()).materialize(
                    indices=[0, 1],
                    model=KumoRelational(),
                    task='classification',
                )
        said = str(raised.value)
        assert 'classification' in said
        assert 'relational' in said
