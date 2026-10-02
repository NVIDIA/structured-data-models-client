# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from predictive_query.ast import Column, Condition, Constant, Filter
from predictive_query.schema.typing import Dtype, RelOp
from sdm_relational_engine.exceptions import NoEligibleEntityError
from sdm_relational_engine.rfm.base.sampler import Sampler

_WINDOW_PAST_THE_DATA = pd.DateOffset(years=5)


class _TimeRange:
    def __init__(self, end: pd.DateOffset) -> None:
        self.end_date_offset = end


class _Aggregation:
    def __init__(self, column: str, window: pd.DateOffset) -> None:
        self._column = column
        self.aggr_time_range = _TimeRange(window)

    def get_target_column_name(self) -> str:
        return self._column


class _Query:
    def __init__(
        self,
        *columns: str,
        window: pd.DateOffset = _WINDOW_PAST_THE_DATA,
        entity_ast: object = None,
    ) -> None:
        self._columns = columns
        self._window = window
        self.entity_ast = entity_ast

    def get_all_target_aggregations(self) -> list[_Aggregation]:
        return [_Aggregation(column, self._window) for column in self._columns]


def _filter_on(value: str) -> Filter:
    return Filter(
        target=Column(fqn='movies.mid'),
        condition=Condition(
            target=Column(fqn='movies.genres'),
            op=RelOp.EQ,
            value=Constant(value=f"'{value}'", dtype_maybe=Dtype.string),
        ),
    )


class _Sampler:
    r"""Carries only what the diagnosis reads, so it needs no real graph."""

    def __init__(self, max_time) -> None:
        self._max_time = max_time

    def get_max_time(self, table_names=None):
        if self._max_time is RuntimeError:
            raise RuntimeError('no time columns')
        return self._max_time

    def get_min_time(self, table_names=None):
        return pd.Timestamp('1950-05-13')

    _empty_sample_reason = Sampler._empty_sample_reason
    _diagnose_empty_sample = Sampler._diagnose_empty_sample
    _empty_sample_error = Sampler._empty_sample_error
    _no_entity_at_anchor = Sampler._no_entity_at_anchor


LAST_ROW = pd.Timestamp('2023-07-30')


def test_an_anchor_at_the_end_of_the_data_is_named_as_the_cause() -> None:
    reason = _Sampler(LAST_ROW)._empty_sample_reason(
        _Query('results.points'), LAST_ROW
    )

    assert 'too restrictive' not in reason
    assert '2023-07-30' in reason
    assert "'results'" in reason
    assert 'anchor_time' in reason


def test_the_range_the_anchor_sits_against_is_reported() -> None:
    r"""An anchor short of the last row still fails when the window does not
    fit, so the range is stated and the caller judges.
    """
    reason = _Sampler(LAST_ROW)._empty_sample_reason(
        _Query('results.points'), pd.Timestamp('2020-01-01')
    )

    assert '2020-01-01' in reason
    assert '2023-07-30' in reason
    assert '1950-05-13' in reason
    assert 'too restrictive' not in reason


def test_a_per_entity_anchor_keeps_the_generic_question() -> None:
    reason = _Sampler(LAST_ROW)._empty_sample_reason(
        _Query('results.points'), 'entity'
    )

    assert reason == ' Is your predictive query too restrictive?'


def test_an_unresolvable_time_range_keeps_the_generic_question() -> None:
    r"""The diagnosis is an improvement on the message, never a new way to
    fail.
    """
    reason = _Sampler(RuntimeError)._empty_sample_reason(
        _Query('results.points'), LAST_ROW
    )

    assert reason == ' Is your predictive query too restrictive?'


def test_an_anchor_with_room_after_it_is_not_blamed() -> None:
    r"""A window that still ends inside the data rules the anchor out, so the
    filter that selected nothing is named instead of a time the caller cannot
    usefully move.
    """
    reason = _Sampler(LAST_ROW)._empty_sample_reason(
        _Query(
            'results.points',
            window=pd.DateOffset(days=7),
            entity_ast=_filter_on('comedy'),
        ),
        pd.Timestamp('2020-01-01'),
    )

    assert 'most likely the reason' not in reason
    assert 'not the reason' in reason
    assert "movies.genres = 'comedy'" in reason
    assert 'anchor_time' not in reason


def test_no_filter_and_no_anchor_to_blame_asks_the_question() -> None:
    reason = _Sampler(LAST_ROW)._empty_sample_reason(
        _Query('results.points', window=pd.DateOffset(days=7)),
        pd.Timestamp('2020-01-01'),
    )

    assert reason == ' Is your predictive query too restrictive?'


def test_one_table_holds_its_data() -> None:
    reason = _Sampler(LAST_ROW)._empty_sample_reason(
        _Query('results.points'), LAST_ROW
    )

    assert "'results' holds data" in reason


def test_several_tables_hold_their_data() -> None:
    reason = _Sampler(LAST_ROW)._empty_sample_reason(
        _Query('results.points', 'laps.time'), LAST_ROW
    )

    assert "'laps', 'results' hold data" in reason


def test_the_anchor_remedy_narrows_the_window() -> None:
    r"""Widening the window asks for more data after the anchor, which is what
    was missing: only a shorter window fits.
    """
    reason = _Sampler(LAST_ROW)._empty_sample_reason(
        _Query('results.points'), LAST_ROW
    )

    assert 'shorten the window' in reason
    assert 'widen the window' not in reason


def _movie_sampler():
    from sdm_relational_engine.rfm import Graph
    from sdm_relational_engine.rfm.backend.local import LocalSampler, LocalTable

    movies = LocalTable(
        df=pd.DataFrame({'MID': range(10), 'GENRES': ['Comedy'] * 10}),
        name='MOVIES',
        primary_key='MID',
    )
    ratings = LocalTable(
        df=pd.DataFrame(
            {
                'RID': range(40),
                'MID': [mid % 10 for mid in range(40)],
                'STARS': [1.0] * 40,
                'TIME': pd.date_range('2020-01-01', periods=40, freq='D'),
            }
        ),
        name='RATING',
        primary_key='RID',
        time_column='TIME',
    )
    graph = Graph([movies, ratings])
    graph.link('RATING', 'MID', 'MOVIES')
    return LocalSampler(graph, verbose=False), graph._to_api_graph_definition()


def _sample_with_filter(value: str, num_train_trials: int = 10) -> str:
    from sdm_relational_engine.rfm.query.parser import parse_query_locally

    sampler, definition = _movie_sampler()
    query = parse_query_locally(
        f'PREDICT COUNT(RATING.STARS, 0, 7, days) FOR EACH MOVIES.MID '
        f"WHERE MOVIES.GENRES = '{value}'",
        definition,
    )
    try:
        sampler.sample_target(
            query,
            num_train_examples=5,
            train_anchor_time=pd.Timestamp('2020-01-10'),
            num_train_trials=num_train_trials,
            num_test_examples=0,
            test_anchor_time=pd.Timestamp('2020-01-10'),
            num_test_trials=0,
        )
    except ValueError as error:
        return str(error)
    return ''


def test_a_filter_matching_nothing_is_named_rather_than_the_anchor() -> None:
    r"""One character of case is the only difference between these two, and
    the anchor the failing one used to blame is the one the other succeeds on.
    """
    assert _sample_with_filter('Comedy') == ''

    reason = _sample_with_filter('comedy')

    assert "MOVIES.GENRES = 'comedy'" in reason
    assert 'most likely the reason' not in reason


def test_the_anchor_reported_is_the_one_the_caller_passed() -> None:
    r"""A temporal query walks its anchor backwards to find candidates, and a
    message quoting where it stopped shows a time the caller never chose.
    """
    reason = _sample_with_filter('comedy', num_train_trials=50)

    assert '2020-01-10' in reason


class _Started(_Sampler):
    r"""Its entity table ``drivers`` has a time column starting at a time."""

    time_column_dict = {'drivers': 'dob'}

    def __init__(self, earliest) -> None:
        super().__init__(LAST_ROW)
        self._earliest = earliest

    def get_min_time(self, table_names=None):
        if table_names == ['drivers']:
            return self._earliest
        return super().get_min_time(table_names)


def _for_each_driver(**kwargs) -> _Query:
    query = _Query('results.points', **kwargs)
    query.entity_table = 'drivers'
    return query


def test_entities_that_start_after_the_anchor_are_named_not_the_anchor() -> (
    None
):
    r"""No anchor in the data helps when every entity starts after it, so the
    column that puts them there is named instead.
    """
    reason = _Started(pd.Timestamp('2024-01-01'))._empty_sample_reason(
        _for_each_driver(), LAST_ROW
    )

    assert "No row of 'drivers' existed at the anchor" in reason
    assert "the earliest 'dob' is 2024-01-01" in reason
    assert "set the time column of 'drivers' to None" in reason
    assert 'anchor_time' not in reason


def test_an_entity_that_existed_by_the_anchor_leaves_the_anchor_reason() -> (
    None
):
    reason = _Started(pd.Timestamp('1950-01-01'))._empty_sample_reason(
        _for_each_driver(), LAST_ROW
    )

    assert 'most likely the reason' in reason
    assert "'drivers'" not in reason


def test_an_all_null_entity_time_column_is_not_a_start_time() -> None:
    r"""A column with no values reports the latest timestamp there is, which
    is no time any entity started.
    """
    reason = _Started(pd.Timestamp.max)._empty_sample_reason(
        _for_each_driver(), LAST_ROW
    )

    assert 'most likely the reason' in reason
    assert "'drivers'" not in reason


def test_a_later_anchor_is_offered_when_a_window_fits_after_the_first_entity():
    reason = _Started(pd.Timestamp('2023-01-01'))._empty_sample_reason(
        _for_each_driver(window=pd.DateOffset(days=7)),
        pd.Timestamp('2020-01-01'),
    )

    assert "No row of 'drivers' existed at the anchor" in reason
    assert 'or pass an anchor_time after 2023-01-01 00:00:00.' in reason


def test_no_later_anchor_is_offered_when_no_window_fits_after_it() -> None:
    reason = _Started(pd.Timestamp('2023-07-28'))._empty_sample_reason(
        _for_each_driver(window=pd.DateOffset(days=7)),
        pd.Timestamp('2020-01-01'),
    )

    assert "No row of 'drivers' existed at the anchor" in reason
    assert 'anchor_time' not in reason


def test_test_examples_are_not_called_context_examples() -> None:
    r"""The same reasons explain an empty test set, whose examples sit at the
    anchor rather than before it.
    """
    started = _Started(pd.Timestamp('2024-01-01'))._empty_sample_reason(
        _for_each_driver(), LAST_ROW, 'test'
    )
    anchored = _Sampler(LAST_ROW)._empty_sample_reason(
        _Query('results.points'), LAST_ROW, 'test'
    )

    for reason in (started, anchored):
        assert 'Context' not in reason
        assert 'Test examples' in reason
    assert 'are asked at the anchor' in anchored


def test_entities_that_start_after_the_anchor_raise_their_own_error() -> None:
    r"""A caller that reads every empty sample as zero labels would report a
    misconfigured time column as a legitimate answer, so this one arrives as
    its own type.
    """
    error = _Started(pd.Timestamp('2024-01-01'))._empty_sample_error(
        _for_each_driver(),
        LAST_ROW,
        'Failed to collect any test examples for evaluation.',
        'test',
    )

    assert isinstance(error, NoEligibleEntityError)
    assert isinstance(error, ValueError)
    assert "No row of 'drivers' existed at the anchor" in str(error)
    assert str(error).startswith('Failed to collect any test examples')


def test_an_anchor_at_the_end_of_the_data_stays_a_plain_value_error() -> None:
    error = _Sampler(LAST_ROW)._empty_sample_error(
        _Query('results.points'),
        LAST_ROW,
        'Failed to collect any context examples.',
    )

    assert type(error) is ValueError


def test_an_entity_start_that_cannot_be_compared_is_not_a_start() -> None:
    r"""A diagnosis is not worth an error of its own, so a time column whose
    values cannot be compared with the anchor leaves the other reasons to
    explain the empty sample.
    """
    sampler = _Started(pd.Timestamp('2024-01-01', tz='UTC'))

    assert sampler._no_entity_at_anchor(_for_each_driver(), LAST_ROW) is None
    assert 'most likely the reason' in sampler._empty_sample_reason(
        _for_each_driver(), LAST_ROW
    )
