# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from sdm_relational_engine.rfm.base.sampler import Sampler


class _Aggregation:
    def __init__(self, column: str) -> None:
        self._column = column

    def get_target_column_name(self) -> str:
        return self._column


class _Query:
    def __init__(self, *columns: str) -> None:
        self._columns = columns

    def get_all_target_aggregations(self) -> list[_Aggregation]:
        return [_Aggregation(column) for column in self._columns]


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
