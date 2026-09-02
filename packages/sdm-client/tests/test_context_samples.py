# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``num_context_samples`` replaced ``run_mode``, and has to be worth more.

``run_mode`` was a four-value enum over a two-value server setting: ``debug``
silently gave you ``fast``, and ``normal`` silently gave you ``best``. Two of
the four words did nothing, which is why it went.

Replacing it with a number is only an improvement if the number is honoured.
Deriving the wire value from it and then gathering whatever the bucket allows
would repeat the same defect in a new shape -- ask for 500, get 4,000 -- so
these cover both halves: which value goes on the wire, and how many examples
are actually gathered.
"""

from __future__ import annotations

import pytest

from sdm_client.errors import StructuredDataError
from sdm_client.models import RELATIONAL_LIMITS
from sdm_client.query import (
    _MATERIALIZE_OPTIONS,
    _narrow_context,
    _run_mode_for,
)

FAST = RELATIONAL_LIMITS['context_rows_fast']
MAX = RELATIONAL_LIMITS['context_rows_max']


class _Task:
    """A task table, as far as narrowing is concerned."""

    def __init__(self, examples: int) -> None:
        self.num_context_examples = examples
        self.narrowed_to: int | None = None

    def narrow_context(self, start: int, length: int) -> _Task:
        assert start == 0, 'narrowing should keep the first examples'
        narrowed = _Task(length)
        narrowed.narrowed_to = length
        return narrowed


# --- which value goes on the wire -----------------------------------------


@pytest.mark.parametrize(
    ('asked', 'expected'),
    [
        (None, 'fast'),
        (1, 'fast'),
        (FAST - 1, 'fast'),
        (FAST, 'fast'),
        (FAST + 1, 'best'),
        (MAX - 1, 'best'),
        (MAX, 'best'),
    ],
)
def test_the_wire_value_is_the_one_that_admits_the_count(asked, expected):
    """The endpoint has two settings; the count decides which one."""
    assert _run_mode_for(asked) == expected


def test_asking_for_more_than_the_endpoint_allows_names_the_ceiling():
    with pytest.raises(StructuredDataError) as error:
        _run_mode_for(MAX + 1)
    message = str(error.value)
    assert '10,000' in message
    assert error.value.code == 'UNSUPPORTED_REQUEST'


def test_the_boundary_is_at_the_documented_row_counts():
    """A pin: these are the two numbers the endpoint actually distinguishes."""
    assert FAST == 1_000
    assert MAX == 10_000


# --- how many examples are actually gathered ------------------------------


def test_the_count_narrows_what_was_gathered():
    """Otherwise the number is decorative, which is the defect it replaced."""
    narrowed = _narrow_context(_Task(4_000), 500)
    assert narrowed.num_context_examples == 500
    assert narrowed.narrowed_to == 500


def test_asking_for_more_than_exists_is_not_an_error():
    """There simply are not that many labelled examples; that is not a fault."""
    task = _Task(240)
    assert _narrow_context(task, 1_000) is task
    assert task.num_context_examples == 240


def test_asking_for_exactly_what_exists_changes_nothing():
    task = _Task(240)
    assert _narrow_context(task, 240) is task


def test_asking_for_nothing_in_particular_keeps_everything():
    """The default gathers what the data holds, up to the endpoint's ceiling."""
    task = _Task(4_000)
    assert _narrow_context(task, None) is task
    assert task.num_context_examples == 4_000


# --- run_mode is gone, and stays gone --------------------------------------


def test_run_mode_is_not_an_accepted_option():
    assert 'run_mode' not in _MATERIALIZE_OPTIONS


def test_num_context_samples_is_not_smuggled_in_as_a_sampler_option():
    """It is an explicit parameter, so it must not also be a passthrough."""
    assert 'num_context_samples' not in _MATERIALIZE_OPTIONS


def test_the_public_surface_offers_no_run_mode():
    import inspect

    from sdm_client import PredictiveQuery

    parameters = inspect.signature(PredictiveQuery.materialize).parameters
    assert 'run_mode' not in parameters
    assert 'num_context_samples' in parameters
