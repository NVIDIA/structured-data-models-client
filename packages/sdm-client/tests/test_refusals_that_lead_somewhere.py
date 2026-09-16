# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refusals that name the real reason, and advice that can be followed.

A tabular model cannot answer a windowed aggregation on any schema, but the
capability check ran after resolution and so never ran at all: the caller got
the resolver's complaint that there is no foreign key from a table to itself,
did what it said, and was refused twice more without the word "time" appearing.

A graph with no time column cannot express an aggregation, which is a real
limit. The remediation was the defect: the refusal named `LIST_DISTINCT`
combined with `RANK` as the one static aggregation, and the model refuses that
too, so the second message sent the caller back to the first.

And a `LIST_DISTINCT` the model does not serve resolved cleanly and died inside
the engine as `INTERNAL_ERROR`, which tells a caller the SDK broke when the
answer is that this model does not serve that kind of task.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sdm_client import (
    KumoRelational,
    KumoTabular,
    PredictiveQuery,
    Relationship,
    Schema,
)
from sdm_client.errors import StructuredDataError

RNG = np.random.default_rng(4)


def single_table() -> Schema:
    events = pd.DataFrame(
        {
            'event_id': range(400),
            'amount': RNG.gamma(2.0, 20.0, 400).round(2),
            'cat': RNG.choice(list('abc'), 400),
            'when': pd.to_datetime('2023-01-01')
            + pd.to_timedelta(RNG.integers(0, 200, 400), 'D'),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data(
            {'events': events}, primary_keys={'events': 'event_id'}
        )


def graph(*, timed: bool, linked: bool = True) -> Schema:
    """A graph of users, movies and ratings, with or without time and links.

    Asked for without time, the column is dropped rather than left undeclared:
    inference finds a timestamp and the refusals rightly point at it, which is
    a different situation from a graph with no time in it at all.
    """
    users = pd.DataFrame({'USERID': range(60), 'AGE': RNG.integers(18, 70, 60)})
    movies = pd.DataFrame(
        {'MOVIEID': range(40), 'GENRE': RNG.choice(list('pqr'), 40)}
    )
    ratings = pd.DataFrame(
        {
            'RID': range(900),
            'USERID': RNG.integers(0, 60, 900),
            'MOVIEID': RNG.integers(0, 40, 900),
            'TAG': RNG.choice(list('klm'), 900),
            'STARS': RNG.integers(1, 6, 900),
            'when': pd.to_datetime('2023-01-01')
            + pd.to_timedelta(RNG.integers(0, 200, 900), 'D'),
        }
    )
    if not timed:
        ratings = ratings.drop(columns=['when'])
    tables = {'users': users, 'ratings': ratings}
    links = [Relationship('ratings', 'USERID', 'users', 'USERID')]
    keys = {'users': 'USERID', 'ratings': 'RID'}
    if linked:
        tables['movies'] = movies
        links.append(Relationship('ratings', 'MOVIEID', 'movies', 'MOVIEID'))
        keys['movies'] = 'MOVIEID'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data(
            tables,
            primary_keys=keys,
            time_columns={'ratings': 'when'} if timed else {},
            relationships=links,
        )


def refuse(query: str, schema: Schema, model) -> str:
    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery(query, schema).materialize(indices=[1, 2], model=model)
    assert raised.value.code == 'INVALID_REQUEST', raised.value.code
    return str(raised.value)


def answer(query: str, schema: Schema, model) -> str:
    return (
        PredictiveQuery(query, schema)
        .materialize(indices=[1, 2], model=model)
        .task.kind
    )


# --- a model that cannot answer this shape says so -------------------------


def test_a_windowed_target_on_a_tabular_model_says_time():
    """It used to say there is no foreign key from a table to itself."""
    message = refuse(
        'PREDICT AVG(events.amount, 0, 7, days) FOR EACH events.event_id',
        single_table(),
        KumoTabular(),
    )
    assert 'target carries a time range' in message
    assert 'foreign key' not in message


def test_the_refusal_says_no_schema_would_help():
    """The loop was three refusals long because each one described something
    the caller could change."""
    message = refuse(
        'PREDICT AVG(events.amount, 0, 7, days) FOR EACH events.event_id',
        single_table(),
        KumoTabular(),
    )
    assert 'No schema makes this answerable' in message


@pytest.mark.parametrize(
    'query',
    [
        'PREDICT events.amount FOR EACH events.event_id',
        'PREDICT events.cat FOR EACH events.event_id',
    ],
)
def test_a_tabular_query_that_is_answerable_is_untouched(query):
    assert answer(query, single_table(), KumoTabular()) in {
        'regression',
        'classification',
    }


# --- a task the model does not serve is refused, not crashed ---------------


def test_a_list_distinct_the_model_cannot_serve_is_refused():
    """It resolved with no errors, reached the engine, and came back as
    `INTERNAL_ERROR: NotImplementedError`."""
    message = refuse(
        'PREDICT LIST_DISTINCT(ratings.TAG, 0, 30, days) FOR EACH users.USERID',
        graph(timed=True),
        KumoRelational(),
    )
    assert 'multilabel' in message
    assert 'foreign key column' in message


def test_the_same_query_ranked_is_refused_the_same_way():
    message = refuse(
        'PREDICT LIST_DISTINCT(ratings.TAG, 0, 30, days) RANK TOP 5 '
        'FOR EACH users.USERID',
        graph(timed=True),
        KumoRelational(),
    )
    assert 'multilabel' in message


def test_a_column_with_no_table_to_point_at_is_refused():
    message = refuse(
        'PREDICT LIST_DISTINCT(ratings.MOVIEID, 0, 30, days) '
        'FOR EACH users.USERID',
        graph(timed=True, linked=False),
        KumoRelational(),
    )
    assert 'multilabel' in message


@pytest.mark.parametrize(
    'query',
    [
        'PREDICT LIST_DISTINCT(ratings.MOVIEID, 0, 30, days) '
        'RANK TOP 5 FOR EACH users.USERID',
        'PREDICT LIST_DISTINCT(ratings.MOVIEID, 0, 30, days) '
        'FOR EACH users.USERID',
    ],
    ids=['ranked', 'plain'],
)
def test_a_foreign_key_list_distinct_still_works(query):
    """Both forms resolve to link prediction, which the model does serve. The
    refusal must not widen onto them."""
    pytest.importorskip('sdm_relational_engine')

    assert answer(query, graph(timed=True), KumoRelational()) == (
        'temporal_link_prediction'
    )


# --- advice on a graph with no time column ---------------------------------


def test_a_static_aggregation_is_not_sent_to_list_distinct():
    """The one static aggregation the language has is refused by this model,
    so naming it sent the caller to a query that refuses and names the first
    one back."""
    message = refuse(
        'PREDICT COUNT(ratings.*) FOR EACH users.USERID',
        graph(timed=False),
        KumoRelational(),
    )
    assert 'missing a time range' in message
    assert 'LIST_DISTINCT combined with RANK' not in message
    assert 'no static aggregation at all' in message


def test_the_advice_names_what_a_time_range_needs():
    message = refuse(
        'PREDICT COUNT(ratings.*) FOR EACH users.USERID',
        graph(timed=False),
        KumoRelational(),
    )
    assert '(0, 30, days)' in message
    assert 'time column on the aggregated table' in message


@pytest.mark.parametrize(
    'query',
    [
        'PREDICT LIST_DISTINCT(ratings.MOVIEID) RANK TOP 5 '
        'FOR EACH users.USERID',
        'PREDICT LIST_DISTINCT(ratings.MOVIEID) FOR EACH users.USERID',
    ],
    ids=['ranked', 'plain'],
)
def test_a_static_list_distinct_is_refused_not_asserted(query):
    """The plain form reached the engine and raised an `AssertionError`."""
    message = refuse(query, graph(timed=False), KumoRelational())
    assert 'without a time range is not supported' in message


def test_the_advice_the_static_refusal_gives_works_when_time_exists():
    """The whole point of changing the wording: adding a time range is a query
    that answers."""
    pytest.importorskip('sdm_relational_engine')

    assert (
        answer(
            'PREDICT COUNT(ratings.*, 0, 30, days) FOR EACH users.USERID',
            graph(timed=True),
            KumoRelational(),
        )
        == 'regression'
    )


def test_training_mode_still_names_the_static_aggregation_it_has():
    """The sentence is true when a model is being trained, and only the
    foundation model refuses it."""
    from predictive_query.profiles import Mode
    from predictive_query.validator.time_range_validator import (
        TimeRangeValidator,
    )

    assert (
        'LIST_DISTINCT combined with RANK'
        in TimeRangeValidator(None, Mode.TRAINING)._static_advice()
    )
    assert (
        'no static aggregation at all'
        in TimeRangeValidator(None, Mode.INFERENCE)._static_advice()
    )


# --- advice that renders to a query the grammar refuses --------------------


def test_a_wildcard_aggregation_is_not_told_to_drop_the_aggregation():
    """The advice was to ask for the target column by name, which renders
    `PREDICT ratings.*` -- and a wildcard is only legal inside COUNT, so the
    remedy for one refusal was a second refusal."""
    message = refuse(
        'PREDICT COUNT(ratings.*, 0, 30, days) FOR EACH users.USERID',
        graph(timed=False),
        KumoRelational(),
    )
    assert 'PREDICT ratings.*' not in message
    assert 'wildcard is only legal inside COUNT' in message
    assert 'column of the entity table' in message


def test_a_named_aggregation_is_pointed_at_the_entity_table():
    """Dropping the aggregation leaves a column of a table the entity points
    at, which has many values per entity and is refused for that instead."""
    message = refuse(
        'PREDICT SUM(ratings.STARS, 0, 30, days) FOR EACH users.USERID',
        graph(timed=False),
        KumoRelational(),
    )
    assert 'PREDICT ratings.STARS' not in message
    assert 'column of the entity table' in message


def test_the_advice_that_message_gives_is_a_query_that_answers():
    pytest.importorskip('sdm_relational_engine')

    assert (
        answer(
            'PREDICT users.AGE FOR EACH users.USERID',
            graph(timed=False),
            KumoRelational(),
        )
        == 'regression'
    )


def test_training_mode_keeps_the_static_rewrite_it_can_use():
    from predictive_query.profiles import Mode
    from predictive_query.validator.time_range_validator import (
        TimeRangeValidator,
    )

    training = TimeRangeValidator(None, Mode.TRAINING)
    assert 'PREDICT ratings.STARS' in training._instead_of_aggregating(
        'ratings.STARS'
    )
    inference = TimeRangeValidator(None, Mode.INFERENCE)
    assert 'PREDICT ratings.STARS' not in inference._instead_of_aggregating(
        'ratings.STARS'
    )
    assert 'wildcard' in training._instead_of_aggregating('ratings.*')


# --- every reason at once, not one per round -------------------------------


def test_a_static_ranked_query_over_the_cap_reports_both_reasons():
    """Reporting only the first sends a caller to fix it and be refused again
    for the second, which is the shape of defect this file is about."""
    message = refuse(
        'PREDICT LIST_DISTINCT(ratings.MOVIEID) RANK TOP 100 '
        'FOR EACH users.USERID',
        graph(timed=False),
        KumoRelational(),
    )
    assert 'without a time range is not supported' in message
    assert 'exceeds maximum' in message


def test_the_cap_alone_is_still_reported_alone():
    message = refuse(
        'PREDICT LIST_DISTINCT(ratings.MOVIEID, 0, 30, days) RANK TOP 100 '
        'FOR EACH users.USERID',
        graph(timed=True),
        KumoRelational(),
    )
    assert 'exceeds maximum' in message
    assert 'without a time range' not in message
