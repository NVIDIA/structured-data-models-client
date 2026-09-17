# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A ranking is only as good as the rows it is a ranking of.

The candidates for link prediction are the rows neighbourhood sampling
reached, which is what an entity has already interacted with. Where an
interaction does not repeat -- where the pair is a key of the fact table --
the rows that will be true are disjoint from that pool by construction, so
recall is zero however good the model is.

Nothing about that is visible from the answer: the query resolves, the request
is sent, and a ranking comes back with plausible scores in descending order,
because the model ranked exactly what it was given.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sdm_client import KumoRelational, PredictiveQuery, Relationship, Schema
from sdm_client.materializers.relational import (
    _how_often_the_answer_was_reachable,
)

RANK = (
    'PREDICT LIST_DISTINCT(buys.iid, 0, 30, days) RANK TOP 5 FOR EACH users.uid'
)
COUNT = 'PREDICT COUNT(buys.*, 0, 30, days) FOR EACH users.uid'


def shop(*, interactions_repeat: bool, seed: int = 0) -> Schema:
    rng = np.random.default_rng(seed)
    users = pd.DataFrame({'uid': range(80), 'seg': rng.choice([*'xy'], 80)})
    items = pd.DataFrame({'iid': range(40), 'cat': rng.choice([*'pqr'], 40)})
    if interactions_repeat:
        pairs = [
            (int(rng.integers(0, 80)), int(rng.integers(0, 12)))
            for _ in range(900)
        ]
    else:
        seen: set[tuple[int, int]] = set()
        while len(seen) < 600:
            seen.add((int(rng.integers(0, 80)), int(rng.integers(0, 40))))
        pairs = sorted(seen)
    buys = pd.DataFrame(
        {
            'bid': range(len(pairs)),
            'uid': [p[0] for p in pairs],
            'iid': [p[1] for p in pairs],
            'ts': pd.to_datetime('2023-01-01')
            + pd.to_timedelta(rng.integers(0, 200, len(pairs)), 'D'),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data(
            {'users': users, 'items': items, 'buys': buys},
            primary_keys={'users': 'uid', 'items': 'iid', 'buys': 'bid'},
            time_columns={'buys': 'ts'},
            relationships=[
                Relationship('buys', 'uid', 'users', 'uid'),
                Relationship('buys', 'iid', 'items', 'iid'),
            ],
        )


def said_materializing(schema: Schema, query: str) -> str:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        PredictiveQuery(query, schema).materialize(
            indices=list(range(20)), model=KumoRelational()
        )
    return ' '.join(
        str(w.message)
        for w in caught
        if 'never going to be right' in str(w.message)
    )


def test_it_is_said_when_no_answer_could_have_been_reached():
    pytest.importorskip('sdm_relational_engine')

    said = said_materializing(shop(interactions_repeat=False), RANK)

    assert 'none of the' in said
    assert 'rows sampled as context had its answer' in said


def test_it_is_quiet_when_the_answers_are_reachable():
    """Interactions that repeat put the answer inside the neighbourhood, and
    there the ranking is over rows that can be right."""
    pytest.importorskip('sdm_relational_engine')

    assert said_materializing(shop(interactions_repeat=True), RANK) == ''


def test_an_aggregation_is_not_a_ranking():
    pytest.importorskip('sdm_relational_engine')

    assert said_materializing(shop(interactions_repeat=False), COUNT) == ''


@pytest.mark.parametrize(
    ('repeat', 'reachable'), [(True, True), (False, False)]
)
def test_what_the_reading_actually_counts(repeat, reachable):
    """The count behind the message, so a change to it is visible as a number
    rather than only as the presence of a sentence."""
    pytest.importorskip('sdm_relational_engine')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = PredictiveQuery(
            RANK, shop(interactions_repeat=repeat)
        ).materialize(indices=list(range(20)), model=KumoRelational())

    found, offered = _how_often_the_answer_was_reachable(request.payloads[0])

    assert offered > 100
    assert (found > 0) is reachable


def test_the_candidates_are_the_ranked_table_not_every_value_beside_it():
    """Reading every value of every related row lets a timestamp or a category
    stand in for a candidate, and a handful of those is enough to report an
    answer as reachable when none of them were."""
    pytest.importorskip('sdm_relational_engine')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = PredictiveQuery(
            RANK, shop(interactions_repeat=False)
        ).materialize(indices=list(range(20)), model=KumoRelational())

    found, offered = _how_often_the_answer_was_reachable(request.payloads[0])

    assert found == 0, (
        f'{found} of {offered} matched something that is not a candidate'
    )


def test_the_request_is_still_built():
    """It is a reading of the context, not a proof about the rows being
    scored, so it is said and not refused."""
    pytest.importorskip('sdm_relational_engine')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = PredictiveQuery(
            RANK, shop(interactions_repeat=False)
        ).materialize(indices=list(range(20)), model=KumoRelational())

    assert request.predict_rows == 20
