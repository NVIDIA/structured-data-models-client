# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A question asked as several questions must still be the question asked.

Ranking and forecasting are served by rewriting them into the tasks the
endpoint does have. The rewrite is only sound if it keeps what the original
asked: the rows it was over, the type of the keys it named, the entity each
answer belongs to, and the numbering the contract gives a step. Each test here
pins one thing the rewrite could quietly drop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdm_client import PredictiveQuery, Schema, StructuredDataError

# --- the graph these questions are asked over ------------------------------


def shop() -> Schema:
    rows = 300
    users = pd.DataFrame({'uid': [1, 2, 3], 'seg': list('abc')})
    items = pd.DataFrame({'iid': [10, 11, 12], 'kind': list('xyz')})
    buys = pd.DataFrame(
        {
            'bid': list(range(rows)),
            'uid': [1, 2, 3] * (rows // 3),
            'iid': [10, 11, 12] * (rows // 3),
            'channel': ['web', 'app'] * (rows // 2),
            'when': pd.to_datetime('2025-06-01')
            + pd.to_timedelta(range(rows), unit='D'),
        }
    )
    return Schema.from_data(
        {'users': users, 'items': items, 'buys': buys},
        primary_keys={'users': 'uid', 'items': 'iid', 'buys': 'bid'},
        time_columns={'buys': 'when'},
    )


def parts_of_query(text, **how):
    return PredictiveQuery(text, shop()).materialize(**how).parts


# --- a candidate keeps the type the caller gave it -------------------------


def test_a_candidate_is_carried_as_the_value_not_as_its_text():
    """`7` must come back `7`, not `'7'`.

    The candidates were once read back out of the query text they had been
    written into, which turned every key into a string and broke the join
    back onto the caller's frame.
    """
    request = PredictiveQuery(
        'PREDICT LIST_DISTINCT(buys.iid, 0, 30, days) RANK TOP 2 '
        'FOR EACH users.uid',
        shop(),
    ).materialize(indices=[1, 2, 3], candidates=[10, 11, 12])
    assert request.about == (10, 11, 12)
    assert all(isinstance(value, int) for value in request.about)


def test_a_numpy_key_past_the_safe_range_is_written_exactly():
    """A key out of a frame is a numpy integer, and a big one loses digits if
    it goes anywhere near a float."""
    from sdm_client.decompose import _as_literal

    big = np.int64(9007199254740993)
    assert _as_literal(big) == '9007199254740993'


@pytest.mark.parametrize(
    'bad',
    ['a"b', "a'b", 'a,b', 'a(b', 'a)b', 'a\\b', '', '   ', None, True, 7.5],
)
def test_a_candidate_that_would_not_read_as_a_key_is_refused(bad):
    """Written into the query unescaped, each of these reads as part of the
    question rather than as the row it names."""
    from sdm_client.decompose import _as_literal

    with pytest.raises(StructuredDataError):
        _as_literal(bad)


def test_a_quote_in_a_candidate_never_reaches_the_query():
    with pytest.raises(StructuredDataError, match='cannot be written'):
        PredictiveQuery(
            'PREDICT LIST_DISTINCT(buys.iid, 0, 30, days) RANK TOP 2 '
            'FOR EACH users.uid',
            shop(),
        ).materialize(indices=[1], candidates=['a" OR "1'])


# --- the rows a question was over are not quietly widened ------------------


def test_a_forecast_keeps_the_filter_its_aggregation_had():
    """`COUNT(buys.* WHERE channel="web", ...)` forecast per step must still
    be about web, not about everything."""
    parts = parts_of_query(
        'PREDICT COUNT(buys.* WHERE buys.channel="web", 0, 7, days) '
        'FORECAST 3 TIMEFRAMES FOR EACH users.uid',
        indices=[1, 2, 3],
    )
    assert len(parts) == 3
    for part in parts:
        assert 'buys.channel="web"' in (part.query or '').replace(' ', '')


def test_a_ranking_over_filtered_rows_is_refused_rather_than_widened():
    """Two filters would have to become one, and a rewrite that dropped
    either would answer a different question without saying so."""
    with pytest.raises(StructuredDataError, match='chosen by a filter'):
        PredictiveQuery(
            'PREDICT LIST_DISTINCT(buys.iid WHERE buys.channel="web", 0, 30, '
            'days) RANK TOP 2 FOR EACH users.uid',
            shop(),
        ).materialize(indices=[1], candidates=[10, 11])


def test_ranking_a_column_that_points_nowhere_never_reaches_the_rewrite():
    """`buys.channel` is a label, not a relationship.

    The rewrite would turn it into a question about which rows an entity
    reaches, which is a different question with a different answer. Nothing
    guards this here because the query language refuses it first, and says
    what to do instead -- so the test pins that the refusal is the one that
    arrives, not that the rewrite declines it.
    """
    with pytest.raises(StructuredDataError, match='foreign key column'):
        PredictiveQuery(
            'PREDICT LIST_DISTINCT(buys.channel, 0, 30, days) RANK TOP 2 '
            'FOR EACH users.uid',
            shop(),
        ).materialize(indices=[1, 2, 3], candidates=['web', 'app'])


@pytest.mark.parametrize('column', ['buys.channel', 'buys.bid', 'buys.uid'])
def test_only_a_link_into_another_table_is_ever_rewritten(column):
    """Every non-link shape is refused before a part is ever written."""
    with pytest.raises(StructuredDataError):
        PredictiveQuery(
            f'PREDICT LIST_DISTINCT({column}, 0, 30, days) RANK TOP 2 '
            f'FOR EACH users.uid',
            shop(),
        ).materialize(indices=[1, 2, 3], candidates=[10, 11])


def test_ranking_a_real_foreign_key_still_works():
    """The guard above must not refuse the case it exists to protect."""
    parts = parts_of_query(
        'PREDICT LIST_DISTINCT(buys.iid, 0, 30, days) RANK TOP 2 '
        'FOR EACH users.uid',
        indices=[1, 2, 3],
        candidates=[10, 11],
    )
    assert len(parts) == 2
    written = [(part.query or '').replace(' ', '') for part in parts]
    assert 'buys.iid=10' in written[0]
    assert 'buys.iid=11' in written[1]


# --- the contract numbers a step from one ---------------------------------


def test_forecast_steps_are_numbered_from_one():
    """`forecast_step` in the contract's own examples is 1, 2, 3."""
    request = PredictiveQuery(
        'PREDICT COUNT(buys.*, 0, 7, days) FORECAST 3 TIMEFRAMES '
        'FOR EACH users.uid',
        shop(),
    ).materialize(indices=[1, 2, 3])
    assert request.about == (1, 2, 3)


def test_each_step_moves_the_window_by_its_own_span():
    parts = parts_of_query(
        'PREDICT COUNT(buys.*, 0, 7, days) FORECAST 3 TIMEFRAMES '
        'FOR EACH users.uid',
        indices=[1, 2, 3],
    )
    windows = [
        (part.query or '').split('buys.*,')[1].split(')')[0] for part in parts
    ]
    assert [w.strip() for w in windows] == [
        '0, 7, days',
        '7, 14, days',
        '14, 21, days',
    ]


# --- what comes back names the row it was asked about ----------------------


def test_a_ranking_answer_keeps_the_row_each_entity_was_asked_on():
    """The merge once numbered rows by the order a dict happened to hold the
    entities, so a result could not be joined back onto the frame scored."""
    from sdm_api.generated.tfm_api import PredictionItem

    from sdm_client.client import _ranked
    from sdm_client.result import PredictionResult

    def answer(scores):
        return PredictionResult(
            model='kumo-relational',
            request_id='r',
            items=[
                PredictionItem(
                    row_index=row,
                    id=entity,
                    probabilities={'true': score, 'false': 1 - score},
                )
                for row, (entity, score) in enumerate(scores, start=5)
            ],
            outputs=('prediction', 'probabilities'),
        )

    merged = _ranked(
        [answer([(1, 0.1), (2, 0.9)]), answer([(1, 0.8), (2, 0.2)])],
        (10, 11),
        'RANK TOP 1',
        None,
    )
    by_id = {item.id: item for item in merged.items}
    assert by_id[1].row_index == 5
    assert by_id[2].row_index == 6
    assert by_id[1].prediction == [11]
    assert by_id[2].prediction == [10]


def test_a_forecast_answer_keeps_the_row_and_adds_the_step():
    """A point in the series is named by the pair, so the row must survive."""

    from sdm_api.generated.tfm_api import PredictionItem

    from sdm_client.client import _by_step
    from sdm_client.result import PredictionResult

    def answer(value):
        return PredictionResult(
            model='kumo-relational',
            request_id='r',
            items=[
                PredictionItem(row_index=row, id=row + 1, prediction=value)
                for row in range(3)
            ],
            outputs=('prediction',),
        )

    request = PredictiveQuery(
        'PREDICT COUNT(buys.*, 0, 7, days) FORECAST 2 TIMEFRAMES '
        'FOR EACH users.uid',
        shop(),
    ).materialize(indices=[1, 2, 3])
    merged = _by_step([answer(1.0), answer(2.0)], request)
    pairs = [(item.row_index, item.forecast_step) for item in merged.items]
    assert pairs == [(0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]


# --- the refusals a caller is most likely to hit ---------------------------


def test_ranking_without_candidates_says_what_to_pass():
    with pytest.raises(StructuredDataError, match='candidates='):
        PredictiveQuery(
            'PREDICT LIST_DISTINCT(buys.iid, 0, 30, days) RANK TOP 2 '
            'FOR EACH users.uid',
            shop(),
        ).materialize(indices=[1])


def test_an_empty_candidate_set_is_refused():
    with pytest.raises(StructuredDataError, match='nothing to rank'):
        PredictiveQuery(
            'PREDICT LIST_DISTINCT(buys.iid, 0, 30, days) RANK TOP 2 '
            'FOR EACH users.uid',
            shop(),
        ).materialize(indices=[1], candidates=[])


def test_candidates_are_refused_on_a_question_that_ranks_nothing():
    with pytest.raises(StructuredDataError):
        PredictiveQuery(
            'PREDICT COUNT(buys.*, 0, 30, days) > 0 FOR EACH users.uid',
            shop(),
        ).materialize(indices=[1], candidates=[10, 11])


def test_a_repeated_candidate_is_asked_about_once():
    parts = parts_of_query(
        'PREDICT LIST_DISTINCT(buys.iid, 0, 30, days) RANK TOP 2 '
        'FOR EACH users.uid',
        indices=[1, 2, 3],
        candidates=[10, 10, 11, 10],
    )
    assert len(parts) == 2
