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
from sdm_client.decompose import Decomposition

# --- the graph these questions are asked over ------------------------------


def shop() -> Schema:
    pytest.importorskip('sdm_relational_engine')
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
    about = [value for _, value in request.decomposition.parts]
    assert about == [10, 11, 12]
    assert all(isinstance(value, int) for value in about)


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
    assert [step for _, step in request.decomposition.parts] == [1, 2, 3]


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
        Decomposition('ranking', (('q10', 10), ('q11', 11)), top=1),
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
    merged = _by_step([answer(1.0), answer(2.0)], request.decomposition)
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


# --- the population asked about is the population answered from ------------


def test_an_entity_filter_survives_into_every_forecast_step():
    """`FOR EACH users.uid WHERE users.seg="a"` chooses which rows become
    labelled examples. A part that drops it is answered from a different
    population than the one asked about, and says nothing about it."""
    parts = parts_of_query(
        'PREDICT COUNT(buys.*, 0, 7, days) FORECAST 2 TIMEFRAMES '
        'FOR EACH users.uid WHERE users.seg="a"',
        indices=[1, 2, 3],
    )
    assert len(parts) == 2
    for part in parts:
        assert 'users.seg = "a"' in (part.query or '')


def test_an_entity_filter_survives_into_every_candidate():
    parts = parts_of_query(
        'PREDICT LIST_DISTINCT(buys.iid, 0, 30, days) RANK TOP 2 '
        'FOR EACH users.uid WHERE users.seg="a"',
        indices=[1, 2, 3],
        candidates=[10, 11],
    )
    assert len(parts) == 2
    for part in parts:
        assert 'users.seg = "a"' in (part.query or '')


# --- what kind of question it is, is not a substring of its text -----------


def test_a_forecast_over_a_table_named_for_ranking_is_still_a_forecast(
    monkeypatch,
):
    """The kind was once read as `'RANK' in text.upper()`, so a table called
    `rank_events` turned a forecast into a ranking: its regression parts were
    asked for probabilities, and the answers were merged as an order.

    The whole answering path is driven here, because recording the right kind
    on the request proves nothing about which kind the merge then uses.
    """
    from sdm_api.generated.tfm_api import PredictionItem

    pytest.importorskip('sdm_relational_engine')

    from sdm_client import SDMClient
    from sdm_client.result import PredictionResult

    events = pd.DataFrame(
        {
            'eid': list(range(300)),
            'uid': [1, 2, 3] * 100,
            'when': pd.to_datetime('2025-06-01')
            + pd.to_timedelta(range(300), unit='D'),
        }
    )
    schema = Schema.from_data(
        {
            'users': pd.DataFrame({'uid': [1, 2, 3], 'seg': list('abc')}),
            'rank_events': events,
        },
        primary_keys={'users': 'uid', 'rank_events': 'eid'},
        time_columns={'rank_events': 'when'},
    )
    request = PredictiveQuery(
        'PREDICT COUNT(rank_events.*, 0, 7, days) FORECAST 2 TIMEFRAMES '
        'FOR EACH users.uid',
        schema,
    ).materialize(indices=[1, 2, 3])
    assert request.decomposition.kind == 'forecast'

    asked_for = []

    def regression(self, part, *, model, outputs=None, **rest):
        asked_for.append(outputs)
        return PredictionResult(
            model='kumo-relational',
            request_id='r',
            items=[
                PredictionItem(row_index=row, id=row + 1, prediction=1.5)
                for row in range(3)
            ],
            outputs=('prediction',),
        )

    monkeypatch.setattr(SDMClient, 'predict', regression)
    client = SDMClient('http://127.0.0.1:9', api_key='none')
    answer = client._answer_in_parts(request, 'kumo-relational', None)

    assert asked_for == [None, None], (
        'a regression part was asked for a probability, which is what the '
        'ranking path does'
    )
    assert [item.forecast_step for item in answer.items] == [1, 1, 1, 2, 2, 2]
    assert all(item.rankings is None for item in answer.items)


# --- how many to keep is what the caller settled on ------------------------


def test_the_top_k_option_overrides_the_number_written_in_the_query():
    """`materialize(top_k=1)` on a query saying `RANK TOP 3` once gave three
    candidates back, because the merge reparsed the original text."""
    request = PredictiveQuery(
        'PREDICT LIST_DISTINCT(buys.iid, 0, 30, days) RANK TOP 3 '
        'FOR EACH users.uid',
        shop(),
    ).materialize(indices=[1, 2, 3], candidates=[10, 11, 12], top_k=1)
    assert request.decomposition.top == 1


def test_the_number_written_in_the_query_is_used_when_nothing_overrides_it():
    request = PredictiveQuery(
        'PREDICT LIST_DISTINCT(buys.iid, 0, 30, days) RANK TOP 2 '
        'FOR EACH users.uid',
        shop(),
    ).materialize(indices=[1, 2, 3], candidates=[10, 11, 12])
    assert request.decomposition.top == 2


def test_a_forecast_refuses_top_k_rather_than_dropping_it():
    """A forecast ranks nothing, so `top_k` describes nothing about it. It was
    quietly discarded, which is how a caller ends up believing an option they
    set had an effect."""
    with pytest.raises(StructuredDataError, match='ranks nothing'):
        PredictiveQuery(
            'PREDICT COUNT(buys.*, 0, 7, days) FORECAST 2 TIMEFRAMES '
            'FOR EACH users.uid',
            shop(),
        ).materialize(indices=[1, 2, 3], top_k=2)


def test_a_part_carries_no_top_k_in_the_task_it_sends():
    """Each part asks one yes-or-no question, and a task that ranks nothing is
    refused by the endpoint for carrying an option it cannot act on:
    `422 does not act on these task options: ['top_k']`. The merge is where
    the number is used, so it must not travel."""
    request = PredictiveQuery(
        'PREDICT LIST_DISTINCT(buys.iid, 0, 30, days) RANK TOP 3 '
        'FOR EACH users.uid',
        shop(),
    ).materialize(indices=[1, 2, 3], candidates=[10, 11], top_k=1)
    assert request.decomposition.top == 1
    for part in request.parts:
        for envelope in part.payloads:
            assert 'top_k' not in envelope['task']


# --- a session pins one context, and this question has several -------------


def test_a_session_refuses_a_rewritten_question(monkeypatch):
    """`fit()` once pinned the first part and answered as though it were the
    whole question: a ranking came back as one candidate's `'true'`."""
    from sdm_client import SDMClient

    client = SDMClient('http://127.0.0.1:9', api_key='none')
    request = PredictiveQuery(
        'PREDICT LIST_DISTINCT(buys.iid, 0, 30, days) RANK TOP 2 '
        'FOR EACH users.uid',
        shop(),
    ).materialize(indices=[1, 2, 3], candidates=[10, 11])
    with pytest.raises(StructuredDataError, match='asked as 2 questions'):
        client.fit(request, model='kumo-relational')


def test_a_session_still_takes_an_ordinary_question():
    """The refusal must not reach past the case it is for. Nothing is sent
    here: the refusal above happens before any call, so reaching the endpoint
    at all is what tells the two apart."""
    from sdm_client import SDMClient

    client = SDMClient('http://127.0.0.1:9', api_key='none', max_retries=0)
    request = PredictiveQuery(
        'PREDICT COUNT(buys.*, 0, 7, days) > 0 FOR EACH users.uid',
        shop(),
    ).materialize(indices=[1, 2, 3])
    assert request.decomposition is None
    with pytest.raises(StructuredDataError) as raised:
        client.fit(request, model='kumo-relational')
    assert 'asked as' not in str(raised.value)


# --- the fields a rewritten answer carries are the ones it builds ----------


def test_a_ranking_refuses_a_field_it_does_not_build():
    """Asking for `quantiles` once gave back a result naming `quantiles` and
    carrying `prediction` and `rankings`."""
    from sdm_client import SDMClient

    client = SDMClient('http://127.0.0.1:9', api_key='none')
    request = PredictiveQuery(
        'PREDICT LIST_DISTINCT(buys.iid, 0, 30, days) RANK TOP 2 '
        'FOR EACH users.uid',
        shop(),
    ).materialize(indices=[1, 2, 3], candidates=[10, 11])
    with pytest.raises(StructuredDataError, match='not among them'):
        client.predict(request, model='kumo-relational', outputs=['quantiles'])


def test_a_ranking_takes_the_fields_it_does_build():
    from sdm_client.client import _refuse_outputs_a_ranking_cannot_give

    for asked in (
        None,
        ['prediction'],
        ['rankings'],
        ['prediction', 'rankings'],
    ):
        _refuse_outputs_a_ranking_cannot_give(asked)


# --- a ranking is the order, so a missing score cannot be read as zero -----


def test_a_candidate_answered_without_a_probability_is_refused():
    """Reading a missing score as 0.0 sorts a candidate the endpoint was
    confident about to the bottom, and gives back an order that looks like an
    answer."""
    from sdm_api.generated.tfm_api import PredictionItem

    from sdm_client.client import _how_sure

    with pytest.raises(StructuredDataError, match='how sure'):
        _how_sure(PredictionItem(row_index=0, id=1, probabilities={}), 10)
    with pytest.raises(StructuredDataError, match='how sure'):
        _how_sure(
            PredictionItem(row_index=0, id=1, probabilities={'yes': 0.9}), 10
        )


@pytest.mark.parametrize('spelling', ['true', 'True', 'TRUE', '1'])
def test_every_spelling_of_the_positive_class_is_read(spelling):
    from sdm_api.generated.tfm_api import PredictionItem

    from sdm_client.client import _how_sure

    item = PredictionItem(row_index=0, id=1, probabilities={spelling: 0.75})
    assert _how_sure(item, 10) == 0.75


def test_a_forecast_asks_each_step_for_the_fields_the_caller_wanted(
    monkeypatch,
):
    """A step is an ordinary regression, so `quantiles` is a field it has.

    The fields were dropped on the way down, so a caller who asked for them
    got a series back with only its predictions and no sign that the rest of
    the request had been ignored.
    """
    from sdm_api.generated.tfm_api import PredictionItem

    from sdm_client import SDMClient
    from sdm_client.result import PredictionResult

    request = PredictiveQuery(
        'PREDICT COUNT(buys.*, 0, 7, days) FORECAST 2 TIMEFRAMES '
        'FOR EACH users.uid',
        shop(),
    ).materialize(indices=[1, 2, 3])

    asked_for = []

    def regression(self, part, *, model, outputs=None, **rest):
        asked_for.append(tuple(outputs) if outputs else None)
        return PredictionResult(
            model='kumo-relational',
            request_id='r',
            items=[PredictionItem(row_index=0, id=1, prediction=1.5)],
            outputs=('prediction', 'quantiles'),
        )

    monkeypatch.setattr(SDMClient, 'predict', regression)
    client = SDMClient('http://127.0.0.1:9', api_key='none')
    answer = client._answer_in_parts(
        request, 'kumo-relational', ['prediction', 'quantiles']
    )

    assert asked_for == [
        ('prediction', 'quantiles'),
        ('prediction', 'quantiles'),
    ]
    assert answer.outputs == ('prediction', 'quantiles')


# --- an entity asked about twice is answered twice -------------------------


def test_the_same_entity_at_two_rows_gets_two_answers():
    """Rows are what a ranking answers, and one entity may sit on several.

    Folding them by entity gave back one row where two were asked, and put
    every candidate into it twice over, so a top-2 could name one candidate
    in both places.
    """
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
                    id=1,
                    probabilities={'true': score, 'false': 1 - score},
                )
                for row, score in enumerate(scores)
            ],
            outputs=('prediction', 'probabilities'),
        )

    merged = _ranked(
        [answer([0.1, 0.9]), answer([0.8, 0.2])],
        Decomposition('ranking', (('q10', 10), ('q11', 11)), top=1),
        None,
    )
    assert len(merged.items) == 2
    by_row = {item.row_index: item for item in merged.items}
    assert by_row[0].prediction == [11]
    assert by_row[1].prediction == [10]
    assert all(item.id == 1 for item in merged.items)


def test_a_candidate_is_never_named_twice_for_one_row():
    from sdm_api.generated.tfm_api import PredictionItem

    from sdm_client.client import _ranked
    from sdm_client.result import PredictionResult

    def answer(score):
        return PredictionResult(
            model='kumo-relational',
            request_id='r',
            items=[
                PredictionItem(
                    row_index=row,
                    id=7,
                    probabilities={'true': score, 'false': 1 - score},
                )
                for row in (0, 1)
            ],
            outputs=('prediction', 'probabilities'),
        )

    merged = _ranked(
        [answer(0.9), answer(0.8)],
        Decomposition('ranking', (('q10', 10), ('q11', 11)), top=2),
        None,
    )
    for item in merged.items:
        assert len(set(item.prediction)) == len(item.prediction)


# --- a ranking builds the fields asked for, and no others ------------------


@pytest.mark.parametrize(
    ('asked', 'has_prediction', 'has_rankings'),
    [
        (None, True, True),
        (['prediction'], True, False),
        (['rankings'], False, True),
        (['prediction', 'rankings'], True, True),
    ],
)
def test_a_ranking_carries_only_the_fields_asked_for(
    asked, has_prediction, has_rankings
):
    """The merge builds these fields itself, so a field not asked for is one
    it should not put in the frame."""
    from sdm_api.generated.tfm_api import PredictionItem

    from sdm_client.client import _ranked
    from sdm_client.result import PredictionResult

    answer = PredictionResult(
        model='kumo-relational',
        request_id='r',
        items=[
            PredictionItem(
                row_index=0, id=1, probabilities={'true': 0.9, 'false': 0.1}
            )
        ],
        outputs=('prediction', 'probabilities'),
    )
    merged = _ranked(
        [answer], Decomposition('ranking', (('q10', 10),), top=1), asked
    )
    item = merged.items[0]
    assert (item.prediction is not None) is has_prediction
    assert (item.rankings is not None) is has_rankings
    columns = list(merged.to_pandas().columns)
    assert ('prediction' in columns) is has_prediction
    assert ('rankings' in columns) is has_rankings


# --- the size reported is the size that will be sent -----------------------


def test_a_rewritten_request_reports_what_all_its_parts_hold():
    """An agent budgeting a call reads these. Reporting the first part's
    counts undercounts by one per candidate, which for a full candidate set
    is two orders of magnitude."""
    request = PredictiveQuery(
        'PREDICT LIST_DISTINCT(buys.iid, 0, 30, days) RANK TOP 2 '
        'FOR EACH users.uid',
        shop(),
    ).materialize(indices=[1, 2, 3], candidates=[10, 11, 12])
    parts = request.parts
    assert len(parts) == 3
    assert len(request.payloads) == sum(len(p.payloads) for p in parts)
    assert request.context_rows == sum(p.context_rows for p in parts)
    assert request.predict_rows == sum(p.predict_rows for p in parts)
    assert request.context_rows > parts[0].context_rows


def test_an_ordinary_request_still_reports_its_own_size():
    request = PredictiveQuery(
        'PREDICT COUNT(buys.*, 0, 7, days) > 0 FOR EACH users.uid',
        shop(),
    ).materialize(indices=[1, 2, 3])
    assert request.parts is None
    assert request.predict_rows == 3
    assert request.context_rows > 0


def test_a_ranking_refuses_an_empty_set_of_fields_like_everything_else():
    """`outputs=[]` says *no fields*, and every other shape refuses it. A
    ranking read it as *the usual*, so the one caller who had said what they
    wanted was the one who did not get it."""
    from sdm_client.client import _refuse_outputs_a_ranking_cannot_give

    with pytest.raises(StructuredDataError, match='at least one field'):
        _refuse_outputs_a_ranking_cannot_give([])
