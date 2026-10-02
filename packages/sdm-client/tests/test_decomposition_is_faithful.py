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

import warnings
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from sdm_client import PredictiveQuery, Schema, StructuredDataError
from sdm_client.decompose import Decomposition

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
    pytest.importorskip('sdm_relational_engine')
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


def test_a_decimal_candidate_is_written_as_the_number_its_column_is():
    """A warehouse id arrives as a `Decimal`, and quoted it asks after a row
    named `"1"` on a column of integers, which the validator refuses."""
    from sdm_client.decompose import _as_literal

    assert _as_literal(Decimal(10)) == '10'
    assert _as_literal(Decimal('10.00')) == '10'


def test_a_decimal_too_wide_for_int64_is_still_written_as_its_digits():
    """The schema layer types such a column as text, so the key is text too,
    and the digits have to survive `str`'s switch to scientific notation."""
    from sdm_client.decompose import _as_literal

    assert _as_literal(Decimal(10) ** 30) == '"1000000000000000000000000000000"'


def test_a_ranking_over_decimal_candidates_materializes_like_an_int_one():
    pytest.importorskip('sdm_relational_engine')
    text = (
        'PREDICT LIST_DISTINCT(buys.iid, 0, 30, days) RANK TOP 2 '
        'FOR EACH users.uid'
    )
    integers = PredictiveQuery(text, shop()).materialize(
        indices=[1, 2, 3], candidates=[10, 11, 12]
    )
    decimals = PredictiveQuery(text, shop()).materialize(
        indices=[1, 2, 3], candidates=[Decimal(10), Decimal(11), Decimal(12)]
    )
    assert [part for part, _ in decimals.decomposition.parts] == [
        part for part, _ in integers.decomposition.parts
    ]


@pytest.mark.parametrize('missing', [None, float('nan'), pd.NA, pd.NaT])
def test_a_missing_candidate_is_refused_however_it_is_spelled(missing):
    """A nullable or Arrow-backed column yields `pd.NA` rather than `None`,
    and one written into the query spends a call asking after a row named
    `<NA>`."""
    from sdm_client.decompose import _as_literal

    with pytest.raises(StructuredDataError, match='missing value'):
        _as_literal(missing)


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
    pytest.importorskip('sdm_relational_engine')
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
    what to do instead, so the test pins that the refusal is the one that
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
    pytest.importorskip('sdm_relational_engine')
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
    pytest.importorskip('sdm_relational_engine')
    request = PredictiveQuery(
        'PREDICT COUNT(buys.*, 0, 7, days) FORECAST 3 TIMEFRAMES '
        'FOR EACH users.uid',
        shop(),
    ).materialize(indices=[1, 2, 3])
    assert [step for _, step in request.decomposition.parts] == [1, 2, 3]


def test_each_step_moves_the_window_by_its_own_span():
    pytest.importorskip('sdm_relational_engine')
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
    pytest.importorskip('sdm_relational_engine')

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


def test_candidates_are_refused_on_a_forecast_rather_than_discarded():
    """A forecast ranks nothing, so candidates describe nothing about it. They
    were accepted and dropped, while `top_k`, the sibling ranking-only
    option, was refused."""
    with pytest.raises(StructuredDataError, match='does not rank anything'):
        PredictiveQuery(
            'PREDICT COUNT(buys.*, 0, 7, days) FORECAST 2 TIMEFRAMES '
            'FOR EACH users.uid',
            shop(),
        ).materialize(indices=[1, 2, 3], candidates=['not', 'a', 'candidate'])


def test_the_candidates_refusal_says_what_to_ask_instead_of_a_kind():
    """It named the query's kind, which is unset for every query that reaches
    the refusal, so it always ended "it is a none query"."""
    with pytest.raises(StructuredDataError, match='RANK TOP k') as refusal:
        PredictiveQuery(
            'PREDICT COUNT(buys.*, 0, 30, days) > 0 FOR EACH users.uid',
            shop(),
        ).materialize(indices=[1], candidates=[10, 11])
    assert 'none query' not in str(refusal.value)


def test_a_repeated_candidate_is_asked_about_once():
    pytest.importorskip('sdm_relational_engine')
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
    pytest.importorskip('sdm_relational_engine')
    parts = parts_of_query(
        'PREDICT COUNT(buys.*, 0, 7, days) FORECAST 2 TIMEFRAMES '
        'FOR EACH users.uid WHERE users.seg="a"',
        indices=[1, 2, 3],
    )
    assert len(parts) == 2
    for part in parts:
        assert 'users.seg = "a"' in (part.query or '')


def test_an_entity_filter_survives_into_every_candidate():
    pytest.importorskip('sdm_relational_engine')
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
    pytest.importorskip('sdm_relational_engine')
    request = PredictiveQuery(
        'PREDICT LIST_DISTINCT(buys.iid, 0, 30, days) RANK TOP 3 '
        'FOR EACH users.uid',
        shop(),
    ).materialize(indices=[1, 2, 3], candidates=[10, 11, 12], top_k=1)
    assert request.decomposition.top == 1


def test_the_number_written_in_the_query_is_used_when_nothing_overrides_it():
    pytest.importorskip('sdm_relational_engine')
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
    pytest.importorskip('sdm_relational_engine')
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
    pytest.importorskip('sdm_relational_engine')
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
    pytest.importorskip('sdm_relational_engine')
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
    pytest.importorskip('sdm_relational_engine')
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
    pytest.importorskip('sdm_relational_engine')
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
    pytest.importorskip('sdm_relational_engine')
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
    pytest.importorskip('sdm_relational_engine')
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


def wide_keys() -> Schema:
    """A shop whose item key straddles int64, so the column travels as text."""
    rows = 300
    keys = [Decimal(2**64 + i) for i in range(3)]
    users = pd.DataFrame({'uid': [1, 2, 3], 'seg': list('abc')})
    items = pd.DataFrame(
        {
            'iid': pd.Series([*keys, Decimal(0)], dtype=object),
            'kind': list('xyzw'),
        }
    )
    buys = pd.DataFrame(
        {
            'bid': list(range(rows)),
            'uid': [1, 2, 3] * (rows // 3),
            'iid': pd.Series([*keys] * (rows // 3), dtype=object),
            'when': pd.to_datetime('2025-06-01')
            + pd.to_timedelta(range(rows), unit='D'),
        }
    )
    return Schema.from_data(
        {'users': users, 'items': items, 'buys': buys},
        primary_keys={'users': 'uid', 'items': 'iid', 'buys': 'bid'},
        time_columns={'buys': 'when'},
    )


def test_a_candidate_is_written_as_its_column_travels_not_as_its_value():
    """A small key out of a column of wide ones is still a key of that column.

    The column travels as text because its values straddle int64, so a
    candidate written as a bare number asks after something the column does
    not hold, and the query the client wrote is refused by the client.
    """
    pytest.importorskip('sdm_relational_engine')
    schema = wide_keys()
    parts = (
        PredictiveQuery(
            'PREDICT LIST_DISTINCT(buys.iid, 0, 30, days) RANK TOP 2 '
            'FOR EACH users.uid',
            schema,
        )
        .materialize(indices=[1, 2], candidates=[Decimal(2**64), Decimal(0)])
        .parts
    )
    written = [(part.query or '').replace(' ', '') for part in parts]
    assert all('buys.iid="' in text for text in written), written


def test_a_candidate_that_cannot_be_written_without_naming_another_row_is_refused():
    from sdm_client.decompose import _as_literal

    with pytest.raises(StructuredDataError, match='different row'):
        _as_literal(Decimal('1.0000000000000000000001'), 'float64')
    assert _as_literal(Decimal(7), 'int64') == '7'
    assert _as_literal(Decimal(7), 'string') == '"7"'
    assert _as_literal(7, 'string') == '"7"'


# --- a failure in one part says which, and stays what it was ---------------


def parts(kind, many):
    from sdm_client.decompose import Decomposition

    about = list(range(1, many + 1)) if kind == 'forecast' else [10, 11, 12]
    return Decomposition(kind, tuple(('q', one) for one in about[:many]))


def test_a_part_failure_keeps_the_exception_it_was():
    """`predict` documents these types, and wrapping every failure in the base
    class dropped the status and the details a caller branches on."""
    from sdm_client.decompose import about_the_part
    from sdm_client.errors import NimRequestError

    raised = NimRequestError(
        503, code='SERVICE_UNAVAILABLE', message='busy.', details={'k': 'v'}
    )
    said = about_the_part(raised, parts('ranking', 3), 11)

    assert isinstance(said, NimRequestError)
    assert said.status_code == 503
    assert said.details == {'k': 'v'}
    assert said.code == 'SERVICE_UNAVAILABLE'
    assert 'busy.' in str(said)


def test_a_ranking_part_is_not_described_as_a_forecast_step():
    """Its number is a candidate and not an ordinal, its window is the one
    written, and asking for fewer candidates cannot shorten a window."""
    from sdm_client.decompose import about_the_part

    said = str(
        about_the_part(
            StructuredDataError('no.', code='INVALID_REQUEST'),
            parts('ranking', 3),
            11,
        )
    )
    assert 'candidate 11' in said
    assert 'window' not in said
    assert 'step' not in said


def test_a_forecast_part_names_the_step_that_failed():
    """Asking for fewer than the total fails in the same place; the failing
    step is the first number that does not."""
    from sdm_client.decompose import about_the_part

    said = str(
        about_the_part(
            StructuredDataError(
                'too far.',
                code='INVALID_REQUEST',
                details={'window_outran_the_data': True},
            ),
            parts('forecast', 40),
            16,
        )
    )
    assert 'step 16 of 40' in said
    assert 'fewer than 16 steps' in said
    assert 'fewer than 40' not in said


def test_a_forecast_part_failure_that_is_not_about_the_window_says_only_which():
    """An unknown model is not answered by asking for fewer steps, and at step
    1 there are no fewer to ask for."""
    from sdm_client.decompose import about_the_part
    from sdm_client.errors import UnknownModelError

    for raised in (
        UnknownModelError('nope', ['kumo-tabular']),
        StructuredDataError('indices is empty', code='INVALID_REQUEST'),
    ):
        said = str(about_the_part(raised, parts('forecast', 3), 1))
        assert 'This is forecast step 1 of 3.' in said
        assert 'fewer than' not in said
        assert 'window' not in said


def test_a_forecast_failure_unrelated_to_its_window_is_not_told_to_shorten():
    """An unknown option failed at step 1 and was told to ask for fewer than
    1 steps. A window that outruns the data, as the engine words it, is still
    told to."""
    pytest.importorskip('sdm_relational_engine')
    forecast = (
        'PREDICT COUNT(buys.*, 0, 30, days) FORECAST {} TIMEFRAMES '
        'FOR EACH users.uid'
    )
    with pytest.raises(StructuredDataError) as unrelated:
        PredictiveQuery(forecast.format(2), shop()).materialize(
            indices=[1, 2], num_neighbours=[4]
        )
    with pytest.raises(StructuredDataError) as outran:
        PredictiveQuery(forecast.format(40), shop()).materialize(indices=[1, 2])

    assert 'This is forecast step 1 of 2.' in str(unrelated.value)
    assert 'fewer than' not in str(unrelated.value)
    assert 'ask for fewer than 10 steps' in str(outran.value)


# --- a candidate no context can teach is left out, and said to be ----------

ENDPOINT = 'http://127.0.0.1:9'


def an_endpoint_that_scores(requests_mock, *scores):
    """Answer the n-th call with `scores[n]` as how sure it is, for every row.

    The calls carry no candidate, so the order they arrive in is what tells
    them apart: one per candidate asked about, in the order they were named.
    """
    calls = iter(scores)

    def answer(call, _):
        sure = next(calls)
        rows = call.json()['predict']['instance_table']['rows']
        return {
            'object': 'prediction.response',
            'id': 'pred',
            'model': 'kumo-relational',
            'metadata': {},
            'predictions': [
                {
                    'id': str(row[0]),
                    'row_index': position,
                    'prediction': 'true' if sure > 0.5 else 'false',
                    'probabilities': {'true': sure, 'false': 1 - sure},
                }
                for position, row in enumerate(rows)
            ],
        }

    return requests_mock.post(f'{ENDPOINT}/v1/predictions', json=answer)


def ranked(candidates, top=3):
    return PredictiveQuery(
        f'PREDICT LIST_DISTINCT(buys.iid, 0, 30, days) RANK TOP {top} '
        'FOR EACH users.uid',
        shop(),
    ).materialize(indices=[1, 2, 3], candidates=candidates)


def answered(request, **how):
    from sdm_client import KumoRelational, SDMClient
    from sdm_client.core.warnings import SDMWarning

    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter('always')
        with SDMClient(ENDPOINT) as client:
            result = client.predict(request, model=KumoRelational(), **how)
    return result, [
        str(warning.message)
        for warning in said
        if issubclass(warning.category, SDMWarning)
    ]


@pytest.mark.parametrize(
    ('candidates', 'cold', 'left_out'),
    [
        (
            [10, 13, 11, 14],
            (13, 14),
            'candidates 13 and 14 are left out of this ranking',
        ),
        ([10, 13, 11], (13,), 'candidate 13 is left out of this ranking'),
    ],
    ids=['two cold', 'one cold'],
)
def test_a_ranking_leaves_out_the_candidates_no_context_teaches(
    requests_mock, candidates, cold, left_out
):
    """Nobody in the shop buys 13 or 14, so the endpoint would refuse their
    questions. Refusing the whole ranking for them gave back nothing about
    the candidates that could be ranked."""
    pytest.importorskip('sdm_relational_engine')
    route = an_endpoint_that_scores(requests_mock, 0.2, 0.9)

    result, said = answered(ranked(candidates))

    # One call each for 10 and 11, and none for those left out.
    assert route.call_count == 2
    assert result.unscored_candidates == cold
    for item in result.items:
        assert item.prediction == [11, 10]
        assert [entry['candidate'] for entry in item.rankings] == [11, 10]
        assert [entry['score'] for entry in item.rankings] == [0.9, 0.2]
    frame = result.to_pandas()
    assert len(frame) == 3
    named = {c for row in frame['prediction'] for c in row}
    assert named == {10, 11}
    (warned,) = [message for message in said if 'unscored' in message]
    assert left_out in warned
    assert 'no entity in it linked to the candidate' in warned
    assert 'num_context_samples' in warned
    assert '(entity, candidate) pairs' in warned


def test_a_ranking_with_no_candidate_left_is_refused_before_any_call(
    requests_mock,
):
    """With every candidate left out there is nothing to rank, and leaving
    them all out would answer with empty rows that look like an answer."""
    pytest.importorskip('sdm_relational_engine')
    from sdm_client.errors import UnsupportedRequestError

    route = an_endpoint_that_scores(requests_mock)

    with pytest.raises(UnsupportedRequestError) as error:
        answered(ranked([13, 14]))

    assert not route.called
    assert error.value.details['candidates'] == [13, 14]
    said = str(error.value)
    assert 'candidates 13 and 14 each hold one answer only' in said
    assert 'nothing to rank' in said
    assert 'Rank without' not in said


def test_the_unscored_candidates_survive_a_merge_and_are_empty_elsewhere(
    requests_mock,
):
    """A ranking split across calls is merged, and dropping the candidates it
    left out there would make every one of them look ranked and lost."""
    pytest.importorskip('sdm_relational_engine')
    from sdm_client.result import PredictionResult

    an_endpoint_that_scores(requests_mock, 0.2, 0.9, 0.4, 0.5, 0.6)
    first, _ = answered(ranked([10, 13, 11]))
    second, _ = answered(ranked([14, 10, 11]))
    assert first.unscored_candidates == (13,)
    assert second.unscored_candidates == (14,)

    merged = PredictionResult.merge([first, second])
    assert merged.unscored_candidates == (13, 14)
    assert len(merged) == len(first) + len(second)

    # The question a ranking asks of candidate 10, asked on its own.
    ordinary, _ = answered(
        PredictiveQuery(
            'PREDICT COUNT(buys.* WHERE buys.iid=10, 0, 30, days) > 0 '
            'FOR EACH users.uid',
            shop(),
        ).materialize(indices=[1, 2, 3])
    )
    assert len(ordinary) == 3
    assert ordinary.unscored_candidates == ()


def test_an_ordinary_ranking_failure_is_not_told_to_drop_the_candidate():
    from sdm_client.decompose import about_the_part
    from sdm_client.errors import NimRequestError

    said = str(
        about_the_part(
            NimRequestError(503, code='SERVICE_UNAVAILABLE', message='busy.'),
            parts('ranking', 3),
            11,
        )
    )
    assert 'ask without it' not in said


@pytest.mark.parametrize(
    ('message', 'joined'),
    [
        ('no room for the window', 'no room for the window. This is'),
        ('no room for the window.', 'no room for the window. This is'),
        ('is there room?', 'is there room? This is'),
    ],
    ids=['no full stop', 'a full stop', 'a question mark'],
)
def test_what_is_appended_to_a_part_failure_reads_as_a_sentence(
    message, joined
):
    """Not every message this wraps ends in one, and two of them ran together
    without it."""
    from sdm_client.decompose import about_the_part

    said = str(
        about_the_part(
            StructuredDataError(message, code='INVALID_REQUEST'),
            parts('ranking', 3),
            11,
        )
    )
    assert joined in said


@pytest.mark.parametrize(
    'outputs', ['prediction', 123, [1, 2]], ids=['a string', 'an int', 'ints']
)
def test_a_ranking_reports_a_bad_outputs_the_way_every_route_does(outputs):
    """A rewritten question is answered in parts and never reaches `_prepare`,
    so the ranking's own output check saw a string first and refused it one
    character at a time."""
    pytest.importorskip('sdm_relational_engine')

    from sdm_client import KumoRelational, SDMClient

    request = PredictiveQuery(
        'PREDICT LIST_DISTINCT(buys.iid, 0, 30, days) RANK TOP 2 '
        'FOR EACH users.uid',
        shop(),
    ).materialize(indices=[1, 2], candidates=[10, 11])
    client = SDMClient('http://127.0.0.1:1', api_key='none')
    try:
        with pytest.raises(StructuredDataError) as raised:
            client.predict(request, model=KumoRelational(), outputs=outputs)
    finally:
        client.close()

    said = str(raised.value)
    assert 'sequence of field names' in said
    assert "'p', 'r', 'e'" not in said
