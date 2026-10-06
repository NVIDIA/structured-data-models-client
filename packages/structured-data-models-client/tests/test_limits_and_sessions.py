# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the client refuses before spending a round trip, and how sessions behave.

The endpoint enforces several sizes, and a request past one of them comes back
as a 500 or a 422 that names neither the limit nor how to get under it. Every
check here exists so the caller reads a sentence instead.
"""

from __future__ import annotations

import warnings

import pytest

from sdm_client import Schema, UnsupportedRequestError
from sdm_client.adapters.relational import RelationalAdapter
from sdm_client.errors import StructuredDataError
from sdm_client.materializers.relational import RelationalMaterializer
from sdm_client.models import KUMO_RELATIONAL, RELATIONAL_LIMITS


def payload(*, predict_rows: int, context_rows: int) -> dict:
    return {
        'task': {'kind': 'regression', 'target': {'column_name': 't'}},
        'schema': {'instance_table': {'columns': {}}},
        'context': {
            'instance_table': {'rows': [[i] for i in range(context_rows)]}
        },
        'predict': {
            'instance_table': {'rows': [[i] for i in range(predict_rows)]}
        },
    }


class _Gathered:
    """The shape a materialized request presents to an adapter."""

    query: str | None = None

    def __init__(self, *payloads: dict) -> None:
        self.payloads = tuple(payloads)
        self.requested_inference: dict = {}
        self.inference: dict = {}


def test_the_declared_limits_are_pinned_to_what_was_read_off_the_endpoint():
    """A pin, not a probe.

    These numbers were read from the relational NIM. This asserts they have
    not drifted in the client, which is all it can do offline. The live suites
    check that the endpoint still agrees.
    """
    assert RELATIONAL_LIMITS['context_rows_fast'] == 1_000
    assert RELATIONAL_LIMITS['context_rows_max'] == 10_000
    assert RELATIONAL_LIMITS['predict_rows_max'] == 1_000
    assert RELATIONAL_LIMITS['predict_rows_temporal_link_prediction'] == 200
    assert RELATIONAL_LIMITS['explanation_predict_rows'] == 1
    assert RELATIONAL_LIMITS['hops_max'] == 6
    assert RELATIONAL_LIMITS['tables_max'] == 15


@pytest.mark.parametrize('bad', ['2', True])
def test_num_hops_must_be_a_whole_number(bad):
    materializer = RelationalMaterializer(Schema({}))
    with pytest.raises(StructuredDataError) as error:
        materializer._check_shape({'num_hops': bad}, RELATIONAL_LIMITS)
    assert 'num_hops must be a whole number' in str(error.value)
    assert error.value.code == 'INVALID_REQUEST'


def test_an_explanation_covering_more_than_one_entity_is_refused():
    """The model explains one entity at a time; more gets an opaque 500."""
    request = _Gathered(payload(predict_rows=4, context_rows=100))
    with pytest.raises(UnsupportedRequestError) as error:
        list(
            RelationalAdapter().prediction_plan(
                request, ('prediction', 'explanation')
            )
        )
    assert 'one at a time' in str(error.value)


def test_an_explanation_over_the_smaller_context_budget_is_refused():
    """Explaining forces the smaller budget, which caps context at 1,000."""
    request = _Gathered(payload(predict_rows=1, context_rows=5_000))
    with pytest.raises(UnsupportedRequestError) as error:
        list(
            RelationalAdapter().prediction_plan(
                request, ('prediction', 'explanation')
            )
        )
    message = str(error.value)
    assert 'num_context_samples' in message
    assert '1,000' in message


def test_an_explanation_within_both_limits_is_allowed():
    request = _Gathered(payload(predict_rows=1, context_rows=500))
    plans = list(
        RelationalAdapter().prediction_plan(
            request, ('prediction', 'explanation')
        )
    )
    assert len(plans) == 1
    assert plans[0]['output']['fields'] == ['prediction', 'explanation']


def test_a_prediction_without_an_explanation_is_not_held_to_those_limits():
    """Both caps are explanation-only; an ordinary batch is unaffected."""
    request = _Gathered(payload(predict_rows=500, context_rows=5_000))
    plans = list(RelationalAdapter().prediction_plan(request, ('prediction',)))
    assert len(plans) == 1


def test_a_relational_request_that_was_never_gathered_says_how_to_build_one():
    import pandas as pd

    from sdm_client import Request

    direct = Request(
        context=pd.DataFrame({'a': [1, 2], 'y': [0, 1]}),
        predict=pd.DataFrame({'a': [3]}),
        target='y',
        task='regression',
    )
    with pytest.raises(UnsupportedRequestError) as error:
        list(RelationalAdapter().prediction_plan(direct, ('prediction',)))
    assert 'materialize' in str(error.value)


def test_a_session_cannot_pin_more_than_one_gathered_context():
    """Each gathered batch carries its own context; a session pins one."""
    request = _Gathered(
        payload(predict_rows=2, context_rows=10),
        payload(predict_rows=2, context_rows=10),
    )
    with pytest.raises(UnsupportedRequestError) as error:
        RelationalAdapter().session_plan(request, ('prediction',))
    assert 'session pins one context' in str(error.value)


def test_a_single_batch_session_pins_the_context_and_not_the_rows():
    request = _Gathered(payload(predict_rows=2, context_rows=10))
    pinned = RelationalAdapter().session_plan(request, ('prediction',))
    assert 'context' in pinned
    assert 'predict' not in pinned

    scoring = list(
        RelationalAdapter().session_prediction_plan(request, ('prediction',))
    )
    assert 'predict' in scoring[0]
    assert 'context' not in scoring[0]


def test_the_default_outputs_follow_the_relational_task_kind():
    adapter = RelationalAdapter()

    class _Task:
        def __init__(self, kind):
            self.kind = kind

    class _Request:
        def __init__(self, kind):
            self.task = _Task(kind)

    assert adapter.default_output(_Request('regression')) == ('prediction',)
    assert adapter.default_output(_Request('binary_classification')) == (
        'prediction',
        'probabilities',
    )
    assert adapter.default_output(_Request('temporal_link_prediction')) == (
        'rankings',
    )


def test_the_relational_model_declares_the_deployment_it_was_read_from():
    """A second build of the same id serves less; this is the shipping one."""
    runtime = KUMO_RELATIONAL.runtime
    assert {'forecasting', 'temporal_link_prediction'} <= runtime.tasks
    assert 'explanation' in runtime.outputs
    # Only these inference options reach the wire; the engine defaults to more.
    assert 'output_type' not in runtime.inference_options
    assert 'num_estimators' in runtime.inference_options


def test_an_inference_option_the_caller_set_is_named_rather_than_dropped():
    """Dropping the engine's defaults is safe; dropping something the caller
    asked for is not, so that case is named."""
    request = _Gathered(payload(predict_rows=1, context_rows=10))
    request.inference = {'inference_config': {'output_type': 'quantiles'}}
    request.requested_inference = {
        'inference_config': {'output_type': 'quantiles'}
    }
    request.payloads[0]['inference'] = {
        'inference_config': {'output_type': 'quantiles', 'num_estimators': 1}
    }
    with pytest.raises(UnsupportedRequestError) as error:
        list(RelationalAdapter().prediction_plan(request, ('prediction',)))
    assert 'output_type' in str(error.value)
    assert 'prediction_statistic=' in str(error.value)


def test_the_engine_s_statistic_travels_on_the_output_instead():
    """Trimming `output_type` once dropped the median the engine asks for,
    and the endpoint answered with its own default, the mean."""
    request = _Gathered(payload(predict_rows=1, context_rows=10))
    request.payloads[0]['inference'] = {
        'inference_config': {
            'output_type': 'median',
            'target_transforms': ['quantile'],
            'num_estimators': 1,
        }
    }
    (plan,) = RelationalAdapter().prediction_plan(request, ('prediction',))
    assert 'inference_config' not in plan['inference']
    assert plan['output']['prediction_statistic'] == 'median'


def test_the_preflight_runs_through_a_real_materialized_request():
    """The fakes above are tight, but they can drift from the real shape.

    This one builds the request the same way ``materialize`` does, so the
    adapter's preflight and the request type cannot diverge unnoticed.
    """
    import pandas as pd

    from sdm_client import Request, Schema

    schema = Schema.from_data({'users': pd.DataFrame({'user_id': [1, 2]})})
    request = Request.materialized(
        payloads=(payload(predict_rows=3, context_rows=50),),
        schema=schema,
        query='PREDICT ... FOR EACH users.user_id',
        indices=[1, 2, 3],
        entity_table='users',
    )
    assert request.payloads
    assert request.query.startswith('PREDICT')

    plans = list(RelationalAdapter().prediction_plan(request, ('prediction',)))
    assert len(plans) == 1
    assert plans[0]['model'] == 'kumo-relational'

    with pytest.raises(UnsupportedRequestError):
        list(
            RelationalAdapter().prediction_plan(
                request, ('prediction', 'explanation')
            )
        )


def correlated_payload(*, rows: int, kind: str = 'regression') -> dict:
    """A batch shaped the way the engine gathers one, with both id columns."""
    return {
        'task': {'kind': kind, 'target': {'column_name': 't'}},
        'schema': {'instance_table': {'columns': {}}},
        'context': {'instance_table': {'rows': [[i] for i in range(10)]}},
        'predict': {
            'instance_table': {
                'columns': ['instance_id', '__kumo_entity_ref_0'],
                'rows': [[200 + i, 10 + i] for i in range(rows)],
            }
        },
    }


def item(**fields):
    from sdm_api.generated.tfm_api import PredictionItem

    return PredictionItem(**fields)


def named(items, payload):
    from sdm_client.adapters.relational import _named_entities

    return _named_entities(tuple(items), payload)


def test_the_caller_entity_replaces_the_transport_key():
    payload = correlated_payload(rows=2)
    out = named(
        [
            item(row_index=0, id='200', prediction=1.0),
            item(row_index=1, id='201', prediction=2.0),
        ],
        payload,
    )
    assert [entry.id for entry in out] == [10, 11]


@pytest.mark.parametrize(
    ('items', 'fragment'),
    [
        ([item(row_index=None, id='200')], 'missing a row_index'),
        ([item(row_index=5, id='200')], 'asked about 2 rows'),
        (
            [item(row_index=0, id='200'), item(row_index=0, id='200')],
            'more than once',
        ),
        ([item(row_index=0, id='999')], 'asked about'),
        ([item(row_index=0, id=None)], 'carries no id'),
        ([item(row_index=0, id='200')], 'unanswered'),
        (
            [
                item(row_index=0, id='200', forecast_step=1),
                item(row_index=1, id='201'),
            ],
            'carries a forecast step',
        ),
    ],
)
def test_a_response_that_does_not_line_up_is_refused(items, fragment):
    """Rewriting an id can misattribute an answer, so every row is checked."""
    from sdm_client.errors import StructuredDataError

    with pytest.raises(StructuredDataError) as error:
        named(items, correlated_payload(rows=2))
    assert fragment in str(error.value)


def test_a_forecast_may_answer_a_row_once_per_step():
    payload = correlated_payload(rows=1, kind='forecasting')
    out = named(
        [
            item(row_index=0, id='200', forecast_step=1, prediction=1.0),
            item(row_index=0, id='200', forecast_step=2, prediction=2.0),
        ],
        payload,
    )
    assert [entry.forecast_step for entry in out] == [1, 2]
    assert {entry.id for entry in out} == {10}


def test_a_forecast_answering_the_same_step_twice_is_refused():
    from sdm_client.errors import StructuredDataError

    with pytest.raises(StructuredDataError) as error:
        named(
            [
                item(row_index=0, id='200', forecast_step=1),
                item(row_index=0, id='200', forecast_step=1),
            ],
            correlated_payload(rows=1, kind='forecasting'),
        )
    assert 'step 1 more than once' in str(error.value)


def test_a_forecast_row_without_a_step_is_refused():
    from sdm_client.errors import StructuredDataError

    with pytest.raises(StructuredDataError) as error:
        named(
            [item(row_index=0, id='200')],
            correlated_payload(rows=1, kind='forecasting'),
        )
    assert 'names no step' in str(error.value)


# --- a session answers the question it pinned -------------------------------


def test_a_session_refuses_a_request_that_asks_something_else():
    """Only the rows travel on a scoring call, so the pinned question is the one
    that would be answered. Two unrelated temporal queries compile to identical
    task and schema payloads, so nothing downstream can tell them apart."""
    pytest.importorskip('sdm_relational_engine')
    import pandas as pd

    from sdm_client import PredictiveQuery, Schema
    from sdm_client.client import FittedModel
    from sdm_client.schema import Relationship

    users = pd.DataFrame({'uid': range(40)})
    a = pd.DataFrame(
        {
            'uid': [i % 40 for i in range(400)],
            'when': pd.date_range('2020-01-01', periods=400, freq='D'),
        }
    )
    b = pd.DataFrame(
        {
            'uid': [i % 40 for i in range(400)],
            'when': pd.date_range('2020-01-01', periods=400, freq='D'),
        }
    )
    schema = Schema.from_data(
        {'users': users, 'a': a, 'b': b},
        primary_keys={'users': 'uid'},
        time_columns={'a': 'when', 'b': 'when'},
        relationships=[
            Relationship('a', 'uid', 'users', 'uid'),
            Relationship('b', 'uid', 'users', 'uid'),
        ],
    )
    kw = dict(indices=[1, 2], anchor_time=pd.Timestamp('2021-01-01'))
    pinned = PredictiveQuery(
        'PREDICT COUNT(a.*, 0, 30, days) > 0 FOR EACH users.uid', schema
    ).materialize(**kw)
    other = PredictiveQuery(
        'PREDICT COUNT(b.*, 0, 30, days) > 0 FOR EACH users.uid', schema
    ).materialize(**kw)

    handle = FittedModel._open(
        None, None, 'sid', ('prediction',), question=pinned._question
    )

    with pytest.raises(StructuredDataError, match='pinned'):
        handle.predict(other)


def test_a_session_refuses_a_request_asked_as_several_questions():
    pytest.importorskip('sdm_relational_engine')

    import pandas as pd

    from sdm_client import PredictiveQuery, Schema
    from sdm_client.client import FittedModel
    from sdm_client.schema import Relationship

    users = pd.DataFrame({'uid': range(40)})
    ev = pd.DataFrame(
        {
            'uid': [i % 40 for i in range(400)],
            'value': [1.0 + i % 5 for i in range(400)],
            'when': pd.date_range('2020-01-01', periods=400, freq='D'),
        }
    )
    schema = Schema.from_data(
        {'users': users, 'ev': ev},
        primary_keys={'users': 'uid'},
        time_columns={'ev': 'when'},
        relationships=[Relationship('ev', 'uid', 'users', 'uid')],
    )
    forecast = PredictiveQuery(
        'PREDICT SUM(ev.value, 0, 30, days) FORECAST 3 TIMEFRAMES FOR EACH users.uid',
        schema,
    ).materialize(indices=[1, 2], anchor_time=pd.Timestamp('2021-01-01'))

    handle = FittedModel._open(
        None, None, 'sid', ('prediction',), question=forecast.query
    )

    with pytest.raises(StructuredDataError, match='3 questions'):
        handle.predict(forecast)


def test_a_session_scoring_call_carries_the_inference_block():
    """The contract's SessionPredictionRequest lists `inference`; dropping it
    made explanations unreachable through a session, because the endpoint only
    explains when inference.run_mode is 'fast'."""
    pytest.importorskip('sdm_relational_engine')
    from sdm_client.adapters.relational import _SESSION_SCORING

    assert 'inference' in _SESSION_SCORING


def test_a_tabular_request_records_the_question_it_was_built_from():
    """The session guard tells requests apart by their question, and this route
    builds the request directly rather than gathering it, so it recorded none
    and the guard could never fire."""
    import numpy as np
    import pandas as pd

    from sdm_client import KumoTabular, PredictiveQuery, Schema

    rows = pd.DataFrame({'k': range(300), 'x': np.arange(300) - 150.0})
    rows['a'] = (rows['x'] > 0).astype(int)
    schema = Schema.from_data({'t': rows}, primary_keys={'t': 'k'})
    schema['t']['a'].stype = 'categorical'

    request = PredictiveQuery('PREDICT t.a FOR EACH t.k', schema).materialize(
        indices=[1, 2], model=KumoTabular()
    )

    assert request.query == 'PREDICT t.a FOR EACH t.k'


def _pinned(request):
    from sdm_client.client import FittedModel

    return FittedModel._open(
        None, None, 'sid', ('prediction',), question=request._question
    )


def test_a_session_refuses_a_frame_built_request_with_another_context():
    """A request built from frames is as much a question as a gathered one.

    Nothing about the two below differs except the labels, which is what the
    model learns from and what a scoring call does not carry.
    """
    import pandas as pd

    from sdm_client import Request

    def asked(labels):
        return Request(
            context=pd.DataFrame({'x': [1, 2, 3, 4], 'y': labels}),
            predict=pd.DataFrame({'x': [5, 6]}),
            target='y',
            task='classification',
        )

    with pytest.raises(StructuredDataError, match='context pinned'):
        _pinned(asked([1, 0, 1, 0])).predict(asked([0, 1, 0, 1]))


def test_a_session_accepts_the_same_frames_built_twice():
    import pandas as pd

    from sdm_client import Request

    def asked():
        return Request(
            context=pd.DataFrame({'x': [1, 2, 3, 4], 'y': [1, 0, 1, 0]}),
            predict=pd.DataFrame({'x': [5, 6]}),
            target='y',
            task='classification',
        )

    assert asked()._question == asked()._question


def test_a_session_refuses_the_same_query_asked_as_another_kind():
    """`task=` changes the question without changing a character of its text."""
    import pandas as pd

    from sdm_client import PredictiveQuery, Schema

    frame = pd.DataFrame({'id': range(40), 'y': [i % 5 for i in range(40)]})
    schema = Schema.from_data({'t': frame}, primary_keys={'t': 'id'})
    query = PredictiveQuery('PREDICT t.y FOR EACH t.id', schema)
    regression = query.materialize(indices=[1, 2], task='regression')
    classification = query.materialize(indices=[1, 2], task='classification')

    assert regression.query == classification.query
    with pytest.raises(StructuredDataError, match='task kind'):
        _pinned(regression).predict(classification)


def test_a_session_accepts_a_literal_written_with_the_other_quote():
    """Two spellings of one literal are one question."""
    from sdm_client.request import _one_spelling_of_each_literal as spelled

    double = (
        'PREDICT COUNT(o.* WHERE o.channel="web", 0, 30, days) FOR EACH u.id'
    )
    single = (
        "PREDICT COUNT(o.* WHERE o.channel='web', 0, 30, days) FOR EACH u.id"
    )
    assert spelled(double) == spelled(single)
    assert spelled('o.a="x" AND o.b=\'y\'') == spelled('o.a=\'x\' AND o.b="y"')
    assert spelled('o.a="x"') != spelled('o.a="z"')


def test_a_frame_built_request_says_when_it_scores_rows_it_shows_the_answers_to():
    """Both gathered routes take such rows out of the context they send, and a
    session says so when it cannot. This route did neither."""
    import pandas as pd

    from sdm_client import Request
    from sdm_client.core.warnings import SDMWarning
    from sdm_client.request import (
        warn_of_rows_scored_from_their_own_answers,
    )

    context = pd.DataFrame(
        {
            'id': range(20),
            'x': [1.0 * i for i in range(20)],
            'y': [i % 2 for i in range(20)],
        }
    )

    def notices(predict):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', SDMWarning)
            warn_of_rows_scored_from_their_own_answers(
                Request(
                    context=context,
                    predict=predict,
                    target='y',
                    task='classification',
                )
            )
        return [
            str(item.message)
            for item in caught
            if 'being scored' in str(item.message)
        ]

    overlapping = notices(context.head(3).drop(columns=['y']))
    assert len(overlapping) == 1
    assert '3 of the 3 rows being scored' in overlapping[0]
    assert 'materialize()' in overlapping[0]

    assert notices(pd.DataFrame({'id': [90, 91], 'x': [9.0, 9.1]})) == []


def test_a_context_whose_answers_are_missing_is_not_called_a_leak():
    """A row with no target teaches nothing and gives nothing away."""
    import pandas as pd

    from sdm_client import Request
    from sdm_client.core.warnings import SDMWarning

    context = pd.DataFrame(
        {
            'id': range(20),
            'x': [1.0 * i for i in range(20)],
            'y': [None] * 3 + [i % 2 for i in range(17)],
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', SDMWarning)
        Request(
            context=context,
            predict=context.head(3).drop(columns=['y']),
            target='y',
            task='classification',
        )
    assert not [item for item in caught if 'being scored' in str(item.message)]


def test_the_session_warning_counts_rows_and_not_keys():
    """`scored_keys()` returns a set, so duplicate scoring rows disappeared
    before either number was taken: two rows asking about one pinned entity
    were reported as one of one."""
    import pandas as pd

    from sdm_client import Request, Schema
    from sdm_client.client import _warn_scoring_what_was_pinned
    from sdm_client.wire.keys import _named_by

    context = pd.DataFrame(
        {'id': [1, 2, 3], 'x': [0.1, 0.2, 0.3], 'y': [0, 1, 0]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'instance': context}, primary_keys={'instance': 'id'}
        )

    def said_for(ids):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            request = Request(
                context=context,
                predict=pd.DataFrame({'id': ids, 'x': [0.1] * len(ids)}),
                schema=schema,
                target='y',
                task='classification',
            )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _warn_scoring_what_was_pinned(
                request, frozenset({_named_by([1])}), False
            )
        return [str(item.message) for item in caught]

    assert said_for([1, 1, 2])[0].startswith('2 of the 3 rows being scored')
    assert said_for([1, 1])[0].startswith('2 of the 2 rows being scored')
    assert said_for([7, 8, 9]) == []


def test_a_request_built_to_score_a_session_is_not_warned_about_its_context():
    """The leak warning is about a pairing that only travels on the one-shot
    call. Asked where the request is built, it fired about a context a session
    scoring call never sends, and failed under warnings-as-errors."""
    import pandas as pd

    from sdm_client import Request
    from sdm_client.core.warnings import SDMWarning

    context = pd.DataFrame(
        {
            'id': range(20),
            'x': [1.0 * i for i in range(20)],
            'y': [i % 2 for i in range(20)],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', SDMWarning)
        Request(
            context=context,
            predict=context.head(3).drop(columns=['y']),
            target='y',
            task='classification',
        )

    assert not [item for item in caught if 'being scored' in str(item.message)]


def test_the_session_warning_knows_one_entity_written_two_ways():
    """The pinned key is read from an encoded envelope and the scored key from
    a frame, so an entity pinned as `1` and scored as `1.0` never met, and a
    session scored rows its context answers without saying so."""
    import pandas as pd

    from sdm_client import Request, Schema
    from sdm_client.client import _warn_scoring_what_was_pinned
    from sdm_client.wire.keys import _named_by

    context = pd.DataFrame(
        {'id': [1, 2, 3], 'x': [0.1, 0.2, 0.3], 'y': [0, 1, 0]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'instance': context}, primary_keys={'instance': 'id'}
        )

    def warned_for(ids):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            request = Request(
                context=context,
                predict=pd.DataFrame({'id': ids, 'x': [0.1] * len(ids)}),
                schema=schema,
                target='y',
                task='classification',
            )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _warn_scoring_what_was_pinned(
                request, frozenset({_named_by([1])}), False
            )
        return bool(caught)

    for named in ([1], [1.0], pd.array([1.0], dtype='Float64'), ['1']):
        assert warned_for(named)
    for named in ([9], [9.0], [1.5]):
        assert not warned_for(named)


def test_a_scoring_call_writes_its_rows_to_the_schema_that_was_pinned():
    """The schema is not sent again on a scoring call: it was pinned when the
    session opened. Typing the rows from the half that travels sent `1.0`
    under a column pinned as `int64`, which the endpoint refused as
    `SESSION_PREDICT_SCHEMA_MISMATCH`, a failure about a value the caller
    never wrote and cannot see."""
    import pandas as pd

    from sdm_client import Request, Schema
    from sdm_client.adapters.tabular import TabularAdapter

    frame = pd.DataFrame(
        {
            'id': range(24),
            'x': [1.0 * i for i in range(24)],
            'y': [i % 2 for i in range(24)],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'instance': frame}, primary_keys={'instance': 'id'}
        )
        opening = Request(
            context=frame,
            predict=frame.head(2)[['id', 'x']],
            schema=schema,
            target='y',
            task='classification',
        )
        scoring = Request(
            context=frame,
            predict=pd.DataFrame({'id': [1.0, 2.0], 'x': [9.0, 8.0]}),
            schema=schema,
            target='y',
            task='classification',
        )

    adapter = TabularAdapter()
    pinned = adapter.session_plan(opening, ('prediction',))
    assert pinned['schema']['instance_table']['columns']['id']['dtype'] == (
        'int64'
    )

    payload = next(
        iter(adapter.session_prediction_plan(scoring, ('prediction',), pinned))
    )
    table = payload['predict']['instance_table']
    at = table['columns'].index('id')
    assert [row[at] for row in table['rows']] == [1, 2]

    unpinned = next(
        iter(adapter.session_prediction_plan(scoring, ('prediction',)))
    )
    table = unpinned['predict']['instance_table']
    at = table['columns'].index('id')
    assert [row[at] for row in table['rows']] == [1.0, 2.0]


@pytest.mark.parametrize(
    ('pinned_keys', 'scored_keys', 'answers'),
    [
        (list(range(24)), [1, 2], True),
        (list(range(24)), [1.0, 2.0], True),
        (list(range(24)), ['abc', 'def'], False),
        ([float(i) for i in range(24)], [1, 2], True),
        ([float(i) for i in range(24)], ['abc', 'def'], False),
        ([str(i) for i in range(24)], [1, 2], True),
        ([str(i) for i in range(24)], ['abc', 'def'], True),
    ],
)
def test_rows_the_pinned_schema_cannot_hold_are_refused_by_name(
    pinned_keys, scored_keys, answers
):
    """Writing the rows to the pinned schema is what makes the call legible to
    the endpoint, and the only thing that can fail for a reason the caller
    cannot see: the failure names a dtype they never wrote, or arrives as a
    bare ValueError from inside pandas."""
    import pandas as pd

    from sdm_client import Request, Schema
    from sdm_client.adapters.tabular import TabularAdapter
    from sdm_client.errors import StructuredDataError

    frame = pd.DataFrame(
        {
            'id': pinned_keys,
            'x': [1.0 * i for i in range(24)],
            'y': [i % 2 for i in range(24)],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'instance': frame}, primary_keys={'instance': 'id'}
        )
        opening = Request(
            context=frame,
            predict=frame.head(2)[['id', 'x']],
            schema=schema,
            target='y',
            task='classification',
        )
        scoring = Request(
            context=frame,
            predict=pd.DataFrame({'id': scored_keys, 'x': [9.0, 8.0]}),
            schema=schema,
            target='y',
            task='classification',
        )

    adapter = TabularAdapter()
    pinned = adapter.session_plan(opening, ('prediction',))

    def plan():
        return next(
            iter(
                adapter.session_prediction_plan(
                    scoring, ('prediction',), pinned
                )
            )
        )

    if answers:
        assert plan()['predict']['instance_table']['rows']
    else:
        with pytest.raises(StructuredDataError) as raised:
            plan()
        assert 'this session pinned' in str(raised.value)
        assert 'fit()' in str(raised.value)
