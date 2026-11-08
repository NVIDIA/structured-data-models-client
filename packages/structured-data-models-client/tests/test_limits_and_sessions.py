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
from test_a_session_can_hold_back_what_it_will_score import pin
from test_limits_and_absent_outputs import sent

from sdm_client import Schema, UnsupportedRequestError
from sdm_client.errors import StructuredDataError
from sdm_client.models import KUMO_RELATIONAL, RELATIONAL_LIMITS


def payload(*, predict_rows: int, context_rows: int) -> dict:
    from test_limits_and_absent_outputs import gathered

    request = gathered(
        context_rows=min(context_rows, 10000),
        run_mode='fast',
        predict_rows=predict_rows,
    )
    body = request.payloads[0]
    if context_rows > 10000:
        rows = body['context']['instance_table']['rows']
        rows.extend([list(rows[-1]) for _ in range(context_rows - len(rows))])
    return body


def _Gathered(*payloads):
    import pandas as pd

    from sdm_client import Request

    return Request.materialized(
        payloads=tuple(payloads),
        schema=Schema.from_data(
            {'users': pd.DataFrame({'uid': range(20), 'label': range(20)})},
            primary_keys={'users': 'uid'},
        ),
        query='PREDICT users.label FOR EACH users.uid',
        indices=None,
        entity_table='users',
    )


def test_the_declared_limits_are_pinned_to_what_was_read_off_the_endpoint():
    """A pin, not a probe.

    These numbers were read from the relational NIM. This asserts they have
    not drifted in the client, which is all it can do offline. The client
    holds a request to no size of its own, so no most is set on context,
    rows scored, bytes, tables or hops.
    """
    assert RELATIONAL_LIMITS == {'context_rows_fast': 1_000, 'classes_max': 10}


@pytest.mark.parametrize('bad', ['2', True])
def test_num_hops_must_be_a_whole_number(bad):
    from test_frame_sessions_core import query

    from sdm_client import KumoRelational

    with pytest.raises(StructuredDataError) as error:
        query().materialize(indices=[1], model=KumoRelational(), num_hops=bad)
    assert 'num_hops must be a whole number' in str(error.value)
    assert error.value.code == 'INVALID_REQUEST'


@pytest.mark.parametrize(
    ('predict_rows', 'context_rows'), [(4, 100), (1, 5000)]
)
def test_explanations_have_no_separate_row_or_context_limit(
    predict_rows, context_rows
):
    request = _Gathered(
        payload(predict_rows=predict_rows, context_rows=context_rows)
    )
    plans = list(sent(request, ('prediction', 'explanation')))
    assert plans[0]['output']['fields'] == ['prediction', 'explanation']


def test_an_explanation_within_both_limits_is_allowed():
    request = _Gathered(payload(predict_rows=1, context_rows=500))
    plans = list(sent(request, ('prediction', 'explanation')))
    assert len(plans) == 1
    assert plans[0]['output']['fields'] == ['prediction', 'explanation']


def test_a_prediction_without_an_explanation_is_not_held_to_those_limits():
    """Both caps are explanation-only; an ordinary batch is unaffected."""
    request = _Gathered(payload(predict_rows=500, context_rows=5_000))
    plans = list(sent(request, ('prediction',)))
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
        list(sent(direct, ('prediction',)))
    assert 'materialize' in str(error.value)


def test_a_session_cannot_pin_more_than_one_gathered_context():
    """Each gathered batch carries its own context; a session pins one."""
    request = _Gathered(
        payload(predict_rows=2, context_rows=10),
        payload(predict_rows=2, context_rows=10),
    )
    with pytest.raises(StructuredDataError) as error:
        pin(request, 'kumo-relational')
    assert error.value.code == 'INVALID_REQUEST'
    assert 'session pins one context' in str(error.value)


def test_the_default_outputs_follow_the_relational_task_kind():
    from test_limits_and_absent_outputs import gathered

    for classification, expected in [
        (False, ['prediction']),
        (True, ['prediction', 'probabilities']),
    ]:
        request = gathered(
            context_rows=50,
            run_mode='fast',
            classification=classification,
            target=[0, 1] * 25 if classification else None,
        )
        (body,) = sent(request, None)
        assert body['output']['fields'] == expected


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
        list(sent(request, ('prediction',)))
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
    (plan,) = sent(request, ('prediction',))
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

    plans = list(sent(request, ('prediction',)))
    assert len(plans) == 1
    assert plans[0]['model'] == 'kumo-relational'

    assert sent(request, ('prediction', 'explanation'))


def gathered_in_calls(*, calls: int, context_rows: int):
    """A relational request already gathered as ``calls`` envelopes."""
    import pandas as pd

    from sdm_client import Request

    return Request.materialized(
        payloads=tuple(
            payload(predict_rows=2, context_rows=context_rows)
            for _ in range(calls)
        ),
        schema=Schema.from_data({'users': pd.DataFrame({'user_id': [1]})}),
        query='PREDICT ... FOR EACH users.user_id',
        indices=[1, 2, 3, 4],
        entity_table='users',
    )


def test_each_call_s_context_is_sent_at_any_size():
    """The client holds a context to no size of its own, in each call or
    across them."""
    sent(gathered_in_calls(calls=2, context_rows=10_001))


def test_a_batched_request_reports_the_context_one_call_carries():
    """The sum reads as a context no call sends: two calls of 1,000 are not a
    2,000-row context, and the cap they are held to is 1,000."""
    request = gathered_in_calls(calls=2, context_rows=1_000)
    assert request.context_rows == 2_000
    assert 'up to 1000 context rows per call across 2 calls' in repr(request)


def test_a_request_built_from_frames_still_describes_itself():
    """`payloads` is None on that route, and counting it is how a description
    of the gathered one stopped every direct request from being printed."""
    import pandas as pd

    from sdm_client import Request

    request = Request(
        context=pd.DataFrame({'x': [1, 2], 'y': [0, 1]}),
        predict=pd.DataFrame({'x': [3]}),
        target='y',
        task='classification',
    )
    assert '2 context rows, 1 to score' in repr(request)


def correlated_payload(*, rows: int, kind: str = 'regression'):
    import pandas as pd

    from sdm_client import KumoRelational, PredictiveQuery

    schema = Schema.from_data(
        {
            'users': pd.DataFrame(
                {'uid': range(10, 40), 'value': [float(i) for i in range(30)]}
            )
        },
        primary_keys={'users': 'uid'},
    )
    schema['users']['value'].stype = 'numerical'
    request = PredictiveQuery(
        'PREDICT users.value FOR EACH users.uid', schema
    ).materialize(indices=list(range(10, 10 + rows)), model=KumoRelational())
    request.payloads[0]['task']['kind'] = kind
    return request


def item(**fields):
    from sdm_api.generated.tfm_api import PredictionItem

    return PredictionItem(**fields)


def named(items, request):
    from dataclasses import asdict

    from test_tabular_direct_core import Endpoint, client

    table = request.payloads[0]['predict']['instance_table']
    key = table['columns'].index('instance_id')
    identifiers = {
        str(200 + i): str(row[key]) for i, row in enumerate(table['rows'])
    }
    predictions = []
    for item in items:
        value = {
            key: value
            for key, value in asdict(item).items()
            if value is not None
        }
        if value.get('id') in identifiers:
            value['id'] = identifiers[value['id']]
        predictions.append(value)
    endpoint = Endpoint(
        {
            'model': 'kumo-relational',
            'id': 'answer',
            'metadata': {},
            'predictions': predictions,
        }
    )
    with client(endpoint) as model_client:
        return model_client.predict(
            request, model='kumo-relational', outputs=('prediction',)
        ).items


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
    fields = {
        'missing a row_index': ('prediction.row_index', 'missing'),
        'asked about 2 rows': ('prediction.row_index', 'out_of_range'),
        'more than once': ('prediction.row_index', 'repeated'),
        'asked about': ('prediction.id', 'not_the_row_s'),
        'carries no id': ('prediction.id', 'missing'),
        'unanswered': ('prediction.row_index', 'unanswered'),
        'carries a forecast step': (
            'prediction.forecast_step',
            'not_a_forecast',
        ),
    }
    assert error.value.code == 'INVALID_RESPONSE'
    assert (
        error.value.details['field'],
        error.value.details['reason'],
    ) == fields[fragment]


# --- a session answers the question it pinned -------------------------------


def test_a_session_refuses_a_request_that_asks_something_else():
    """Only the rows travel on a scoring call, so the pinned question is the one
    that would be answered. Two unrelated temporal queries compile to identical
    task and schema payloads, so nothing downstream can tell them apart."""
    import pandas as pd
    from test_frame_sessions_core import Endpoint
    from test_tabular_direct_core import client

    from sdm_client import PredictiveQuery, Schema
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

    endpoint = Endpoint()
    with client(endpoint) as model_client:
        with model_client.fit(
            pinned, model='kumo-relational', outputs=('prediction',)
        ) as handle:
            with pytest.raises(StructuredDataError, match='pinned'):
                handle.predict(other)
    assert not endpoint.scored


def test_a_session_refuses_a_request_asked_as_several_questions():

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
        score_pinned(asked([1, 0, 1, 0]), asked([0, 1, 0, 1]))


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
        score_pinned(regression, classification)


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


def test_a_scoring_call_writes_its_rows_to_the_schema_that_was_pinned():
    """The schema is not sent again on a scoring call: it was pinned when the
    session opened. Typing the rows from the half that travels sent `1.0`
    under a column pinned as `int64`, which the endpoint refused as
    `SESSION_PREDICT_SCHEMA_MISMATCH`, a failure about a value the caller
    never wrote and cannot see."""
    import pandas as pd

    from sdm_client import Request, Schema

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

    pinned, payload = score_pinned(opening, scoring)
    assert (
        pinned['schema']['instance_table']['columns']['id']['dtype'] == 'int64'
    )
    table = payload['predict']['instance_table']
    at = table['columns'].index('id')
    assert [row[at] for row in table['rows']] == [1, 2]
    assert all(type(row[at]) is int for row in table['rows'])


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

    def plan():
        return score_pinned(opening, scoring)[1]

    if answers:
        assert plan()['predict']['instance_table']['rows']
    else:
        with pytest.raises(StructuredDataError) as raised:
            plan()
        assert 'this session pinned' in str(raised.value)
        assert 'fit()' in str(raised.value)


def score_pinned(opening, scoring):
    from test_entity_id_parity import StubTransport
    from test_tabular_direct_core import client

    class SessionTransport(StubTransport):
        def close(self):
            pass

    transport = SessionTransport(
        {
            'model': 'kumo-tabular',
            'id': 'answer',
            'metadata': {},
            'predictions': [
                {'row_index': i, 'prediction': 0}
                for i in range(scoring.predict_rows)
            ],
        }
    )
    with client(transport) as model_client:
        with model_client.fit(
            opening, model='kumo-tabular', outputs=('prediction',)
        ) as fitted:
            fitted.predict(scoring)
    assert len(transport.scored) == 1
    return transport.created, transport.scored[0]


@pytest.mark.parametrize(
    ('ids', 'overlap'),
    [
        ([1, 1, 2], 2),
        ([1, 1], 2),
        ([7, 8, 9], 0),
        ([1], 1),
        ([1.0], 1),
        ([9.0], 0),
    ],
)
def test_session_overlap_warning_counts_rows_and_normalizes_keys(ids, overlap):
    import pandas as pd

    from sdm_client import Request

    context = pd.DataFrame(
        {'id': [1, 4, 5], 'x': [0.1, 0.2, 0.3], 'y': [0, 1, 0]}
    )
    schema = Schema.from_data(
        {'instance': context}, primary_keys={'instance': 'id'}
    )
    opening = Request(
        context=context,
        predict=pd.DataFrame({'id': [6], 'x': [0.1]}),
        schema=schema,
        target='y',
        task='classification',
    )
    scoring = Request(
        context=context,
        predict=pd.DataFrame({'id': ids, 'x': [0.1] * len(ids)}),
        schema=schema,
        target='y',
        task='classification',
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        score_pinned(opening, scoring)
    notices = [
        str(item.message)
        for item in caught
        if "session's pinned context" in str(item.message)
    ]
    if overlap:
        assert len(notices) == 1
        assert notices[0].startswith(
            f'{overlap} of the {len(ids)} rows being scored'
        )
    else:
        assert notices == []
