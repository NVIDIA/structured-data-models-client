# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the endpoint will not take, and what the answer did not contain.

Two silences. The tabular endpoint caps a context at 10,000 rows and the client
did not know it, so a caller could encode and transmit a table of any size to
be refused at the far end -- while the identical overage on the other model was
refused locally with a message naming the limit.

And a field a caller names is validated against the model and never against
what comes back. The endpoint answers 200, leaves the field out, and the frame
simply has no such column. That happens on both models for different reasons,
so what is reported is what the answer did not carry, never a guess at why.
"""

from __future__ import annotations

import warnings
from functools import partial

import numpy as np
import pandas as pd
import pytest

from sdm_client import PredictiveQuery, Schema
from sdm_client.client import _report_absent
from sdm_client.errors import StructuredDataError
from sdm_client.models import KUMO_TABULAR
from sdm_client.result import PredictionResult


def warnings_from(build):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        built = build()
    return built, [str(w.message) for w in caught]


def table(rows: int) -> Schema:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            'rid': np.arange(rows),
            'amount': rng.gamma(2.0, 30.0, rows).round(2),
            'region': rng.choice(list('NSEW'), rows),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data({'t': frame}, primary_keys={'t': 'rid'})


def query(schema: Schema) -> PredictiveQuery:
    return PredictiveQuery('PREDICT t.amount FOR EACH t.rid', schema)


# --- the context the endpoint will take ------------------------------------


def test_the_declared_cap_is_the_one_the_endpoint_enforces():
    """Measured at the boundary: 10,000 answer, 10,001 refuse."""
    assert KUMO_TABULAR.runtime.limits['context_rows_max'] == 10_000


def test_a_context_above_the_cap_is_refused_before_it_is_encoded():
    schema = table(200)
    with pytest.raises(StructuredDataError) as raised:
        query(schema).materialize(indices=[0, 1], num_context_samples=50_000)
    assert 'accepts at most 10,000 context rows' in str(raised.value)


def test_the_cap_itself_is_allowed():
    schema = table(200)
    request = query(schema).materialize(
        indices=[0, 1], num_context_samples=10_000
    )
    assert len(request.context.instance_table) == 198


def test_an_unnamed_context_is_capped_and_says_so():
    schema = table(12_000)
    request, said = warnings_from(
        partial(query(schema).materialize, indices=[0, 1])
    )
    assert len(request.context.instance_table) == 10_000
    assert any('were sampled' in s for s in said), said
    assert any('num_context_samples' in s for s in said)


def test_a_context_under_the_cap_is_left_whole_and_unremarked():
    schema = table(500)
    request, said = warnings_from(
        partial(query(schema).materialize, indices=[0, 1])
    )
    assert len(request.context.instance_table) == 498
    assert not any('were sampled' in s for s in said)


def test_the_capped_context_is_a_sample_and_not_the_first_rows():
    """Taking the head would bias the context toward whatever the table is
    ordered by, which for a table ordered by time is the oldest rows."""
    schema = table(12_000)
    request = query(schema).materialize(indices=[0])
    taken = request.context.instance_table['rid']
    assert taken.max() > 11_000
    assert len(taken) == 10_000


def test_the_predict_frame_route_is_capped_too():
    """The route with no knob at all. The schema's table became the context
    whole, so a large table was encoded and sent in full every time, to be
    refused."""
    schema = table(12_000)
    rows = schema['t'].data.head(3).drop(columns=['amount'])
    request, said = warnings_from(
        partial(query(schema).materialize, predict=rows)
    )
    assert len(request.context.instance_table) == 10_000
    assert any('were sampled' in s for s in said)


# --- what the answer did not carry -----------------------------------------


class FakeItem:
    def __init__(self, **fields):
        for name, value in fields.items():
            setattr(self, name, value)


def result(outputs=(), explanation=None, **fields) -> PredictionResult:
    return PredictionResult(
        model='kumo-tabular',
        request_id='r1',
        items=[FakeItem(row_index=0, **fields)],
        outputs=outputs,
        explanation=explanation,
    )


def test_a_field_the_answer_carries_is_not_reported():
    answer = result(prediction=1.0)
    assert answer.absent(['prediction']) == ()


@pytest.mark.parametrize('empty', [None, {}, [], ()])
def test_a_field_present_but_empty_counts_as_absent(empty):
    """An empty mapping is what a 200 carrying no quantiles looks like."""
    assert result(quantiles=empty).absent(['quantiles']) == ('quantiles',)


def test_several_absent_fields_are_reported_together():
    answer = result(prediction=1.0, quantiles={}, probabilities={})
    assert answer.absent(['prediction', 'quantiles', 'probabilities']) == (
        'quantiles',
        'probabilities',
    )


def test_a_repeated_request_is_reported_once():
    assert result(quantiles={}).absent(['quantiles', 'quantiles']) == (
        'quantiles',
    )


def test_explanation_is_looked_for_where_it_lives():
    assert result(prediction=1.0).absent(['explanation']) == ('explanation',)
    assert result(prediction=1.0, explanation=None).absent(['explanation']) == (
        'explanation',
    )
    carried = PredictionResult(
        model='m',
        request_id='r',
        items=[FakeItem(row_index=0)],
        explanation={'feature_importance': {}},
    )
    assert carried.absent(['explanation']) == ()


def test_a_field_this_type_cannot_look_for_is_not_called_absent():
    """Reporting it would be a guess, and an unknown field is refused before
    anything is sent."""
    assert result(prediction=1.0).absent(['something_else']) == ()


def test_only_the_fields_the_caller_named_are_reported():
    answer = result(prediction=1.0, quantiles={})
    _, said = warnings_from(partial(_report_absent, answer, None, 'kumo-x'))
    assert said == []


def test_a_named_field_the_answer_lacks_is_announced():
    answer = result(prediction=1.0, quantiles={})
    _, said = warnings_from(
        partial(_report_absent, answer, ['prediction', 'quantiles'], 'kumo-x')
    )
    assert len(said) == 1
    assert "kumo-x answered without ['quantiles']" in said[0]
    assert 'to_pandas()' in said[0]


def test_the_report_says_what_happened_and_not_why():
    """`kumo-relational` returns `prediction` for a regression asked alone and
    omits it when quantiles are asked alongside, so any sentence about what the
    model produces for a task kind would be false in that case."""
    answer = result(prediction=None, quantiles={'0.5': 1.0})
    _, said = warnings_from(
        partial(_report_absent, answer, ['prediction'], 'kumo-relational')
    )
    assert 'does not produce' not in said[0]
    assert 'task kind' not in said[0]


def test_the_result_is_returned_unchanged():
    answer = result(prediction=1.0, quantiles={})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert _report_absent(answer, ['quantiles'], 'kumo-x') is answer


def test_a_materialized_context_above_the_cap_is_refused_not_merely_the_number():
    """A small table makes this a check on the argument. The rows have to be
    there for it to be a check on the context."""
    schema = table(12_000)
    with pytest.raises(StructuredDataError) as raised:
        query(schema).materialize(indices=[0, 1], num_context_samples=10_001)
    assert 'accepts at most 10,000 context rows' in str(raised.value)


def test_the_predict_route_advises_something_that_route_allows():
    """Its own refusal says `num_context_samples` cannot be used here, so the
    warning must not tell the caller to pass it."""
    schema = table(12_000)
    rows = schema['t'].data.head(3).drop(columns=['amount'])
    _, said = warnings_from(partial(query(schema).materialize, predict=rows))
    capped = [s for s in said if 'were sampled' in s]
    assert capped
    remedy = capped[0].split('sampled.')[1]
    assert 'Pass num_context_samples' not in remedy
    assert 'takes no num_context_samples' in remedy
    assert 'building the schema' in remedy

    with pytest.raises(StructuredDataError) as raised:
        query(schema).materialize(predict=rows, num_context_samples=100)
    assert 'cannot be used with predict=<frame>' in str(raised.value)


def test_an_empty_answer_reports_nothing_missing():
    """A zero-row batch is a supported request with a real empty answer, so
    every field looks absent and none of them was dropped."""
    empty = PredictionResult(
        model='kumo-tabular', request_id='r1', items=[], outputs=()
    )
    assert empty.absent(['prediction', 'quantiles']) == ()
    _, said = warnings_from(
        partial(_report_absent, empty, ['prediction'], 'kumo-tabular')
    )
    assert said == []


def test_a_fit_time_output_list_is_not_read_again_later():
    """The caller's list is theirs to mutate; what was pinned is not."""
    from sdm_client.client import FittedModel

    named = ['prediction', 'quantiles']
    fitted = FittedModel(
        client=None,
        adapter=None,
        session_id='s1',
        outputs=('prediction', 'quantiles'),
        named_outputs=named,
    )
    named.append('embeddings')
    assert fitted._named_outputs == ('prediction', 'quantiles')


# --- what the relational endpoint enforces, read the way it reads it --------


def gathered(*, context_rows: int, run_mode: str | None, target=None) -> object:
    values = target if target is not None else list(range(context_rows))
    payload = {
        'task': {'kind': 'regression', 'target': {'column_name': 'TARGET'}},
        'schema': {'instance_table': {'columns': {}}},
        'context': {
            'instance_table': {
                'columns': ['TARGET'],
                'rows': [[value] for value in values],
            }
        },
        'predict': {'instance_table': {'columns': ['TARGET'], 'rows': [[0]]}},
    }
    if run_mode is not None:
        payload['inference'] = {'run_mode': run_mode}

    class Gathered:
        payloads = (payload,)
        requested_inference: dict = {}

    return Gathered()


def explanation_plan(request):
    from sdm_client.adapters.relational import RelationalAdapter

    return list(
        RelationalAdapter().prediction_plan(
            request, ('prediction', 'explanation')
        )
    )


def test_an_explanation_is_refused_on_the_mode_the_endpoint_refuses_it_on():
    """A graph that supplies 120 rows for a request of 1,500 still selects the
    larger-context mode, and that is what the endpoint gates explaining on."""
    with pytest.raises(StructuredDataError) as raised:
        explanation_plan(gathered(context_rows=120, run_mode='best'))
    assert 'num_context_samples' in str(raised.value)


def test_a_small_context_asked_for_smally_still_explains():
    assert explanation_plan(gathered(context_rows=120, run_mode='fast'))


def test_a_payload_naming_no_mode_is_judged_on_the_rows_it_carries():
    """Nothing the engine gathers arrives without a run_mode, but a hand-built
    payload can, and the rows are then all there is to read."""
    assert explanation_plan(gathered(context_rows=120, run_mode=None))
    with pytest.raises(StructuredDataError):
        explanation_plan(gathered(context_rows=1_001, run_mode=None))


# --- a context that teaches nothing, on both routes -------------------------


def relational_constant(kind: str) -> object:
    request = gathered(context_rows=50, run_mode='fast', target=[50] * 50)
    request.payloads[0]['task']['kind'] = kind
    return request


def plan(request):
    from sdm_client.adapters.relational import RelationalAdapter

    return list(RelationalAdapter().prediction_plan(request, ('prediction',)))


def test_a_constant_classification_target_is_refused_relationally_too():
    """The tabular route refused this locally and the relational one sent it,
    so the same unlearnable question was answered by one model of the two."""
    with pytest.raises(StructuredDataError) as raised:
        plan(relational_constant('binary_classification'))
    assert 'at least two classes' in str(raised.value)


def test_a_constant_regression_target_is_remarked_on_and_still_sent():
    built, said = warnings_from(
        partial(plan, relational_constant('regression'))
    )
    assert built
    assert any('every context row has the same TARGET' in s for s in said), said


def test_a_target_that_varies_is_left_alone():
    _, said = warnings_from(
        partial(plan, gathered(context_rows=50, run_mode='fast'))
    )
    assert not any('every context row' in s for s in said)


def test_a_context_past_the_models_cap_is_refused_before_it_is_sent():
    """The client's own preamble says it refuses first where it can tell, and
    the model is known by the time this check runs, so sending 12,500 rows to
    be told the limit at the far end was work nobody needed to do."""
    import numpy as np
    import pandas as pd

    from sdm_client import KumoTabular, Request, UnsupportedRequestError
    from sdm_client.base import check_request_against_model

    rng = np.random.default_rng(0)
    rows = 12_500
    context = pd.DataFrame({'k': range(rows), 'x': rng.normal(size=rows)})
    context['y'] = (context['x'] > 0).astype(int)
    request = Request(
        context=context,
        predict=context.drop(columns=['y']).head(3),
        target='y',
        task='classification',
    )

    with pytest.raises(UnsupportedRequestError, match='12,500'):
        check_request_against_model(
            request, KumoTabular().spec, ('prediction',)
        )


def test_a_context_within_the_cap_is_not_refused():
    import numpy as np
    import pandas as pd

    from sdm_client import KumoTabular, Request
    from sdm_client.base import check_request_against_model

    rng = np.random.default_rng(0)
    context = pd.DataFrame({'k': range(500), 'x': rng.normal(size=500)})
    context['y'] = (context['x'] > 0).astype(int)
    request = Request(
        context=context,
        predict=context.drop(columns=['y']).head(3),
        target='y',
        task='classification',
    )

    check_request_against_model(request, KumoTabular().spec, ('prediction',))
