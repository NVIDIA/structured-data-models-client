# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the client refuses before spending a round trip, and how sessions behave.

The endpoint enforces several sizes, and a request past one of them comes back
as a 500 or a 422 that names neither the limit nor how to get under it. Every
check here exists so the caller reads a sentence instead.
"""

from __future__ import annotations

import pytest

from sdm_client import KumoRelational, UnsupportedRequestError
from sdm_client.adapters.relational import RelationalAdapter
from sdm_client.models import RELATIONAL_LIMITS


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

    def __init__(self, *payloads: dict) -> None:
        self.payloads = tuple(payloads)
        self.requested_inference: dict = {}
        self.inference: dict = {}


def test_the_declared_limits_are_the_ones_the_endpoint_enforces():
    """Read off the relational NIM; a change here should be deliberate."""
    assert RELATIONAL_LIMITS['context_rows_fast'] == 1_000
    assert RELATIONAL_LIMITS['context_rows_max'] == 10_000
    assert RELATIONAL_LIMITS['predict_rows_max'] == 1_000
    assert RELATIONAL_LIMITS['predict_rows_temporal_link_prediction'] == 200
    assert RELATIONAL_LIMITS['explanation_predict_rows'] == 1
    assert RELATIONAL_LIMITS['hops_max'] == 6
    assert RELATIONAL_LIMITS['tables_max'] == 15


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
    runtime = KumoRelational().spec.runtime
    assert {'forecasting', 'temporal_link_prediction'} <= runtime.tasks
    assert 'explanation' in runtime.outputs
    # Only these inference options reach the wire; the engine defaults to more.
    assert 'output_type' not in runtime.inference_options
    assert 'num_estimators' in runtime.inference_options
