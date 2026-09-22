# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from predictive_query.query import ValidatedPredictiveQuery
from sdm_relational_engine.exceptions import HTTPException
from sdm_relational_engine.rfm import Graph, KumoRelational

_CARDINALITY_DETAIL = (
    '{"detail": "categorical cardinality 15001 exceeds limit 10000."}'
)


class FailingAPI:
    r"""Fake RFMAPI whose predict always fails with a fixed status, counting
    how many times the retry loop actually called it.
    """

    def __init__(self, status: int, detail: str = 'boom') -> None:
        self.status = status
        self.detail = detail
        self.n_predict = 0

    def predict(
        self,
        request: Any,
        *,
        entity_ids: Any = None,
        instance_ids: Any = None,
        anchor_times: Any = None,
    ) -> Any:
        self.n_predict += 1
        raise HTTPException(self.status, self.detail)


def _model(graph: Graph, api: FailingAPI) -> KumoRelational:
    model = KumoRelational(graph, verbose=False)
    model._client = api  # type: ignore[assignment]
    return model


def _predict(
    model: KumoRelational,
    query: ValidatedPredictiveQuery,
    num_retries: int = 2,
) -> str:
    with model.retry(num_retries=num_retries):
        with pytest.raises(RuntimeError) as info:
            model.predict(query, indices=[0, 1], verbose=False)
    return str(info.value)


def test_rejected_request_is_not_retried(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    api = FailingAPI(422, _CARDINALITY_DETAIL)
    message = _predict(_model(user_store_graph, api), ltv)
    assert api.n_predict == 1
    assert 'categorical cardinality 15001 exceeds limit 10000' in message
    assert 'create an issue' not in message


def test_unavailable_nim_is_retried(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    api = FailingAPI(503)
    message = _predict(_model(user_store_graph, api), ltv, num_retries=2)
    assert api.n_predict == 3
    assert 'could not complete' in message


def test_rate_limit_is_retried(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    api = FailingAPI(429)
    _predict(_model(user_store_graph, api), ltv, num_retries=1)
    assert api.n_predict == 2


def test_oversized_request_is_not_retried_and_carries_the_remedy(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    r"""413 is the caller's to fix, so it is raised as ValueError rather than
    dressed up as a defect to report.

    Asserting the remedy and not merely the failure is the point: the retry
    classifier already treats 413 as non-transient, so deleting this branch
    would still raise -- just as a RuntimeError with no batch_size advice, and
    nothing would fail.
    """
    api = FailingAPI(413, 'request entity too large')
    model = _model(user_store_graph, api)
    with model.retry(num_retries=2), pytest.raises(ValueError) as info:
        model.predict(ltv, indices=[0, 1], verbose=False)

    message = str(info.value)
    assert api.n_predict == 1, 'a deterministic payload must not be retried'
    assert 'request entity too large' in message
    assert 'carries 2 entities' in message
    assert "'batch_size'" in message
    assert "'num_neighbors'" in message


def test_cardinality_rejection_carries_the_remedy_end_to_end(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    r"""This fixture's payload is far below the limit, so the columns cannot be
    named; the remedy still has to survive the whole predict path. Naming the
    columns is covered against real payloads in ``test_explain_summary``.
    """
    api = FailingAPI(422, _CARDINALITY_DETAIL)
    message = _predict(_model(user_store_graph, api), ltv)
    assert 'Stype.ID' in message
    assert 'does not lift the limit' in message
    assert 'instance_table' not in message
