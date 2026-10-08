# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sdm_client import KumoRelational, PredictiveQuery, StructuredDataError

_SKILLS = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / 'tests/test_skills.py')
)
_QUERY = (
    'PREDICT LIST_DISTINCT(orders.item, 0, 30, days) RANK TOP 2 '
    'FOR EACH users.user_id'
)


@pytest.mark.parametrize(
    'factory', [list, tuple, range, pd.Series, np.array, iter]
)
def test_candidates_are_iterated_at_the_public_materialize_boundary(factory):
    candidates = factory(4) if factory is range else factory(range(4))
    request = PredictiveQuery(
        _QUERY, _SKILLS['schema'].__wrapped__()
    ).materialize(indices=[1, 2], model=KumoRelational(), candidates=candidates)
    assert request.task.kind == 'binary_classification'
    assert len(request.parts) == 4
    assert [value for _, value in request._decomposition.parts] == [0, 1, 2, 3]


def test_candidate_iteration_is_after_query_validation():
    class Unreadable:
        def __iter__(self):
            raise AssertionError('candidates were read before the query')

    with pytest.raises(StructuredDataError, match='not a query'):
        PredictiveQuery(
            'not a query', _SKILLS['schema'].__wrapped__()
        ).materialize(
            indices=[1, 2], model=KumoRelational(), candidates=Unreadable()
        )


def test_candidate_iteration_preserves_the_original_error():
    original = RuntimeError('candidate iteration failed')

    class Unreadable:
        def __iter__(self):
            raise original

    with pytest.raises(RuntimeError) as caught:
        PredictiveQuery(_QUERY, _SKILLS['schema'].__wrapped__()).materialize(
            indices=[1, 2], model=KumoRelational(), candidates=Unreadable()
        )
    assert caught.value is original


def test_date_array_condition_reaches_public_materialization():
    request = _SKILLS['resolve'](
        _SKILLS['filtered']('IN', 'a time'), _SKILLS['schema'].__wrapped__()
    )
    assert request.task.kind == 'regression'
