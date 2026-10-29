# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asking ``kumo-tabular`` for an explanation.

The endpoint explains every scored row, and the client hands each row's
explanation back under the key naming that row.
"""

from __future__ import annotations

import pytest

from sdm_client.adapters.tabular import TabularAdapter
from sdm_client.base import check_request_against_model
from sdm_client.models import KUMO_TABULAR

EXPLAIN = ('prediction', 'explanation')


def response(**record) -> dict:
    """A one-record ``kumo-tabular`` response carrying ``record``'s fields."""
    return {
        'id': 'r',
        'model': 'kumo-tabular',
        'predictions': [{'row_index': 0, 'prediction': 1, **record}],
        'metadata': {},
    }


@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_allowed(tabular_request, task):
    """kumo-tabular accepts explanation as an output on both of its tasks.

    Assumptions:
    - The request scores one row.
    """
    check_request_against_model(
        tabular_request(task=task), KUMO_TABULAR, EXPLAIN
    )


def test_one_row_sent(tabular_request):
    """An explanation over one row is sent with explanation in its fields.

    Assumptions:
    - The request scores exactly one row.
    """
    (envelope,) = TabularAdapter().prediction_plan(tabular_request(), EXPLAIN)
    assert envelope['output']['fields'] == list(EXPLAIN)


def test_rows_sent(tabular_request):
    """An explanation over two rows is sent with explanation in its fields.

    Assumptions:
    - The endpoint explains every scored row.
    """
    (envelope,) = TabularAdapter().prediction_plan(
        tabular_request(rows=2), EXPLAIN
    )
    assert envelope['output']['fields'] == list(EXPLAIN)


def test_pin_sends_no_rows(tabular_request):
    """Pinning a session context for explanations sends no predict rows."""
    pinned = TabularAdapter().session_plan(tabular_request(rows=2), EXPLAIN)
    assert 'predict' not in pinned


def test_session_rows_sent(tabular_request):
    """Session scoring sends an explanation over two rows.

    Assumptions:
    - The endpoint explains every scored row.
    """
    (envelope,) = TabularAdapter().session_prediction_plan(
        tabular_request(rows=2), EXPLAIN
    )
    assert envelope['output']['fields'] == list(EXPLAIN)


def test_result_carries_explanation():
    """The row's explanation reaches the result under its row_index.

    Assumptions:
    - The endpoint returns one record, with no id, and its explanation.
    """
    explanation = {
        'format': 'sdm_gradient_v1',
        'details': {
            'outputs': [{'kind': 'prediction', 'statistic': 'mean'}],
            'instance_table': {
                'row_indices': [0],
                'columns': ['x'],
                'gradients': [[[0.5]]],
            },
            'related_tables': {},
        },
    }
    result = TabularAdapter().normalize(
        response(explanation=explanation), EXPLAIN
    )
    assert result.explanations == {
        0: {
            'outputs': [{'kind': 'prediction', 'statistic': 'mean'}],
            'instance': {'columns': ['x'], 'gradients': [[0.5]]},
        }
    }


def test_result_without_explanation():
    """A response without an explanation leaves the result's explanations
    empty.

    Assumptions:
    - The record carries no explanation field.
    """
    result = TabularAdapter().normalize(response(), ('prediction',))
    assert result.explanations == {}
