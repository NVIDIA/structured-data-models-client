# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asking ``kumo-tabular`` for an explanation.

The endpoint explains every scored row, and the client hands each row's
explanation back under the key naming that row.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sdm_client.adapters.tabular import TabularAdapter
from sdm_client.base import check_request_against_model
from sdm_client.models import KUMO_TABULAR

EXPLAIN = ('prediction', 'explanation')
CLASSES = [
    {'kind': 'probability', 'class': '0'},
    {'kind': 'probability', 'class': '1'},
]


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


def test_result_carries_explanation(gradient_explanation):
    """The row's explanation reaches the result under its row_index.

    Assumptions:
    - The endpoint returns one record, with no id, and its explanation.
    """
    explanation = gradient_explanation(
        outputs=[{'kind': 'prediction', 'statistic': 'mean'}],
        row=0,
        columns=['x'],
        gradients=[[0.5]],
    )
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


def answered(request, gradient_explanation, *, keyed=True, outputs=None):
    """Send ``request`` through the adapter and read back a mocked answer.

    Each scored row ``i`` comes back as a NIM answers it: under ``'[<key>]'``
    when the request has a key, with probabilities, and with its own
    explanation, whose gradients are ``-(i + 1)`` and ``i + 1`` per class.
    """
    adapter = TabularAdapter()
    (envelope,) = adapter.prediction_plan(request, EXPLAIN)
    table = envelope['predict']['instance_table']
    records = []
    for index, row in enumerate(table['rows']):
        record = {
            'row_index': index,
            'prediction': 1,
            'probabilities': {'0': 0.25, '1': 0.75},
            'explanation': gradient_explanation(
                outputs=outputs or CLASSES,
                row=index,
                columns=['age'],
                gradients=[[-(index + 1.0)], [index + 1.0]],
            ),
        }
        if keyed:
            record['id'] = f'[{row[table["columns"].index("row_id")]}]'
        records.append(record)
    return adapter.normalize(
        {
            'id': 'r',
            'model': 'kumo-tabular',
            'predictions': records,
            'metadata': {},
        },
        EXPLAIN,
        envelope,
    )


def test_every_row_is_explained_under_its_id(
    tabular_request, gradient_explanation
):
    """Each scored row's explanation is under that row's id, as the frame
    names it.

    Assumptions:
    - The request's rows carry the key ``row_id`` (90 and 91).
    - The NIM answers each row under ``'[<key>]'`` with its own explanation.
    """
    result = answered(tabular_request(rows=2), gradient_explanation)

    assert sorted(result.explanations) == [90, 91]
    assert sorted(result.explanations) == sorted(result.to_pandas()['id'])


def test_each_row_keeps_its_own_gradients(
    tabular_request, gradient_explanation
):
    """A row's instance gradients are its own, as ``[output][column]``.

    Assumptions:
    - The NIM gives row ``i`` the gradients ``-(i + 1)`` and ``i + 1``.
    """
    result = answered(tabular_request(rows=2), gradient_explanation)

    assert result.explanations[90]['instance'] == {
        'columns': ['age'],
        'gradients': [[-1.0], [1.0]],
    }
    assert result.explanations[91]['instance']['gradients'] == [[-2.0], [2.0]]


def test_rows_without_a_key_are_explained_under_their_row_index(
    tabular_request, gradient_explanation
):
    """With no key column, each explanation is under the row's row_index.

    Assumptions:
    - The request has no ``row_id`` column, so the NIM sends no id.
    """
    result = answered(
        tabular_request(rows=3, keyed=False), gradient_explanation, keyed=False
    )

    assert sorted(result.explanations) == [0, 1, 2]
    assert result.explanations[2]['instance']['gradients'] == [[-3.0], [3.0]]


def test_a_tabular_entry_has_no_related_tables(
    tabular_request, gradient_explanation
):
    """kumo-tabular has one table, so an entry carries no ``related`` key.

    Assumptions:
    - The NIM sends an empty ``related_tables``, as it always does here.
    """
    result = answered(tabular_request(rows=2), gradient_explanation)

    for entry in result.explanations.values():
        assert set(entry) == {'outputs', 'instance'}


def test_outputs_and_columns_are_passed_through(
    tabular_request, gradient_explanation
):
    """``outputs`` and ``columns`` are kept as the NIM returned them.

    Assumptions:
    - The NIM names a quantile output with a float level and an extra key,
      which the client does not read or rewrite.
    """
    outputs = [
        {'kind': 'prediction', 'statistic': 'median'},
        {'kind': 'quantile', 'level': 0.1, 'note': 'kept'},
    ]
    result = answered(
        tabular_request(rows=1), gradient_explanation, outputs=outputs
    )

    assert result.explanations[90]['outputs'] == outputs
    assert result.explanations[90]['instance']['columns'] == ['age']


def test_the_flat_table_has_a_line_per_row_output_and_column(
    tabular_request, gradient_explanation
):
    """``to_pandas()`` gives one line per (row, output, column), keyed as the
    frame is, each on the row's own block at hop 0.

    Assumptions:
    - Two rows, two class outputs and one column: four lines.
    """
    result = answered(tabular_request(rows=2), gradient_explanation)

    flat = result.explanations.to_pandas()

    assert list(flat.columns) == [
        'id',
        'table',
        'hop',
        'key',
        'row',
        'output',
        'column',
        'gradient',
    ]
    assert len(flat) == 4
    assert set(flat['table']) == {'instance'}
    assert set(flat['hop']) == {0}
    assert list(flat['key']) == list(flat['id'])
    assert flat.loc[flat['id'] == 91, 'gradient'].tolist() == [-2.0, 2.0]


def test_the_flat_table_names_a_keyless_row_by_its_row_index(
    tabular_request, gradient_explanation
):
    """With no key, ``to_pandas()`` names a row by ``row_index``, and its
    ``key`` is empty.

    Assumptions:
    - The request has no key column.
    """
    result = answered(
        tabular_request(rows=2, keyed=False), gradient_explanation, keyed=False
    )

    flat = result.explanations.to_pandas()

    assert list(flat.columns)[0] == 'row_index'
    assert sorted(set(flat['row_index'])) == [0, 1]
    assert flat['key'].isna().all()


@pytest.mark.parametrize(
    ('task', 'outputs', 'returned'),
    [
        (
            'classification',
            CLASSES,
            {'probabilities': {'0': 0.25, '1': 0.75}},
        ),
        (
            'regression',
            [
                {'kind': 'prediction', 'statistic': 'mean'},
                {'kind': 'quantile', 'level': 0.1},
                {'kind': 'quantile', 'level': 0.9},
            ],
            {'quantiles': {'0.1': 1.0, '0.9': 3.0}},
        ),
    ],
    ids=['probabilities', 'quantiles'],
)
def test_the_flat_table_names_each_output_as_the_frame_does(
    tabular_request, gradient_explanation, task, outputs, returned
):
    """Each explained score is named by the column of ``result.to_pandas()``
    that holds it, so a gradient can be read beside its score.

    Assumptions:
    - The NIM returns the scores it explains: every class probability, or the
      prediction and each asked quantile keyed by its level as text.
    """
    adapter = TabularAdapter()
    (envelope,) = adapter.prediction_plan(tabular_request(task=task), EXPLAIN)
    result = adapter.normalize(
        response(
            id='[90]',
            explanation=gradient_explanation(
                outputs=outputs,
                row=0,
                columns=['age'],
                gradients=[[0.1]] * len(outputs),
            ),
            **returned,
        ),
        EXPLAIN,
        envelope,
    )

    named = result.explanations.to_pandas()['output']

    assert len(set(named)) == len(outputs)
    assert set(named) <= set(result.to_pandas().columns)


def test_the_flat_table_of_no_explanation_is_empty_with_its_columns(
    tabular_request,
):
    """With nothing explained, ``to_pandas()`` is an empty frame that still
    has its columns.

    Assumptions:
    - The response carries no explanation.
    """
    result = TabularAdapter().normalize(response(), ('prediction',))

    flat = result.explanations.to_pandas()

    assert isinstance(flat, pd.DataFrame)
    assert flat.empty
    assert 'gradient' in flat.columns
