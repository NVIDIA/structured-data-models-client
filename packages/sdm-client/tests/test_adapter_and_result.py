# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a request declares, what a refusal says, and what an empty answer is."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdm_client import (
    KumoRelational,
    PredictionResult,
    Request,
    UnsupportedRequestError,
)
from sdm_client.adapters.relational import RelationalAdapter
from sdm_client.adapters.tabular import TabularAdapter
from sdm_client.errors import StructuredDataError


def context(rows: int = 6, **columns) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            'row_id': range(rows),
            'v': [float(index) for index in range(rows)],
            'y': [index % 2 for index in range(rows)],
        }
    )
    for name, values in columns.items():
        frame[name] = values
    return frame


def envelope(request: Request, outputs=('prediction',)) -> dict:
    return next(iter(TabularAdapter().prediction_plan(request, outputs)))


# --- a column with no values decides nothing --------------------------------


@pytest.mark.parametrize(
    ('scored', 'expected'),
    [([42], 'int64'), ([3.5], 'float64'), (['alpha'], 'string')],
)
def test_an_empty_context_column_takes_the_type_of_the_rows_to_score(
    scored, expected
):
    """It used to be called 'string' first, which sent 42 as the text '42'."""
    request = Request(
        context=context(x=[None] * 6),
        predict=pd.DataFrame({'row_id': [900], 'v': [1.0], 'x': scored}),
        target='y',
        task='classification',
    )
    plan = envelope(request)
    assert plan['schema']['instance_table']['columns']['x']['dtype'] == expected
    assert plan['predict']['instance_table']['rows'] == [[900, 1.0, scored[0]]]


def test_a_column_empty_in_both_frames_is_still_a_string():
    """With no values anywhere there is nothing to infer, and string carries
    whatever arrives."""
    request = Request(
        context=context(x=[None] * 6),
        predict=pd.DataFrame({'row_id': [900], 'v': [1.0], 'x': [None]}),
        target='y',
        task='classification',
    )
    plan = envelope(request)
    assert plan['schema']['instance_table']['columns']['x']['dtype'] == 'string'


def test_an_ordinary_column_is_unaffected():
    request = Request(
        context=context(x=[1.0] * 6),
        predict=pd.DataFrame({'row_id': [900], 'v': [1.0], 'x': [2.0]}),
        target='y',
        task='classification',
    )
    plan = envelope(request)
    assert (
        plan['schema']['instance_table']['columns']['x']['dtype'] == 'float64'
    )


# --- a numpy scalar is a number ---------------------------------------------


def test_a_numpy_scalar_in_inference_becomes_a_python_one():
    """It used to die inside the JSON encoder as a bare TypeError."""
    import json

    request = Request(
        context=context(),
        predict=pd.DataFrame({'row_id': [900], 'v': [1.0]}),
        target='y',
        task='classification',
        inference={'seed': np.int64(4), 'nested': {'a': np.float64(1.5)}},
    )
    plan = envelope(request)
    assert plan['inference'] == {'seed': 4, 'nested': {'a': 1.5}}
    assert json.loads(json.dumps(plan['inference']))


# --- an empty answer is still an answer -------------------------------------


def test_an_empty_result_carries_the_columns_the_request_implies():
    result = PredictionResult(
        model='kumo-tabular',
        request_id='r',
        items=(),
        outputs=('prediction',),
    )
    frame = result.to_pandas()
    assert list(frame.columns) == ['row_index', 'prediction']
    assert frame.empty


def test_an_empty_result_can_be_joined_on_row_index():
    frame = PredictionResult(
        model='kumo-tabular', request_id='r', items=(), outputs=('prediction',)
    ).to_pandas()
    to_score = pd.DataFrame({'a': []})
    joined = to_score.reset_index(names='row_index').merge(
        frame, on='row_index', how='left'
    )
    assert joined.empty


def test_an_empty_context_is_refused_rather_than_sent():
    with pytest.raises(StructuredDataError) as error:
        Request(
            context=context(0),
            predict=pd.DataFrame({'row_id': [900], 'v': [1.0]}),
            target='y',
            task='classification',
        )
    assert 'no rows' in str(error.value)


def test_an_empty_predict_frame_is_still_allowed():
    """The contract puts no minimum on rows, and a no-op batch is legitimate."""
    request = Request(
        context=context(),
        predict=pd.DataFrame(
            {
                'row_id': pd.Series(dtype='int64'),
                'v': pd.Series(dtype='float64'),
            }
        ),
        target='y',
        task='classification',
    )
    assert envelope(request)['predict']['instance_table']['rows'] == []


# --- what the endpoint enforces, said locally -------------------------------


def relational_payload(*, classes: int, kind: str = 'classification') -> dict:
    return {
        'task': {'kind': kind, 'target': {'column_name': 'TARGET'}},
        'schema': {
            'instance_table': {
                'columns': {
                    'TARGET': {'dtype': 'string', 'stype': 'categorical'}
                }
            },
            'related_tables': {
                'orders': {
                    'columns': {'note': {'dtype': 'string', 'stype': 'text'}}
                }
            },
        },
        'context': {
            'instance_table': {
                'columns': ['TARGET'],
                'rows': [[f'class-{index % classes}'] for index in range(50)],
            }
        },
        'predict': {'instance_table': {'columns': ['TARGET'], 'rows': [['x']]}},
    }


class _Gathered:
    def __init__(self, *payloads: dict) -> None:
        self.payloads = tuple(payloads)
        self.requested_inference: dict = {}
        self.inference: dict = {}


def test_more_classes_than_the_model_takes_is_refused_locally():
    request = _Gathered(relational_payload(classes=21))
    with pytest.raises(UnsupportedRequestError) as error:
        list(RelationalAdapter().prediction_plan(request, ('prediction',)))
    message = str(error.value)
    assert 'at most 10 classes' in message
    assert '21' in message


def test_a_target_within_the_ceiling_is_allowed():
    request = _Gathered(relational_payload(classes=4))
    assert list(RelationalAdapter().prediction_plan(request, ('prediction',)))


def test_a_regression_target_is_not_counted():
    request = _Gathered(relational_payload(classes=40, kind='regression'))
    assert list(RelationalAdapter().prediction_plan(request, ('prediction',)))


def test_the_ceiling_is_declared_where_the_other_limits_are():
    assert KumoRelational().spec.runtime.limits['classes_max'] == 10


def test_an_unsupported_dtype_refusal_names_the_columns():
    """The endpoint reports the type and not the column. The payload knows."""
    from sdm_client.errors import NimRequestError

    error = NimRequestError(
        422,
        code='UNSUPPORTED_DTYPE',
        message="semantic type 'text' is unsupported",
    )
    enriched = RelationalAdapter().explain_error(
        error, relational_payload(classes=3)
    )
    message = str(enriched)
    assert 'orders.note' in message
    assert "stype = 'categorical'" in message
    assert enriched.status_code == 422


def test_an_unrelated_error_is_returned_unchanged():
    from sdm_client.errors import NimRequestError

    error = NimRequestError(500, code='INTERNAL_ERROR', message='boom')
    assert (
        RelationalAdapter().explain_error(error, relational_payload(classes=3))
        is error
    )


def test_the_tabular_adapter_does_not_pretend_to_explain():
    error = StructuredDataError('boom', code='X')
    assert TabularAdapter().explain_error(error, {}) is error
