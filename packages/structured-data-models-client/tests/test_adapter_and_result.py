# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a request declares, what a refusal says, and what an empty answer is."""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

from sdm_client import (
    KumoRelational,
    KumoTabular,
    PredictionResult,
    PredictiveQuery,
    Request,
    Schema,
    UnsupportedRequestError,
)
from sdm_client.adapters.relational import RelationalAdapter
from sdm_client.adapters.tabular import TabularAdapter
from sdm_client.errors import StructuredDataError
from sdm_client.models import KUMO_RELATIONAL


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


def predict_recorded(request: Request, outputs=('prediction',)):
    from test_tabular_direct_core import Endpoint, client

    endpoint = Endpoint(
        {
            'model': 'kumo-tabular',
            'id': 'answer',
            'metadata': {},
            'predictions': [
                {'row_index': i, 'prediction': 0}
                for i in range(request.predict_rows)
            ],
        }
    )
    with client(endpoint) as model_client:
        result = model_client.predict(
            request, model=KumoTabular(), outputs=outputs
        )
    return result, endpoint.calls


def envelope(request: Request, outputs=('prediction',)) -> dict:
    result, calls = predict_recorded(request, outputs)
    assert len(result) == request.predict_rows
    assert len(calls) == 1
    return calls[0]


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
    assert list(frame.columns) == ['row_index', 'id', 'prediction']
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
    result, calls = predict_recorded(request)
    assert result.to_pandas().empty
    assert calls == []


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


def relational_request(classes, kind='classification', **options):
    frame = pd.DataFrame(
        {
            'uid': range(100),
            'feature': [i % 7 for i in range(100)],
            'label': [
                f'class-{i % classes}' if kind == 'classification' else float(i)
                for i in range(100)
            ],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'users': frame}, primary_keys={'users': 'uid'}
        )
        return PredictiveQuery(
            'PREDICT users.label FOR EACH users.uid', schema
        ).materialize(
            indices=[0, 1],
            model=KumoRelational(),
            **options,
        )


def predict_relational(request):
    import json

    from test_tabular_direct_core import Endpoint, client

    class RelationalEndpoint(Endpoint):
        def predict_body(self, body):
            payload = json.loads(body)
            self.calls.append(payload)
            table = payload['predict']['instance_table']
            at = table['columns'].index('instance_id')
            return json.dumps(
                {
                    'model': 'kumo-relational',
                    'id': 'answer',
                    'metadata': {},
                    'predictions': [
                        {'row_index': i, 'id': str(row[at]), 'prediction': 1.0}
                        for i, row in enumerate(table['rows'])
                    ],
                }
            ).encode()

    endpoint = RelationalEndpoint()
    with client(endpoint) as model_client:
        result = model_client.predict(request, model=KumoRelational())
    assert len(result) == 2
    assert len(endpoint.calls) == 1
    return endpoint.calls[0]


def test_more_classes_than_the_model_takes_is_refused_locally(monkeypatch):
    from sdm_client import SDMClient

    sent = []

    def unexpected_send(*args):
        sent.append(args)
        raise AssertionError('class limit must be checked before transport')

    monkeypatch.setattr(SDMClient, '_send_body', unexpected_send)
    with pytest.raises(UnsupportedRequestError) as error:
        predict_relational(relational_request(21))
    message = str(error.value)
    assert 'at most 10 classes' in message
    assert '21' in message
    assert sent == []


def test_a_target_within_the_ceiling_is_allowed():
    payload = predict_relational(relational_request(4))
    assert payload['task']['kind'] == 'multiclass_classification'


def test_a_regression_target_is_not_counted():
    payload = predict_relational(relational_request(40, kind='regression'))
    assert payload['task']['kind'] == 'regression'


def test_the_ceiling_is_declared_where_the_other_limits_are():
    assert KUMO_RELATIONAL.runtime.limits['classes_max'] == 10


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


# --- merging keeps the shape ------------------------------------------------


def empty(request_id: str, outputs=('prediction',)) -> PredictionResult:
    return PredictionResult(
        model='kumo-tabular', request_id=request_id, items=(), outputs=outputs
    )


def test_merging_keeps_the_columns_the_request_implied():
    """A split request that scored nothing must still be joinable."""
    merged = PredictionResult.merge([empty('a'), empty('b')])
    assert merged.outputs == ('prediction',)
    assert list(merged.to_pandas().columns) == ['row_index', 'id', 'prediction']


def test_merging_parts_that_wanted_different_outputs_is_refused():
    """They would not be slices of one request."""
    with pytest.raises(StructuredDataError, match='different outputs'):
        PredictionResult.merge([empty('a'), empty('b', ('scores',))])


# --- the inference config the caller handed over ----------------------------


def test_empty_inference_config_is_not_sent():
    payload = predict_relational(relational_request(4, inference_config={}))
    assert 'inference_config' not in payload.get('inference', {})


def test_a_dropped_option_is_refused_with_a_remedy_that_works():
    with pytest.raises(UnsupportedRequestError) as error:
        predict_relational(
            relational_request(
                4, kind='regression', inference_config={'output_type': 'mean'}
            )
        )
    message = str(error.value)
    assert 'output_type' in message
    assert 'drop them from inference_config' in message
    # Following the remedy reaches the endpoint successfully.
    predict_relational(
        relational_request(4, kind='regression', inference_config={})
    )


@dataclasses.dataclass
class RegressionConfig:
    output_type: str = 'median'
    num_estimators: int = 1


@pytest.mark.parametrize(
    ('config', 'expected'),
    [
        (RegressionConfig(), set()),
        (RegressionConfig(output_type='mean'), {'output_type'}),
        ({}, set()),
        ({'output_type': 'mean'}, {'output_type'}),
    ],
)
def test_dataclass_defaults_do_not_become_explicit_inference_options(
    config, expected
):
    from sdm_client.adapters.relational import _keys_the_caller_set

    assert _keys_the_caller_set(config) == expected
