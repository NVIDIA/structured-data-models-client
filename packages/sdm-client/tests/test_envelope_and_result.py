# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The payload that goes out, and the frame that comes back."""

from __future__ import annotations

import pandas as pd
import pytest
from sdm_api.generated.tfm_api import PredictionItem

from sdm_client import PredictionResult, Request
from sdm_client.adapters.tabular import TabularAdapter
from sdm_client.errors import StructuredDataError


def request_for(task: str = 'classification', **kwargs) -> Request:
    n = 12
    context = pd.DataFrame(
        {
            'row_id': range(n),
            'age': [20.0 + i for i in range(n)],
            'label': [i % 2 for i in range(n)],
        }
    )
    predict = pd.DataFrame({'row_id': [90, 91], 'age': [33.0, 41.0]})
    return Request(
        context=context,
        predict=predict,
        target='label',
        task=task,
        **kwargs,
    )


def envelope(task: str = 'classification', outputs=('prediction',)) -> dict:
    return next(TabularAdapter().prediction_plan(request_for(task), outputs))


def test_the_envelope_carries_every_section_the_contract_requires():
    body = envelope()
    for section in ('model', 'task', 'schema', 'context', 'predict', 'output'):
        assert section in body, f'missing {section}'
    assert body['model'] == 'kumo-tabular'


def test_tables_are_encoded_as_the_arrays_format():
    body = envelope()
    table = body['context']['instance_table']
    assert table['format'] == 'arrays'
    assert table['columns'] == ['row_id', 'age', 'label']
    assert len(table['rows']) == 12
    assert table['rows'][0] == [0, 20.0, 0]


def test_the_target_dtype_matches_the_schema_dtype_for_that_column():
    """They must be equal; the endpoint answers 422 when they are not."""
    body = envelope()
    target = body['task']['target']
    declared = body['schema']['instance_table']['columns'][
        target['column_name']
    ]
    assert target['dtype'] == declared['dtype']


def test_a_classification_target_is_declared_categorical_not_numerical():
    """Inference is task-agnostic, so an integer label arrives numerical."""
    columns = envelope('classification')['schema']['instance_table']['columns']
    assert columns['label']['stype'] == 'categorical'


def test_a_regression_target_is_not_forced_to_categorical():
    columns = envelope('regression')['schema']['instance_table']['columns']
    assert columns['label']['stype'] == 'numerical'


def _text_request(distinct: int = 200) -> Request:
    n = 400
    context = pd.DataFrame(
        {
            'row_id': range(n),
            'code': [f'RES-{i % distinct:05d}' for i in range(n)],
            'region': [f'R{i % 3}' for i in range(n)],
            'label': [i % 2 for i in range(n)],
        }
    )
    predict = pd.DataFrame(
        {
            'row_id': [90, 91],
            'code': ['RES-00001', 'RES-00002'],
            'region': ['R0', 'R1'],
        }
    )
    return Request(
        context=context, predict=predict, target='label', task='classification'
    )


def test_a_text_column_is_sent_as_categorical_because_tabular_cannot_read_text():
    """`text` is refused outright by the endpoint; `categorical` never is.

    So the column travels as labels rather than costing the whole request a
    422 that names no column.
    """
    request = _text_request()
    assert request.schema['instance']['code'].stype == 'text'
    with pytest.warns(UserWarning, match='cannot read a text column'):
        body = next(TabularAdapter().prediction_plan(request, ('prediction',)))
    columns = body['schema']['instance_table']['columns']
    assert columns['code']['stype'] == 'categorical'
    assert columns['code']['dtype'] == 'string'


def test_the_warning_names_every_rewritten_column_once():
    """Two real datasets hit 9 and 13 such columns; one warning, not thirteen."""
    request = _text_request()
    request.schema['instance']['region'].stype = 'text'
    with pytest.warns(UserWarning) as caught:
        next(TabularAdapter().prediction_plan(request, ('prediction',)))
    messages = [
        str(w.message) for w in caught if 'cannot read a text' in str(w.message)
    ]
    assert len(messages) == 1
    assert 'code' in messages[0] and 'region' in messages[0]


def test_the_task_kind_is_sent_as_given_and_never_narrowed():
    assert envelope('classification')['task']['kind'] == 'classification'


def test_a_session_pins_the_context_and_scoring_sends_only_the_rows():
    adapter = TabularAdapter()
    request = request_for()
    pinned = adapter.session_plan(request, ('prediction',))
    assert set(pinned) == {'model', 'task', 'schema', 'context', 'metadata'}
    assert 'predict' not in pinned

    scoring = next(adapter.session_prediction_plan(request, ('prediction',)))
    assert set(scoring) == {'predict', 'output', 'metadata'}
    assert 'context' not in scoring


def test_every_request_carries_an_id_that_correlates_it_with_server_logs():
    assert envelope()['metadata']['request_id'].startswith('sdm-')


def test_inference_knobs_travel_only_when_given():
    assert 'inference' not in envelope()
    request = request_for(inference={'num_context_samples': 500})
    body = next(TabularAdapter().prediction_plan(request, ('prediction',)))
    assert body['inference'] == {'num_context_samples': 500}


def result_with(*items: dict) -> PredictionResult:
    return PredictionResult(
        model='kumo-tabular',
        request_id='r-1',
        items=[PredictionItem.from_dict(item) for item in items],
        metadata={},
    )


def test_a_classification_result_spreads_probabilities_into_columns():
    frame = result_with(
        {
            'row_index': 0,
            'prediction': 0,
            'probabilities': {'0': 0.6, '1': 0.4},
        },
        {
            'row_index': 1,
            'prediction': 1,
            'probabilities': {'0': 0.3, '1': 0.7},
        },
    ).to_pandas()
    assert list(frame.columns) == [
        'row_index',
        'prediction',
        'probability_0',
        'probability_1',
    ]
    assert frame.loc[0, 'probability_0'] == pytest.approx(0.6)


def test_a_regression_result_does_not_invent_probability_columns():
    frame = result_with(
        {'row_index': 0, 'prediction': 1.5},
        {'row_index': 1, 'prediction': 2.5},
    ).to_pandas()
    assert 'probability_0' not in frame.columns
    assert list(frame['prediction']) == [1.5, 2.5]


def test_rows_come_back_ordered_by_row_index_whatever_order_they_arrived_in():
    """The endpoint is not required to preserve request order."""
    frame = result_with(
        {'row_index': 2, 'prediction': 'c'},
        {'row_index': 0, 'prediction': 'a'},
        {'row_index': 1, 'prediction': 'b'},
    ).to_pandas()
    assert list(frame['row_index']) == [0, 1, 2]
    assert list(frame['prediction']) == ['a', 'b', 'c']


def test_the_entity_key_is_carried_when_the_model_has_one():
    frame = result_with(
        {'row_index': 0, 'id': 'user-7', 'prediction': 1}
    ).to_pandas()
    assert frame.loc[0, 'id'] == 'user-7'


def test_an_explanation_is_absent_rather_than_a_different_return_type():
    result = result_with({'row_index': 0, 'prediction': 1})
    assert result.explanation is None
    assert isinstance(result.to_pandas(), pd.DataFrame)


def test_quantiles_are_spread_under_a_readable_prefix():
    frame = result_with(
        {'row_index': 0, 'prediction': 2.0, 'quantiles': {'0.5': 2.0}}
    ).to_pandas()
    assert 'quantile_0.5' in frame.columns


def test_a_split_request_keeps_every_batch_rather_than_the_last():
    """A request past the endpoint's cap is several calls, one answer."""
    first = result_with({'row_index': 0, 'prediction': 'a'})
    second = result_with({'row_index': 1, 'prediction': 'b'})
    merged = PredictionResult.merge([first, second])
    assert len(merged) == 2
    assert list(merged.to_pandas()['prediction']) == ['a', 'b']
    assert merged.metadata['batches'] == 2


def test_merging_one_part_returns_it_unchanged():
    only = result_with({'row_index': 0, 'prediction': 'a'})
    assert PredictionResult.merge([only]) is only


def test_merging_nothing_is_an_error_rather_than_an_empty_answer():
    with pytest.raises(StructuredDataError):
        PredictionResult.merge([])


def test_a_malformed_two_hundred_stays_inside_the_client_error_contract():
    """A body that does not match the contract is still this client's failure."""
    from sdm_client.errors import StructuredDataError

    adapter = TabularAdapter()
    with pytest.raises(StructuredDataError) as error:
        adapter.normalize(
            {'model': 'kumo-tabular', 'predictions': [], 'metadata': {}},
            ('prediction',),
        )
    assert error.value.code == 'INVALID_RESPONSE'

    with pytest.raises(StructuredDataError):
        adapter.normalize(
            {
                'id': 'r',
                'model': 'kumo-tabular',
                'predictions': [None],
                'metadata': {},
            },
            ('prediction',),
        )


def test_a_forecast_that_carries_no_explanation_says_so():
    """The single-request path reports a field the answer did not carry, and
    the decomposed path returned its merge without passing through it, so a
    forecast asked for an explanation and given none said nothing."""
    import warnings

    from sdm_api.generated.tfm_api import PredictionItem

    from sdm_client.client import _by_step, _report_absent
    from sdm_client.decompose import Decomposition
    from sdm_client.result import PredictionResult

    answers = [
        PredictionResult(
            model='kumo-relational',
            request_id='r',
            items=[PredictionItem(row_index=0, prediction=1.0)],
            outputs=('prediction', 'explanation'),
            explanation=None,
        )
        for _ in range(2)
    ]
    merged = _by_step(
        answers, Decomposition('forecast', (('q1', 1), ('q2', 2)))
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _report_absent(merged, ['prediction', 'explanation'], 'kumo-relational')

    assert any('explanation' in str(item.message) for item in caught)
