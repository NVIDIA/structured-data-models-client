# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated Python response bindings enforce field types and preserve values."""

import pytest
from sdm_api.generated.tfm_api import PredictionResponse, ResponseContractError


def envelope():
    return {
        'id': 'response',
        'model': 'kumo-relational',
        'metadata': {},
        'predictions': [{'id': 'row', 'row_index': 0, 'prediction': 1.25}],
    }


def test_typed_values_and_wire_order():
    body = envelope()
    body['predictions'] = [
        {
            'id': 'second',
            'row_index': 1,
            'prediction': 2.5,
            'scores': [0.25, 0.75],
        },
        {'id': 'first', 'row_index': 0.0, 'prediction': 1.25, 'unused': True},
    ]
    body['unused'] = [1, 2]
    got = PredictionResponse.from_dict(body)
    assert got.id == 'response'
    assert got.model == 'kumo-relational'
    assert [item.row_index for item in got.predictions] == [1, 0]
    assert [item.id for item in got.predictions] == ['second', 'first']
    assert [item.prediction for item in got.predictions] == [2.5, 1.25]
    assert tuple(got.predictions[0].scores) == (0.25, 0.75)


@pytest.mark.parametrize('items', [[], {}])
def test_empty_prediction_collection(items):
    body = envelope()
    body['predictions'] = items
    assert not PredictionResponse.from_dict(body).predictions


@pytest.mark.parametrize('missing', [False, True])
def test_optional_item_id(missing):
    body = envelope()
    if missing:
        del body['predictions'][0]['id']
    else:
        body['predictions'][0]['id'] = None
    assert PredictionResponse.from_dict(body).predictions[0].id is None


@pytest.mark.parametrize('field', ['id', 'model', 'predictions', 'metadata'])
def test_required_envelope_fields(field):
    body = envelope()
    del body[field]
    with pytest.raises((ResponseContractError, TypeError, ValueError)):
        PredictionResponse.from_dict(body)


@pytest.mark.parametrize(
    'field, value',
    [
        ('id', 1),
        ('model', 1),
        ('predictions', 'bad'),
        ('predictions', 1),
        ('predictions', {'x': 1}),
        ('predictions', [1]),
        ('metadata', []),
    ],
)
def test_invalid_envelope_fields(field, value):
    body = envelope()
    body[field] = value
    with pytest.raises((ResponseContractError, TypeError, ValueError)):
        PredictionResponse.from_dict(body)


@pytest.mark.parametrize(
    'field, value',
    [
        ('id', 1),
        ('row_index', None),
        ('row_index', 0.25),
        ('row_index', True),
        ('row_index', '0'),
        ('row_index', float('inf')),
        ('forecast_step', 0.25),
    ],
)
def test_invalid_item_fields(field, value):
    body = envelope()
    body['predictions'][0][field] = value
    with pytest.raises(
        (ResponseContractError, TypeError, ValueError, OverflowError)
    ):
        PredictionResponse.from_dict(body)


def test_nonobject_response_and_missing_row_index():
    with pytest.raises((ResponseContractError, TypeError, ValueError)):
        PredictionResponse.from_dict([])
    body = envelope()
    del body['predictions'][0]['row_index']
    with pytest.raises((ResponseContractError, TypeError, ValueError)):
        PredictionResponse.from_dict(body)
