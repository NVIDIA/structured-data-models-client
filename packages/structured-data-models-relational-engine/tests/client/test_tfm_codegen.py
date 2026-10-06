# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import fields
from pathlib import Path

import pytest
from sdm_api.endpoints import HTTPMethod
from sdm_api.generated.tfm_api import (
    TFM_ENDPOINTS_BY_OPERATION_ID,
    TFM_MODEL_KUMO_RELATIONAL,
    TFM_OUTPUT_FIELD_EMBEDDINGS,
    TFM_OUTPUT_FIELD_EXPLANATION,
    TFM_SCHEMA_NAMES,
    PredictionItem,
    PredictionResponse,
    ResponseContractError,
    TFMOperations,
)
from sdm_relational_engine.client import NimClient
from sdm_relational_engine.client.rfm import (
    _prediction_item_to_row,
    _prediction_response_to_rfm,
)

# The Universal TFM OpenAPI contract is not vendored here, so the tests that
# replay it skip unless a checkout is pointed at explicitly. The default is the
# sibling-directory layout the maintainers use; anyone else sets
# KUMO_RELATIONAL_CONTRACT_DIR.
_ENV_CONTRACT_DIR = 'KUMO_RELATIONAL_CONTRACT_DIR'
CANONICAL_SPEC = (
    Path(os.environ.get(_ENV_CONTRACT_DIR, '../structured-data-api'))
    / 'nim-sd.openapi.yaml'
)


def test_generated_tfm_api_runtime_metadata() -> None:
    operation = TFMOperations.run_prediction

    assert TFM_MODEL_KUMO_RELATIONAL == 'kumo-relational'
    assert TFM_OUTPUT_FIELD_EMBEDDINGS == 'embeddings'
    assert TFM_OUTPUT_FIELD_EXPLANATION == 'explanation'
    assert operation.operation_id == 'runPrediction'
    assert operation.request_schema == 'PredictionRequest'
    assert operation.response_schema == 'PredictionResponse'
    assert operation.endpoint.method == HTTPMethod.POST
    assert operation.endpoint.get_path() == '/v1/predictions'
    assert TFM_ENDPOINTS_BY_OPERATION_ID['runPrediction'] == (
        operation.endpoint
    )
    assert 'HealthResponse' not in TFM_SCHEMA_NAMES
    assert 'ProblemDetails' in TFM_SCHEMA_NAMES


def test_generated_tfm_api_paths_are_service_root_relative() -> None:
    client = NimClient('https://example.test', api_key=None)

    assert client._format_endpoint_url(
        TFMOperations.run_prediction.endpoint.get_path()
    ) == ('https://example.test/v1/predictions')

    with pytest.raises(ValueError, match='must start'):
        client._format_endpoint_url('rfm/validate_query')


def test_generated_prediction_response_parser() -> None:
    response = PredictionResponse.from_dict(
        {
            'id': 'pred-1',
            'model': 'kumo-relational',
            'predictions': [
                {
                    # A string, as the contract types it and as the NIM sends
                    # it: an integer-keyed entity comes back as '180', not 180.
                    'id': '7',
                    # Integers, not strings that look like them: the contract
                    # types row_index and forecast_step as int32, and its
                    # base-10-string rule is scoped to table cells, which these
                    # are not. Coercing them let a row_index of 1.2 become
                    # row 1.
                    'row_index': 3,
                    'forecast_step': 2,
                    'prediction': True,
                    'probabilities': {
                        'false': 0.25,
                        'true': 0.75,
                    },
                    'scores': [0.4, '0.6'],
                    'rankings': [
                        {
                            'id': 11,
                            'score': '0.7',
                        }
                    ],
                    'embeddings': [0.1, 0.2],
                    'quantiles': {
                        '0.5': '1.25',
                    },
                    'explanation': {
                        'format': 'natural_language_summary',
                        'summary': 'Order frequency dropped.',
                    },
                    'metadata': {
                        'adapter_status': 'stubbed',
                    },
                }
            ],
            'metadata': {
                'version': 'v1',
                'task_kind': 'classification',
            },
        }
    )

    assert response.id == 'pred-1'
    assert response.model == 'kumo-relational'
    assert response.metadata['task_kind'] == 'classification'
    assert len(response.predictions) == 1
    item = response.predictions[0]
    assert item.id == '7'
    assert item.row_index == 3
    assert item.forecast_step == 2
    assert item.prediction is True
    assert item.probabilities == {'false': 0.25, 'true': 0.75}
    assert item.scores == (0.4, 0.6)
    assert item.rankings == ({'id': 11, 'score': '0.7'},)
    assert item.embeddings == (0.1, 0.2)
    assert item.quantiles == {'0.5': 1.25}
    assert item.explanation == {
        'format': 'natural_language_summary',
        'summary': 'Order frequency dropped.',
    }
    assert item.metadata == {'adapter_status': 'stubbed'}


def test_prediction_item_adapter_maps_known_fields() -> None:
    item = PredictionItem(
        id='7',
        row_index=3,
        prediction='yes',
        probabilities={
            'no': 0.2,
            'yes': 0.8,
        },
        scores=(0.3, 0.7),
        rankings=(
            {
                'id': 'merchant-1',
                'score': 0.9,
            },
        ),
        embeddings=(0.1, 0.2),
        quantiles={
            'p50': 12.5,
        },
        explanation={
            'format': 'natural_language_summary',
            'summary': 'Recent purchases increased.',
        },
        metadata={
            'adapter_status': 'stubbed',
        },
    )

    row = _prediction_item_to_row(item, entity_id='customer-7')

    assert row == {
        'ENTITY': 'customer-7',
        'PREDICTION': 'yes',
        'NO_PROB': 0.2,
        'YES_PROB': 0.8,
        'SCORES': [0.3, 0.7],
        'RANKINGS': [
            {
                'id': 'merchant-1',
                'score': 0.9,
            }
        ],
        'EMBEDDINGS': [0.1, 0.2],
        'Q_P50': 12.5,
        'EXPLANATION': {
            'format': 'natural_language_summary',
            'summary': 'Recent purchases increased.',
        },
    }

    mapped_fields = {
        'id',
        'row_index',
        'prediction',
        'probabilities',
        'scores',
        'rankings',
        'embeddings',
        'quantiles',
        'explanation',
    }
    parsed_fields = {field.name for field in fields(PredictionItem)}
    intentionally_unmapped_fields = {'forecast_step', 'metadata'}
    assert parsed_fields == mapped_fields | intentionally_unmapped_fields


def test_prediction_response_correlates_opaque_ids_to_repeated_entities() -> (
    None
):
    response = PredictionResponse(
        id='pred-1',
        model='kumo-relational',
        predictions=(
            PredictionItem(id='21', row_index=1, prediction='second'),
            PredictionItem(id='20', row_index=0, prediction='first'),
        ),
        metadata={'task_kind': 'regression'},
    )

    converted = _prediction_response_to_rfm(
        response,
        entity_ids=('user-7', 'user-7'),
        instance_ids=(20, 21),
    )

    assert converted.prediction == {
        'columns': ['ENTITY', 'PREDICTION'],
        'data': [
            ['user-7', 'first'],
            ['user-7', 'second'],
        ],
    }


def test_ranking_response_expands_rankings_to_compatibility_rows() -> None:
    response = PredictionResponse(
        id='pred-rank-1',
        model='kumo-relational',
        predictions=(
            PredictionItem(
                id='20',
                row_index=0,
                rankings=(
                    {'id': 'item-a', 'score': 0.9},
                    {'id': 'item-b', 'score': 0.7},
                ),
            ),
        ),
        metadata={
            'task_kind': 'temporal_link_prediction',
            'output_type': 'rankings',
        },
    )

    converted = _prediction_response_to_rfm(
        response,
        entity_ids=('user-7',),
        instance_ids=(20,),
    )

    assert converted.prediction == {
        'columns': ['ENTITY', 'CLASS', 'SCORE'],
        'data': [
            ['user-7', 'item-a', 0.9],
            ['user-7', 'item-b', 0.7],
        ],
    }


@pytest.mark.parametrize(
    ('item', 'error'),
    [
        (
            PredictionItem(
                id='20',
                row_index=0,
                prediction='item-a',
                rankings=({'id': 'item-a', 'score': 0.9},),
            ),
            'must not include prediction',
        ),
        (PredictionItem(id='20', row_index=0), 'missing rankings'),
        (
            PredictionItem(id='20', row_index=0, rankings=({'score': 0.9},)),
            'missing id',
        ),
        (
            PredictionItem(id='20', row_index=0, rankings=({'id': 'item-a'},)),
            'missing score',
        ),
        (
            PredictionItem(
                id='20',
                row_index=0,
                rankings=({'id': 7, 'score': 0.9},),
            ),
            'id must be a string',
        ),
        (
            PredictionItem(
                id='20',
                row_index=0,
                rankings=({'id': 'item-a', 'score': 0.9, 'label': 'x'},),
            ),
            'unexpected fields',
        ),
    ],
)
def test_ranking_response_rejects_malformed_rankings(
    item: PredictionItem,
    error: str,
) -> None:
    response = PredictionResponse(
        id='pred-rank-1',
        model='kumo-relational',
        predictions=(item,),
        metadata={
            'task_kind': 'temporal_link_prediction',
            'output_type': 'rankings',
        },
    )

    with pytest.raises(ValueError, match=error):
        _prediction_response_to_rfm(
            response,
            entity_ids=('user-7',),
            instance_ids=(20,),
        )


def test_forecast_response_allows_multiple_records_per_request_row() -> None:
    response = PredictionResponse(
        id='pred-forecast-1',
        model='kumo-relational',
        predictions=(
            PredictionItem(
                id='20',
                row_index=0,
                prediction=10.0,
                quantiles={'0.5': 10.0},
                forecast_step=1,
            ),
            PredictionItem(
                id='20',
                row_index=0,
                prediction=12.0,
                quantiles={'0.5': 12.0},
                forecast_step=2,
            ),
        ),
        metadata={
            'task_kind': 'forecasting',
            'output_type': 'forecast',
        },
    )

    converted = _prediction_response_to_rfm(
        response,
        entity_ids=('item-42',),
        instance_ids=(20,),
    )

    assert converted.prediction == {
        'columns': ['ENTITY', 'PREDICTION', 'Q_0.5', 'FORECAST_STEP'],
        'data': [
            ['item-42', 10.0, 10.0, 1],
            ['item-42', 12.0, 12.0, 2],
        ],
    }


@pytest.mark.parametrize(
    ('predictions', 'error'),
    [
        (
            (PredictionItem(id='20', row_index=0, prediction=10.0),),
            'missing forecast_step',
        ),
        (
            (
                PredictionItem(
                    id='20', row_index=0, prediction=10.0, forecast_step=1
                ),
                PredictionItem(
                    id='20', row_index=0, prediction=11.0, forecast_step=1
                ),
            ),
            'duplicate forecast_step',
        ),
    ],
)
def test_forecast_response_rejects_malformed_steps(
    predictions: tuple[PredictionItem, ...],
    error: str,
) -> None:
    response = PredictionResponse(
        id='pred-forecast-1',
        model='kumo-relational',
        predictions=predictions,
        metadata={
            'task_kind': 'forecasting',
            'output_type': 'forecast',
        },
    )

    with pytest.raises(ValueError, match=error):
        _prediction_response_to_rfm(
            response,
            entity_ids=('item-42',),
            instance_ids=(20,),
        )


def test_a_prediction_item_cannot_be_built_without_its_row_index() -> None:
    """The binding refuses it, so the engine never sees a keyless item.

    This used to be one of the response-validation cases below: an item with no
    ``row_index`` was constructible and the engine reported it downstream. The
    contract makes the field required, so the generated binding now refuses the
    construction outright and the invalid state cannot be reached.
    """
    with pytest.raises(TypeError):
        PredictionItem(id='20', prediction=1.0)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ('predictions', 'entity_ids', 'instance_ids', 'error'),
    [
        (
            (PredictionItem(id='20', row_index=1, prediction=1.0),),
            (3,),
            (20,),
            'row_index is out of range',
        ),
        (
            (PredictionItem(id='wrong', row_index=0, prediction=1.0),),
            (3,),
            (20,),
            'id does not match request instance_id',
        ),
        (
            (
                PredictionItem(id='20', row_index=0, prediction=1.0),
                PredictionItem(id='20', row_index=0, prediction=2.0),
            ),
            (3, 3),
            (20, 21),
            'duplicate row_index',
        ),
        (
            (PredictionItem(id='20', row_index=-1, prediction=1.0),),
            (3,),
            (20,),
            'row_index is out of range',
        ),
        (
            (PredictionItem(row_index=0, prediction=1.0),),
            (3,),
            (20,),
            'id does not match request instance_id',
        ),
        (
            (
                PredictionItem(id='20', row_index=0, prediction=1.0),
                PredictionItem(id='20', row_index=1, prediction=2.0),
            ),
            (3, 4),
            (20, 21),
            'id does not match request instance_id',
        ),
        (
            (PredictionItem(id='20', row_index=0, prediction=1.0),),
            (3, 4),
            (20, 21),
            'response count does not match the request',
        ),
    ],
)
def test_prediction_response_rejects_invalid_correlation(
    predictions: tuple[PredictionItem, ...],
    entity_ids: tuple[object, ...],
    instance_ids: tuple[object, ...],
    error: str,
) -> None:
    response = PredictionResponse(
        id='pred-1',
        model='kumo-relational',
        predictions=predictions,
        metadata={'task_kind': 'regression'},
    )

    with pytest.raises(ValueError, match=error):
        _prediction_response_to_rfm(
            response,
            entity_ids=entity_ids,
            instance_ids=instance_ids,
        )


# The generator, its drift checks and the canonical-spec comparison used to live
# here. They moved with the code: `structured-data-models-api` owns the generated bindings now, and
# `packages/structured-data-models-api/tests/test_generated_bindings.py` plus
# `scripts/check_contract_pin.py` cover them. Keeping a second copy here meant
# tests that only passed when run from one directory.


def test_a_prediction_id_that_is_not_a_string_is_refused() -> None:
    """The contract types the entity id as a string and the NIM sends one, so
    a number here is a server that is not keeping the contract.

    Coercing it produced an id the caller never sent, which then correlated
    against their rows as though it had.
    """
    with pytest.raises(ResponseContractError):
        PredictionResponse.from_dict(
            {
                'id': 'pred-1',
                'model': 'kumo-relational',
                'predictions': [{'id': 7, 'row_index': 0}],
                'metadata': {},
            }
        )
