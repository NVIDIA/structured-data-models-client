# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest
from sdm_api.endpoints import HTTPMethod
from sdm_api.generated.tfm_api import (
    TFM_ENDPOINTS_BY_OPERATION_ID,
    TFM_MODEL_KUMO_RELATIONAL,
    TFM_OUTPUT_FIELD_EMBEDDINGS,
    TFM_OUTPUT_FIELD_EXPLANATION,
    TFM_OUTPUT_FIELD_VALUES,
    TFM_SCHEMA_NAMES,
    PredictionItem,
    PredictionResponse,
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
                    'id': 7,
                    'row_index': '3',
                    'forecast_step': '2',
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


def test_generator_creates_minimal_bindings(tmp_path: Path) -> None:
    spec = tmp_path / 'api_spec.json'
    output = tmp_path / 'generated.py'
    spec.write_text(json.dumps(_minimal_openapi_spec()))

    subprocess.run(
        [
            sys.executable,
            'scripts/generate_tfm_api.py',
            '--spec',
            str(spec),
            '--output',
            str(output),
        ],
        check=True,
    )

    generated = output.read_text()
    assert 'Source: ' in generated
    assert 'class TFMOperations' in generated
    assert 'class PredictionResponse' in generated
    assert 'run_prediction: Final[TFMOperation]' in generated
    assert "path='/v1/predictions'" in generated
    assert (
        "TFM_MODEL_KUMO_RELATIONAL: Final[str] = 'kumo-relational'" in generated
    )
    assert "TFM_OUTPUT_FIELD_EMBEDDINGS: Final[str] = 'embeddings'" in generated
    assert (
        "TFM_OUTPUT_FIELD_EXPLANATION: Final[str] = 'explanation'" in generated
    )

    subprocess.run(
        [
            sys.executable,
            'scripts/generate_tfm_api.py',
            '--spec',
            str(spec),
            '--output',
            str(output),
            '--check',
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            'scripts/generate_tfm_api.py',
            '--spec',
            str(spec),
            '--output',
            str(output),
            '--validate-contract',
        ],
        check=True,
    )

    output.write_text(generated + '\n# stale edit\n')
    result = subprocess.run(
        [
            sys.executable,
            'scripts/generate_tfm_api.py',
            '--spec',
            str(spec),
            '--output',
            str(output),
            '--check',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert 'not up to date' in result.stderr


def test_generator_source_label_is_repository_relative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.generate_tfm_api import generate_code_from_source

    source_repo = tmp_path / 'source-repo'
    (source_repo / '.git').mkdir(parents=True)
    spec = source_repo / 'specs' / 'api_spec.json'
    spec.parent.mkdir()
    spec.write_text(json.dumps(_minimal_openapi_spec()))
    monkeypatch.chdir(tmp_path)

    relative_code = generate_code_from_source(
        'source-repo/specs/api_spec.json',
        output_path=tmp_path / 'relative.py',
    )
    absolute_code = generate_code_from_source(
        str(spec),
        output_path=tmp_path / 'absolute.py',
    )

    assert relative_code == absolute_code
    assert '# Source: specs/api_spec.json' in relative_code
    assert str(tmp_path) not in relative_code


def test_generator_validation_reports_spec_drift(tmp_path: Path) -> None:
    from scripts.generate_tfm_api import (
        generate_code,
        validate_generated_code,
    )

    spec = _minimal_openapi_spec()
    spec_text = json.dumps(spec)
    code = generate_code(
        spec,
        spec_text=spec_text,
        source='inline',
        output_path=tmp_path / 'generated.py',
    )
    assert validate_generated_code(spec, code) == []

    del spec['components']['schemas']['PredictionItem']['properties'][
        'metadata'
    ]
    errors = validate_generated_code(spec, code)
    assert any('PredictionItem fields differ' in error for error in errors)


def test_generator_validation_reports_output_enum_drift(tmp_path: Path) -> None:
    from scripts.generate_tfm_api import (
        generate_code,
        validate_generated_code,
    )

    spec = _minimal_openapi_spec()
    spec_text = json.dumps(spec)
    code = generate_code(
        spec,
        spec_text=spec_text,
        source='inline',
        output_path=tmp_path / 'generated.py',
    )
    assert validate_generated_code(spec, code) == []

    spec['components']['schemas']['OutputSpec']['properties']['fields'][
        'items'
    ]['enum'].append('attributions')
    errors = validate_generated_code(spec, code)
    assert any('TFM_OUTPUT_FIELD_VALUES differs' in error for error in errors)


@pytest.mark.skipif(
    not CANONICAL_SPEC.exists(),
    reason=f'contract spec not found at {CANONICAL_SPEC}; set '
    f'{_ENV_CONTRACT_DIR} to a checkout to run this',
)
def test_generated_tfm_api_matches_local_canonical_spec() -> None:
    spec = _load_local_canonical_spec_at_generated_revision()
    from scripts.generate_tfm_api import generate_code_from_source

    output = Path('src/sdm_relational_engine/client/generated/tfm_api.py')
    expected = generate_code_from_source(
        str(CANONICAL_SPEC),
        output_path=output,
    )
    assert output.read_text() == expected

    # Keep the loaded spec live so the skip guard above cannot be accidentally
    # removed without also updating this test.
    assert spec['paths']['/v1/predictions']['post']['operationId'] == (
        'runPrediction'
    )


@pytest.mark.skipif(
    not CANONICAL_SPEC.exists(),
    reason=f'contract spec not found at {CANONICAL_SPEC}; set '
    f'{_ENV_CONTRACT_DIR} to a checkout to run this',
)
def test_generated_tfm_api_contract_matches_local_canonical_spec() -> None:
    spec = _load_local_canonical_spec_at_generated_revision()
    from scripts.generate_tfm_api import validate_generated_code

    output = Path('src/sdm_relational_engine/client/generated/tfm_api.py')
    assert validate_generated_code(spec, output.read_text()) == []

    schemas = spec['components']['schemas']
    assert tuple(field.name for field in fields(PredictionItem)) == tuple(
        schemas['PredictionItem']['properties']
    )
    assert tuple(field.name for field in fields(PredictionResponse)) == tuple(
        schemas['PredictionResponse']['properties']
    )
    assert set(schemas['PredictionItem'].get('required', [])) <= {
        field.name for field in fields(PredictionItem)
    }
    assert set(schemas['PredictionResponse']['required']) <= {
        field.name for field in fields(PredictionResponse)
    }
    assert (
        tuple(schemas['OutputSpec']['properties']['fields']['items']['enum'])
        == TFM_OUTPUT_FIELD_VALUES
    )
    assert TFMOperations.run_prediction.response_schema == (
        'PredictionResponse'
    )


@pytest.mark.skipif(
    not CANONICAL_SPEC.exists(),
    reason=f'contract spec not found at {CANONICAL_SPEC}; set '
    f'{_ENV_CONTRACT_DIR} to a checkout to run this',
)
def test_documented_prediction_response_examples_parse() -> None:
    spec = _load_local_canonical_spec_at_generated_revision()
    examples = spec['paths']['/v1/predictions']['post']['responses']['200'][
        'content'
    ]['application/json']['examples']
    required = set(
        spec['components']['schemas']['PredictionResponse']['required']
    )

    for name, example in examples.items():
        response = PredictionResponse.from_dict(example['value'])
        assert required <= set(example['value'])
        assert response.id
        assert response.model
        assert response.metadata['task_kind']
        assert response.predictions, name
        for item in response.predictions:
            assert item.id is None or isinstance(item.id, str)
            assert item.id is not None or item.row_index is not None
            if item.row_index is not None:
                assert isinstance(item.row_index, int)
            if item.probabilities is not None:
                assert all(
                    isinstance(value, float)
                    for value in item.probabilities.values()
                )
            if item.embeddings is not None:
                assert isinstance(item.embeddings, tuple)
            if item.scores is not None:
                assert isinstance(item.scores, tuple)
            if item.explanation is not None:
                assert isinstance(item.explanation, dict)
            if item.metadata is not None:
                assert isinstance(item.metadata, dict)


@pytest.mark.skipif(
    not CANONICAL_SPEC.exists(),
    reason=f'contract spec not found at {CANONICAL_SPEC}; set '
    f'{_ENV_CONTRACT_DIR} to a checkout to run this',
)
def test_documented_prediction_request_examples_match_envelope_shape() -> None:
    spec = _load_local_canonical_spec_at_generated_revision()
    request_schema = spec['components']['schemas']['PredictionRequest']
    examples = spec['paths']['/v1/predictions']['post']['requestBody'][
        'content'
    ]['application/json']['examples']
    property_names = set(request_schema['properties'])
    required = set(request_schema['required'])

    assert {'kumo_tabular_arrays', 'kumo_relational_arrays'} <= set(examples)
    for example in examples.values():
        value = example['value']
        assert set(value) <= property_names
        assert required <= set(value)
        _assert_request_envelope_shape(value)


def _generated_source_sha(path: Path) -> str:
    match = re.search(
        r'^# Source SHA256: ([0-9a-f]+)$', path.read_text(), flags=re.MULTILINE
    )
    assert match is not None
    return match.group(1)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_local_canonical_spec_at_generated_revision() -> dict:
    yaml = pytest.importorskip('yaml')
    output = Path('src/sdm_relational_engine/client/generated/tfm_api.py')
    generated_source_sha = _generated_source_sha(output)
    if generated_source_sha != _file_sha256(CANONICAL_SPEC):
        pytest.skip(
            'the local contract checkout is not the generated spec revision'
        )
    return yaml.safe_load(CANONICAL_SPEC.read_text())


def _assert_request_envelope_shape(value: dict) -> None:
    assert value['model'] in {'kumo-tabular', 'kumo-relational'}
    assert 'kind' in value['task']
    assert 'instance_table' in value['schema']
    assert 'instance_table' in value['context']
    assert 'instance_table' in value['predict']
    assert isinstance(value['output']['fields'], list)
    assert value['output']['fields']
    assert 'operation' not in value.get('metadata', {})
    assert 'evaluate' not in value


def _minimal_openapi_spec() -> dict:
    return {
        'openapi': '3.0.0',
        'paths': {
            '/v1/predictions': {
                'post': {
                    'operationId': 'runPrediction',
                    'summary': 'Run a one-shot TFM prediction',
                    'requestBody': {
                        'content': {
                            'application/json': {
                                'schema': {
                                    '$ref': '#/components/schemas/PredictionRequest',
                                },
                            },
                        },
                    },
                    'responses': {
                        '200': {
                            'content': {
                                'application/json': {
                                    'schema': {
                                        '$ref': '#/components/schemas/'
                                        'PredictionResponse',
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
        'components': {
            'schemas': {
                'PredictionRequest': {
                    'required': [
                        'version',
                        'model',
                        'task',
                        'schema',
                        'context',
                        'predict',
                        'output',
                    ],
                    'properties': {
                        'version': {
                            'enum': ['v1'],
                        },
                        'model': {
                            'enum': ['kumo-tabular', 'kumo-relational'],
                        },
                        'task': {},
                        'schema': {},
                        'context': {},
                        'predict': {},
                        'output': {},
                        'inference': {},
                        'metadata': {},
                    },
                },
                'PredictionItem': {
                    'properties': {
                        'id': {},
                        'row_index': {},
                        'forecast_step': {},
                        'prediction': {},
                        'probabilities': {},
                        'scores': {},
                        'rankings': {},
                        'embeddings': {},
                        'quantiles': {},
                        'explanation': {},
                        'metadata': {},
                    },
                },
                'PredictionResponse': {
                    'required': [
                        'id',
                        'model',
                        'predictions',
                        'metadata',
                    ],
                    'properties': {
                        'id': {},
                        'model': {},
                        'predictions': {},
                        'metadata': {},
                    },
                },
                'OutputSpec': {
                    'properties': {
                        'fields': {
                            'items': {
                                'enum': [
                                    'prediction',
                                    'probabilities',
                                    'embeddings',
                                    'explanation',
                                ],
                            },
                        },
                    },
                },
                'TaskSpec': {
                    'properties': {
                        'kind': {
                            'enum': ['classification', 'regression'],
                        },
                    },
                },
            },
        },
    }
