# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import requests
from rfm_nim_payloads import (
    NIM_HEALTH_READY_PATH,
    NIM_V1_PREDICTION_PATH,
    NIM_V1_SESSIONS_PATH,
    nim_v1_session_create_payload,
    nim_v1_session_predict_minimal_payload,
    nim_v1_smoke_payload,
)
from sdm_api.endpoints import HTTPMethod
from sdm_api.generated.tfm_api import (
    TFM_ENDPOINTS_BY_OPERATION_ID,
    TFMOperations,
)
from sdm_relational_engine.client import NimClient
from sdm_relational_engine.client.rfm import RFMAPI

MOCK_URL = 'https://nim.test'
SDK_V1_MODELS_PATH = '/v1/models'


class JsonRequestCapture:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None
        self.headers: requests.structures.CaseInsensitiveDict[str] | None = None

    def __call__(self, request: Any) -> bool:
        self.payload = request.json()
        self.headers = request.headers
        return True


def test_rfm_api_predict_posts_current_v1_payload_and_parses_response(
    mock_api: Any,
) -> None:
    capture = JsonRequestCapture()
    mock_api.post(
        f'{MOCK_URL}{NIM_V1_PREDICTION_PATH}',
        additional_matcher=capture,
        json={
            'id': 'pred-contract-test',
            'model': 'kumo-relational',
            'predictions': [
                {
                    'id': '601',
                    'row_index': 0,
                    'prediction': True,
                    'probabilities': {
                        'True': 0.65,
                        'False': 0.35,
                    },
                }
            ],
            'metadata': {
                'adapter': 'mock',
            },
        },
    )

    api = RFMAPI(NimClient(MOCK_URL, api_key='DISABLED'))
    payload = nim_v1_smoke_payload()
    result = api.predict(
        payload,
        entity_ids=[601],
        instance_ids=[601],
    )

    assert capture.payload == payload
    assert capture.headers is not None
    assert capture.headers['Content-Type'] == 'application/json'
    assert result.prediction == {
        'columns': ['ENTITY', 'PREDICTION', 'TRUE_PROB', 'FALSE_PROB'],
        'data': [[601, True, 0.65, 0.35]],
    }


def test_rfm_api_predict_reattaches_anchor_times(mock_api: Any) -> None:
    mock_api.post(
        f'{MOCK_URL}{NIM_V1_PREDICTION_PATH}',
        json={
            'id': 'pred-anchor-test',
            'model': 'kumo-relational',
            'predictions': [
                {
                    'id': '601',
                    'row_index': 0,
                    'prediction': True,
                    'probabilities': {
                        'True': 0.65,
                        'False': 0.35,
                    },
                }
            ],
            'metadata': {
                'adapter': 'mock',
            },
        },
    )

    api = RFMAPI(NimClient(MOCK_URL, api_key='DISABLED'))
    result = api.predict(
        nim_v1_smoke_payload(),
        entity_ids=[601],
        instance_ids=[601],
        anchor_times=['2025-02-01T00:00:00Z'],
    )

    assert result.prediction == {
        'columns': [
            'ENTITY',
            'ANCHOR_TIMESTAMP',
            'PREDICTION',
            'TRUE_PROB',
            'FALSE_PROB',
        ],
        'data': [[601, '2025-02-01T00:00:00Z', True, 0.65, 0.35]],
    }


def test_rfm_api_predict_rejects_misaligned_anchor_times(
    mock_api: Any,
) -> None:
    mock_api.post(
        f'{MOCK_URL}{NIM_V1_PREDICTION_PATH}',
        json={
            'id': 'pred-anchor-mismatch-test',
            'model': 'kumo-relational',
            'predictions': [
                {
                    'id': '601',
                    'row_index': 0,
                    'prediction': True,
                }
            ],
            'metadata': {
                'adapter': 'mock',
            },
        },
    )

    api = RFMAPI(NimClient(MOCK_URL, api_key='DISABLED'))
    with pytest.raises(ValueError, match='different lengths'):
        api.predict(
            nim_v1_smoke_payload(),
            entity_ids=[601],
            instance_ids=[601],
            anchor_times=['2025-02-01T00:00:00Z', '2025-02-02T00:00:00Z'],
        )


def test_sdk_prediction_endpoint_matches_current_nim_v1_route() -> None:
    client = NimClient(MOCK_URL, api_key='DISABLED')
    endpoint = TFMOperations.run_prediction.endpoint

    assert endpoint.get_path() == NIM_V1_PREDICTION_PATH
    assert client._format_endpoint_url(endpoint.get_path()) == (
        f'{MOCK_URL}{NIM_V1_PREDICTION_PATH}'
    )


def test_sdk_health_endpoint_matches_current_nim_route() -> None:
    client = NimClient(MOCK_URL, api_key='DISABLED')

    assert NIM_HEALTH_READY_PATH == '/v1/health/ready'
    assert client._format_endpoint_url(NIM_HEALTH_READY_PATH) == (
        f'{MOCK_URL}{NIM_HEALTH_READY_PATH}'
    )


@pytest.mark.parametrize(
    ('operation_id', 'method', 'current_nim_path'),
    [
        ('createSession', HTTPMethod.POST, NIM_V1_SESSIONS_PATH),
        (
            'runSessionPrediction',
            HTTPMethod.POST,
            f'{NIM_V1_SESSIONS_PATH}/{{session_id}}/predictions',
        ),
        (
            'deleteSession',
            HTTPMethod.DELETE,
            f'{NIM_V1_SESSIONS_PATH}/{{session_id}}',
        ),
    ],
)
def test_sdk_generated_session_endpoints_match_current_nim_routes(
    operation_id: str,
    method: HTTPMethod,
    current_nim_path: str,
) -> None:
    client = NimClient(MOCK_URL, api_key='DISABLED')
    endpoint = TFM_ENDPOINTS_BY_OPERATION_ID[operation_id]

    assert endpoint.method == method
    assert client._format_endpoint_url(endpoint.get_path()) == (
        f'{MOCK_URL}{current_nim_path}'
    )


def test_sdk_rest_authenticate_accepts_current_nim_surface(
    mock_api: Any,
) -> None:
    mock_api.get(f'{MOCK_URL}{NIM_HEALTH_READY_PATH}', json={'status': 'ready'})
    mock_api.get(
        f'{MOCK_URL}{SDK_V1_MODELS_PATH}',
        json={
            'data': [
                {
                    'id': 'kumo-relational',
                }
            ]
        },
    )

    client = NimClient(MOCK_URL)
    client.authenticate()


def test_rfm_api_predict_does_not_mutate_request_payload(
    mock_api: Any,
) -> None:
    mock_api.post(
        f'{MOCK_URL}{NIM_V1_PREDICTION_PATH}',
        json={
            'id': 'pred-contract-test',
            'model': 'kumo-relational',
            'predictions': [
                {
                    'id': '601',
                    'row_index': 0,
                }
            ],
            'metadata': {},
        },
    )
    payload = nim_v1_smoke_payload()
    original = deepcopy(payload)

    api = RFMAPI(NimClient(MOCK_URL, api_key='DISABLED'))
    api.predict(
        payload,
        entity_ids=[601],
        instance_ids=[601],
    )

    assert payload == original


def test_rfm_api_predict_maps_varied_prediction_item_shapes(
    mock_api: Any,
) -> None:
    mock_api.post(
        f'{MOCK_URL}{NIM_V1_PREDICTION_PATH}',
        json={
            'id': 'pred-varied-test',
            'model': 'kumo-relational',
            'predictions': [
                {
                    'id': '601',
                    'row_index': 0,
                    'prediction': 'gold',
                    'scores': ['0.5', 1],
                    'rankings': [
                        {
                            'entity_id': 'item-1',
                            'score': 0.9,
                        }
                    ],
                    'embeddings': ['0.1', 0.2],
                    'quantiles': {
                        'p50': '12.5',
                    },
                    'explanation': {
                        'reason': 'fixture',
                    },
                },
                {
                    'id': '602',
                    'row_index': 1,
                    'prediction': 3.14,
                },
                {
                    'id': '603',
                    'row_index': 2,
                    'probabilities': {
                        'A': '0.25',
                        'B': 0.75,
                    },
                },
            ],
            'metadata': {
                'adapter': 'mock',
            },
        },
    )

    api = RFMAPI(NimClient(MOCK_URL, api_key='DISABLED'))
    payload = nim_v1_smoke_payload()
    payload['predict']['instance_table']['rows'] = [
        [601, '2025-02-01T00:00:00Z'],
        [602, '2025-02-02T00:00:00Z'],
        [603, '2025-02-03T00:00:00Z'],
    ]
    result = api.predict(
        payload,
        entity_ids=['account-a', 'account-a', 'account-b'],
        instance_ids=[601, 602, 603],
    )

    assert result.prediction == {
        'columns': [
            'ENTITY',
            'PREDICTION',
            'SCORES',
            'RANKINGS',
            'EMBEDDINGS',
            'Q_P50',
            'EXPLANATION',
            'A_PROB',
            'B_PROB',
        ],
        'data': [
            [
                'account-a',
                'gold',
                [0.5, 1.0],
                [
                    {
                        'entity_id': 'item-1',
                        'score': 0.9,
                    }
                ],
                [0.1, 0.2],
                12.5,
                {
                    'reason': 'fixture',
                },
                None,
                None,
            ],
            [
                'account-a',
                3.14,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            [
                'account-b',
                None,
                None,
                None,
                None,
                None,
                None,
                0.25,
                0.75,
            ],
        ],
    }


def test_rfm_api_predict_renders_multiclass_long_format(
    mock_api: Any,
) -> None:
    mock_api.post(
        f'{MOCK_URL}{NIM_V1_PREDICTION_PATH}',
        json={
            'id': 'pred-multiclass-test',
            'model': 'kumo-relational',
            'predictions': [
                {
                    'id': '601',
                    'row_index': 0,
                    'prediction': 'pro',
                    'probabilities': {
                        'free': 0.1,
                        'pro': 0.6,
                        'enterprise': 0.2,
                        'trial': 0.1,
                    },
                }
            ],
            'metadata': {
                'adapter': 'mock',
                'task_kind': 'multiclass_classification',
            },
        },
    )

    api = RFMAPI(NimClient(MOCK_URL, api_key='DISABLED'))
    result = api.predict(
        nim_v1_smoke_payload(),
        entity_ids=[601],
        instance_ids=[601],
        anchor_times=['2025-02-01T00:00:00Z'],
    )

    assert result.prediction == {
        'columns': [
            'ENTITY',
            'ANCHOR_TIMESTAMP',
            'CLASS',
            'SCORE',
            'PREDICTED',
        ],
        'data': [
            [601, '2025-02-01T00:00:00Z', 'pro', 0.6, True],
            [601, '2025-02-01T00:00:00Z', 'enterprise', 0.2, False],
            [601, '2025-02-01T00:00:00Z', 'free', 0.1, False],
            [601, '2025-02-01T00:00:00Z', 'trial', 0.1, False],
        ],
    }


def test_prediction_response_rejects_bad_probability_shape(
    mock_api: Any,
) -> None:
    mock_api.post(
        f'{MOCK_URL}{NIM_V1_PREDICTION_PATH}',
        json={
            'id': 'pred-bad-probabilities',
            'model': 'kumo-relational',
            'predictions': [
                {
                    'id': '1',
                    # Present so the item parses far enough to reach the
                    # mis-shaped probabilities, which is what this asserts.
                    'row_index': 0,
                    'probabilities': ['not', 'a', 'mapping'],
                }
            ],
            'metadata': {},
        },
    )

    from sdm_relational_engine.exceptions import InvalidResponseError

    api = RFMAPI(NimClient(MOCK_URL, api_key='DISABLED'))
    # A mis-shaped *response* is the server's error, not the caller's, so it
    # surfaces as InvalidResponseError with the driver reason preserved rather
    # than as a bare TypeError from deep inside the parser.
    with pytest.raises(
        InvalidResponseError, match='does not match the contract'
    ) as excinfo:
        api.predict(
            nim_v1_smoke_payload(),
            entity_ids=[601],
            instance_ids=[601],
        )
    # The cause is the generated parser's own contract error rather than a
    # bare TypeError, so the reason names the field shape it expected.
    from sdm_api.generated.tfm_api import ResponseContractError

    assert isinstance(excinfo.value.__cause__, ResponseContractError)
    assert 'requires' in str(excinfo.value.__cause__)


def test_payload_factories_return_isolated_deep_copies() -> None:
    first = nim_v1_smoke_payload()
    second = nim_v1_smoke_payload()

    first['context']['instance_table']['rows'][0][1] = False
    first['schema']['related_tables']['accounts']['columns']['segment'][
        'dtype'
    ] = 'mutated'

    assert second['context']['instance_table']['rows'][0][1] is True
    assert (
        second['schema']['related_tables']['accounts']['columns']['segment'][
            'dtype'
        ]
        == 'string'
    )


def test_container_smoke_payload_tracks_current_v1_wire_shape() -> None:
    payload = nim_v1_smoke_payload()

    assert set(payload) == {
        'model',
        'task',
        'schema',
        'context',
        'predict',
        'output',
        'inference',
    }
    assert 'version' not in payload
    assert 'metadata' not in payload
    assert payload['model'] == 'kumo-relational'
    assert payload['task']['kind'] == 'classification'
    assert payload['output']['fields'] == ['prediction', 'probabilities']
    assert payload['inference'] == {'run_mode': 'best'}


def test_session_payload_helpers_match_current_v1_session_contract() -> None:
    create = nim_v1_session_create_payload()
    predict = nim_v1_session_predict_minimal_payload()

    assert set(create) == {'model', 'task', 'schema', 'context'}
    assert 'predict' not in create
    assert 'output' not in create
    assert 'inference' not in create

    assert set(predict) == {'predict', 'output'}
    assert predict['output']['fields'] == ['prediction', 'probabilities']


class _StubResponse:
    r"""Minimal stand-in for the ``requests`` response the parser reads."""

    def __init__(self, body: Any, error: Exception | None = None) -> None:
        self._body = body
        self._error = error

    def json(self) -> Any:
        if self._error is not None:
            raise self._error
        return self._body


def test_non_json_prediction_body_is_an_invalid_response_error() -> None:
    """A body that will never parse must not be reported as a transient failure
    the caller should retry.
    """
    from sdm_relational_engine.exceptions import InvalidResponseError

    response = _StubResponse(
        None,
        requests.exceptions.JSONDecodeError('Expecting value', '<html>', 0),
    )

    with pytest.raises(InvalidResponseError) as excinfo:
        RFMAPI._parse_predict_response(
            response, entity_ids=[1], instance_ids=['1'], anchor_times=None
        )
    assert 'does not match the contract' in str(excinfo.value)
    assert isinstance(excinfo.value, ValueError)


def test_prediction_item_missing_id_is_an_invalid_response_error() -> None:
    """No bare KeyError escapes."""
    from sdm_relational_engine.exceptions import InvalidResponseError

    response = _StubResponse(
        {
            'id': 'pred-1',
            'model': 'kumo-relational',
            'predictions': [{'row_index': 0, 'prediction': True}],
        }
    )

    with pytest.raises(InvalidResponseError):
        RFMAPI._parse_predict_response(
            response, entity_ids=[1], instance_ids=['1'], anchor_times=None
        )


def test_identity_mapping_mismatch_stays_a_caller_error() -> None:
    r"""Identity mappings describe the request, not the response.

    ``entity_ids``/``instance_ids``/``anchor_times`` are built by the caller, so
    a length mismatch among them must not be reported as a malformed NIM
    response.
    """
    from sdm_relational_engine.exceptions import InvalidResponseError

    body = {
        'id': 'pred-1',
        'model': 'kumo-relational',
        'predictions': [{'id': '1', 'row_index': 0, 'prediction': True}],
    }

    with pytest.raises(ValueError) as excinfo:
        RFMAPI._parse_predict_response(
            _StubResponse(body),
            entity_ids=[1, 2],
            instance_ids=['1'],
            anchor_times=None,
        )
    assert not isinstance(excinfo.value, InvalidResponseError)
    assert 'identity mappings have different lengths' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        RFMAPI._parse_predict_response(
            _StubResponse(body),
            entity_ids=[1],
            instance_ids=['1'],
            anchor_times=[None, None],
        )
    assert not isinstance(excinfo.value, InvalidResponseError)
    assert 'anchor times' in str(excinfo.value)


def test_response_count_mismatch_is_still_an_invalid_response() -> None:
    r"""The neighbouring response-side check keeps its class."""
    from sdm_relational_engine.exceptions import InvalidResponseError

    response = _StubResponse(
        {
            'id': 'pred-1',
            'model': 'kumo-relational',
            'predictions': [],
        }
    )

    with pytest.raises(InvalidResponseError):
        RFMAPI._parse_predict_response(
            response, entity_ids=[1], instance_ids=['1'], anchor_times=None
        )
