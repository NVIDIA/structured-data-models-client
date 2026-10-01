# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import quote

import pytest
import requests
from rfm_nim_hardening_cases import (
    EXPECTED_REJECTION_CASES,
    REGRESSION_REJECTION_CASES,
    ExpectedRejectionCase,
    RejectionCase,
)
from rfm_nim_live_harness import (
    LiveNimClient,
    assert_prediction_response,
    assert_problem_details,
    assert_ready_and_model_available,
)
from rfm_nim_payloads import (
    NIM_V1_PREDICTION_PATH,
    NIM_V1_SESSIONS_PATH,
    nim_v1_empty_predict_rows_payload,
    nim_v1_explicit_utc_offset_timestamp_payload,
    nim_v1_fast_run_mode_payload,
    nim_v1_multiclass_payload,
    nim_v1_prediction_only_output_payload,
    nim_v1_regression_payload,
    nim_v1_reordered_predict_rows_payload,
    nim_v1_session_create_payload,
    nim_v1_session_predict_minimal_payload,
    nim_v1_smoke_payload,
    nim_v1_text_stringlist_payload,
    nim_v1_two_predict_rows_payload,
    nim_v1_without_inference_payload,
)
from sdm_relational_engine.client import NimClient
from sdm_relational_engine.client.rfm import RFMAPI

# The client package and the end-to-end script gate on the KUMO_RELATIONAL_*
# names; accept those first so one export runs every live suite, and keep the
# older RFM_NIM_* spellings working.
_ENV_VAR = 'KUMO_RELATIONAL_NIM_BASE_URL'
_LEGACY_ENV_VAR = 'RFM_NIM_BASE_URL'


def _live_url() -> str | None:
    return os.environ.get(_ENV_VAR) or os.environ.get(_LEGACY_ENV_VAR)


def _live_api_key() -> str | None:
    return os.environ.get('KUMO_RELATIONAL_NIM_API_KEY') or os.environ.get(
        'RFM_NIM_API_KEY'
    )


_PREFLIGHT_ERROR: str | None = None

pytestmark = [
    pytest.mark.live_nim,
    pytest.mark.skipif(
        not _live_url(),
        reason=f'set {_ENV_VAR} to run live KumoRelational NIM tests',
    ),
]


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.lower() not in {'0', 'false', 'no'}


@pytest.fixture(scope='module')
def live_nim() -> Iterator[LiveNimClient]:
    client = LiveNimClient(
        _live_url(),
        timeout_seconds=float(os.environ.get('RFM_NIM_TIMEOUT_SECONDS', '30')),
        verify_ssl=_env_bool('RFM_NIM_VERIFY_SSL', True),
        api_key=_live_api_key(),
    )
    yield client
    client.session.close()


@pytest.fixture(scope='module', autouse=True)
def live_nim_preflight(live_nim: LiveNimClient) -> None:
    global _PREFLIGHT_ERROR
    if _PREFLIGHT_ERROR is not None:
        pytest.skip(f'RFM NIM preflight already failed: {_PREFLIGHT_ERROR}')
    try:
        assert_ready_and_model_available(live_nim)
    except (requests.RequestException, AssertionError) as exc:
        _PREFLIGHT_ERROR = str(exc) or (
            'readiness or model-discovery response violated the live contract'
        )
        pytest.fail(
            f'could not reach RFM NIM at {live_nim.base_url}: '
            f'{_PREFLIGHT_ERROR}',
            pytrace=False,
        )


def _post_prediction(
    live_nim: LiveNimClient,
    payload: dict[str, Any],
) -> requests.Response:
    return live_nim.request(
        'POST',
        NIM_V1_PREDICTION_PATH,
        json=payload,
    )


def _assert_service_recovers(live_nim: LiveNimClient) -> None:
    assert_ready_and_model_available(live_nim)
    payload = nim_v1_smoke_payload()
    assert_prediction_response(_post_prediction(live_nim, payload), payload)


def _cleanup_created_session(
    live_nim: LiveNimClient,
    response: requests.Response,
) -> None:
    if response.status_code != 201:
        return
    session_id = str(response.json()['session_id'])
    session_url = f'{NIM_V1_SESSIONS_PATH}/{quote(session_id, safe="")}'
    cleanup = live_nim.request('DELETE', session_url)
    assert cleanup.status_code == 204


def _assert_rejection_and_recovers(
    live_nim: LiveNimClient,
    case: RejectionCase,
) -> None:
    response = live_nim.request('POST', case.path, **case.request_kwargs())
    contract_error: Exception | None = None
    try:
        assert_problem_details(
            response,
            expected_statuses=case.expected_statuses,
        )
    except Exception as exc:  # Preserve the contract failure after recovery.
        contract_error = exc

    _cleanup_created_session(live_nim, response)
    _assert_service_recovers(live_nim)
    if contract_error is not None:
        raise contract_error


@pytest.mark.live_nim_smoke
def test_live_nim_smoke_preflight(live_nim: LiveNimClient) -> None:
    assert live_nim.base_url


@pytest.mark.live_nim_smoke
def test_live_nim_smoke_public_sdk_prediction(
    live_nim: LiveNimClient,
) -> None:
    client = NimClient(
        live_nim.base_url,
        api_key=live_nim.api_key,
        verify_ssl=live_nim.verify_ssl,
    )
    client.authenticate()

    payload = nim_v1_smoke_payload()
    result = RFMAPI(client).predict(
        payload,
        entity_ids=['account-601'],
        instance_ids=[601],
    )

    prediction = result.prediction
    assert isinstance(prediction, dict)
    assert isinstance(prediction.get('columns'), list)
    assert isinstance(prediction.get('data'), list)
    assert len(prediction['data']) == 1
    entity_index = prediction['columns'].index('ENTITY')
    assert prediction['data'][0][entity_index] == 'account-601'
    assert 'PREDICTION' in prediction['columns']


@pytest.mark.live_nim_smoke
def test_live_nim_smoke_session_lifecycle(
    live_nim: LiveNimClient,
) -> None:
    created = live_nim.request(
        'POST',
        NIM_V1_SESSIONS_PATH,
        json=nim_v1_session_create_payload(),
    )
    assert created.status_code == 201
    created_body = created.json()
    session_id = str(created_body['session_id'])
    assert session_id
    assert created_body['ttl_seconds'] > 0
    assert created_body['expires_at']

    session_url = f'{NIM_V1_SESSIONS_PATH}/{quote(session_id, safe="")}'
    session_predictions_url = f'{session_url}/predictions'
    deleted = False
    try:
        payload = nim_v1_session_predict_minimal_payload()
        response = live_nim.request(
            'POST',
            session_predictions_url,
            json=payload,
        )
        full_payload = nim_v1_smoke_payload()
        full_payload['predict'] = payload['predict']
        full_payload['output'] = payload['output']
        assert_prediction_response(response, full_payload)

        delete_response = live_nim.request('DELETE', session_url)
        assert delete_response.status_code == 204
        deleted = True

        second_delete = live_nim.request('DELETE', session_url)
        assert second_delete.status_code == 204

        after_delete = live_nim.request(
            'POST',
            session_predictions_url,
            json=payload,
        )
        assert_problem_details(after_delete, expected_statuses=(404,))
    finally:
        if not deleted:
            cleanup = live_nim.request('DELETE', session_url)
            assert cleanup.status_code == 204


@pytest.mark.live_nim_full
@pytest.mark.parametrize(
    'payload_factory',
    [
        pytest.param(
            nim_v1_prediction_only_output_payload,
            id='prediction-only',
        ),
        pytest.param(nim_v1_fast_run_mode_payload, id='fast-run-mode'),
        pytest.param(nim_v1_without_inference_payload, id='default-inference'),
        pytest.param(nim_v1_two_predict_rows_payload, id='two-rows'),
        pytest.param(
            nim_v1_reordered_predict_rows_payload,
            id='reordered-rows',
        ),
        pytest.param(nim_v1_empty_predict_rows_payload, id='empty-batch'),
        pytest.param(
            nim_v1_explicit_utc_offset_timestamp_payload,
            id='timestamp-offsets',
        ),
        pytest.param(
            nim_v1_text_stringlist_payload,
            id='text-stringlist',
        ),
        pytest.param(nim_v1_regression_payload, id='regression'),
        pytest.param(nim_v1_multiclass_payload, id='multiclass'),
    ],
)
def test_live_nim_full_prediction_variants(
    live_nim: LiveNimClient,
    payload_factory: Callable[[], dict[str, Any]],
) -> None:
    payload = payload_factory()
    assert_prediction_response(_post_prediction(live_nim, payload), payload)


def _invalid_run_mode_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['inference']['run_mode'] = 'turbo'
    return payload


def _unsupported_output_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['output']['fields'].append('feature_importances')
    return payload


def _unsupported_task_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['task']['kind'] = 'not-a-task'
    return payload


def _missing_schema_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload.pop('schema')
    return payload


def _missing_target_column_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['task']['target']['column_name'] = 'missing_status'
    return payload


def _invalid_timestamp_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['predict']['instance_table']['rows'][0][1] = 'not-a-timestamp'
    return payload


def _relationship_mismatch_payload() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['schema']['relationships'][0]['source_columns'] = [
        'missing_instance_id',
    ]
    return payload


@pytest.mark.live_nim_full
@pytest.mark.parametrize(
    'payload_factory',
    [
        pytest.param(_invalid_run_mode_payload, id='invalid-run-mode'),
        pytest.param(_unsupported_output_payload, id='unsupported-output'),
        pytest.param(_unsupported_task_payload, id='unsupported-task'),
        pytest.param(_missing_schema_payload, id='missing-required-schema'),
        pytest.param(
            _missing_target_column_payload,
            id='missing-target-column',
        ),
        pytest.param(_invalid_timestamp_payload, id='invalid-timestamp'),
        pytest.param(
            _relationship_mismatch_payload,
            id='relationship-mismatch',
        ),
    ],
)
def test_live_nim_full_rejects_invalid_requests_and_recovers(
    live_nim: LiveNimClient,
    payload_factory: Callable[[], dict[str, Any]],
) -> None:
    response = _post_prediction(live_nim, payload_factory())
    assert_problem_details(response)
    _assert_service_recovers(live_nim)


@pytest.mark.live_nim_full
@pytest.mark.parametrize(
    'case',
    EXPECTED_REJECTION_CASES,
    ids=lambda case: case.case_id,
)
def test_live_nim_full_expected_http_rejections(
    live_nim: LiveNimClient,
    case: ExpectedRejectionCase,
) -> None:
    response = live_nim.request(
        case.method,
        case.path,
        **case.request_kwargs(),
    )
    if case.problem_details:
        assert_problem_details(
            response,
            expected_statuses=(case.expected_status,),
        )
    else:
        assert response.status_code == case.expected_status
    _assert_service_recovers(live_nim)


@pytest.mark.live_nim_full
@pytest.mark.parametrize(
    'case',
    REGRESSION_REJECTION_CASES,
    ids=lambda case: case.case_id,
)
def test_live_nim_full_regression_rejections(
    live_nim: LiveNimClient,
    case: RejectionCase,
) -> None:
    _assert_rejection_and_recovers(live_nim, case)


@pytest.mark.live_nim_full
def test_live_nim_full_accepts_request_metadata(
    live_nim: LiveNimClient,
) -> None:
    payload = nim_v1_smoke_payload()
    payload['metadata'] = {'request_id': 'rfm-sdk-live-validation'}
    assert_prediction_response(_post_prediction(live_nim, payload), payload)


@pytest.mark.live_nim_full
def test_live_nim_full_session_validation_and_cleanup(
    live_nim: LiveNimClient,
) -> None:
    created = live_nim.request(
        'POST',
        NIM_V1_SESSIONS_PATH,
        json=nim_v1_session_create_payload(),
    )
    assert created.status_code == 201
    session_id = str(created.json()['session_id'])
    session_url = f'{NIM_V1_SESSIONS_PATH}/{quote(session_id, safe="")}'
    try:
        payload = nim_v1_session_predict_minimal_payload()
        payload['predict']['related_tables'].pop('accounts')
        response = live_nim.request(
            'POST',
            f'{session_url}/predictions',
            json=payload,
        )
        assert_problem_details(response)
        _assert_service_recovers(live_nim)
    finally:
        cleanup = live_nim.request('DELETE', session_url)
        assert cleanup.status_code == 204
