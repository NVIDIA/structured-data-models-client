# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests
from rfm_nim_payloads import NIM_HEALTH_READY_PATH

_JSON_MEDIA_TYPE = 'application/json'
_PROBLEM_MEDIA_TYPE = 'application/problem+json'


def normalize_base_url(base_url: str) -> str:
    """Return a validated HTTP(S) base URL without trailing slashes."""
    candidate = base_url.strip()
    parsed = urlsplit(candidate)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise ValueError(
            'RFM NIM base URL must be an absolute http:// or https:// URL.'
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(
            'RFM NIM base URL cannot contain credentials; use the API key '
            'environment variable.'
        )
    if parsed.query or parsed.fragment:
        raise ValueError('RFM NIM base URL cannot contain a query or fragment.')
    path = parsed.path.rstrip('/')
    return urlunsplit((parsed.scheme, parsed.netloc, path, '', ''))


@dataclass
class LiveNimClient:
    base_url: str
    timeout_seconds: float = 30
    verify_ssl: bool = True
    api_key: str | None = None
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        self.base_url = normalize_base_url(self.base_url)
        if self.api_key:
            self.session.headers['X-API-Key'] = self.api_key

    def request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> requests.Response:
        if not path.startswith('/'):
            raise ValueError('RFM NIM request paths must start with /.')
        kwargs.setdefault('timeout', self.timeout_seconds)
        kwargs.setdefault('verify', self.verify_ssl)
        return self.session.request(
            method,
            f'{self.base_url}{path}',
            **kwargs,
        )


def assert_json_object(response: requests.Response) -> dict[str, Any]:
    assert response.headers.get('content-type', '').startswith(_JSON_MEDIA_TYPE)
    body = response.json()
    assert isinstance(body, dict)
    return body


def assert_problem_details(
    response: requests.Response,
    *,
    expected_statuses: tuple[int, ...] = (400, 422),
) -> dict[str, Any]:
    """Assert only the problem-details fields required by the API contract."""
    assert response.status_code in expected_statuses
    assert response.headers.get('content-type', '').startswith(
        _PROBLEM_MEDIA_TYPE
    )
    body = response.json()
    assert isinstance(body, dict)
    assert body.get('status') == response.status_code
    for key in ('type', 'title', 'detail'):
        assert isinstance(body.get(key), str)
        assert body[key]
    return body


def assert_prediction_response(
    response: requests.Response,
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    """Validate Kumo Relational response semantics without pinning model output."""
    assert response.status_code == 200
    body = assert_json_object(response)
    assert isinstance(body.get('id'), str) and body['id']
    assert body.get('model') == request_payload['model']
    assert isinstance(body.get('metadata'), dict)

    request_table = request_payload['predict']['instance_table']
    columns = request_table['columns']
    rows = request_table['rows']
    instance_id_index = columns.index('instance_id')
    expected_ids = [str(row[instance_id_index]) for row in rows]

    predictions = body.get('predictions')
    assert isinstance(predictions, list)
    assert len(predictions) == len(expected_ids)
    if not predictions:
        return body

    by_row_index: dict[int, dict[str, Any]] = {}
    for prediction in predictions:
        assert isinstance(prediction, dict)
        row_index = prediction.get('row_index')
        assert isinstance(row_index, int) and not isinstance(row_index, bool)
        assert 0 <= row_index < len(expected_ids)
        assert row_index not in by_row_index
        assert prediction.get('id') == expected_ids[row_index]
        by_row_index[row_index] = prediction
    assert set(by_row_index) == set(range(len(expected_ids)))

    requested_fields = set(request_payload['output']['fields'])
    for prediction in predictions:
        for field_name in requested_fields:
            assert field_name in prediction
        probabilities = prediction.get('probabilities')
        if probabilities is not None:
            assert isinstance(probabilities, dict) and probabilities
            values = list(probabilities.values())
            assert all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in values
            )
            assert all(0 <= value <= 1 for value in values)
            assert math.isclose(sum(values), 1.0, rel_tol=1e-5, abs_tol=1e-5)
    return body


def assert_ready_and_model_available(client: LiveNimClient) -> None:
    ready_response = client.request('GET', NIM_HEALTH_READY_PATH)
    assert ready_response.status_code == 200
    ready = assert_json_object(ready_response)
    status = str(ready.get('status', '')).lower()
    check = str(ready.get('check', '')).lower()
    assert status == 'ready' or (status == 'healthy' and check == 'ready')

    models_response = client.request('GET', '/v1/models')
    assert models_response.status_code == 200
    models = assert_json_object(models_response).get('data')
    assert isinstance(models, list)
    assert any(
        isinstance(model, dict) and model.get('id') == 'kumo-relational'
        for model in models
    )
