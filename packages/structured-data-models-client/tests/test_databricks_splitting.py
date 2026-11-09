# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Databricks serving endpoint drops a call too large as a connection, so a
call past its byte limit is split before it is sent; a 413, the gateway's
400 saying a request is too large, a 504 or a call
that timed out is asked again in halves; a deadline spent waiting on an
unavailable endpoint is a 503, never split; and a 500 is tried once more."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest
import requests
from test_databricks_serving import (
    ENDPOINT,
    ApiClient,
    Endpoints,
    Workspace,
    asked,
)

from sdm_client import SDMClient
from sdm_client.core.databricks_serving import _status_of


def answer(sent: dict[str, Any]) -> dict[str, Any]:
    r"""The served model's answer to ``sent``: one prediction per row."""
    table = sent['predict']['instance_table']
    keys = (
        sent.get('schema', {}).get('instance_table', {}).get('primary_key')
        or []
    )
    at = table['columns'].index(keys[0]) if keys else None
    return {
        'object': 'prediction.response',
        'id': 'pred',
        'model': 'kumo-tabular',
        'metadata': {},
        'predictions': [
            {
                'id': f'[{row[at]}]' if at is not None else str(index),
                'row_index': index,
                'prediction': 'yes',
            }
            for index, row in enumerate(table['rows'])
        ],
    }


class Failing(Endpoints):
    r"""Raises each of ``failures`` on the calls in turn, then answers."""

    def __init__(self, failures: list[BaseException]) -> None:
        super().__init__(answer=answer)
        self.failures = list(failures)

    def query(self, *, name: str, **kwargs: Any) -> Any:
        if self.failures:
            self.calls.append({'name': name, **kwargs})
            raise self.failures.pop(0)
        return super().query(name=name, **kwargs)


class Sized(ApiClient):
    r"""Records the bytes of each envelope posted."""

    sizes: list[int]

    def do(self, method: str, path: str, *, headers, data, raw) -> Any:
        body = data.read()
        self.sizes.append(len(body))
        return super().do(
            method, path, headers=headers, data=io.BytesIO(body), raw=raw
        )


def opened(failures=(), **kwargs: Any) -> tuple[SDMClient, Workspace]:
    workspace = Workspace()
    workspace.serving_endpoints = Failing(list(failures))
    workspace.api_client = Sized(workspace)
    workspace.api_client.sizes = []
    client = SDMClient.for_databricks_serving(
        ENDPOINT, workspace_client=workspace, **kwargs
    )
    return client, workspace


def sdk_error(status: int, reason: str, body: bytes, kind: str) -> OSError:
    r"""The error the installed Databricks SDK raises for an HTTP answer of
    ``status``, read by its own parser, as ``api_client.do`` raises it."""
    parser = pytest.importorskip('databricks.sdk.errors.parser')
    response = requests.Response()
    response.status_code = status
    response.reason = reason
    response._content = body
    response.headers['Content-Type'] = kind
    response.request = requests.Request(
        'POST',
        f'https://example.invalid/serving-endpoints/{ENDPOINT}/invocations',
    ).prepare()
    response.url = response.request.url
    error = parser._Parser().get_api_error(response)
    assert error is not None
    return error


HTML = 'text/html'
JSON = 'application/json'
#: The SDK's errors for each status, as the gateway answers them.
STATUS = {
    413: lambda: sdk_error(
        413,
        'Request Entity Too Large',
        b'<html><pre>too large</pre></html>',
        HTML,
    ),
    502: lambda: sdk_error(
        502, 'Bad Gateway', b'<html><title>Bad Gateway</title></html>', HTML
    ),
    504: lambda: sdk_error(504, 'Gateway Timeout', b'', 'text/plain'),
    500: lambda: sdk_error(
        500,
        'Internal Server Error',
        b'{"error_code": "INTERNAL_ERROR", "message": "boom"}',
        JSON,
    ),
    429: lambda: sdk_error(
        429,
        'Too Many Requests',
        b'{"error_code": "REQUEST_LIMIT_EXCEEDED", "message": "slow"}',
        JSON,
    ),
    # The gateway's refusal of a call too large.
    400: lambda: sdk_error(
        400,
        'Bad Request',
        b'{"error_code": "BAD_REQUEST", "message": "Request size cannot exceed 16MB"}',
        JSON,
    ),
    503: lambda: sdk_error(
        503,
        'Service Unavailable',
        b'{"error_code": "TEMPORARILY_UNAVAILABLE", "message": "later"}',
        JSON,
    ),
}


def test_max_request_bytes_is_passed_through():
    client, _ = opened(max_request_bytes=1234)
    assert client._transport._max_request_bytes == 1234


def test_a_tabular_call_past_the_byte_limit_is_split_before_it_is_sent(caplog):
    whole, workspace = opened()
    whole.predict(asked(), model='kumo-tabular')
    (size,) = workspace.api_client.sizes
    caplog.set_level('INFO', logger='sdm_core')
    split, workspace = opened(max_request_bytes=size - 1)
    result = split.predict(asked(), model='kumo-tabular')
    assert len(workspace.api_client.sizes) == 2
    assert all(sent <= size - 1 for sent in workspace.api_client.sizes)
    assert result.to_pandas()['row_index'].tolist() == [0, 1]
    assert any('smaller calls' in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize('status', [413, 400, 504])
def test_a_tabular_call_refused_as_too_large_is_asked_in_halves(status):
    error = STATUS[status]()
    assert _status_of(error) == status, type(error).__mro__
    client, workspace = opened([error])
    result = client.predict(asked(), model='kumo-tabular')
    calls = workspace.serving_endpoints.calls
    assert len(calls) == 3
    sent = [
        json.loads(c['dataframe_records'][0]['request_json']) for c in calls
    ]
    assert [len(s['predict']['instance_table']['rows']) for s in sent] == [
        2,
        1,
        1,
    ]
    assert result.to_pandas()['row_index'].tolist() == [0, 1]


@pytest.mark.parametrize('status', [500, 502])
def test_a_500_or_a_502_is_tried_once_more_at_its_size(status, monkeypatch):
    from sdm_client.core import databricks_serving

    monkeypatch.setattr(databricks_serving, '_RETRY_PAUSE_SECONDS', 0)
    client, workspace = opened([STATUS[status]()])
    client.predict(asked(), model='kumo-tabular')
    calls = workspace.serving_endpoints.calls
    assert len(calls) == 2
    # Never split: both calls carry every row.
    sent = [
        json.loads(c['dataframe_records'][0]['request_json']) for c in calls
    ]
    assert [len(s['predict']['instance_table']['rows']) for s in sent] == [2, 2]


def test_a_deadline_spent_waiting_is_a_503_and_a_call_timed_out_a_504():
    def timed_out(cause: BaseException | None) -> TimeoutError:
        error = TimeoutError('Timed out after 0:05:00')
        error.__cause__ = cause
        return error

    assert _status_of(timed_out(STATUS[429]())) == 503
    assert _status_of(timed_out(STATUS[503]())) == 503
    assert _status_of(timed_out(requests.ConnectionError('refused'))) == 503
    assert _status_of(timed_out(requests.ReadTimeout('slow'))) == 504
    assert _status_of(timed_out(None)) == 504
    assert _status_of(STATUS[504]()) == 504
