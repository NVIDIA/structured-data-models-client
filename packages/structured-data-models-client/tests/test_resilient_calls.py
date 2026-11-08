# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The client holds a call to no size of its own and recovers from what the
endpoint says: a call refused as too large, by a gateway, or timed out is
asked again in halves, each with the whole context; a 422 is not."""

from __future__ import annotations

import json

import pytest
import requests
from test_limits_and_absent_outputs import gathered
from test_tabular_direct_core import client

from sdm_client.core.transport import _to_nim_error
from sdm_client.errors import StructuredDataError


class Scripted:
    r"""An endpoint that fails the first call as ``failure`` says, then
    answers every call with a prediction for each row it scores."""

    url = 'http://localhost'

    def __init__(self, failure: Exception) -> None:
        self.failure: Exception | None = failure
        self.calls: list[dict] = []

    def predict_body(self, body: bytes) -> bytes:
        payload = json.loads(body)
        self.calls.append(payload)
        if self.failure is not None:
            failure, self.failure = self.failure, None
            raise failure
        table = payload['predict']['instance_table']
        key = table['columns'].index('instance_id')
        return json.dumps(
            {
                'id': 'answer',
                'model': 'kumo-relational',
                'metadata': {},
                'predictions': [
                    {'id': str(row[key]), 'row_index': at, 'prediction': 1.0}
                    for at, row in enumerate(table['rows'])
                ],
            }
        ).encode()

    def predict(self, payload: dict) -> dict:
        return json.loads(self.predict_body(json.dumps(payload).encode()))

    def close(self) -> None:
        pass


def _timed_out() -> StructuredDataError:
    r"""The transport's account of an answer slower than its timeout."""
    error = StructuredDataError(
        'Request to http://localhost/v1/predictions timed out after 1 seconds',
        code='TRANSPORT_ERROR',
        details={'timeout': 1.0},
    )
    error.__cause__ = requests.ReadTimeout()
    return error


@pytest.mark.parametrize(
    'failure',
    [
        _to_nim_error(413, b'{"detail": "too large"}'),
        _to_nim_error(502, b'{}'),
        _to_nim_error(504, b'{}'),
        _timed_out(),
    ],
    ids=['413', '502', '504', 'timeout'],
)
def test_a_call_too_large_or_too_slow_is_asked_again_in_halves(failure):
    request = gathered(context_rows=30, run_mode='fast', predict_rows=4)
    endpoint = Scripted(failure)
    result = client(endpoint).predict(request, model='kumo-relational')
    assert list(result.to_pandas()['row_index']) == [0, 1, 2, 3]
    whole, *halves = endpoint.calls
    assert [len(c['predict']['instance_table']['rows']) for c in halves] == [
        2,
        2,
    ]
    assert all(c['context'] == whole['context'] for c in halves)


def test_a_422_is_not_asked_again_and_says_what_the_endpoint_said():
    request = gathered(context_rows=30, run_mode='fast', predict_rows=4)
    endpoint = Scripted(_to_nim_error(422, b'{"detail": "N_test 4 exceeds"}'))
    with pytest.raises(StructuredDataError) as refused:
        client(endpoint).predict(request, model='kumo-relational')
    assert len(endpoint.calls) == 1
    assert 'N_test 4 exceeds' in str(refused.value)


def test_a_call_of_one_row_refused_as_too_large_is_the_endpoint_s_refusal():
    request = gathered(context_rows=30, run_mode='fast', predict_rows=1)
    endpoint = Scripted(_to_nim_error(413, b'{"detail": "context too large"}'))
    with pytest.raises(StructuredDataError) as refused:
        client(endpoint).predict(request, model='kumo-relational')
    assert len(endpoint.calls) == 1
    assert 'context too large' in str(refused.value)
