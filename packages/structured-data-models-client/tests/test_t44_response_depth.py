# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual frozen public nested responses, through the real transport host."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
from test_t44_mutable_prediction import _request

from sdm_client import KumoRelational, SDMClient

REFERENCE = json.loads(
    Path(__file__).with_name('t44_response_depth_reference.json').read_text()
)


def nested(levels):
    value = 'retained-leaf'
    for _ in range(levels):
        value = [value]
    return value


@pytest.mark.parametrize('core', [False, True], ids=['original', 'core'])
@pytest.mark.parametrize('levels', [0, 90, 125])
def test_public_builtin_response_depth_matches_frozen_client(
    monkeypatch, core, levels
):
    cell = '310' if sys.version_info[:2] == (3, 10) else '312'
    expected = next(
        record
        for record in REFERENCE['cells'][cell]
        if record['levels'] == levels
    )
    monkeypatch.setenv(
        'SDM_CORE_ROUTES', 'client.predict/regression/frames' if core else ''
    )
    sent = []
    events = []

    def post(url, *, data, **kwargs):
        events.append('http-post')
        body = json.loads(data)
        sent.append(body)
        table = body['predict']['instance_table']
        at = table['columns'].index('instance_id')
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(
            {
                'id': 'frozen-response-depth',
                'model': 'kumo-relational',
                'metadata': {},
                'predictions': [
                    {
                        'row_index': index,
                        'id': str(row[at]),
                        'prediction': nested(levels),
                    }
                    for index, row in enumerate(table['rows'])
                ],
            }
        ).encode()
        response._content_consumed = True
        return response

    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        request = _request()
        identifiers = [
            body['metadata']['request_id'] for body in expected['sent']
        ]
        with patch(
            'sdm_client.adapters.relational._request_id',
            side_effect=identifiers,
        ):
            with SDMClient('http://localhost:1', max_retries=0) as client:
                client._transport._session.post = post
                result = client.predict(request, model=KumoRelational())
                frame = result.to_pandas()
        predictions = frame['prediction'].tolist()
        for prediction in predictions:
            value = prediction
            for _ in range(levels):
                assert type(value) is list and len(value) == 1
                value = value[0]
            assert type(value) is str and value == 'retained-leaf'
        observed = {
            'levels': levels,
            'complete_response_depth': levels + 3,
            'outcome': 'success',
            'result': {
                'columns': frame.columns.tolist(),
                'data': frame.values.tolist(),
            },
            'builtin_list_levels': [levels] * len(predictions),
            'warnings': [
                {
                    'type': type(item.message).__name__,
                    'message': str(item.message),
                }
                for item in notices
            ],
            'events': events,
            'sent': sent,
        }
    assert observed == expected
