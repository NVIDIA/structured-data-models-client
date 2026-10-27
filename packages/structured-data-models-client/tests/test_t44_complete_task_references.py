# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A later invalid batch must not consume the continuing neighbor stream."""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient
from sdm_client import StructuredDataError

from test_t44_text_frames import (
    Scripted,
    core as core,
    dt3_no_time_body,
    inspection,
    p3e2_warning_codes,
    shop,
)


def question(frames):
    schema = Schema.from_data(
        frames,
        primary_keys={'users': 'user_id', 'orders': 'order_id'},
        time_columns={},
    )
    return PredictiveQuery(
        'PREDICT users.age FOR EACH users.user_id WHERE users.user_id = 1',
        schema,
    )


def neighbors(asked):
    checkpoint = asked._relational_materializer._core_sampler_state['held'][
        'frame_checkpoint'
    ]
    return json.loads(checkpoint['metadata'])['neighbors']


def test_complete_task_validation_precedes_all_neighbor_draws_and_seed_resets(
    core,
):
    observed = json.loads(
        Path(__file__)
        .with_name('t44_complete_task_references_reference.json')
        .read_text()
    )['records']
    expected = copy.deepcopy(observed)
    if core:
        for record in expected:
            for body in record.get('inspection', {}).get('payloads', []):
                dt3_no_time_body(body)
            for body in record['sent']:
                dt3_no_time_body(body)
            record['warnings'] = p3e2_warning_codes(record['warnings'])
    frames = shop()
    asked = question(frames)
    initial_neighbors = None
    continued = None
    request_ids = [
        body['metadata']['request_id']
        for record in expected
        for body in record['sent']
    ]
    with patch(
        'sdm_client.adapters.relational._request_id', side_effect=request_ids
    ):
        for record in expected:
            if record['label'] == 'fresh-valid':
                asked = question(frames)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')

                def gather():
                    return asked.materialize(
                        indices=record['indices'],
                        num_context_samples=1,
                        num_neighbors=[3],
                        num_hops=1,
                        rows_per_call=1,
                        random_seed=record['seed'],
                    )

                if 'error' in record:
                    with pytest.raises(StructuredDataError) as failure:
                        gather()
                    assert {
                        'type': type(failure.value).__name__,
                        'code': failure.value.code,
                        'message': str(failure.value),
                    } == record['error']
                    if core:
                        current = neighbors(asked)
                        if initial_neighbors is None:
                            initial_neighbors = current
                            assert current['position'] == 624
                        assert current == initial_neighbors
                else:
                    request = gather()
                    assert inspection(request) == record['inspection']
                    scripted = Scripted('age')
                    with SDMClient(
                        'http://localhost:1', max_retries=0
                    ) as client:
                        client._transport.close()
                        client._transport = scripted
                        result = client.predict(
                            request, model=KumoRelational()
                        ).to_pandas()
                    assert scripted.calls == record['sent']
                    assert (
                        json.loads(result.to_json(orient='split'))
                        == record['result']
                    )
                    if core:
                        assert request._core is not None
                        current = neighbors(asked)
                        assert current['position'] == 12
                        if record['label'] == 'continuing-valid':
                            continued = current
                        else:
                            assert continued is not None
                            assert current == continued
            assert [
                {
                    'type': type(item.message).__name__,
                    'message': str(item.message),
                }
                for item in caught
            ] == record['warnings']
