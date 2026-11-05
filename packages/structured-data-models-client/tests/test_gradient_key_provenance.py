# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual sampled keyless rows and genuine similarly named source keys."""

import copy

import pandas as pd
import pytest
from test_tabular_direct_core import client

from sdm_client import PredictiveQuery, Schema


class GradientEndpoint:
    url = 'http://localhost'
    serves_sessions = True

    def __init__(self):
        self.calls = []
        self.pinned = None

    def close(self):
        pass

    def create_session(self, payload):
        self.calls.append(copy.deepcopy(payload))
        self.pinned = payload
        return {'id': 'gradient-key-session'}

    def delete_session(self, session_id):
        pass

    def session_predict(self, session_id, payload):
        return self.predict({**self.pinned, **payload})

    def predict(self, payload):
        self.calls.append(copy.deepcopy(payload))
        table = payload['predict']['instance_table']
        related = payload['predict']['related_tables']
        records = []
        for index, row in enumerate(table['rows']):
            instance = row[table['columns'].index('instance_id')]
            blocks = {}
            for name, held in related.items():
                positions = [
                    at
                    for at, values in enumerate(held['rows'])
                    if values[held['columns'].index('instance_id')] == instance
                ]
                blocks[name] = {
                    'row_indices': positions,
                    'columns': [],
                    'gradients': [[[] for _ in positions]],
                }
            records.append(
                {
                    'row_index': index,
                    'id': str(instance),
                    'prediction': 1.0,
                    'explanation': {
                        'format': 'sdm_gradient_v1',
                        'details': {
                            'input_space': 'preprocessed_numerical',
                            'outputs': [
                                {'kind': 'prediction', 'statistic': 'mean'}
                            ],
                            'instance_table': {
                                'row_indices': [index],
                                'columns': [],
                                'gradients': [[[]]],
                            },
                            'related_tables': blocks,
                        },
                    },
                }
            )
        return {
            'model': 'kumo-relational',
            'id': 'r',
            'metadata': {},
            'predictions': records,
        }


def sampled(*, genuine=False, forecast=False):
    events = pd.DataFrame(
        {
            'user_id': [i // 3 for i in range(36)],
            'value': [i % 2 for i in range(36)],
        }
    )
    if genuine:
        events['__node_id_custom'] = range(100, 136)
    schema = Schema.from_data(
        {
            'users': pd.DataFrame({'user_id': range(12), 'age': range(20, 32)}),
            'events': events,
        },
        primary_keys={'users': 'user_id'},
    )
    schema['events'].primary_key = '__node_id_custom' if genuine else None
    return PredictiveQuery(
        'EXPLAIN PREDICT users.age FOR EACH users.user_id', schema
    ).materialize(
        indices=[0, 1], num_context_samples=6, num_neighbors=[3], random_seed=1
    )


@pytest.mark.parametrize('genuine', [False, True])
@pytest.mark.parametrize('session', [False, True])
def test_generated_keys_are_internal_but_source_names_are_preserved(
    genuine, session
):
    request = sampled(genuine=genuine)
    endpoint = GradientEndpoint()
    from test_runtime_ownership import operations, require_operations

    with operations() as observed, client(endpoint) as model_client:
        if session:
            with model_client.fit(request, model='kumo-relational') as fitted:
                result = fitted.predict(request)
        else:
            result = model_client.predict(request, model='kumo-relational')
    require_operations(
        observed,
        *(
            [
                'frame_session:fit',
                'frame_session:predict',
                'frame_session:close',
            ]
            if session
            else ['predict']
        ),
    )
    for entry in result.explanations.gradients.values():
        block = entry['related']['events']
        assert block['rows']
        assert ('keys' in block) == genuine
        assert all(
            ('__node_id_custom' in row) == genuine for row in block['rows']
        )
        assert all('__node_id' not in row for row in block['rows'])
    assert endpoint.calls
    assert all(
        'client_generated_keys' not in payload for payload in endpoint.calls
    )
