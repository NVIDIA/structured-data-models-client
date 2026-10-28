# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Later native casts may edit earlier graph meaning before graph creation."""

from __future__ import annotations

import copy
import json
import warnings
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_t44_text_frames import dt3_no_time_body, p3e2_warning_codes
from t45_frame_routes import FRAME_ROUTES

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Schema,
    SDMClient,
)
from sdm_client.materializers.relational import RelationalMaterializer

PANDAS = int(pd.__version__.split('.')[0])
REFERENCE = json.loads(
    Path(__file__)
    .with_name(f't44_live_graph_metadata_pandas{PANDAS}_reference.json')
    .read_text()
)


def frames():
    values = ['', None, 'é', '😺', 'a\x00b', '東京', 'red', 'blue']

    def text(values):
        return pd.Series(values, dtype=pd.StringDtype(storage='python'))

    return {
        'users': pd.DataFrame(
            {
                'user_id': np.arange(1, 9),
                'age': np.arange(21, 29, dtype=np.int32),
                'category': text(
                    ['red', 'blue', 'red', None, 'blue', 'red', 'blue', 'red']
                ),
                'note': text(values),
                'signal': np.arange(8, dtype='int64') + 100,
            }
        ),
        'orders': pd.DataFrame(
            {
                'order_id': np.arange(64),
                'user_id': np.repeat(np.arange(1, 9), 8),
                'marker': np.arange(64),
                'note': text(values * 8),
                'amount': np.arange(64, dtype='float64'),
            }
        ),
    }


class Scripted:
    url = 'scripted://text-feature-reference'
    serves_sessions = True

    def __init__(self):
        self.calls = []

    def predict(self, payload):
        self.calls.append(copy.deepcopy(payload))
        table = payload['predict']['instance_table']
        at = table['columns'].index('instance_id')
        return {
            'id': 'text-feature-reference',
            'model': 'kumo-relational',
            'metadata': {},
            'predictions': [
                {
                    'row_index': i,
                    'id': str(row[at]),
                    'prediction': 100.0 + len(self.calls),
                }
                for i, row in enumerate(table['rows'])
            ],
        }

    def close(self):
        pass


def failure(error):
    chain = []
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append(
            {
                'module': type(error).__module__,
                'type': type(error).__name__,
                'message': str(error),
                'code': getattr(error, 'code', None),
            }
        )
        error = (
            error.__cause__
            if error.__cause__ is not None
            else error.__context__
        )
    return {'kind': 'error', 'error': {'chain': chain}}


def observation(request, data):
    assert type(request.payloads) is tuple
    observed = {
        'kind': 'success',
        'inspection': {
            'payloads': copy.deepcopy(list(request.payloads)),
            'context_rows': request.context_rows,
            'predict_rows': request.predict_rows,
        },
    }
    scripted = Scripted()
    with SDMClient('http://localhost:1', max_retries=0) as client:
        client._transport.close()
        client._transport = scripted
        result = client.predict(request, model=KumoRelational()).to_pandas()
    observed.update(
        sent=scripted.calls,
        result=json.loads(result.to_json(orient='split')),
        merged=json.loads(
            result.merge(
                data['users'], left_on='id', right_on='user_id'
            ).to_json(orient='split')
        ),
    )
    return observed


@pytest.mark.parametrize('core', [False, True], ids=['original', 'core'])
@pytest.mark.parametrize(
    'case',
    REFERENCE['records'],
    ids=lambda case: f'{case["declared"]}-mutation-{case["mutation"]}',
)
def test_graph_observes_logical_metadata_after_all_declared_casts(
    monkeypatch, core, case
):
    monkeypatch.setenv('SDM_CORE_ROUTES', FRAME_ROUTES if core else '')
    if core:

        def legacy(*args, **kwargs):
            raise AssertionError('core graph called the legacy materializer')

        monkeypatch.setattr(RelationalMaterializer, '_get_model', legacy)
        monkeypatch.setattr(
            'sdm_client.materializers.relational._as_typed', legacy
        )
        monkeypatch.setattr(
            'sdm_client.materializers.relational._to_declared', legacy
        )
    data = frames()
    schema = Schema.from_data(
        data,
        primary_keys={'users': 'user_id', 'orders': 'order_id'},
        time_columns={},
    )
    schema['users']['signal'].stype = 'numerical'
    trace = []
    changed = False

    class NativeDecimal(Decimal):
        def _observe(self, method):
            nonlocal changed
            trace.append({'event': method})
            if case['mutation'] and not changed:
                changed = True
                schema['users']['signal'].stype = 'categorical'
                schema['users']['signal'].dtype = 'string'
                trace.append(
                    {
                        'event': 'effect',
                        'stype': str(schema['users']['signal'].stype),
                        'dtype': schema['users']['signal'].dtype,
                    }
                )

        def __float__(self):
            self._observe('float')
            return super().__float__()

        def __str__(self):
            self._observe('str')
            return super().__str__()

    data['orders']['amount'] = pd.Series(
        [NativeDecimal(str(Decimal(i) + Decimal('0.25'))) for i in range(64)],
        dtype=object,
    )
    schema['orders']['amount'].dtype = case['declared']
    query = PredictiveQuery('PREDICT users.age FOR EACH users.user_id', schema)
    expected = copy.deepcopy(case['calls'])
    request_ids = iter(
        body['metadata']['request_id']
        for call in expected
        for body in call['outcome'].get('sent', [])
    )
    monkeypatch.setattr(
        'sdm_client.adapters.relational._request_id', request_ids.__next__
    )
    if core:
        for call in expected:
            outcome = call['outcome']
            if outcome['kind'] == 'success':
                for body in [
                    *outcome['inspection']['payloads'],
                    *outcome['sent'],
                ]:
                    dt3_no_time_body(body)
            call['warnings'] = p3e2_warning_codes(call['warnings'])
    calls = []
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        for _ in range(2):
            first, first_warning = len(trace), len(notices)
            try:
                request = query.materialize(
                    indices=[7, 8],
                    num_context_samples=3,
                    num_neighbors=[3],
                    num_hops=1,
                    random_seed=7,
                )
                outcome = observation(request, data)
            except Exception as error:
                outcome = failure(error)
            calls.append(
                {
                    'outcome': outcome,
                    'trace': copy.deepcopy(trace[first:]),
                    'warnings': [
                        {
                            'type': type(w.message).__name__,
                            'message': str(w.message),
                        }
                        for w in notices[first_warning:]
                    ],
                    'schema_after': {
                        'dtype': schema['users']['signal'].dtype,
                        'stype': str(schema['users']['signal'].stype),
                        'physical': str(data['users']['signal'].dtype),
                    },
                }
            )
    assert changed == case['effect_reached']
    assert calls == expected
    # The second call uses the exact retained graph. Schema dtype mutation
    # neither replays the Decimal callbacks nor resets its actual ownership.
    assert calls[1]['trace'] == []
